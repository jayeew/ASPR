"""One bounded CLI scheduler for all model tasks in the five-dimension study."""
from __future__ import annotations

import asyncio
import errno
import json
import math
import os
import re
import shutil
import signal
import tempfile
import time
from collections import Counter, deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from figure_pipeline.fig3_revision.resources import available_memory_gib
from gear.codex_cli import _strict_response_schema

from .config import MODEL, PROTOCOL
from .io import encoded, fingerprint, read, write


class RefusalError(RuntimeError):
    pass


class ToolUseError(RuntimeError):
    pass


def transient_failure(error: BaseException) -> bool:
    if isinstance(error, (RefusalError, ToolUseError)):
        return False
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, OSError):
        return error.errno in {errno.EAGAIN, errno.EINTR, errno.ECONNRESET,
                               errno.ECONNREFUSED, errno.ETIMEDOUT, errno.EPIPE,
                               errno.ENETDOWN, errno.ENETUNREACH, errno.EHOSTUNREACH}
    return bool(re.search(
        r'\b(?:429|502|503|504)\b|rate[ _]limit|timed? out|timeout|temporar|'
        r'connection (?:reset|closed|lost|refused)|stream disconnected|broken pipe', str(error), re.I))


def explicit_refusal(text: str) -> bool:
    return bool(re.match(
        r"\s*(?:(?:I(?:'m| am) sorry[,，]?\s*(?:but\s+)?)?I "
        r"(?:cannot|can't|won't|am unable to) (?:assist|help|comply|provide|fulfill)\b|"
        r'(?:抱歉|对不起)[，,。\s]*(?:我)?(?:无法|不能)(?:协助|帮助|提供|满足))', text, re.I))


def parse_layout_json(text: str) -> Any:
    """Recover a value only by removing a whole fence or commas before ]/} outside strings."""
    fenced = re.fullmatch(r'\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*', text)
    original = fenced[1] if fenced else text
    chars, quoted, escaped = [], False, False
    for i, char in enumerate(original):
        if quoted:
            chars.append(char)
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            chars.append(char)
        elif char == ',' and original[i + 1:].lstrip().startswith(('}', ']')):
            continue
        else:
            chars.append(char)
    return json.loads(''.join(chars))


def same_scalar_representation(before: Any, after: Any) -> bool:
    if type(before) is type(after):
        return before == after
    if before is None or after is None:
        return False
    if isinstance(before, bool) or isinstance(after, bool):
        return ((isinstance(before, str) and before in ('true', 'false') and after is (before == 'true'))
                or (isinstance(after, str) and after in ('true', 'false') and before is (after == 'true')))
    if not all(isinstance(value, (str, int, float)) for value in (before, after)):
        return False
    try:
        first, second = Decimal(str(before)), Decimal(str(after))
        return first.is_finite() and second.is_finite() and first == second
    except InvalidOperation:
        return False


def verify_lossless_format(before: Any, after: Any, extra_paths: set[tuple[Any, ...]],
                           path: tuple[Any, ...] = ()) -> None:
    if isinstance(before, dict) and isinstance(after, dict):
        required = {key for key in before if path + (key,) not in extra_paths}
        if set(after) != required:
            raise ValueError(f'Format repair changed retained fields at {path}')
        for key in after:
            verify_lossless_format(before[key], after[key], extra_paths, path + (key,))
    elif isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            raise ValueError(f'Format repair changed list length at {path}')
        for i, (first, second) in enumerate(zip(before, after)):
            verify_lossless_format(first, second, extra_paths, path + (i,))
    elif isinstance(before, (dict, list)) or isinstance(after, (dict, list)) or not same_scalar_representation(before, after):
        raise ValueError(f'Format repair changed original scientific content at {path}')


class Scheduler:
    def __init__(self, output: Path, maximum: int = 64, initial: int = 16,
                 timeout: int = 900, single_attempt: bool = False) -> None:
        if maximum < 1 or initial < 1 or timeout < 1:
            raise ValueError('Maximum, initial concurrency and timeout must be positive')
        self.output = output
        self.maximum = maximum
        self.limit = min(initial, maximum)
        self.timeout = timeout
        self.single_attempt = single_attempt
        self.active = 0
        self.peak = 0
        self.window: list[str] = []
        self.counts: Counter[str] = Counter()
        self.condition = asyncio.Condition()
        self.started = time.monotonic()
        self.admission_cycle = ('evaluation', 'reference', 'evaluation', 'report')
        self.admission_cursor = 0
        self.waiting: dict[str, deque[object]] = {stage: deque() for stage in set(self.admission_cycle)}

    def memory_ceiling(self) -> int:
        return max(0, math.floor((available_memory_gib() - 4.0) / 0.5))

    def next_stage(self) -> tuple[str, int]:
        for offset in range(len(self.admission_cycle)):
            position = (self.admission_cursor + offset) % len(self.admission_cycle)
            stage = self.admission_cycle[position]
            if self.waiting[stage]:
                return stage, (position + 1) % len(self.admission_cycle)
        raise RuntimeError('No queued task is available')

    async def acquire(self, task: str = '') -> None:
        prefix = task.split('/', 1)[0]
        stage = ('evaluation' if prefix in {'evaluation', 'evaluate'} else
                 'reference' if prefix == 'reference' else 'report')
        token, acquired = object(), False
        async with self.condition:
            self.waiting[stage].append(token)
        try:
            while True:
                async with self.condition:
                    chosen, cursor = self.next_stage()
                    if (chosen == stage and self.waiting[stage][0] is token
                            and self.active < min(self.limit, self.memory_ceiling())):
                        self.waiting[stage].popleft()
                        self.admission_cursor = cursor
                        self.active += 1
                        self.peak = max(self.peak, self.active)
                        acquired = True
                        return
                await asyncio.sleep(1)
        finally:
            if not acquired:
                async with self.condition:
                    self.waiting[stage].remove(token)

    async def release(self, state: str) -> None:
        async with self.condition:
            self.active -= 1
            self.counts[state] += 1
            self.window.append(state)
            if state == 'rate_limit':
                self.limit = max(1, self.limit // 2)
            if len(self.window) >= 32:
                if self.window.count('timeout') / len(self.window) >= .1:
                    self.limit = max(1, self.limit - 8)
                elif all(s == 'completed' for s in self.window):
                    self.limit = min(self.maximum, self.limit + 8)
                self.window.clear()
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        return {'active': self.active, 'adaptive_limit': self.limit, 'maximum': self.maximum,
                'memory_ceiling': self.memory_ceiling(), 'peak_active': self.peak,
                'available_memory_gib': round(available_memory_gib(), 2),
                'elapsed_seconds': round(time.monotonic() - self.started, 1),
                'calls': dict(self.counts),
                'queued_by_stage': {stage: len(queue) for stage, queue in self.waiting.items()}}

    async def ask(self, task: str, instruction: str, payload: dict[str, Any],
                  schema: type[BaseModel], effort: str) -> dict[str, Any]:
        if self.single_attempt:
            from .full_rerun import ask_once
            return await ask_once(self, task, instruction, payload, schema, effort)
        shape = _strict_response_schema(schema.model_json_schema())
        digest = fingerprint([PROTOCOL, MODEL, effort, instruction, payload, shape])
        cached = self.output / 'checkpoints' / task / (digest + '.json')
        if cached.exists():
            self.counts['reused'] += 1
            return schema.model_validate(read(cached)).model_dump(mode='json')
        attempts, format_attempted = 0, False
        original_value: Any = None
        extra_paths: set[tuple[Any, ...]] = set()
        current_task, current_instruction, current_payload = task, instruction, payload
        while True:
            raw: Any = None
            try:
                raw = await self.invoke(current_task, current_instruction, current_payload, shape, effort, digest)
                if isinstance(raw, dict) and raw.get('refusal'):
                    raise RefusalError(str(raw['refusal']))
                try:
                    result = schema.model_validate(raw).model_dump(mode='json')
                finally:
                    if format_attempted:
                        verify_lossless_format(original_value, raw, extra_paths)
            except (ValidationError, json.JSONDecodeError) as exc:
                if format_attempted:
                    raise
                # Missing dimensions/fields, applicability and score changes are scientific edits.
                lossless_types = {'extra_forbidden', 'int_parsing', 'int_type', 'float_parsing',
                                  'float_type', 'bool_parsing', 'bool_type', 'string_type'}
                if isinstance(exc, ValidationError) and any(
                        error['type'] not in lossless_types for error in exc.errors()):
                    raise
                original_response = exc.doc if isinstance(exc, json.JSONDecodeError) else raw
                if isinstance(original_response, str) and explicit_refusal(original_response):
                    raise RefusalError(original_response) from exc
                if isinstance(exc, json.JSONDecodeError):
                    # An unparseable response has no trustworthy scientific value to rewrite.
                    recovered = parse_layout_json(original_response)
                    result = schema.model_validate(recovered).model_dump(mode='json')
                    write(cached, result)
                    return result
                original_value = raw
                extra_paths = {tuple(error['loc']) for error in exc.errors() if error['type'] == 'extra_forbidden'}
                current_payload = {'response_to_format': original_response, 'format_error': str(exc)}
                current_instruction = ('Repair JSON formatting only, using this schema. Preserve all scientific '
                                       'judgments, scores, wording, IDs and quotations. Never add missing '
                                       'scientific fields or dimension entries. Do not call any tools.')
                current_task = task + '/format_repair'
                format_attempted = True
                continue
            except (RefusalError, ToolUseError):
                raise
            except (TimeoutError, OSError, RuntimeError) as exc:
                if not transient_failure(exc) or attempts >= 1:
                    raise
                attempts += 1
                await asyncio.sleep(10 if '429' in str(exc) else 2)
                continue
            write(cached, result)
            return result

    async def invoke(self, task: str, instruction: str, payload: dict[str, Any],
                     shape: dict[str, Any], effort: str, digest: str) -> Any:
        directory = self.output / 'logs/calls' / uuid4().hex
        directory.mkdir(parents=True)
        prompt = instruction + '\n\nINPUT:\n' + encoded(payload)
        (directory / 'prompt.txt').write_text(prompt)
        write(directory / 'schema.json', shape)
        record = {'task': task, 'model': MODEL, 'reasoning_effort': effort,
                  'protocol': PROTOCOL, 'request_hash': digest,
                  'quotation_protocol': payload.get('quotation_protocol'),
                  'provider': 'inherited CLI configuration', 'service_tier': 'inherited CLI configuration',
                  'request_characters': len(prompt), 'state': 'queued', 'usage': None,
                  'started_at': datetime.now(timezone.utc).isoformat()}
        write(directory / 'record.json', record)
        await self.acquire(task)
        started, state = time.monotonic(), 'failed'
        try:
            record.update(state='running', active_at_start=self.active,
                          admitted_at=datetime.now(timezone.utc).isoformat())
            write(directory / 'record.json', record)
            response = await self.process(directory, prompt, effort, record)
            if explicit_refusal(response):
                raise RefusalError(response)
            result = parse_layout_json(response) if self.single_attempt else json.loads(response)
            if isinstance(result, dict) and result.get('refusal'):
                raise RefusalError(str(result['refusal']))
            write(directory / 'response.json', result)
            state = 'completed'
            return result
        except asyncio.CancelledError:
            state = 'interrupted'
            record['error'] = 'Task cancelled; request and available process output retained.'
            raise
        except TimeoutError as exc:
            state = 'timeout'
            record['error'] = str(exc)
            raise
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            record['error'] = str(exc)
            state = 'rate_limit' if re.search(r'\b429\b|rate[ _]limit', str(exc), re.I) else 'failed'
            raise
        finally:
            record.update(state=state, seconds=time.monotonic() - started,
                          finished_at=datetime.now(timezone.utc).isoformat())
            try:
                write(directory / 'record.json', record)
            finally:
                await self.release(state)

    async def process(self, directory: Path, prompt: str, effort: str,
                      record: dict[str, Any]) -> str:
        executable = shutil.which('codex')
        if executable is None:
            raise RuntimeError('Codex CLI not found')
        command = [executable, 'exec', '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only',
                   '--model', MODEL, '-c', f'model_reasoning_effort="{effort}"', '--json',
                   '--output-schema', str((directory / 'schema.json').resolve()),
                   '--output-last-message', str((directory / 'answer.json').resolve()), '-']
        with tempfile.TemporaryDirectory(prefix='innovation-five-') as cwd:
            if self.single_attempt:
                ledger_path = self.output / 'status/full_rerun_calls.json'
                ledger = read(ledger_path)
                ledger[record['task']]['state'] = 'started'
                write(ledger_path, ledger)
            process = await asyncio.create_subprocess_exec(*command, cwd=cwd,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, start_new_session=True)
            record['pid'] = process.pid
            write(directory / 'record.json', record)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(prompt.encode()), self.timeout)
            except (TimeoutError, asyncio.CancelledError):
                if process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                stdout, stderr = await process.communicate()
                (directory / 'events.jsonl').write_bytes(stdout)
                (directory / 'stderr.txt').write_bytes(stderr)
                raise
        (directory / 'events.jsonl').write_bytes(stdout)
        (directory / 'stderr.txt').write_bytes(stderr)
        inspect_events(stdout.decode(errors='replace'), record)
        record['returncode'] = process.returncode
        answer = directory / 'answer.json'
        if process.returncode:
            raise RuntimeError(f'CLI exit {process.returncode}: ' + stderr.decode(errors='replace')[-2000:]
                               + str(record.get('cli_errors', []))[-1000:])
        if not answer.is_file():
            raise RuntimeError('CLI returned no final response')
        return answer.read_text()


def inspect_events(text: str, record: dict[str, Any]) -> None:
    violations = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get('type') == 'turn.completed':
            record['usage'] = event.get('usage')
        if event.get('type') in {'error', 'turn.failed'}:
            record.setdefault('cli_errors', []).append(event)
        item = event.get('item') or {}
        item_type = item.get('type', '') if isinstance(item, dict) else ''
        if item_type in {'command_execution', 'file_change', 'web_search', 'tool_call',
                         'function_call', 'computer_call', 'image_generation_call'} or item_type.endswith('_tool_call'):
            violations.append(item_type)
    record['tool_use_violations'] = violations
    if violations:
        raise ToolUseError('Unpermitted tool use in experimental call: ' + ', '.join(violations))
