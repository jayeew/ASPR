from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from functools import partial
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from gear.codex_cli import _strict_response_schema
from figure_pipeline.fig3_revision.client import BASE, Client
from figure_pipeline.fig3_revision.materials import fits
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_mechanisms.execution import Engine as Scheduler

from .models import Config


class TaskUnavailable(RuntimeError):
    """An explicitly recorded missing task, never a zero scientific score."""


def error_category(message: str) -> str:
    if re.search(r'limited access.*safety|content restriction|safety reasons|model refusal', message, re.I | re.S):
        return 'provider_content_restriction'
    if re.search(r'\b429\b|rate[ _]limit', message, re.I):
        return 'rate_limit'
    if re.search(r'selected model is at capacity', message, re.I):
        return 'transient_service'
    if re.search(r'timed? ?out|TimeoutExpired|stream.*disconnect|connection.*(reset|closed)|\b50[234]\b', message, re.I):
        return 'transient_transport'
    return 'technical_failure'


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.scheduler = Scheduler(config, config.workers)
        path = (config.ledger_root or config.output) / 'call_ledger.jsonl'
        self.entries = [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []

    def status(self, paper: str, stage: str, condition: str, state: str, **details: Any) -> None:
        write(self.config.output / 'task_status' / stage / condition / f'{paper}.json', {
            'paper_id': paper, 'stage': stage, 'condition': condition, 'state': state,
            'time': datetime.now(timezone.utc).isoformat(), **details})

    def records(self, paper: str, stage: str, condition: str) -> list[Any]:
        result = []
        for path in (self.config.output / 'logs/calls').glob('*/record.json'):
            record = read(path)
            if (record.get('paper_id'), record.get('stage'), record.get('method')) == (paper, stage, condition):
                result.append((path, record))
        return sorted(result, key=lambda item: item[1]['time'])

    def recover(self, paper: str, stage: str, condition: str, schema: type[BaseModel]) -> dict[str, Any] | None:
        for path, record in reversed(self.records(paper, stage, condition)):
            for filename in ('response.json', 'answer.json'):
                candidate = path.with_name(filename)
                if candidate.exists():
                    try:
                        result = schema.model_validate_json(candidate.read_text()).model_dump()
                    except ValueError:
                        continue
                    write(self.config.output / stage / condition / f'{paper}.json', result)
                    self.status(paper, stage, condition, 'completed', recovered_from=str(candidate))
                    return result
            if record['state'] == 'running' and record.get('pid'):
                try:
                    os.kill(record['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise TaskUnavailable(f'Task process still exists: {paper}/{stage}/{condition}')
        return None

    def reserve(self, paper: str, stage: str, condition: str, effort: str, additional: bool) -> str:
        additional = additional or self.config.supplemental_round
        extra = sum(e['additional'] for e in self.entries)
        ordinary = len(self.entries) - extra
        if len(self.entries) >= self.config.call_limit or (additional and extra >= self.config.additional_limit):
            raise TaskUnavailable('Authorized total/additional request limit reached')
        if not additional and ordinary >= self.config.ordinary_limit:
            raise TaskUnavailable('Authorized ordinary request limit reached')
        attempt = uuid4().hex
        entry = {'attempt_id': attempt, 'paper_id': paper, 'stage': stage, 'condition': condition,
                 'output_directory': str(self.config.output),
                 'model': self.config.model, 'effort': effort, 'additional': additional,
                 'time': datetime.now(timezone.utc).isoformat()}
        for root in {self.config.output, self.config.ledger_root or self.config.output}:
            with (root / 'call_ledger.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.entries.append(entry)
        return attempt

    async def call(self, paper: str, stage: str, condition: str, prompt: str, material: Any,
                   schema: type[BaseModel], effort: str) -> dict[str, Any]:
        path = self.config.output / stage / condition / f'{paper}.json'
        if path.exists():
            return schema.model_validate(read(path)).model_dump()
        for blocked in read(self.config.output / 'known_restrictions.json'):
            if (blocked['paper_id'], blocked['stage'], blocked['condition']) == (paper, stage, condition):
                self.status(paper, stage, condition, 'blocked', category=blocked['category'], reason=blocked['detail'])
                raise TaskUnavailable(f'Known restriction: {paper}/{stage}/{condition}')
        result = self.recover(paper, stage, condition, schema)
        if result is not None:
            return result
        previous = self.records(paper, stage, condition)
        if previous:
            last = previous[-1][1]
            category = error_category(str(last.get('cli_error', '')) + str(last.get('error', '')))
            if len(previous) >= 2 or category not in ('transient_transport', 'transient_service', 'rate_limit'):
                raise TaskUnavailable(f'Previous unresolved attempt: {paper}/{stage}/{condition}: {category}')
        config = self.config.model_copy(update={'efforts': {**self.config.efforts, 'novelty': effort}})
        if not fits(config, material, BASE + '\n' + prompt + '\nINPUT:\n', _strict_response_schema(schema.model_json_schema())):
            raise TaskUnavailable(f'Original material exceeds request capacity: {paper}/{stage}/{condition}')
        for index in range(len(previous), 2):
            try:
                result = await self.attempt(paper, stage, condition, prompt, material, schema, config, index > 0)
                write(path, result)
                return result
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                records = self.records(paper, stage, condition)
                detail = str(records[-1][1].get('cli_error', '')) if records else ''
                category = error_category(detail + str(exc))
                self.status(paper, stage, condition, 'unresolved', category=category, reason=str(exc))
                if category not in ('transient_transport', 'transient_service', 'rate_limit') or index == 1:
                    raise TaskUnavailable(f'{paper}/{stage}/{condition}: {category}: {exc}') from exc
                await asyncio.sleep(2)
        raise TaskUnavailable('No valid response')

    async def attempt(self, paper: str, stage: str, condition: str, prompt: str, material: Any,
                      schema: type[BaseModel], config: Config, additional: bool) -> dict[str, Any]:
        await self.scheduler.acquire(stage, paper, condition)
        state = 'failed'
        try:
            attempt = self.reserve(paper, stage, condition, config.efforts['novelty'], additional)
            self.status(paper, stage, condition, 'running', attempt_id=attempt)
            client = Client(config, context={'paper_id': paper, 'stage': stage, 'method': condition, 'attempt_id': attempt})
            result = await asyncio.get_running_loop().run_in_executor(
                self.scheduler.pool, partial(client.call, 'novelty', prompt, material, schema, True))
            state = 'completed'
            self.status(paper, stage, condition, state)
            print(f'COMPLETE {paper} {stage}/{condition} [{config.model}/{config.efforts["novelty"]}]', flush=True)
            return result
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            state = 'rate_limit' if error_category(str(exc)) == 'rate_limit' else 'failed'
            raise
        finally:
            await self.scheduler.release(state)

    async def close(self) -> None:
        self.scheduler.closed = True
        self.scheduler.ready.set()
        if self.scheduler.dispatcher is not None:
            await self.scheduler.dispatcher
        for pool in (self.scheduler.pool, self.scheduler.io_pool, self.scheduler.download_pool, self.scheduler.gpu_pool):
            pool.shutdown(wait=True)
