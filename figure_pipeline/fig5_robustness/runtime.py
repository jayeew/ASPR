from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from figure_pipeline.fig3_revision.client import BASE, Client
from figure_pipeline.fig3_revision.materials import fits
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.runtime import error_category
from gear.codex_cli import _strict_response_schema

from .models import Config


class Unavailable(RuntimeError):
    """Recorded missing task, not a scientific zero."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def upstream_processes() -> list[int]:
    found = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(a.startswith(b'figure_pipeline.fig4_') for a in args) and b'run' in args:
            found.append(int(path.parent.name))
    return found


def memory_available() -> float:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError('MemAvailable unavailable')


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.slots = asyncio.Semaphore(config.cli_limit)
        ledger = config.output / 'call_ledger.jsonl'
        self.entries = [json.loads(line) for line in ledger.read_text().splitlines() if line] if ledger.exists() else []

    def status(self, paper: str, stage: str, condition: str, state: str, **details: Any) -> None:
        write(self.config.output / 'task_status' / stage / condition / f'{paper}.json',
              dict(paper_id=paper, stage=stage, condition=condition, state=state, time=now(), **details))

    def reserve(self, paper: str, stage: str, condition: str, model: str, effort: str, extra: bool) -> str:
        ordinary = sum(not r['additional'] for r in self.entries)
        additional = len(self.entries) - ordinary
        if len(self.entries) >= self.config.call_limit or (extra and additional >= self.config.additional_limit) or (not extra and ordinary >= self.config.ordinary_limit):
            raise Unavailable('Fig5 request budget exhausted')
        attempt = uuid4().hex
        row = {'attempt_id': attempt, 'paper_id': paper, 'stage': stage, 'condition': condition,
                   'model': model, 'effort': effort, 'additional': extra, 'time': now()}
        path = self.config.output / 'call_ledger.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        self.entries.append(row)
        return attempt

    def records(self, paper: str, stage: str, condition: str) -> list[tuple[Path, dict[str, Any]]]:
        result = []
        for path in (self.config.output / 'logs/calls').glob('*/record.json'):
            row = read(path)
            if (row.get('paper_id'), row.get('stage'), row.get('method')) == (paper, stage, condition):
                result.append((path, row))
        return sorted(result, key=lambda r: r[1]['time'])

    def recover(self, paper: str, stage: str, condition: str, schema: type[BaseModel]) -> dict[str, Any] | None:
        for path, record in reversed(self.records(paper, stage, condition)):
            for name in ('response.json', 'answer.json'):
                candidate = path.with_name(name)
                if candidate.exists():
                    try:
                        value = schema.model_validate_json(candidate.read_text()).model_dump()
                    except ValueError:
                        continue
                    self.status(paper, stage, condition, 'completed', recovered_from=str(candidate))
                    return value
            if record['state'] == 'running' and record.get('pid'):
                try:
                    os.kill(record['pid'], 0)
                except ProcessLookupError:
                    continue
                raise Unavailable('Prior request still running')
        return None

    def fits(self, prompt: str, material: Any, schema: type[BaseModel]) -> bool:
        return fits(self.config, material, BASE + '\n' + prompt + '\nINPUT:\n',
                    _strict_response_schema(schema.model_json_schema()))

    async def call(self, paper: str, stage: str, condition: str, prompt: str, material: Any,
                   schema: type[BaseModel], model: str = 'gpt-6.1-sol', effort: str = 'high',
                   additional: bool = False) -> dict[str, Any]:
        target = self.config.output / stage / condition / f'{paper}.json'
        if target.exists():
            return schema.model_validate(read(target)).model_dump()
        value = self.recover(paper, stage, condition, schema)
        if value is not None:
            write(target, value)
            return value
        records = self.records(paper, stage, condition)
        attempts = [r for r in self.entries if (r['paper_id'], r['stage'], r['condition']) == (paper, stage, condition)]
        if attempts:
            last = records[-1][1] if records else {}
            category = error_category(str(last.get('error', '')) + str(last.get('cli_error', '')))
            if len(attempts) >= 2 or category not in ('transient_transport', 'rate_limit'):
                raise Unavailable(f'Unresolved previous request: {category}')
        if not self.fits(prompt, material, schema):
            raise Unavailable('Original input exceeds capacity; no truncation or compression')
        cfg = self.config.model_copy(update={'model': model, 'efforts': {**self.config.efforts, 'novelty': effort},
                                             'repair_attempts': 0, 'network_retries': 0})
        for index in range(len(attempts), 2):
            try:
                async with self.slots:
                    while memory_available() < self.config.memory_reserve_gib:
                        self.status(paper, stage, condition, 'waiting_memory')
                        await asyncio.sleep(10)
                    attempt = self.reserve(paper, stage, condition, model, effort, additional or index > 0)
                    self.status(paper, stage, condition, 'running', attempt_id=attempt)
                    client = Client(cfg, context={'paper_id': paper, 'stage': stage, 'method': condition, 'attempt_id': attempt})
                    result = await asyncio.to_thread(client.call, 'novelty', prompt, material, schema, True)
                write(target, result)
                self.status(paper, stage, condition, 'completed', attempt_id=attempt)
                print(f'COMPLETE {paper} {stage}/{condition} {model}/{effort}', flush=True)
                return result
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                records = self.records(paper, stage, condition)
                detail = records[-1][1].get('cli_error', '') if records else ''
                category = error_category(str(exc) + str(detail))
                self.status(paper, stage, condition, 'unresolved', category=category, reason=str(exc))
                if category not in ('transient_transport', 'rate_limit') or index == 1:
                    raise Unavailable(f'{category}: {exc}') from exc
                await asyncio.sleep(2)
        raise Unavailable('No valid response')
