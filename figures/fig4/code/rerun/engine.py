"""Reuse scheduling/HTTP; exactly one client attempt per authorized task."""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from figure_pipeline.fig3_revision.client import BASE, Client
from figure_pipeline.fig3_revision.materials import fits
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_mechanisms.execution import Engine as Scheduler

from .config import Config


def ledger(root: Path) -> list[dict[str, Any]]:
    path = root / 'call_ledger.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def adopted_entries(rows: list[dict[str, Any]], config: Config) -> list[dict[str, Any]]:
    """Final-version writer plus final-version evaluation attempts; retain history separately."""
    return [r for r in rows if r['stage'] == 'generate'
            or r.get('evaluation_round', 'v1') == config.evaluation_round]


class Engine(Scheduler):
    def __init__(self, config: Config, cohort: list[str], ceiling: int) -> None:
        super().__init__(config, config.workers)
        self.entries = ledger(config.output)
        self.cohort, self.cohort_ceiling = set(cohort), ceiling

    def recovered(self, paper: str, condition: str, stage: str, schema: type[BaseModel]) -> dict[str, Any] | None:
        for path in (self.config.output / 'logs/calls').glob('*/record.json'):
            record = read(path)
            if (record.get('paper_id'), record.get('method'), record.get('stage')) != (paper, condition, stage):
                continue
            if stage == 'evaluate' and record.get('evaluation_round', 'v1') != self.config.evaluation_round:
                continue
            for name in ('response.json', 'answer.json'):
                candidate = path.with_name(name)
                if candidate.exists():
                    try:
                        return schema.model_validate_json(candidate.read_text()).model_dump()
                    except ValueError:
                        continue
        return None

    async def once(self, paper: str, condition: str, stage: str, instruction: str,
                   payload: Any, schema: type[BaseModel]) -> dict[str, Any]:
        from gear.codex_cli import _strict_response_schema

        checkpoint_stage = stage if stage != 'evaluate' else stage + '_' + self.config.evaluation_round
        path = self.config.output / 'checkpoints' / checkpoint_stage / condition / f'{paper}.json'
        if path.exists():
            return read(path)
        if any((r['paper_id'], r['condition'], r['stage']) == (paper, condition, stage)
               and (stage != 'evaluate' or r.get('evaluation_round', 'v1') == self.config.evaluation_round)
               for r in self.entries):
            recovered = self.recovered(paper, condition, stage, schema)
            if recovered is None:
                raise RuntimeError('Previous attempt has no valid response; no automatic retry authorized')
            write(path, recovered)
            return recovered
        formatted = _strict_response_schema(schema.model_json_schema())
        if not fits(self.config, payload, BASE + '\n' + instruction + '\nINPUT:\n', formatted):
            raise ValueError('Original input exceeds request limit; no summarization or model call made')
        await self.acquire(stage, paper, condition)
        state = 'completed'
        try:
            adopted = adopted_entries(self.entries, self.config)
            if len(adopted) >= 1400 or sum(r['paper_id'] in self.cohort for r in adopted) >= self.cohort_ceiling:
                raise RuntimeError('Authorized call ceiling reached')
            entry = {'call_number': len(self.entries) + 1, 'paper_id': paper, 'condition': condition, 'stage': stage}
            if stage == 'evaluate':
                entry['evaluation_round'] = self.config.evaluation_round
            ledger_path = self.config.output / 'call_ledger.jsonl'
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open('a') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self.entries.append(entry)
            client = Client(self.config, context={'paper_id': paper, 'method': condition, 'stage': stage,
                                                 'step': checkpoint_stage,
                                                 'evaluation_round': self.config.evaluation_round if stage == 'evaluate' else None})
            role = 'baseline' if stage == 'generate' else 'novelty'
            result = await asyncio.get_running_loop().run_in_executor(
                self.pool, lambda: client.call(role, instruction, payload, schema, True))
            write(path, result)
            return result
        except subprocess.TimeoutExpired:
            state = 'timeout'
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            state = 'rate_limit' if re.search(r'\b429\b|rate[ _]limit', str(exc), re.I) else 'failed'
            raise
        finally:
            await self.release(state)
