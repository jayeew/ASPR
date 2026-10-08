from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from figure_pipeline.fig3_revision.client import Client
from figure_pipeline.fig3_revision.storage import read, write

from .models import Config


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.slots = asyncio.Semaphore(config.workers)

    async def call(self, paper: str, stage: str, condition: str, prompt: str, material: Any,
                   schema: type[BaseModel], model: str = 'gpt-6-astra', effort: str = 'xhigh') -> dict[str, Any]:
        path = self.config.output / stage / condition / f'{paper}.json'
        if path.exists():
            return read(path)
        blocked_path = self.config.output / 'unresolved_tasks.json'
        for blocked in read(blocked_path) if blocked_path.exists() else []:
            if (blocked['paper_id'], blocked['stage'], blocked['condition']) == (paper, stage, condition):
                raise RuntimeError(f'Explicit unresolved task, no automatic retry: {paper}/{stage}/{condition}: {blocked["category"]}')
        records = self.config.output / 'logs/calls'
        previous_states = []
        for previous in records.glob('*/record.json'):
            value = read(previous)
            if (value.get('paper_id'), value.get('stage'), value.get('method')) != (paper, stage, condition):
                continue
            for name in ('response.json', 'answer.json'):
                saved = previous.with_name(name)
                if saved.exists():
                    try:
                        result = schema.model_validate_json(saved.read_text()).model_dump()
                        write(path, result)
                        return result
                    except ValueError:
                        continue
            previous_states.append(value['state'])
        if previous_states and ('running' in previous_states or not self.config.retry_failed):
            raise RuntimeError(f'Previous incomplete attempt: {paper}/{stage}/{condition}; inspect before another request')
        async with self.slots:
            config = self.config.model_copy(update={'model': model, 'efforts': {**self.config.efforts, 'novelty': effort}})
            self.config.output.mkdir(parents=True, exist_ok=True)
            entry = {'time': datetime.now(timezone.utc).isoformat(), 'paper_id': paper, 'stage': stage,
                     'condition': condition, 'model': model, 'effort': effort, 'executable': config.executable}
            if os.environ.get('FIG4_EXPLANATION_RESUME_SESSION'):
                entry['resume_session_id'] = os.environ['FIG4_EXPLANATION_RESUME_SESSION']
            with (self.config.output / 'call_ledger.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            client = Client(config, context={'paper_id': paper, 'stage': stage, 'method': condition})
            result = await asyncio.to_thread(client.call, 'novelty', prompt, material, schema, True)
            write(path, result)
            print(f'COMPLETE {stage} {paper} {condition} [{model}/{effort}]', flush=True)
            return result
