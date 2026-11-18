from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel

from figure_pipeline.fig3_revision.storage import write
from figure_pipeline.fig5_robustness.runtime import Runner as PreviousRunner
from figure_pipeline.fig5_robustness.runtime import Unavailable


class Runner(PreviousRunner):
    """Permit supervised code repairs after current calls finish, without losing requests."""

    def reserve(self, paper: str, stage: str, condition: str, model: str, effort: str, extra: bool) -> str:
        if (self.config.output / 'pause_dispatch').exists():
            raise Unavailable('Dispatch paused for supervised repair; no new request was spent')
        return super().reserve(paper, stage, condition, model, effort, extra)

    async def call(self, paper: str, stage: str, condition: str, prompt: str, material: Any,
                   schema: type[BaseModel], model: str = 'gpt-6.1-sol', effort: str = 'high',
                   additional: bool = False) -> dict[str, Any]:
        target = self.config.output / stage / condition / f'{paper}.json'
        while not target.exists() and (self.config.output / 'pause_dispatch').exists():
            self.status(paper, stage, condition, 'waiting_dispatch', reason='Existing requests finish before supervised repair')
            await asyncio.sleep(3)
        try:
            return await super().call(paper, stage, condition, prompt, material, schema, model, effort, additional)
        except Unavailable:
            records = self.records(paper, stage, condition)
            if not records or 'Selected model is at capacity' not in records[-1][1].get('cli_error', ''):
                raise
            # Preserve the failed call and experimental model; at most one capacity-retry key.
            retry = condition + '__capacity_retry'
            if not self.records(paper, stage, retry):
                await asyncio.sleep(60)
            result = await super().call(paper, stage, retry, prompt, material, schema, model, effort, True)
            write(target, result)
            self.status(paper, stage, condition, 'completed', recovered_from_condition=retry)
            return result


async def protected(runner: Runner, paper: str, stage: str, condition: str, job: Awaitable[Any]) -> Any:
    try:
        result = await job
    except (OSError, ValueError, RuntimeError) as exc:
        runner.status(paper, stage, condition, 'unresolved', reason=str(exc))
        print(f'UNRESOLVED {paper} {stage}/{condition}: {str(exc)[:250]}', flush=True)
        return None
    runner.status(paper, stage, condition, 'completed')
    return result
