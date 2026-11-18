"""Fig4 scheduler: transport timeouts do not imply an overloaded model service."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from figure_pipeline.fig3_revision.execution import Engine as BaseEngine
from figure_pipeline.fig3_revision.progress import log


class Engine(BaseEngine):
    async def release(self, state: str) -> None:
        async with self.condition:
            self.active -= 1
            old = self.limit
            if state == 'rate_limit':
                self.limit = max(min(8, self.ceiling), self.limit//2)
                self.window.clear()
            elif state == 'completed':
                self.window.append(state)
                if len(self.window) >= 8:
                    self.limit = min(self.ceiling, self.limit+8)
                    self.window.clear()
            if old != self.limit:
                log(self.config, '并发调整', f'Fig4 CLI上限 {old} → {self.limit} ({state})')
            self.condition.notify_all()
            self.ready.set()

    async def ask(self, paper: str, stage: str, method: str, step: str, role: str,
                  instruction: str, payload: Any, schema: type[BaseModel], synthesis: bool = False) -> dict[str, Any]:
        if self.config.efforts.get(role) == 'xhigh' or role == 'clusters':
            synthesis = True
        for attempt in range(2):
            try:
                return await super().ask(paper, stage, method, step, role, instruction,
                                         payload, schema, synthesis)
            except subprocess.TimeoutExpired:
                if self.stop or attempt:
                    raise
                key = str(self.config.output/'annotations/checkpoints'/stage/(method or 'papers')/paper/f'{step}.json')
                self.shared.pop(key, None)
                self.config = self.config.model_copy(update={
                    'executable': str(Path(__file__).with_name('codex_http'))})
                log(self.config, '超时续跑', f'{step}: 保留失败记录，原始科学任务重试一次',
                    paper_id=paper, stage=stage, method=method)
                await asyncio.sleep(2)
        raise RuntimeError('No task result')
