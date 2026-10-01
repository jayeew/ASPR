"""One cooperative scheduler: awaiting parents occupy no worker or model slot."""
from __future__ import annotations

import asyncio
import re
import subprocess
from collections import Counter, OrderedDict, deque
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel

from .client import BASE, Client
from .config import Config
from .materials import batches, encoded, fits, fragments
from .models import Digest
from .progress import log
from .resources import available_memory_gib
from .storage import read, write

_ACTIVITY: ContextVar[dict[str, int] | None] = ContextVar('fig3_activity', default=None)


class Engine:
    def __init__(self, config: Config, workers: int, transport: Callable[..., Any] | None = None) -> None:
        self.config, self.transport = config, transport
        self.ceiling = min(workers, config.cli_limit)
        self.limit = min(config.cli_initial, self.ceiling)
        self.active = 0
        self.window: list[str] = []
        self.stop = False
        self.pool = ThreadPoolExecutor(max_workers=max(1, self.ceiling))
        download_workers = max(1, config.network_limit//4)
        self.io_pool = ThreadPoolExecutor(max_workers=max(1, config.network_limit-download_workers))
        self.download_pool = ThreadPoolExecutor(max_workers=download_workers)
        self.gpu_pool = ThreadPoolExecutor(max_workers=1)
        self.condition = asyncio.Condition()
        self.ready = asyncio.Event()
        self.waiting: OrderedDict[str, OrderedDict[tuple[str, str], deque[Any]]] = OrderedDict()
        self.dispatcher: asyncio.Task[Any] | None = None
        self.closed = False
        self.counts: Counter[str] = Counter()
        self.shared: dict[str, asyncio.Task[Any]] = {}
        self.artifacts: dict[str, Any] = {}
        self.downloads: dict[str, asyncio.Task[Any]] = {}
        self.material_cache: dict[tuple[Any, ...], Any] = {}
        self.neutral_readers: dict[str, asyncio.Task[Any]] = {}


    def load(self, path: Any) -> Any:
        key = str(path)
        if key not in self.artifacts:
            self.artifacts[key] = read(path)
        return self.artifacts[key]

    async def io(self, function: Callable[[], Any], gpu: bool = False, download: bool = False) -> Any:
        if self.stop:
            raise InterruptedError('已停止启动新子任务')
        pool = self.gpu_pool if gpu else self.download_pool if download else self.io_pool
        return await asyncio.get_running_loop().run_in_executor(pool, function)

    async def acquire(self, stage: str = 'test', paper: str = '', method: str = '') -> None:
        if self.stop:
            raise InterruptedError('已停止启动模型')
        future = asyncio.get_running_loop().create_future()
        self.waiting.setdefault(stage, OrderedDict()).setdefault((paper, method), deque()).append(future)
        if self.dispatcher is None:
            self.dispatcher = asyncio.create_task(self.dispatch())
        self.ready.set()
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                await self.release('cancelled')
            raise

    def take_waiter(self) -> Any:
        stage, groups = self.waiting.popitem(last=False)
        key, queue = groups.popitem(last=False)
        future = queue.popleft()
        if queue:
            groups[key] = queue
        if groups:
            self.waiting[stage] = groups
        return future

    async def dispatch(self) -> None:
        while not self.closed:
            self.ready.clear()
            slots = max(0, int((available_memory_gib()-self.config.memory_reserve_gib)
                              / max(self.config.task_memory_gib, .001)))
            while self.waiting and (self.stop or self.active < self.limit and slots > 0):
                future = self.take_waiter()
                if future.cancelled():
                    continue
                if self.stop:
                    future.set_exception(InterruptedError('已停止启动模型'))
                else:
                    self.active += 1
                    slots -= 1
                    future.set_result(None)
            try:
                await asyncio.wait_for(self.ready.wait(), 1)
            except asyncio.TimeoutError:
                pass

    async def release(self, state: str) -> None:
        async with self.condition:
            self.active -= 1
            self.window.append(state)
            old = self.limit
            if state == 'rate_limit':
                self.limit = max(1, self.limit//2)
            if len(self.window) >= 32:
                if self.window.count('timeout')/len(self.window) >= .1:
                    self.limit = max(1, self.limit-8)
                elif all(s == 'completed' for s in self.window) and available_memory_gib() > self.config.memory_reserve_gib+8*self.config.task_memory_gib:
                    self.limit = min(self.ceiling, self.limit+8)
                self.window.clear()
            if old != self.limit:
                log(self.config, '并发调整', f'CLI上限 {old} → {self.limit}')
            self.condition.notify_all()
            self.ready.set()

    async def ask(self, paper: str, stage: str, method: str, step: str, role: str,
                  instruction: str, payload: Any, schema: type[BaseModel], synthesis: bool = False) -> dict[str, Any]:
        from gear.codex_cli import _strict_response_schema
        path = self.config.output/'annotations/checkpoints'/stage/(method or 'papers')/paper/f'{step}.json'
        key = str(path)
        if key in self.shared:
            if _ACTIVITY.get() is not None:
                _ACTIVITY.get()['reuse'] += 1
            return await self.shared[key]

        async def execute() -> dict[str, Any]:
            if path.exists():
                self.counts['reused'] += 1
                if _ACTIVITY.get() is not None:
                    _ACTIVITY.get()['reuse'] += 1
                return read(path)
            formatted = _strict_response_schema(schema.model_json_schema())
            material = payload() if callable(payload) else payload
            request_config = self.config
            if role != 'read' and stage in {'gear', 'graph', 'full', 'reference', 'extract', 'support',
                    'novelty', 'quality', 'checklist', 'concerns', 'clusters', 'fusion', 'preference', 'review_dynamics', 'recheck'}:
                # Original report/assessment text takes precedence over the old arbitrary
                # small-packet limit. Ordinary background readers retain their small budget.
                request_config = self.config.model_copy(update={'material_max_chars': 64000,
                    'request_max_chars': 96000, 'request_max_bytes': 384000})
                if stage in {'gear', 'graph', 'full'}:
                    request_config = request_config.model_copy(update={'material_max_chars': 96000,
                        'request_max_chars': 128000, 'request_max_bytes': 512000})
            if not fits(request_config, material, BASE+'\n'+instruction+'\nINPUT:\n', formatted):
                # Keep the object being judged separate from background reading.
                fixed_keys = {'claim_id', 'claim_text', 'claim', 'target_claim', 'core', 'predictions',
                              'report', 'reports', 'units', 'errors', 'sections', 'concerns', 'checklist',
                              'reference', 'candidate_ids', 'findings', 'decisions', 'candidate_clusters',
                              'historical_comparisons', 'contribution_sections', 'assessment'}
                fixed = {k: v for k, v in material.items() if k in fixed_keys} if isinstance(material, dict) else {}
                context = {k: v for k, v in material.items() if k not in fixed_keys} if isinstance(material, dict) else material
                if not fits(request_config, fixed, BASE+'\n'+instruction+'\nINPUT:\n', formatted):
                    raise ValueError(f'{stage}/{step}: original task objects require smaller tasks; not summarized')
                reading = await self.reduce(paper, stage, method, step+'_read', context)
                material = {**fixed, 'background_reading': reading}
                if not fits(request_config, material, BASE+'\n'+instruction+'\nINPUT:\n', formatted):
                    reading = await self.ask(paper, stage, method, step+'_context', 'read',
                        'Condense background only. Preserve source IDs, exact quotations and evidence limits. '
                        'This is not a report, claim or evaluator judgment. Aim below 1500 characters.',
                        {'background_reading': reading}, Digest)
                    material = {**fixed, 'background_reading': reading}
            if not fits(request_config, material, BASE+'\n'+instruction+'\nINPUT:\n', formatted):
                raise ValueError('Instruction/schema or indivisible material exceeds request budget')
            parts = material.get('parts', []) if isinstance(material, dict) else []
            neutral = stage == 'concerns' and role == 'read' and parts and all(
                p.get('path', '').startswith(('$.checklist.', '$.evidence_blocks[', '$.coverage[')) for p in parts)
            if neutral:
                shared_key = encoded([paper, role, instruction, material, formatted, self.config.model, self.config.efforts[role]])
                current = asyncio.current_task()
                if shared_key in self.neutral_readers and self.neutral_readers[shared_key] is not current:
                    result = await asyncio.shield(self.neutral_readers[shared_key])
                    write(path, result)
                    self.counts['reused'] += 1
                    if _ACTIVITY.get() is not None:
                        _ACTIVITY.get()['reuse'] += 1
                    return result
                self.neutral_readers[shared_key] = current
            await self.acquire(stage, paper, method)
            state = 'completed'
            try:
                client = Client(request_config, self.transport, {'paper_id': paper, 'stage': stage, 'method': method, 'step': step})
                result = await asyncio.get_running_loop().run_in_executor(
                    self.pool, lambda: client.call(role, instruction, material, schema, synthesis))
                write(path, result)
                self.counts['completed'] += 1
                if _ACTIVITY.get() is not None:
                    _ACTIVITY.get()['new'] += 1
                return result
            except subprocess.TimeoutExpired:
                state = 'timeout'
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                state = 'rate_limit' if re.search(r'\b429\b|rate[ _]limit', str(exc), re.IGNORECASE) else 'failed'
                raise
            finally:
                await self.release(state)
        self.shared[key] = asyncio.create_task(execute())
        return await self.shared[key]

    async def reduce(self, paper: str, stage: str, method: str, name: str, material: Any) -> Any:
        current = material
        for level in range(8):
            if len(encoded(current)) <= self.config.material_target_chars:
                return current
            pieces = fragments(current, self.config.block_chars)
            payloads = [{'parts': group, 'note': 'Preserve parent field/object/source IDs and unresolved limitations; do not infer omitted facts.'}
                        for group in batches(pieces, 8, 6500)]
            results = await self.map(paper, stage, method, f'{name}_L{level}', payloads,
                lambda p, i, level=level: self.ask(paper, stage, method, f'{name}_L{level}_{i:05d}', 'read',
                    'Read this fragment, retaining scientific facts, exact short quotes, source/claim IDs, '
                    'counterevidence, scope and uncertainty. Output a compact evidence digest, not a verdict. '
                    'Do not treat partial absence as global absence. Aim at less than 900 characters.', p, Digest))
            current = results
        raise ValueError('Material reduction did not converge; no oversized request was sent')

    async def map(self, paper: str, stage: str, method: str, name: str, payloads: list[Any],
                  action: Callable[[Any, int], Awaitable[Any]]) -> list[Any]:
        path = self.config.output/'inputs/tasks'/stage/(method or 'papers')/paper/f'{name}.json'
        if path.exists():
            payloads = read(path)['tasks']
        else:
            write(path, {'tasks': payloads})
        counts: Counter[str] = Counter()
        async def one(payload: Any, index: int) -> Any:
            activity = {'new': 0, 'reuse': 0}
            token = _ACTIVITY.set(activity)
            try:
                value = await action(payload, index)
                counts['reused' if activity['reuse'] and not activity['new'] else 'success'] += 1
                return value
            except asyncio.CancelledError:
                counts['cancelled'] += 1
                raise
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
                counts['failed'] += 1
                return exc
            finally:
                _ACTIVITY.reset(token)
                log(self.config, '子任务进度', f'[{name} {sum(counts.values())}/{len(payloads)}] '
                    f'成功{counts["success"]}，复用{counts["reused"]}，失败{counts["failed"]}，取消{counts["cancelled"]}；运行中{self.active}',
                    paper_id=paper, stage=stage, method=method)
        results = await asyncio.gather(*(one(p, i) for i, p in enumerate(payloads)))
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise RuntimeError(f'{name}: {len(errors)}个子任务失败: {errors[0]}')
        return results

    async def close(self) -> None:
        self.closed = True
        self.ready.set()
        if self.dispatcher is not None:
            self.dispatcher.cancel()
            await asyncio.gather(self.dispatcher, return_exceptions=True)
        for pool in (self.pool, self.io_pool, self.download_pool, self.gpu_pool):
            pool.shutdown(wait=True, cancel_futures=True)
