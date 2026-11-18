"""Named, ordinary step files. Existence-only reuse and explicit task reset."""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import Config
from .progress import duration, log
from .storage import Store, read, write


class Checkpoints:
    def __init__(self, config: Config, context: dict[str, str], stop: threading.Event | None = None) -> None:
        self.config, self.context, self.stop = config, context, stop

    def directory(self) -> Path:
        return (self.config.output/'annotations/checkpoints'/self.context['stage']/
                (self.context.get('method') or 'papers')/self.context['paper_id'])

    def run(self, step: str, action: Callable[[], Any], path: Path | None = None) -> Any:
        path = path if path is not None else self.directory()/f'{step}.json'
        if self.stop is not None and self.stop.is_set():
            raise InterruptedError('用户已请求停止，保留已落盘断点，不再启动下一子步骤')
        fields = {**self.context, 'step': step, 'checkpoint': str(path)}
        if path.exists():
            log(self.config, '断点复用', '读取已完成子步骤，不再次调用模型或检索。', **fields)
            return read(path)
        started = time.monotonic()
        log(self.config, '步骤开始', '开始处理，成功后立即保存断点。', **fields)
        try:
            result = action()
            write(path, result)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError,
                subprocess.SubprocessError, sqlite3.Error) as exc:
            log(self.config, '步骤失败', f'{type(exc).__name__}: {exc}；本步骤未完成，下次重试。', **fields)
            raise
        elapsed = time.monotonic()-started
        log(self.config, '步骤完成', f'用时{duration(elapsed)}，结果已保存。', seconds=elapsed, **fields)
        return result


def reset_task(config: Config, stage: str, ident: str, variant: str) -> None:
    """Only a user-selected --overwrite task loses its prior results/checkpoints."""
    paths = [Store(config).path(stage, ident, variant),
             config.output/'annotations/checkpoints'/stage/(variant or stage)/ident]
    if stage in {'eacl', 'reviewgrounder'}:
        paths.append(config.output/'reports'/stage/'steps'/ident)
    elif stage == 'core':
        paths.append(config.output/'annotations/core_selection'/f'{ident}.json')
    elif stage == 'support':
        paths.append(config.output/'annotations/support_batches'/variant/ident)
    elif stage == 'recheck':
        paths.append(config.output/'annotations/recheck_details'/ident)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    log(config, '显式重跑', '已清除本任务旧结果和子步骤断点；其他任务及下游结果保持原样。',
        stage=stage, paper_id=ident, method=variant)
