from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import signal
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from figure_pipeline.fig3_revision.client import terminate_active

from .aggregate import aggregate, status
from .materials import prepare, sync_baselines
from .models import CONDITIONS, Config


def package(config: Config) -> Path:
    aggregate(config)
    summary = status(config)
    (config.output / 'DATA_README.md').write_text(
        '# Fig5 鲁棒性实验数据\n\n'
        f'100篇观察队列；20篇配对扰动；5篇原配置重复。报告实际{summary["available_reports"]}/85份；'
        f'本轮请求{summary["new_calls"]}/325次。当前覆盖以run_summary.json为准。\n\n'
        'paper_coverage/cohort记录领域、覆盖和固定抽样；inputs/source_masks记录实际条件材料；'
        'baseline记录复用来源；validated_reference与evaluation_inputs记录私有参考与真实评价输入；'
        'answer_parts保留报告原句、保留判断与技术状态；paper_condition_metrics、paired_summary、'
        'repeat_variation、observational_*为绘图源表。\n\n'
        '科学未决不等于无错误，unsupported_definitive是已确认比例；固定分母不因缺证缩小。'
        '20篇是压力样本，重复5篇只作观察波动参照；全部新评价是模型评价。\n\n'
        'new_spending记录新增支出；attributable_costs包含可归属复用调用，不可再累计为新增支出。'
        '计时限定固定输入的分析写作，不含历史建库、检索、嵌入。调用秒数之和不是墙钟时间；'
        'usage_missing不能视为零成本。不绘图、不换算美元。\n', encoding='utf-8')
    target = config.output / 'Fig5_data.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(config.output.rglob('*')):
            relative = path.relative_to(config.output)
            if not path.is_file() or path == target or path.suffix == '.lock' or path.name in ('runner.log',):
                continue
            if 'logs' in relative.parts and path.name != 'record.json':
                continue
            archive.write(path, Path('fig5_robustness') / relative)
    return target


async def execute(config: Config, args: argparse.Namespace) -> None:
    from .pipeline import run
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=config.cli_limit))
    def stop() -> None:
        terminate_active()
        if task is not None:
            task.cancel()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop)
    await run(config, args.stage, args.paper_id, args.condition, args.wait_for_fig4)
    aggregate(config)
    if args.stage == 'all':
        package(config)


def main() -> None:
    parser = argparse.ArgumentParser(description='Fig5 fixed-input robustness experiment; data only')
    parser.add_argument('command', choices=['prepare', 'run', 'aggregate', 'status', 'package'])
    parser.add_argument('--stage', choices=['reference', 'generate', 'evaluate', 'all'], default='all')
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--condition', action='append', choices=CONDITIONS)
    parser.add_argument('--wait-for-fig4', action='store_true')
    parser.add_argument('--cli-limit', type=int, default=32)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.cli_limit < 1:
        parser.error('--cli-limit must be positive')
    config = Config(cli_limit=args.cli_limit, cli_initial=args.cli_limit, workers=args.cli_limit,
                    **({'output': args.output.resolve()} if args.output else {}))
    if args.command == 'status':
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return
    config.output.mkdir(parents=True, exist_ok=True)
    with (config.output / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            prepare(config)
            aggregate(config)
        elif args.command == 'run':
            asyncio.run(execute(config, args))
        elif args.command == 'aggregate':
            sync_baselines(config)
            aggregate(config)
        else:
            print(package(config))


if __name__ == '__main__':
    main()
