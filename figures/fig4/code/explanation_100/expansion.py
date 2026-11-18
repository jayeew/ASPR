"""Accepted five-paper protocol applied to the remaining 95; data outputs only."""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write

from . import prompts
from .models import Config
from .refinement import configuration as pilot_configuration, topology


def configuration() -> Config:
    return pilot_configuration().model_copy(update={
        'supplemental_round': False, 'expanded_protocol': True})


def prepare(config: Config) -> None:
    original = config.ledger_root
    assert original is not None
    cohort = read(config.output / 'cohort.json')
    for paper in cohort['paper_ids']:
        for directory in ('inputs/papers', 'inputs/reference'):
            target = config.output / directory / f'{paper}.json'
            if not target.exists():
                shutil.copyfile(original / directory / f'{paper}.json', target)
        target = config.output / 'factual_opportunities' / f'{paper}.json'
        if not target.exists():
            write(target, topology(read(config.output / 'inputs/papers' / f'{paper}.json')['graph']))
    acceptance = config.output / 'pilot_acceptance.json'
    if not acceptance.exists():
        write(acceptance, dict(accepted=True, time=datetime.now(timezone.utc).isoformat(),
              user_instruction='继续追加启动剩余95篇吧，仅仅完成数据准备即可 不用再画图了',
              accepted_pilot='focused_J_S_v2 with uniform joint criterion review',
              reuse_pilot_results=True, render=False))
    archive = config.output / 'protocol_pilot.json'
    if not archive.exists():
        shutil.copyfile(config.output / 'protocol.json', archive)
    write(config.output / 'protocol.json', dict(
        version='accepted_J_S_v2_expansion', scope='100_candidates_including_5_development_papers',
        generation_model=config.model, generation_effort=config.generation_effort,
        evaluation_effort=config.evaluation_effort, input_masks_unchanged=True,
        reconstruction_allowed=True, private_reference_not_in_writer=True,
        new_reference_calls_per_paper=1, joint_review_criterion_applied_in_initial_evaluation=True,
        pilot_results_reused=True, remaining_authorized=True, render=False,
        ordinary_limit=config.ordinary_limit, additional_limit=config.additional_limit,
        shared_total_limit=config.call_limit))
    print('Prepared 100 factual inputs; accepted five reused; 95 authorized; no model calls.', flush=True)


def status(config: Config) -> dict[str, Any]:
    records = [read(p) for p in (config.output / 'logs/calls').glob('*/record.json')]
    statuses = [read(p) for p in (config.output / 'task_status').rglob('*.json')]
    original = config.ledger_root or config.output
    ledger = [json.loads(s) for s in (original / 'call_ledger.jsonl').read_text().splitlines() if s]
    return dict(shared_calls=len(ledger), shared_additional=sum(r['additional'] for r in ledger),
                call_states=dict(Counter(r['state'] for r in records)),
                stages=dict(Counter(r['stage'] + '/' + r['state'] for r in records)),
                reports=len(list((config.output / 'reports').glob('*/*.json'))),
                screened=len(list((config.output / 'screening').glob('*.json'))),
                unresolved=[r for r in statuses if r['state'] in ('unresolved', 'blocked')])


def package(config: Config) -> Path:
    root = config.output
    summary = read(root / 'run_summary.json')
    budget = status(config)
    write(root / 'data_delivery_status.json', budget)
    shutil.copyfile((config.ledger_root or root) / 'call_ledger.jsonl', root / 'shared_call_ledger.jsonl')
    (root / 'DATA_README.md').write_text(
        '# Fig4 百篇数据准备\n\n'
        '采用已审阅五篇的联合图／结构数值口径；五篇结果复用，其余论文单次建立全部参考。'
        '本轮不绘图，压缩包不包含旧五篇图件及旧审阅统计。\n\n'
        f'候选{summary["candidate_papers"]}篇，已筛选{summary["screened_papers"]}篇，'
        f'科学纳入{len(summary["included_papers"])}篇，报告{summary["reports"]}份。'
        '科学排除、方面不适用和技术缺失分别记录；未决不算确认正确。\n\n'
        f'共享账本累计{budget["shared_calls"]}次，其中补充{budget["shared_additional"]}次。'
        '生成Sol/medium，参考与评价Sol/high；均为模型评价。\n\n'
        'content_coverage为每篇明确回应且科学内容与必要范围正确的要点数／固定适用要点数；'
        'grounded_coverage进一步要求独立原文、原生图或合法图推导依据。没有合成总分。'
        '先逐论文计算再等权平均；配置绝对比较采用该方面七配置共同完整集合，'
        '差值采用对应完整配对，区间为10000次论文bootstrap。\n\n'
        '- graph_screening.csv、aspect_applicability.csv：筛选与固定分母。\n'
        '- public_tasks/、reference/papers/：共同问题及私有参考。\n'
        '- analysis_inputs/、analysis/、report_inputs/、reports/：实际生成材料与报告。\n'
        '- aligned_evaluation/、answer_parts.json：逐要点评价与报告原句。\n'
        '- paper_aspect_metrics.csv、configuration_summary.csv：论文与配置结果。\n'
        '- paired_effects_summary.csv、paired_changes.json：配对差值及保留／新增／遗漏。\n'
        '- *_without_development.csv：排除原五篇的结果。\n'
        '- run_summary.json、data_delivery_status.json、call_ledger.jsonl：用量与缺失。\n\n'
        'call_ledger.jsonl为本版目录记录；shared_call_ledger.jsonl另含前期试跑请求，'
        '用于核对全程1980次预算，不将旧评分混入本版统计。\n\n'
        '工程核对仅阅读代码与实际产物；未编写或运行回归测试。\n', encoding='utf-8')
    destination = root.parent / 'Fig4_100_data_v2.zip'
    excluded = {'figure', 'logs'}
    old_documents = {'SCIENTIFIC_REVIEW.md', 'PILOT_REVIEW.md', 'index.html'}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if path.is_file() and not (set(relative.parts) & excluded) and path.name not in old_documents:
                archive.write(path, Path('fig4_data_v2') / relative)
        for path in sorted((root / 'logs/calls').glob('*/record.json')):
            archive.write(path, Path('fig4_data_v2') / path.relative_to(root))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'aggregate', 'package', 'status'])
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--stage', choices=['all', 'reference', 'reports', 'evaluate'], default='all')
    args = parser.parse_args()
    config = configuration()
    prompts.GRAPH_DEFINITIONS += '\n' + prompts.STRUCTURE_DEFINITIONS
    if args.command == 'prepare':
        prepare(config)
    elif args.command == 'run':
        from .pipeline import run
        asyncio.run(run(config, False, args.paper_id, args.stage))
    elif args.command == 'aggregate':
        from .aggregate import aggregate
        aggregate(config)
    elif args.command == 'package':
        print(package(config))
    else:
        print(json.dumps(status(config), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
