"""Fixed-core rates, paper pairing and reused bootstrap; no information statistics."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from figure_pipeline.fig3_revision.aggregate import bootstrap, write_csv
from figure_pipeline.fig3_revision.storage import read, write

from .config import CONDITIONS, CONTRASTS, METRICS, NAMES, Config
from .engine import adopted_entries, ledger


def status(config: Config) -> dict[str, Any]:
    pilot = read(config.output / 'pilot.json')['paper_ids']
    ids = [p['paper_id'] for p in read(config.output / 'papers.json')]
    rows = ledger(config.output)
    adopted = adopted_entries(rows, config)
    stages = {}
    for stage, directory in [('generate', 'reports'), ('evaluate', 'evaluation')]:
        complete = [(p, c) for p in ids for c in CONDITIONS
                    if (config.output / directory / c / f'{p}.json').exists()]
        stages[stage] = {'complete': len(complete), 'pilot_complete': sum(p in pilot for p, _ in complete)}
    records = [read(p) for p in (config.output / 'logs/calls').glob('*/record.json')]
    log_path = config.output / 'run_log.jsonl'
    events = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    last = {(r['paper_id'], r['condition'], r['stage']): r for r in events}
    return {'evaluation_round': config.evaluation_round,
            'call_limit': 1400, 'calls_started': len(adopted),
            'historical_calls_started': len(rows), 'superseded_calls_started': len(rows) - len(adopted),
            'pilot_calls_started': sum(r['paper_id'] in pilot for r in adopted),
            'remaining_cohort_calls_started': sum(r['paper_id'] not in pilot for r in adopted),
            'calls_remaining': 1400 - len(adopted), 'call_states': dict(Counter(r['state'] for r in records)),
            **stages, 'active': [{'paper_id': r['paper_id'], 'condition': r['method'], 'stage': r['stage'],
                                 'pid': r.get('pid'), 'seconds': r.get('seconds')} for r in records if r['state'] == 'running'],
            'usage': {k: sum((r.get('usage') or {}).get(k, 0) for r in records)
                      for k in ('input_tokens', 'cached_input_tokens', 'output_tokens')},
            'usage_missing_calls': sum(r.get('usage') is None for r in records),
            'failed_objects': [r for r in last.values() if r['state'] == 'failed']}


def paper_metrics(config: Config, paper: str, condition: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs = read(config.output / 'inputs/evaluation' / f'{paper}.json')
    reference = {r['core_id']: r for r in inputs['reference']}
    report_path = config.output / 'reports' / condition / f'{paper}.json'
    eval_path = config.output / 'evaluation' / condition / f'{paper}.json'
    row = {'paper_id': paper, 'condition': condition, 'name': NAMES[condition],
           'C': None, 'H': None, 'S': None, 'body_chars': None,
           'C_denominator': len(inputs['cores']),
           'H_denominator': sum(r['historical_comparison_applicable'] for r in reference.values()),
           'S_denominator': sum(r['scope_applicable'] for r in reference.values()), 'state': 'pending'}
    if report_path.exists():
        row['body_chars'] = read(report_path)['body_chars']
    if not eval_path.exists():
        attempts = [r for r in ledger(config.output) if r['paper_id'] == paper and r['condition'] == condition]
        if attempts:
            row['state'] = 'incomplete'
        return row, []
    items = read(eval_path)['items']
    count = {'C': sum(r['addressed'] for r in items),
             'H': sum(r['addressed'] and r['historical_comparison_correct'] is True and r['scope_correct'] is True
                      for r in items if reference[r['core_id']]['historical_comparison_applicable']),
             'S': sum(r['addressed'] and r['scope_correct'] is True
                      for r in items if reference[r['core_id']]['scope_applicable'])}
    row.update(state='complete')
    for metric in METRICS:
        row[metric + '_numerator'] = count[metric]
        denominator = row[metric + '_denominator']
        row[metric] = 100 * count[metric] / denominator if denominator else None
    row['unknown_history_cores'] = sum(r['addressed'] and r['historical_comparison_correct'] is None
                                     for r in items if reference[r['core_id']]['historical_comparison_applicable'])
    row['unknown_scope_cores'] = sum(r['addressed'] and r['scope_correct'] is None
                                   for r in items if reference[r['core_id']]['scope_applicable'])
    return row, [{'paper_id': paper, 'condition': condition, **r, 'reference': reference[r['core_id']]} for r in items]


def aggregate(config: Config) -> dict[str, Any]:
    roster = read(config.output / 'papers.json')
    ids = [r['paper_id'] for r in roster]
    rows, core_rows = [], []
    for paper in ids:
        for condition in CONDITIONS:
            row, items = paper_metrics(config, paper, condition)
            rows.append(row)
            core_rows.extend(items)
    keyed = {(r['paper_id'], r['condition']): r for r in rows}
    write_csv(config.output / 'paper_metrics.csv', rows)
    write(config.output / 'core_evaluations.json', core_rows)
    means = []
    for condition in CONDITIONS:
        chosen = [r for r in rows if r['condition'] == condition and r['state'] == 'complete']
        item = {'condition': condition, 'name': NAMES[condition], 'n_papers': len(chosen)}
        item.update({m: float(np.mean([r[m] for r in chosen if r[m] is not None]))
                     if any(r[m] is not None for r in chosen) else None for m in METRICS})
        item['body_chars_median'] = float(np.median([r['body_chars'] for r in chosen])) if chosen else None
        means.append(item)
    write_csv(config.output / 'condition_summary.csv', means)
    common_ids = [p for p in ids if all(keyed[p, c]['state'] == 'complete' for c in CONDITIONS)]
    common_means = []
    for condition in CONDITIONS:
        chosen = [keyed[p, condition] for p in common_ids]
        common_means.append({'condition': condition, 'name': NAMES[condition], 'n_papers': len(chosen),
                             **{m: float(np.mean([r[m] for r in chosen])) if chosen else None for m in METRICS}})
    write_csv(config.output / 'condition_summary_common.csv', common_means)
    effects, summaries = [], []
    for component, other in CONTRASTS.items():
        for metric in METRICS:
            complete = []
            for paper in ids:
                f, control = keyed[paper, 'F'][metric], keyed[paper, other][metric]
                delta = f - control if f is not None and control is not None else None
                effects.append({'paper_id': paper, 'component': component, 'metric': metric,
                                'full': f, 'control': control, 'delta': delta})
                if delta is not None:
                    complete.append(delta)
            stats = bootstrap(complete, config)
            quartiles = np.percentile(complete, [25, 50, 75]) if complete else [None] * 3
            summaries.append({'component': component, 'metric': metric, 'mean_delta': stats['estimate'],
                              'ci_low': stats['low'], 'ci_high': stats['high'], 'n': stats['n'],
                              'q1': quartiles[0], 'median': quartiles[1], 'q3': quartiles[2]})
    write_csv(config.output / 'component_effects_paper.csv', effects)
    write_csv(config.output / 'component_effects_summary.csv', summaries)
    interaction, joint = [], []
    for paper in ids:
        values = [keyed[paper, c]['H'] for c in ('T', 'E', 'G', 'F')]
        interaction.append({'paper_id': paper, 'T': values[0], 'E': values[1], 'G': values[2], 'F': values[3],
                            'I': values[3] - values[1] - values[2] + values[0] if all(v is not None for v in values) else None})
        structure = read(config.output / 'inputs/joint_structure' / f'{paper}.json')
        f, no_joint = keyed[paper, 'F']['H'], keyed[paper, 'F_noJ']['H']
        joint.append({'paper_id': paper, 'J_topo': structure['J_topo'], 'H_F': f, 'H_noJ': no_joint,
                      'delta_H': f - no_joint if f is not None and no_joint is not None else None})
    write_csv(config.output / 'interaction_paper.csv', interaction)
    valid = [r['I'] for r in interaction if r['I'] is not None]
    common = [r for r in interaction if r['I'] is not None]
    write(config.output / 'interaction_summary.json', {'metric': 'H', 'statistics': bootstrap(valid, config),
          'condition_means': {c: float(np.mean([r[c] for r in common])) if common else None for c in ('T', 'E', 'G', 'F')},
          'positive': sum(v > 0 for v in valid), 'zero': sum(v == 0 for v in valid), 'negative': sum(v < 0 for v in valid)})
    write_csv(config.output / 'joint_effects_paper.csv', joint)
    examples = []
    for effect in effects:
        if effect['delta'] is not None and effect['delta'] < 0:
            paper, metric = effect['paper_id'], effect['metric']
            other = CONTRASTS[effect['component']]
            examples.append({**effect, 'control_name': NAMES[other],
                             'full_evaluation': read(config.output / 'evaluation/F' / f'{paper}.json')['items'],
                             'control_evaluation': read(config.output / 'evaluation' / other / f'{paper}.json')['items']})
    write(config.output / 'full_lower_examples.json', examples)
    write_csv(config.output / 'paper_order.csv', sorted(
        [{'paper_id': r['paper_id'], 'field': r.get('field_name', '')} for r in roster], key=lambda r: (r['field'], r['paper_id'])))
    summary = status(config)
    write(config.output / 'summary.json', summary)
    pilot_review(config, means, common_means, summary, examples)
    return summary


def pilot_review(config: Config, means: list[dict[str, Any]], common_means: list[dict[str, Any]],
                 summary: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    pilot_limit = 70
    main_cohort = summary['remaining_cohort_calls_started'] > 0
    lines = ['# Fig4 精简重跑：' + ('100篇采纳版本运行摘要' if main_cohort else '试跑审阅'), '',
             f"评价版本：{summary['evaluation_round']}。",
             f"采纳版本任务启动：{summary['calls_started']}/1400；采纳试跑：{summary['pilot_calls_started']}/{pilot_limit}。",
             f"全部历史任务启动：{summary['historical_calls_started']}；被替代的旧评价：{summary['superseded_calls_started']}。",
             f"试跑报告完成：{summary['generate']['pilot_complete']}/35；评价完成：{summary['evaluate']['pilot_complete']}/35。", '',
             f"全队列报告完成：{summary['generate']['complete']}/700；评价完成：{summary['evaluate']['complete']}/700。", '',
             '|配置|有效论文数|回应覆盖率|正确历史比较覆盖率|范围正确回应率|正文中位字数|',
             '|---|---:|---:|---:|---:|---:|']
    for row in means:
        values = [str(row['n_papers'])] + [f"{row[m]:.2f}%" if row[m] is not None else 'NA' for m in METRICS]
        values.append(str(row['body_chars_median']) if row['body_chars_median'] is not None else 'NA')
        lines.append('|' + row['name'] + '|' + '|'.join(values) + '|')
    lines.extend(['', '上表各配置有效论文集合不同，不用于直接排名；组件效应使用各自完整配对。技术缺失不填零。', '',
                  '## 七配置共同完整论文', '', '|配置|共同论文数|回应覆盖率|正确历史比较覆盖率|范围正确回应率|',
                  '|---|---:|---:|---:|---:|'])
    for row in common_means:
        values = [str(row['n_papers'])] + [f"{row[m]:.2f}%" if row[m] is not None else 'NA' for m in METRICS]
        lines.append('|' + row['name'] + '|' + '|'.join(values) + '|')
    cohort_note = ('用户已采纳最新版5篇并授权剩余95篇。技术缺失保留，不自动增加请求。'
                   if main_cohort else '试跑后暂停，等待用户审阅，未启动剩余95篇。')
    lines.extend(['', cohort_note, '', '## 未完成对象', ''])
    for row in summary['failed_objects']:
        lines.append(f"- {row['paper_id']} / {NAMES[row['condition']]}：{row['error']}。原始响应保留，未新增请求。")
    lines.extend(['',
                  '## 完整系统低于消融的逐篇记录', ''])
    for example in examples:
        lines.extend([f"### {example['paper_id']} · {METRICS[example['metric']]} · 对照：{example['control_name']}",
                      f"完整：{example['full']:.2f}%；对照：{example['control']:.2f}%；差：{example['delta']:.2f}个百分点。"])
        for label, key in [('完整系统', 'full_evaluation'), (example['control_name'], 'control_evaluation')]:
            for item in example[key]:
                lines.extend([f"- {label} / {item['core_id']}：回应={item['addressed']}；历史正确={item['historical_comparison_correct']}；范围正确={item['scope_correct']}。",
                              '  原句：' + ' / '.join(item['report_quotes']), '  理由：' + item['reason'],
                              '  未决：' + item['unresolved_reason']])
    if not examples:
        lines.append('当前没有已完成配对中完整系统低于消融的记录；缺失配对不作为胜出。')
    (config.output / 'pilot_review.md').write_text('\n'.join(lines) + '\n')
