from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.storage import read

from .models import ASPECTS, NAMES, Config


def pct(value: float | None) -> str:
    return 'NA' if value is None else f'{100 * value:.1f}%'


def review(config: Config) -> Path:
    root = config.output
    summary = read(root / 'run_summary.json')
    absolute = read(root / 'configuration_summary.json')
    effects = read(root / 'paired_effects_summary.json')
    transitions = read(root / 'paired_changes.json')
    lines = ['# Fig4 新版试跑审阅', '',
             '范围：原五篇开发样本；剩余95篇尚待用户审阅。所有评分均为模型评价，不是专家标签。', '',
             f"已准备 {summary['candidate_papers']} 篇，科学筛选 {summary['screened_papers']} 篇；"
             f"已生成 {summary['reports']} 份报告，启动 {summary['calls_started']} 次请求。", '',
             '生成：gpt-6.1-sol / medium；参考与评价：gpt-6.1-sol / high。', '',
             '## 七配置共同完整样本', '', '|配置|方面|正确覆盖率|外部证据落实率|共同n|', '|---|---|---:|---:|---:|']
    grounded = {(r['condition'], r['aspect']): r for r in absolute if r['metric'] == 'grounded_coverage'}
    for row in absolute:
        if row['metric'] == 'content_coverage':
            ext = grounded[(row['condition'], row['aspect'])]
            lines.append(f"|{row['configuration']}|{ASPECTS[row['aspect']]}|{pct(row['estimate'])}|{pct(ext['estimate'])}|{row['n']}|")
    lines.extend(['', '## 加入分支与组件的配对差值', '', '|比较|方面|差值（百分点）|95%区间|配对n|提高/持平/降低|', '|---|---|---:|---|---:|---|'])
    for row in effects:
        if row['metric'] != 'content_delta':
            continue
        estimate = 'NA' if row['estimate'] is None else f"{100 * row['estimate']:+.1f}"
        interval = 'NA' if row['low'] is None else f"[{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]"
        lines.append(f"|{row['contrast']}|{ASPECTS[row['aspect']]}|{estimate}|{interval}|{row['n']}|{row['improved']}/{row['tied']}/{row['reduced']}|")
    lines.extend(['', '## 完整系统遗漏或下降的逐要点记录', '', '以下逐项保留原句与评价理由；不把模型判定自动当作最终科学结论。'])
    for item in transitions:
        if item['transition'] != 'lost':
            continue
        lines.extend(['', f"### {item['paper_id']} · {item['contrast']} · {item['part_id']}",
                      item['question'], '', '**完整系统原句**', *item['full']['report_quotes'],
                      '', '**对照原句**', *item['comparison']['report_quotes'],
                      '', '**评价理由**', item['full']['reason'], item['comparison']['reason']])
    lines.extend(['', '## 未解决任务', '', '```json', json.dumps(summary['unresolved_tasks'], ensure_ascii=False, indent=2), '```', '',
                  '## 审阅文件', '', '- public_tasks/：七配置共用问题；reference/papers/：私有科学参考。',
                  '- report_inputs/、analysis_inputs/：实际模型输入；reports/：完整报告。',
                  '- aligned_evaluation/：匿名评分、原段落定位与原句。',
                  '- answer_parts.json、paired_changes.json：逐要点与保留／新增／遗漏。',
                  '- input_information.csv：联合图独有边、记录到的引用接触及可重建性。',
                  '- call_ledger.jsonl、run_summary.json：失败请求也计数；未知用量不填零。', '',
                  '确认任务合理、证据对应和评分尺度一致后，才追加95篇。完整系统是否获胜不是技术验收条件。'])
    path = root / 'PILOT_REVIEW.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def package(config: Config) -> Path:
    root = config.output
    review_path = review(config)
    body = html.escape(review_path.read_text())
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fig4 pilot review</title>'
        '<style>body{font:15px/1.7 sans-serif;max-width:1200px;margin:40px auto;color:#243547}pre{white-space:pre-wrap}img{width:100%}</style>'
        '<h1>Fig4 原五篇审阅</h1><a href="SCIENTIFIC_REVIEW.md">科学结论与待商议口径</a> · '
        '<a href="PILOT_REVIEW.md">评分与原句</a> · <a href="configuration_summary.csv">配置源表</a>'
        '<img src="figure/Fig4_preview.png"><pre>' + body + '</pre>', encoding='utf-8')
    filename = 'Fig4_joint_structure_refinement_pilot.zip' if config.focused_refinement else 'Fig4_explanation_100_pilot_review.zip'
    destination = root.parent / filename
    allowed = {'.json', '.jsonl', '.csv', '.md', '.html', '.svg', '.pdf', '.png'}
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix not in allowed or 'logs' in path.relative_to(root).parts:
                continue
            archive.write(path, Path(root.name) / path.relative_to(root))
        # Execution summaries are sufficient for cost/state auditing; internal model event streams stay local.
        for path in sorted((root / 'logs/calls').glob('*/record.json')):
            archive.write(path, Path(root.name) / path.relative_to(root))
    return destination
