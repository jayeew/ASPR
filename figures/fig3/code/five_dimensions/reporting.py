"""Render the new report body before the unchanged original analysis attachment."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import DAMAGE_ID, DIMENSIONS
from .io import write
from .models import Report


def render_report(report: dict[str, Any]) -> str:
    value = Report.model_validate(report)
    pieces = ['## 贡献概述', value.contribution_overview]
    dimensions = {item.dimension: item for item in value.dimensions}
    status_labels = {'supported': '有据', 'mixed': '部分支持或存在分歧', 'unresolved': '待核实'}
    for key, label in DIMENSIONS.items():
        row = dimensions[key]
        pieces.extend([f'## {label}', row.judgment, row.rationale,
                       f'**成立范围与判断状态：** {row.scope}（{status_labels[row.status]}）'])
        for source in row.evidence:
            pieces.append(f'> {source.quote}\n\n来源：`{source.source_id}`。{source.explanation}')
    pieces.extend(['## 综合结论', value.synthesis])
    return '\n\n'.join(pieces) + '\n'


def write_report(output: Path, paper_id: str, method: str, report: dict[str, Any]) -> None:
    value = Report.model_validate(report).model_dump(mode='json')
    path = output / 'reports' / method / f'{paper_id}.json'
    write(path, value)
    body = render_report(value)
    with (output / 'inputs/original_reports' / method / f'{paper_id}.md').open(encoding='utf-8', newline='') as handle:
        original = handle.read()
    introduction = '\n---\n\n## 详细分析材料：原分析正文\n\n以下保留原分析正文，不属于上方五维主体的新评分对象。\n\n'
    if paper_id == DAMAGE_ID:
        introduction += '原保存报告的末段存在截断，以下按原样保留；五维主体生成未使用该不完整末段。\n\n'
    for target, text in ((path.with_suffix('.md'), body + introduction + original),
                         (path.with_suffix('.body.md'), body)):
        temporary = target.with_name(target.name + f'.{os.getpid()}.tmp')
        temporary.write_text(text, encoding='utf-8', newline='')
        temporary.replace(target)
