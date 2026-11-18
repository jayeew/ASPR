"""Editable, frequency-sized scientific phrase clouds from saved predictions only."""
from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import Scene, measure
from .data import OUT, read, write
from .predict import CATEGORIES, frequencies

W = 960
INK, MUTED, BORDER = '#202C3B', '#596B7E', '#A9CDE4'
COLORS = ('#1763AD', '#087F86', '#B65C21')


def lines_for(phrase: str, size: float, maximum: float) -> list[str]:
    lines, line = [], ''
    for word in phrase.split():
        trial = (line + ' ' + word).strip()
        if measure(word, size) > maximum:
            raise ValueError(f'Unbreakable scientific token exceeds panel width: {word}')
        if line and measure(trial, size) > maximum:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def intersect(a: tuple[float, ...], b: tuple[float, ...], gap: float = 9) -> bool:
    return not (a[0]+a[2]+gap <= b[0] or b[0]+b[2]+gap <= a[0]
                or a[1]+a[3]+gap <= b[1] or b[1]+b[3]+gap <= a[1])


def cloud_layout(rows: list[dict[str, Any]], width: float, height: float) -> list[dict[str, Any]]:
    """Seeded phrase packing; never shrink or randomly alter font sizes."""
    rng = random.Random(20261003)
    placed: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(rows, key=lambda r: (-r['frequency'], -len(r['phrase']), r['phrase']))):
        vertical = index in {4, 9, 14, 19}
        size = row['font_size_pt']
        wrap_width = max((108, 112, 116)[index % 3],
                         max(measure(word, size) for word in row['phrase'].split()) + 1)
        lines = [row['phrase']] if vertical else lines_for(row['phrase'], size, wrap_width)
        block_w = size * 1.16 if vertical else max(measure(line, size) for line in lines)
        block_h = measure(row['phrase'], size) if vertical else len(lines) * size * 1.16
        xs = {5.0, width-block_w-5, (width-block_w)/2}
        ys = {5.0, height-block_h-5, (height-block_h)/2}
        for item in placed:
            px, py, pw, ph = item['box']
            xs.update((px-block_w-10, px+pw+10, px, px+pw-block_w, px+pw/2-block_w/2))
            ys.update((py-block_h-10, py+ph+10, py, py+ph-block_h))
        candidates = [(x, y) for x in sorted(xs) for y in sorted(ys)
                      if 5 <= x <= width-block_w-5 and 5 <= y <= height-block_h-5]
        target_x = width/2 if row['frequency'] > 1 else width * (.28 if index % 2 else .72)
        if vertical:
            target_x = width * (.08 if index % 2 else .92)
        priorities = {(x, y): ((x+block_w/2-target_x)/(width/2))**2
                      + ((y+block_h/2-height/2)/(width/2))**2 + rng.random()*1e-5
                      for x, y in candidates}
        candidates.sort(key=lambda xy: priorities[xy])
        for x, y in candidates:
            box = (x, y, block_w, block_h)
            if x < 5 or y < 5 or x+block_w > width-5 or y+block_h > height-5:
                continue
            if not any(intersect(box, tuple(p['box'])) for p in placed):
                placed.append({**row, 'box': list(box), 'lines': lines, 'vertical': vertical})
                break
        else:
            raise ValueError('Cloud needs a taller panel at the authorized font sizes')
    return placed


def build_clouds(rows: list[dict[str, Any]]) -> tuple[float, list[list[dict[str, Any]]]]:
    for height in (390, 460, 530, 600, 700, 850, 1000, 1200):
        try:
            layouts = [cloud_layout([r for r in rows if r['displayed'] and r['category'] == cat], 288, height)
                       for cat in CATEGORIES]
            return height, layouts
        except ValueError as exc:
            if 'taller panel' not in str(exc):
                raise
    raise ValueError('Cannot fit all phrases without changing requested font scale')


def draw_header(scene: Scene, summary: dict[str, Any], directions: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    scene.text(16, 30, 'Fig. 7 | Graph-guided forecasts of future research directions', 23, INK, True)
    scene.rect(12, 46, W-24, 125, '#F8FBFE', BORDER, 4)
    scene.text(24, 68, 'a', 22, INK, True)
    scene.text(51, 68, 'Whole-corpus evidence to prospective research themes', 17, INK, True)
    scene.text(28, 94, f'{summary["claims"]:,} claims  /  {summary["papers"]:,} papers  /  {summary["edges"]:,} semantic edges', 15, INK)
    scene.text(28, 115, '2023–2025 corpus', 13, MUTED)
    scene.arrow(176, 112, 220, 112, MUTED, 1.2)
    scene.text(234, 115, 'Activity + cross-group links + method associations', 13, INK)
    scene.arrow(540, 112, 580, 112, MUTED, 1.2)
    scene.text(596, 115, f'{len(directions)} retained future directions', 13, INK, True)
    if directions:
        selected = sorted(directions, key=lambda d: (len(d['direction']), d['candidate_id']))[0]
        refs = ' + '.join(dict.fromkeys(evidence[c['source_id']]['paper_id'] for c in selected['citations']))
        scene.paragraph(28, 136, f'Example {selected["candidate_id"]}: {selected["direction"]}', W-58, 11, 13, max_lines=2)
        scene.text(28, 163, f'Source papers: {refs}', 9, MUTED)


def draw_panels(scene: Scene, height: float, layouts: list[list[dict[str, Any]]]) -> None:
    descriptions = ('Prospective methods and technical routes', 'Specific problems for further investigation',
                    'Proposed combinations of scientific approaches')
    for i, (category, color, layout) in enumerate(zip(CATEGORIES, COLORS, layouts)):
        x, y = 12 + i*316, 184
        scene.rect(x, y, 304, height+79, '#FFFFFF', BORDER, 4)
        scene.rect(x+1, y+1, 302, 38, '#F1F7FC')
        scene.text(x+12, y+27, 'bcd'[i], 22, INK, True)
        scene.text(x+40, y+26, category, 21, color, True)
        scene.text(x+12, y+55, descriptions[i], 10, MUTED)
        for item in layout:
            bx, by, bw, _ = item['box']
            size = item['font_size_pt']
            if item['vertical']:
                _, _, _, bh = item['box']
                local = Scene(bh, bw, f'phrase_{i}_{len(scene.text_boxes)}')
                local.text(0, size*.34, item['phrase'], size, color, anchor='middle')
                local.root.set('transform', f'translate({x+8+bx+bw/2} {y+65+by+bh/2}) rotate(-90)')
                scene.add(local.root)
                scene.text_boxes.append({'text': item['phrase'], 'x': x+8+bx, 'y': y+65+by,
                                         'width': bw, 'height': bh, 'font_size': size})
            else:
                for j, line in enumerate(item['lines']):
                    scene.text(x+8+bx+bw/2, y+65+by+size*.83+j*size*1.16,
                               line, size, color, anchor='middle')
        if not layout:
            scene.text(x+152, y+height/2, 'No eligible phrases', 14, MUTED, anchor='middle')


def usage_summary() -> dict[str, Any]:
    records = [read(p) for p in sorted((OUT/'logs/calls').glob('*/record.json'))]
    known = [r for r in records if r.get('usage') is not None]
    return {'recorded_attempts': len(records),
            'completed_calls': sum(r['state'] == 'completed' for r in records),
            'calls_without_reported_usage': len(records)-len(known),
            'input_tokens_known': sum(r['usage'].get('input_tokens', 0) for r in known),
            'output_tokens_known': sum(r['usage'].get('output_tokens', 0) for r in known),
            'cached_input_tokens_known': sum(r['usage'].get('cached_input_tokens', 0) for r in known),
            'call_seconds_known': sum(r.get('seconds', 0) for r in records),
            'money_cost': None,
            'note': 'Token usage is provider-reported; unknown usage is not zero. No monetary conversion.'}


def methods_text(summary: dict[str, Any], directions: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    calls = [json.loads(s) for s in (OUT/'calls.jsonl').read_text().splitlines()]
    usage = usage_summary()
    recovery = ''
    if (OUT/'qa/consolidation_recovery.json').exists():
        info = read(OUT/'qa/consolidation_recovery.json')
        recovery = (f"跨包邻接社区及边端点 ID 校验修正后，{info['restored_candidates']} 条原始候选恢复进入完整清单；"
                    f"预留的一次额外调用用于重新归并全部 {info['corrected_source_checked_candidates']} 条候选，原始预测响应保持不变。")
    return f'''# Fig7：全量 Claim Graph 驱动的未来方向词云

数据截至 2025-12-31：{summary['claims']:,} 条摘要 claims，{summary['papers']:,} 篇论文，
{summary['edges']:,} 条 cosine > 0.5 且时间有效、排除同论文的语义边。
全部节点参加统计；{summary['groups']:,} 个材料分组全部进入八个预测包。
其中 {summary['communities']:,} 个为固定的全时期社区，未分配社区的 {summary['unassigned_claims']:,} 条
claims 按已有主题元数据组织。后者不作为新发现的社区。

本地计算年度论文占比（分母为对应年份语料中的全部有 claims 论文）、实际跨组连接，
及至少一个 METHOD 端点的关联。边的年份为较晚端点年份；跨组连接比例以该组当年较晚端点
发出的全部保留边为分母。材料中的方法关联包括方法端点位于邻接组的情况，不等于方法已迁移。
同一论文可涉及多个分组，因此各组论文占比不必加和为一。
年度占比不能完全消除期刊构成和收录覆盖变化，后期节点拥有更大的历史邻居池。
社区属于固定的全时期划分，年度统计不构成独立的历史社区检测。

为适应上下文，所有分组均有统计摘要，模型读取代表性原文而非全部七万条原文。
代表例优先近期、跨组关联和方法类，按论文去重；小组保留一个代表及实际邻接边端点，
较大组增加代表。分包按长度均衡，保留跨包邻接主题和边端点。小组边的时间、端点和余弦可追溯。
本轮 gpt-6.1-sol / medium 实际启动 {len(calls)} 次调用（最多 10 次）；
每包最多 12 个候选，全局合并同义词并排除重复、复述结果或依据不足的方向。
最终保留 {len(directions)} 条未来方向。模型是当前模型，并未模拟历史时点可用模型。
{recovery}
已记录输入 tokens：{usage['input_tokens_known']:,}，输出 tokens：{usage['output_tokens_known']:,}；
{usage['calls_without_reported_usage']} 次调用未报告用量。以上不换算为货币费用。
这些是来源约束下的前瞻性假说，不是经过回测的预测准确率，也不是全科学文献覆盖。

## 词云

三类分别表示 Method、Problem、Combination。每类显示频次最高的最多 20 个完整短语，
并列以固定随机种子抽选，避免首字母顺序偏向。频次为包含标准化短语的不同保留方向数；同一方向最多一次。
字号 = 8 + 10 × sqrt(频次 / 全部展示短语最大频次) pt。三图共同尺度，
不按置信度、原论文词频或支持论文数加权；色彩仅区分类别。
固定随机种子 20261003；各类别同频抽选使用该种子加类别序号。
短语完整保留，以横排为主、少量纵排，不缩写或随机放大。

## English caption

**Fig. 7 | Graph-guided forecasts of future research directions.**
(a) Whole-corpus aggregation of the 2023–2025 abstract-derived Claim Graph and a source-linked
forecast example. All nodes enter aggregation; representative claim texts and complete group
summaries form eight model input packets. Fixed graph communities and topic-organized unassigned
claims provide context for activity, cross-group semantic connectivity and method associations.
(b–d) Method, Problem and Combination phrases from {len(directions)} retained prospective directions.
Font size encodes the number of distinct retained directions containing each normalized phrase,
using a common square-root scale; it does not encode forecast probability. At most 20 phrases per
category are displayed, with seeded selection among frequency ties. Semantic edges motivate interpretation, not causal or support claims.
Predictions are hypotheses grounded in the available corpus, not retrospectively validated outcomes.

## Files and reproduction

`tables/predicted_directions.csv`: future statements, graph rationale, exact cited claim quotations and labels.
`tables/direction_sources.csv`: original claim IDs, article IDs, DOI, source fragments and each direction mapping.
`tables/wordcloud_frequencies.csv`: all phrases, direction IDs, counts, font sizes and display inclusion.
`data/group_profiles.json`: graph observations referenced by each forecast.
`data/coverage.json`: packet coverage and input sizes.
`logs/calls/*/record.json`: measured timing and usage where available; missing usage is not zero.

From the repository root, with the project Python environment:

```bash
python -m figure_pipeline.fig7_reference prepare
python -m figure_pipeline.fig7_reference predict
python -m figure_pipeline.fig7_reference render
```

Predict reuses saved results. Render is offline and never starts model calls. The ten-call ledger
is shared across resumed execution. Old design-prototype outputs are separate from this forecast directory.
'''


def render() -> None:
    directions = read(OUT/'data/directions.json')
    rows = read(OUT/'data/wordcloud_frequencies.json')
    assert rows == frequencies(directions), 'Saved phrase counts differ from prediction records'
    summary = read(OUT/'data/summary.json')
    evidence = read(OUT/'data/evidence.json')
    height, layouts = build_clouds(rows)
    total_height = height+330
    scene = Scene(W, total_height, 'fig7_forecast')
    scene.rect(0, 0, W, total_height, '#FFFFFF')
    draw_header(scene, summary, directions, evidence)
    draw_panels(scene, height, layouts)
    scene.text(16, total_height-43, 'Word size: number of distinct predicted directions containing each phrase; common scale across panels.', 12, INK)
    scene.text(16, total_height-24, 'Prospective hypotheses from the available corpus • Size is not forecast probability • Up to 20 phrases per category', 11, MUTED)
    final = OUT/'final/Fig7.svg'
    scene.save(final)
    root = ET.parse(final)
    root.getroot().set('width', f'{W}pt')
    root.getroot().set('height', f'{total_height}pt')
    root.write(final, encoding='utf-8', xml_declaration=True)
    fonts(OUT)
    convert(final, 1.5)
    outline(final, OUT/'final/Fig7_outlined.svg')
    import cairosvg
    from PIL import Image
    high = OUT/'final/Fig7_600dpi.png'
    cairosvg.svg2png(url=str(final), write_to=str(high), output_width=round(W/72*600),
                    output_height=round(total_height/72*600))
    with Image.open(high) as img:
        img.save(high, dpi=(600, 600))
    boxes = scene.text_boxes
    assert all(b['x'] >= 0 and b['y'] >= 0 and b['x']+b['width'] <= W+.01
               and b['y']+b['height'] <= total_height for b in boxes)
    write(OUT/'qa/layout.json', {'width_pt': W, 'height_pt': total_height, 'seed': 20261003,
          'clouds': layouts, 'all_text_inside_canvas': True, 'font_sizes_match_counts': True})
    write(OUT/'data/usage_summary.json', usage_summary())
    (OUT/'methods_and_caption.md').write_text(methods_text(summary, directions, rows), encoding='utf-8')
    with zipfile.ZipFile(OUT/'Fig7_forecast_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('final', 'tables'):
            for path in sorted((OUT/folder).glob('*')):
                archive.write(path, path.relative_to(OUT))
        for name in ('methods_and_caption.md', 'data/directions.json', 'data/group_profiles.json',
                     'data/coverage.json', 'data/summary.json', 'data/wordcloud_frequencies.json',
                     'data/usage_summary.json', 'data/rejected_candidates.json', 'predictions/merge.json'):
            archive.write(OUT/name, name)
    print(f'Exported {final}; {len(directions)} directions, {sum(r["displayed"] for r in rows)} displayed phrases', flush=True)
