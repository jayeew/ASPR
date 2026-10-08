"""Single glyph-packed word cloud with source-derived prominence, no model calls."""
from __future__ import annotations

import io
import math
from functools import lru_cache
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFont
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import Scene, font_path
from .data import OUT, read, write
from .predict import CATEGORIES, csv_write, frequencies

W, H = 1200, 870
BASE = {'Method': '#124F93', 'Problem': '#087B79', 'Combination': '#C65C19'}


def prominence_rows() -> list[dict[str, Any]]:
    directions = read(OUT/'data/directions.json')
    by_id = {d['candidate_id']: d for d in directions}
    sources = read(OUT/'data/evidence.json')
    rows = frequencies(directions)
    aliases = read(Path(__file__).with_name('short_labels.json'))
    for row in rows:
        papers = {sources[c['source_id']]['paper_id'] for k in row['direction_ids']
                  for c in by_id[k]['citations']}
        row['source_paper_ids'] = sorted(papers)
        row['source_paper_count'] = len(papers)
        row['prominence'] = row['frequency'] * math.log2(1 + len(papers))
        row['displayed'] = False
    selected = []
    for category in CATEGORIES:
        group = [r for r in rows if r['category'] == category]
        random.Random(20261003).shuffle(group)
        group.sort(key=lambda r: -r['prominence'])
        selected.extend(group[:24])
    lo, hi = min(r['prominence'] for r in rows), max(r['prominence'] for r in rows)
    selected_keys = {(r['category'],r['phrase']) for r in selected}
    for row in rows:
        row['displayed'] = (row['category'],row['phrase']) in selected_keys
        row['display_label'] = aliases.get(row['phrase'])
        if row['displayed']:
            assert row['display_label'] is not None
        weight = ((row['prominence']-lo)/(hi-lo))**.45
        row['font_size_pt'] = round(12 + 48 * weight)
        row['bold'] = row['prominence'] >= 4
        base = BASE[row['category']]
        lightness = .32 * (1-weight)
        rgb = [int(base[i:i+2], 16) for i in (1, 3, 5)]
        row['color'] = '#' + ''.join(f'{round(c*(1-lightness)+255*lightness):02x}' for c in rgb)
    return rows


@lru_cache(maxsize=400)
def glyph_mask(label: str, size: float, bold: bool, vertical: bool) -> tuple[np.ndarray, tuple[int, ...]]:
    import cairosvg
    font = ImageFont.truetype(str(font_path(bold)), round(size))
    canvas_w, canvas_h = math.ceil(font.getlength(label)) + 128, round(size*2)+64
    baseline = round(size)+24
    svg = ET.Element('svg', xmlns='http://www.w3.org/2000/svg', width=str(canvas_w), height=str(canvas_h))
    text = ET.SubElement(svg, 'text', x='32', y=str(baseline), **{
        'font-family': 'Times New Roman', 'font-size': str(round(size)),
        'font-weight': 'bold' if bold else 'normal', 'fill': '#000000'})
    text.text = label
    raw = cairosvg.svg2png(bytestring=ET.tostring(svg))
    alpha = Image.open(io.BytesIO(raw)).getchannel('A')
    left, top, right, bottom = alpha.getbbox()
    crop = alpha.crop((left, top, right, bottom))
    canvas = Image.new('L', (crop.width+8, crop.height+8), 0)
    canvas.paste(crop, (4, 4))
    if vertical:
        canvas = canvas.transpose(Image.Transpose.ROTATE_90)
    bounds = (left-32, top-baseline, right-32, bottom-baseline)
    return binary_dilation(np.asarray(canvas) > 8, iterations=3), bounds


def pack(rows: list[dict[str, Any]], width: int, height: int, gap: int = 0,
         rectangular: bool = False) -> list[dict[str, Any]]:
    yy, xx = np.mgrid[:height, :width]
    occupied = ((xx-width/2)/(width*.49))**2 + ((yy-height/2)/(height*.48))**2 > 1
    if rectangular:
        occupied = (xx<8) | (xx>=width-8) | (yy<8) | (yy>=height-8)
    rng = np.random.default_rng(20261003)
    ordered = [r.copy() for r in rows if r['displayed']]
    random.Random(20261003).shuffle(ordered)
    ordered.sort(key=lambda r: (-r['prominence'], -len(r['display_label'])))
    placed = []
    for index, row in enumerate(ordered):
        orientations = [False, True] if index % 7 else [True, False]
        if row['prominence'] >= 4 or row.get('horizontal_only',False):
            orientations = [False]
        for vertical in orientations:
            mask, bounds = glyph_mask(row['display_label'], row['font_size_pt'], row['bold'], vertical)
            if gap:
                mask = np.pad(mask,gap)
            mh, mw = mask.shape
            if mh > height or mw > width:
                continue
            overlap = fftconvolve(occupied.astype(np.float32), mask[::-1, ::-1].astype(np.float32), mode='valid')
            ys, xs = np.where(overlap < .25)
            if not len(xs):
                continue
            costs = ((xs+mw/2-width/2)/(width/2))**2 + ((ys+mh/2-height/2)/(height/2))**2
            costs += rng.uniform(0, .035, len(xs))
            best = int(np.argmin(costs))
            x, y = int(xs[best]), int(ys[best])
            assert not np.any(occupied[y:y+mh, x:x+mw] & mask)
            occupied[y:y+mh, x:x+mw] |= mask
            placed.append({**row, 'x': x+gap, 'y': y+gap, 'width': mw-2*gap, 'height': mh-2*gap,
                           'bounds': list(bounds), 'vertical': vertical})
            break
        else:
            raise ValueError(f'Cannot pack phrase at its encoded size: {row["display_label"]}')
    return placed


def draw_words(scene: Scene, layout: list[dict[str, Any]], ox: float, oy: float) -> None:
    for i, row in enumerate(layout):
        left, top, right, bottom = row['bounds']
        glyph_width, glyph_height = right-left+8, bottom-top+8
        local = Scene(glyph_width, glyph_height, f'keyword_{i:03d}')
        # The same integer font size used for packing is exported, avoiding metric drift.
        local.text(4-left, 4-top, row['display_label'], round(row['font_size_pt']),
                   row['color'], row['bold'])
        x, y = ox+row['x'], oy+row['y']
        transform = (f'translate({x} {y+glyph_width}) rotate(-90)' if row['vertical']
                     else f'translate({x} {y})')
        local.root.set('transform', transform)
        title = ET.SubElement(local.root, '{http://www.w3.org/2000/svg}title')
        title.text = (f'{row["category"]}: {row["phrase"]}\n{row["frequency"]} directions; '
                      f'{row["source_paper_count"]} source papers; prominence {row["prominence"]:.3f}')
        scene.add(local.root)


def render() -> None:
    fonts(OUT)
    rows = prominence_rows()
    for height in (680, 760, 840, 920):
        try:
            layout = pack(rows, 1120, height)
            break
        except ValueError:
            if height == 920:
                raise
    canvas_h = height + 180
    scene = Scene(W, canvas_h, 'fig7_unified_wordcloud')
    scene.rect(0, 0, W, canvas_h, '#FFFFFF')
    scene.text(40, 43, 'Graph-guided future research directions', 29, '#1E2B3C', True)
    scene.text(40, 67, '2023–2025 Claim Graph  ·  70,034 claims  ·  95 retained forecasts', 13, '#647184')
    for i, category in enumerate(CATEGORIES):
        x = 40+i*147
        scene.rect(x, 88, 11, 11, BASE[category], radius=2)
        scene.text(x+18, 98, category, 13, BASE[category], True)
    scene.text(1160, 98, 'Larger + darker = higher evidence-weighted prominence', 12, '#647184', anchor='end')
    draw_words(scene, layout, 40, 117)
    scene.text(40, canvas_h-24, 'Prominence = direction frequency × log₂(1 + distinct source papers).  Short labels map to full sourced predictions.', 11, '#647184')
    final = OUT/'final/Fig7.svg'
    scene.save(final)
    tree = ET.parse(final)
    tree.getroot().set('width', f'{W}pt')
    tree.getroot().set('height', f'{canvas_h}pt')
    tree.write(final, encoding='utf-8', xml_declaration=True)
    fonts(OUT)
    convert(final, 1.4)
    outline(final, OUT/'final/Fig7_outlined.svg')
    import cairosvg
    high = OUT/'final/Fig7_600dpi.png'
    cairosvg.svg2png(url=str(final), write_to=str(high), output_width=round(W/72*600),
                    output_height=round(canvas_h/72*600))
    with Image.open(high) as image:
        image.save(high, dpi=(600, 600))
    write(OUT/'data/unified_wordcloud.json', rows)
    csv_write(OUT/'tables/unified_wordcloud.csv', [{k: r.get(k) for k in (
        'category','phrase','display_label','frequency','direction_ids','source_paper_count',
        'source_paper_ids','prominence','font_size_pt','color','displayed')} for r in rows])
    write(OUT/'qa/unified_layout.json', {'width_pt': W, 'height_pt': canvas_h, 'layout': layout,
          'pixel_collision_check_passed': True, 'seed': 20261003})
    update_notes(rows)
    package()
    print(f'Unified cloud: {len(layout)} labels, {W} × {canvas_h} pt; no new model calls', flush=True)


def update_notes(rows: list[dict[str, Any]]) -> None:
    path = OUT/'methods_and_caption.md'
    original = path.read_text(encoding='utf-8')
    before = original.split('## 词云')[0]
    after = original.split('## Files and reproduction')[1]
    section = '''## 词云（合并版）

按用户修订要求，三类合并为一张词云。蓝色 Method，青色 Problem，橙色 Combination。
图中使用简短主题标签；完整预测短语、方向 ID 和来源论文在 unified_wordcloud.csv 中对应保留。
短标签是展示缩写，不改变预测正文或新增预测。同频、同论文覆盖的标签具有相同字号及深浅。

显著度 S = 方向频次 × log₂(1 + 不同来源论文数)。来源论文为关联方向所引论文的去重并集。
该指标仅用于本图的相对视觉强调，不是经验证的科学重要性、预测成功概率或因果证据强度。
每类按 S 选取 24 个标签，同分以固定种子 20261003 抽选，三类在同一画布中混合排布。
全部 284 个原始短语仍保留在数据表中。以全部短语的 S 最小值和最大值归一化，
w = ((S − min S)/(max S − min S))^0.45，字号为 round(12 + 48w) pt；w 越高颜色越深。
字体和色深采用同一尺度，文字像素碰撞检测后紧密排布；少量纵向词填充空隙，不拆行成段落。
本次仅本地改图，模型调用总数仍为 10。

## English caption

**Fig. 7 | Graph-guided future research directions.** A unified cloud of abbreviated theme
labels from 95 retained prospective directions grounded in the full 2023–2025 Claim Graph.
Hue distinguishes Method, Problem and Combination. Font size and shade encode a descriptive
prominence index: direction frequency multiplied by log₂(1 + distinct cited source papers).
Twenty-four labels per category are displayed, with seeded tie selection. The index is not
validated scientific importance or forecast probability. Full phrases, direction IDs and
source-paper mappings are supplied in the companion table. Forecasts remain prospective hypotheses.

'''
    path.write_text(before+section+'## Files and reproduction'+after, encoding='utf-8')


def package() -> None:
    with zipfile.ZipFile(OUT/'Fig7_forecast_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('final', 'tables'):
            for path in sorted((OUT/folder).glob('*')):
                archive.write(path, path.relative_to(OUT))
        for name in ('methods_and_caption.md','data/directions.json','data/group_profiles.json',
                     'data/coverage.json','data/summary.json','data/unified_wordcloud.json',
                     'data/wordcloud_frequencies.json','data/usage_summary.json',
                     'qa/unified_layout.json','predictions/merge.json'):
            archive.write(OUT/name,name)


if __name__ == '__main__':
    render()
