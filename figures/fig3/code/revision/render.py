"""Forty independent vector components, eight panels and a 420 × 760 mm master."""
from __future__ import annotations

import copy
import math
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .aggregate import COMPONENTS
from .config import METHODS, Config
from .progress import log
from .storage import read, write

NS = 'http://www.w3.org/2000/svg'
INK, LIGHT = '#213047', '#e7ecf1'
COLORS = dict(zip(METHODS, ['#778794', '#9680b5', '#bb8e5c', '#487fae', '#51a196', '#bf5362']))
NAMES = dict(zip(METHODS, ['Direct-A', 'EACL', 'ReviewGrounder', 'GEAR', 'Graph', 'Full']))
HEIGHTS = {'a': 340, 'b': 470, 'c': 510, 'd': 420, 'e': 530, 'f': 530, 'g': 420, 'h': 580}
TITLES = {'a': 'Study design', 'b': 'Historical increment', 'c': 'Report quality and contributions',
          'd': 'Source support and traceability', 'e': 'Human reviewer comparison', 'f': 'Valid scientific information',
          'g': 'Fusion retention and error handling', 'h': 'Preference, repeat evaluation and controls'}
SHORT = {'supported_difference': 'Difference', 'substantially_covered': 'Covered', 'bounded_increment': 'Bounded',
    'insufficient_material': 'Insufficient', 'positive_increment': 'Positive', 'substantially_known': 'Known',
    'limited_increment': 'Limited', 'explicit_abstention': 'Abstain', 'not_addressed': 'Absent',
    'cannot_judge': 'Undecidable', 'order_sensitive': 'Order sensitive', 'technical_failure': 'Technical failure'}


def root(width: float, height: float, ident: str) -> ET.Element:
    ET.register_namespace('', NS)
    return ET.Element(f'{{{NS}}}svg', {'viewBox': f'0 0 {width} {height}', 'width': str(width),
                                    'height': str(height), 'id': ident})


def element(parent: ET.Element, tag: str, **attrs: Any) -> ET.Element:
    return ET.SubElement(parent, f'{{{NS}}}{tag}', {k.replace('_', '-'): str(v) for k, v in attrs.items()})


def text(parent: ET.Element, x: float, y: float, value: str, size: int = 14, bold: bool = False) -> None:
    element(parent, 'text', x=x, y=y, font_size=size, font_family='DejaVu Sans, Microsoft YaHei, sans-serif',
            fill=INK, font_weight='bold' if bold else 'normal').text = re.sub(
                r'[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]', '\ufffd', value)


def label(value: str) -> str:
    return NAMES.get(value, SHORT.get(value, value.replace('quality_', '').replace('error_prevalence_', '').replace('_', ' ')))


def wrap(value: str, width: int) -> list[str]:
    # CJK glyphs occupy roughly twice the width of Latin characters.
    lines, line, used = [], '', 0
    for char in value:
        weight = 2 if ord(char) > 255 else 1
        if char == '\n' or used+weight > width:
            lines.append(line)
            line, used = '', 0
        if char != '\n':
            line += char
            used += weight
    if line:
        lines.append(line)
    return lines


def intervals(node: ET.Element, data: dict[str, Any], width: int, height: int) -> None:
    rows = [r for r in data['rows'] if r.get('estimate') is not None]
    if not rows:
        text(node, 15, 75, 'NA — no estimable observations')
        return
    metrics = list(dict.fromkeys(r.get('metric', r.get('category', 'value')) for r in rows))
    left, right, top, bottom = 190, width-30, 66, height-42
    lower = min([0.0]+[r.get('low', r['estimate']) for r in rows])
    upper = max([0.0]+[r.get('high', r['estimate']) for r in rows])
    if lower == upper:
        upper = lower+1
    pad = (upper-lower)*.07
    lower, upper = lower-pad, upper+pad
    pos = lambda x: left+(x-lower)/(upper-lower)*(right-left)
    for tick in range(5):
        value = lower+(upper-lower)*tick/4
        x = pos(value)
        element(node, 'line', x1=x, x2=x, y1=top-8, y2=bottom+5, stroke=LIGHT)
        text(node, x-12, bottom+23, f'{value:.2g}', 12)
    step = (bottom-top)/max(1, len(metrics))
    for index, metric in enumerate(metrics):
        group = [r for r in rows if r.get('metric', r.get('category', 'value')) == metric]
        y = top+(index+.5)*step
        words = wrap(label(metric), 26)
        for j, line in enumerate(words[:2]):
            text(node, 10, y+(j-.3)*14, line, 13)
        for k, row in enumerate(group):
            offset = (k-(len(group)-1)/2)*min(5, step/max(len(group)+1, 1))
            color = COLORS.get(row.get('method'), '#53647d')
            low, high = row.get('low', row['estimate']), row.get('high', row['estimate'])
            element(node, 'line', x1=pos(low), x2=pos(high), y1=y+offset, y2=y+offset, stroke=color, stroke_width=1.8)
            element(node, 'circle', cx=pos(row['estimate']), cy=y+offset, r=2.8, fill=color)
    text(node, 12, height-8, 'Dots: paper macro means; whiskers: 95% paper bootstrap CI. See CSV for n.', 11)


def scatter(node: ET.Element, data: dict[str, Any], width: int, height: int) -> None:
    rows = [r for r in data['rows'] if r.get('x') is not None and r.get('y') is not None]
    if not rows:
        text(node, 15, 75, 'NA — no paired axes available')
        return
    x_max = max([r.get('x_high') or r['x'] for r in rows]+[.01])*1.12
    y_max = max([r.get('y_high') or r['y'] for r in rows]+[.01])*1.12
    left, right, top, bottom = 65, width-35, 60, height-45
    xp = lambda v: left+v/x_max*(right-left)
    yp = lambda v: bottom-v/y_max*(bottom-top)
    for tick in range(4):
        vx, vy = x_max*tick/3, y_max*tick/3
        element(node, 'line', x1=left, x2=right, y1=yp(vy), y2=yp(vy), stroke=LIGHT)
        text(node, left-45, yp(vy)+4, f'{vy:.2g}', 12)
        text(node, xp(vx)-8, bottom+17, f'{vx:.2g}', 12)
    for row in rows:
        x, y, color = xp(row['x']), yp(row['y']), COLORS.get(row['method'], INK)
        if row.get('x_low') is not None:
            element(node, 'line', x1=xp(row['x_low']), x2=xp(row['x_high']), y1=y, y2=y, stroke=color)
        if row.get('y_low') is not None:
            element(node, 'line', x1=x, x2=x, y1=yp(row['y_low']), y2=yp(row['y_high']), stroke=color)
        element(node, 'circle', cx=x, cy=y, r=4, fill=color)
        text(node, x+6, y-5, label(row['method']), 12)
    text(node, left, height-6, 'x: '+label(data['x_label'])+'   /   y: '+label(data['y_label']), 12)


def heatmap(node: ET.Element, data: dict[str, Any], width: int, height: int) -> None:
    rows = data['rows']
    if not rows:
        text(node, 15, 75, 'NA — no observed categories')
        return
    if 'metric' in rows[0]:
        intervals(node, data, width, height)
        return
    categories = list(dict.fromkeys(r['category'] for r in rows))
    methods = list(dict.fromkeys(r['method'] for r in rows))
    # Matrix datasets retain x/y explicitly; grid cells display each joint category per method.
    left, top, right, bottom = 118, 90, width-12, height-30
    cell_w, cell_h = (right-left)/max(len(categories), 1), (bottom-top)/max(len(methods), 1)
    index = {(r['method'], r['category']): r for r in rows}
    for j, category in enumerate(categories):
        name = label(category)
        for k, line in enumerate(wrap(name, max(5, int(cell_w/6)))[:3]):
            text(node, left+j*cell_w+2, 48+k*12, line, 10)
    for i, method in enumerate(methods):
        for k, line in enumerate(wrap(label(method), 16)[:2]):
            text(node, 8, top+(i+.5)*cell_h+k*12, line, 11)
        for j, category in enumerate(categories):
            row = index.get((method, category))
            value = row.get('estimate') if row else None
            color = '#f3f5f7' if value is None else f'rgb({int(240-155*value)},{int(245-125*value)},{int(249-100*value)})'
            element(node, 'rect', x=left+j*cell_w, y=top+i*cell_h, width=max(0, cell_w-2), height=max(0, cell_h-2), fill=color)
            if cell_w > 22 and cell_h > 12:
                text(node, left+j*cell_w+3, top+(i+.5)*cell_h+3, 'NA' if value is None else f'{value:.0%}', 10)
    text(node, 8, height-8, 'Cell: within-paper share, macro averaged. Exact counts and denominators in component CSV.', 10)


def table(node: ET.Element, data: dict[str, Any], width: int, height: int) -> None:
    y = 52
    for row in data['rows']:
        value = ' | '.join(f'{label(str(k))}: {label(str(v))}' for k, v in row.items())
        for line in wrap(value, int((width-30)/7)):
            if y > height-22:
                text(node, 15, height-8, 'Continued in component JSON / CSV.', 11)
                return
            text(node, 15, y, line, 13)
            y += 17
        y += 7


def case(node: ET.Element, data: dict[str, Any], width: int, height: int) -> None:
    if not data['rows']:
        text(node, 15, 75, 'No eligible case available')
        return
    first, y = data['rows'][0], 52
    text(node, 12, y, first['paper_id']+' · ID-ordered eligible case', 12, True)
    y += 20
    if data['id'] == 'c6':
        rows = [{'heading': label(r['method'])+' / '+str(r.get('P')), 'quote': ' '.join(r.get('quotes', []))}
                for r in data['rows']]
    else:
        rows = ([{'heading': 'Report assertion', 'quote': first['report_quote']}]+
                [{'heading': e['source_id']+' · '+e['source_type'], 'quote': e['quote']+' — '+e['reason']}
                 for e in first['evidence']])
    for row in rows:
        if y > height-50:
            break
        text(node, 12, y, row['heading'], 12, True)
        y += 16
        for line in wrap(row['quote'], int((width-30)/7))[:3]:
            if y > height-24:
                break
            text(node, 12, y, line, 12)
            y += 16
        y += 7
    text(node, 12, height-7, 'Figure excerpts; complete quotes and evidence are retained in component JSON / CSV.', 10)


def component(data: dict[str, Any], width: int, height: int) -> ET.Element:
    node = root(width, height, data['id'])
    element(node, 'rect', x=0, y=0, width=width, height=height, fill='white')
    text(node, 12, 25, data['id']+'  '+data['title'], 17, True)
    renderers = {'interval': intervals, 'flow': flow, 'scatter': scatter, 'category': heatmap,
                 'matrix': matrix, 'table': table, 'case': case}
    renderers[data['kind']](node, data, width, height)
    return node


def flow(node: ET.Element, data: dict[str, Any], width: int, height: int) -> None:
    values = {r['metric']: r['estimate'] for r in data['rows'] if r['estimate'] is not None}
    if not values:
        text(node, 15, 75, 'NA — no fusion observations')
        return
    retained = values.get('fusion_retained', 0)
    lost, added = values.get('fusion_not_retained', 0), values.get('fusion_added', 0)
    left, right, middle = 120, width-140, height*.52
    maximum = max(retained+lost, retained+added, 1)
    for y1, y2, value, color in [(middle, middle, retained, '#51a196'),
            (middle, height-48, lost, '#bf5362'), (65, middle, added, '#487fae')]:
        if value > 0:
            element(node, 'path', d=f'M {left} {y1} C {width*.45} {y1}, {width*.55} {y2}, {right} {y2}',
                    fill='none', stroke=color, stroke_width=max(1, value/maximum*35), opacity=.65)
    text(node, 12, middle-20, f'Branch union: {retained+lost:.2f}', 13)
    text(node, right-15, middle-20, f'Full: {retained+added:.2f}', 13)
    text(node, width*.38, middle-23, f'Retained {retained:.2f}', 13)
    text(node, left, 53, f'Added {added:.2f}', 13)
    text(node, right-40, height-25, f'Not retained {lost:.2f}', 13)
    text(node, 12, height-7, 'Mean within-paper cluster counts; no inference of explicit error correction.', 11)


def matrix(node: ET.Element, data: dict[str, Any], width: int, height: int) -> None:
    rows = data['rows']
    if not rows:
        text(node, 15, 75, 'NA — no observed matrix entries')
        return
    methods = list(dict.fromkeys(r['method'] for r in rows))
    xs, ys = sorted({r['x'] for r in rows}), sorted({r['y'] for r in rows})
    columns = min(3, len(methods))
    facet_w, facet_h = width/columns, (height-52)/math.ceil(len(methods)/columns)
    index = {(r['method'], r['x'], r['y']): r for r in rows}
    for i, method in enumerate(methods):
        ox, oy = i % columns*facet_w, 40+i//columns*facet_h
        text(node, ox+7, oy+12, label(method), 12, True)
        left, top = ox+75, oy+22
        cell_w, cell_h = (facet_w-83)/max(len(xs), 1), (facet_h-40)/max(len(ys), 1)
        for xj, xname in enumerate(xs):
            text(node, left+xj*cell_w+2, top+len(ys)*cell_h+12, label(xname)[:6], 9)
        for yj, yname in enumerate(ys):
            text(node, ox+4, top+(yj+.6)*cell_h, label(yname)[:10], 9)
            for xj, xname in enumerate(xs):
                row = index.get((method, xname, yname))
                value = row.get('estimate') if row else 0.0
                color = '#f0f3f6' if value is None else f'rgb({int(240-155*value)},{int(245-125*value)},{int(249-100*value)})'
                element(node, 'rect', x=left+xj*cell_w, y=top+yj*cell_h,
                        width=max(0, cell_w-1), height=max(0, cell_h-1), fill=color)
                if cell_w > 23 and cell_h > 11:
                    text(node, left+xj*cell_w+2, top+(yj+.75)*cell_h, 'NA' if value is None else f'{value:.0%}', 9)
    text(node, 12, height-3, 'Joint shares within paper; full labels, counts and denominators in component CSV.', 10)


def outline(node: ET.Element) -> ET.Element:
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MplPath
    from matplotlib.textpath import TextPath
    result = copy.deepcopy(node)
    cjk = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    for parent in result.iter():
        for child in list(parent):
            if child.tag != f'{{{NS}}}text':
                continue
            value = child.text or ''
            font = FontProperties(fname=str(cjk)) if cjk.exists() and any(ord(c) > 255 for c in value) else FontProperties(
                family='DejaVu Sans', weight=child.get('font-weight', 'normal'))
            path = TextPath((0, 0), value, size=float(child.get('font-size', '14')), prop=font)
            commands = []
            names = {MplPath.MOVETO: 'M', MplPath.LINETO: 'L', MplPath.CURVE3: 'Q', MplPath.CURVE4: 'C', MplPath.CLOSEPOLY: 'Z'}
            for vertices, code in path.iter_segments(curves=True, simplify=False):
                commands.append(names[code]+(' '.join(f'{v:.4f}' for v in vertices) if code != MplPath.CLOSEPOLY else ''))
            element(parent, 'path', d=' '.join(commands), fill=child.get('fill', INK),
                transform=f'translate({child.get("x", "0")} {child.get("y", "0")}) scale(1 -1)')
            parent.remove(child)
    return result


def export(node: ET.Element, path: Path, config: Config) -> None:
    import cairosvg
    width, height = map(float, node.get('viewBox', '').split()[2:])
    node.set('width', f'{config.figure_width_mm*width/2100}mm')
    node.set('height', f'{config.figure_width_mm*height/2100}mm')
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(node).write(path, encoding='utf-8', xml_declaration=True)
    outlined = outline(node)
    outlined_path = path.with_name(path.stem+'_outlined.svg')
    ET.ElementTree(outlined).write(outlined_path, encoding='utf-8', xml_declaration=True)
    # Raster/PDF from outlined paths retain the same CJK glyph shapes across exporters.
    cairosvg.svg2pdf(url=str(outlined_path), write_to=str(path.with_suffix('.pdf')))
    cairosvg.svg2png(url=str(outlined_path), write_to=str(path.with_suffix('.png')),
                    output_width=round(width*2), output_height=round(height*2))


def render_component(config: Config, key: str, width: int, height: int) -> str:
    data = read(config.output/'derived/components'/f'{key}.json')
    child = component(data, width, height)
    export(copy.deepcopy(child), config.output/'figures/components'/f'{key}.svg', config)
    return ET.tostring(child, encoding='unicode')


def render(config: Config, panel: str | None = None, component: str | None = None) -> None:
    selected = [component[0]] if component else [panel] if panel else list(HEIGHTS)
    layout = []
    for name in selected:
        keys = [k for k in COMPONENTS if k[0] == name]
        columns = 2 if len(keys) == 4 else 3
        height = (HEIGHTS[name]-65)//math.ceil(len(keys)/columns)
        for index, key in enumerate(keys):
            if not component or key == component:
                layout.append({'component': key, 'panel': name, 'x': index % columns*(2100//columns),
                               'y': 55+index//columns*height, 'width': 2100//columns, 'height': height})
    rendered = {}
    # Matplotlib font/path state is isolated per CPU process; only the parent logs.
    with ProcessPoolExecutor(max_workers=min(len(layout) or 1, os.cpu_count() or 1, 8)) as pool:
        futures = {pool.submit(render_component, config, row['component'], row['width'], row['height']): row for row in layout}
        for future in as_completed(futures):
            key = futures[future]['component']
            rendered[key] = future.result()
            log(config, '组件完成', f'[绘图 {len(rendered)}/{len(layout)}] {key}四种格式已保存。', stage='render')
    if not component:
        for name in selected:
            node = root(2100, HEIGHTS[name], 'panel_'+name)
            text(node, 12, 30, name+'  '+TITLES[name], 24, True)
            for i, method in enumerate(METHODS):
                element(node, 'circle', cx=790+i*210, cy=22, r=5, fill=COLORS[method])
                text(node, 802+i*210, 27, NAMES[method], 14)
            for row in (r for r in layout if r['panel'] == name):
                child = ET.fromstring(rendered[row['component']])
                child.set('x', str(row['x'])); child.set('y', str(row['y']))
                node.append(child)
            export(node, config.output/'figures/panels'/f'{name}.svg', config)
    write(config.output/'figures/placements.json', layout)


def assemble(config: Config) -> None:
    log(config, '组装开始', '读取8个panel，导出420×760 mm母版及其他格式。', stage='assemble')
    height = 2100*config.figure_height_mm/config.figure_width_mm
    node = root(2100, height, 'Fig3')
    element(node, 'rect', x=0, y=0, width=2100, height=height, fill='white')
    y = 0
    for name, panel_height in HEIGHTS.items():
        child = ET.parse(config.output/'figures/panels'/f'{name}.svg').getroot()
        child.set('x', '0')
        child.set('y', str(y))
        child.set('width', '2100')
        child.set('height', str(panel_height))
        node.append(child)
        y += panel_height
    export(node, config.output/'figures/Fig3.svg', config)
    log(config, '组装完成', '母版四种格式已保存。', stage='assemble')
