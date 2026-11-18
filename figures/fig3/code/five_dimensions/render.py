"""Update the five-dimensional score panels and authorised workflow labels in Fig3.

No scientific values are invented and no original image or result is overwritten.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from .config import DIMENSIONS, METHOD_LABELS, OUTPUT, ROOT
from .io import read, write

SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
NS = {'s': SVG}
SOURCE_SVG = ROOT / 'figures/fig3/assets/layout_template.svg'
LAYOUT = ROOT / 'outputs/fig3_reference/layouts/style.json'
METHOD_ORDER = ('direct_a', 'gear', 'graph', 'full', 'eacl', 'reviewgrounder')
OTHERS = tuple(m for m in METHOD_ORDER if m != 'full')
LABELS = dict(METHOD_LABELS, reviewgrounder='ReviewG')
COLORS = dict(zip(METHOD_ORDER, ('#84909E', '#E16C35', '#1269DF', '#713BCB', '#219D91', '#92944F')))
ABBREVIATIONS = dict(zip(DIMENSIONS, ('HO', 'UC', 'KO', 'EB', 'SV')))
REPLACED_IDS = tuple(f'axes_{n}' for n in range(13, 20)) + ('text_692', 'text_693')
FLOW_TEXT_IDS = ('text_8',) + tuple(f'text_{n}' for n in range(69, 82))
PANEL_TITLE_IDS = ('text_694', 'text_695', 'replacement_c6_text_10')
PT_PER_MM = 72 / 25.4


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float | None:
    if value is None or value == '' or str(value).casefold() == 'nan':
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _data(output: Path, allow_incomplete: bool) -> tuple[list[dict], list[dict], dict]:
    scores = _csv(output / 'tables/dimension_scores.csv')
    summaries = _csv(output / 'tables/summary.csv')
    pairs = _csv(output / 'tables/paired_effects.csv')
    summary = read(output / 'summary.json')
    completion = summary['completion']
    if summary.get('full_rerun'):
        completion = {**completion, 'full_rerun': True}
    expected = completion['main_expected_evaluations']
    completed = completion['main_completed_evaluations']
    partial = completion.get('main_partial_evaluations', 0)
    failed = completion.get('main_technical_failure_evaluations', 0)
    if completion['main_papers'] != 100 or expected != 600:
        raise ValueError('The formal replacement requires the original 100-paper, six-method roster.')
    if completed + partial + failed < expected and not allow_incomplete:
        raise ValueError(f'Only {completed} complete + {partial} partial + {failed} technical failures '
                         f'of {expected} evaluations recorded; no formal Fig3 rendered. '
                         'Use allow_incomplete=True only to explicitly retain documented missing results.')
    scores = [r for r in scores if r['cohort'] == 'main']
    pairs = [r for r in pairs if r['cohort'] == 'main']
    index = {(r['method'], r['dimension']): r for r in summaries if r['cohort'] == 'main'}
    for method in METHOD_ORDER:
        for dimension in DIMENSIONS:
            found = [r for r in scores if r['method'] == method and r['dimension'] == dimension]
            assert len(found) == len({r['paper_id'] for r in found}) == 100
            values = [int(r['score']) for r in found if r['status'] == 'assessed']
            assert all(0 <= v <= 3 for v in values)
            assert len(values) == int(index[(method, dimension)]['n'])
    pair_keys = {(r['comparator'], r['dimension']) for r in pairs}
    assert len(pairs) == len(pair_keys) == len(OTHERS) * len(DIMENSIONS)
    if not any(int(r['n']) for r in pairs):
        raise ValueError('No assessed paired results; cannot render invented or placeholder effects.')
    return scores, pairs, completion


def _style() -> Any:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for name in ('arial.ttf', 'arialbd.ttf'):
        path = Path('/mnt/c/Windows/Fonts') / name
        if path.is_file():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                         'text.color': '#10213B', 'axes.labelcolor': '#647487',
                         'xtick.color': '#647487', 'ytick.color': '#647487',
                         'axes.edgecolor': '#8295A7', 'axes.linewidth': .6})
    return plt


def _save(fig: Any, prefix: Path, plt: Any) -> None:
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(prefix.with_suffix('.' + extension), transparent=True, dpi=300)
    plt.close(fig)


def _score_panel(scores: list[dict], output: Path, box: list[float]) -> None:
    plt = _style()
    width, height = box[2:]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor='none')
    fig.text(0, .975, 'c1  Five-dimensional analysis performance', fontsize=11, weight='bold', va='top')
    for j, dimension in enumerate(DIMENSIONS):
        ax = fig.add_axes([.065 + j * .187, .24, .18, .54])
        ax.set_facecolor('none')
        ax.axhspan(2.55, 3.45, color=COLORS['full'], alpha=.06, zorder=0)
        for i, method in enumerate(METHOD_ORDER):
            values = [int(r['score']) for r in scores if r['dimension'] == dimension
                      and r['method'] == method and r['status'] == 'assessed']
            for score, count in Counter(values).items():
                ax.scatter(score, i, s=110 * count / len(values), color=COLORS[method],
                           edgecolors='white', linewidths=.3, zorder=3)
            ax.text(3.9, i, str(len(values)), fontsize=6.4, color='#647487', va='center', ha='center')
        ax.set_xlim(-.5, 4.5)
        ax.set_ylim(5.55, -.55)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks(range(6), [LABELS[m] for m in METHOD_ORDER] if j == 0 else [])
        ax.tick_params(length=0, labelsize=7.4, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for tick, method in zip(ax.get_yticklabels(), METHOD_ORDER):
            tick.set_color(COLORS[method])
        ax.set_title(ABBREVIATIONS[dimension], fontsize=9, pad=4)
        ax.text(3.9, -1.13, 'n', fontsize=6.5, color='#647487', ha='center')
    fig.text(.01, .105, 'HO Originality  ·  UC Understanding/capability  ·  KO Knowledge organisation  ·  EB Evidence  ·  SV Value',
             fontsize=6.8, color='#647487')
    fig.text(.01, .025, 'Final five-dimensional report only; score 0–3 measures analysis performance. Bubble area = within-method proportion.',
             fontsize=6.6, color='#647487')
    _save(fig, output / 'Fig3_c1_five_dimensions', plt)


def _paired_panel(pairs: list[dict], output: Path, box: list[float]) -> None:
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    plt = _style()
    width, height = box[2:]
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), facecolor='none')
    fig.text(0, .975, 'c2  Paired analysis differences · Full − comparator', fontsize=9.8, weight='bold', va='top')
    ax = fig.add_axes([.06, .05, .92, .68])
    ax.set_facecolor('none')
    lookup = {(r['comparator'], r['dimension']): r for r in pairs}
    effects = np.full((5, 5), np.nan)
    magnitudes = []
    for i, dimension in enumerate(DIMENSIONS):
        for j, method in enumerate(OTHERS):
            row = lookup[(method, dimension)]
            value = _number(row['estimate'])
            if value is not None:
                effects[i, j] = value
                magnitudes.extend(abs(float(row[k])) for k in ('estimate', 'low', 'high'))
    limit = max(.1, *magnitudes)
    cmap = LinearSegmentedColormap.from_list('five_dim_effect', ['#EAA678', '#FFFFFF', '#9671D0'])
    cmap.set_bad('#ECEFF2')
    ax.imshow(effects, cmap=cmap, vmin=-limit, vmax=limit, aspect='auto')
    for i, dimension in enumerate(DIMENSIONS):
        for j, method in enumerate(OTHERS):
            row = lookup[(method, dimension)]
            value = _number(row['estimate'])
            if value is None:
                ax.text(j, i, 'NA', fontsize=7, ha='center', va='center', color='#647487')
                continue
            ax.text(j, i - .12, f'{value:+.2f}', ha='center', va='center', fontsize=7.6)
            low, high = float(row['low']), float(row['high'])
            ax.plot([j + low / limit * .34, j + high / limit * .34], [i + .25] * 2,
                    color='#10213B', linewidth=.7)
            ax.plot([j, j], [i + .17, i + .33], color='#8A98AB', linewidth=.5)
    for method in ('gear', 'graph'):
        j = OTHERS.index(method)
        ax.add_patch(Rectangle((j - .49, -.49), .98, 4.98, fill=False, lw=.8, ec=COLORS[method]))
    ax.set_xticks(range(5), [LABELS[m] for m in OTHERS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(5), list(ABBREVIATIONS.values()))
    ax.tick_params(length=0, labelsize=7, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, output / 'Fig3_c2_paired_differences', plt)


def _element_hash(node: ET.Element) -> str:
    return hashlib.sha256(ET.tostring(node)).hexdigest()


def _unchanged_hash(node: ET.Element) -> str:
    """Exclude only explicitly authorised label nodes, retaining their surrounding art."""
    copied = copy.deepcopy(node)
    if copied.get('id') in PANEL_TITLE_IDS:
        for child in list(copied):
            copied.remove(child)
    for parent in copied.iter():
        for child in list(parent):
            if child.get('id') in FLOW_TEXT_IDS + PANEL_TITLE_IDS:
                parent.remove(child)
    return _element_hash(copied)


def _panel_numbers(root: ET.Element) -> list[dict[str, str]]:
    """Renumber only the three visible retained panel titles; preserve their art."""
    index = {node.get('id'): node for node in root.iter()}
    audit = []
    for identifier, old, new in [('text_694', '4', '3'), ('text_695', '5', '4')]:
        uses = index[identifier].findall('.//s:use', NS)
        attribute = f'{{{XLINK}}}href'
        assert uses[0].get(attribute) == '#TimesNewRomanPS-BoldMT-63'
        assert uses[1].get(attribute) == f'#TimesNewRomanPS-BoldMT-{ord(old):x}'
        uses[1].set(attribute, f'#TimesNewRomanPS-BoldMT-{ord(new):x}')
        audit.append({'node': identifier, 'old_prefix': 'c' + old, 'new_prefix': 'c' + new})
    title = index['replacement_c6_text_10'].find('s:text', NS)
    assert title is not None and title.text and title.text.startswith('c6  ')
    title.text = 'c5' + title.text[2:]
    audit.append({'node': 'replacement_c6_text_10', 'old_prefix': 'c6', 'new_prefix': 'c5'})
    return audit


def _flow_labels(root: ET.Element, completion: dict[str, Any], rescued: ET.Element) -> dict[str, Any]:
    """Keep the historical 594 count; identify new scoring and unchanged diagnostics."""
    completed, expected = completion['main_completed_evaluations'], completion['main_expected_evaluations']
    partial = completion.get('main_partial_evaluations', 0)
    entries = {
        'text_8': (201.7, 189, 8, ['old quality', 'assessments']),
        'text_69': (719, 177, 9, ['Common checklist']),
        'text_70': (719, 189, 8.7, ['Manuscript', '+ original sources']),
        'text_71': (858, 177, 9, ['Five-dimension body']),
        'text_72': (858, 189, 8.7, ['New final analysis', 'for each method']),
        'text_73': (788, 229, 9.2, ['Independent verification']),
        'text_74': (788, 242, 8.5, ['Body + checklist + source material']),
        'text_75': (788, 261, 7.8,
                    [f'New: {completed} complete + {partial} partial / {expected}; n by dimension',
                     'Other diagnostics: original detailed analysis']),
        'text_76': (1044.5, 177, 10, ['Report construction']),
        'text_77': (1044.5, 190, 8.7, ['Original analysis + own evidence']),
        'text_78': (1044.5, 210, 10, ['Shared checklist']),
        'text_79': (1044.5, 223, 8.7, ['Manuscript + common evidence']),
        'text_80': (1044.5, 243, 10, ['Independent verification']),
        'text_81': (1044.5, 256, 8.4, ['New body + checklist + cited evidence']),
    }
    if completion.get('full_rerun'):
        entries['text_72'] = (858, 189, 8.7, ['Full updated', 'Other methods retained'])
    index = {node.get('id'): node for node in root.iter()}
    for identifier, (x, y, size, lines) in entries.items():
        group = index[identifier]
        for definitions in group.findall('.//s:defs', NS):
            for definition in definitions:
                rescued.append(copy.deepcopy(definition))
        for child in list(group):
            group.remove(child)
        for i, line in enumerate(lines):
            text = ET.SubElement(group, f'{{{SVG}}}text', {
                'x': str(x), 'y': str(y + i * 10.2), 'text-anchor': 'middle',
                'font-family': 'Arial', 'font-size': str(size), 'fill': '#10213B',
            })
            text.text = line
    return {'updated_text_nodes': list(entries), 'original_quality_count_retained': 594,
            'new_completed_evaluations': completed, 'new_expected_evaluations': expected,
            'new_partial_evaluations': partial,
            'new_denominator_source': 'summary.json completion',
            'meaning': 'New five-dimensional body verification; remaining diagnostics use original analysis.'}


def _prefix_ids(node: ET.Element, prefix: str) -> None:
    mapping = {el.attrib['id']: prefix + el.attrib['id'] for el in node.iter() if 'id' in el.attrib}
    for element in node.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id':
                element.set(key, mapping[value])
            elif value.startswith('#') and value[1:] in mapping:
                element.set(key, '#' + mapping[value[1:]])
            else:
                element.set(key, re.sub(r'url\(#([^)]*)\)',
                                        lambda match: 'url(#' + mapping.get(match[1], match[1]) + ')', value))


def _check_original_positions(figure: ET.Element, layout: dict) -> None:
    """Refuse to replace these IDs if the authoritative SVG has changed layout."""
    by_id = {node.get('id'): node for node in figure}
    for identifier in REPLACED_IDS:
        if identifier not in by_id:
            raise ValueError(f'Expected source SVG element is absent: {identifier}')
    for identifier, component in [('axes_13', 'c1'), ('axes_17', 'c1'), ('axes_19', 'c3')]:
        path = by_id[identifier].find('s:g/s:path', NS)
        assert path is not None
        values = [float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+)', path.get('d', ''))]
        xs, ys = values[::2], values[1::2]
        x, y, w, h = [v * PT_PER_MM for v in layout['components'][component]['page_box_mm']]
        assert x <= min(xs) <= max(xs) <= x + w and y <= min(ys) <= max(ys) <= y + h


def _replace_panels(source: Path, score_svg: Path, pair_svg: Path, target: Path,
                    completion: dict[str, Any] | None = None) -> dict[str, Any]:
    ET.register_namespace('', SVG)
    ET.register_namespace('xlink', XLINK)
    root = ET.parse(source).getroot()
    figure = root.find("s:g[@id='figure_1']", NS)
    if figure is None:
        raise ValueError('Source SVG lacks the expected matplotlib figure_1 group.')
    layout = read(LAYOUT)
    _check_original_positions(figure, layout)
    retained_before = {node.get('id'): _unchanged_hash(node) for node in figure
                       if node.get('id') not in REPLACED_IDS}
    # Later axes can reference font glyphs or markers defined inside removed axes.
    # Preserve those definitions without preserving the old scores or text.
    rescued = ET.Element(f'{{{SVG}}}defs', {'id': 'five_dimensions_preserved_source_definitions'})
    for child in list(figure):
        if child.get('id') in REPLACED_IDS:
            for definitions in child.findall('.//s:defs', NS):
                for definition in definitions:
                    rescued.append(copy.deepcopy(definition))
            figure.remove(child)
    root.insert(1, rescued)
    flow_audit = _flow_labels(root, completion, rescued) if completion is not None else None
    panel_audit = _panel_numbers(root)
    for component, filename in [('c1', score_svg), ('c3', pair_svg)]:
        panel = ET.parse(filename).getroot()
        _prefix_ids(panel, 'five_dim_' + component + '_')
        x, y, w, h = [v * PT_PER_MM for v in layout['components'][component]['page_box_mm']]
        vx, vy, vw, vh = [float(v) for v in panel.get('viewBox', '').split()]
        if vx != 0 or vy != 0:
            raise ValueError('Replacement SVG requires a zero-origin viewBox.')
        group = ET.SubElement(figure, f'{{{SVG}}}g', {
            'id': 'five_dimensions_replacement_' + component,
            'transform': f'translate({x:.9f} {y:.9f}) scale({w / vw:.9f} {h / vh:.9f})',
        })
        for child in panel:
            if child.tag != f'{{{SVG}}}metadata':
                group.append(child)
    retained_after = {node.get('id'): _unchanged_hash(node) for node in figure
                      if node.get('id') in retained_before}
    assert retained_before == retained_after
    ids = [element.get('id') for element in root.iter() if element.get('id')]
    duplicates = [identifier for identifier, n in Counter(ids).items() if n > 1]
    assert not duplicates, duplicates
    from .reflow import reflow_panel_c
    reflow_audit = reflow_panel_c(root)
    final_ids = [n.get('id') for n in root.iter() if n.get('id')]
    assert len(final_ids) == len(set(final_ids))
    ET.ElementTree(root).write(target, encoding='utf-8', xml_declaration=True)
    return {'panel_c_reflow': reflow_audit, 'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'removed_source_ids': list(REPLACED_IDS), 'preserved_top_level_nodes': len(retained_before),
            'preserved_nodes_unchanged_except_authorised_labels_and_panel_c_reflow': True,
            'authorised_label_updates': flow_audit, 'rescued_definition_count': len(rescued),
            'authorised_panel_title_updates': panel_audit,
            'unique_svg_ids': True,
            'replaced_boxes_mm': {key: layout['components'][key]['page_box_mm'] for key in ('c1', 'c3')}}


def _export_composite(directory: Path) -> None:
    """Use the already installed renderer even when its environment is not active."""
    candidate = shutil.which('cairosvg')
    executable = Path(candidate) if candidate else Path('/home/jayee/miniconda3/envs/openrlhf/bin/cairosvg')
    if not executable.is_file():
        raise FileNotFoundError('The existing CairoSVG renderer could not be located.')
    environment = os.environ.copy()
    fonts = ROOT / 'outputs/fig1_reference/layouts/fonts.conf'
    if fonts.is_file():
        environment['FONTCONFIG_FILE'] = str(fonts)
    for extension in ('pdf', 'png'):
        command = [str(executable), str(directory / 'Fig3.svg'), '-f', extension,
                   '-o', str(directory / ('Fig3.' + extension))]
        if extension == 'png':
            command.extend(['--output-width', '3000'])
        subprocess.run(command, check=True, env=environment, capture_output=True, text=True)


def render(output: Path = OUTPUT, allow_incomplete: bool = False) -> dict[str, Any]:
    """Render actual scores and update authorised workflow labels in the accepted Fig3."""
    scores, pairs, completion = _data(output, allow_incomplete)
    directory = output / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    layout = read(LAYOUT)
    original_hash = hashlib.sha256(SOURCE_SVG.read_bytes()).hexdigest()
    _score_panel(scores, directory, layout['components']['c1']['page_box_mm'])
    _paired_panel(pairs, directory, layout['components']['c3']['page_box_mm'])
    audit = _replace_panels(SOURCE_SVG, directory / 'Fig3_c1_five_dimensions.svg',
                            directory / 'Fig3_c2_paired_differences.svg', directory / 'Fig3.svg', completion)
    _export_composite(directory)
    assert hashlib.sha256(SOURCE_SVG.read_bytes()).hexdigest() == original_hash
    audit.update(status='complete', completion=completion, allow_incomplete=allow_incomplete,
                 original_file_unchanged=True,
                 statistics_sha256={name: hashlib.sha256((output / 'tables' / name).read_bytes()).hexdigest()
                                    for name in ('dimension_scores.csv', 'summary.csv', 'paired_effects.csv')})
    write(directory / 'fig3_replacement_audit.json', audit)
    (directory / 'fig3_update_notes_zh.md').write_text(
        '# Fig3五维面板更新说明\n\n'
        f"主队列100篇、六种方法；{completion['main_completed_evaluations']}份完整五维评价，"
        f"{completion.get('main_partial_evaluations', 0)}份部分评价，预期共600份。"
        '每个维度按其实际可评分论文计算分布；配对比较仅使用双方均可评分的同一论文。\n\n'
        '参考引文技术失败单独记为technical_failure，不改为科学未决或零分；其余维度保留。'
        '逐维缺失计数见tables/summary.csv和tables/technical_failures.csv。\n\n'
        'panel c重新排版：c1放大、移除底部两行小字，增大五维名称与比例，跨上下两行；'
        '中间c2为上方五维评分分布，下方c3配对差异与c4贡献覆盖呈品字布局；右侧c5断言组成按三行两列排列并放大圆圈。'
        '删除原c5图谱关系与稿件证据案例框，其余面板及所有统计数据不变。'
        'c1沿用1000篇名册分母，994篇可分析、6篇材料不可用；'
        'c3删除底部两行小字并增加矩阵高度；c3和c4放大，c2/c3左移以收紧与c1的间距。配对置信区间仍采用共同尺度，中心竖线表示差值0。'
        'HO历史原创性；UC认识与能力推进；KO知识组织与整合；EB证据基础；SV科学价值。'
        '0–3分衡量五维报告的分析表现，不是论文创新程度，不计算总分。\n\n'
        '新c2/c3评价最终报告前部的五维分析；断言、贡献覆盖、信息与偏好面板仍评价保留的详细分析正文。'
        '594明确标为原质量评价可用集，新五维有效完成数从统计表读取。'
        '旧评分不改名复用，原整图文件及授权区域外的图形节点保持不变。图注和正文需同步明确这两个评价对象。\n',
        encoding='utf-8')
    if completion.get('full_rerun'):
        with (directory / 'fig3_update_notes_zh.md').open('a', encoding='utf-8') as handle:
            handle.write('\n本轮仅更新Full报告与核验，其他五种方法的报告和评分复用；'
                         'Full使用完整稿件及已有证据并集，正文不限篇幅。核验指令、共用清单和评分规则不变。'
                         '生成拒绝、调用超时或其他执行缺失见tables/missing.csv，不计为0分；'
                         '论文草稿及其中图件副本未更新。\n')
    return audit


if __name__ == '__main__':
    print(json.dumps(render(), ensure_ascii=False))
