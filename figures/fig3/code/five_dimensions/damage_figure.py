"""Update the DAMAGE report panel and its manuscript figure-number heading.

The English editorial map must be prepared from an existing generated report.
This module never invents a missing report or translates it with an extra model call.
"""
from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import ImageFont

from .config import DAMAGE_ID, DIMENSIONS, OUTPUT, ROOT
from .io import encoded, read, write
from .models import Report

SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
ORIGINAL = ROOT / 'outputs/fig1-11_final/Fig6/Fig6.svg'
FONT = Path('/mnt/c/Windows/Fonts/times.ttf')
CAIROSVG = Path('/home/jayee/miniconda3/envs/openrlhf/bin/cairosvg')
ORIGINAL_TITLE = 'Fig. 6 | From an RNA-responsive system to a traceable report'
MANUSCRIPT_TITLE = 'Extended Data Fig. 1 | From an RNA-responsive system to a traceable report'
LABELS = {'historical_originality': 'Historical originality',
          'understanding_capability': 'Advances in understanding and capability',
          'knowledge_organisation': 'Knowledge organisation and integration',
          'evidential_basis': 'Evidential basis', 'scientific_value': 'Scientific value'}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def panel(root: ET.Element, identifier: str) -> ET.Element:
    found = root.find(f'.//*[@id="{identifier}"]')
    if found is None:
        raise ValueError(f'Authoritative figure lacks {identifier}')
    return found


def title_node(root: ET.Element) -> ET.Element:
    parent = panel(root, 'fig6_damage')
    nodes = [node for node in parent if node.tag == f'{{{SVG}}}text'
             and node.get('x') == '14' and node.get('y') == '29']
    if len(nodes) != 1 or nodes[0].text not in (ORIGINAL_TITLE, MANUSCRIPT_TITLE):
        raise ValueError('Expected the unique original or renumbered DAMAGE heading')
    return nodes[0]


def outside_report(root: ET.Element, restore_title: bool = False) -> bytes:
    copied = copy.deepcopy(root)
    if restore_title:
        title_node(copied).text = ORIGINAL_TITLE
    for parent in copied.iter():
        for child in list(parent):
            if child.attrib.get('id') == 'panel_e':
                parent.remove(child)
    return ET.tostring(copied, encoding='utf-8')


def load_editorial(report: dict[str, Any], path: Path) -> dict[str, str]:
    """Require source-bound, report-specific editorial summaries for all five rows."""
    record = read(path)
    if record.get('report_sha256') != digest(encoded(report).encode()):
        raise ValueError('English editorial map does not identify the current generated report')
    rows = record.get('dimensions', [])
    if len(rows) != 5 or {row['dimension'] for row in rows} != set(DIMENSIONS):
        raise ValueError('Editorial map must cover exactly the five report dimensions')
    dimensions = {row['dimension']: row for row in report['dimensions']}
    summaries = {}
    for row in rows:
        source = dimensions[row['dimension']]
        prose = [source[key] for key in ('judgment', 'rationale', 'scope')]
        quotations = row.get('supporting_report_quotes', [])
        if not quotations or any(not any(quote in text for text in prose) for quote in quotations):
            raise ValueError(f"{row['dimension']}: editorial source quotation absent from report prose")
        summary = row['summary_en'].strip()
        if not summary:
            raise ValueError('Empty English summary')
        summaries[row['dimension']] = summary
    return summaries


def add_text(group: ET.Element, x: float, y: float, value: str,
             size: float = 16, bold: bool = False, colour: str = '#24282d') -> None:
    element = ET.SubElement(group, f'{{{SVG}}}text', {
        'x': str(x), 'y': str(y), 'font-family': 'Times New Roman',
        'font-size': str(size), 'font-weight': 'bold' if bold else 'normal',
        'font-style': 'normal', 'fill': colour, 'text-anchor': 'start'})
    element.text = value


def wrap(value: str, width: float, size: float) -> list[str]:
    font = ImageFont.truetype(str(FONT), round(size * 4))
    lines: list[str] = []
    current = ''
    for word in value.split():
        candidate = f'{current} {word}'.strip()
        if current and font.getlength(candidate) / 4 > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if any(font.getlength(line) / 4 > width for line in lines):
        raise ValueError('Unbreakable editorial text exceeds the figure panel width')
    return lines


def replace_panel(root: ET.Element, summaries: dict[str, str]) -> None:
    target = panel(root, 'panel_e')
    children = list(target)
    for child in children[3:]:
        target.remove(child)
    add_text(target, 36, 23, 'Five-dimensional innovation analysis', 21, True)
    add_text(target, 14, 55, 'Editorial summaries of the generated five-dimensional report', 15,
             colour='#4c78a8')
    for index, dimension in enumerate(DIMENSIONS):
        top = 66 + index * 78
        ET.SubElement(target, f'{{{SVG}}}rect', {
            'x': '13', 'y': str(top), 'width': '591', 'height': '73',
            'fill': '#f5f8fa', 'stroke': '#e0e8f0', 'rx': '3', 'stroke-width': '0.65'})
        add_text(target, 23, top + 18, f'D{index+1}  {LABELS[dimension]}', 16.5, True)
        size = 16.0
        lines = wrap(summaries[dimension], 568, size)
        if len(lines) > 3:
            size = 14.5
            lines = wrap(summaries[dimension], 568, size)
        if len(lines) > 3:
            raise ValueError(f'{dimension}: shorten the source-bound English editorial summary')
        for offset, text in enumerate(lines):
            add_text(target, 23, top + 36 + offset * 16, text, size)
    add_text(target, 14, 469, 'Source quotations and judgment scope are retained in the full report.',
             13.5, colour='#4c78a8')


def export(svg: Path) -> None:
    if not CAIROSVG.is_file():
        raise FileNotFoundError(f'Existing CairoSVG entrypoint unavailable: {CAIROSVG}')
    environment = dict(os.environ)
    fonts = ROOT / 'outputs/fig1_reference/layouts/fonts.conf'
    if fonts.is_file():
        environment['FONTCONFIG_FILE'] = str(fonts)
    subprocess.run([str(CAIROSVG), str(svg), '-f', 'pdf', '-o', str(svg.with_suffix('.pdf'))],
                   check=True, env=environment)
    subprocess.run([str(CAIROSVG), str(svg), '-f', 'png', '--output-width', '4252',
                    '-o', str(svg.with_suffix('.png'))], check=True, env=environment)


def verification_details(evaluation: dict[str, Any]) -> dict[str, Any]:
    rows = evaluation['dimensions']
    if not any(row['status'] == 'assessed' for row in rows):
        raise ValueError('DAMAGE requires at least one independently assessed dimension for display')
    failures = [row['dimension'] for row in rows if row['status'] == 'technical_failure']
    return {'verification_status': 'partial' if failures else 'complete',
            'technical_failure_dimensions': failures,
            'unscored_dimensions': [row['dimension'] for row in rows if row['score'] is None],
            'dimension_statuses': [{'dimension': row['dimension'], 'status': row['status'],
                                    'score': row['score'], 'origin': row.get('origin')}
                                   for row in rows],
            'all_five_dimensions_assessed': all(row['status'] == 'assessed' for row in rows)}


def render(output: Path = OUTPUT, editorial_path: Path | None = None) -> dict[str, Any]:
    from .aggregate import load_result

    evaluation, error = load_result(output, DAMAGE_ID, 'full')
    if error is not None or evaluation is None:
        raise ValueError(f'DAMAGE requires current independent verification before rendering: {error}')
    verification = verification_details(evaluation)
    report_path = output / 'reports/full' / f'{DAMAGE_ID}.json'
    if not report_path.is_file():
        raise FileNotFoundError('DAMAGE five-dimensional report is not yet available; no figure generated')
    report = Report.model_validate(read(report_path)).model_dump(mode='json')
    editorial_path = editorial_path or output / 'figures/DAMAGE_editorial_en.json'
    if not editorial_path.is_file():
        raise FileNotFoundError('Prepare a source-bound English editorial map from the actual report first')
    summaries = load_editorial(report, editorial_path)
    original_bytes = ORIGINAL.read_bytes()
    root = ET.fromstring(original_bytes)
    before_outside = digest(outside_report(root))
    before_panels = {name: digest(ET.tostring(panel(root, name)))
                     for name in ('panel_a', 'panel_b', 'panel_c', 'panel_d')}
    replace_panel(root, summaries)
    title_node(root).text = MANUSCRIPT_TITLE
    after_panels = {name: digest(ET.tostring(panel(root, name))) for name in before_panels}
    if before_panels != after_panels or before_outside != digest(outside_report(root, restore_title=True)):
        raise ValueError('A region outside the report panel and authorised heading changed')
    destination = output / 'figures/Fig6.svg'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))
    export(destination)
    audit = {'source': str(ORIGINAL), 'source_sha256': digest(original_bytes),
             'report_path': str(report_path), 'report_sha256': digest(encoded(report).encode()),
             'editorial_path': str(editorial_path), 'changed_region': 'panel_e',
             'outside_report_unchanged': False,
             'outside_report_unchanged_except_authorised_title': True,
             'authorised_title_change': {'parent_id': 'fig6_damage', 'x': '14', 'y': '29',
                                         'before': ORIGINAL_TITLE, 'after': MANUSCRIPT_TITLE,
                                         'attributes_unchanged': True},
             'outside_report_original_sha256': before_outside,
             'outside_report_restored_title_sha256': digest(outside_report(root, restore_title=True)),
             'original_panels_sha256': before_panels, 'new_panels_sha256': after_panels,
             'output_files': [str(destination.with_suffix(suffix)) for suffix in ('.svg', '.pdf', '.png')],
             'editorial_source_quotes_checked': True,
             'current_independent_evaluation_checked': True,
             **verification,
             'meaning': 'Literal source links verified; scientific fidelity of the English summary requires editorial review.'}
    write(output / 'figures/Fig6_unchanged_regions.json', audit)
    return audit
