"""Update report labels in the whole Fig2 without changing scientific records."""
from __future__ import annotations

import copy
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import ImageFont

from .config import OUTPUT, ROOT
from .damage_figure import export
from .io import write

ORIGINAL = ROOT / 'outputs/fig1-11_final/Fig2/Fig2.svg'
SVG = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
CHANGES = (
    ('a_claim_output', '20', '134', 'Limitation', 'Scope', 108),
    ('a_paper_output', '45', '19.0', 'Whole-paper', 'Five-dimensional', 83),
    ('a_paper_output', '45', '44.0', 'Traceable report', 'Source-linked analysis', 83),
    ('a_paper_output', '20', '66', 'Contributions', 'Historical originality', 108),
    ('a_paper_output', '20', '83', 'Historical increment', 'Understanding & capability', 108),
    ('a_paper_output', '20', '100', 'Knowledge relations', 'Knowledge organisation', 108),
    ('a_paper_output', '20', '117', 'Graph interpretation', 'Evidential basis', 108),
    ('a_paper_output', '20', '134', 'Limitations', 'Scientific value', 108),
    ('a_paper_output', '20', '151', 'Evidence links', 'Detailed analysis retained', 108),
    ('a_outputs', '816', '35', 'Whole-paper report', 'Five-dimensional report', 182),
    ('a_outputs', '816', '54', 'Integrated, source-linked analysis',
     'Five dimensions + detailed analysis', 182),
    ('e_report', '6', '14.5', 'Innovation analysis report', 'Detailed analysis (retained)', 165),
    ('e_synthesis_method', '9', '56', 'Retain disagreement and limitations',
     'Retain disagreement and scope', 185),
    ('e_report', '7', '102', 'Knowledge relations · Limitations',
     'Knowledge relations · Scope', 165),
)


def fitted_size(node: ET.Element, text: str, width: float) -> float:
    original = float(node.attrib['font-size'])
    font_name = 'timesbd.ttf' if node.get('font-weight') == 'bold' else 'times.ttf'
    font = ImageFont.truetype(str(Path('/mnt/c/Windows/Fonts') / font_name), 1000)
    size = min(original, width / (font.getlength(text) / 1000))
    return round(size, 4)


def update(root: ET.Element) -> list[dict[str, Any]]:
    audit = []
    for group, x, y, before, after, width in CHANGES:
        parent = root.find(f'.//*[@id="{group}"]')
        if parent is None:
            raise ValueError(f'Missing figure group: {group}')
        nodes = [e for e in parent.iter(f'{{{SVG}}}text')
                 if e.get('x') == x and e.get('y') == y and e.text == before]
        if len(nodes) != 1:
            raise ValueError(f'Expected exactly one unchanged text node: {group} {before}')
        node = nodes[0]
        old_attributes = dict(node.attrib)
        size = fitted_size(node, after, width)
        if size < float(node.attrib['font-size']):
            node.set('font-size', str(size))
        node.text = after
        audit.append({'group': group, 'x': x, 'y': y, 'before': before, 'after': after,
                      'attributes_before': old_attributes, 'attributes_after': dict(node.attrib),
                      'fitted_width': width})
    return audit


def render(output: Path = OUTPUT) -> dict[str, Any]:
    original = ORIGINAL.read_bytes()
    root = ET.fromstring(original)
    before = copy.deepcopy(root)
    changes = update(root)
    restored = copy.deepcopy(root)
    for change in changes:
        parent = restored.find(f'.//*[@id="{change["group"]}"]')
        node = next(e for e in parent.iter(f'{{{SVG}}}text')
                    if e.get('x') == change['x'] and e.get('y') == change['y']
                    and e.text == change['after'])
        node.text = change['before']
        node.attrib.clear()
        node.attrib.update(change['attributes_before'])
    if ET.tostring(restored) != ET.tostring(before):
        raise ValueError('Unexpected modification outside the declared text nodes')
    destination = output / 'figures/Fig2.svg'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))
    export(destination)
    audit = {'source': str(ORIGINAL), 'source_sha256': hashlib.sha256(original).hexdigest(),
             'changed_text_nodes': changes, 'only_declared_text_nodes_changed': True,
             'all_other_elements_and_attributes_unchanged': True,
             'scientific_content_and_numbers_unchanged': True,
             'output_files': [str(destination.with_suffix(s)) for s in ('.svg', '.pdf', '.png')],
             'note': 'Only labels and necessary font fitting changed; original report excerpt retained.'}
    write(output / 'figures/Fig2_text_changes.json', audit)
    return audit


if __name__ == '__main__':
    result = render()
    print(f"Updated {len(result['changed_text_nodes'])} text nodes; scientific content retained.")
