"""Place approved replacement SVGs into existing Fig3 slots."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg

ROOT = Path(__file__).resolve().parents[2]
FIG3 = ROOT / 'outputs/fig3_reference'
ASSETS = ROOT / 'outputs/fig3_extensions/abcd/fig3_replacements'
SVG = 'http://www.w3.org/2000/svg'
XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG)
ET.register_namespace('xlink', XLINK)
FILES = {'b_upper_comparisons': 'B_precision_joint_and_historical.svg', 'c6': 'D_c6_graph_relation.svg'}


def insert_replacements(root: ET.Element, layout: dict) -> None:
    """Keep geometry unchanged and prevent cross-SVG clip/marker ID collisions."""
    for child in list(root):
        if child.get('id', '').startswith('replacement_'):
            root.remove(child)
    for key, name in FILES.items():
        group_id = 'replacement_' + key
        for child in list(root):
            if child.get('id') == group_id:
                root.remove(child)
        x, y, width, height = [v*72/25.4 for v in layout['components'][key]['page_box_mm']]
        group = ET.SubElement(root, f'{{{SVG}}}g', {'id': group_id})
        # Cover only the approved slot; existing source judgments are untouched.
        ET.SubElement(group, f'{{{SVG}}}rect', {'x':str(x),'y':str(y),
            'width':str(width),'height':str(height),'fill':'#FBFDFF'})
        component = ET.parse(ASSETS/name).getroot()
        identities = {node.get('id'): group_id+'_'+node.get('id') for node in component.iter() if node.get('id')}
        for node in component.iter():
            for attribute, value in list(node.attrib.items()):
                if attribute == 'id':
                    node.set(attribute, identities[value])
                elif value.startswith('#') and value[1:] in identities:
                    node.set(attribute, '#' + identities[value[1:]])
                elif 'url(#' in value:
                    for old, new in identities.items():
                        value = value.replace('url(#'+old+')', 'url(#'+new+')')
                    node.set(attribute, value)
        component.set('x', str(x)); component.set('y', str(y))
        component.set('width', str(width)); component.set('height', str(height))
        group.append(component)


def export(path: Path) -> None:
    cairosvg.svg2pdf(url=str(path), write_to=str(path.with_suffix('.pdf')))
    cairosvg.svg2png(url=str(path), write_to=str(path.with_suffix('.png')), scale=1.875)


def main() -> None:
    os.environ['FONTCONFIG_FILE'] = str(FIG3/'layouts/fonts.conf')
    layout = json.loads((FIG3/'layouts/style.json').read_text())
    master = FIG3/'final/Fig3.svg'
    root = ET.parse(master).getroot()
    insert_replacements(root, layout)
    ET.ElementTree(root).write(master, encoding='utf-8', xml_declaration=True)
    export(master)
    for key in ('b', 'c'):
        panel = layout['panels'][key]
        x, y, width, height = [panel[k]*72/25.4 for k in ('x','y','w','h')]
        node = copy.deepcopy(root)
        node.set('viewBox', f'{x} {y} {width} {height}')
        node.set('width', f'{width}pt'); node.set('height', f'{height}pt')
        path = FIG3/'panels'/f'{key}.svg'
        ET.ElementTree(node).write(path, encoding='utf-8', xml_declaration=True)
        export(path)
    print('Updated Fig3 and panels b/c in SVG, PDF and PNG.')


if __name__ == '__main__':
    main()
