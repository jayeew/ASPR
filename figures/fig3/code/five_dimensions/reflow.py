"""Recompose panel c from existing vector art without changing statistics."""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

from .render import NS, SVG, XLINK, PT_PER_MM, ROOT, _prefix_ids


def reflow_panel_c(root: ET.Element) -> dict:
    figure = root.find("s:g[@id='figure_1']", NS)
    assert figure is not None
    index = {node.get('id'): node for node in figure}
    coverage_ids = ['axes_20', 'text_694']
    composition_ids = [f'axes_{i}' for i in range(21, 27)] + [f'text_{i}' for i in range(695, 701)]
    groups = {}
    for name, identifiers in [('coverage', coverage_ids), ('composition', composition_ids)]:
        group = ET.Element(f'{{{SVG}}}g')
        for identifier in identifiers:
            node = index[identifier]
            figure.remove(node)
            group.append(node)
        groups[name] = group
    # The source retains an obscured old case as well as its replacement.
    rescued = ET.SubElement(root, f'{{{SVG}}}defs', {'id':'panel_c_removed_art_definitions'})
    def rescue(node: ET.Element) -> None:
        for definitions in node.findall('.//s:defs', NS):
            for definition in definitions:
                identifier = definition.get('id')
                if identifier and not any(d.get('id') == identifier for d in rescued):
                    rescued.append(copy.deepcopy(definition))
    for identifier in ['axes_27', 'text_701']:
        rescue(index[identifier])
        figure.remove(index[identifier])
    old_case = root.find("s:g[@id='replacement_c6']", NS)
    assert old_case is not None
    rescue(old_case)
    root.remove(old_case)
    score = index['five_dimensions_replacement_c1']
    pair = index['five_dimensions_replacement_c3']
    for node in [score, pair]:
        figure.remove(node)
    for node, old, new in [(score, 'c1 ', 'c2 '), (pair, 'c2 ', 'c3 ')]:
        for text in node.iter(f'{{{SVG}}}text'):
            if text.text and text.text.startswith(old):
                text.text = new + text.text[len(old):]
    for group, identifier, new in [(groups['coverage'], 'text_694', '4'),
                                    (groups['composition'], 'text_695', '5')]:
        title = next(n for n in group.iter() if n.get('id') == identifier)
        title.findall('.//s:use', NS)[1].set(f'{{{XLINK}}}href',
                                           f'#TimesNewRomanPS-BoldMT-{ord(new):x}')
    def place(node: ET.Element, old: tuple, new: tuple, name: str) -> None:
        ox, oy, ow, oh = old
        nx, ny, nw, nh = new
        scale = min(nw / ow, nh / oh)
        tx, ty = (nx - ox * scale) * PT_PER_MM, (ny - oy * scale) * PT_PER_MM
        wrapper = ET.SubElement(figure, f'{{{SVG}}}g',
            {'id': 'panel_c_reflow_' + name,
             'transform': f'translate({tx} {ty}) scale({scale})'})
        wrapper.append(node)
    place(score, (12,221,254,45), (112,221,234,43), 'scores')
    place(pair, (269,221,139,45), (112,266,116,38), 'paired')
    place(groups['coverage'], (12,269,123,34), (233,266,115,38), 'coverage')
    # Three rows by two columns; preserve every original wedge and method label.
    composition = groups['composition']
    for j in range(6):
        node = next(n for n in composition if n.get('id') == f'axes_{21+j}')
        old_x = (429.448819 + (j % 3)*84.188976) / PT_PER_MM
        old_y = (798.660 + (j // 3)*31.804724) / PT_PER_MM
        for label in node.findall('s:g', NS):
            if label.get('id', '').startswith('text_'):
                px, py = old_x*PT_PER_MM, old_y*PT_PER_MM
                label.set('transform',f'translate({px} {py}) scale(0.7) translate({-px} {-py})')
        new_x, new_y = 362 + (j % 2)*29, 240 + (j // 2)*23
        place(node, (old_x-6,old_y-6,12,12),
              (new_x-10,new_y-10,20,20), f'assertion_{j}')
    title = ET.SubElement(figure, f'{{{SVG}}}text', {
        'x':str(351*PT_PER_MM),'y':str(224*PT_PER_MM),
        'font-family':'Arial','font-size':'9.5','font-weight':'bold','fill':'#10213B'})
    title.text = 'c5 Assertion composition'
    for j in range(5):
        node = next(n for n in composition if n.get('id') == f'text_{696+j}')
        ox = [138,169,200,138,169][j]
        oy = [299.1469,299.1469,299.1469,302.2469,302.2834][j]
        place(node,(ox,oy,1,1),(351+(j%3)*19,298+(j//3)*4,0.8,0.8),f'assertion_legend_{j}')
    diagram_path = ROOT / 'outputs/reviewer_open_coding_pilot_20261116/open_discovery/fig_reviewer_concerns_four_layer/reviewer_five_dimensions_four_layer.svg'
    diagram = ET.parse(diagram_path).getroot()
    # Enlarge the central dimension names and counts independently of the legends.
    for i, (name_id, count_id) in enumerate([(2,3),(5,6),(11,12),(15,16),(22,23)]):
        angle = math.radians(i*72+36)
        x = (455+157*math.sin(angle))*.72
        y = (630-157*math.cos(angle))*.72
        for identifier, cy, factor in [(name_id,y-19*.72,1.3),(count_id,y+37*.72,1.25)]:
            label=diagram.find(f".//s:g[@id='text_{identifier}']",NS)
            assert label is not None
            label.set('transform',f'translate({x} {cy}) scale({factor}) translate({-x} {-cy})')
    _prefix_ids(diagram, 'reviewer_concerns_')
    # Keep the complete diagram and its legends; the panel heading replaces its large title.
    for parent in diagram.iter():
        for child in list(parent):
            if child.tag == f'{{{SVG}}}text' and child.text and (child.text == 'Reviewer concerns informing the five dimensions' or child.text.startswith('Selected concerns are highlighted') or child.text.startswith('1000 roster papers:')):
                parent.remove(child)
    nested = ET.SubElement(figure, f'{{{SVG}}}svg',
        {'id': 'panel_c_reflow_framework', 'x': str(12*PT_PER_MM), 'y': str(227*PT_PER_MM),
         'width': str(113*PT_PER_MM), 'height': str(76*PT_PER_MM),
         'viewBox': '0 82 890 736', 'preserveAspectRatio': 'xMinYMid meet'})
    for child in diagram:
        if child.tag != f'{{{SVG}}}metadata':
            nested.append(child)
    heading = ET.SubElement(figure, f'{{{SVG}}}text',
        {'x':str(12*PT_PER_MM), 'y':str(224*PT_PER_MM), 'font-family':'Arial',
         'font-size':'9', 'font-weight':'bold', 'fill':'#10213B'})
    heading.text = 'c1 Reviewer concerns informing the five dimensions'
    return {'layout':'c1 enlarged and spans two rows; central c2 above c3/c4; c5 three rows by two columns',
            'removed':'Old c5 Graph relation and manuscript evidence',
            'framework_source':str(diagram_path), 'statistics_changed':False}
