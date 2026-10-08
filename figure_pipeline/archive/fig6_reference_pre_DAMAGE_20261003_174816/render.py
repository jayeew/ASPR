"""Editable SVG artwork using the same scene graph and fonts as Fig.1–3."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from figure_pipeline.fig1_reference.svg import Scene, el, mix
from .data import OUT, read, write
from .editorial import ANCHORS, CLAIM_TEXT, COLORS, REPORT_ROWS, SYNTHESIS, SYNTHESIS_SOURCE

W, H = 1100, 1375
INK, MUTED, EDGE, PALE = '#24282D', '#627181', '#A9CDE4', '#EEF6FC'
PANELS = {'a': (12, 63, 1076, 239), 'b': (12, 314, 446, 555),
          'c': (470, 314, 618, 555), 'd': (12, 881, 446, 458),
          'e': (470, 881, 618, 458)}
TITLES = {'a': 'Manuscript and grounded contributions',
          'b': 'Historical evidence and differences',
          'c': 'From claim neighborhoods to joint structure',
          'd': 'Evidence-constrained judgments',
          'e': 'Traceable report and scientific takeaways'}
# Layout positions are editorial, not a metric embedding. Reused in all local views.
POSITIONS = {'C01': (70, 40), 'C02': (250, 40), 'C03': (70, 215),
             'C04': (270, 207), 'C05': (502, 92), 'C06': (505, 259),
             'H01': (195, 125), 'H02': (72, 125), 'H03': (320, 25),
             'H04': (333, 112), 'H05': (463, 23), 'H06': (398, 190),
             'H07': (491, 174), 'H08': (572, 162), 'H09': (352, 275),
             'H10': (392, 57)}


def para(s: Scene, x: float, y: float, value: str, width: float,
         size: float = 17.3, leading: float = 20.5, **kwargs: Any) -> float:
    return s.paragraph(x, y, value, width, size, leading, **kwargs)


def tag(s: Scene, x: float, y: float, text: str, color: str, w: float = 42) -> None:
    s.rect(x, y, w, 22, mix(color, '#ffffff', .9), color, 3, sw=.7)
    s.text(x + w / 2, y + 16.5, text, 16.5, color, True, anchor='middle', sans=True)


class Figure:
    def __init__(self) -> None:
        self.data = read(OUT / 'data/snapshot.json')
        self.components: dict[str, Scene] = {}
        self.panels: dict[str, Scene] = {}
        self.placements: dict[str, Any] = {}

    def component(self, ident: str, width: float, height: float) -> Scene:
        result = Scene(width, height, ident)
        result.root.set('data-component', ident)
        self.components[ident] = result
        return result

    def place(self, parent: Scene, child: Scene, x: float, y: float) -> None:
        parent.use(child, x, y)
        self.placements[child.ident] = {'parent': parent.ident, 'x': x, 'y': y,
                                        'width': child.width, 'height': child.height}

    def panel(self, key: str) -> Scene:
        _, _, w, h = PANELS[key]
        s = Scene(w, h, 'panel_' + key)
        s.rect(0, 0, w, h, '#FCFDFF', EDGE, 4, sw=1)
        s.rect(1, 1, w - 2, 32, PALE)
        s.text(10, 24, key, 27, INK, True)
        s.text(36, 23, TITLES[key], 22 if key in 'ac' else 21, INK, True)
        self.panels[key] = s
        return s

    def manuscript(self) -> None:
        s = self.panel('a')
        identity = self.component('a1_manuscript', 267, 203)
        para(identity, 0, 18, 'Ultrathin liquid cells for microsecond time-resolved cryo-EM',
             255, 20, 21, bold=True)
        identity.text(0, 85, 'Curtis et al. · Nat. Commun. (2026)', 15.5, MUTED)
        identity.text(0, 106, 'Published article · cutoff: 22 Jan 2026', 15.5, MUTED)
        para(identity, 0, 123, 'How can cryo-EM capture slower molecular motion?', 255, 17.3, 20, bold=True)
        identity.rect(3, 158, 211, 5, '#A7CDE4')
        identity.rect(3, 165, 211, 17, '#E9F5FB')
        identity.rect(3, 184, 211, 5, '#A7CDE4')
        for x in (25, 58, 103, 137, 179):
            identity.node(x, 173, fill='#8396AB', radius=4, stroke='#60758C')
        identity.text(218, 166, 'SiO₂', 14.5, MUTED)
        identity.text(218, 189, 'SiO₂', 14.5, MUTED)
        identity.text(4, 198, 'M01 · two 1.4-nm layers · schematic', 14.5, MUTED)
        self.place(s, identity, 13, 34)
        s.line(285, 47, 285, 225, EDGE)
        for i, claim in enumerate(self.data['claims']):
            card = self.component(f'a{i+2}_C{i+1:02}', 249, 90)
            color = COLORS[i]
            card.rect(0, 0, 249, 90, mix(color, '#ffffff', .965), mix(color, '#ffffff', .5), 3)
            tag(card, 7, 6, f'C{i+1:02}', color)
            card.text(57, 22, self.data['claim_labels'][i], 17.3, INK, True)
            para(card, 8, 45, CLAIM_TEXT[i], 233, 17.3, 18.5, max_lines=2)
            card.text(8, 82, ANCHORS[i], 14.8, MUTED)
            self.place(s, card, 299 + (i % 3) * 257, 42 + (i // 3) * 95)
        s.text(1060, 234, 'C01–C06: system summaries; M01–M06: manuscript anchors', 14, MUTED, anchor='end')

    def historical(self) -> None:
        s = self.panel('b')
        matrix = self.component('b1_history_matrix', 420, 262)
        matrix.text(0, 17, 'Selected claims × independent prior works', 19, INK, True)
        matrix.text(0, 39, 'Saved GEAR assessments · A = abstract, F = full text', 15.5, MUTED)
        heads = [('E01', 'Membranes', '2020 · F'), ('E02', 'Laser melting', '2021 · A'),
                 ('E03', 'Orientation', '2025 · F'), ('E04', 'Translocation', '2021 · F')]
        for j, (eid, title, year) in enumerate(heads):
            x = 111 + j * 84
            matrix.text(x, 62, eid, 17.3, INK, True, anchor='middle')
            matrix.text(x, 81, title, 15, MUTED, anchor='middle')
            matrix.text(x, 99, year, 15, MUTED, anchor='middle')
        style = {'PARTIAL_ANTECEDENT': ('P', '#237A67'), 'EXTENSION': ('E', '#713BCB'),
                 'PARALLEL': ('L', '#616F86'), 'BUILDING_BLOCK': ('B', '#1269DF')}
        lookup = {(r['claim'], r['work']): r['assessment'] for r in self.data['relations']}
        for i, ci in enumerate((1, 2, 3, 5)):
            y = 108 + i * 28
            matrix.rect(0, y, 420, 27, '#F2F6FA' if i % 2 == 0 else '#ffffff')
            matrix.text(9, y + 20, f'C{ci:02}', 18, COLORS[ci-1], True, sans=True)
            for j in range(4):
                relation = lookup[f'C{ci:02}', f'E{j+1:02}']
                symbol, color = style[relation['relation_label']] if relation else ('?', '#A3ADB8')
                matrix.text(111 + j * 84, y + 20, symbol, 20, color, True, anchor='middle', sans=True)
        matrix.text(0, 237, 'P  Partial antecedent     E  Extension     B  Building block', 15.5, MUTED)
        matrix.text(0, 257, 'L  Parallel     ?  No saved assessment for this pair', 15.5, MUTED)
        self.place(s, matrix, 13, 43)
        compare = self.component('b2_source_comparison', 420, 237)
        compare.text(0, 18, 'C02 · Extend the laser-melting route', 19, INK, True)
        for x, label, sub, quote, color in [
            (0, 'Target · M02', 'Results, p. 2',
             '“…allows us to reach longer timescales with a higher yield of intact samples.”', COLORS[1]),
            (214, 'Prior work · E02', '2021 abstract',
             '', '#65798F')]:
            compare.rect(x, 30, 206, 140, '#ffffff', EDGE, 3)
            compare.text(x + 8, 50, label, 17.3, color, True)
            compare.text(x + 8, 70, sub, 14.8, MUTED)
            if quote:
                para(compare, x + 8, 93, quote, 190, 17.3, 19, italic=True)
        prior = self.data['display_prior_quote']
        para(compare, 222, 93, '“' + prior + '”', 190, 17.3, 19, italic=True)
        para(compare, 0, 192, 'Shared basis: laser melting and revitrification. Residual difference: sealed SiO₂ cells plus short-pulse trains extend the usable time window.',
             420, 17.3, 20)
        self.place(s, compare, 13, 314)

    def network(self, s: Scene, targets: list[str], x: float, y: float,
                scale: float, miniature: bool = False) -> None:
        ids = {v['claim_id']: k for k, v in self.data['historical_nodes'].items()}
        ids.update({c['claim_id']: f'C{i:02}' for i, c in enumerate(self.data['claims'], 1)})
        insertion = [(ids[a], ids[b]) for a, b in self.data['joint']['insertion_edges']
                     if a in ids and b in ids and ids[a] in targets]
        active = set(targets) | {b for _, b in insertion}
        history = [(ids[a], ids[b]) for a, b in self.data['joint']['historical_edges']
                   if a in ids and b in ids and ids[a] in active and ids[b] in active]
        def xy(key: str) -> tuple[float, float]:
            px, py = POSITIONS[key]
            return x + px * scale, y + py * scale
        for a, b in history:
            extra = frozenset((a, b)) in {frozenset((ids[u], ids[v])) for u, v in self.data['joint_only_edges'] if u in ids and v in ids}
            s.line(*xy(a), *xy(b), '#394C61' if extra else '#B2BDC8',
                   2.1 if extra and not miniature else .8, dash='5 3' if extra else None)
        for a, b in insertion:
            s.line(*xy(a), *xy(b), COLORS[int(a[-2:])-1], .65 if miniature else 1.05, opacity=.42)
        for key in sorted(active):
            px, py = xy(key)
            target = key.startswith('C')
            r = (10 if target else 3.2) if miniature else (18 if target else 10)
            color = COLORS[int(key[-2:])-1] if target else '#E8EDF2'
            s.node(px, py, fill=color, radius=r, stroke='#ffffff' if target else '#8FA0B2', sw=.8)
            if target or not miniature:
                s.text(px, py + (4 if miniature else 5), key, 11 if miniature else (15.5 if target else 13.8),
                       '#ffffff' if target else '#435164', True, anchor='middle', sans=True)
            elif miniature:
                s.text(px, py - 5, key, 11, MUTED, anchor='middle')

    def graph(self) -> None:
        s = self.panel('c')
        local = self.component('c1_local_neighborhoods', 592, 100)
        for j, key in enumerate(('C01', 'C04', 'C05')):
            x = j * 199
            local.rect(x, 0, 194, 99, '#ffffff', EDGE, 3)
            local.text(x + 9, 19, key + ' · local view', 17.3, COLORS[int(key[-2:])-1], True)
            self.network(local, [key], x + 9, 21, .30, True)
        self.place(s, local, 13, 41)
        net = self.component('c2_joint_network', 592, 314)
        net.text(0, 18, 'Union view · all 6 targets, selected 10 of 50 neighbors', 18.5, INK, True)
        net.text(0, 38, 'All edges among the displayed nodes are retained.', 15.5, MUTED)
        self.network(net, [f'C{i:02}' for i in range(1, 7)], 35, 40, .9)
        net.text(440, 148, 'G02', 15, '#394C61', True)
        net.text(148, 159, 'G01', 15, COLORS[1], True)
        net.text(548, 225, 'G03', 15, COLORS[4], True)
        self.place(s, net, 13, 150)
        notes = self.component('c3_graph_facts', 592, 87)
        notes.line(0, 0, 592, 0, EDGE)
        para(notes, 0, 20, 'G01  C01 + C04 share specimen and interface work (H01, H02).', 592, 17.3, 20)
        para(notes, 0, 42, 'G03  C05 + C06 share a prior L1-stalk result (H08).', 592, 17.3, 20)
        para(notes, 0, 64, 'G02  An existing ribosome-assembly / in situ-imaging edge (H05–H06) links the C05 and C04 neighborhoods.', 592, 16.6, 19)
        self.place(s, notes, 13, 464)

    def judgments(self) -> None:
        s = self.panel('d')
        s.text(13, 52, 'Specific statements, not whole-claim verdicts', 16.8, MUTED)
        rows = [
            ('SUPPORTED', '#53766A', 'C04 · Imaging quality',
             'M04: sampling compensation factor 0.53 → 0.99, with near-atomic reconstructions.',
             'Retain performance in the tested specimens.', 'M04 → GEAR C04 → report paragraph 4'),
            ('NARROWED', '#A77335', 'C02 · Define the stability benefit',
             'Extracted candidate: limiting “sample damage”. Grounding supports less liquid-film breakup.',
             'Keep the observed integrity benefit.', 'M02 → shared grounding → R02'),
            ('WITHHELD', '#63748C', 'C06 · Helix 68 as the cause',
             'A slow L1-stalk response suggests additional interactions; it does not identify their cause.',
             'Keep helix 68 as a candidate mechanism.', 'M06 → GEAR C06 → R04')]
        for i, (state, color, heading, evidence, final, links) in enumerate(rows):
            c = self.component(f'd{i+1}_judgment', 420, 116)
            c.rect(0, 0, 420, 116, '#ffffff', mix(color, '#ffffff', .5), 3)
            c.rect(0, 0, 420, 24, mix(color, '#ffffff', .91), radius=3)
            c.text(9, 18, state, 16, color, True, sans=True)
            c.text(410, 18, heading, 17, INK, True, anchor='end')
            pos = para(c, 9, 45, evidence, 402, 17.3, 19.5)
            para(c, 9, pos + 2, final, 402, 17.3, 19.5, bold=True)
            c.text(9, 110, links, 15, MUTED)
            self.place(s, c, 13, 65 + i * 122)
        para(s, 13, 438, 'The candidate in the middle row is system-extracted, not an author quotation.', 420, 15.5, 17)

    def report(self) -> None:
        s = self.panel('e')
        s.text(13, 52, 'Saved fusion report · faithful English translations', 16.8, MUTED)
        s.text(438, 52, 'Reader takeaways', 17.3, INK, True)
        for i, row in enumerate(REPORT_ROWS):
            assert row['source'] in self.data['report']['body'], row['id']
            c = self.component(f'e{i+1}_{row["id"]}', 592, 88 if i == 0 else 73)
            c.rect(0, 0, 405, 88 if i == 0 else 73, '#ffffff', '#D8E4EF', 3)
            color = COLORS[[0, 1, 4, 5][i]]
            tag(c, 7, 5, row['id'], color)
            c.text(57, 21, row['title'], 18, INK, True)
            para(c, 8, 42, row['english'], 387, 16.6, 18.2)
            c.rect(420, 0, 172, 88 if i == 0 else 73, mix(color, '#ffffff', .965), mix(color, '#ffffff', .62), 3)
            c.line(405, 40, 420, 40, color, 1)
            c.text(428, 18, row['question'], 17, color, True)
            para(c, 428, 39, row['takeaway'], 155, 16.6, 18.3)
            self.place(s, c, 13, [64, 163, 247, 331][i])
            s.text(21, [160, 244, 328, 412][i], row['links'], 13.7, MUTED)
        assert SYNTHESIS_SOURCE in self.data['report']['body']
        s.rect(13, 418, 592, 33, '#FFF5E7', '#E3C195', 3)
        para(s, 21, 431, SYNTHESIS, 576, 15.5, 16, bold=True)

    def render(self) -> None:
        self.manuscript()
        self.historical()
        self.graph()
        self.judgments()
        self.report()
        scene = Scene(W, H, 'Fig6')
        scene.rect(0, 0, W, H, '#ffffff')
        scene.text(W/2, 31, 'Fig. 6 | From a scientific paper to an evidence-constrained report', 30, INK, True, anchor='middle')
        scene.text(W/2, 52, 'Ultrathin liquid cells for microsecond time-resolved cryo-EM', 19, MUTED, anchor='middle')
        for key, panel in self.panels.items():
            self.save(panel, OUT / 'panels' / (key + '.svg'))
            x, y, _, _ = PANELS[key]
            scene.use(panel, x, y)
        scene.text(13, 1356, 'Graph: colored lines = semantic insertion; gray = historical edges; dashed = visible only in the joint view.', 15.3, MUTED)
        scene.text(13, 1373, '*G02 is a figure-added graph cross-reference. Graph links do not establish causality or priority. Full sources accompany the figure.', 14.5, MUTED)
        self.save(scene, OUT / 'final/Fig6.svg')
        for ident, component in self.components.items():
            self.save(component, OUT / 'components' / (ident + '.svg'))
        write(OUT / 'layouts/style.json', {'width_mm': 180, 'height_mm': 225, 'viewbox': [W,H],
              'panels': PANELS, 'components': self.placements, 'network_positions': POSITIONS,
              'claim_colors': COLORS, 'body_font_pt': 17.3 * 180 / 1100 * 72 / 25.4})
        write(OUT / 'layouts/text_boxes.json', scene.text_boxes)
        write(OUT / 'data/report_translations.json', REPORT_ROWS + [{'id':'R05', 'source':SYNTHESIS_SOURCE,'english':SYNTHESIS,'paragraph':1}])

    @staticmethod
    def save(scene: Scene, path: Path) -> None:
        root = scene.document()
        root.set('width', f'{scene.width * 180 / W:.6f}mm')
        root.set('height', f'{scene.height * 180 / W:.6f}mm')
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def render() -> None:
    Figure().render()
