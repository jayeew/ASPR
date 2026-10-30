"""Render Fig.9 using the editable SVG and export machinery of Fig.1–8."""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import Scene, el, measure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fig9_reference'
BLUE, ORANGE, GREEN, PURPLE = '#006BFF', '#EF4B08', '#00A878', '#8428E8'
INK, TEAL, GRAY, RED = '#101D64', '#229FAF', '#8093A6', '#ED244A'
W, H = 1680, 960


class Figure(Scene):
    def label(self, x: float, y: float, value: str, size: float = 15,
              color: str = INK, bold: bool = False) -> None:
        self.text(x, y, value, size, color, bold)

    def box(self, x: float, y: float, w: float, h: float,
            color: str = BLUE, fill: str = '#FFFFFF') -> None:
        self.rect(x, y, w, h, fill, color, radius=4, sw=.65)

    def arrow(self, x: float, y: float, xx: float, yy: float,
              color: str = BLUE, width: float = 1.6, dash: bool = False,
              head_length: float | None = None) -> None:
        self.line(x, y, xx, yy, color, width, '5 3' if dash else None)
        a = math.atan2(yy-y, xx-x)
        length = head_length if head_length is not None else min(7 if width < 3 else 11, math.hypot(xx-x, yy-y)*.65)
        pts = [(xx, yy)] + [(xx-length*math.cos(a+t), yy-length*math.sin(a+t))
                           for t in (-.48, .48)]
        self.path('M'+' L'.join(f'{u},{v}' for u, v in pts)+' Z', fill=color)

    def doc(self, x: float, y: float, color: str = BLUE, scale: float = 1) -> None:
        self.path(f'M{x},{y} h{14*scale} l{6*scale},{6*scale} v{23*scale} h{-20*scale} Z',
                  fill='#FFFFFF', stroke=color, width=1.5)
        self.path(f'M{x+14*scale},{y} v{6*scale} h{6*scale}', stroke=color, width=1)
        for i in range(4):
            self.line(x+4*scale, y+(11+i*4)*scale, x+16*scale,
                      y+(11+i*4)*scale, color, .8)

    def search(self, x: float, y: float, color: str = BLUE) -> None:
        self.add(el('circle', cx=x, cy=y, r=9, fill='none', stroke=color, stroke_width=2))
        self.line(x+7, y+7, x+16, y+16, color, 2.5)

    def network(self, x: float, y: float, color: str = BLUE, r: float = 24) -> None:
        for a in (0, 1.3, 2.5, 3.8, 5):
            xx, yy = x+r*math.cos(a), y+r*math.sin(a)
            self.line(x, y, xx, yy, color, 1.4)
            self.add(el('circle', cx=xx, cy=yy, r=4, fill=color))
        self.add(el('circle', cx=x, cy=y, r=6, fill='white', stroke=color, stroke_width=2))

    def shield(self, x: float, y: float, color: str = BLUE) -> None:
        self.path(f'M{x},{y} l11,4 v10 q0,9 -11,16 q-11,-7 -11,-16 v-10 Z',
                  fill='white', stroke=color, width=1.7)
        self.path(f'M{x-6},{y+14} l4,4 l8,-9', stroke=color, width=2)

    def mark(self, x: float, y: float, mode: str) -> None:
        color = GREEN if mode == 'match' else ORANGE if mode == 'different' else GRAY
        self.add(el('circle', cx=x, cy=y, r=7, fill=color))
        if mode == 'match':
            self.path(f'M{x-4},{y} l3,3 l5,-6', stroke='white', width=1.4)
        else:
            self.text(x, y+4, '!' if mode == 'different' else '?',
                      11, 'white', True, anchor='middle', sans=True)


def icon(f: Figure, kind: str, x: float, y: float,
         color: str = BLUE, scale: float = 1) -> None:
    """Pure vector pictograms, with no platform-dependent symbol glyphs."""
    g = Figure(40, 40, f'icon-{kind}-{x}-{y}')
    if kind == 'gear':
        points = []
        for i in range(48):
            angle = i*math.pi/24
            radius = 16 if i % 6 in (1, 2, 3, 4) else 12.5
            points.append((20+radius*math.cos(angle), 20+radius*math.sin(angle)))
        g.path('M'+' L'.join(f'{u},{v}' for u, v in points)+' Z', fill=color)
        g.add(el('circle', cx=20, cy=20, r=7, fill='white'))
    elif kind == 'database':
        g.path('M7,10 v22 c0,8 26,8 26,0 v-22 Z', fill=color)
        for yy in (10, 18, 25, 32):
            g.path(f'M7,{yy} c0,8 26,8 26,0', stroke='white', width=1.4)
        g.add(el('ellipse', cx=20, cy=10, rx=13, ry=5, fill=color, stroke='white', stroke_width=1))
    elif kind == 'brain':
        for sign in (-1, 1):
            g.path(f'M20,7 C{20+sign*8},1 {20+sign*13},7 {20+sign*11},12 '
                   f'C{20+sign*19},12 {20+sign*19},22 {20+sign*13},24 '
                   f'C{20+sign*18},31 {20+sign*10},37 {20+sign*6},33 '
                   f'C{20+sign*1},38 20,34 20,29 Z', stroke=color, width=1.8)
            g.path(f'M{20+sign*11},12 l{-sign*6},3 v7 l{sign*7},3 M{20+sign*6},25 v6', stroke=color, width=1.5)
    elif kind == 'clock':
        g.add(el('circle', cx=20, cy=20, r=14, fill='none', stroke=color, stroke_width=2))
        g.path('M20,10 v11 l8,5', stroke=color, width=2)
    elif kind == 'refresh':
        g.path('M7,18 A14,14 0 0 1 31,10 M33,22 A14,14 0 0 1 9,30', stroke=color, width=2)
        g.path('M24,10 h8 v-8 M16,30 h-8 v8', stroke=color, width=2)
    elif kind == 'rank':
        for i, h in enumerate((12, 23, 32)):
            g.rect(5+i*11, 37-h, 7, h, color)
    elif kind == 'link':
        g.path('M18,24 l-4,4 c-8,8 -17,-1 -9,-9 l8,-8 c6,-6 13,-2 13,3 '
               'M22,16 l4,-4 c8,-8 17,1 9,9 l-8,8 c-6,6 -13,2 -13,-3', stroke=color, width=3)
    elif kind == 'search':
        g.search(17, 16, color)
    elif kind == 'network':
        g.network(20, 20, color, 14)
    elif kind == 'shield':
        g.shield(20, 4, color)
    elif kind == 'doc':
        g.doc(10, 5, color)
    elif kind == 'stop':
        g.add(el('circle', cx=20, cy=20, r=13, fill='none', stroke=color, stroke_width=2))
        g.rect(13, 13, 14, 14, color)
    g.root.set('transform', f'translate({x} {y}) scale({scale})')
    f.add(g.root)


def block_arrow(f: Figure, x: float, y: float, w: float = 22) -> None:
    f.path(f'M{x},{y-7} h{w-10} v-7 l10,14 l-10,14 v-7 h{-w+10} Z',
           fill=ORANGE, stroke='#FFFFFF', width=.7)


def supervisor(f: Figure) -> None:
    f.box(12, 80, 1656, 358, ORANGE, '#FFF0E1')
    icon(f, 'gear', 31, 82, ORANGE, 1.05)
    f.label(80, 112, 'Evidence Supervisor', 29, '#C92B0A', True)
    f.label(355, 111, '— State-aware decision and bounded orchestration', 22, '#C92B0A', True)
    constraints = [('Retrieval budget', 'database', 942, 135), ('Temporal cutoff', 'clock', 1083, 132),
                   ('Candidate cap', 'doc', 1221, 122), ('12-round bound', 'refresh', 1349, 138),
                   ('Legal action enforcement', 'shield', 1493, 160)]
    for text, kind, x, width in constraints:
        f.box(x, 91, width, 30, '#FFE2CB', '#FFFCF9')
        icon(f, kind, x+5, 94, BLUE, .6)
        f.label(x+31, 111, text, 12.2)
    modules = [(26, 410, 'S1', 'Observe evidence state', 'Summarize progress, queue, and resources'),
               (466, 378, 'S2', 'Generate legal actions', 'Determine allowable actions from state'),
               (874, 334, 'S3', 'Select next action', 'Choose one action from the legal set Aₜ'),
               (1238, 416, 'S4', 'Dispatch or finalize', 'Call tools or terminate based on the action')]
    for x, width, n, title, subtitle in modules:
        f.box(x, 132, width, 266, '#EDA77F', '#FFFCF9')
        f.rect(x+10, 139, 49, 35, ORANGE, radius=5)
        f.label(x+20, 165, n, 25, 'white', True)
        f.label(x+71, 157, title, 21, '#C92B0A', True)
        f.label(x+71, 176, subtitle, 14)
    for x in (440, 848, 1212):
        block_arrow(f, x, 266)
    f.box(36, 188, 390, 199, '#8DCEF0', '#FFFFFF')
    f.rect(37, 189, 388, 25, '#EDF7FF')
    f.label(48, 207, 'Current evidence state (Eₜ)', 16, BLUE, True)
    for i, (name, kind) in enumerate(zip(
            ['Search progress', 'Evidence queue', 'Relation evidence', 'Control resources'],
            ['search', 'doc', 'network', 'database'])):
        y = 215+i*42
        if i:
            f.line(42, y, 419, y, '#DBE9F0')
        icon(f, kind, 42, y+5, BLUE, .7)
        f.label(78, y+24, name, 14, INK, True)
    f.line(192, 215, 192, 386, '#DFE9F2')
    for y, label, width, done in ((230, 'normal search', 48, True), (247, 'contrastive search', 36, False)):
        f.label(200, y, label, 11.5)
        f.rect(287, y-8, 68, 8, '#DFECF6')
        f.rect(287, y-8, width, 8, '#51A8FF')
        if done:
            f.mark(371, y-4, 'match')
        else:
            f.add(el('circle', cx=371, cy=y-4, r=6, fill='white', stroke='#92ACCA', stroke_width=1.4))
        f.label(382, y, 'done' if done else 'pending', 10.5)
    f.label(200, 278, 'Unclassified works', 12.5)
    for x in (309, 335, 361):
        f.doc(x, 265, scale=.66)
    f.label(392, 279, '···', 18, BLUE)
    for x, mode, text in ((207, 'match', 'direct'), (271, 'different', 'partial'), (336, 'unresolved', 'unresolved')):
        f.mark(x, 321, mode)
        f.label(x+10, 325, text, 11.5)
    f.label(200, 357, 'Retrieval budget', 12)
    f.rect(287, 350, 68, 7, '#DFECF6')
    f.rect(287, 350, 48, 7, BLUE)
    f.label(362, 357, 'remaining', 10.5)
    f.label(200, 376, 'Used actions', 12)
    for i in range(7):
        f.add(el('circle', cx=291+i*18, cy=372, r=3.7, fill=BLUE if i < 3 else '#D9E7F1'))
    f.box(476, 188, 358, 199, '#8DCEF0')
    f.rect(477, 189, 356, 25, '#EDF7FF')
    f.label(489, 207, 'Legal action set Aₜ   (ordered condition → action)', 14.5, BLUE, True)
    rows = [('Initial search not done', 'Normal search', 'search'),
            ('Unclassified works exist', 'Verify relation', 'doc'),
            ('Contrastive search pending', 'Contrastive search', 'search'),
            ('Antecedent + check due', 'Stability check', 'shield'),
            ('No antecedent + expandable', 'Expand / Finalize', 'network'),
            ('No further follow-up required', 'Finalize', 'stop')]
    for i, (condition, action, kind) in enumerate(rows):
        y = 232+i*28
        if i % 2 == 0:
            f.rect(477, y-15, 356, 26, '#F7FAFD')
        icon(f, kind, 481, y-18, RED if i == 5 else BLUE, .48)
        f.label(506, y, condition, 12.7)
        f.arrow(677, y-5, 693, y-5, BLUE, 1.2)
        f.label(704, y, action, 12.7, GREEN if i == 1 else PURPLE if i > 2 else BLUE, True)
    for x, title in ((884, 'Deterministic selection'), (1045, 'Constrained LLM planner')):
        f.box(x, 194, 153, 88, '#99CFEB', '#F7FCFF')
        f.label(x+8, 216, title, 12.4, BLUE, True)
    icon(f, 'gear', 894, 232, '#438FFF', .82)
    f.paragraph(934, 240, 'If only one legal action exists', 96, 13, 16)
    icon(f, 'brain', 1054, 230, BLUE, .85)
    f.paragraph(1094, 240, 'If multiple legal actions exist', 98, 13, 16)
    f.box(923, 306, 236, 39, ORANGE, '#FFF9F4')
    f.doc(936, 312, ORANGE, .8)
    f.label(967, 331, 'Selected action  aₜ in Aₜ', 17, '#C92B0A', True)
    f.arrow(960, 283, 960, 303)
    f.arrow(1121, 283, 1121, 303)
    for x, width in ((884, 153), (1045, 153)):
        f.box(x, 356, width, 32, '#BBD8EF', '#FAFDFF')
    icon(f, 'shield', 888, 358, BLUE, .6)
    f.label(916, 376, 'Legal-set constraint', 12.2, BLUE)
    icon(f, 'gear', 1048, 358, BLUE, .6)
    f.label(1077, 368, 'Planner failure →', 11.5)
    f.label(1077, 382, 'deterministic fallback', 11.5)
    dispatch = [(1248, 145, 'Evidence acquisition', BLUE),
                (1399, 147, 'Evidence adjudication', '#087F88'),
                (1552, 92, 'Termination', PURPLE)]
    for x, width, title, color in dispatch:
        f.box(x, 194, width, 160, '#9AD7E9' if color != PURPLE else '#CFACF1', '#FAFDFF' if color != PURPLE else '#FBF7FF')
        f.label(x+9, 214, title, 12.6, color, True)
        if color != PURPLE:
            f.label(x+9, 230, '(retrieval tools)' if color == BLUE else '(verification tools)', 12.5, color)
        f.box(x+5, 241, width-10, 105, '#C9DCEB' if color != PURPLE else '#D7BFF5')
    icon(f, 'search', 1257, 251, BLUE, .7)
    f.label(1292, 268, 'Normal search', 12)
    icon(f, 'network', 1257, 289, BLUE, .7)
    f.label(1292, 299, 'Contrastive search', 11.5)
    f.label(1292, 328, 'Citation expansion', 11.5)
    icon(f, 'network', 1408, 259, '#087F88', .85)
    f.label(1448, 273, 'Verify relation', 12.5, '#087F88')
    f.label(1448, 303, 'Stability check', 12.5, '#087F88')
    f.doc(1567, 271, PURPLE, .8)
    f.label(1591, 290, 'Finalize', 12.5, PURPLE)
    f.label(1409, 330, 'Enabled; not yet done', 11, '#087F88')
    # The actual tool-return lane below closes the loop at S1; no S4→S1 shortcut.
    f.label(535, 421, 'Tool results → Evidence-state update → Next decision round', 18, '#C92B0A', True)


def port(f: Figure, x: float, y: float, color: str) -> None:
    f.add(el('circle', cx=x, cy=y, r=2.5, fill='white', stroke=color, stroke_width=1.3))


def interface(f: Figure) -> None:
    """Action-specific routes; curved crossings do not denote junctions."""
    # Acquisition: normal search → 02; contrastive / expansion → 04.
    f.path('M1320,354 V448 H514 V482', stroke=ORANGE, width=3.6)
    f.arrow(514, 475, 514, 492, ORANGE, 3.6, head_length=17)
    f.line(1172, 448, 1172, 482, ORANGE, 3.6)
    f.arrow(1172, 475, 1172, 492, ORANGE, 3.6, head_length=17)
    f.add(el('circle', cx=1172, cy=448, r=2.5, fill=ORANGE))
    # Adjudication: verify_relation / stability_check → 03 only.
    # Hump over the acquisition source at x=1320; hump over its rail at y=448.
    f.path('M1326,440 Q1320,430 1314,440', stroke='#FFF0E1', width=8)
    f.path('M853,443 Q863,448 853,453', stroke='white', width=8)
    f.path('M1473,354 V440 H1326 Q1320,430 1314,440 H853 V443 '
           'Q863,448 853,453 V482', stroke=ORANGE, width=3.6)
    f.arrow(853, 475, 853, 492, ORANGE, 3.6, head_length=17)
    # Termination has no feedback; it proceeds to post-loop coverage and judgment.
    f.line(1598, 354, 1598, 482, ORANGE, 3.6)
    f.arrow(1598, 475, 1598, 492, ORANGE, 3.6, head_length=17)
    for x in (1320, 1473, 1598):
        port(f, x, 354, ORANGE)
    # Tool results return from the actual top borders of 02, 03, 04.
    # Continuous bridged rail: no breaks that could look like dangling segments.
    for x in (514, 853, 1172):
        f.path(f'M{x+6},460 Q{x},470 {x-6},460', stroke='white', width=7)
    f.path('M1250,460 H1178 Q1172,470 1166,460 H859 Q853,470 847,460 '
           'H520 Q514,470 508,460 H177 V408',
           stroke=TEAL, width=2.6, dash='6 3')
    f.arrow(177, 416, 177, 398, TEAL, 2.6, True, head_length=17)
    for x in (610, 965, 1250):
        f.arrow(x, 492, x, 460, TEAL, 2.6, True, head_length=12)
        port(f, x, 492, TEAL)
        f.add(el('circle', cx=x, cy=460, r=2.2, fill=TEAL))
    f.label(33, 480, 'Tool results → S1', 13.5, TEAL)
    f.label(394, 480, 'Normal search', 13, '#C92B0A', True)
    f.label(617, 480, 'Retrieved works', 12.5, TEAL)
    f.label(866, 476, 'Verified relations /', 11.7, TEAL)
    f.label(866, 488, 'stability outcomes', 11.7, TEAL)
    f.label(1015, 476, 'Contrastive search /', 11.7, '#C92B0A', True)
    f.label(1015, 488, 'citation expansion', 11.7, '#C92B0A', True)
    f.label(1190, 480, 'New works', 12, TEAL)
    f.label(1483, 480, 'Finalize', 13.5, '#C92B0A', True)


def append_scaled(f: Figure, child: Figure, x: float, width: float,
                  new_width: float) -> None:
    scale = new_width / width
    child.root.set('transform', f'translate({x*(1-scale)} 0) scale({scale} 1)')
    f.add(child.root)
    for box in child.text_boxes:
        f.text_boxes.append({**box, 'x': x+(box['x']-x)*scale, 'width': box['width']*scale})


def steps(f: Figure) -> None:
    specs = [(16, 342, BLUE, 'Claim grounding', 'Extract and verify claims from manuscript', grounding),
             (370, 288, ORANGE, 'Prior-art retrieval', 'Retrieve candidate prior works per claim', retrieval),
             (670, 326, GREEN, 'Paired verification', 'Compare claim and prior work across facets', verification),
             (1008, 296, BLUE, 'Adaptive evidence acquisition', 'Acquire additional evidence when needed', adaptive),
             (1316, 348, PURPLE, 'Evidence judgment & finalization', 'Aggregate verified evidence for claim assessment', judgment)]
    roles = ['Claim Miner  |  Consolidator  |  Internal Verifier',
             'Query Planner  |  PriorArtService  |  Ranker',
             'Relation Classifier  |  Independent Verifier',
             'Query Planner  |  Retrieval  |  Relation reuse',
             'Coverage evaluator  |  GearClaimCard  |  Assessment']
    for i, ((x, w, color, title, sub, draw), role) in enumerate(zip(specs, roles)):
        panel = Figure(W, H, f'step-{i+1}')
        panel.box(x, 492, w, 329, color, '#FCFEFF')
        panel.rect(x+9, 500, 47, 39, color, radius=5)
        panel.label(x+17, 530, f'{i+1:02}', 29, 'white', True)
        size = min(22, (w-75)/measure(title, 1, True))
        panel.label(x+65, 516, title, size, color, True)
        panel.paragraph(x+65, 533, sub, w-75, 12, 13)
        draw(panel)
        panel.rect(x, 826, w, 22, '#F0F5FA', color, sw=.4)
        panel.text(x+w/2, 841, role, 11.5, color, anchor='middle')
        append_scaled(f, panel, x, w, w-10 if i < 4 else w)
        if i < 4:
            block_arrow(f, x+w-8, 650, 18)


def grounding(f: Figure) -> None:
    f.box(25, 548, 150, 190, '#A7CDE9')
    f.label(34, 565, 'Input: manuscript', 15, BLUE, True)
    f.box(35, 574, 130, 152, '#CBD7E5')
    for i in range(20):
        y = 581+i*6.8
        f.rect(42, y, 75+(i*17 % 40), 3, '#CED9E5')
    for y, color, tag in ((611, BLUE, 'S₁'), (652, ORANGE, 'S₂'), (694, GREEN, 'S₃')):
        f.rect(42, y, 92, 6, color, opacity=.55)
        f.rect(136, y-3, 24, 15, color, radius=3)
        f.label(140, y+8, tag, 12, 'white', True)
    for i, (title, text) in enumerate((('Mine', 'Find claim candidates from source spans'),
                                     ('Consolidate', 'Merge and deduplicate claims'),
                                     ('Verify', 'Check support within manuscript'))):
        y = 548+i*65
        f.box(184, y, 164, 54, '#A7CDE9', '#F5FAFF')
        if i == 0:
            f.search(203, y+23)
        elif i == 1:
            f.network(204, y+28, r=13)
        else:
            f.shield(204, y+13)
        f.label(225, y+18, title, 16, BLUE, True)
        f.paragraph(225, y+34, text, 116, 13, 13)
        if i < 2:
            f.arrow(266, y+55, 266, y+64)
    f.box(25, 744, 150, 67, '#A7CDE9')
    f.label(34, 761, 'Grounded claims Cᵢ', 15, BLUE, True)
    for i in range(3):
        f.doc(41+i*43, 772, scale=.75)
        f.label(42+i*43, 807, f'C{"₁₂₃"[i]}', 13, BLUE)
    f.box(188, 749, 160, 60, '#ECAABC', '#FFF2F5')
    f.label(197, 771, 'Internally unsupported', 14, RED, True)
    f.label(197, 790, 'Skip external search', 14, RED)
    f.arrow(176, 778, 188, 778, RED)
    f.line(349, 778, 366, 778, RED, 1.1, '4 3')
    f.arrow(366, 778, 366, 853, RED, 1.1, True)


def retrieval(f: Figure) -> None:
    for y, h, title in ((548, 68, 'Query planning'), (629, 64, 'Retrieve'), (704, 107, 'Rank & filter')):
        f.box(380, y, 268, h, '#EFC2A8', '#FFFCF8')
        f.label(427, y+19, title, 17, '#D52A0B', True)
    f.doc(391, 557, ORANGE, .9)
    f.label(427, 584, 'Queries from claim and key concepts', 13)
    for i, txt in enumerate(('Q₁', 'Q₂ ···', 'Q₃ ···')):
        f.box(409+i*76, 592, 61, 22, '#B6D2EF', '#F3F9FF')
        f.label(426+i*76, 608, txt, 15, BLUE)
    icon(f, 'database', 385, 635, ORANGE, .85)
    f.label(427, 666, 'Search prior-art sources', 13)
    for i in range(5):
        f.doc(427+i*41, 673, scale=.55)
    icon(f, 'rank', 389, 709, ORANGE, .55)
    for i, width in enumerate((133, 105, 78, 54)):
        y = 740+i*17
        f.label(394, y+3, f'W{"₁₂₃₄"[i]}', 14, BLUE, True)
        f.rect(428, y-7, 174, 9, '#ECF0F4')
        f.rect(428, y-7, width, 9, '#8AC3FA')
    f.label(427, 805, 'Cutoff · identity exclusion · candidate cap', 12)
    f.arrow(514, 617, 514, 628, ORANGE)
    f.arrow(514, 694, 514, 703, ORANGE)


def verification(f: Figure) -> None:
    f.box(680, 548, 306, 65, '#A4DAD0', '#F9FFFC')
    f.label(690, 564, 'Target claim Cᵢ', 16, BLUE, True)
    f.label(871, 564, 'Prior work Wⱼ', 16, BLUE, True)
    f.label(690, 579, '(manuscript)', 13)
    f.label(872, 579, '(retrieval)', 13)
    f.doc(726, 583, scale=.8)
    f.doc(916, 583, scale=.8)
    f.arrow(801, 585, 864, 585)
    f.arrow(864, 585, 801, 585)
    f.label(810, 604, 'Compare', 13)
    f.box(680, 619, 306, 92, '#A4DAD0')
    f.label(690, 634, 'Facet-level comparison', 15, BLUE, True)
    for i, (name, status) in enumerate((('Object (entities / materials)', 'match'),
                                      ('Method / mechanism', 'different'),
                                      ('Conditions / scope', 'match'),
                                      ('Outcome / contribution', 'unresolved'))):
        y = 649+i*17
        f.label(690, y, name, 13, INK, True)
        f.mark(882, y-4, status)
        f.label(897, y, status, 13, GREEN if status == 'match' else GRAY)
    f.box(680, 717, 306, 58, '#EFC2A8', '#FFF5EC')
    icon(f, 'link', 687, 725, ORANGE, .8)
    f.label(728, 734, 'RelationCard (Rᵢⱼ)', 16, '#D52A0B', True)
    f.label(728, 752, 'Relation · common / difference facets', 12.5)
    f.label(728, 768, 'Evidence status · source-linked rationale', 12.5)
    f.shield(694, 781)
    f.label(715, 790, 'Independent verification', 14, BLUE, True)
    f.label(715, 806, 'For direct-antecedent candidates', 13)


def adaptive(f: Figure) -> None:
    f.box(1018, 548, 276, 95, '#A7CDE9', '#F8FCFF')
    f.search(1045, 570)
    f.label(1070, 567, 'Contrastive search', 17, BLUE, True)
    f.paragraph(1070, 585, 'Counter and complementary evidence under alternative queries', 211, 13, 14)
    for i in range(3):
        f.doc(1065+i*70, 611, scale=.7)
    f.box(1018, 651, 276, 101, '#A7CDE9', '#F8FCFF')
    f.label(1070, 671, 'Citation expansion', 17, BLUE, True)
    f.label(1070, 689, 'Optional citation-neighbor follow-up', 13)
    f.network(1051, 713, r=23)
    f.doc(1044, 702, scale=.65)
    f.arrow(1084, 718, 1114, 718)
    for i in range(3):
        f.doc(1126+i*55, 704, scale=.8)
    f.box(1018, 760, 276, 51, '#A7CDE9')
    f.label(1030, 778, 'New works → Relation re-verification', 15, BLUE, True)
    f.label(1030, 799, 'Return to Step 03 through the supervisor', 13)


def judgment(f: Figure) -> None:
    f.box(1326, 548, 328, 54, '#CAB2ED', '#FCF9FF')
    for x, width in ((1333, 79), (1427, 100), (1542, 104)):
        f.box(x, 555, width, 30, '#D7C2F3', '#FFFFFF')
    f.doc(1339, 559, PURPLE, .65)
    f.label(1358, 575, 'Finalize', 13, PURPLE)
    icon(f, 'shield', 1429, 557, PURPLE, .65)
    f.label(1456, 575, 'Coverage audit', 11.5, PURPLE)
    f.doc(1548, 559, PURPLE, .65)
    f.label(1566, 575, 'GearClaimCard', 11.5, PURPLE, True)
    f.arrow(1415, 570, 1424, 570, PURPLE)
    f.arrow(1530, 570, 1539, 570, PURPLE)
    f.label(1337, 597, 'Coverage is evaluated after the action loop', 12.5)
    f.box(1326, 610, 328, 89, '#CAB2ED')
    f.label(1337, 628, 'Coverage audit', 16, PURPLE, True)
    for i, txt in enumerate(('Query-role breadth', 'Contrastive completed', 'Candidate base',
                             'Compared works', 'Dual ranking', 'Service health')):
        x, y = 1343+(i//3)*158, 646+(i%3)*19
        f.mark(x, y-4, 'match')
        f.label(x+12, y, txt, 13)
    f.label(1335, 718, 'Coverage constrains no-antecedent inference', 15, PURPLE, True)
    for x, width, color, title, detail in (
            (1326, 158, GREEN, 'Adequate evidence', 'Bounded interpretation'),
            (1492, 162, RED, 'Limited / insufficient', 'Preserve limitations')):
        f.box(x, 733, width, 53, color)
        f.add(el('circle', cx=x+18, cy=754, r=12, fill=color))
        if color == GREEN:
            f.path(f'M{x+11},754 l5,5 l10,-12', stroke='white', width=2)
        else:
            f.path(f'M{x+12},748 l12,12 M{x+24},748 l-12,12', stroke='white', width=2)
        f.label(x+35, 748, title, 12.5, color, True)
        f.label(x+35, 761, 'coverage', 13, color, True)
        f.label(x+10, 780, detail, 12)
    f.line(1643, 700, 1643, 723, PURPLE, 1)
    f.line(1573, 723, 1643, 723, PURPLE, 1)
    f.line(1404, 723, 1573, 723, PURPLE, 1)
    f.arrow(1404, 723, 1404, 731, PURPLE)
    f.arrow(1573, 723, 1573, 731, PURPLE)
    f.label(1335, 798, 'Direct / partial antecedents follow verified relations;', 13)
    f.label(1335, 814, 'limited coverage is not evidence of no innovation.', 13, RED)


def ledger(f: Figure) -> None:
    f.box(16, 855, 1648, 94, '#A98AE0', '#FCFAFF')
    f.label(28, 875, 'Evidence ledger & traceable output', 20, INK, True)
    f.label(373, 875, 'From source spans to final assessment, with traceable provenance', 14, PURPLE)
    nodes = [('Source spans Sᵢ', 'Manuscript source spans', BLUE, 'doc'),
             ('Retrieved works Wⱼ', 'Candidate prior works', BLUE, 'doc'),
             ('Relation evidence Rᵢⱼ', 'Type and rationale', TEAL, 'network'),
             ('Coverage', 'Retrieval evidence limits', TEAL, 'database'),
             ('GearClaimCard', 'Claim evidence status', ORANGE, 'doc'),
             ('Assessment', 'Findings and scope', PURPLE, 'shield'),
             ('Fusion handoff', 'With independent Graph', PURPLE, 'network')]
    for i, (title, subtitle, color, kind) in enumerate(nodes):
        x = 27+i*183
        f.box(x, 884, 164, 47, '#D3DEF4')
        icon(f, kind, x+3, 890, color, .8)
        f.label(x+36, 900, title, 12.5, color, True)
        f.paragraph(x+36, 915, subtitle, 124, 11.5, 12)
        if i < 6:
            f.arrow(x+168, 908, x+179, 908, BLUE)
    # Low-level source linkage, drawn below the cards so it never crosses text.
    f.line(109, 940, 1207, 940, '#A7A5D2', 1, '4 3')
    for i in range(7):
        x = 109+i*183
        f.line(x, 932, x, 940, '#A7A5D2', 1, '4 3')
        f.add(el('circle', cx=x, cy=940, r=3.5, fill=nodes[i][2]))
    f.box(1315, 861, 340, 82, '#CEB5F2', '#FCFAFF')
    f.label(1325, 876, 'Final evidence outcome states', 13, PURPLE, True)
    f.label(1495, 876, '(categories, not a ranking)', 9.5, GRAY)
    states = [(1324, 883, 102, 31, 'internally\nunsupported', RED, '#FDEAF0'),
              (1432, 883, 102, 31, 'antecedent\nfound', '#00835F', '#E8F7F1'),
              (1540, 883, 106, 31, 'residual\nextension', '#D54B16', '#FFF0E2'),
              (1324, 919, 173, 19, 'bounded no antecedent', BLUE, '#E7F1FF'),
              (1503, 919, 143, 19, 'inconclusive', PURPLE, '#F1E8FF')]
    for x, y, width, height, value, color, fill in states:
        f.box(x, y, width, height, color, fill)
        lines = value.split('\n')
        for j, line in enumerate(lines):
            f.text(x+width/2, y+(12 if len(lines) > 1 else 13)+j*12,
                   line, 11.5, color, True, anchor='middle')


def main() -> None:
    final = OUT / 'final'
    final.mkdir(parents=True, exist_ok=True)
    fonts(OUT)
    f = Figure(W, H, 'fig9-evidence-supervised-agent')
    f.rect(0, 0, W, H, 'white')
    title = 'Fig. 9 | Evidence-supervised agent architecture for bounded and traceable scientific claim verification'
    f.label(28, 34, title, 29, INK, True)
    f.label(28, 61, 'A claim-level workflow for prior-art retrieval, paired verification, conditional follow-up, and traceable evidence judgment', 23, '#24477B')
    supervisor(f)
    steps(f)
    ledger(f)
    interface(f)
    root = el('svg', width='420mm', height='240mm', viewBox=f'0 0 {W} {H}')
    root.append(f.root)
    svg = final / 'Fig9.svg'
    ET.ElementTree(root).write(svg, encoding='utf-8', xml_declaration=True)
    outline(svg, final / 'Fig9_outlined.svg')
    convert(svg, scale=2)
    import cairosvg
    cairosvg.svg2png(url=str(svg), write_to=str(final/'Fig9_600dpi.png'),
                    output_width=round(420/25.4*600), output_height=round(240/25.4*600))
    outside = [b for b in f.text_boxes if b['x'] < 0 or b['x']+b['width'] > W or b['y']+b['height'] > H]
    overlaps = []
    for i, left in enumerate(f.text_boxes):
        for right in f.text_boxes[i+1:]:
            dx = min(left['x']+left['width'], right['x']+right['width'])-max(left['x'], right['x'])
            dy = min(left['y']+left['height'], right['y']+right['height'])-max(left['y'], right['y'])
            if dx > 1 and dy > 1:
                overlaps.append([left['text'], right['text'], round(dx, 2), round(dy, 2)])
    sources = ['gear/evidence_supervisor.py', 'gear/prior_art.py', 'gear/innovation/shared.py',
               'gear/innovation/analysis.py', 'gear/innovation/pipeline.py']
    qa = {'canvas_mm': [420, 240], 'text_elements': len(f.text_boxes),
          'outside_canvas': outside, 'text_overlaps': overlaps,
          'interface_connections': {
              'normal_search': 'S4 acquisition → Step 02',
              'contrastive_search / citation_expand': 'S4 acquisition → Step 04',
              'verify_relation / stability_check': 'S4 adjudication → Step 03',
              'finalize': 'S4 termination → Step 05; no feedback',
              'retrieved_works': 'Step 02 → S1',
              'relations / stability_outcomes': 'Step 03 → S1',
              'new_works': 'Step 04 → S1',
          },
          'interface_geometry': {'source_ports_y': 354, 'step_border_y': 492,
                                 'feedback_s1_border': [177, 398],
                                 'crossings': 'curved bridges with background knockout; no junction'},
          'source_sha256': {
              p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}}
    (OUT/'qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(final), 'outside_canvas': outside, 'text_overlaps': overlaps}))


if __name__ == '__main__':
    main()
