"""English display text and exact mappings to saved report sentences."""
from __future__ import annotations

from typing import Any

COLORS = ['#1269DF', '#178F9D', '#713BCB', '#D47626', '#56739C', '#9D557D']
CLAIM_TEXT = [
    'Two SiO₂ layers seal a\nvitrified cryo-EM sample.',
    '30-µs pulse trains limit film\nbreakup over longer times.',
    'Interfaces and melting help\nparticles detach and rotate.',
    'Near-atomic resolution with\nnearly isotropic sampling.',
    'Hotter particles: wider L1 motion\nat 300 µs, but not 30 µs.',
    'Slow motion suggests added\ninteractions with nearby rRNA.'
]
ANCHORS = ['Method · M01 · p. 2', 'Method · M02 · p. 2',
           'Mechanism · M03 · p. 3', 'Finding · M04 · pp. 2–4',
           'Finding · M05 · p. 3', 'Mechanism · M06 · p. 5']
REPORT_ROWS: list[dict[str, Any]] = [
    {'id': 'R01', 'title': 'The specific increment',
     'source': '这项工作的主要贡献，是把样品封装、激光时序控制、近原子分辨率成像和时间分辨结构分析组合成一种超薄液体池cryo-EM方案。',
     'english': 'The work combines sample sealing, laser timing, near-atomic imaging and time-resolved structural analysis into an ultrathin liquid-cell cryo-EM approach.',
     'links': 'C01 + C02 + C04 + C05 · M01–M05', 'question': 'What is added?',
     'takeaway': 'A combined experimental capability.', 'paragraph': 1},
    {'id': 'R02', 'title': 'Extend an existing route',
     'source': '第二项贡献是把有效观察窗口推进到数百微秒。',
     'english': 'The second contribution extends the effective observation window to hundreds of microseconds.',
     'links': 'C02 · M02 · E02', 'question': 'How far does it go?',
     'takeaway': 'Longer windows; intact specimens.', 'paragraph': 3},
    {'id': 'R03', 'title': 'An empirical biological result',
     'source': '本文增加的是L1 stalk温度—延迟响应的经验性刻画。',
     'english': 'The added contribution is an empirical characterization of the L1 stalk’s temperature–delay response.',
     'links': 'C05 · M05 · E04 · G02*', 'question': 'What can it reveal?',
     'takeaway': 'Delayed L1-stalk motion.', 'paragraph': 5},
    {'id': 'R04', 'title': 'Keep the mechanism open',
     'source': '其绝对历史优先性、普适性和helix 68的因果作用仍未确定。',
     'english': 'Its absolute priority, general applicability and the causal role of helix 68 remain unresolved.',
     'links': 'C06 · M06 · report paragraph 6', 'question': 'What remains open?',
     'takeaway': 'Helix 68 causality.', 'paragraph': 6},
]
SYNTHESIS_SOURCE = '它主要是既有微秒cryo-EM、超薄膜成像和优选取向控制的延续、扩展与组合'
SYNTHESIS = 'Primarily a continuation, extension and combination of microsecond cryo-EM, ultrathin-membrane imaging and orientation control.'
