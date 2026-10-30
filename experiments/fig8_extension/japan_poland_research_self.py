"""Official standalone research identities; no graduate-program university mapping."""
from __future__ import annotations

from .academy_self_entities import run

SOURCES = [
    ('https://openalex.org/I205401836', 'National Institute for Materials Science',
     'https://www.nims.go.jp/eng/nims/profile.html',
     'Official profile names NIMS and describes its national research and development '
     'agency status from 2015 and designated institute status from 2016.'),
    ('https://openalex.org/I4210110652', 'RIKEN',
     'https://www.riken.jp/en/about/',
     'Official description identifies RIKEN as a national research and development agency '
     'and comprehensive research institute. Whole RIKEN signature only, not center aliases.'),
    ('https://openalex.org/I99542240', 'Polish Academy of Sciences',
     'https://pan.pl/en/about-us/',
     'Official academy description distinguishes the scholarly academy and network of '
     'research institutes. Academy self signature does not identify a partner university.'),
]


if __name__ == '__main__':
    run(SOURCES, 'japan_poland_research_self')
