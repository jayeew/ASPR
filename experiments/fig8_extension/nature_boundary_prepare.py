"""Collect targeted primary evidence without changing earlier reviews."""
from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lxml import html

from .nature_boundary_sources import OUT, PREV, capture, publisher
from .stage0 import write_json

URLS = [
    'https://weill.cornell.edu/our-story/history',
    'https://med.stanford.edu/about.html',
    'https://med.stanford.edu/about/principles.html',
    'https://www.bjmu.edu.cn/ylfw/zsfsyy/index.htm',
    'https://www.bjmu.edu.cn/ylfw/gjfsyy/index.htm',
    'https://news.pku.edu.cn/info/7921/3049761.htm',
    'https://www.biomembrane.tsinghua.edu.cn/p/shiyanshijianjie.html',
    'https://identityguide.hms.harvard.edu/page-5',
    'https://uthscsa.edu/news/utsa-and-ut-health-san-antonio-complete-merger-become-university-texas-san-antonio',
    'https://www.mskcc.org/research/ski',
    'https://www.volastratx.com/',
    'https://wordpress-hk.stonewise.cn/protocal',
    'https://teach.weill.cornell.edu/mdphd',
    'https://alleninstitute.org/news/the-paul-g-allen-frontiers-group-announces-allen-discovery-center-for-neuroimmune-interactions-at-icahn-school-of-medicine-at-mount-sinai',
]


def article_extract(doi: str) -> dict[str, Any]:
    pub = publisher(doi)
    raw = Path(pub['source_receipt']).with_suffix('.gz')
    tree = html.fromstring(gzip.decompress(raw.read_bytes()))
    nodes = tree.xpath('//*[self::h2 or self::h3][contains(text(),"Competing interests")]/following-sibling::p[1]')
    return {'doi': doi, 'source_url': pub['source_url'], 'receipt': pub['source_receipt'],
            'affiliations': pub['affiliations'], 'disclosure': '\n'.join(x.text_content() for x in nodes)}


def main() -> None:
    with ThreadPoolExecutor(max_workers=5) as pool:
        receipts = list(pool.map(capture, URLS))
    write_json(OUT / 'primary_source_receipts.json', receipts)
    pattern = r'Lynxi|Synfuels|Trip\.com|Glbizzia|Shuimu|Vidrovr|GRIN Therapeutics|VantAI|Fauna Robotics|DP Technology|Xscape|Tsientang|Nanhu Laboratory|Advanced Algorithms|New Cornerstone|Nasal Disease|Aero.Engine|Multiscale Spin|Southwest United'
    units = json.loads((PREV / 'remaining_weighted_units.json').read_text())
    dois = sorted({d for u in units if re.search(pattern, u['address'], re.IGNORECASE) for d in u['dois']})
    with ThreadPoolExecutor(max_workers=4) as pool:
        evidence = list(pool.map(article_extract, dois))
    write_json(OUT / 'targeted_publisher_evidence.json', evidence)
    print(f'{len(receipts)} primary sources and {len(evidence)} publisher pages retained')


if __name__ == '__main__':
    main()
