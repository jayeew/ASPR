"""Finite goal acquisition; reuse the shared bounded request implementation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import three_journal_round as bounded

AUDIT = bounded.ROOT / 'supplement/closure_goal'


def setup() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    path = AUDIT / 'budget.json'
    if not path.exists():
        start = datetime.fromisoformat('2026-10-23T07:11:09+00:00')
        bounded.save(path, {'start_utc': start.isoformat(), 'deadline_utc': (start+timedelta(minutes=90)).isoformat(),
                           'network_deadline_utc': (start+timedelta(minutes=60)).isoformat(),
                           'request_cap':300,'doi_cap':200,'annual_request_cap':30,'annual_requests':0,
                           'requests':0,'dois':[],'blocked_hosts':[],'status':'running'})
    bounded.AUDIT = AUDIT


def get(url: str, params: dict | None = None, dois: list[str] | None = None, annual: bool = False) -> dict:
    setup()
    state = bounded.init()
    if datetime.now(timezone.utc) >= datetime.fromisoformat(state['network_deadline_utc']):
        raise RuntimeError('60-minute network deadline')
    if annual:
        if state['annual_requests'] >= 30:
            raise RuntimeError('Annual source request cap')
        state['annual_requests'] += 1
        bounded.save(AUDIT/'budget.json',state)
    return bounded.get(url,params,dois)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['init','annual'])
    args=parser.parse_args()
    setup()
    if args.action == 'annual':
        queries=[('https://www.science.org/content/page/about-science',None),
                 ('https://www.cell.com/cell/about',None),
                 ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',{'db':'nlmcatalog','id':'0404511,0413066','retmode':'xml'})]
        for url,params in queries:
            r=get(url,params,annual=True)
            print(json.dumps(r),flush=True)


if __name__ == '__main__':
    main()


def pnas_pilot() -> None:
    import hashlib
    from collections import defaultdict
    from .three_journal_round import load
    from .supplement_crossref import SELECT
    decisions, _, _, _ = load()
    pools = defaultdict(list)
    for d in decisions.values():
        if d['source_id'] == bounded.SIDS[2] and d['eligibility_status'] == 'pending_primary_online_date':
            pools[(d['field'],d['year_openalex'])].append(d['doi'])
    for pool in pools.values():
        pool.sort(key=lambda doi:hashlib.sha256(('20261021:'+doi).encode()).hexdigest())
    targets=[]
    while any(pools.values()):
        for pool in pools.values():
            if pool:
                targets.append(pool.pop(0))
    bounded.save(AUDIT/'pnas_date_queue.json',targets)
    group=targets[:20]
    r=get('https://api.crossref.org/works',{'filter':','.join('doi:'+d for d in group),'select':SELECT,'rows':len(group)},group)
    print(json.dumps(r),flush=True)
