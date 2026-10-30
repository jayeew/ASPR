"""Map explicit UK constituent-country suffixes in matched native signatures."""
from __future__ import annotations

import json
from collections import defaultdict

import pyarrow.parquet as pq

from . import terminal_country_apply as apply
from .nature_backfill_resolve import clean_address
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/uk_signed_country'
MAPPING = {'england': 'GB', 'scotland': 'GB', 'wales': 'GB', 'northern ireland': 'GB'}


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    scope_path = FOLDER / 'scope.json'
    if not scope_path.exists():
        db = connect()
        selected = set()
        for tid, payload in db.execute("""SELECT task_id,payload FROM tasks WHERE kind='institution'
            AND active=1 AND (payload LIKE '%England%' OR payload LIKE '%Scotland%'
                OR payload LIKE '%Wales%' OR payload LIKE '%Northern Ireland%')"""):
            item = json.loads(payload)
            suffix = clean_address(item.get('name', '')).rsplit(',', 1)[-1].strip().casefold()
            if suffix in MAPPING and item.get('missing_country'):
                selected.add(tid)
        groups = defaultdict(set)
        for tid in selected:
            for wid, shard in db.execute("""SELECT p.work_id,p.shard FROM task_papers t
                JOIN papers p USING(work_id) WHERE t.task_id=? AND p.status='included_verified_research'""", (tid,)):
                groups[shard].add(wid)
        db.close()
        papers = []
        for shard, ids in groups.items():
            rows = pq.read_table(ROOT / 'fields/papers' / shard,
                columns=['work_id', 'doi', 'authors_json'], filters=[('work_id', 'in', list(ids))]).to_pylist()
            for paper in rows:
                slots = []
                for author in json.loads(paper['authors_json']):
                    if author.get('nature_v1_slots') is not None:
                        continue
                    receipt = author.get('publisher_affiliation_source')
                    if not receipt:
                        continue
                    for slot in author.get('publisher_affiliation_slots') or []:
                        address = clean_address(slot.get('text', ''))
                        suffix = address.rsplit(',', 1)[-1].strip().casefold()
                        if not slot.get('country_codes') and suffix in MAPPING and ';' not in address:
                            slots.append({'address': address, 'code': MAPPING[suffix], 'receipt': receipt})
                if slots:
                    papers.append({'work_id': paper['work_id'], 'doi': paper['doi'], 'shard': shard, 'slots': slots})
        atomic_json(scope_path, {'papers': papers, 'mapping': MAPPING,
            'scope': 'Current qualified tasks with explicit terminal UK constituent-country names. Native identity and unique author-address connections must match at application; preserve Nature and existing country assignments.',
            'definition_source': 'https://www.gov.uk/hmrc-internal-manuals/employment-income-manual/eim40032',
            'definition_excerpt': 'The United Kingdom means England, Wales, Scotland and Northern Ireland.',
            'applicable_publication_years': [2023, 2024, 2025],
            'not_inferred_from': ['institution headquarters', 'author nationality', 'university parent relationship'],
            'network_requests': 0})
    if (FOLDER / 'application.json').exists():
        return
    apply.FOLDER = FOLDER
    apply.RULE = str(scope_path.relative_to(ROOT))
    apply.run()


if __name__ == '__main__':
    run()
