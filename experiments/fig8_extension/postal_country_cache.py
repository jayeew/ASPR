"""Recover an explicit ISO country name joined to its postal address segment."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

import pyarrow.parquet as pq

from . import terminal_country_apply as application
from .nature_backfill_resolve import clean_address
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/postal_country_cache'
ISO = ROOT / 'supplement/cureus_country_codes/iso_3166-1.json'


def explicit_country(address: str) -> str | None:
    text = clean_address(address)
    if text.count(',') < 2 or ';' in text:
        return None
    tail = text.rsplit(',', 1)[-1].strip().rstrip('.').casefold()
    if re.search(r"(?<!\d)\d{6}\s+people['’]s republic of china$", tail):
        return 'CN'
    return None


def prepare() -> dict:
    entry = next(e for e in json.loads(ISO.read_text())['3166-1'] if e['alpha_2'] == 'CN')
    assert entry['official_name'] == "People's Republic of China"
    db = connect()
    groups = defaultdict(set)
    query = """SELECT DISTINCT p.work_id,p.shard FROM tasks t
        JOIN task_papers l ON l.task_id=t.task_id JOIN papers p USING(work_id)
        WHERE t.kind='institution' AND t.active=1 AND p.status='included_verified_research'
        AND t.payload LIKE '%republic of china%'"""
    for wid, shard in db.execute(query):
        groups[shard].add(wid)
    db.close()
    papers, mapping = [], {}
    for shard, ids in groups.items():
        rows = pq.read_table(ROOT / 'fields/papers' / shard,
            columns=['work_id', 'doi', 'authors_json'], filters=[('work_id', 'in', sorted(ids))]).to_pylist()
        for paper in rows:
            slots = []
            for author in json.loads(paper['authors_json']):
                if author.get('nature_v1_slots') is not None or not author.get('publisher_affiliation_source'):
                    continue
                for slot in author.get('publisher_affiliation_slots') or []:
                    address = clean_address(slot.get('text', ''))
                    code = explicit_country(address)
                    if not slot.get('country_codes') and code:
                        slots.append({'address': address, 'code': code, 'receipt': author['publisher_affiliation_source']})
                        mapping[address.rsplit(',', 1)[-1].strip().rstrip('.').casefold()] = code
            if slots:
                papers.append({'work_id': paper['work_id'], 'doi': paper['doi'], 'shard': shard, 'slots': slots})
    return {'papers': papers, 'mapping': mapping, 'iso_receipt': str(ISO.relative_to(ROOT)),
        'iso_sha256': hashlib.sha256(ISO.read_bytes()).hexdigest(),
        'scope': 'Current qualified institution-task papers; empty native affiliation country slots ending in a six-digit postal token followed by the full ISO official name People\'s Republic of China. Curly apostrophe is a typography alias only. No country inferred from postal code, headquarters or institution parent. Native author/address and exact paper identity rechecked before writing.',
        'network_requests': 0, 'final_task_closure': False}


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    scope = FOLDER / 'scope.json'
    if not scope.exists():
        result = prepare()
        atomic_json(scope, result)
        print(json.dumps({'selected_papers': len(result['papers']),
            'selected_author_slots': sum(len(p['slots']) for p in result['papers']), 'network_requests': 0}), flush=True)
    if (FOLDER / 'application.json').exists():
        return
    application.FOLDER = FOLDER
    application.RULE = str(scope.relative_to(ROOT))
    application.run()


if __name__ == '__main__':
    run()
