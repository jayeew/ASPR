"""Reuse retained ISO names for explicit native terminal-country statements."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from . import terminal_country_apply as apply
from .nature_backfill_resolve import clean_address
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/iso_terminal_country'
ISO = ROOT / 'supplement/cureus_country_codes/iso_3166-1.json'


def mapping() -> dict[str, str]:
    entries = json.loads(ISO.read_text())['3166-1']
    previous = json.loads((ROOT / 'supplement/explicit_terminal_country/scope.json').read_text())['mapping']
    held = {'georgia', 'jersey', 'congo'}
    result = {}
    for entry in entries:
        for field in ('name', 'official_name', 'common_name'):
            name = entry.get(field, '').casefold()
            if name and name not in held and name not in previous and ',' not in name:
                result[name] = entry['alpha_2']
    return result


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    scope_path = FOLDER / 'scope.json'
    if not scope_path.exists():
        countries = mapping()
        db = connect()
        selected = []
        for tid, payload in db.execute("SELECT task_id,payload FROM tasks WHERE kind='institution' AND active=1"):
            item = json.loads(payload)
            address = clean_address(item.get('name', ''))
            suffix = address.rsplit(',', 1)[-1].strip().rstrip('.').casefold()
            if ',' in address and ';' not in address and suffix in countries and item.get('missing_country'):
                selected.append(tid)
        groups = defaultdict(set)
        for tid in selected:
            for wid, shard in db.execute("SELECT p.work_id,p.shard FROM task_papers t JOIN papers p USING(work_id) WHERE t.task_id=? AND p.status='included_verified_research'", (tid,)):
                groups[shard].add(wid)
        db.close()
        papers, counts = [], Counter()
        for shard, ids in groups.items():
            rows = pq.read_table(ROOT / 'fields/papers' / shard,
                columns=['work_id', 'doi', 'authors_json'], filters=[('work_id', 'in', sorted(ids))]).to_pylist()
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
                        suffix = address.rsplit(',', 1)[-1].strip().rstrip('.').casefold()
                        if (not slot.get('country_codes') and ',' in address and ';' not in address
                                and suffix in countries):
                            slots.append({'address': address, 'code': countries[suffix], 'receipt': receipt})
                            counts[countries[suffix]] += 1
                if slots:
                    papers.append({'work_id': paper['work_id'], 'doi': paper['doi'], 'shard': shard, 'slots': slots})
        atomic_json(scope_path, {'papers': papers, 'mapping': countries, 'country_slot_counts': dict(counts),
            'iso_receipt': str(ISO.relative_to(ROOT)), 'iso_sha256': hashlib.sha256(ISO.read_bytes()).hexdigest(),
            'scope': 'Current qualified institution tasks with missing countries; exact terminal ISO country names only; native DOI/title/journal/unique author-address links required at application. Existing countries and completed Nature preserved.',
            'held_ambiguous_names': ['Georgia', 'Jersey', 'Congo'],
            'existing_ten_country_rule_not_repeated': True, 'network_requests': 0})
    if (FOLDER / 'application.json').exists():
        return
    apply.FOLDER = FOLDER
    apply.RULE = str(scope_path.relative_to(ROOT))
    apply.run()


if __name__ == '__main__':
    run()
