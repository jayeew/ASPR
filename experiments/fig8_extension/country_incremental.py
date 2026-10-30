"""Apply existing literal-country rules only to new, unreviewed signatures."""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections import defaultdict

import pyarrow.parquet as pq

from . import terminal_country_apply as application
from .country_change_validation import run as validate
from .iso_postal_country_cache import country_patterns, explicit_code
from .nature_backfill_resolve import clean_address
from .postal_country_cache import ISO
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/country_incremental'


def known_signatures() -> set[tuple[str, str, str]]:
    known = set()
    bases = ['explicit_terminal_country', 'iso_terminal_country', 'postal_country_cache',
             'iso_postal_country_cache', 'country_incremental']
    for base in bases:
        for path in (ROOT / 'supplement' / base).rglob('scope.json'):
            if not path.with_name('application.json').exists():
                continue
            for paper in json.loads(path.read_text()).get('papers', []):
                known.update((paper['work_id'], clean_address(s['address']), s['receipt'])
                             for s in paper['slots'])
    return known


def identity_changed(before: str, after: str) -> bool:
    """Reopen an address only when its author-identity context has actually changed."""
    def signature(value: str) -> list[tuple]:
        authors = json.loads(value)
        if isinstance(authors, str):
            authors = json.loads(authors)
        fields = ('position', 'name', 'author_id', 'corroborated_orcid',
                  'publisher_author_identity_match', 'publisher_affiliation_source')
        return [tuple(author.get(field) for field in fields) for author in authors]
    return signature(before) != signature(after)


def prepare(low: int, high: int) -> dict:
    db = connect(); grouped = defaultdict(set); reopened = set()
    query = """SELECT p.work_id,p.shard,c.before_value,c.after_value FROM changes c JOIN papers p USING(work_id)
        WHERE c.rowid>? AND c.rowid<=? AND c.field='authors_json'
        AND p.status='included_verified_research'"""
    for wid, shard, before, after in db.execute(query, (low, high)):
        grouped[shard].add(wid)
        if identity_changed(before, after):
            reopened.add(wid)
    db.close()
    known = known_signatures(); patterns = country_patterns()
    papers, mapping = [], {}
    for shard, ids in grouped.items():
        rows = pq.read_table(ROOT / 'fields/papers' / shard,
                            columns=['work_id', 'doi', 'authors_json'],
                            filters=[('work_id', 'in', sorted(ids))]).to_pylist()
        for row in rows:
            slots = []
            for author in json.loads(row['authors_json']):
                receipt = author.get('publisher_affiliation_source')
                if not receipt or author.get('nature_v1_slots') is not None:
                    continue
                for slot in author.get('publisher_affiliation_slots') or []:
                    address = clean_address(slot.get('text', ''))
                    if slot.get('country_codes') or ((row['work_id'], address, receipt) in known
                            and row['work_id'] not in reopened):
                        continue
                    code = explicit_code(address, patterns)
                    if code:
                        slots.append({'address': address, 'code': code, 'receipt': receipt})
                        mapping[address.rsplit(',', 1)[-1].strip().rstrip('.').casefold()] = code
            if slots:
                papers.append({'work_id': row['work_id'], 'doi': row['doi'], 'shard': shard, 'slots': slots})
    return {'papers': papers, 'mapping': mapping, 'changes_low': low, 'changes_high': high,
            'changed_qualified_papers_examined': sum(map(len, grouped.values())),
            'identity_context_reopened_work_ids': sorted(reopened),
            'iso_receipt': str(ISO.relative_to(ROOT)),
            'iso_sha256': hashlib.sha256(ISO.read_bytes()).hexdigest(),
            'rule': 'Existing ISO postal-country rule; native identity and exact author/address rechecked; previously reviewed signatures skipped unless author identity/roster evidence changed',
            'network_requests': 0, 'final_task_closure': False}


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'incremental.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        state_path = FOLDER / 'state.json'
        db = connect()
        if state_path.exists():
            state = json.loads(state_path.read_text())
            low = state['completed_through_change_rowid']
            high = state.get('pending_high') or db.execute('SELECT max(rowid) FROM changes').fetchone()[0]
        else:
            low = db.execute("SELECT min(rowid) FROM changes WHERE receipt=?",
                             ('supplement/iso_postal_country_cache/scope.json',)).fetchone()[0]
            if low is None:
                raise ValueError('Existing country repair baseline is required')
            high = db.execute('SELECT max(rowid) FROM changes').fetchone()[0]
        db.close()
        atomic_json(state_path, {'completed_through_change_rowid': low, 'pending_high': high})
        folder = FOLDER / f'{low}-{high}'; folder.mkdir(exist_ok=True)
        scope_path = folder / 'scope.json'
        if not scope_path.exists():
            atomic_json(scope_path, prepare(low, high))
        scope = json.loads(scope_path.read_text())
        print(json.dumps({'selected_papers': len(scope['papers']),
                          'slots': sum(len(p['slots']) for p in scope['papers'])}), flush=True)
        if scope['papers']:
            application.FOLDER = folder; application.RULE = str(scope_path.relative_to(ROOT))
            if not (folder / 'application.json').exists():
                application.run(allow_unique_accent_names=True)
            validate(folder)
        atomic_json(state_path, {'completed_through_change_rowid': high,
                                'last_scope': str(scope_path.relative_to(ROOT)), 'network_requests': 0})


if __name__ == '__main__':
    run()
