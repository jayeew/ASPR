"""Apply explicit terminal ISO country names after postal tokens in cached signatures."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from . import terminal_country_apply as application
from .nature_backfill_resolve import clean_address
from .postal_country_cache import ISO
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/iso_postal_country_cache'


def country_patterns() -> list[tuple[re.Pattern, str]]:
    names = {}
    for entry in json.loads(ISO.read_text())['3166-1']:
        for field in ('name', 'official_name', 'common_name'):
            name = entry.get(field, '').casefold().replace('’', "'")
            if name and ',' not in name and name not in {'georgia', 'jersey', 'congo'}:
                names[name] = entry['alpha_2']
    return [(re.compile(r"(?<![\w])(?:\d{4,6}|\d{3}-\d{4}|\d{5}-\d{4})\s+" + re.escape(name) + r'$'), code)
            for name, code in sorted(names.items(), key=lambda pair: -len(pair[0]))]


def explicit_code(address: str, patterns: list[tuple[re.Pattern, str]]) -> str | None:
    text = clean_address(address).casefold().replace('’', "'")
    if text.count(',') < 2 or ';' in text:
        return None
    tail = text.rsplit(',', 1)[-1].strip().rstrip('.')
    codes = {code for pattern, code in patterns if pattern.search(tail)}
    if len(codes) != 1:
        return None
    code = codes.pop()
    if code == 'CN' and any(name in text for name in ('hong kong', 'macau', 'macao', 'taiwan')):
        return None
    return code


def prepare() -> dict:
    patterns = country_patterns()
    db = connect(); selected = []
    for tid, payload in db.execute("SELECT task_id,payload FROM tasks WHERE kind='institution' AND active=1"):
        item = json.loads(payload)
        if item.get('missing_country') and explicit_code(item.get('name', ''), patterns):
            selected.append(tid)
    grouped = defaultdict(set)
    for tid in selected:
        for wid, shard in db.execute("SELECT p.work_id,p.shard FROM task_papers l JOIN papers p USING(work_id) WHERE l.task_id=? AND p.status='included_verified_research'", (tid,)):
            grouped[shard].add(wid)
    db.close()
    papers, mapping, counts = [], {}, Counter()
    for shard, ids in grouped.items():
        rows = pq.read_table(ROOT / 'fields/papers' / shard, columns=['work_id','doi','authors_json'], filters=[('work_id','in',sorted(ids))]).to_pylist()
        for paper in rows:
            slots = []
            for author in json.loads(paper['authors_json']):
                if author.get('nature_v1_slots') is not None or not author.get('publisher_affiliation_source'):
                    continue
                for slot in author.get('publisher_affiliation_slots') or []:
                    if slot.get('country_codes'):
                        continue
                    address = clean_address(slot.get('text', ''));code = explicit_code(address, patterns)
                    if code:
                        slots.append({'address':address,'code':code,'receipt':author['publisher_affiliation_source']})
                        mapping[address.rsplit(',',1)[-1].strip().rstrip('.').casefold()] = code
                        counts[code] += 1
            if slots:
                papers.append({'work_id':paper['work_id'],'doi':paper['doi'],'shard':shard,'slots':slots})
    return {'papers':papers,'mapping':mapping,'country_slots':dict(counts),
        'iso_receipt':str(ISO.relative_to(ROOT)),'iso_sha256':hashlib.sha256(ISO.read_bytes()).hexdigest(),
        'scope':'Current qualified institution tasks; explicit terminal full ISO country names after numeric postal tokens; no postcode-derived country. Native identity and author/address links required. Ambiguous Georgia/Jersey/Congo names and CN signatures naming Hong Kong/Macao/Taiwan withheld.',
        'network_requests':0,'final_task_closure':False}


def run() -> None:
    FOLDER.mkdir(exist_ok=True);scope=FOLDER/'scope.json'
    if not scope.exists():
        result=prepare();atomic_json(scope,result)
        print(json.dumps({'selected_papers':len(result['papers']),'country_slots':result['country_slots'],'network_requests':0}),flush=True)
    if not (FOLDER/'application.json').exists():
        application.FOLDER=FOLDER;application.RULE=str(scope.relative_to(ROOT));application.run()


if __name__ == '__main__':
    run()
