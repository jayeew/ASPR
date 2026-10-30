"""Candidate-bound reparse of cached Young Minds educational article pages."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq
from lxml import html

from .fig8b_qualification_build import key, match, write_decisions
from .publisher_delta_apply import evidence_index, stamp, updated_decision
from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/frontiers_young_minds'
RULE = 'supplement/frontiers_young_minds/type_rule.json'


def parse(raw: bytes, doi: str) -> dict | None:
    tree = html.fromstring(raw)
    titles = tree.xpath('//h1[contains(concat(" ",normalize-space(@class)," ")," fulltext-heading ")]')
    badges = tree.xpath('//span[contains(concat(" ",normalize-space(@class)," ")," fulltext-badge ") and contains(@class,"article-type")]')
    citation = tree.xpath('//aside[contains(@class,"fulltext-aside")]/h3[normalize-space(.)="Citation"]/following-sibling::div[1]')
    if len(titles) != 1 or len(badges) != 1 or len(citation) != 1:
        return None
    native_dois = re.findall(r'10\.3389/frym\.\d{4}\.\d+', citation[0].text_content().lower())
    if native_dois != [doi]: return None
    kind = ' '.join(badges[0].text_content().split())
    if kind not in {'Core Concept', 'New Discovery'}: return None
    dates = tree.xpath('//aside[contains(@class,"fulltext-aside")]/h3[normalize-space(.)="Publishing dates"]/following-sibling::div[1]')
    date = ''
    if len(dates) == 1:
        found = re.search(r'Published online:\s*([A-Za-z]+ \d{1,2}, \d{4})', dates[0].text_content())
        if found: date = datetime.strptime(found[1], '%B %d, %Y').date().isoformat()
    return {'family': 'publisher_frontiers_young_minds', 'title': ' '.join(titles[0].text_content().split()),
            'kind': kind, 'journal_titles': ['Frontiers for Young Minds'], 'categories': [],
            'online_date': date, 'date_precision': 'day' if date else 'missing', 'type_rule_receipt': RULE}


def run() -> dict:
    receipt = json.loads((FOLDER / 'guidelines.json').read_text())
    if receipt['http_status'] != 200: raise ValueError('Official type source unavailable')
    text = html.fromstring(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())).text_content()
    if 'already been peer-reviewed and published' not in text or 'explain fundamental ideas' not in text:
        raise ValueError('Official type definitions not found')
    atomic_json(ROOT / RULE, {'status': 'reviewed', 'official_url': receipt['url'], 'source': receipt['raw_receipt'],
        'types': {'Core Concept': 'Educational synthesis of established concepts', 'New Discovery': 'Adaptation of prior peer-reviewed research'},
        'scope': 'Existing 2023–2025 candidates; exact native citation DOI/title/journal plus explicit article badge; applies only to these two educational article types', 'classification': 'nonprimary'})
    stats = Counter(); q = ROOT / 'fields/qualification'
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect(); evidence_index(db, q)
        journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year'] for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        additions = []; selected = defaultdict(dict)
        rows = db.execute("SELECT p.work_id,p.doi,p.title,p.journal,p.source_id,p.shard,j.receipt FROM papers p JOIN publisher_remaining_jobs j USING(doi) WHERE p.doi LIKE '10.3389/frym.%' AND j.receipt<>''").fetchall()
        considered = len(rows)
        for wid, doi, title, journal, sid, shard, cached in rows:
            r = json.loads((ROOT / cached).read_text())
            if r.get('http_status') != 200 or urlsplit(r.get('final_url', '')).hostname != 'kids.frontiersin.org': continue
            item = parse(gzip.decompress((q / r['raw_receipt']).read_bytes()), doi)
            if not item: stats['unresolved_native_fields'] += 1; continue
            item.update(receipt=r['raw_receipt'], url=r['final_url'])
            e = {'work_id': wid, 'doi': doi, **match({'work_id': wid, 'title': title, 'journal_name': journal, 'source_id': sid}, item, journals[sid])}
            if e['identity'] != 'matched': stats['identity_unresolved'] += 1; continue
            prior = {}; secondary = []
            for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (wid,)):
                v = json.loads(payload)
                if kind == 'primary': prior[v['id']] = v
                else: secondary = v
            if e['id'] in prior: continue
            prior[e['id']] = e; additions.append(e); selected[shard][wid] = (list(prior.values()), secondary)
            attempt(db, hashlib.sha256(('qualification:' + doi).encode()).hexdigest()[:32], 'young_minds_native_type', 'identity_and_educational_type_confirmed', r['raw_receipt'], {'rule': RULE, 'type': item['kind']})
        with gzip.open(q / 'decision_evidence.jsonl.gz', 'at') as stream, (q / 'curated_evidence.jsonl').open('a') as curated:
            for e in additions:
                line = json.dumps(e, ensure_ascii=False) + '\n'; stream.write(line); curated.write(line)
                db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)', (e['work_id'], e['id'], 'primary', json.dumps(e, ensure_ascii=False)))
        db.commit()
        for shard, targets in selected.items():
            path = q / 'decisions' / shard; rows = pq.read_table(path).to_pylist()
            for row in rows:
                if row['work_id'] not in targets: continue
                primary, secondary = targets[row['work_id']]; new = updated_decision(row, primary, secondary, annual)
                stats[row['eligibility_status'] + ' -> ' + new['eligibility_status']] += 1; row.update(new)
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], row['work_id']))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?", ('resolved' if row['eligibility_status'].startswith(('included','excluded')) else 'pending', row['reason'], row['doi']))
            write_decisions(path, rows, q / 'decision_changes.jsonl.gz'); db.commit()
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)', ('primary', stamp(q / 'decision_evidence.jsonl.gz')))
        db.commit(); db.close()
        # Keep the portable offline decision rule synchronized within the writer lock.
        source = Path('experiments/fig8_extension/fig8b_qualification_decide.py')
        (q / 'decide.py').write_bytes(source.read_bytes())
    result = {'cached_pages_considered': considered, 'new_evidence': len(additions), **stats}
    atomic_json(FOLDER / 'apply_status.json', result); print(json.dumps(result), flush=True)
    return result
