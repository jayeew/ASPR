"""Reuse ISSN-bound native journal titles to repair exact cached JATS identities."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache

import pyarrow.parquet as pq
from lxml import etree

from .fig8b_qualification_build import key, match, write_decisions
from .fig8b_qualification_evidence import parse_front
from .publisher_delta_apply import evidence_index, stamp, updated_decision
from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

RULE = 'supplement/cached_journal_identity_rules.json'


@lru_cache(maxsize=1024)
def native(receipt: str) -> dict:
    raw = gzip.decompress((ROOT / receipt).read_bytes())
    if raw.rstrip().endswith(b'</front>'):
        raw += b'</article>'
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    if etree.QName(tree).localname != 'article':
        articles = tree.xpath('//*[local-name()="metadata"]/*[local-name()="article"]')
        if len(articles) != 1:
            raise ValueError('Expected one archive article')
        tree = articles[0]
    for node in tree.iter():
        if isinstance(node.tag, str):
            node.tag = etree.QName(node).localname
    return parse_front(etree.tostring(tree))


def corroborates(paper: dict, evidence: dict, record: dict, rule: dict) -> bool:
    """Require native DOI/title and a uniquely ISSN-bound exact journal title."""
    doi = paper['doi'].lower()
    return bool(doi and doi in {v.lower() for v in record['doi']}
                and key(paper['title']) == key(evidence['title'])
                and key(paper['title']) in {key(v) for v in record['title']}
                and not record['issn'] and not evidence.get('issns')
                and {key(v) for v in record['journal_titles']} & set(rule['title_keys'])
                and {key(v) for v in evidence.get('journal_titles', [])} & set(rule['title_keys']))


def run() -> dict:
    q = ROOT / 'fields/qualification'
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    owners = defaultdict(set)
    for sid, row in journals.items():
        for issn in json.loads(row['issn'] or '[]'):
            owners[issn].add(sid)
    rules = {}; stats = Counter()
    proofs = json.loads((ROOT / 'supplement/pmc_journal_domain_proofs.json').read_text())
    for sid, proof in proofs.items():
        shared = set(proof['issns']) & set(json.loads(journals[sid]['issn'] or '[]'))
        if not shared or any(owners[i] != {sid} for i in shared):
            continue
        try:
            record = native(proof['receipt'])
        except (OSError, ValueError, etree.XMLSyntaxError):
            stats['proof_raw_unresolved'] += 1
            continue
        if not shared <= set(record['issn']):
            continue
        titles = {key(v) for v in record['journal_titles']}
        if titles != set(proof['title_keys']):
            continue
        rules[sid] = {**proof, 'validated_issns': sorted(shared),
                      'rule': 'Exact native journal title with ISSN uniquely mapped in current registry; PMC domain is not identity evidence'}
    atomic_json(ROOT / RULE, {'rules': rules, 'network_requests': 0})
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect(); evidence_index(db, q)
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year'] for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        selected = defaultdict(dict); additions = []
        candidates = db.execute("SELECT work_id,doi,title,journal,source_id,shard FROM papers WHERE status='pending_primary_identity_mismatch'").fetchall()
        for wid, doi, title, journal, sid, shard in candidates:
            if sid not in rules or not doi:
                continue
            prior = {}; secondary = []
            for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (wid,)):
                item = json.loads(payload)
                if kind == 'primary': prior[item['id']] = item
                else: secondary = item
            paper = {'work_id': wid, 'doi': doi, 'title': title, 'journal_name': journal, 'source_id': sid}
            new = []
            for item in list(prior.values()):
                if item['family'] != 'archived_jats' or item.get('issns') or item['identity'] == 'matched':
                    continue
                receipt = str((q / item['receipt']).relative_to(ROOT))
                try:
                    record = native(receipt)
                except (OSError, ValueError, etree.XMLSyntaxError):
                    stats['raw_receipt_unresolved'] += 1
                    continue
                if not corroborates(paper, item, record, rules[sid]):
                    continue
                revised = {k: v for k, v in item.items() if k not in {'id', 'identity', 'class'}}
                revised.update(source_id=sid, journal_identity_rule=RULE,
                               journal_identity_proof=rules[sid]['receipt'], supersedes_identity_evidence=item['id'])
                revised = match(paper, revised, journals[sid])
                if revised['identity'] != 'matched' or revised['id'] in prior:
                    continue
                prior[revised['id']] = revised; new.append(revised)
            if not new: continue
            additions.extend(new); selected[shard][wid] = (list(prior.values()), secondary)
            task_id = hashlib.sha256(('qualification:' + doi).encode()).hexdigest()[:32]
            attempt(db, task_id, 'cached_issn_bound_journal_identity', 'identity_confirmed_qualification_recomputed', RULE,
                    {'source_id': sid, 'native_proof': rules[sid]['receipt'], 'new_evidence': [e['id'] for e in new]})
        with gzip.open(q / 'decision_evidence.jsonl.gz', 'at') as stream, (q / 'curated_evidence.jsonl').open('a') as curated:
            for item in additions:
                line = json.dumps(item, ensure_ascii=False) + '\n'; stream.write(line); curated.write(line)
                db.execute('INSERT OR REPLACE INTO current_evidence_index VALUES(?,?,?,?)', (item['work_id'], item['id'], 'primary', json.dumps(item, ensure_ascii=False)))
        db.commit()
        for shard, targets in selected.items():
            path = q / 'decisions' / shard; rows = pq.read_table(path).to_pylist()
            for row in rows:
                if row['work_id'] not in targets: continue
                primary, secondary = targets[row['work_id']]
                revised = updated_decision(row, primary, secondary, annual)
                stats[row['eligibility_status'] + ' -> ' + revised['eligibility_status']] += 1
                row.update(revised)
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (row['eligibility_status'], row['work_id']))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?", ('resolved' if row['eligibility_status'].startswith(('included', 'excluded')) else 'pending', row['reason'], row['doi']))
            write_decisions(path, rows, q / 'decision_changes.jsonl.gz'); db.commit()
        db.execute('INSERT OR REPLACE INTO current_evidence_files VALUES(?,?)', ('primary', stamp(q / 'decision_evidence.jsonl.gz')))
        db.commit(); db.close()
    result = {'status': 'identity_repaired_derived_refresh_required', 'new_evidence_records': len(additions), 'papers_rechecked': sum(map(len, selected.values())), 'network_requests': 0, **stats}
    atomic_json(ROOT / 'supplement/cached_journal_identity_status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
