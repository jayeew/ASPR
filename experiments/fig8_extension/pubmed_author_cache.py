"""Reuse scoped PubMed XML author statements; never assert primary byline completeness."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from .derived_refresh import refresh
from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import country_names
from .secondary_author_cache import enrich
from .snapshot_location_recovery import FOLDER as RECOVERY
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = RECOVERY / 'pubmed_author_cache'


def metadata(article: etree._Element, receipt: str) -> dict:
    source = article.find('./MedlineCitation/Article')
    authors = []
    if source is None:
        return {}
    for author in source.findall('./AuthorList/Author'):
        first, last = author.findtext('ForeName') or '', author.findtext('LastName') or ''
        if len(first.replace('.', '').strip()) < 2 or not last:
            continue
        affiliations = [{'text': ' '.join(''.join(a.itertext()).split()), 'countries': [],
                         'identifiers': [], 'names': []} for a in author.findall('./AffiliationInfo/Affiliation')]
        authors.append({'name': first + ' ' + last, 'affiliations': [a for a in affiliations if a['text']]})
    title = source.find('ArticleTitle')
    return {'doi': [d.strip().lower() for d in article.xpath('./PubmedData/ArticleIdList/ArticleId[@IdType="doi"]/text()')],
        'title': [''.join(title.itertext()) if title is not None else ''],
        'issns': article.xpath('./MedlineCitation/Article/Journal/ISSN/text() | ./MedlineCitation/MedlineJournalInfo/ISSNLinking/text()'),
        'journal_titles': [source.findtext('./Journal/Title') or ''], 'authors': authors,
        'evidence_receipt': receipt, 'source_url': 'https://pubmed.ncbi.nlm.nih.gov/' + (article.findtext('./MedlineCitation/PMID') or '') + '/',
        'authority': 'PubMed indexed author-affiliation statements; no primary byline completeness or university ownership assertion'}


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (FOLDER / 'application.json').exists():
            return
        scope = json.loads((RECOVERY / 'pubmed_author_cache_candidates.json').read_text())['papers']
        sources = defaultdict(list)
        for receipt in sorted(set(scope.values())):
            tree = etree.fromstring(gzip.decompress((ROOT / receipt).read_bytes()), etree.XMLParser(resolve_entities=False, no_network=True))
            for article in tree.findall('PubmedArticle'):
                item = metadata(article, receipt)
                for doi in item.get('doi', []):
                    sources[doi].append(item)
        registry, by_name = defaultdict(list), defaultdict(list)
        for r in json.loads((ROOT / 'fields/institution_registry.json').read_text()):
            if r.get('ror'):
                registry[r['ror'].rstrip('/')].append(r['id'])
            by_name[key(r['display_name'])].append(r)
        names = country_names()
        for name in ['korea', 'georgia', 'congo']:
            names.pop(normalized(name), None)
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        counts, audit = Counter(), []
        with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = connect(); groups = defaultdict(set)
            for wid in scope:
                row = db.execute("SELECT shard FROM papers WHERE work_id=? AND status='included_verified_research'", (wid,)).fetchone()
                if row:
                    groups[row[0]].add(wid)
            for shard, ids in groups.items():
                path = ROOT / 'fields/papers' / shard
                table = pq.read_table(path); rows = table.to_pylist(); dirty = False
                for paper in rows:
                    wid = paper['work_id']
                    if wid not in ids:
                        continue
                    matched = [s for s in sources[paper['doi']] if signature_match(paper, s, journals[paper['source_id']])]
                    before = json.loads(paper['authors_json'])
                    # Existing secondary statements need an explicit source comparison, not silent replacement.
                    after, stats = enrich(before, matched, registry, by_name, names,
                                          source_label='pubmed', preserve_secondary=True)
                    state = 'matched_no_new_fields' if matched else 'identity_not_confirmed'
                    if after != before:
                        state = 'secondary_statements_updated'; counts.update(stats); counts['papers_changed'] += 1
                        value = json.dumps(after, ensure_ascii=False)
                        receipts = sorted({s['evidence_receipt'] for s in matched})
                        db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                            (wid, 'authors_json', paper['authors_json'], value, json.dumps(receipts),
                             'Exact PubMed DOI/title/journal and unique full author-name association; secondary authority retained; primary and existing secondary signatures preserved'))
                        paper['authors_json'] = value; dirty = True
                        observed = metrics(paper)
                        db.execute('UPDATE papers SET abstract_gap=?,institution_gap=?,country_gap=? WHERE work_id=?',
                            (not observed['abstract'], not observed['every_author_institution'], not observed['every_author_country'], wid))
                    counts[state] += 1
                    detail = {'work_id': wid, 'status': state, 'candidate_receipt': scope[wid],
                              'matched_receipts': sorted({s['evidence_receipt'] for s in matched}),
                              'existing_secondary_authors_preserved': sum(bool(a.get('secondary_affiliation_slots')) for a in before),
                              'final_task_closure': False}
                    audit.append(detail)
                    for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='metadata' AND entity_key=?", (wid,)).fetchall():
                        if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                                      (tid, 'pubmed_cached_author_statements', scope[wid])).fetchone():
                            continue
                        db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                                   (tid, 'pubmed_cached_author_statements', state, scope[wid], json.dumps(detail)))
                        db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
                if dirty:
                    temporary = path.with_suffix('.parquet.tmp')
                    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd'); temporary.replace(path)
                db.commit()
            db.close()
        atomic_json(FOLDER / 'application.json', {'status': 'fixed_cached_pubmed_author_scope_processed',
            **counts, 'audit': audit, 'network_requests': 0, 'final_task_closure': False})
        refresh()
        print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
