"""Fill qualified APS abstract gaps from native cached batch metadata only."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .aps_authors import journal_issns
from .derived_refresh import refresh
from .fig8b_qualification_recompute import metrics
from .fig8b_qualification_build import plain
from .supplement_apply import abstract_ok, signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def abstract_text(article: dict) -> str:
    value = article.get('abstract') or {}
    if not isinstance(value, dict) or value.get('format') not in {'html', 'text', 'plain', None}:
        return ''
    return plain(value.get('value', ''))


def run(output_folder: Path | None = None, cached_paths: list[Path] | None = None) -> None:
    folder = output_folder if output_folder is not None else ROOT / 'supplement/aps_cached_abstracts'
    folder.mkdir(exist_ok=True)
    scope_path = folder / 'scope.json'
    db = connect()
    if not scope_path.exists():
        rows = db.execute("SELECT work_id,doi,shard,title,journal,source_id FROM papers "
                          "WHERE status='included_verified_research' AND abstract_gap=1 AND route='aps'").fetchall()
        atomic_json(scope_path, {'papers': [dict(zip(['work_id', 'doi', 'shard', 'title', 'journal_name', 'source_id'], r))
                                          for r in rows], 'scope': 'Current qualified APS abstract gaps', 'network_requests': 0})
    db.close()
    papers = json.loads(scope_path.read_text())['papers']
    by_doi = defaultdict(list)
    for paper in papers:
        by_doi[paper['doi']].append(paper)
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    found = defaultdict(list)
    rejected = []
    paths = cached_paths if cached_paths is not None else sorted((ROOT / 'fields/qualification/aps_all_raw').glob('*.json.gz'))
    for path in paths:
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record.get('http_status') != 200:
            continue
        for article in record.get('response', {}).get('data', []):
            doi = article.get('identifiers', {}).get('doi', '').lower()
            if doi not in by_doi:
                continue
            payload = {'issns': journal_issns(article), 'title': [plain(article.get('title', {}).get('value', ''))],
                       'journal_titles': [article.get('journal', {}).get('name', '')]}
            text = abstract_text(article)
            for paper in by_doi[doi]:
                item = {'work_id': paper['work_id'], 'doi': doi, 'receipt': str(path.resolve().relative_to(ROOT.resolve())),
                        'source_url': record.get('url'), 'locator': 'response.data[identifiers.doi=' + doi + '].abstract',
                        'native_article_sha256': hashlib.sha256(json.dumps(article, sort_keys=True).encode()).hexdigest()}
                if not signature_match(paper, payload, journals[paper['source_id']]):
                    rejected.append({**item, 'status': 'identity_unmatched'})
                elif not abstract_ok(text):
                    rejected.append({**item, 'status': 'abstract_missing_or_quality_unconfirmed'})
                else:
                    found[paper['work_id']].append({**item, 'abstract': text})
    selected = {}
    for paper in papers:
        wid = paper['work_id']
        values = found.get(wid, [])
        variants = {' '.join(v['abstract'].split()) for v in values}
        if len(variants) == 1:
            selected[wid] = {**values[0], 'corroborating_receipts': [v['receipt'] for v in values]}
        else:
            rejected.append({'work_id': wid, 'doi': paper['doi'],
                             'status': 'cached_abstract_conflict' if variants else 'no_confirmed_abstract_in_batch_cache',
                             'receipts': [v['receipt'] for v in values]})
    atomic_json(folder / 'matched_sources.json', list(selected.values()))
    atomic_json(folder / 'unresolved.json', rejected)
    counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        for shard in sorted({p['shard'] for p in papers if p['work_id'] in selected}):
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            dirty = False
            for paper in rows:
                source = selected.get(paper['work_id'])
                if not source:
                    continue
                status = db.execute('SELECT status FROM papers WHERE work_id=?', (paper['work_id'],)).fetchone()[0]
                if status != 'included_verified_research':
                    counts['no_longer_qualified'] += 1
                    continue
                if paper['abstract']:
                    counts['existing_abstract_preserved'] += 1
                    if paper['abstract_source'] != 'matched_publisher:' + source['receipt']:
                        continue
                    actual = metrics(paper)
                    final = ('observed_fields_present_not_publisher_verified' if all(actual[k] for k in
                             ['abstract', 'every_author_institution', 'every_author_country']) else 'gaps_remain')
                    if paper['metadata_final_status'] == final:
                        continue
                    values = {'metadata_final_status': final}
                    counts['derived_status_repaired'] += 1
                else:
                    values = {'abstract': source['abstract'], 'abstract_source': 'matched_publisher:' + source['receipt'],
                              'after_abstract': 1}
                    actual = metrics({**paper, **values})
                    values['metadata_final_status'] = ('observed_fields_present_not_publisher_verified'
                        if all(actual[k] for k in ['abstract', 'every_author_institution', 'every_author_country'])
                        else 'gaps_remain')
                    counts['abstracts_filled'] += 1
                for field, value in values.items():
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (paper['work_id'], field, json.dumps(paper[field]), json.dumps(value), source['receipt'],
                                'Native APS batch abstract; exact DOI/title/journal; agreeing cached variants; fill empty value only'))
                    paper[field] = value
                db.execute('UPDATE papers SET abstract_gap=0 WHERE work_id=?', (paper['work_id'],))
                dirty = True
            if dirty:
                tmp = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), tmp, compression='zstd')
                tmp.replace(path)
            db.commit()
        db.close()
    result = {**counts, 'fixed_papers': len(papers), 'matched_papers': len(selected),
              'remaining_unconfirmed_papers': len(papers) - len(selected), 'batch_files_reviewed': len(paths),
              'network_requests': 0, 'status': 'applied_derived_refresh_pending'}
    with (folder / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    atomic_json(folder / 'status.json', result)
    refresh()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
