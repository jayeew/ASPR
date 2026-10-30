"""Translate official AIMS structured author links into the native metadata adapter."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from .aims_native_collect import FOLDER
from .fig8b_qualification_build import plain
from .supplement_apply import apply
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def metadata(article: dict, receipt: str) -> dict:
    affiliations = article.get('affiliations') or []
    by_id = {a['tagId']: a for a in affiliations if a.get('tagId')}
    all_authors = article.get('authors') or []
    shared = len(affiliations) == 1 and not affiliations[0].get('labelId') and all(
        not a.get('addressTagIds') and not a.get('addressLabels') for a in all_authors)
    authors = []
    for author in all_authors:
        if author.get('contribType') != 'author':
            continue
        refs = author.get('addressTagIds') or []
        linked = [by_id[v] for v in refs if v in by_id]
        if shared:
            linked = affiliations
        authors.append({'name': plain(author.get('authorNameEn') or ''),
            'orcid': author.get('orcid') or '', 'nonpersonal': False,
            'unresolved_affiliation_refs': [v for v in refs if v not in by_id] or
                ([] if linked else ['native_author_affiliation_link_absent']),
            'affiliations': [{'text': plain(a.get('affliEn') or ''), 'countries': [],
                              'identifiers': [], 'names': []} for a in linked]})
    journal = article.get('journalMeta') or {}
    return {'doi': [article['doi'].casefold()], 'title': [plain(article.get('titleEn') or '')],
        'journal_titles': [journal.get('titleEn') or ''],
        'issns': [journal[k] for k in ['issnEpub', 'issnPpub'] if journal.get(k)],
        'abstract': plain(article.get('abstractinfoEn') or ''), 'authors': authors,
        'byline_has_etal': any(not a['name'] for a in authors),
        'evidence_receipt': receipt, 'source_url': 'https://www.aimspress.com/article/doi/' + article['doi'],
        'native_author_adapter': 'AIMS addressTagIds; unlabeled single shared affiliation only when all byline links are empty'}


def run() -> dict:
    scope = {r['doi'] for r in json.loads((FOLDER / 'scope.json').read_text())['records']}
    extra = FOLDER / 'additional_existing_candidate_reuse.json'
    if extra.exists():
        scope.update(r['doi'] for r in json.loads(extra.read_text())['records'])
    counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); db = connect(); selected = set()
        for doi in scope:
            if db.execute("SELECT 1 FROM papers WHERE doi=? AND status='included_verified_research'", (doi,)).fetchone():
                selected.add(doi)
        for path in sorted((FOLDER / 'issues').glob('*.json')):
            record = json.loads(path.read_text())
            if record.get('http_status') != 200 or not record.get('raw_receipt'):
                continue
            raw_path = Path(record['raw_receipt']); raw = gzip.decompress(raw_path.read_bytes())
            data = json.loads(raw).get('data')
            if not isinstance(data, dict):
                continue
            receipt = str(raw_path.relative_to(ROOT))
            for article in data.get('articles', []):
                doi = (article.get('doi') or '').casefold()
                if doi not in selected:
                    continue
                payload = metadata(article, receipt)
                db.execute('INSERT OR REPLACE INTO source_metadata VALUES(?,?,?,?,?)',
                    (receipt + '#' + doi, doi, 'jats', hashlib.sha256(raw).hexdigest(), json.dumps(payload, ensure_ascii=False)))
                counts['native_records_indexed'] += 1
        db.commit(); db.close()
        result = apply(ROOT, scope_dois=selected, index_sources=False) if selected else {}
        output = {'qualified_scope': len(selected), **counts, 'application': result}
        atomic_json(FOLDER / 'metadata_application.json', output)
    return output


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
