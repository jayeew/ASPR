"""Apply individual native-content reviews of Frontiers Methods papers."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .mdpi_brief_content_apply import native_article_xml
from .supplement_apply import apply as apply_metadata
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_bytes, atomic_json, jats_metadata, parse_publisher

FOLDER = ROOT / 'supplement/frontiers_remaining_formats'


def prepare(only_dois: set[str] | None = None) -> tuple[dict, dict]:
    scope = {r['doi']: r for r in json.loads((FOLDER / 'methods_content_scope.json').read_text())['records']}
    reviews = json.loads((FOLDER / 'methods_content_reviews.json').read_text())['records']
    db = connect()
    sources, metadata = {}, {}
    for review in reviews:
        if only_dois is not None and review['doi'] not in only_dois:
            continue
        if review['classification'] not in {'research', 'nonprimary'}:
            continue
        doi = review['doi']
        row = scope[doi]
        raw_path = ROOT / 'fields/qualification' / row['native_receipt']
        raw = gzip.decompress(raw_path.read_bytes())
        publisher = row.get('native_family') == 'publisher_frontiers'
        parsed = parse_publisher(raw) if publisher else jats_metadata(native_article_xml(raw))
        if publisher:
            parsed.update(authors=[], byline_has_etal=True, native_metadata_kind='publisher')
        abstract = parsed.get('abstract', '')
        if (doi not in parsed.get('doi', []) or abstract != review['full_native_abstract']
                or hashlib.sha256(raw).hexdigest() != review['raw_sha256']
                or hashlib.sha256(abstract.encode()).hexdigest() != review['abstract_sha256']):
            raise ValueError('Native content identity or reviewed content changed: ' + doi)
        body_evidence = {}
        if review.get('full_body_receipt'):
            body = json.loads((ROOT / review['full_body_receipt']).read_text())
            body_raw = gzip.decompress((ROOT / body['retrieval_receipt']['raw_receipt']).read_bytes())
            if (body['doi'] != doi or body['status'] != 'matched_native_body_ready_for_individual_review'
                    or hashlib.sha256(body['body'].encode()).hexdigest() != review['body_sha256']
                    or hashlib.sha256(body_raw).hexdigest() != body['raw_sha256']):
                raise ValueError('Reviewed full body identity or content changed: ' + doi)
            body_evidence = {k: review[k] for k in ('full_body_receipt', 'body_sha256')}
        prior = {json.loads(p)['id']: json.loads(p) for p, in db.execute(
            "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],))}
        backing = prior[row['evidence_id']]
        targets = [e['id'] for e in prior.values() if e.get('identity') == 'matched' and e.get('doi') == doi
                   and ((e.get('family') == 'archived_jats' and e.get('kind') == 'methods-article')
                        or (e.get('family') == 'publisher_frontiers' and e.get('kind') == 'Methods'))]
        sources[doi] = [{'family': 'reviewed_study_design', 'doi': doi, 'title': backing['title'],
            'issns': backing.get('issns', []), 'journal_titles': backing.get('journal_titles', []),
            'kind': 'Individual native Methods content review', 'categories': [], 'online_date': '',
            'date_precision': 'not_a_date_source', 'study_design_class': review['classification'],
            'review_protocol': 'native_frontiers_methods_design_1', 'reviewed_primary_evidence_ids': targets,
            'source_evidence_ids': [backing['id']],
            'review_rule_receipt': 'supplement/frontiers_remaining_formats/methods_content_reviews.json',
            'format_policy_receipt': 'supplement/frontiers_remaining_formats/methods_policy_review.json',
            'study_design_reason': review['reason'], 'raw_sha256': review['raw_sha256'],
            'abstract_sha256': review['abstract_sha256'], 'content_locator': review['content_locator'],
            **body_evidence,
            'receipt': '../../supplement/frontiers_remaining_formats/methods_content_reviews.json',
            'url': 'https://doi.org/' + doi}]
        metadata[doi] = {**parsed, 'issns': parsed.get('issns', parsed.get('issn', [])),
            'evidence_receipt': str(raw_path.resolve().relative_to(ROOT.resolve())),
            'source_url': 'https://doi.org/' + doi, 'raw_sha256': review['raw_sha256']}
    db.close()
    return sources, metadata


def run(only_dois: set[str] | None = None) -> dict:
    sources, metadata = prepare(only_dois)
    folder = FOLDER / 'content_application'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'sources.json', sources)
    qualification = apply_sources(set(sources), folder, explicit_sources=sources)
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        selected = set()
        for doi, payload in metadata.items():
            if not db.execute("SELECT 1 FROM papers WHERE doi=? AND status='included_verified_research'", (doi,)).fetchone():
                continue
            selected.add(doi)
            db.execute('INSERT OR REPLACE INTO source_metadata VALUES(?,?,?,?,?)',
                       (payload['evidence_receipt'] + '#' + doi, doi, payload.get('native_metadata_kind', 'jats'),
                        payload['raw_sha256'], json.dumps(payload)))
        db.commit()
        db.close()
        merged = apply_metadata(ROOT, scope_dois=selected, index_sources=False) if selected else {}
        source = Path(__file__).with_name('fig8b_qualification_decide.py')
        target = ROOT / 'fields/qualification/decide.py'
        before, after = target.read_bytes(), source.read_bytes()
        if before != after:
            atomic_bytes(target, after)
            with (ROOT / 'supplement/offline_rule_sync_events.jsonl').open('a') as stream:
                stream.write(json.dumps({'source': str(source), 'target': str(target),
                    'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': hashlib.sha256(after).hexdigest(),
                    'reason': 'Bound individual Frontiers Methods review; preserve date and source conflicts',
                    'updated_at': datetime.now(timezone.utc).isoformat()}) + '\n')
    result = {'qualification': qualification, 'qualified_scope': len(selected), 'metadata': merged,
              'network_requests': 0, 'final_task_closure': False}
    with (folder / 'applications.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    atomic_json(folder / 'application.json', result)
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
