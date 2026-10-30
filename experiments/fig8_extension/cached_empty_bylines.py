"""Reparse qualified cached JATS records missed by the old author-group parser."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter

from lxml import etree

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata

FOLDER = ROOT / 'supplement/cached_empty_bylines'
REVISION = 'explicit-jats-author-group-1'


def run() -> set[str]:
    FOLDER.mkdir(exist_ok=True); stats = Counter(); affected = set()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX); db = connect()
        rows = db.execute("SELECT DISTINCT s.receipt,s.doi,s.payload FROM source_metadata s JOIN papers p USING(doi) WHERE s.kind='jats' AND p.status='included_verified_research' AND json_array_length(json_extract(s.payload,'$.authors'))=0").fetchall()
        with (FOLDER / 'reparse_events.jsonl').open('a') as events:
            for index, (receipt, doi, payload) in enumerate(rows):
                old = json.loads(payload)
                if old.get('empty_byline_reparse_revision') == REVISION:
                    stats['already_checked'] += 1; continue
                raw_path = ROOT / old['evidence_receipt']
                try:
                    raw = gzip.decompress(raw_path.read_bytes())
                    if raw.rstrip().endswith(b'</front>'): raw += b'</article>'
                    parsed = jats_metadata(raw)
                except (OSError, ValueError, etree.XMLSyntaxError) as error:
                    stats['unresolved_raw_or_parser'] += 1
                    events.write(json.dumps({'receipt': receipt, 'doi': doi, 'status': type(error).__name__}) + '\n')
                    continue
                if doi not in {v.lower() for v in parsed.get('doi', [])}:
                    stats['native_doi_conflict'] += 1; continue
                new = {**old, 'empty_byline_reparse_revision': REVISION}
                if parsed.get('authors'):
                    new.update(authors=parsed['authors'], byline_has_etal=parsed.get('byline_has_etal', False))
                    affected.add(doi); stats['bylines_recovered'] += 1
                    stats['native_author_records_recovered'] += len(parsed['authors'])
                else: stats['still_no_explicit_authors'] += 1
                db.execute('UPDATE source_metadata SET payload=? WHERE receipt=?', (json.dumps(new, ensure_ascii=False), receipt))
                events.write(json.dumps({'receipt': receipt, 'doi': doi, 'before_authors': old.get('authors'),
                    'after_authors': new.get('authors'), 'raw_receipt': old['evidence_receipt'],
                    'reason': 'Reuse existing explicit JATS author-group parser on formerly empty byline; no network or inferred links', 'revision': REVISION}, ensure_ascii=False) + '\n')
                if index % 500 == 0:
                    db.commit(); events.flush()
                    atomic_json(FOLDER / 'status.json', {'status': 'reparsing_fixed_cached_sources', 'fixed_sources': len(rows), 'processed': index + 1, **stats})
            db.commit(); db.close()
    atomic_json(FOLDER / 'scope.json', {'dois': sorted(affected), 'scope': 'Currently qualified papers with newly recovered explicit cached byline'})
    atomic_json(FOLDER / 'status.json', {'status': 'reparsed_metadata_application_pending', 'fixed_sources': len(rows), 'affected_dois': len(affected), **stats})
    print(json.dumps({'affected_dois': len(affected), **stats}), flush=True)
    return affected
