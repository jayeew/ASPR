"""Apply cached deposited country statements in the retained fixed residual scope."""
from __future__ import annotations

import json

from . import terminal_country_apply as application
from .country_change_validation import run as validate
from .country_scope_summary import run as summarize
from .iso_postal_country_cache import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = FOLDER / 'crossref_deposited'
    folder.mkdir(exist_ok=True)
    scope = folder / 'scope.json'
    if not scope.exists():
        review = json.loads((FOLDER / 'remaining_native_review.json').read_text())
        pairs = {(r['work_id'], r['source_receipt']) for r in review['records']
                 if r['status'] == 'native_adapter_or_author_links_missing'
                 and r['source_receipt'].startswith('fields/crossref_raw/')}
        original = json.loads((FOLDER / 'scope.json').read_text())
        papers = []
        for paper in original['papers']:
            slots = [s for s in paper['slots'] if (paper['work_id'], s['receipt']) in pairs]
            if slots:
                papers.append({**paper, 'slots': slots})
        atomic_json(scope, {**original, 'papers': papers,
            'source_authority': 'Crossref publisher-deposited author-affiliation metadata; not native complete byline',
            'parent_scope': str((FOLDER / 'scope.json').relative_to(ROOT))})
    if not (folder / 'application.json').exists():
        application.FOLDER = folder
        application.RULE = str(scope.relative_to(ROOT))
        application.run(deposited_country_metadata_only=True)
    validate(folder)
    summarize(FOLDER)


if __name__ == '__main__':
    run()
