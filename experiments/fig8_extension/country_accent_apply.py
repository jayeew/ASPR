"""Retry only reviewed unmatched native names with unique Latin accent folding."""
from __future__ import annotations

import json

from . import terminal_country_apply as application
from .country_change_validation import run as validate
from .country_scope_summary import run as summarize
from .iso_postal_country_cache import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = FOLDER / 'accent_names'
    folder.mkdir(exist_ok=True)
    scope_path = folder / 'scope.json'
    if not scope_path.exists():
        review = json.loads((FOLDER / 'remaining_native_review.json').read_text())
        records = [r for r in review['records'] if r['status'] == 'native_author_name_not_matched']
        pairs = {(r['work_id'], r['source_receipt']) for r in records}
        original = json.loads((FOLDER / 'cleaned_native/scope.json').read_text())
        papers = []
        for paper in original['papers']:
            slots = [s for s in paper['slots'] if (paper['work_id'], s['receipt']) in pairs]
            if slots:
                papers.append({**paper, 'slots': slots})
        atomic_json(scope_path, {**original, 'papers': papers,
            'parent_scope': str((FOLDER / 'scope.json').relative_to(ROOT)),
            'name_review_receipt': str((FOLDER / 'remaining_native_review.json').relative_to(ROOT)),
            'matching_rule': 'Unique full Latin name after accent folding in both rosters, exact native address, matched DOI/title/journal; no initials expansion or name reordering.',
            'final_task_closure': False})
    if not (folder / 'application.json').exists():
        application.FOLDER = folder
        application.RULE = str(scope_path.relative_to(ROOT))
        application.run(allow_unique_accent_names=True)
    validate(folder)
    summarize(FOLDER)


if __name__ == '__main__':
    run()
