"""Apply DOI-matched publisher MODS types without inferring online dates or bylines."""
from __future__ import annotations

import json
import os

from .pensoft_bulk import FOLDER, URL
from .publisher_delta_apply import apply_available
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    staged = FOLDER / 'staged'
    staged.mkdir(exist_ok=True)
    jobs = []
    for path in (FOLDER / 'matched').glob('*.json'):
        source = json.loads(path.read_text())
        parsed = source['parsed']
        record = {'source': 'publisher_mods', 'route': 'pensoft', 'doi': source['doi'],
                  'url': URL, 'status': 'parsed',
                  'raw_receipt': os.path.relpath(ROOT / source['page_receipt']['raw_receipt'], ROOT / 'fields/qualification'),
                  'mods_source_receipt': str(path.relative_to(ROOT)),
                  'parsed': {**parsed, 'authors': [], 'date_precision': '',
                             'title': [t.lstrip('\ufeff') for t in parsed['title']],
                             'publisher': ['Pensoft Publishers']},
                  'limitations': ['No online date inferred from MODS dateAvailable',
                                  'Native author records retained separately; not yet normalized or applied']}
        target = staged / path.name
        atomic_json(target, record)
        jobs.append((source['doi'], str(target.relative_to(ROOT))))
    result = apply_available(supplied_jobs=jobs) if jobs else {}
    atomic_json(FOLDER / 'application.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()))
