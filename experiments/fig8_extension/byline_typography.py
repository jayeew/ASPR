"""Conservative full-byline comparison for punctuation-only name variants."""
from __future__ import annotations

import json
import unicodedata
from collections import Counter

from . import byline_complete_cache as application
from .primary_byline_comparison import compare
from .supplement_queue import ROOT


def name_key(value: str) -> str:
    # Preserve word boundaries: Hongying is not equated with Hong Ying.
    text = ''.join(' ' if unicodedata.category(c) == 'Pd' else c for c in value)
    return ' '.join(text.replace('.', '').casefold().split())


def compare_typography(current: list[dict], incoming: dict) -> str:
    old = [name_key(a['name']) for a in current]
    new = [name_key(a['name']) for a in incoming.get('authors', [])]
    if not old or old != new:
        return 'author_sequence_differs'
    if any(n != 1 for n in Counter(old).values()):
        return 'duplicate_author_names_require_review'
    adjusted = json.loads(json.dumps(incoming))
    for a, b in zip(current, adjusted['authors']):
        b['name'] = a['name']
    return compare(current, adjusted)


def run() -> None:
    application.FOLDER = ROOT / 'supplement/native_typography_byline'
    application.run(name_typography=True)


if __name__ == '__main__':
    run()
