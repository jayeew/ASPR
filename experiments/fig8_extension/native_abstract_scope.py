"""Retain explicit native abstract/type conflicts for adjudication."""
from __future__ import annotations

import hashlib
import re

from .fig8b_qualification_decide import type_class

REVIEW_DESCRIPTION = re.compile(
    r'\b(?:in this (?:narrative |systematic |comprehensive |mini[- ]?)?review|'
    r'(?:here|herein|in this article),? we review|'
    r'this review (?:summari[sz]es|discusses|provides|presents|describes|focuses))\b',
    re.I,
)


def native_abstract_scope(parsed: dict, kind: str) -> dict:
    """A self-described review conflicts with a generic research label, not identity."""
    if type_class(kind, 'archived_jats', parsed.get('categories', [])) != 'research':
        return {}
    abstract = parsed.get('abstract') or ''
    found = REVIEW_DESCRIPTION.search(abstract)
    if not found:
        return {}
    return {
        'scope_review_required': True,
        'scope_review_note': 'Native abstract self-describes review content; generic research label requires subtype adjudication, not automatic exclusion.',
        'review_abstract': abstract,
        'review_quote': found.group(0),
        'abstract_sha256': hashlib.sha256(abstract.encode()).hexdigest(),
        'review_rule_receipt': 'supplement/target_500k/pmc_exact_public/abstract_scope_review/rule.json',
    }
