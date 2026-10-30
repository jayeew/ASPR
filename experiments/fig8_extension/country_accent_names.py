"""Country-slot matching for unique Latin full-name accent variants only."""
from __future__ import annotations

import json
import unicodedata
from collections import Counter

from .cureus_country_apply import enrich as exact_enrich
from .fig8b_qualification_build import key


def accent_key(name: str) -> str | None:
    folded=''.join(c for c in unicodedata.normalize('NFKD',name) if not unicodedata.combining(c))
    if any(c.isalpha() and not c.isascii() for c in folded):
        return None
    return key(folded) or None


def aliases_for(authors: list[dict], source: dict) -> dict[str, str]:
    old_counts=Counter(accent_key(a['name']) for a in authors)
    new_counts=Counter(accent_key(a['name']) for a in source.get('authors',[]))
    native={accent_key(a['name']):a['name'] for a in source.get('authors',[])}
    return {a['name']:native[accent_key(a['name'])] for a in authors
        if accent_key(a['name']) and old_counts[accent_key(a['name'])]==new_counts[accent_key(a['name'])]==1
        and key(a['name'])!=key(native[accent_key(a['name'])])}


def enrich(authors: list[dict], source: dict, receipt: str, rules: dict[str,str],
           rule_receipt: str, basis: str) -> tuple[list[dict], int, int]:
    aliases=aliases_for(authors,source)
    reverse={native:retained for retained,native in aliases.items()}
    adjusted=json.loads(json.dumps(source))
    for author in adjusted.get('authors',[]):
        author['name']=reverse.get(author['name'],author['name'])
    result, slots, people=exact_enrich(authors,adjusted,receipt,rules,rule_receipt,basis)
    for old,new in zip(authors,result):
        if old['name'] not in aliases:continue
        for a,b in zip(old.get('publisher_affiliation_slots') or [],new.get('publisher_affiliation_slots') or []):
            if not a.get('country_codes') and b.get('country_codes'):
                b['country_evidence'].update(native_author_name=aliases[old['name']],
                    retained_author_name=old['name'],name_matching='Unique Latin full name after Unicode accent folding; exact native address; roster and identity unchanged')
    return result, slots, people
