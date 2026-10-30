"""Extract explicit online-release dates from a DOI-identified publisher first page."""
from __future__ import annotations

import re
from datetime import datetime


def extract(first_page: str, doi: str) -> dict:
    lines = first_page.splitlines()
    standalone = {m.group(1).lower().rstrip('.,;') for line in lines
                  if (m := re.fullmatch(r'\s*https?://(?:dx\.)?doi\.org/(10\.\S+)\s*', line))}
    if standalone != {doi.lower()}:
        return {'status': 'first_page_standalone_doi_not_unique_or_mismatched'}
    value = ' '.join(first_page.split())
    matches = re.findall(r'\bAvailable online\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})\b', value)
    dates = set()
    for match in matches:
        try:
            dates.add(datetime.strptime(match, '%d %B %Y').date().isoformat())
        except ValueError:
            return {'status': 'invalid_explicit_online_date'}
    if len(dates) != 1:
        return {'status': 'explicit_online_date_absent_or_conflicting'}
    return {'status': 'explicit_online_date_identified', 'online_date': dates.pop(),
            'date_precision': 'day', 'matched_doi': doi.lower(), 'source_text': matches}
