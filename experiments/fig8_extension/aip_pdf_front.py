"""Bind explicit AIP cover types to the same publication's dated native front."""
from __future__ import annotations

import re
from datetime import datetime

from .fig8b_qualification_decide import key


def extract(pages: list[str], doi: str, title: str, journal: str) -> dict:
    if not doi.startswith('10.1063/') or key(journal) not in key(' '.join(pages)):
        return {'status': 'journal_or_doi_scope_unconfirmed'}
    covers, fronts = [], []
    for number, text in enumerate(pages, 1):
        if key(title) not in key(text):
            continue
        if re.search(r'Accepted\s+Manuscript|Just\s+Accepted', text, re.I):
            continue
        ids = {m.group(1).lower().rstrip('.,;') for line in text.splitlines()
               if (m := re.fullmatch(r'\s*https?://(?:dx\.)?doi\.org/(10\.\S+)\s*', line))}
        labels = re.findall(r'^\s*(RESEARCH ARTICLE|REVIEW ARTICLE|EDITORIAL|LETTER)\s*\|\s*'
                            r'([A-Z]+\s+\d{1,2}\s+\d{4})\s*$', text, re.M)
        if ids == {doi.lower()} and len(labels) == 1:
            try:
                day = datetime.strptime(labels[0][1], '%B %d %Y').date().isoformat()
            except ValueError:
                continue
            covers.append({'page': number, 'kind': labels[0][0], 'date': day})
        native_ids = {v.lower().rstrip('.,;') for v in re.findall(r'doi:\s*(10\.1063/[\w.]+)', text, re.I)}
        dates = re.findall(r'Published\s+Online:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})', text)
        if (native_ids == {doi.lower()} and len(set(dates)) == 1
                and re.search(r'pubs\.aip\.org/aip/[a-z]+', text)):
            try:
                day = datetime.strptime(dates[0], '%d %B %Y').date().isoformat()
            except ValueError:
                continue
            fronts.append({'page': number, 'date': day, 'date_quote': 'Published Online: ' + dates[0]})
    if len(covers) != 1 or len(fronts) != 1 or covers[0]['date'] != fronts[0]['date']:
        return {'status': 'native_type_date_identity_absent_or_conflicting',
                'covers': covers, 'fronts': fronts}
    return {'status': 'native_aip_front_confirmed', 'doi': doi,
            'kind': covers[0]['kind'], 'online_date': fronts[0]['date'],
            'cover_page': covers[0]['page'], 'native_front_page': fronts[0]['page'],
            'date_quote': fronts[0]['date_quote'],
            'journal_identity_basis': 'Identity-matched repository journal/cover plus identical DOI and title on original publisher cover and dated front'}
