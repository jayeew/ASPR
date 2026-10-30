"""Read native ACS front metadata; repository cover dates never qualify."""
from __future__ import annotations

import re
from datetime import datetime


def extract(page: str, doi: str) -> dict:
    if re.search(r'Just\s+Accepted|Accepted\s+Manuscript', page, re.I):
        return {'status': 'provisional_manuscript_not_promoted'}
    identifiers = {m.group(1).lower().rstrip('.,;') for line in page.splitlines()
                   if (m := re.fullmatch(r'\s*https?://(?:dx\.)?doi\.org/(10\.\S+)\s*', line))}
    if not doi.startswith('10.1021/') or identifiers != {doi.lower()}:
        return {'status': 'native_doi_unconfirmed'}
    labels = re.findall(r'^\s*(Research Article|Article|Review|Communication|Letter)\s*pubs\.acs\.org/[a-zA-Z0-9/]+\s*$', page, re.M)
    if len(set(labels)) != 1 or 'American Chemical Society' not in page:
        return {'status': 'native_publisher_header_unconfirmed'}
    dates = re.findall(r'^Published:\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})\s*$', page, re.M)
    if len(set(dates)) != 1:
        return {'status': 'native_published_date_absent_or_conflicting'}
    try:
        day = datetime.strptime(dates[0], '%B %d, %Y').date().isoformat()
    except ValueError:
        return {'status': 'invalid_published_date'}
    return {'status': 'native_acs_front_confirmed', 'doi': doi, 'kind': labels[0],
            'online_date': day, 'date_quote': 'Published: ' + dates[0],
            'policy_url': 'https://pubs.acs.org/page/4authors/submission/howtosubmit.html',
            'policy_scope': 'ACS states first web publication is the publication date; repository deposit and issue dates are not used.'}
