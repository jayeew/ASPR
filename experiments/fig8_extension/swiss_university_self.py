"""Reuse the Swiss university chamber list for exact signed university names."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'switzerland': (
        'https://www.swissuniversities.ch/en/organisation/bodies/chamber-of-universities/member',
        'University of Basel|University of St. Gallen|University of Lausanne|'
        'University of Fribourg|University of Lucerne|Università della Svizzera italiana|'
        'University of Geneva|University of Bern|University of Zurich|University of Neuchâtel',
    ),
}


if __name__ == '__main__':
    run(SOURCES, 'swiss_self', {'switzerland': 'CH'},
        'Official swissuniversities Chamber of universities membership names, '
        'read alongside its accredited-institutions list. Current membership '
        'corroborates exact university-self identities explicitly signed in '
        '2023–2025 papers, not historical affiliate ownership. ETH entry has '
        'a source spelling error and is not crosswalked; EPFL already reviewed. '
        'Nonmatching names, hospitals, departments, joint centers and branch '
        'campuses remain held. Country is not inferred from the directory. '
        'Already reviewed entity states and Nature records are preserved.')
