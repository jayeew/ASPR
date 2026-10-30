"""Corroborate exact university-self names from HEC parent institution entries."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'pakistan': ('https://www.hec.gov.pk/english/universities/Pages/DAIs/HEC-recognized-Campuses.aspx',
        'Bahria University|Air University|Virtual University of Pakistan|'
        'National Textile University|University of Punjab|University of Education|'
        'University of Gujrat|University of Sargodha|'
        'Pir Mehr Ali Shah Arid Agriculture University|Al-Kawthar University|'
        'University of Sindh|Shah Abdul Latif University|Sukkur IBA University|'
        'Sindh Agriculture University|Dow University of Health Sciences|'
        'University of Swat|Gomal University|Hazara University|University of Malakand|'
        'University of Turbat|Sardar Bahadur Khan Women University|'
        'Balochistan University of Information Technology, Engineering and Management Sciences|'
        'University of Balochistan|University of Lahore|Preston University|'
        'Isra University|Iqra University|Abasyn University|Hamdard University|'
        'Qurtuba University of Science and Information Technology|'
        'Riphah International University|Al-Hamd Islamic University|Greenwich University|'
        'Hajvery University|Aga Khan University|Iqra National University|Ilma University|'
        'Superior University|Ibadat International University|DHA Suffa University|'
        'Indus University|Lahore Leads University'),
}


if __name__ == '__main__':
    run(SOURCES, 'pakistan_hec_self', {'pakistan': 'PK'},
        'Use only the parent University/DAI name column to corroborate exact '
        'university-self identity alongside the 2023–2025 primary signature and '
        'unique existing education entity. This is not the complete Pakistan '
        'university roster or an accreditation/date adjudication. Campus locations, '
        'degree restrictions, later closures/upgrades, hospitals and departments '
        'are not assigned. Do not infer current or historical subsidiary ownership '
        'or author country. Ambiguous location-qualified names, spelling errors, '
        'COMSATS predecessor name, NUST abbreviation and nonmatching names remain '
        'unresolved. Existing reviewed identities and Nature records preserved.')
