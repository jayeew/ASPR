"""Reuse UKA's explicit Universities category for existing Swedish university identities."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'sweden': (
        'https://www.uka.se/swedish-higher-education-authority/about-higher-education/universities-university-colleges-and-other-education-providers/higher-education-institutions',
        'Chalmers University of Technology|Karlstad University|Karolinska Institutet|KTH Royal Institute of Technology|Linköping University|Linnaeus University|Luleå University of Technology|Lund University|Malmö University|Mid Sweden University|Stockholm School of Economics|Stockholm University|Swedish University of Agricultural Sciences|Umeå University|University of Gothenburg|Uppsala University|Örebro University|Mälardalen University'),
}

if __name__ == '__main__':
    run(SOURCES, 'sweden_self', {'sweden': 'SE'},
        'Only UKA Universities section, last updated 2023-11-13. University colleges and other providers are outside this rule, not automatically excluded from the study. Exact declared university self identity only; do not infer hospital, institute or joint-center parentage. Nonmatching names and already reviewed entities retained for separate handling. No geographical or employment inference.')
