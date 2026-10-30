"""Exact university-self names corroborated by institutional English profiles."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'south_china': ('https://english.usc.edu.cn/info/1034/1869.htm',
                    'University of South China'),
    'jinan': ('https://www.ujn.edu.cn/jndxen/About/About_the_UJN.htm',
              'University of Jinan'),
    'south_china_agricultural': ('https://english.scau.edu.cn/417/main.psp',
                                'South China Agricultural University'),
    'punjab': ('https://pu.edu.pk/', 'University of the Punjab'),
}


if __name__ == '__main__':
    run(SOURCES, 'high_frequency_asia_self',
        {'south_china': 'CN', 'jinan': 'CN', 'south_china_agricultural': 'CN', 'punjab': 'PK'},
        'Whole exact signed names only. University of Jinan is not Jinan University; '
        'University of the Punjab is not Panjab University. No inferred aliases, '
        'departments, affiliated hospitals, campuses or joint centers.',
        evidence_locator='Official university profile or homepage: full institution name and university description')
