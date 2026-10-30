"""Apply reviewed government university lists, excluding known cross-country homonyms."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'australia': ('https://www.education.gov.au/research-block-grants/higher-education-providers-eligible-research-block-grants',
        'Australian Catholic University|Avondale University|Bond University|Central Queensland University|Charles Darwin University|Charles Sturt University|Curtin University|Deakin University|Edith Cowan University|Federation University Australia|Flinders University|Griffith University|James Cook University|La Trobe University|Macquarie University|Monash University|Murdoch University|Queensland University of Technology|Southern Cross University|Swinburne University of Technology|The Australian National University|The University of Melbourne|The University of Notre Dame Australia|The University of Queensland|The University of Sydney|The University of Western Australia|Torrens University of Australia|University of Canberra|University of Divinity|University of New South Wales|University of Southern Queensland|University of Tasmania|University of Technology Sydney|University of the Sunshine Coast|University of Wollongong|Western Sydney University'),
    'new_zealand': ('https://www.govt.nz/organisations/universities-nz/',
        'Auckland University of Technology|Massey University|The University of Auckland|University of Canterbury|University of Otago|University of Waikato|Victoria University of Wellington'),
}

if __name__ == '__main__':
    run(SOURCES, 'oceania_self', {'australia': 'AU', 'new_zealand': 'NZ'},
        'No Adelaide 2026 merger back-projection; University of Adelaide/South Australia retain their historical identities. Newcastle, New England, Victoria and Lincoln bare homonyms held; nonuniversity institutes, RMIT shorthand, all hospitals, centers and overseas campuses excluded. NZ list updated 2022-11-15; Australian list updated 2026-04-21 is identity corroboration only, not historical ownership proof.')
