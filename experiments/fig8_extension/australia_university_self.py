"""Apply exact university-self names from Austrade's official university list."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {'australia': ('https://www.studyaustralia.gov.au/en/plan-your-studies/list-of-australian-universities',
    'Australian Catholic University|Australian National University|Avondale University|Bond University|Charles Darwin University|Charles Sturt University|CQ University|Curtin University|Deakin University|Edith Cowan University|Federation University of Australia|Flinders University|Griffith University|James Cook University|Latrobe University|Macquarie University|Monash University|Murdoch University|Queensland University of Technology|RMIT University|Southern Cross University|Swinburne University of Technology|Torrens University Australia|University of Canberra|University of Divinity|University of Melbourne|University of New England|University of New South Wales|University of Newcastle|University of Notre Dame Australia|University of Queensland|University of Southern Queensland|University of Sydney|University of Tasmania|University of Technology Sydney|University of the Sunshine Coast|University of Western Australia|University of Wollongong|Victoria University|Western Sydney University')}


if __name__ == '__main__':
    run(SOURCES, 'australia_self', {'australia': 'AU'},
        'Austrade Study Australia university headings, not campus-location bullets. Current official university list corroborates exact university-self names in 2023–2025 signatures. Adelaide University merger and Australian University of Theology renamed status are excluded from this rule, with no successor back-projection. Nonunique names (e.g. Victoria/New England/Newcastle where ambiguous), registry-name mismatches, departments, hospitals, joint units and campuses remain held. No allocation of country from headquarters; declared author and slot countries unchanged. Omitted universities are not excluded from Fig8.')
