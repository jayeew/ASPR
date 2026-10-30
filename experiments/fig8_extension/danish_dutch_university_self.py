"""Apply reviewed Danish agency and Nuffic names to existing exact university identities."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'denmark': ('https://ufsn.dk/english/education/the-higher-education-system/universities/',
        'University of Copenhagen|Aarhus University|University of Southern Denmark|Roskilde University|Aalborg University|Technical University of Denmark|Copenhagen Business School|IT University of Copenhagen'),
    'netherlands': ('https://www.studyinnl.org/dutch-education/research-universities',
        'Delft University of Technology|Eindhoven University of Technology|Erasmus University Rotterdam|Leiden University|Maastricht University|Nyenrode Business University|Radboud University Nijmegen|The Protestant Theological University|Tilburg University|University of Amsterdam|University of Groningen|University of Humanistic Studies|University of Twente|Utrecht University|VU Amsterdam'),
}

if __name__ == '__main__':
    run(SOURCES, 'danish_dutch_self', {'denmark': 'DK', 'netherlands': 'NL'},
        'Exact whole named university-self identity only, corroborated by signed 2023–2025 paper affiliation; current list does not establish historical subsidiary ownership. Bare Open University homonyms, Wageningen University and Research Centre composite, TIAS school, hospitals, university medical centers, joint centers and campuses excluded from this rule. Missing or nonunique registry names remain held; no automatic study exclusion.')
