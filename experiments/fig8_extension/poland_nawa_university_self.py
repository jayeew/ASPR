"""Corroborate standalone university names using the Polish NAWA HEI directory."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'poland': ('https://study.gov.pl/higher-education-institutions',
        'Adam Mickiewicz University in Poznan|Cardinal Stefan Wyszyński University|'
        'Jagiellonian University in Krakow|Jan Dlugosz University in Czestochowa|'
        'Jan Kochanowski University in Kielce|John Paul II Catholic University of Lublin|'
        'Kazimierz Wielki University in Bydgoszcz|Maria Curie-Skłodowska University|'
        'Nicolaus Copernicus University in Torun|University of Białystok|'
        'University of Gdansk|University of Lodz|University of Opole|'
        'University of Rzeszów|University of Silesia in Katowice|University of Szczecin|'
        'University of Warmia and Mazury in Olsztyn|University of Warsaw|'
        'University of Wrocław|University of Zielona Góra|'
        'AGH University of Science and Technology in Krakow|Bialystok University of Technology|'
        'University of Science and Technology in Bydgoszcz|Czestochowa University of Technology|'
        'Gdansk University of Technology|Kielce University of Technology|'
        'Koszalin University of Technology|Lodz University of Technology|'
        'Lublin University of Technology|Opole University of Technology|'
        'Poznan University of Technology|Rzeszow University of Technology|'
        'Silesian University of Technology in Gliwice|'
        'Tadeusz Kosciuszko Cracow University of Technology|University of Bielsko-Biala|'
        'Warsaw University of Technology|West Pomeranian University of Technology in Szczecin|'
        'Wrocław University of Technology|Medical University of Bialystok|'
        'Medical University of Gdansk|Medical University of Lodz|Medical University of Lublin|'
        'Medical University of Silesia|Medical University of Warsaw|'
        'Pomeranian Medical University in Szczecin|Poznan University of Medical Sciences|'
        'Wroclaw Medical University|Cracow University of Economics|'
        'Poznan University of Economics and Business|University of Economics in Katowice|'
        'Warsaw School of Economics|Wroclaw University of Economics|'
        'Poznań University of Life Sciences|University of Agriculture in Krakow|'
        'University of Life Sciences in Lublin|Warsaw University of Life Sciences – SGGW|'
        'Wrocław University of Environmental and Life Sciences|'
        'Jan Długosz University in Częstochowa|Maria Grzegorzewska University|'
        'Pedagogical University of Cracow|Pomeranian University in Słupsk|'
        'Siedlce University of Natural Sciences and Humanities|Gdynia Maritime University|'
        'Maritime University of Szczecin|Military University of Technology'),
}


if __name__ == '__main__':
    run(SOURCES, 'poland_nawa_self', {'poland': 'PL'},
        'NAWA government directory university, technical, medical, economics, '
        'agriculture, pedagogy and selected maritime/military university entries; '
        'exact whole signed name plus unique education entity only. Medical '
        'colleges/faculties within Jagiellonian, Nicolaus Copernicus and Warmia '
        'institutions excluded from university-self rules. No hospital, department, '
        'campus, successor, translation or historical merger inference. Current '
        'directory corroborates entity identity, not historical accreditation. '
        'Nonmatching spelling/location/name variants remain unresolved. Countries '
        'and reviewed Nature records preserved.')
