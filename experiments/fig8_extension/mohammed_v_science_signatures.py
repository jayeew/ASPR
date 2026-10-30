"""Resolve explicit Mohammed V science-faculty signatures using its official directory."""
from __future__ import annotations

from . import physics_university_signatures as worker
from .supplement_queue import ROOT

URL = 'https://www.fsr.ac.ma/content/pr%C3%A9sentation-de-la-facult%C3%A9-des-sciences-de-rabat'


def run() -> None:
    worker.FOLDER = ROOT / 'supplement/official_entities/mohammed_v_science'
    worker.SPECS = [('https://openalex.org/I126477371', 'Mohammed V University', URL, [
        'faculty of sciences, mohammed v university in rabat, rabat 10040, morocco',
        'faculty of sciences, university mohammed v, bp 1014, rabat, morocco',
        'faculté des sciences, université mohammed v, rabat, morocco',
    ])]
    worker.run({
        'sources': [URL], 'access': 'Official faculty page rendered with web tool',
        'locator': 'Présentation de la Faculté des Sciences de Rabat, opening paragraph and Contactez-nous',
        'findings': 'Faculty identifies itself as an institution of Université Mohammed V and gives Rabat BP 1014 address; six teaching/research departments include physics.',
        'scope': 'Only three reviewed whole science-faculty signatures explicitly naming Mohammed V and Rabat, from the current institution queue. Existing registry entity I126477371 is uniquely Mohammed V University, education, MA.',
        'temporal_scope': 'Official current faculty identity corroborates explicit university names in original 2023–2025 signatures, not historical affiliate ownership inference.',
        'excluded': 'Mohammed VI institutions, hospitals, joint centers, laboratory acronyms, numerically contaminated text and all other unsigned relationships. No country changes.',
        'reviewed_at': '2026-10-24',
    })


if __name__ == '__main__':
    run()
