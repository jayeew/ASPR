"""Resolve reviewed faculty/school bylines with explicit university names."""
from __future__ import annotations

from . import physics_university_signatures as worker
from .supplement_queue import ROOT

SPECS = [('https://openalex.org/I165779595', 'The University of Melbourne', 'https://physics.unimelb.edu.au/about', ['school of physics, the university of melbourne, melbourne, vic 3010 australia', 'school of physics, university of melbourne, 3010 parkville, victoria australia', 'school of physics, university of melbourne, melbourne, vic 3010 australia', 'school of physics, university of melbourne, melbourne, vic australia', 'school of physics, university of melbourne, melbourne, victoria 3010, australia', 'school of physics, university of melbourne, melbourne, victoria, australia', 'school of physics, university of melbourne, parkville, vic 3010 australia', 'school of physics, university of melbourne, parkville, vic 3010, australia', 'school of physics, university of melbourne, parkville, vic 3010, australia.', 'school of physics, university of melbourne, parkville, victoria 3010, australia', 'school of physics, university of melbourne, parkville, victoria, australia', 'school of physics, university of melbourne, tin alley and swanston street, melbourne, 3010, victoria, australia', 'school of physics, university of melbourne, victoria 3010, australia', 'school of physics, university of melbourne, victoria, australia']), ('https://openalex.org/I74788687', 'Comenius University Bratislava', 'https://fmph.uniba.sk/en', ['faculty of mathematics, physics and informatics, comenius university in bratislava, 842 48 bratislava, slovakia', 'faculty of mathematics, physics and informatics, comenius university in bratislava, mlynská dolina f1, 842 48, bratislava, slovakia', 'faculty of mathematics, physics and informatics, comenius university, 842 48, bratislava, slovakia', 'faculty of mathematics, physics and informatics, comenius university, 84248, bratislava, slovakia', 'faculty of mathematics, physics and informatics, comenius university, bratislava slovak republic', 'faculty of mathematics, physics and informatics, comenius university, bratislava, slovak republic', 'faculty of mathematics, physics and informatics, comenius university, bratislava, slovakia', 'faculty of mathematics, physics and informatics, comenius university, mlynská dolina f1, 842 48 bratislava, slovakia'])]


def run() -> None:
    worker.FOLDER = ROOT / 'supplement/official_entities/melbourne_comenius'
    worker.SPECS = SPECS
    worker.run({
        'sources': [s[2] for s in SPECS], 'access': 'Both official faculty/school pages opened and read with web tool',
        'locator': 'Melbourne School of Physics About introduction; Comenius FMPI official site identity and university navigation',
        'findings': 'The School of Physics explicitly belongs to the University of Melbourne; Faculty of Mathematics, Physics and Informatics explicitly identifies Comenius University Bratislava.',
        'scope': 'Fixed individually reviewed complete bylines naming school/faculty and university. No substring matching.',
        'excluded': 'Combined OzGrav/ARC center, malformed university names, email-contaminated faculty signatures and all hospitals/branches remain outside this rule.',
        'temporal_scope': 'Official current department identity corroborates university explicitly named in 2023–2025 paper signatures; no retrospective joint-center ownership assignment.',
        'reviewed_at': '2026-10-24',
    })


if __name__ == '__main__':
    run()
