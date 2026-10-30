"""Apply individually reviewed Fudan unit signatures supported by official pages."""
from __future__ import annotations

from . import physics_university_signatures as worker
from .supplement_queue import ROOT

DIRECTORY = 'https://www.fudan.edu.cn/en/142/list.htm'
PHYSICS = 'https://phys.fudan.edu.cn/eng/wbout/list.htm'
NAMES = [
    'school of life sciences, fudan university, shanghai, 200433 china',
    'department of chemistry, fudan university, shanghai, 200433, china',
    'department of physics, fudan university, shanghai 200433, china',
    'department of materials science, fudan university, shanghai, 200433 p. r. china',
    'department of physics, fudan university, shanghai, 200433 china',
    'department of physics, fudan university, shanghai, 200433 pr china',
    'department of physics, fudan university, shanghai, 200438 p. r. china',
    'school of computer science, fudan university, shanghai, 200433 china',
    'school of life sciences, fudan university shanghai china',
    'school of life sciences, fudan university, shanghai, 200438 china',
    'school of public health, fudan university, room 435, bld #8, 130 dongan road, xuhui district, shanghai, 200032 china',
    'department of physics and state key laboratory of surface physics, fudan university, shanghai 200433, china',
    'state key laboratory of surface physics and department of physics, fudan university, shanghai 200438, china',
    'state key laboratory of surface physics and department of physics, fudan university, shanghai 200433, china',
    "department of physics and state key laboratory of surface physics, fudan university, shanghai 200433, people's republic of china",
    'state key laboratory of surface physics, department of physics, fudan university, shanghai 200438, china',
]


def run() -> None:
    worker.FOLDER = ROOT / 'supplement/official_entities/fudan_departments'
    worker.SPECS = [('https://openalex.org/I24943067', 'Fudan University', DIRECTORY, NAMES)]
    worker.run({
        'sources': [DIRECTORY, PHYSICS],
        'access': 'Official directory and physics department About Us opened and read with web tool',
        'locator': 'Schools and Departments entries; Physics About Us Present Status paragraph naming three department laboratories',
        'findings': 'Directory confirms the named standalone departments/schools. Physics page explicitly states that the department runs State Key Laboratory of Surface Physics.',
        'scope': 'Sixteen fixed complete signatures individually reviewed from current paper slots and institution tasks.',
        'rule_scope': 'Exact whole signatures explicitly naming Fudan and listed school/department, optionally Surface Physics laboratory expressly run by its physics department. Signed country unchanged.',
        'excluded': 'Westlake multi-institution signature, hospitals including Zhongshan, smart drug delivery laboratory, collaborative centers, Institute of Computational Physical Sciences and unreviewed laboratory combinations are retained unresolved.',
        'temporal_scope': 'Contemporary university signatures in 2023–2025 corroborated by official departmental identity/history; no retrospective subsidiary transfer inference.',
        'reviewed_at': '2026-10-24',
    })


if __name__ == '__main__':
    run()
