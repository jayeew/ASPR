"""Resolve explicit SJTU signatures; standalone TDLI identity remains separately scoped."""
from __future__ import annotations

from . import physics_university_signatures as worker
from .supplement_queue import ROOT

URL = 'https://en.sjtu.edu.cn/news-events/notice/1823'
NAMES = [
    'tsung-dao lee institute & school of physics and astronomy, shanghai jiao tong university , shanghai 200240 ,',
    'tsung-dao lee institute (tdli) & school of physics and astronomy (spa), shanghai jiao tong university (sjtu), shanghai, 200240, china',
    'tsung-dao lee institute (tdli) and school of physics and astronomy (spa), shanghai jiao tong university (sjtu), shanghai 200240, china',
    'tsung-dao lee institute, shanghai jiao tong university, shanghai 200240',
    'tsung-dao lee institute, shanghai jiao tong university, shanghai, 200240 china',
    'tsung-dao lee institute, shanghai jiao tong university, shanghai, 201210 china',
]


def run() -> None:
    worker.FOLDER = ROOT / 'supplement/official_entities/sjtu_tdli_explicit'
    worker.SPECS = [('https://openalex.org/I183067930', 'Shanghai Jiao Tong University', URL, NAMES)]
    worker.run({
        'sources': [URL], 'access': 'Official university article read with web tool',
        'source_publication_date': '2024-09-23',
        'locator': 'Memorial Service for Professor Tsung-Dao Lee; institute of SJTU and institute at SJTU descriptions',
        'findings': 'University article identifies TDLI as its institute, corroborating the university explicitly named in the six reviewed whole paper signatures.',
        'scope': 'Only the six explicit SJTU whole signatures. This is declared university identity, not an inference from ROR or from the author nationality.',
        'excluded': 'Bare TDLI signatures remain pending a separate institute entity/ownership representation. Mixed Nordita/KTH/Stockholm signature and mojibake text are not accepted by this rule. Countries unchanged, including signatures without country.',
        'temporal_scope': '2024 primary institutional source plus explicit 2023-2025 original-paper names; no expansion to new geographic branches.',
        'reviewed_at': '2026-10-24',
    })


if __name__ == '__main__':
    run()
