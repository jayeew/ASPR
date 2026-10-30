"""Resolve reviewed standalone Milan physics signatures; preserve joint INFN cases."""
from __future__ import annotations
from . import physics_university_signatures as worker
from .supplement_queue import ROOT

NAMES = ['dipartimento di fisica, universita’ di milano, via celoria 16, 20133, milano, mi, italy', 'dipartimento di fisica, università degli studi di milano, 20133 milano, italy', 'dipartimento di fisica, università degli studi di milano, i-20133 milano, italy', 'dipartimento di fisica, università degli studi di milano, milan, italy', 'dipartimento di fisica, università degli studi di milano, via celoria 16, 20133 milano, italy', 'dipartimento di fisica, università degli studi di milano, via celoria 16, 20133, milano, italy', 'dipartimento di fisica, università degli studi di milano, via celoria 16, i-20133 milano, italy', 'dipartimento di fisica, università degli studi di milano, via celoria 16, i-20133, milano, italy', 'dipartimento di fisica, università degli studi di milano, via celoria 16, milano 20133, italy', 'dipartimento di fisica, università di milano, i-20133 milano, italy', 'dipartimento di fisica, università di milano, milan italy', 'dipartimento di fisica, università di milano, milano, italy', 'dipartimento di fisica, università di milano, via celoria 16, 20133, milan, italy', 'dipartimento di fisica, università di milano, via celoria 16, 20133, milano mi, italy', 'dipartimento di fisica, università di milano, via celoria 16, 20133, milano, mi, italy', 'dipartimento di fisica, università di milano, via celoria 16, i-20133, milano, italy', 'dipartimento di fisica, universitá degli studi di milano, milan, italy', 'dipartimento di fisica, universitá degli studi di milano, via celoria 16, i-20133 milano, italy', 'dipartimento di fisica, universitá di milano, i-20133 milano, italy']
URL = 'https://fisica.unimi.it/system/files/2025-06/Piano%20Triennale%20Fisica%202025-27_0.pdf'


def run() -> None:
    worker.FOLDER = ROOT / 'supplement/official_entities/milan_physics'
    worker.SPECS = [('https://openalex.org/I189158943', 'University of Milan', URL, NAMES)]
    worker.run({
        'sources': [URL, 'https://www.unimi.it/it/ugov/ou-structure/dipartimento-di-fisica-aldo-pontremoli'],
        'access': 'Official 2025-27 departmental plan PDF read with web tool, page 2; official directory search-index text reviewed, direct directory returned 403, not retried.',
        'locator': 'PDF Missione del Dipartimento; university directory department address',
        'findings': 'University of Milan physics department reconstituted in 2012, named Aldo Pontremoli in 2017; directory gives Via Celoria 16. INFN section hosted by agreement has its own staff and is not treated as university-owned.',
        'scope': '19 individually reviewed whole signatures explicitly naming Milan university and physics department. No arbitrary substring assignment.',
        'excluded': 'Milan-Bicocca, joint INFN signatures, malformed names and conflicting Celoria 6 addresses remain outside this rule. Countries unchanged.',
        'temporal_scope': '2025 official plan corroborates department history and explicit 2023-2025 paper bylines; plan approval/date placeholders are not treated as approval evidence.',
        'reviewed_at': '2026-10-24',
    })


if __name__ == '__main__':
    run()
