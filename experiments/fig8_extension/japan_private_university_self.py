"""Exact university names corroborated by the JASSO private-university directory."""
from __future__ import annotations

import json

from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .uk_devolved_university_self import run as apply_names

FOLDER = ROOT / 'supplement/official_entities/japan_private_self'
PREFECTURES = set('Hokkaido Aomori Iwate Miyagi Akita Yamagata Fukushima Ibaraki Tochigi '
                  'Gunma Saitama Chiba Tokyo Kanagawa Niigata Toyama Ishikawa Fukui '
                  'Yamanashi Nagano Gifu Shizuoka Aichi Mie Shiga Kyoto Osaka Hyogo Nara '
                  'Wakayama Tottori Shimane Okayama Hiroshima Yamaguchi Tokushima Kagawa '
                  'Ehime Kochi Fukuoka Saga Nagasaki Kumamoto Oita Miyazaki Kagoshima Okinawa'.split())


def run() -> None:
    receipt = json.loads((FOLDER / 'current_directory_receipt.json').read_text())
    pages = json.loads((FOLDER / 'pages.json').read_text())
    rows = []
    for page, text in enumerate(pages, 1):
        if 'List of Private Universities using EJU' not in text:
            raise ValueError('Unexpected directory section')
        for line in text.splitlines():
            prefecture, separator, name = line.strip().partition(' ')
            if prefecture in PREFECTURES and separator and name != 'region':
                rows.append({'name': name.strip(), 'prefecture_label': prefecture, 'page': page})
    names = sorted({r['name'] for r in rows})
    atomic_json(FOLDER / 'directory_extraction.json', {'rows': rows, 'reviewed_names': len(names),
                'raw_receipt': receipt['raw_receipt'], 'scope': 'University-self identity corroboration only; not a claim about 2023–2025 EJU participation or accreditation'})
    apply_names({'jasso_directory': (receipt['url'], '|'.join(names))}, 'japan_private_self', {'jasso_directory': 'JP'},
                'Exact whole university self names only; no hospitals, subdivisions, fuzzy name aliases, '
                'predecessors, successor mergers or country replacement. Directory date is 2026; '
                'it corroborates named university identity, not historical EJU participation.',
                evidence_access='Retained JASSO PDF and extracted page text; current_directory_receipt.json and pages.json',
                evidence_locator='Private Universities table, exact whole Name of University cell; page recorded in directory_extraction.json')
    proof = FOLDER / 'jasso_directory_review.json'
    review = json.loads(proof.read_text())
    review.update(raw_http_retained=True, raw_receipt=receipt['raw_receipt'])
    atomic_json(proof, review)
    atomic_json(FOLDER / 'source_status.json', {'status': 'current_official_pdf_processed',
                'current_receipt': 'current_directory_receipt.json',
                'prior_non_pdf_response_preserved': 'directory_receipt.json', 'final_task_closure': False})


if __name__ == '__main__':
    run()
