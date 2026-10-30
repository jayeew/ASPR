"""Merge explicit PubMed electronic dates with existing primary research evidence."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .cached_date_apply import run as apply_dates
from .fig8b_qualification_decide import NONPRIMARY, SPECIAL, key
from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json


def run(folder: Path | None = None) -> dict:
    folder = folder if folder is not None else ROOT / 'supplement/pubmed_date_followup'
    review = json.loads((folder / 'date_review.json').read_text())
    rule = {'source_url': 'https://www.nlm.nih.gov/bsd/licensee/elements_article_source.html',
            'locator': 'ArticleDate DateType Electronic; Print-Electronic and Electronic-eCollection examples',
            'scope': 'Fixed DOI/PMID cohort in adjacent scope.json; exact DOI, title and journal identity; one valid explicit electronic date',
            'effect': 'Date evidence only; does not establish original research type. Preserve prior evidence and conflicts.',
            'excluded_fields': ['PubDate', 'PubmedPubDate', 'DateCompleted', 'DateRevised'],
            'family': 'pubmed_publisher_electronic_date'}
    atomic_json(folder / 'date_rule.json', rule)
    sources = {}
    held = []
    for row in review['records']:
        if row['status'] != 'matched_electronic_date_requires_primary_evidence_merge':
            continue
        flags = {key(t) for t in row['publication_types']} & (NONPRIMARY | SPECIAL | {'retractedpublication', 'expressionofconcern'})
        if flags:
            held.append({'doi': row['doi'], 'flags': sorted(flags), 'receipt': row['receipt']})
            continue
        identity = row['identity']
        sources[row['doi']] = [{'family': rule['family'], 'title': identity['title'][0],
            'issns': identity['issns'], 'journal_titles': identity['journal_titles'],
            'kind': 'unspecified', 'categories': [], 'online_date': row['electronic_dates'][0],
            'date_precision': 'day', 'receipt': '../../' + row['receipt'],
            'url': 'https://pubmed.ncbi.nlm.nih.gov/' + row['pmid'] + '/',
            'pmid': row['pmid'], 'date_rule_receipt': '../../' + str((folder / 'date_rule.json').relative_to(ROOT)),
            'native_date_locator': './MedlineCitation/Article/ArticleDate[@DateType="Electronic"]'}]
    atomic_json(folder / 'type_flags_held.json', {'records': held, 'eligibility_unchanged_for_held': True})
    result = apply_dates({r['doi'] for r in review['records']}, folder, explicit_sources=sources)
    with (ROOT / 'supplement/checkpoint.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        db = connect()
        for row in review['records']:
            receipt = row.get('receipt', str((folder / 'date_review.json').relative_to(ROOT)))
            for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (row['doi'],)).fetchall():
                if db.execute("SELECT 1 FROM attempts WHERE task_id=? AND source='pubmed_electronic_date_batch' AND receipt=?", (tid, receipt)).fetchone():
                    continue
                flags_held = row['doi'] in {r['doi'] for r in held}
                outcome = 'pubmed_type_flags_require_primary_review' if flags_held else row['status']
                attempt(db, tid, 'pubmed_electronic_date_batch', outcome, receipt,
                        {'doi': row['doi'], 'pmid': row['pmid'], 'date_only': True,
                         'type_flags_held': flags_held,
                         'rule': str((folder / 'date_rule.json').relative_to(ROOT)), 'final_task_closure': False})
        db.commit()
        db.close()
    return result


if __name__ == '__main__':
    run()
