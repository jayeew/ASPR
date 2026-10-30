"""Apply explicit document-abstract and Polymer Journal original-article policies."""
from __future__ import annotations

import json
from collections import Counter

from .fig8b_qualification_decide import journal_research_format
from .qa_type_apply import run as apply_review
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/native_abstract_polymer_types'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER / 'status.json').exists():
        return
    if not (FOLDER / 'review.json').exists():
        db = connect(); papers = {}; checks = []; sampled = set()
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        query = '''SELECT work_id,payload FROM current_evidence_index WHERE kind='primary'
            AND (payload LIKE '%"kind": "abstract"%' OR
                 (payload LIKE '%"Polymer Journal"%' AND payload LIKE '%"nature_page"%'))'''
        for wid, payload in db.execute(query):
            e = json.loads(payload)
            abstract = e.get('family') == 'archived_jats' and e.get('kind') == 'abstract'
            polymer = journal_research_format(e) and e.get('family') == 'nature_page' and 'Polymer Journal' in e.get('journal_titles', [])
            if e.get('identity') != 'matched' or not (abstract or polymer):
                continue
            row = db.execute('SELECT doi,title,journal,source_id,shard,status FROM papers WHERE work_id=?', (wid,)).fetchone()
            if not row:
                continue
            doi, title, journal, sid, shard, status = row
            paper = papers.setdefault(wid, {'work_id': wid, 'doi': doi, 'title': title,
                'journal_name': journal, 'source_id': sid, 'shard': shard, 'before_status': status, 'evidence': []})
            paper['evidence'].append(e)
            group = ('abstract' if abstract else 'polymer', sid, (e.get('online_date') or '')[:4])
            if group in sampled:
                continue
            native = native_metadata('fields/qualification/' + e['receipt'], doi)
            native_type = native.get('article_type') == 'abstract' if abstract else bool(
                {'Original Article', 'OriginalPaper'} & set(native.get('article_types', [])))
            if (doi not in {v.strip().lower() for v in native.get('doi', [])}
                    or not signature_match(paper, native, journals[sid]) or not native_type):
                raise ValueError('Native type or identity validation failed: ' + wid)
            sampled.add(group)
            checks.append({'work_id': wid, 'rule': group[0], 'receipt': e['receipt'],
                           'status': 'native_doi_title_journal_and_explicit_type_confirmed'})
        db.close()
        atomic_json(FOLDER / 'review.json', {'papers': list(papers.values()), 'native_checks': checks,
            'rules': [
                {'source_url': 'https://pmc.ncbi.nlm.nih.gov/tagging-guidelines/article/dobs/',
                 'locator': 'Article Types / Abstracts',
                 'interpretation': 'The document itself is an abstract; exclude from the full original-research-paper denominator. Embedded abstract elements are unaffected.',
                 'scope': 'Identity-matched archived_jats kind abstract; retain all other conflicting evidence.'},
                {'source_url': 'https://www.nature.com/pj/authors-and-referees/authors',
                 'supporting_source_url': 'https://www.nature.com/pj/journal-information',
                 'interpretation': 'The journal distinguishes original research articles from reviews, notes and other formats.',
                 'scope': 'Native nature_page Original Article or OriginalPaper, exact Polymer Journal title and ISSN 0032-3896 or 1349-0540; no overriding content or source holds.'}],
            'status_counts': dict(Counter(p['before_status'] for p in papers.values())),
            'scope': 'All current statuses with the exact source types, not a new literature search; window eligibility and conflicts recalculated from all evidence.',
            'paper_network_requests': 0})
    result = apply_review(FOLDER.name)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
