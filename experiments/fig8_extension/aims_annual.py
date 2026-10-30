"""Use matched, dated AIMS native publishing statements as annual working rules."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict

from .aims_native_collect import FOLDER
from .cell_press_annual_cache import apply_proposals
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    sources = json.loads((FOLDER / 'qualification_sources.json').read_text())
    db = connect(); groups = defaultdict(dict)
    for doi in sources:
        papers = db.execute('SELECT work_id,source_id,journal FROM papers WHERE doi=?', (doi,)).fetchall()
        if len(papers) != 1:
            continue
        wid, sid, journal = papers[0]
        for payload, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,)):
            e = json.loads(payload); online = e.get('online_date', '')
            if (e.get('family') != 'publisher_aims' or e.get('identity') != 'matched'
                    or e.get('date_field') != 'onlineDate' or e.get('publishers') != ['AIMS Press']
                    or len(online) != 10 or int(online[:4]) not in {2023, 2024, 2025}):
                continue
            raw_path = (ROOT / 'fields/qualification' / e['receipt']).resolve()
            raw = gzip.decompress(raw_path.read_bytes())
            data = json.loads(raw)['data']['articles']
            matching = [a for a in data if (a.get('doi') or '').casefold() == doi
                        and a.get('journalMeta', {}).get('publisherNameEn') == 'AIMS Press'
                        and (a.get('onlineDate') or '')[:10] == online]
            if len(matching) != 1:
                continue
            groups[(sid, int(online[:4]), journal)][doi] = {
                'doi': doi, 'publication_date': online, 'publisher': 'AIMS Press',
                'statement': 'journalMeta.publisherNameEn=AIMS Press; article onlineDate=' + online,
                'raw_receipt': str(raw_path.relative_to(ROOT.resolve())),
                'raw_sha256': hashlib.sha256(raw).hexdigest(), 'source_url': e['url'],
                'locator': 'data.articles[doi].journalMeta.publisherNameEn + onlineDate',
                'issns': e['issns'], 'evidence_id': e['id']}
    db.close(); rules = []; unknown = []
    for (sid, year, journal), articles in groups.items():
        proofs = sorted(articles.values(), key=lambda v: (v['publication_date'], v['doi']))
        if len(proofs) < 2:
            unknown.append({'source_id': sid, 'year': year, 'reason': 'Only one matched dated native publisher statement', 'proofs': proofs})
            continue
        first, last = proofs[0], proofs[-1]
        rules.append({'source_id': sid, 'year': year, 'journal': journal, 'label': 'False',
            'publisher': 'AIMS Press', 'source_url': first['source_url'], 'raw_receipt': first['raw_receipt'],
            'raw_sha256': first['raw_sha256'], 'first_dated_proof': first, 'last_dated_proof': last,
            'matched_article_proofs': len(proofs),
            'scope': 'Annual working rule from at least two distinct DOI/title/ISSN-matched native AIMS article publishing statements in this online year. No cross-year extrapolation; reopen on dated transfer or conflicting publisher evidence.'})
    folder = FOLDER / 'annual'; folder.mkdir(exist_ok=True)
    atomic_json(folder / 'review.json', {'journal_year_groups': len(groups), 'proposals': len(rules),
                                       'network_requests': 0, 'article_statement_scope': True})
    return apply_proposals(rules, folder, unknown, 'dated_aims_native_publishing_statements',
                           'Matched native article onlineDate and explicit AIMS Press publisher', 'aims_native_annual')


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
