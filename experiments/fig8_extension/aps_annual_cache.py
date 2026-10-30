"""Reuse dated APS native publication and copyright metadata for annual labels."""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date

from .aps_authors import journal_issns
from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_build import key, plain
from .supplement_queue import ROOT, connect
from .three_journal_apply import read_csv


def copyright_year(article: dict) -> int | None:
    rights = article.get('rights', {})
    try:
        year = date.fromisoformat(article.get('date', '')).year
    except ValueError:
        return None
    if (year not in {2023, 2024, 2025} or rights.get('copyrightYear') != year
            or article.get('publisher', {}).get('name') != 'APS'
            or 'American Physical Society' not in rights.get('rightsStatement', '')
            or not (any(p.get('name') == 'American Physical Society' for p in rights.get('copyrightHolders', []))
                    or (rights.get('creativeCommons') is True and rights.get('rightsStatement') == 'Published by the American Physical Society'))):
        return None
    return year


def run(selected_pairs: set[tuple[str, int]] | None = None,
        selected_receipts: set[str] | None = None) -> dict:
    folder = ROOT / 'supplement/aps_annual_cache'
    folder.mkdir(exist_ok=True)
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = connect()
    candidates = {}
    for doi, sid, title, journal in db.execute("SELECT doi,source_id,title,journal FROM papers WHERE doi LIKE '10.1103/%'"):
        if selected_pairs is None or sid in {pair[0] for pair in selected_pairs}:
            candidates.setdefault(doi, []).append((sid, title, journal))
    db.close()
    annual = {(r['source_id'], int(r['year'])): r for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
    rules_path = folder / 'rules.json'
    rules = {(r['source_id'], r['year']): r for r in json.loads(rules_path.read_text())} if rules_path.exists() else {}
    unmatched = []
    for path in sorted((ROOT / 'fields/qualification/aps_all_raw').glob('*.json.gz')):
        if selected_receipts is not None and str(path.relative_to(ROOT)) not in selected_receipts:
            continue
        raw = gzip.decompress(path.read_bytes())
        record = json.loads(raw)
        if record.get('http_status') != 200:
            continue
        digest = hashlib.sha256(raw).hexdigest()
        for article in record.get('response', {}).get('data', []):
            doi = article.get('identifiers', {}).get('doi', '').lower()
            year = copyright_year(article)
            if doi not in candidates or year is None:
                continue
            matched = [(sid, journal) for sid, title, journal in candidates[doi]
                       if key(title) == key(plain(article.get('title', {}).get('value', '')))
                       and (key(journal) == key(article.get('journal', {}).get('name', ''))
                            or bool(registry[sid] & set(journal_issns(article))))]
            sids = {p[0] for p in matched}
            if len(sids) != 1:
                unmatched.append({'doi': doi, 'status': 'candidate_title_or_journal_identity_not_unique'})
                continue
            sid, journal = matched[0]
            if (sid, year) not in annual or (selected_pairs is not None and (sid, year) not in selected_pairs):
                continue
            proof = {'doi': doi, 'publication_date': article['date'], 'native_journal_id': article['journal']['id'],
                     'validated_journal_id_issns': journal_issns(article), 'registry_issns': sorted(registry[sid]),
                     'raw_receipt': str(path.relative_to(ROOT)), 'raw_sha256': digest,
                     'source_url': record['url'], 'copyright': article['rights'],
                     'locator': 'response.data[identifiers.doi=' + doi + '].rights'}
            if (sid, year) not in rules:
                rules[(sid, year)] = {'source_id': sid, 'year': year, 'label': 'False', 'journal': journal,
                    'publisher': 'American Physical Society', 'source_url': record['url'],
                    'raw_receipt': proof['raw_receipt'], 'raw_sha256': digest,
                    'scope': 'Working annual rule anchored to native APS article publication dates and same-year rights record naming APS as copyright holder or explicit Creative Commons publishing body; exact candidate DOI/title/journal; no extrapolation across years; reopen on transfer evidence',
                    'first_dated_proof': proof, 'last_dated_proof': proof}
            rule = rules[(sid, year)]
            if proof['publication_date'] < rule['first_dated_proof']['publication_date']:
                rule['first_dated_proof'] = proof
            if proof['publication_date'] > rule['last_dated_proof']['publication_date']:
                rule['last_dated_proof'] = proof
    return apply_proposals(list(rules.values()), folder, unmatched,
                           rule_status='dated_native_aps_publication_copyright',
                           reason='Native APS publication and same-year APS copyright with exact DOI/title/journal identity',
                           route='native_aps_copyright', apply_pairs=selected_pairs)


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
