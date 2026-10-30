"""Use dated native named imprints with independent publication-date corroboration."""
from __future__ import annotations

import calendar
import gzip
import hashlib
import json
from collections import defaultdict

import pyarrow.dataset as ds
from lxml import etree

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_decide import electronic_date, imprint_dates
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv

ENTITIES = {'American Society for Microbiology', 'American Society for Microbiology (ASM)', 'PeerJ, Inc', 'PeerJ Inc.'}
CONTEXT = 'supplement/publisher_identity_context.json'
ADDITIONAL_ENTITIES = {'Impact Journals, LLC', 'Dove Press', 'Life Science Alliance LLC'}
ADDITIONAL_CONTEXT = 'supplement/independent_imprint_context.json'
ENTITIES |= ADDITIONAL_ENTITIES
SPECIALIST_ENTITIES = {'Ivyspring International Publisher', 'Spandidos Publications', 'Company of Biologists'}
SPECIALIST_CONTEXT = 'supplement/specialist_imprint_context.json'
ENTITIES |= SPECIALIST_ENTITIES
SOCIETY_ENTITIES = {'American Society for Clinical Investigation', 'Pensoft Publishers',
                    'International Union of Crystallography', 'The Rockefeller University Press',
                    'American Society for Cell Biology'}
SOCIETY_CONTEXT = 'supplement/society_imprint_context.json'
ENTITIES |= SOCIETY_ENTITIES
MEDICAL_ENTITIES = {'Wolters Kluwer -- Medknow Publications', 'Wolters Kluwer Health',
                    'KeAi Publishing', 'BMJ Publishing Group', 'Baishideng Publishing Group Inc', 'De Gruyter'}
MEDICAL_CONTEXT = 'supplement/medical_imprint_context.json'
ENTITIES |= MEDICAL_ENTITIES
REMAINING_ENTITIES = {'Copernicus Publications', 'The Royal Society', 'AME Publications'}
REMAINING_CONTEXT = 'supplement/remaining_named_imprint_context.json'
ENTITIES |= REMAINING_ENTITIES
COPERNICUS_ENTITIES = {'Copernicus Publications (Copernicus GmbH)'}
ENTITIES |= COPERNICUS_ENTITIES
SPJ_ENTITIES = {'AAAS Science Partner Journal Program'}
ENTITIES |= SPJ_ENTITIES
PRESS_ENTITIES = {'SAGE Publications', 'Taylor & Francis', 'Cold Spring Harbor Laboratory Press',
                  'American Association for Cancer Research',
                  'Association for Research in Vision and Ophthalmology'}
ENTITIES |= PRESS_ENTITIES
OPEN_PUBLISHER_ENTITIES = {'Beilstein-Institut', 'Microbiology Society', 'Termedia Publishing',
                           'JMIR Publications Inc.', 'Tech Science Press'}
ENTITIES |= OPEN_PUBLISHER_ENTITIES
CUREUS_ENTITIES = {'Cureus Inc.'}
ENTITIES |= CUREUS_ENTITIES
EDP_ENTITIES = {'EDP Sciences'}
ENTITIES |= EDP_ENTITIES


def date_supported(native: dict, online: str, primary: list[dict]) -> bool:
    explicit, _ = electronic_date(native['dates'])
    return explicit == online or (online in imprint_dates(native['dates']) and any(
        e.get('identity') == 'matched' and e.get('online_date') == online and e.get('receipt') for e in primary))


def native_imprint(raw: bytes, doi: str, family: str = 'archived_jats') -> dict:
    if family == 'publisher_edp_native':
        from .edp_rich_candidates import extract
        native = extract(raw, doi)
        if native is None:
            raise ValueError('No unique EDP native DOI')
        return native
    if family == 'publisher_copernicus':
        from .copernicus_bulk import parse_article
        document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
        tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
        articles = [tree] if etree.QName(tree).localname == 'article' else tree.xpath('//*[local-name()="metadata"]/*[local-name()="article"]')
        matches = [parse_article(a) for a in articles if doi.lower() in [d.lower() for d in a.xpath('.//*[local-name()="article-id" and @pub-id-type="doi"]/text()')]]
        if len(matches) != 1:
            raise ValueError('No unique Copernicus imprint DOI')
        native = matches[0]
        months = {v.casefold():str(i) for i,v in enumerate(calendar.month_name) if v}
        months.update({v.casefold():str(i) for i,v in enumerate(calendar.month_abbr) if v})
        native['dates'] = [{**d, 'month':months.get(str(d.get('month','')).casefold(),d.get('month',''))} for d in native['dates']]
        return native
    return jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)


def run(selected_source_ids: set[str] | None = None) -> dict:
    folder = ROOT / 'supplement/additional_dated_imprints'
    folder.mkdir(exist_ok=True)
    rows = ds.dataset(ROOT / 'fields/qualification/decisions', format='parquet').to_table(
        columns=['work_id','doi','title','source_id','online_date','online_year','nature_portfolio_at_verified_online_year'],
        filter=ds.field('eligibility_status') == 'included_verified_research').to_pylist()
    registry = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    groups = defaultdict(list)
    for row in rows:
        if row['nature_portfolio_at_verified_online_year'] not in {'True','False'} and (selected_source_ids is None or row['source_id'] in selected_source_ids):
            groups[(row['source_id'], row['online_year'])].append(row)
    path = folder / 'rules.json'
    rules = {(r['source_id'], r['year']): r for r in json.loads(path.read_text())} if path.exists() else {}
    db = connect()
    for (sid, year), candidates in groups.items():
        proofs = []
        ordered = sorted(candidates, key=lambda r: r['online_date'])
        for sequence in [ordered, list(reversed(ordered))]:
            for row in sequence:
                primary = [json.loads(v[0]) for v in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],))]
                proof = None
                for e in primary:
                    if e.get('family') not in {'archived_jats', 'publisher_copernicus', 'publisher_edp_native'} or e.get('identity') != 'matched' or not set(e.get('publishers', [])) & ENTITIES:
                        continue
                    raw_path = ROOT / 'fields/qualification' / e['receipt']
                    raw = gzip.decompress(raw_path.read_bytes())
                    native = native_imprint(raw, row['doi'], e['family'])
                    native['issns'] = native['issn']
                    publishers = set(native['publisher'])
                    if not publishers or not publishers <= ENTITIES:
                        continue
                    if (row['doi'] not in {v.lower() for v in native['doi']} or not date_supported(native, row['online_date'], primary)
                            or not signature_match({'title':row['title'],'journal_name':registry[sid]['journal_name'],'source_id':sid}, native, set(json.loads(registry[sid]['issn'])))):
                        continue
                    date_evidence = []
                    for item in primary:
                        if item.get('identity') != 'matched' or item.get('online_date') != row['online_date'] or not item.get('receipt'):
                            continue
                        p = ROOT / 'fields/qualification' / item['receipt'];b = p.read_bytes()
                        if p.suffix == '.gz': b = gzip.decompress(b)
                        date_evidence.append({**item, 'checked_raw_sha256':hashlib.sha256(b).hexdigest()})
                    proof = {**row, 'native_family':e['family'], 'raw_receipt':str(raw_path.relative_to(ROOT)), 'raw_sha256':hashlib.sha256(raw).hexdigest(),
                             'source_url':e['url'], 'publishers':sorted(publishers), 'date_evidence':date_evidence,
                             'journal':registry[sid]['journal_name'], 'registry_issns':json.loads(registry[sid]['issn'])}
                    break
                if proof:
                    proofs.append(proof)
                    break
        if len(proofs) == 2 and year in {2023,2024,2025}:
            context = ADDITIONAL_CONTEXT if any(set(p['publishers']) & ADDITIONAL_ENTITIES for p in proofs) else CONTEXT
            if any(set(p['publishers']) & SPECIALIST_ENTITIES for p in proofs):
                context = SPECIALIST_CONTEXT
            if any(set(p['publishers']) & SOCIETY_ENTITIES for p in proofs):
                context = SOCIETY_CONTEXT
            if any(set(p['publishers']) & MEDICAL_ENTITIES for p in proofs):
                context = MEDICAL_CONTEXT
            if any(set(p['publishers']) & REMAINING_ENTITIES for p in proofs):
                context = REMAINING_CONTEXT
            if any(set(p['publishers']) & COPERNICUS_ENTITIES for p in proofs):
                context = 'supplement/copernicus_imprint_context.json'
            if any(set(p['publishers']) & SPJ_ENTITIES for p in proofs):
                context = 'supplement/aaas_spj_imprint_context.json'
            if any(set(p['publishers']) & PRESS_ENTITIES for p in proofs):
                context = 'supplement/press_society_imprint_context.json'
            if any(set(p['publishers']) & OPEN_PUBLISHER_ENTITIES for p in proofs):
                context = 'supplement/open_publisher_imprint_context.json'
            if any(set(p['publishers']) & CUREUS_ENTITIES for p in proofs):
                context = 'supplement/cureus_brand_context.json'
            if any(set(p['publishers']) & EDP_ENTITIES for p in proofs):
                context = 'supplement/edp_imprint_context.json'
            context_hash = hashlib.sha256((ROOT / context).read_bytes()).hexdigest()
            rules[(sid,year)] = {'source_id':sid,'year':year,'label':'False','journal':registry[sid]['journal_name'],
                'publisher':' | '.join(sorted({p for proof in proofs for p in proof['publishers']})),
                'source_url':proofs[0]['source_url'],'raw_receipt':proofs[0]['raw_receipt'],'raw_sha256':proofs[0]['raw_sha256'],
                'proofs':proofs,'context_receipt':context,'context_sha256':context_hash,
                'scope':'Same journal/year native named imprint with explicit online date or exact independently corroborated full publication date; not current-profile extrapolation. Records imprint and non-Portfolio membership, not historical group ownership. Reopen on contrary transfer evidence.'}
    db.close()
    return apply_proposals(list(rules.values()),folder,[],rule_status='additional_dated_native_imprint',
                           reason='Exact native named imprint and dated publication evidence, with official publisher identity context',route='additional_native_imprint', apply_pairs=set(groups) & set(rules))


if __name__ == '__main__':
    print(json.dumps(run()),flush=True)
