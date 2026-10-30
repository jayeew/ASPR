"""Evidence-bound updates of the current abstract/author tables, never fuzzy ownership."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_build import key, plain, pnas_journal_alias
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import address_country, country_names
from .supplement_queue import ROOT, connect
from .supplement_sources import jats_metadata, separate_affiliation_notes


def abstract_ok(value: str) -> bool:
    if len(value.strip()) < 80:
        return False
    paragraphs = [p.strip() for p in value.split('\n') if p.strip()]
    lowered = value.strip().casefold()
    return not lowered.startswith(('abstract not available', 'no abstract available', 'abstract unavailable')) and len(set(paragraphs)) == len(paragraphs)


@lru_cache(maxsize=1)
def explicit_country_codes() -> frozenset[str]:
    """Keep valid declared codes even when callers exclude ambiguous country names."""
    return frozenset(country_names().values())


def countries(affiliations: list[dict], names: dict[str, str]) -> list[str]:
    if not affiliations:
        return []
    output = set()
    for aff in affiliations:
        current = set()
        for country in aff['countries']:
            code = country['code'].upper()
            if code not in explicit_country_codes():
                code = names.get(normalized(country['text']), '')
            if code:
                current.add(code)
        if not current:
            explicit=address_country(aff['text'],names)
            if explicit:current.add(explicit)
            else:return []
        address = key(aff['text'])
        if 'CN' in current and any(t in address for t in ['hongkong', 'taiwan', 'macao', 'macau']):
            return []
        output.update(current)
    return sorted(output)


def load_index(root: Path, db: sqlite3.Connection) -> dict:
    q = root / 'fields/qualification'
    db.execute('CREATE TABLE IF NOT EXISTS source_metadata(receipt TEXT PRIMARY KEY,doi TEXT,kind TEXT,raw_hash TEXT,payload TEXT)')
    db.execute('CREATE INDEX IF NOT EXISTS source_metadata_doi ON source_metadata(doi)')
    seen = dict(db.execute('SELECT receipt,raw_hash FROM source_metadata'))
    count = Counter()
    paths = [*(q/'jats_parsed').glob('*.json'),*(q/'oai_parsed').glob('*.json'),*(q/'supplement_parsed').glob('*.json')]
    for i, path in enumerate(paths, 1):
        r = json.loads(path.read_text())
        if r.get('status') != 'parsed' or not r.get('raw_receipt'):
            continue
        receipt = str(path.relative_to(root))
        h = r.get('raw_sha256', '')
        if receipt in seen and seen[receipt] == h:
            continue
        p = r['parsed']
        is_jats = path.parent.name in ['jats_parsed','oai_parsed'] or r.get('source')=='pmc' or r.get('route')=='copernicus'
        if is_jats:
            if 'abstract' not in p or 'authors' not in p:
                raw = gzip.decompress((q / r['raw_receipt']).read_bytes())
                p = jats_metadata(raw + b'</article>')
            doi_values = p['doi']
        else:
            p = {**p, 'title': p['title'], 'authors': []}
            doi_values = p['doi']
        for doi in doi_values:
            payload = {**p, 'title': p['title'] if isinstance(p['title'], list) else [p['title']],
                       'issns': p.get('issns', p.get('issn', [])), 'journal_titles': p.get('journal_titles', []),
                       'evidence_receipt': 'fields/qualification/' + r['raw_receipt'], 'source_url': r.get('source_url',r['url'])}
            db.execute('INSERT OR REPLACE INTO source_metadata VALUES(?,?,?,?,?)',
                       (receipt, doi.lower(), 'jats' if is_jats else 'publisher', h, json.dumps(payload, ensure_ascii=False)))
            count['indexed'] += 1
        if i % 1000 == 0:
            db.commit()
            print('metadata sources indexed', i, flush=True)
    db.commit()
    return dict(count)


def deposited_metadata(root: Path, db: sqlite3.Connection) -> int:
    """Publisher-deposited affiliation text is separate from a verified complete byline."""
    db.execute('CREATE TABLE IF NOT EXISTS deposit_file_index (receipt TEXT PRIMARY KEY, stamp TEXT NOT NULL)')
    indexed_files = dict(db.execute('SELECT receipt,stamp FROM deposit_file_index'))
    count=0
    for path in (root/'fields/crossref_raw').glob('*.gz'):
        file_receipt = str(path.relative_to(root))
        stat = path.stat()
        stamp = f'{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}:deposit-parser-1'
        if indexed_files.get(file_receipt) == stamp:
            continue
        seen = dict(db.execute('SELECT receipt,raw_hash FROM source_metadata WHERE receipt GLOB ?',
                               (file_receipt + '#*',)))
        with gzip.open(path,'rt') as stream:record=json.load(stream)
        if record.get('status')!='success':continue
        for item in record['response']['message']['items']:
            receipt=str(path.relative_to(root))+'#'+item['DOI'].lower()
            raw_hash=hashlib.sha256(json.dumps(item,sort_keys=True).encode()).hexdigest()
            if seen.get(receipt)==raw_hash:continue
            authors=[]
            for author in item.get('author',[]):
                name=' '.join(filter(None,[author.get('given'),author.get('family')])) or author.get('name','')
                if not name:continue
                affiliations=[]
                for aff in author.get('affiliation',[]):
                    identifiers=[{'type':i.get('id-type',''),'value':i.get('id','')} for i in aff.get('id',[]) if isinstance(i,dict) and i.get('asserted-by')=='publisher']
                    affiliations.append({'text':aff.get('name',''),'countries':[],'identifiers':identifiers,'names':[aff.get('name','')]})
                authors.append({'name':name,'nonpersonal':False,'orcid':author.get('ORCID',''),'affiliations':affiliations,'unresolved_affiliation_refs':[]})
            payload={'title':item.get('title',[]),'issns':item.get('ISSN',[]),'journal_titles':item.get('container-title',[]),
                'abstract':plain(item.get('abstract','')),'authors':authors,'byline_has_etal':True,
                'evidence_receipt':str(path.relative_to(root)),'source_url':record['url'],
                'authority':'Publisher-deposited text and affiliation statements; completeness of registered byline not asserted'}
            db.execute('INSERT OR REPLACE INTO source_metadata VALUES(?,?,?,?,?)',(receipt,item['DOI'].lower(),'deposit',raw_hash,json.dumps(payload,ensure_ascii=False)))
            count+=1
        db.execute('INSERT OR REPLACE INTO deposit_file_index VALUES(?,?)', (file_receipt, stamp))
    db.commit()
    return count


@lru_cache(maxsize=1)
def cached_journal_rules() -> dict:
    path = ROOT / 'supplement/cached_journal_identity_rules.json'
    return json.loads(path.read_text())['rules'] if path.exists() else {}


def signature_match(paper: dict, item: dict, issns: set[str]) -> bool:
    title = key(paper['title']) in {key(plain(t)) for t in item.get('title', [])}
    journal = bool(issns & set(item.get('issns', []))) or key(paper['journal_name']) in {key(t) for t in item.get('journal_titles', [])}
    if not item.get('issns'):
        journal = journal or pnas_journal_alias(paper['journal_name'],item.get('journal_titles',[]),issns)
        rule = cached_journal_rules().get(paper.get('source_id'), {})
        if issns & set(rule.get('validated_issns', [])):
            journal = journal or bool({key(t) for t in item.get('journal_titles', [])} & set(rule.get('title_keys', [])))
    return bool(title and journal)


def institution_ids(author: dict, registry: dict, by_name: dict, names: dict) -> tuple[list[str], list[dict]]:
    found = set()
    slots = []
    for aff in author['affiliations']:
        ids = set()
        site = countries([aff], names)
        for identifier in aff.get('identifiers', []):
            if identifier['type'].casefold() == 'ror':
                ids.update(registry.get(identifier['value'].rstrip('/'), []))
        declared_names=list(aff.get('names',[]))+[part.strip() for part in aff['text'].split(',')]
        for name in declared_names:
            candidates = by_name.get(key(name), [])
            candidates = [r for r in candidates if site and r.get('country_code') in site]
            if len(candidates) == 1:
                ids.add(candidates[0]['id'])
        found.update(ids)
        slots.append({**aff, 'institution_ids': sorted(ids), 'identity_status': 'exact_declared_identifier_or_name_and_site' if ids else 'unresolved',
                      'country_codes': site, 'university_ownership_status': 'not_inferred_from_registry'})
    return sorted(found), slots


def preserve_official_slots(previous: list[dict], incoming: list[dict]) -> tuple[list[dict], bool]:
    """Retain reviewed identity/country evidence for the same complete signature."""
    reviewed=[v for v in previous if v.get('official_identity_evidence') or v.get('country_evidence')]
    if not reviewed:return incoming,False
    result=json.loads(json.dumps(incoming))
    for old in reviewed:
        matches=[v for v in result if key(v.get('text',''))==key(old.get('text',''))]
        if len(matches)!=1:return incoming,True
        new=matches[0]
        if old.get('official_identity_evidence') and new.get('institution_ids') and set(new['institution_ids'])!=set(old.get('institution_ids',[])):
            return incoming,True
        if old.get('official_identity_evidence'):
            for field in ['institution_ids','identity_status','entity_type','university_ownership_status','official_identity_evidence']:
                if field in old:new[field]=old[field]
        if old.get('country_evidence') and old.get('country_codes'):
            if new.get('country_codes') and set(new['country_codes']) != set(old['country_codes']):
                return incoming,True
            new['country_codes'] = old['country_codes']
            new['country_evidence'] = old['country_evidence']
        if not new.get('country_codes') and old.get('country_codes'):
            new['country_codes']=old['country_codes']
            new['retained_country_evidence']='previous_identical_publisher_signature'
    return result,False


def enrich_authors(current: list[dict], source: dict, registry: dict, by_name: dict, names: dict) -> tuple[list[dict], dict]:
    result = json.loads(json.dumps(current))
    candidates = defaultdict(list)
    for a in source.get('authors', []):
        candidates[key(a['name'])].append(a)
    counts = Counter(key(a['name']) for a in current)
    stats = Counter()
    for author in result:
        matches = candidates.get(key(author['name']), [])
        if len(matches) != 1 or counts[key(author['name'])] != 1:
            stats['unresolved_author_link'] += 1
            continue
        matched = separate_affiliation_notes(matches[0])
        if matched['unresolved_affiliation_refs']:
            stats['unresolved_affiliation_reference'] += 1
            continue
        ids, slots = institution_ids(matched, registry, by_name, names)
        slots, official_conflict = preserve_official_slots(author.get('publisher_affiliation_slots',[]) or [],slots)
        if official_conflict:
            author['official_identity_source_conflict']=True
            if any(old.get('country_evidence') and new.get('country_codes')
                   and key(old.get('text','')) == key(new.get('text',''))
                   and set(old.get('country_codes',[])) != set(new['country_codes'])
                   for old in author.get('publisher_affiliation_slots',[]) or [] for new in slots):
                author['country_conflict_retained'] = True
                author['country_conflict_disposition'] = 'reviewed_signature_country_conflicts_with_new_primary_source'
            author['conflicting_publisher_slots']={'slots':slots,'receipt':source['evidence_receipt']}
            stats['official_identity_source_conflicts']+=1
            continue
        site = countries(matched['affiliations'], names)
        if ids and not author['institution_ids_or_verified_local_entities']:
            author['institution_ids_or_verified_local_entities'] = ids
            author['identity_source'] = 'publisher_signature_exact_registry_identifier_or_unique_name_site'
            stats['institution_author_filled'] += 1
        if site and not author['countries']:
            author['countries'] = site
            author['country_source'] = 'publisher_explicit_affiliation_countries'
            stats['country_author_filled'] += 1
        elif site and set(site) != set(author['countries']):
            author['country_conflict_retained'] = True
            author['country_conflict_disposition'] = 'publisher_signature_conflict_requires_review'
            stats['country_conflict'] += 1
        if matched.get('non_affiliation_notes'):
            author['publisher_nonaffiliation_notes'] = matched['non_affiliation_notes']
        author['publisher_affiliation_slots'] = slots
        author['publisher_affiliation_source'] = source['evidence_receipt']
        author['publisher_author_nonpersonal'] = matched['nonpersonal']
        author['publisher_author_identity_match'] = 'unique_exact_name_within_matched_paper'
        if matched.get('raw_name_identity_evidence'):
            author['publisher_author_identity_match'] = 'unique_publisher_raw_name_and_original_author_id'
            author['publisher_author_identity_evidence'] = matched['raw_name_identity_evidence']
        if matched['nonpersonal']:
            author['nonpersonal_verified'] = True
    return result, dict(stats)


def byline_fingerprint(authors: list[dict]) -> str:
    """Use the existing exact name key; retain all order and affiliation differences."""
    return json.dumps([{**separate_affiliation_notes(author), 'name': key(author['name'])} for author in authors], sort_keys=True)


def complete_byline(current: list[dict], source: dict) -> tuple[list[dict], bool]:
    """Append only declared byline members when every old author links uniquely."""
    publisher=source.get('authors',[])
    if not publisher or source.get('byline_has_etal'):return current,False
    old=Counter(key(a['name']) for a in current)
    new=Counter(key(a['name']) for a in publisher)
    if '' in new or any(n!=1 for n in new.values()) or any(n!=1 or new[k]!=1 for k,n in old.items()):
        return current,False
    result=json.loads(json.dumps(current))
    for author in publisher:
        if key(author['name']) in old:continue
        result.append({'position':len(result),'name':author['name'],'author_id':'',
            'original_institution_ids':[],'original_countries':[],'institution_ids_or_verified_local_entities':[],
            'countries':[],'nonpersonal_verified':author['nonpersonal'],
            'identity_source':'publisher_byline_added_identity_pending','country_source':'missing',
            'country_conflict_retained':False,'explicit_raw_countries':[],
            'country_conflict_disposition':'not_detected','raw_affiliation_strings':[a['text'] for a in author['affiliations']],
            'status':'publisher_declared_author','supplementary_addresses':[],'supplementary_source':source['evidence_receipt'],
            'nature_v1_slots':None,'publisher_orcid':author.get('orcid','')})
    return result,True


def secondary_abstracts(root: Path, missing: set[str], db: sqlite3.Connection) -> int:
    if not missing:
        return 0
    count = 0
    q = root / 'fields/qualification'
    cache = root / 'supplement/bulk_cache.sqlite'
    if cache.exists():
        index = sqlite3.connect(cache)
        selected = set()
        for doi in missing:
            selected.update(r[0] for r in index.execute('SELECT path FROM records WHERE doi=?', (doi,)))
        index.close()
        paths = [root / name for name in sorted(selected)]
    else:
        paths = [*((root / 'fields/epmc_raw').glob('*.gz')), *((q / 'epmc_raw').glob('*.gz'))]
    for path in paths:
        with gzip.open(path, 'rt') as stream:
            r = json.load(stream)
        if r.get('status') != 'complete':
            continue
        for item in r.get('response', {}).get('resultList', {}).get('result', []):
            doi = (item.get('doi') or '').lower()
            if doi not in missing or not item.get('abstractText'):
                continue
            journal = item.get('journalInfo', {}).get('journal', {})
            payload = {'title': [item.get('title', '')], 'abstract': plain(item['abstractText']), 'authors': [],
                       'issns': [journal.get('issn', ''), journal.get('essn', '')], 'journal_titles': [journal.get('title', '')],
                       'evidence_receipt': str(path.relative_to(root)), 'source_url': 'https://europepmc.org/article/' + item.get('source', 'MED') + '/' + item['id']}
            ident = str(path.relative_to(root)) + '#' + doi
            db.execute('INSERT OR REPLACE INTO source_metadata VALUES(?,?,?,?,?)',
                       (ident, doi, 'secondary', '', json.dumps(payload, ensure_ascii=False)))
            count += 1
    db.commit()
    return count


def apply(root: Path = ROOT, scope_dois: set[str] | None = None, index_sources: bool = True) -> dict:
    db = connect(root)
    stats = Counter(load_index(root, db) if index_sources else {})
    stats['publisher_deposits_indexed']=deposited_metadata(root,db)
    q = root / 'fields/qualification'
    observed_gaps = {r[0] for r in db.execute('SELECT doi FROM papers WHERE abstract_gap=1')}
    scoped_shards = None
    if scope_dois is not None:
        observed_gaps &= scope_dois
        scoped_shards = {shard for doi, shard in db.execute('SELECT doi,shard FROM papers') if doi in scope_dois}
    missing = set()
    for path in (q / 'decisions').glob('*.parquet'):
        if scoped_shards is not None and path.name not in scoped_shards:
            continue
        missing.update(r['doi'] for r in pq.read_table(path, columns=['doi', 'eligibility_status']).to_pylist()
                       if r['eligibility_status'] == 'included_verified_research' and r['doi'] in observed_gaps)
    stats['secondary_abstract_records_indexed'] = secondary_abstracts(root, missing, db)
    records = json.loads((root / 'fields/institution_registry.json').read_text())
    registry = defaultdict(list)
    by_name = defaultdict(list)
    for r in records:
        if r.get('ror'):
            registry[r['ror'].rstrip('/')].append(r['id'])
        by_name[key(r['display_name'])].append(r)
    names = country_names()
    for ambiguous in ['korea', 'georgia', 'congo']:
        names.pop(normalized(ambiguous), None)
    with (root / 'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in csv.DictReader(stream)}
    for path in sorted((root / 'fields/papers').glob('*.parquet')):
        if scoped_shards is not None and path.name not in scoped_shards:
            continue
        decisions = {r['work_id']: r for r in pq.read_table(q / 'decisions' / path.name, columns=['work_id', 'eligibility_status']).to_pylist()}
        table = pq.read_table(path)
        rows = table.to_pylist()
        changed = False
        for paper in rows:
            if scope_dois is not None and paper['doi'] not in scope_dois:
                continue
            if decisions[paper['work_id']]['eligibility_status'] != 'included_verified_research':
                continue
            items = [(kind, json.loads(payload)) for kind, payload in db.execute('SELECT kind,payload FROM source_metadata WHERE doi=? ORDER BY CASE kind WHEN \'jats\' THEN 0 WHEN \'publisher\' THEN 1 WHEN \'deposit\' THEN 2 ELSE 3 END', (paper['doi'],))]
            items = [(k, v) for k, v in items if signature_match(paper, v, journals[paper['source_id']])]
            if not items:
                continue
            before = {k: paper[k] for k in ['abstract', 'abstract_source', 'authors_json','author_list_may_be_truncated','authorships_source_receipt','full_authorships_status','metadata_final_status',*[k for k in paper if k.startswith('after_')]]}
            receipts={k:items[0][1]['evidence_receipt'] for k in before}
            if not paper['abstract']:
                abstracts = [(k, v) for k, v in items if abstract_ok(v.get('abstract', ''))]
                if abstracts:
                    highest = abstracts[0][0]
                    selected = [v for k, v in abstracts if k == highest]
                    variants = {' '.join(v['abstract'].split()) for v in selected}
                    if len(variants) == 1:
                        paper['abstract'] = selected[0]['abstract']
                        paper['abstract_source'] = 'matched_' + highest + ':' + selected[0]['evidence_receipt']
                        receipts['abstract']=receipts['abstract_source']=selected[0]['evidence_receipt']
                        stats['abstract_filled'] += 1
                    else:
                        stats['abstract_source_conflict'] += 1
            sources = [v for k,v in items if k=='jats' and v.get('authors')]
            if not sources:sources=[v for k,v in items if k=='deposit' and v.get('authors')]
            if sources:
                # Multiple archives must agree on the byline and addresses before updating.
                fingerprints = {byline_fingerprint(v['authors']) for v in sources}
                if len(fingerprints) == 1:
                    current=json.loads(paper['authors_json'])
                    extended,complete=complete_byline(current,sources[0])
                    if complete and (len(extended)>len(current) or paper['author_list_may_be_truncated']):
                        stats['publisher_byline_authors_added']+=len(extended)-len(current)
                        paper['author_list_may_be_truncated']=False
                        paper['authorships_source_receipt']=sources[0]['evidence_receipt']
                        paper['full_authorships_status']='publisher_byline_reconstructed_unique_exact_links'
                    authors, delta = enrich_authors(extended, sources[0], registry, by_name, names)
                    stats.update(delta)
                    paper['authors_json'] = json.dumps(authors, ensure_ascii=False)
                else:
                    stats['publisher_author_source_conflict'] += 1
            actual=metrics(paper)
            for metric,value in actual.items():
                if 'after_'+metric in paper:paper['after_'+metric]=value
            paper['metadata_final_status']='gaps_remain' if not actual['abstract'] or not actual['every_author_institution'] or not actual['every_author_country'] else 'observed_fields_present_not_publisher_verified'
            for field, old in before.items():
                if old == paper[field]:
                    continue
                changed = True
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (paper['work_id'], field, old if isinstance(old,str) else json.dumps(old), paper[field] if isinstance(paper[field],str) else json.dumps(paper[field]), receipts[field],
                            'Matched DOI, title and journal; nonempty source value; exact author identity/site checks; optional exact journal alias proof: supplement/cached_journal_identity_rules.json'))
                stats['changed_' + field] += 1
        if changed:
            temporary = path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
            with temporary.open('rb') as durable:os.fsync(durable.fileno())
            temporary.replace(path)
            descriptor=os.open(path.parent,os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        db.commit()
        print('applied metadata', path.name, 'changed', changed, flush=True)
    result = dict(stats)
    (root / 'supplement/metadata_apply_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    db.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(apply(args.root)))


if __name__ == '__main__':
    main()
