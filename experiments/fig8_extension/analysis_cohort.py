"""Select the current complete-input Fig8 cohort without changing source eligibility."""
from __future__ import annotations

import fcntl
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8_preview import digest, write_csv, write_json
from .journal_readiness import abstract_screen
from .supplement_queue import ROOT

OUT = ROOT / 'analysis/cohort'
FIELDS = {'biochemistry', 'materials', 'physics'}
ORDER = ['field_outside_scope', 'identity_or_title_invalid', 'abstract_missing',
         'abstract_quality_flag', 'abstract_format_invalid', 'authors_format_invalid',
         'author_list_truncated', 'no_personal_authors', 'author_country_missing',
         'author_country_format_invalid', 'author_country_conflict', 'affiliation_country_missing']


def country_codes_valid(values: object) -> bool:
    return isinstance(values, list) and bool(values) and all(
        isinstance(v, str) and re.fullmatch('[A-Z]{2}', v) and v not in {'ZZ', 'XX'} for v in values)


def author_flags(row: dict) -> tuple[list[str], dict]:
    flags = []
    stats = {'personal_authors': 0, 'institution_complete': False, 'country_sources': []}
    try:
        authors = json.loads(row['authors_json'] or 'null')
    except (ValueError, TypeError):
        return ['authors_format_invalid'], stats
    if not isinstance(authors, list) or any(not isinstance(a, dict) for a in authors):
        return ['authors_format_invalid'], stats
    if row.get('author_list_may_be_truncated'):
        flags.append('author_list_truncated')
    personal = [a for a in authors if not a.get('nonpersonal_verified')]
    stats['personal_authors'] = len(personal)
    if not personal:
        return flags + ['no_personal_authors'], stats
    stats['institution_complete'] = all(a.get('institution_ids_or_verified_local_entities') for a in personal)
    stats['country_sources'] = sorted({str(a.get('country_source') or 'unspecified') for a in personal})
    for author in personal:
        countries = author.get('countries')
        if not countries:
            flags.append('author_country_missing')
        elif not country_codes_valid(countries):
            flags.append('author_country_format_invalid')
        if author.get('country_conflict_retained') or author.get('secondary_country_conflict'):
            flags.append('author_country_conflict')
        slots = author.get('nature_v1_slots') or author.get('publisher_affiliation_slots') or []
        if not isinstance(slots, list) or any(not isinstance(s, dict) for s in slots):
            flags.append('authors_format_invalid')
            continue
        for slot in slots:
            codes = slot.get('country_codes')
            if not codes:
                flags.append('affiliation_country_missing')
            elif not country_codes_valid(codes):
                flags.append('author_country_format_invalid')
            elif country_codes_valid(countries) and not set(codes) <= set(countries):
                flags.append('author_country_conflict')
    return sorted(set(flags)), stats


def assess(row: dict, decision: dict, reviewed: dict, provenance: dict) -> dict:
    reasons = []
    if decision['field'] not in FIELDS:
        reasons.append('field_outside_scope')
    if not re.fullmatch(r'10\.\d{4,9}/\S+', row['doi'] or '', re.I) or not (row['title'] or '').strip():
        reasons.append('identity_or_title_invalid')
    abstract = row['abstract'] or ''
    screen = abstract_screen(abstract, row['doi'], reviewed, provenance)
    if screen == 'missing':
        reasons.append('abstract_missing')
    elif screen == 'quality_review':
        reasons.append('abstract_quality_flag')
    if '\ufffd' in abstract or re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', abstract) or re.search(r'</?(?:html|body|script|jats:abstract)\b', abstract, re.I):
        reasons.append('abstract_format_invalid')
    af, stats = author_flags(row)
    reasons += af
    reasons = [r for r in ORDER if r in reasons]
    return {'work_id': row['work_id'], 'doi': row['doi'], 'source_id': decision['source_id'],
            'journal': row['journal_name'], 'field': decision['field'], 'online_year': decision['online_year'],
            'paper_shard': decision['base_shard'], 'portfolio': decision['nature_portfolio_at_verified_online_year'],
            'included': not reasons, 'primary_exclusion': reasons[0] if reasons else '',
            'exclusion_reasons': reasons, 'abstract_status': screen, 'abstract_sha256': digest(abstract),
            'personal_author_n': stats['personal_authors'], 'all_personal_authors_have_institution': stats['institution_complete'],
            'country_sources': stats['country_sources'], 'old_claims_used': False}


def source_reviews() -> tuple[dict, dict]:
    reviewed, provenance = {}, {}
    for item in json.loads((ROOT / 'supplement/pnas_abstract_review/current.json').read_text()).get('papers', []):
        if item['status'] in {'native_text_confirmed', 'repaired_notice_as_abstract'} and (ROOT / item['receipt']).exists():
            reviewed[item['doi']] = item['abstract_sha256']
    for item in json.loads((ROOT / 'supplement/pbj_native_formats/abstract_availability.json').read_text()).get('records', []):
        if item['status'] == 'existing_text_without_native_abstract_review' and item.get('current_abstract_sha256'):
            provenance[item['doi']] = item['current_abstract_sha256']
    return reviewed, provenance


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reviewed, provenance = source_reviews()
    counts, excluded, reasons = Counter(), Counter(), Counter()
    groups = defaultdict(Counter)
    journals = {}
    writer = None
    temp = OUT / 'selection.tmp.parquet'
    seen = set()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            for path in sorted((ROOT / 'fields/papers').glob('*.parquet')):
                decisions = {d['work_id']: d for d in pq.read_table(ROOT / 'fields/qualification/decisions' / path.name,
                             filters=[('eligibility_status', '=', 'included_verified_research')]).to_pylist()}
                columns = ['work_id','doi','title','journal_name','abstract','authors_json','author_list_may_be_truncated']
                selected = []
                for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=1000):
                    for row in batch.to_pylist():
                        if row['work_id'] not in decisions:
                            continue
                        d = decisions[row['work_id']]
                        assert row['doi'] not in seen
                        seen.add(row['doi'])
                        item = assess(row, d, reviewed, provenance)
                        item['paper_shard'] = path.name
                        selected.append(item)
                        counts['original_qualified'] += 1
                        counts['included' if item['included'] else 'excluded'] += 1
                        if item['included']:
                            counts['retained_missing_institution'] += not item['all_personal_authors_have_institution']
                            counts['retained_brand_unknown'] += item['portfolio'] not in {'True','False'}
                        else:
                            excluded[item['primary_exclusion']] += 1
                            reasons.update(item['exclusion_reasons'])
                        journals[item['source_id']] = item['journal']
                        for key in [('field',item['field']),('field_year',item['field']+'|'+str(item['online_year'])),
                                    ('journal',item['source_id']),('journal_field',item['source_id']+'|'+item['field'])]:
                            groups[key]['original_n'] += 1
                            groups[key]['included_n'] += item['included']
                            groups[key]['excluded_n'] += not item['included']
                table = pa.Table.from_pylist(selected)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression='zstd')
                writer.write_table(table.cast(writer.schema))
                print(path.name,counts['original_qualified'],counts['included'],flush=True)
        finally:
            if writer:
                writer.close()
    assert counts['original_qualified'] == 500002 == counts['included'] + counts['excluded']
    temp.replace(OUT / 'selection.parquet')
    write_csv(OUT / 'group_counts.csv', [{'dimension':k[0],'group':k[1],
                'journal':journals.get(k[1].split('|')[0],''),**dict(v)} for k,v in sorted(groups.items())])
    priority = [dict(source_id=s,journal=journals[s],**groups[('journal',s)]) for s in
                ['https://openalex.org/S137773608','https://openalex.org/S3880285','https://openalex.org/S110447773','https://openalex.org/S125754415']]
    summary = {'created_at':datetime.now(timezone.utc).isoformat(),'counts':dict(counts),
               'exclusive_exclusions':dict(excluded),'overlapping_exclusion_flags':dict(reasons),
               'fields':[dict(field=f,**groups[('field',f)]) for f in sorted(FIELDS)],'priority_journals':priority,
               'journal_n':sum(v['included_n']>0 for k,v in groups.items() if k[0]=='journal'),
               'llm_first_pass_requests':counts['included'],'reuse_old_results':False,'new_acquisition':False,
               'interpretation':'Input-complete observed subset, not a random or unbiased full-journal cohort; screen pass is not scientific validation.',
               'country_rule':'All personal authors have country codes; no truncated byline or flagged country conflict; all recorded affiliation slots have countries.',
               'country_evidence_rule':'Retains recorded source, including inherited OpenAlex; nonmissing does not imply publisher verification.',
               'university_rule':'University ownership and institution completeness remain separate readiness dimensions; not silently treated as verified.',
               'old_claim_failure_rule':'Old claim text, formatting and extraction status do not determine new paper inclusion.'}
    write_json(OUT / 'summary.json',summary)
    activate(summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def activate(summary: dict) -> None:
    """Change the active analysis queue, never overwrite source research eligibility."""
    from .current_extract import JOURNALS, MODEL
    from .panel_c_scope import run as apply_panel_c

    rows=pq.read_table(OUT/'selection.parquet',columns=['doi','source_id','included','exclusion_reasons']).to_pylist()
    target_n=sum(r['included'] and r['source_id'] in JOURNALS for r in rows)
    db=sqlite3.connect(ROOT/'analysis/contributions/contributions.sqlite')
    try:
        db.execute('CREATE TABLE IF NOT EXISTS cohort_selection(doi TEXT PRIMARY KEY,included INTEGER,reasons TEXT)')
        db.executemany('INSERT OR REPLACE INTO cohort_selection VALUES (?,?,?)',[(r['doi'],int(r['included']),json.dumps(r['exclusion_reasons'])) for r in rows])
        db.execute('''UPDATE papers SET status='excluded_analysis_input' WHERE doi IN
            (SELECT doi FROM cohort_selection WHERE included=0)''')
        db.executemany('''UPDATE papers SET status=CASE WHEN status IN
            ('pending','metadata_only','excluded_analysis_input') THEN ? ELSE status END,
            legacy_status='discarded_by_user',legacy_id='' WHERE doi=?''',
            [('pending' if r['source_id'] in JOURNALS else 'metadata_only',r['doi'])
             for r in rows if r['included']])
        db.commit()
    finally:
        db.close()
    scope_path=ROOT/'analysis/analysis_scope.json'
    scope=json.loads(scope_path.read_text()) if scope_path.exists() else {}
    scope.update({'status':'current','parent_qualified_n':500002,
        'analysis_papers':summary['counts']['included'],'fields':sorted(FIELDS),'journal_restriction':None,
        'selection':'analysis/cohort/selection.parquet','reuse_old_contributions':False,
        'all_contributions_require_fresh_extraction':True,'first_pass_llm_requests':target_n,
        'extraction_source_ids':list(JOURNALS),'extraction_model':MODEL,
        'extraction_reasoning_effort':'low',
        'excluded_records_preserved_in_master':True,'no_new_paper_acquisition':True})
    write_json(scope_path,scope)
    summary['llm_first_pass_requests']=target_n
    write_json(OUT/'summary.json',summary)
    write_json(ROOT/'analysis/contributions/batch_gate.json',{'allow_batch':True,
        'reason':'Fresh extraction restricted to the four approved journals.',
        'analysis_papers':summary['counts']['included'],'extraction_papers':target_n,
        'source_ids':list(JOURNALS),'model':MODEL,'reasoning_effort':'low','reuse_old_results':False})
    apply_panel_c()


if __name__=='__main__':
    run()
