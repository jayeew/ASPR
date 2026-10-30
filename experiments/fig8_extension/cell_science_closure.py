"""Finite Cell annual imprint and Science type evidence audit."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone

from . import three_journal_round as bounded

AUDIT = bounded.ROOT / 'supplement/cell_science_closure'


def get(url: str, params: dict | None = None, dois: list[str] | None = None) -> dict:
    """Use durable receipts and tighter limits than the previous bounded collector."""
    bounded.AUDIT = AUDIT
    state = bounded.init()
    if datetime.now(timezone.utc) >= datetime.fromisoformat(state['network_stop_utc']):
        raise RuntimeError('Network phase closed')
    if state['requests'] >= state['request_cap']:
        raise RuntimeError('Request cap')
    for path in (AUDIT / 'requests').glob('*.json'):
        if not path.stem.isdigit():
            continue
        old = json.loads(path.read_text())
        if old['url'] == url and old.get('params') == params:
            return old
    return bounded.get(url, params, dois)


def cache() -> None:
    """Inspect existing evidence once, scoped to Science and Cell."""
    decisions, evidence, secondary, sources = bounded.load()
    selected = {wid:d for wid,d in decisions.items() if d['source_id'] in bounded.SIDS[:2]}
    bounded.save(AUDIT/'scoped_evidence_index.json', {'evidence':{w:evidence[w] for w in selected},
                  'secondary':{w:secondary[w] for w in selected}, 'sources':sources})
    bounded.save(AUDIT/'baseline_counts.json', {s:sum(d['source_id']==s for d in selected.values()) for s in bounded.SIDS[:2]})
    print('Scoped cached evidence:',len(selected))


def prepare() -> None:
    """Propose only Cell annual labels and one recovered Science record."""
    if (AUDIT/'decision_delta.json').exists():
        print('Existing proposal retained; no overwrite on resume.')
        return
    from lxml import etree
    from .fig8b_qualification_build import match
    from .fig8b_qualification_decide import decide, electronic_date
    from .three_journal_round import row_for

    decisions, evidence, secondary, sources = bounded.load()
    proofs = []
    for number in ['009', '012']:
        path = AUDIT/'requests'/f'{number}.body.gz'
        root = etree.fromstring(gzip.decompress(path.read_bytes()))
        for article in root.findall('PubmedArticle'):
            journal = article.findtext('.//Journal/Title')
            issn = article.findtext('.//Journal/ISSN')
            dates = article.findall('.//ArticleDate[@DateType="Electronic"]')
            notice = article.findtext('.//CopyrightInformation') or ''
            if journal != 'Cell' or issn not in {'0092-8674','1097-4172'} or not dates or 'Elsevier Inc.' not in notice:
                continue
            date = '-'.join(dates[0].findtext(t) for t in ['Year','Month','Day'])
            if not date.startswith('2023-'):
                continue
            pmid = article.findtext('.//PMID')
            proofs.append({'id':f'cell-pubmed-publisher-notice-{pmid}', 'receipt':'../../supplement/cell_science_closure/requests/'+path.name,
                           'locator':f'PubmedArticle[MedlineCitation/PMID={pmid}]/MedlineCitation/Article/Abstract/CopyrightInformation',
                           'doi':article.findtext('./PubmedData/ArticleIdList/ArticleId[@IdType="doi"]'), 'date':date,
                           'issns':[issn], 'journal':journal, 'publisher_notice':notice,
                           'source_url':'https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/'})
    assert {p['date'][5:7] for p in proofs} >= {'01','07','12'}
    annual = [dict(source_id=bounded.SIDS[1], year=2023, label='False', publisher='Elsevier / Cell Press',
                   status='dated_publisher_notices_in_official_pubmed', proof_count=len(proofs), proofs=proofs,
                   scope='Cell ISSNs matched; publisher copyright notices with electronic dates across January, July and December 2023. Annual working rule, reopened on transfer/conflict evidence; not a claim that every issue was audited.')]
    for year, receipt, locator, status in [(2024,'002.body.gz','PDF pages 1, 4 and 15: dated Elsevier agreement and Cell Press Hybrid Eligible Journals, Cell 0092-8674','dated_elsevier_contract_annex'),
                                          (2025,'008.body.gz','PDF page 1: Università di Genova 02/04/2025, Cell Press Journals Updated April 2025; Cell 0092-8674','dated_cell_press_title_list')]:
        proof = {'id':f'cell-annual-{year}-dated-title-list','receipt':'../../supplement/cell_science_closure/requests/'+receipt,'locator':locator,
                 'source_url':json.loads((AUDIT/'requests'/receipt.replace('.body.gz','.json')).read_text())['url']}
        annual.append(dict(source_id=bounded.SIDS[1],year=year,label='False',publisher='Elsevier / Cell Press',status=status,proof_count=1,proofs=[proof],
                           scope='Dated institutional copy of publisher contract/title list; exact Cell ISSN. 2024 agreement corroborates 2025 term. Reopen if journal-transfer evidence conflicts.'))
    bounded.save(AUDIT/'annual_rules.json',annual)
    parsed = json.loads((AUDIT/'science_retry_parsed.json').read_text())
    d = next(r for r in decisions.values() if r['doi']=='10.1126/science.adj9223')
    assert parsed['doi']==[d['doi']]
    date, precision = electronic_date(parsed['dates'])
    e = match(row_for(d,sources), {'family':'archived_jats','title':' '.join(parsed['title']),'issns':parsed['issn'],
              'journal_titles':parsed['journal_titles'],'kind':parsed['article_type'],'online_date':date,'date_precision':precision,
              'categories':parsed['categories'],'publishers':parsed['publisher'],'receipt':'../../supplement/cell_science_closure/requests/013.body.gz',
              'url':json.loads((AUDIT/'requests/013.json').read_text())['final_url'],
              'scope_review_note':'Abstract describes the authors species identification and comparative study, not a teaser summarizing another paper; current JATS research rule retained, no independent full-text certification.'},sources[d['source_id']])
    e.update(work_id=d['work_id'],doi=d['doi'])
    assert e['identity']=='matched'
    evidence[d['work_id']].append(e)
    proposals=[]
    for wid, before in decisions.items():
        after=dict(before)
        if wid==d['work_id']:
            r=decide(evidence[wid],secondary[wid],before['duplicate_of'])
            after.update(online_year=r['online_year'],online_date=r['online_date'],eligibility_status=r['status'],reason=r['reason'],
                         publisher_types=r['type'],evidence_ids=json.dumps(r['evidence_ids']),rejected_evidence_ids=json.dumps(r.get('rejected_evidence_ids',[])))
            if r['online_year'] in {2023,2024,2025}:
                after['nature_portfolio_at_verified_online_year']='False'
                after['portfolio_evidence_ids']=json.dumps([f'nlm-0404511-imprint-{r["online_year"]}'])
        rule=next((r for r in annual if before['source_id']==r['source_id'] and after['online_year']==r['year']),None)
        if rule:
            assert before['nature_portfolio_at_verified_online_year']!='True'
            after['nature_portfolio_at_verified_online_year']='False'
            after['portfolio_evidence_ids']=json.dumps([p['id'] for p in rule['proofs']])
        if after!=before:proposals.append({'work_id':wid,'before':before,'after':after})
    bounded.save(AUDIT/'decision_delta.json',proposals)
    bounded.save(AUDIT/'evidence_delta.json',[e])
    print('Proposed changes',len(proposals), 'Science status',next(r['after']['eligibility_status'] for r in proposals if r['work_id']==d['work_id']))


def refresh_metadata() -> None:
    """Increment existing metadata reports only for the one newly included paper."""
    from collections import Counter
    from .closure_metadata import norm
    from .three_journal_apply import read_csv, write_csv, target_papers

    if (AUDIT/'metadata_reports_applied.json').exists():
        return
    changes = json.loads((AUDIT/'decision_delta.json').read_text())
    new = [r['after'] for r in changes if r['before']['eligibility_status']!='included_verified_research' and r['after']['eligibility_status']=='included_verified_research']
    papers=target_papers()
    notes=set(json.loads((bounded.ROOT/'supplement/closure_goal/noninstitution_note_rules.json').read_text())['exact_normalized_texts'])
    rows=read_csv(bounded.OUT/'three_journal_author_institution_coverage.csv')
    unresolved=read_csv(bounded.OUT/'three_journal_metadata_unresolved.csv')
    for d in new:
        p=papers[d['work_id']]; authors=json.loads(p['authors_json']); c=Counter(papers=1,authors=len(authors))
        c['papers_all_authors_known_institution']=bool(authors) and all(a['institution_ids_or_verified_local_entities'] for a in authors)
        c['papers_all_authors_known_country']=bool(authors) and all(a['countries'] for a in authors)
        c['papers_author_list_may_be_truncated']=bool(p['author_list_may_be_truncated'])
        for a in authors:
            c['authors_known_institution']+=bool(a['institution_ids_or_verified_local_entities']);c['authors_known_country']+=bool(a['countries'])
            c['authors_country_conflict']+=bool(a.get('country_conflict_retained'))
            slots=a.get('publisher_affiliation_slots') or [];c['authors_without_publisher_slot_structure']+=not slots
            for s in slots:
                if norm(s['text']) in notes:c['noninstitution_note_slots_excluded']+=1;continue
                c['represented_publisher_institution_slots']+=1;c['slots_known_institution']+=bool(s.get('institution_ids'));c['slots_known_country']+=bool(s.get('country_codes'));c['slots_unresolved_identity']+=s.get('identity_status')=='unresolved'
        row=next(r for r in rows if r['journal']==p['journal_name'])
        for k in row:
            if k!='journal':row[k]=int(row[k])+int(c[k])
        unresolved.append(dict(work_id=d['work_id'],doi=d['doi'],journal=p['journal_name'],field=d['field'],
            authors_missing_known_institution=len(authors)-c['authors_known_institution'],authors_missing_known_country=len(authors)-c['authors_known_country'],
            authors_country_conflict=c['authors_country_conflict'],authors_without_publisher_slot_structure=c['authors_without_publisher_slot_structure'],unresolved_publisher_institution_slots=c['slots_unresolved_identity'],
            authors_inherited_unverified_identity=sum('openalex' in (a.get('identity_source') or '').lower() for a in authors),status='unresolved_retained_after_bounded_cache_review',
            next_evidence_needed='author_signature_and_official_entity_or_location_evidence; no inferred university parent',source_receipts=json.dumps(sorted({a['publisher_affiliation_source'] for a in authors if a.get('publisher_affiliation_source')}))))
    write_csv(bounded.OUT/'three_journal_author_institution_coverage.csv',rows)
    write_csv(bounded.OUT/'three_journal_metadata_unresolved.csv',unresolved)
    bounded.save(AUDIT/'metadata_reports_applied.json',{'added_papers':len(new),'institution_or_country_assignments_added':0})


def verify() -> None:
    """Offline scoped validation, including provenance and exported counts."""
    import csv
    import hashlib
    from .fig8b_qualification_decide import decide, portfolio_label
    from .three_journal_apply import read_csv

    D,E,S,_=bounded.load()
    annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in read_csv(bounded.ROOT/'fields/journal_publisher_by_year.csv')}
    for wid,d in D.items():
        r=decide(E[wid],S[wid],d['duplicate_of'])
        label,_=portfolio_label(annual.get((d['source_id'],d['online_year']),''),E[wid],d['online_year'])
        assert (r['status'],r['online_year'],label)==(d['eligibility_status'],d['online_year'],d['nature_portfolio_at_verified_online_year'])
    roster=read_csv(bounded.Q/'research_roster.csv')
    assert len(roster)==len({r['doi'] for r in roster})
    summary=json.loads((bounded.Q/'recomputed_results.json').read_text())
    assert len(roster)==summary['verified_research_count']
    pending={w for w,d in D.items() if d['eligibility_status'].startswith('pending')};found=set();n=0
    with gzip.open(bounded.Q/'unresolved.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            n+=1
            if r['work_id'] in D:found.add(r['work_id'])
    assert pending==found and n==summary['unresolved_count']
    records=[json.loads(p.read_text()) for p in (AUDIT/'requests').glob('*.json') if p.stem.isdigit()]
    budget=json.loads((AUDIT/'budget.json').read_text())
    assert len(records)==budget['requests']<=budget['request_cap']
    assert {d for r in records for d in r['dois']}==set(budget['dois'])
    assert len(budget['dois'])<=budget['doi_cap'] and budget['search_queries']<=budget['search_query_cap']
    for r in records:
        if 'raw_receipt' in r:assert hashlib.sha256(gzip.decompress((bounded.ROOT/r['raw_receipt']).read_bytes())).hexdigest()==r['sha256']
    for rule in json.loads((AUDIT/'annual_rules.json').read_text()):
        assert all((bounded.Q/p['receipt']).is_file() for p in rule['proofs'])
    journals=read_csv(bounded.OUT/'priority_journals.csv')
    assert all(int(r['candidate_n'])==int(r['verified_n'])+int(r['excluded_n'])+int(r['pending_n']) for r in journals)
    bounded.save(AUDIT/'completion_checks.json',{'scope_candidate_n':len(D),'offline_decision_errors':0,'global_roster_n':len(roster),'global_roster_dois_unique':True,
                 'global_unresolved_export_n':n,'scope_pending_ids_exact':True,'raw_hashes_valid':True,'annual_proof_paths_valid':True,
                 'global_scientific_requalification':False,'request_n':len(records),'explicit_doi_queries_n':len(budget['dois']),
                 'checked_at':datetime.now(timezone.utc).isoformat()})
    print('Offline validation passed',len(D),len(roster),n)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['cache','prepare','verify'])
    args=parser.parse_args()
    {'cache':cache,'prepare':prepare,'verify':verify}[args.action]()
