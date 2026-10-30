import csv
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.fig8_extension.publisher_delta_apply import apply_available
from experiments.fig8_extension.supplement_queue import connect


@pytest.mark.parametrize('duplicate',[False,True])
@pytest.mark.parametrize('supplied',[False,True])
def test_incremental_apply_identity_hold_and_idempotence(tmp_path,duplicate,supplied):
    root=tmp_path;q=root/'fields/qualification'
    for path in [q/'decisions',q/'supplement_parsed',root/'fields/papers']:path.mkdir(parents=True)
    for name in ['decision_evidence.jsonl.gz','secondary_evidence.jsonl.gz']:
        with gzip.open(q/name,'wt') as stream:stream.write('')
    sources=[('journal_registry.csv',[{'source_id':'S1','journal_name':'Example','issn':'["1234-5678"]'}]),('journal_publisher_by_year.csv',[{'source_id':'S1','year':'2024','nature_portfolio_year':'False'}])]
    for name,rows in sources:
        with (root/'fields'/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    papers=[];decisions=[];db=connect(root)
    for i in [1,2]:
        doi='10.1007/a' if i==1 or duplicate else '10.1007/b'
        paper={'work_id':f'W{i}','doi':doi,'title':'A valid paper','source_id':'S1','journal_name':'Example','field':'physics','abstract':'','abstract_source':'','after_abstract':0,'author_list_may_be_truncated':False,'authors_json':json.dumps([{'name':'A','countries':[],'institution_ids_or_verified_local_entities':[]}])}
        papers.append(paper)
        decisions.append({'work_id':f'W{i}','doi':doi,'source_id':'S1','duplicate_of':'','online_year':0,'online_date':'','publisher_types':'','eligibility_status':'pending_no_primary_type_date','reason':'No source','evidence_ids':'[]','rejected_evidence_ids':'[]','nature_portfolio_at_verified_online_year':'unknown','portfolio_evidence_ids':'[]'})
        db.execute('INSERT INTO papers(work_id,doi,title,source_id,journal,shard,status) VALUES(?,?,?,?,?,?,?)',(f'W{i}',doi,'A valid paper','S1','Example','a.parquet','pending_no_primary_type_date'))
    pq.write_table(pa.Table.from_pylist(papers),root/'fields/papers/a.parquet');pq.write_table(pa.Table.from_pylist(decisions),q/'decisions/a.parquet')
    record={'route':'springer','url':'https://link.springer.com/article/10.1007/a','raw_receipt':'raw.html.gz','parsed':{'title':['A valid paper'],'issns':['1234-5678'],'journal_titles':['Example'],'article_types':['OriginalPaper'],'online_date':'2024-01-02','date_precision':'day','abstract':'word '*70}}
    (q/'supplement_parsed/a.json').write_text(json.dumps(record))
    db.execute('CREATE TABLE publisher_remaining_jobs(doi TEXT PRIMARY KEY,route TEXT,status TEXT,receipt TEXT)')
    db.execute('INSERT INTO publisher_remaining_jobs VALUES(?,?,?,?)',('10.1007/a','springer','parsed','fields/qualification/supplement_parsed/a.json'));db.commit();db.close()
    jobs=[('10.1007/a','fields/qualification/supplement_parsed/a.json')] if supplied else None
    author_jobs=[]
    result=apply_available(root,supplied_jobs=jobs,author_jobs_out=author_jobs);rows=pq.read_table(q/'decisions/a.parquet').to_pylist()
    assert author_jobs == ([] if duplicate else [('W1','a.parquet','fields/qualification/supplement_parsed/a.json')])
    if duplicate:
        assert all(r['eligibility_status']=='pending_special_subtype_review' for r in rows)
        assert result.get('abstracts_filled',0)==0
        assert len((q/'curated_evidence.jsonl').read_text().splitlines())==2
    else:
        assert rows[0]['eligibility_status']=='included_verified_research'
        assert rows[1]==decisions[1]
        assert result['abstracts_filled']==1
    before=pq.read_table(root/'fields/papers/a.parquet').to_pylist()
    assert before[1] == papers[1]
    if not duplicate:
        assert before[0]['abstract'] == 'word '*70
        assert before[0]['after_abstract'] == 1
        assert {k:v for k,v in before[0].items() if k not in {'abstract','abstract_source','after_abstract'}} == {k:v for k,v in papers[0].items() if k not in {'abstract','abstract_source','after_abstract'}}
    repeat_jobs=[]
    assert apply_available(root,supplied_jobs=jobs,author_jobs_out=repeat_jobs)=={'applied_source_records':0}
    assert repeat_jobs == []
    if supplied:
        db=connect(root)
        assert 'applied' not in {r[1] for r in db.execute('PRAGMA table_info(publisher_remaining_jobs)')}
        assert db.execute('SELECT status FROM publisher_remaining_jobs').fetchone()[0]=='parsed'
        db.close()
    assert pq.read_table(root/'fields/papers/a.parquet').to_pylist()==before


def test_archive_adapter_preserves_type_categories_and_date_semantics():
    from experiments.fig8_extension.publisher_delta_apply import publisher_items
    record={'source':'pmc','route':'pmc_archive_priority','url':'https://example.org/PMC1','raw_receipt':'front.xml.gz',
        'parsed':{'title':['A paper'],'issn':['1234-5678'],'journal_titles':['Journal'],'article_type':'research-article','categories':['Profile'],
        'publisher':['Publisher'],'dates':[{'attributes':{},'year':'2024','month':'1','day':'2'}]}}
    value=publisher_items(record)[0]
    assert value['family']=='archived_jats'
    assert value['categories']==['Profile']
    assert value['online_date']==''
    assert value['publication_dates']==['2024-01-02']


def test_archive_adapter_keeps_native_doi_for_spelling_identity():
    from experiments.fig8_extension.fig8b_qualification_build import match
    from experiments.fig8_extension.publisher_delta_apply import publisher_items
    record={'source':'pmc','doi':'10.1234/example','route':'pmc_archive_priority',
        'url':'https://example.org/PMC1','raw_receipt':'front.xml.gz',
        'parsed':{'doi':['10.1234/EXAMPLE'],'title':['Cell signaling'],
                  'issn':['1234-5678'],'journal_titles':['Journal'],
                  'article_type':'research-article','categories':[], 'dates':[]}}
    value=publisher_items(record)[0]
    assert value['doi']=='10.1234/example'
    paper={'work_id':'W1','doi':'10.1234/example','title':'Cell signalling',
           'source_id':'S1','journal_name':'Journal'}
    assert match(paper,value,{'issn':'["1234-5678"]'})['identity']=='matched'
    record['parsed']['doi']=['10.1234/other']
    assert match(paper,publisher_items(record)[0],{'issn':'["1234-5678"]'})['identity']!='matched'


@pytest.mark.parametrize('heading',[
    'Research Article', 'Original Article', 'Research Articles', 'Original articles',
    'Original papers', 'Original study articles',
    'Clinical/Basic Science Research Article', 'Experimental Section: Research Article'])
def test_explicit_research_heading_refines_only_coarse_root_type(heading: str) -> None:
    from experiments.fig8_extension.publisher_delta_apply import publisher_items
    record={'source':'pmc','route':'pmc_archive_priority','url':'https://example.org/PMC1','raw_receipt':'front.xml.gz','parsed':{'doi':['10.1234/a'],'title':['An empirical study'],'issn':['1234-5678'],'journal_titles':['Journal'],'article_type':'other','categories':[heading],'dates':[]}}
    value=publisher_items(record)[0]
    assert value['kind']=='research-article'
    assert value['native_section_type_evidence']['native_article_type']=='other'
    record['parsed']['article_type']='review'
    assert publisher_items(record)[0]['kind']=='review'
    record['parsed']['article_type']='other';record['parsed']['categories']=['Article']
    assert publisher_items(record)[0]['kind']=='other'
    record['parsed']['categories']=[heading];record['parsed']['abstract']='In this review, we discuss published studies.'
    assert publisher_items(record)[0]['scope_review_required'] is True
