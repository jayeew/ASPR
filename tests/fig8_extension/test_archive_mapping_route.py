import copy
import json

from experiments.fig8_extension.archive_mapping_route import mapping_only


def test_routing_requires_complete_native_byline_and_country_links():
    a={'name':'Alice','publisher_affiliation_source':'raw.xml.gz',
       'publisher_author_identity_match':'unique_exact_name_within_matched_paper','countries':['US'],
       'publisher_affiliation_slots':[{'text':'Example University, USA','country_codes':['US'],'institution_ids':[]}]}
    p={'full_authorships_status':'publisher_byline_reconstructed_unique_exact_links',
       'author_list_may_be_truncated':False,'abstract':'research '*80,
       'authorships_source_receipt':'raw.xml.gz','authors_json':json.dumps([a])}
    assert mapping_only(p)
    for field,value in [('full_authorships_status','openalex'),('author_list_may_be_truncated',True),('abstract',''),('abstract',None)]:
        assert not mapping_only({**p,field:value})
    for field,value in [('country_conflict_retained',True),('secondary_country_conflict',True),('nature_v1_slots',[]),('countries',[]),('publisher_affiliation_slots',[])]:
        b={**a,field:value}; assert not mapping_only({**p,'authors_json':json.dumps([b])})
    b=copy.deepcopy(a);b['publisher_affiliation_slots'][0]['country_codes']=[]
    assert not mapping_only({**p,'authors_json':json.dumps([b])})
    assert not mapping_only({**p,'authors_json':json.dumps([a,a])})


def test_assessment_keeps_pending_qualification_and_missing_receipts(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from experiments.fig8_extension import archive_mapping_route as module
    folder=tmp_path/'fields/papers';folder.mkdir(parents=True)
    author={'name':'A','publisher_affiliation_source':'raw.xml.gz','countries':['US'],
            'publisher_author_identity_match':'unique_exact_name_within_matched_paper',
            'publisher_affiliation_slots':[{'text':'Institution, USA','country_codes':['US']}]}
    row={'work_id':'W1','doi':'10.1/x','authors_json':json.dumps([author]),
         'full_authorships_status':'publisher_byline_reconstructed_unique_exact_links',
         'author_list_may_be_truncated':False,'authorships_source_receipt':'raw.xml.gz','abstract':'research '*80}
    pq.write_table(pa.Table.from_pylist([row]),folder/'test.parquet')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    paper={'work_id':'W1','doi':'10.1/x','shard':'test.parquet','eligibility_status':'included_verified_research'}
    assert module.assess([('PMC1',[paper])]) == {}
    (tmp_path/'raw.xml.gz').write_bytes(b'retained source')
    assert 'PMC1' in module.assess([('PMC1',[paper])])
    assert module.assess([('PMC1',[{**paper,'eligibility_status':'pending_primary_type'}])]) == {}
    assert module.assess([('PMC1',[paper,{**paper,'work_id':'W2','eligibility_status':'pending_primary_type'}])]) == {}
    assert module.assess([('PMC1',[{**paper,'doi':'10.1/other'}])]) == {}
