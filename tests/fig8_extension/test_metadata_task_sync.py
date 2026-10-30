import json
from experiments.fig8_extension.metadata_task_sync import requests_for


def test_known_author_country_does_not_hide_unresolved_second_slot():
    author={'name':'Jane Doe','position':0,'countries':['GB'],'institution_ids_or_verified_local_entities':['I1'],
            'publisher_affiliation_source':'receipt','publisher_affiliation_slots':[
                {'text':'Known institution','institution_ids':['I1'],'country_codes':['GB']},
                {'text':'Other institution','institution_ids':[],'country_codes':[]}]}
    paper={'work_id':'W1','doi':'10.1/a','source_id':'S1','journal_name':'Journal','authors_json':json.dumps([author]),'abstract':'An abstract','author_list_may_be_truncated':False}
    rows=requests_for(paper,{'nature_portfolio_at_verified_online_year':'False','online_year':2024})
    units=[r for r in rows if r[0]=='institution']
    assert len(units)==1 and units[0][3]['name']=='Other institution'
    assert units[0][3]['missing_identity'] and units[0][3]['missing_country']
    assert any(r[0]=='metadata' for r in rows)
    author['nature_v1_slots']=[{'identity_missing':True}]
    paper['authors_json']=json.dumps([author])
    assert not any(r[0] in {'institution','university_ownership'} for r in requests_for(paper,{'nature_portfolio_at_verified_online_year':'False','online_year':2024}))


def test_known_identity_and_country_keep_unknown_university_in_queue():
    author={'name':'Jane Doe','position':0,'countries':['CN'],
            'institution_ids_or_verified_local_entities':['local-lab'],
            'publisher_affiliation_source':'publisher.xml',
            'publisher_affiliation_slots':[{'text':'Named laboratory, China',
                'institution_ids':['local-lab'],'country_codes':['CN'],
                'identity_status':'official_exact_declared_name',
                'university_ownership_status':'unverified','official_identity_evidence':'official.json'}]}
    paper={'work_id':'W1','doi':'10.1/a','source_id':'S1','journal_name':'Journal',
           'authors_json':json.dumps([author]),'abstract':'Abstract','author_list_may_be_truncated':False}
    decision={'nature_portfolio_at_verified_online_year':'False','online_year':2024}
    rows=requests_for(paper,decision)
    assert not any(r[0]=='institution' for r in rows)
    ownership=[r for r in rows if r[0]=='university_ownership']
    assert len(ownership)==1 and ownership[0][1]=='local-lab'
    assert ownership[0][3]['source']=='publisher.xml' and ownership[0][4]==0
    author['publisher_affiliation_slots'][0]['university_ownership_status']='officially_confirmed_nonuniversity'
    paper['authors_json']=json.dumps([author])
    assert not any(r[0]=='university_ownership' for r in requests_for(paper,decision))
    author['publisher_affiliation_slots'][0]['university_ownership_status']='unverified'
    paper['authors_json']=json.dumps([author])
    assert not any(r[0]=='university_ownership' for r in requests_for(paper,decision,{'main_personal_direct':{'local-lab'}}))
