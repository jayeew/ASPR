import gzip
import hashlib
import json

from experiments.fig8_extension import country_verified_identity as module


def test_source_bound_raw_identity_and_country_only_fill(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    raw=json.dumps({'response':{'id':'W1','doi':'https://doi.org/10.1/a','authorships':[
        {'author':{'id':'A1','display_name':'A. Li'},'raw_author_name':'Alice Li'}]}}).encode()
    (tmp_path/'raw.gz').write_bytes(gzip.compress(raw))
    proof={'author_id':'A1','canonical_name':'A. Li','publisher_raw_receipt':'native.xml.gz',
           'doi':'10.1/a','work_id':'W1','openalex_receipt':'raw.gz','openalex_raw_sha256':hashlib.sha256(raw).hexdigest(),'raw_author_name':'Alice Li'}
    receipt='fields/qualification/native.xml.gz'
    author={'author_id':'A1','name':'A. Li','countries':['US'],
            'publisher_affiliation_source':receipt,'publisher_author_identity_match':'unique_publisher_raw_name_and_original_author_id',
            'publisher_author_identity_evidence':proof,'publisher_affiliation_slots':[{'text':'Lab, Canada','country_codes':[]}]}
    source={'doi':['10.1/a'],'authors':[{'name':'Alice Li','affiliations':[{'text':'Lab, Canada'}]}]}
    value,n,persons=module.enrich([author],source,receipt,{'Lab, Canada':'CA'},'rule','terminal')
    assert n==1 and persons==0 and value[0]['countries']==['US']
    assert value[0]['country_conflict_retained'] and value[0]['name']=='A. Li'
    assert module.enrich(value,source,receipt,{'Lab, Canada':'CA'},'rule','terminal')==(value,0,0)
    assert module.enrich([{**author,'nature_v1_slots':[]}],source,receipt,{'Lab, Canada':'CA'},'rule','terminal')[1]==0
    assert module.match(author,[author],{**source,'doi':['10.1/wrong']},receipt) is None
    assert module.match(author,[author],{**source,'authors':source['authors']*2},receipt) is None
    assert module.match(author,[author],source,'other.xml.gz') is None
    broken={**author,'publisher_author_identity_evidence':{**proof,'openalex_raw_sha256':'bad'}}
    assert module.match(broken,[broken],source,receipt) is None


def test_orcid_requires_unique_checksum_valid_identity():
    a={'name':'A. Li','corroborated_orcid':'0000-0002-1825-0097',
       'publisher_author_identity_match':'unique_checksum_valid_orcid_in_both_paper_bylines'}
    s={'authors':[{'name':'Alice Li','orcid':a['corroborated_orcid'],'affiliations':[{'text':'Lab, Canada'}]}]}
    assert module.match(a,[a],s,'receipt')==s['authors'][0]
    assert module.match(a,[a],{**s,'authors':s['authors']*2},'receipt') is None
    assert module.match(a,[a,{**a,'name':'Other'}],s,'receipt') is None
    assert module.match({**a,'corroborated_orcid':'0000-0002-1825-0098'},[a],s,'receipt') is None
