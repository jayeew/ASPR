from __future__ import annotations
import pytest
from experiments.fig8_extension.fig8b_qualification_decide import decide,virta_original_research


def evidence() -> dict:
    return {'id':'v1','family':'institution_virta','identity':'matched','kind':'A1','class':'unresolved_type','title':'Measurements of a new material','categories':['A1'],'issns':['1234-5678'],'journal_titles':['Journal'],'online_date':'','date_precision':'not_an_online_date','type_policy_receipt':'../../supplement/target_500k/virta_public/rule.json','source_record_status':'2','source_record_id':'p1','raw_sha256':'abc','record_sha256':'def','virta_attributes':{'MuotoKoodi':'0','YleisoKoodi':'0','EmojulkaisuntyyppiKoodi':'0','ArtikkelityyppiKoodi':'1','VertaisarvioituKytkin':'1'}}


def date_evidence() -> dict:
    return {'id':'d1','family':'crossref_publisher_deposit','identity':'matched','class':'unresolved_type','kind':'journal-article','title':'Measurements of a new material','categories':[],'issns':['1234-5678'],'online_date':'2024-02-03'}


def test_approved_original_requires_independent_online_date() -> None:
    e=evidence()
    assert decide([e],[])['status']=='pending_primary_online_date'
    d=decide([e,date_evidence()],[])
    assert d['status']=='included_verified_research'
    assert 'institution-reported' in d['reason']
    assert d['online_year']==2024


@pytest.mark.parametrize('field,value',[
    ('source_record_status','1'),('identity','title_or_journal_mismatch'),
    ('kind','A2'),('raw_sha256',''),('scope_review_required',True)])
def test_unapproved_review_mismatch_or_content_hold_not_promoted(field: str,value: str|bool) -> None:
    e=evidence();e[field]=value
    assert not virta_original_research(e)


def test_declared_data_article_cannot_become_original() -> None:
    e=evidence();e['virta_attributes']['ArtikkelityyppiKoodi']='3'
    assert not virta_original_research(e)
    assert decide([e,date_evidence()],[])['status']!='included_verified_research'


def test_indexed_case_report_conflict_is_retained() -> None:
    e=evidence()
    d=decide([e,date_evidence()],[{'types':['Case Reports']}])
    assert d['status']=='pending_primary_secondary_type_conflict'


def test_out_of_window_electronic_year_is_excluded() -> None:
    d=date_evidence();d['online_date']='2022-12-01'
    assert decide([evidence(),d],[])['status']=='excluded_primary_online_year_outside_window'


def test_institutional_classification_does_not_override_native_identity_conflict() -> None:
    native={'id':'native','family':'publisher_elsevier','identity':'title_or_journal_mismatch','class':'research','kind':'Research Article','title':'A different paper','categories':[],'online_date':'2024-02-03'}
    assert decide([evidence(),date_evidence(),native],[])['status']=='pending_primary_identity_mismatch'
