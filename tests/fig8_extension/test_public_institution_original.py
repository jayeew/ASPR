from __future__ import annotations

import pytest

from experiments.fig8_extension.fig8b_qualification_decide import decide, public_institution_original_research
from tests.fig8_extension.test_virta_original import date_evidence


def source(family: str) -> dict:
    common={'id':'native','family':family,'identity':'matched','class':'unresolved_type',
            'title':'Measured properties','categories':[],'online_date':'','issns':['1234-5678'],
            'source_record_id':'official:123','raw_sha256':'raw','record_sha256':'record',
            'source_peer_reviewed':True,'journal_titles':['Journal']}
    if family=='institution_muenster':
        return {**common,'kind':'Research article (journal)','source_publication_status':'Published',
                'type_policy_receipt':'../../supplement/target_500k/muenster_public/rule.json'}
    return {**common,'kind':'1.01 Original Scientific Article','source_typology_code':'1.01',
            'source_approved':True,'source_publication_status':'Objavljeno','source_container_type':'Revija',
            'type_policy_receipt':'../../supplement/target_500k/slovenia_public/rule.json'}


@pytest.mark.parametrize('family',['institution_muenster','institution_slovenia'])
def test_source_explicit_original_type_still_needs_online_year(family: str) -> None:
    item=source(family)
    assert public_institution_original_research(item)
    assert decide([item],[])['status']=='pending_primary_online_date'
    assert decide([item,date_evidence()],[])['status']=='included_verified_research'
    assert decide([item,date_evidence()],[{'types':['Case Reports']}])['status']=='pending_primary_secondary_type_conflict'


@pytest.mark.parametrize('field,value',[('kind','Review'),('source_peer_reviewed',False),
    ('source_publication_status','In press'),('source_approved',False),('source_container_type','Book'),
    ('source_typology_code','1.02'),('scope_review_required',True),('raw_sha256','')])
def test_native_typology_and_curation_negative_controls(field: str,value: object) -> None:
    item=source('institution_slovenia');item[field]=value
    assert not public_institution_original_research(item)
