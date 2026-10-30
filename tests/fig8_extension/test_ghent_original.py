from __future__ import annotations

import pytest

from experiments.fig8_extension.fig8b_qualification_decide import decide, ghent_original_research
from tests.fig8_extension.test_virta_original import date_evidence


def evidence() -> dict:
    return {'id':'g1', 'family':'institution_ghent', 'identity':'matched', 'class':'unresolved_type',
            'kind':'original', 'title':'Measurements of a new material', 'categories':['original'],
            'issns':['1234-5678'], 'journal_titles':['Journal'], 'online_date':'',
            'source_publication_type':'journalArticle', 'source_classification':'A1',
            'source_wos_type':'Article', 'source_publication_status':'published',
            'source_record_id':'record1', 'raw_sha256':'a', 'record_sha256':'b',
            'type_policy_receipt':'../../supplement/target_500k/ghent_public/rule.json'}


def test_original_type_requires_independent_online_date() -> None:
    assert decide([evidence()],[])['status']=='pending_primary_online_date'
    result=decide([evidence(),date_evidence()],[])
    assert result['status']=='included_verified_research'
    assert 'Ghent' in result['reason'] and 'institution-reported' in result['reason']
    assert 'Finnish' not in result['reason']


@pytest.mark.parametrize('field,value', [('kind','review'),('source_classification','A4'),
    ('source_wos_type','Review'),('source_publication_status','inpress'),('scope_review_required',True)])
def test_negative_source_controls_and_exclusions(field: str, value: str|bool) -> None:
    item=evidence();item[field]=value
    assert not ghent_original_research(item)
    assert decide([item,date_evidence()],[])['status']!='included_verified_research'


def test_secondary_case_conflict_and_rejected_identity_remain_pending() -> None:
    assert decide([evidence(),date_evidence()],[{'types':['Case Reports']}])['status']=='pending_primary_secondary_type_conflict'
    native={**date_evidence(),'id':'bad','identity':'title_or_journal_mismatch'}
    assert decide([evidence(),date_evidence(),native],[])['status']=='pending_primary_identity_mismatch'


@pytest.mark.parametrize('classification',['A2','A3'])
def test_explicit_original_peer_reviewed_articles_do_not_require_wos(classification: str) -> None:
    item={**evidence(),'source_classification':classification,'source_wos_type':''}
    assert decide([item,date_evidence()],[])['status']=='included_verified_research'
