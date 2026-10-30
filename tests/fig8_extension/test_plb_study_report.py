from __future__ import annotations
import hashlib
import pytest
from experiments.fig8_extension.fig8b_qualification_decide import decide


def evidence(**changes: object) -> dict:
    abstract='We calculate the scattering cross section using a new perturbative expansion.'
    item=dict(id='sco',family='publisher_elsevier',identity='matched',doi='10.1016/j.physletb.2024.139214',journal_titles=['Physics Letters B'],kind='Short communication',categories=['Short communication'],native_pit='sco',native_journal_code='PLB',native_research_report_abstract=abstract,native_research_report_quote='We calculate',native_abstract_sha256=hashlib.sha256(abstract.encode()).hexdigest(),type_policy_receipt='../../supplement/target_500k/plb_cached_type_review/rule.json',title='Calculation of a scattering cross section',online_date='2024-12-01',date_precision='day',**{'class':'unresolved_type'})
    item.update(changes)
    return item


def test_native_research_report_and_date_required() -> None:
    assert decide([evidence()],[])['status']=='included_verified_research'
    assert decide([evidence(online_date='')],[])['status']=='pending_primary_online_date'
    assert decide([evidence(online_date='2022-12-01')],[])['status']=='excluded_primary_online_year_outside_window'


@pytest.mark.parametrize('changes',[{'native_pit':'mis'},{'native_journal_code':'NUPHB'},{'native_research_report_quote':''},{'native_abstract_sha256':'wrong'},{'scope_review_required':True},{'categories':['Short communication','Data Paper']},{'title':'A systematic review of scattering'}])
def test_ambiguous_or_conflicting_reports_remain_pending(changes: dict) -> None:
    assert decide([evidence(**changes)],[])['status']!='included_verified_research'


def test_index_review_conflict_is_preserved() -> None:
    assert decide([evidence()],[{'types':['Review'],'receipt':'index.json'}])['status']=='pending_primary_secondary_type_conflict'
