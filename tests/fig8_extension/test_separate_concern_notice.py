from __future__ import annotations

from experiments.fig8_extension.fig8b_qualification_decide import decide, type_class


def notice(**changes: object) -> dict:
    return {'id':'n1','identity':'matched','family':'archived_jats','kind':'expression-of-concern',
            'class':type_class('expression-of-concern','archived_jats'),
            'title':'Expression of Concern for Original study', 'online_date':'2025-11-24',
            'separate_notice_rule':'native_related_article_object_of_concern',
            'notice_target_dois':['10.1234/target'], **changes}


def test_separate_notice_is_excluded_and_prior_notice_hold_does_not_block() -> None:
    old = notice(id='n0',separate_notice_rule='')
    assert decide([old,notice()],[])['status'] == 'excluded_primary_nonresearch'


def test_research_integrity_flag_without_separate_notice_remains_unresolved() -> None:
    assert decide([notice(title='Original study')],[])['status'] == 'pending_integrity_review'
    assert decide([notice(notice_target_dois=[])],[])['status'] == 'pending_integrity_review'
    assert decide([notice(kind='retracted-publication')],[])['status'] == 'pending_integrity_review'


def test_notice_with_scope_conflict_is_not_promoted() -> None:
    assert decide([notice(scope_review_required=True)],[])['status'] == 'pending_integrity_review'


def test_separate_notice_can_identify_target_with_pmc_instead_of_doi() -> None:
    assert decide([notice(notice_target_dois=[],notice_target_pmcids=['PMC12345'])],[])['status'] == 'excluded_primary_nonresearch'
