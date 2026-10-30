from __future__ import annotations

from experiments.fig8_extension.fig8b_qualification_decide import decide, type_class


def item(issn: str='1367-4803', heading: str='Applications Note', kind: str='brief-report', **extra: object) -> dict:
    return {'id':heading or 'empty','family':'archived_jats','identity':'matched','issns':[issn],
        'kind':kind,'categories':[heading] if heading else [],'class':type_class(kind,'archived_jats',[heading]),
        'title':'A new computational method','online_date':'2024-02-03',**extra}


def test_named_formats_require_journal_and_category() -> None:
    assert decide([item()],[])['status']=='included_verified_research'
    assert decide([item('2635-0041','Application Note')],[])['status']=='included_verified_research'
    assert decide([item('2635-0041',heading='')],[])['status']=='pending_special_subtype_review'
    assert decide([item('2352-3042','Rapid Communication')],[])['status']=='included_verified_research'
    assert decide([item('1467-5463','Case Study','case-report')],[])['status']=='included_verified_research'
    assert decide([item('9999-9999')],[])['status']=='pending_special_subtype_review'
    assert decide([item(heading='')],[])['status']=='pending_special_subtype_review'


def test_precise_heading_refines_only_same_journal_and_type() -> None:
    assert decide([item(heading=''),item()],[])['status']=='included_verified_research'
    assert decide([item(heading='',issn='9999-9999'),item()],[])['status']=='pending_special_subtype_review'
    assert decide([item(kind='letter',heading=''),item()],[])['status']=='pending_special_subtype_review'


def test_content_identity_and_date_holds_remain() -> None:
    assert decide([item(title='A systematic review')],[])['status']=='pending_special_subtype_review'
    assert decide([item(scope_review_required=True)],[])['status']=='pending_special_subtype_review'
    assert decide([item(online_date='')],[])['status']=='pending_primary_online_date'
    assert decide([item(identity='title_or_journal_mismatch')],[])['status']=='pending_primary_identity_mismatch'
    assert decide([item()],[{'types':['Review']}])['status']=='pending_primary_secondary_type_conflict'
    assert decide([item(),item(heading='Review',kind='review-article')],[])['status']=='pending_conflicting_primary_types'


def test_frontiers_native_heading_and_publisher_required() -> None:
    original = item('1664-8021', 'Brief Research Report', doi='10.3389/fgene.2024.123', publishers=['Frontiers Media SA'], journal_titles=['Frontiers in Genetics'])
    assert decide([original], [])['status'] == 'included_verified_research'
    assert decide([{**original, 'journal_titles': ['Acta Biochimica Polonica']}], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**original, 'journal_titles': []}], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**original, 'publishers': ['Other']}], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**original, 'categories': []}], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**original, 'title': 'A case report'}], [])['status'] == 'pending_special_subtype_review'


def test_pbj_brief_communication_keeps_conflicts_and_requires_native_heading() -> None:
    original = item('1467-7652', 'Brief Communication')
    assert decide([original], [])['status'] == 'included_verified_research'
    assert decide([item('1467-7644', 'Brief Communications')], [])['status'] == 'included_verified_research'
    assert decide([item('1467-7652', '')], [])['status'] == 'pending_special_subtype_review'
    assert decide([item('9999-9999', 'Brief Communication')], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**original, 'scope_review_required': True}], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**original, 'online_date': ''}], [])['status'] == 'pending_primary_online_date'
    assert decide([original, item('1467-7652', 'Review', 'review-article')], [])['status'] == 'pending_conflicting_primary_types'


def test_pbj_incomplete_copy_requires_exact_paper_and_journal_identity() -> None:
    precise = item('1467-7652', 'Brief Communication', doi='10.1111/pbi.example',
                   journal_titles=['Plant Biotechnology Journal'])
    legacy = {**precise, 'id': 'legacy', 'issns': [], 'categories': []}
    assert decide([legacy, precise], [])['status'] == 'included_verified_research'
    for changed in ({'doi': '10.1111/pbi.other'}, {'title': 'Other paper'},
                    {'journal_titles': ['Other journal']}, {'issns': ['9999-9999']},
                    {'scope_review_required': True}):
        assert decide([{**legacy, **changed}, precise], [])['status'] == 'pending_special_subtype_review'
