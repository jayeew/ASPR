"""Keep STAR Protocols format decisions limited to identity-matched native evidence."""
from experiments.fig8_extension.fig8b_qualification_decide import decide, star_protocol_format


def native_protocol() -> dict:
    return {
        'id': 'precise', 'identity': 'matched', 'family': 'archived_jats',
        'kind': 'research-article', 'class': 'subtype_review_required',
        'categories': ['Protocol'], 'journal_titles': ['STAR Protocols'],
        'issns': ['2666-1667'], 'doi': '10.1016/j.xpro.example',
        'title': 'Example experimental procedure', 'online_date': '2024-01-01',
    }


def test_only_native_journal_protocol_is_excluded() -> None:
    evidence = native_protocol()
    assert decide([evidence], [])['status'] == 'excluded_primary_nonresearch'
    for change in [
        {'identity': 'mismatch'}, {'family': 'crossref_publisher_deposit'},
        {'issns': []}, {'journal_titles': ['Cell']}, {'categories': ['Methods']},
        {'categories': ['Protocol', 'Case Report']}, {'class': 'integrity_hold'},
        {'scope_review_required': True}, {'class': 'research'},
    ]:
        assert not star_protocol_format({**evidence, **change})


def test_same_paper_legacy_front_can_be_refined() -> None:
    precise = native_protocol()
    legacy = {**precise, 'id': 'legacy', 'class': 'research', 'categories': [], 'issns': []}
    assert decide([legacy, precise], [])['status'] == 'excluded_primary_nonresearch'
    for change in [
        {'doi': '10.1/other'}, {'title': 'Another paper'}, {'journal_titles': ['Cell']},
        {'issns': ['0000-0000']}, {'family': 'publisher_elsevier'},
        {'scope_review_required': True}, {'categories': ['Original Research']},
    ]:
        assert decide([{**legacy, **change}, precise], [])['status'] == 'pending_conflicting_primary_types'
    assert decide([legacy], [])['status'] == 'included_verified_research'


def test_integrity_and_duplicate_decisions_remain_prioritized() -> None:
    precise = native_protocol()
    concern = {**precise, 'id': 'concern', 'class': 'integrity_hold', 'kind': 'retracted-publication'}
    assert decide([precise, concern], [])['status'] == 'pending_integrity_review'
    assert decide([precise], [], duplicate_of='retained-work')['status'] == 'excluded_duplicate_wrong_journal'
