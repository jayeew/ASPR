import pytest

from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


@pytest.mark.parametrize('sid,formats', [
    ('S4210214063', ['Letter', 'Brief Communication', 'Resource']),
    ('S4210232714', ['Letter', 'Brief Communication', 'Resource']),
    ('S151741590', ['Resource', 'Technical Report']),
    ('S2764671299', ['Letter', 'Brief Communication', 'Resource']),
    ('S2764926557', ['Analysis', 'Brief Communication', 'Resource']),
    ('S139253143', ['Analysis', 'Brief Communication', 'Resource', 'Technical Report']),
    ('S103895331', ['Letter']),
    ('S41063453', ['Letter', 'Resource', 'Technical Report']),
    ('S4210228084', ['Analysis', 'Brief Communication', 'Resource']),
    ('S2298632', ['Brief Communication', 'Resource', 'Technical Report']),
    ('S203256638', ['Brief Communication', 'Resource']),
])
def test_journal_specific_native_formats(sid: str, formats: list[str]) -> None:
    for kind in formats:
        e = {'id': 'native', 'identity': 'matched', 'family': 'nature_catalogue',
             'source_id': 'https://openalex.org/'+sid, 'kind': kind,
             'class': 'subtype_review_required', 'title': 'An experimental investigation',
             'online_date': '2024-02-03', 'categories': []}
        assert journal_research_format(e)
        assert decide([e], [], '')['status'] == 'included_verified_research'
        for change in [{'identity': 'mismatch'}, {'family': 'archived_jats'},
                       {'scope_review_required': True}, {'categories': ['Review']},
                       {'title': 'Systematic review of the literature'}]:
            assert not journal_research_format({**e, **change})


def test_mixed_or_unverified_formats_stay_unresolved() -> None:
    for sid, kind in [('S4210214063', 'Analysis'), ('S4210214063', 'Technical Report'),
                      ('S151741590', 'Letter'), ('S103895331', 'Resource'),
                      ('S203256638', 'Analysis'), ('S203256638', 'Letter'),
                      ('S203256638', 'Technical Report'), ('S2298632', 'Registered Report')]:
        e = {'id': 'held', 'identity': 'matched', 'family': 'nature_page',
             'source_id': 'https://openalex.org/'+sid, 'kind': kind,
             'class': 'subtype_review_required', 'online_date': '2024-02-03'}
        assert not journal_research_format(e)
        assert decide([e], [], '')['status'] == 'pending_special_subtype_review'
