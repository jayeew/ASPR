from experiments.fig8_extension.fig8b_qualification_decide import decide, type_class


def test_ampersand_qa_reuses_nonprimary_definition_without_changing_plain_qa() -> None:
    assert type_class('Q&A', 'nature_catalogue') == 'nonprimary'
    assert type_class('Q & A', 'nature_page') == 'nonprimary'
    assert type_class('QA', 'archived_jats') == 'unresolved_type'
    item = {'id':'qa','kind':'Q&A','identity':'matched','family':'nature_catalogue',
            'class':'unresolved_type','online_date':'2024-05-01','title':'An interview'}
    assert decide([item], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([{**item,'identity':'unmatched'}], [])['status'] == 'pending_primary_identity_mismatch'
    assert decide([{**item,'class':'integrity_hold'}], [])['status'] == 'pending_integrity_review'
    research = {**item,'id':'research','kind':'Research Article','class':'research'}
    assert decide([item,research], [])['status'] == 'pending_conflicting_primary_types'


def test_primeview_is_scoped_to_reviewed_journals_and_preserves_conflicts() -> None:
    item = {'id': 'pv', 'kind': 'PrimeView', 'identity': 'matched',
            'family': 'nature_page', 'class': 'unresolved_type',
            'source_id': 'https://openalex.org/S4210231269',
            'online_date': '2024-05-01', 'title': 'A graphical summary'}
    assert decide([item], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([{**item, 'source_id': 'unreviewed'}], [])['status'] == 'pending_primary_type'
    assert decide([{**item, 'identity': 'unmatched'}], [])['status'] == 'pending_primary_identity_mismatch'
    assert decide([{**item, 'class': 'integrity_hold'}], [])['status'] == 'pending_integrity_review'
    research = {**item, 'id': 'research', 'kind': 'Research Article', 'class': 'research'}
    assert decide([item, research], [])['status'] == 'pending_conflicting_primary_types'
