from experiments.fig8_extension.fig8b_qualification_decide import decide, nature_protocol_format


def protocol() -> dict:
    return {'id': 'publisher', 'identity': 'matched', 'family': 'nature_page',
            'class': 'subtype_review_required', 'source_id': 'https://openalex.org/S109387254',
            'doi': '10.1038/s41596-024-00000-x', 'kind': 'Protocol', 'categories': [],
            'title': 'A validated laboratory procedure', 'online_date': '2024-02-01'}


def test_protocol_is_journal_specific() -> None:
    source = protocol()
    assert decide([source], [])['status'] == 'excluded_primary_nonresearch'
    assert nature_protocol_format({**source, 'kind': 'Protocol Extension'})
    assert nature_protocol_format({**source, 'kind': 'Protocol Update'})
    for patch in ({'source_id': 'different'}, {'family': 'crossref_publisher_deposit'},
                  {'kind': 'Article'}, {'class': 'integrity_hold'}, {'identity': 'mismatch'},
                  {'scope_review_required': True}, {'categories': ['Original Research']},
                  {'doi': '10.1038/s41586-024-00000-x'}):
        assert not nature_protocol_format({**source, **patch})


def test_generic_archive_refinement_preserves_real_conflicts() -> None:
    source = protocol()
    archive = {**source, 'id': 'archive', 'family': 'archived_jats', 'class': 'research',
               'kind': 'research-article', 'categories': ['Article'], 'issns': ['1750-2799'],
               'journal_titles': ['Nature protocols']}
    assert decide([source, archive], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([source, {**archive, 'issns': []}], [])['status'] == 'excluded_primary_nonresearch'
    for patch in ({'doi': 'other'}, {'title': 'Another manuscript'},
                  {'categories': ['Research Article']}, {'issns': ['0000-0000']}):
        assert decide([source, {**archive, **patch}], [])['status'] == 'pending_conflicting_primary_types'
    assert decide([source, {**archive, 'class': 'integrity_hold'}], [])['status'] == 'pending_integrity_review'
