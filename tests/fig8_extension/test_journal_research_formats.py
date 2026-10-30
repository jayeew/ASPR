from experiments.fig8_extension.fig8b_qualification_decide import decide


def evidence(**changes: object) -> dict:
    return {'id': 'brief', 'identity': 'matched', 'class': 'subtype_review_required',
            'source_id': 'https://openalex.org/S127827428', 'family': 'nature_catalogue',
            'kind': 'Brief Communication', 'title': 'A new experimental imaging method',
            'categories': [], 'online_date': '2024-02-02', **changes}


def test_journal_specific_format_is_not_global_short_article_acceptance() -> None:
    assert decide([evidence()], [])['status'] == 'included_verified_research'
    for changed in [{'source_id': 'other'}, {'family': 'archived_jats'}, {'kind': 'brief-report'},
                    {'title_scope_flag': True}, {'scope_review_required': True},
                    {'categories': ['Review']}, {'title': 'A systematic review of imaging methods'}]:
        assert decide([evidence(**changed)], [])['status'] == 'pending_special_subtype_review'


def test_format_keeps_date_identity_primary_secondary_and_integrity_checks() -> None:
    assert decide([evidence(online_date='')], [])['status'] == 'pending_primary_online_date'
    assert decide([evidence(online_date='2022-01-01')], [])['status'] == 'excluded_primary_online_year_outside_window'
    assert decide([evidence(identity='title_or_journal_mismatch')], [])['status'] == 'pending_primary_identity_mismatch'
    assert decide([evidence()], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
    other = evidence(id='other', kind='Review', **{'class': 'nonprimary'})
    assert decide([evidence(), other], [])['status'] == 'pending_conflicting_primary_types'
    other['class'] = 'integrity_hold'
    assert decide([evidence(), other], [])['status'] == 'pending_integrity_review'


def test_methods_analysis_resource_are_journal_specific_research_formats() -> None:
    for kind in ['Analysis', 'Resource']:
        assert decide([evidence(kind=kind, categories=[kind])], [])['status'] == 'included_verified_research'
        assert decide([evidence(kind=kind, source_id='https://openalex.org/S106963461')], [])['status'] == 'pending_special_subtype_review'
        assert decide([evidence(kind=kind, categories=['Study Protocol'])], [])['status'] == 'pending_special_subtype_review'
        assert decide([evidence(kind=kind, categories=['Expression of Concern'])], [])['status'] == 'pending_special_subtype_review'


def test_acs_letter_needs_native_label_journal_and_all_existing_guards() -> None:
    item = evidence(source_id='https://openalex.org/S143846845', family='archived_jats',
                    kind='rapid-communication', categories=['Letter'], issns=['1530-6984'],
                    **{'class': 'unresolved_type'})
    assert decide([item], [])['status'] == 'included_verified_research'
    for changes in [{'issns': ['other']}, {'categories': []}, {'categories': ['Letter', 'Review']},
                    {'kind': 'letter'}, {'family': 'crossref_publisher_deposit'},
                    {'title_scope_flag': True}, {'scope_review_required': True},
                    {'title': 'Systematic review of new materials'}]:
        assert decide([{**item, **changes}], [])['status'] != 'included_verified_research'
    assert decide([{**item, 'online_date': ''}], [])['status'] == 'pending_primary_online_date'
    assert decide([item], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
    assert decide([item, evidence(id='review', **{'class': 'nonprimary'})], [])['status'] == 'pending_conflicting_primary_types'
    assert decide([{**item, 'identity': 'unmatched'}], [])['status'] == 'pending_primary_identity_mismatch'
