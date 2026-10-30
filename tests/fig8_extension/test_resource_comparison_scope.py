from experiments.fig8_extension.fig8b_qualification_decide import decide


def evidence() -> list[dict]:
    native = {'id': 'native', 'identity': 'matched', 'doi': '10.1128/mra.test',
              'family': 'archived_jats', 'kind': 'brief-report',
              'class': 'subtype_review_required', 'issns': ['2576-098X'],
              'categories': ['Genome Sequences'], 'online_date': '2023-01-12'}
    review = {'id': 'scope', 'identity': 'matched', 'doi': native['doi'],
              'family': 'reviewed_comparison_scope', 'kind': 'Original data resource announcement',
              'class': 'unresolved_type', 'online_date': '',
              'scope_rule': 'mra_genome_sequences_extension_20261025',
              'comparison_scope': 'extension_only', 'source_evidence_ids': ['native']}
    return [native, review]


def test_resource_exclusion_is_not_nonresearch_and_preserves_year() -> None:
    result = decide(evidence(), [])
    assert result['status'] == 'excluded_resource_announcement_main_scope'
    assert result['online_year'] == 2023
    assert set(result['evidence_ids']) == {'native', 'scope'}
    assert decide(evidence()[:1], [])['status'] == 'pending_special_subtype_review'


def test_scope_needs_matching_native_subtype_and_identity() -> None:
    for changes in [{'issns': ['0000-0000']}, {'categories': ['Protocols and Workflows']},
                    {'doi': 'different'}, {'identity': 'mismatch'},
                    {'scope_review_required': True}]:
        native, review = evidence()
        native.update(changes)
        assert decide([native, review], [])['status'] != 'excluded_resource_announcement_main_scope'


def test_existing_date_integrity_and_duplicate_holds_take_precedence() -> None:
    native, review = evidence()
    other = {**native, 'id': 'other', 'online_date': '2024-01-12'}
    assert decide([native, review, other], [])['status'] == 'pending_conflicting_primary_online_years'
    other.update({'class': 'integrity_hold'})
    assert decide([native, review, other], [])['status'] == 'pending_integrity_review'
    assert decide([native, review], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
    assert decide([native, review], [], 'another-work')['status'].startswith('excluded_duplicate')
