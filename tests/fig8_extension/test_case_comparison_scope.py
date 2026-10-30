from experiments.fig8_extension.fig8b_qualification_decide import decide


def evidence() -> list[dict]:
    native = {'id': 'native', 'identity': 'matched', 'doi': '10.1128/mra.test',
              'family': 'archived_jats', 'kind': 'case-report',
              'class': 'subtype_review_required', 'issns': ['2576-098X'],
              'categories': ['Case Report'], 'online_date': '2023-01-12'}
    review = {'id': 'scope', 'identity': 'matched', 'doi': native['doi'],
              'family': 'reviewed_comparison_scope', 'kind': 'Original data case_report announcement',
              'class': 'unresolved_type', 'online_date': '',
              'scope_rule': 'case_report_extension_20261026',
              'comparison_scope': 'extension_only', 'source_evidence_ids': ['native']}
    return [native, review]


def test_case_report_exclusion_is_not_nonresearch_and_preserves_year() -> None:
    result = decide(evidence(), [])
    assert result['status'] == 'excluded_case_report_main_scope'
    assert result['online_year'] == 2023
    assert set(result['evidence_ids']) == {'native', 'scope'}
    assert decide(evidence()[:1], [])['status'] == 'pending_special_subtype_review'


def test_scope_needs_matching_native_subtype_and_identity() -> None:
    for changes in [{'kind': 'research-article', 'categories': ['Cohort Study']}, {'kind': 'case-series', 'categories': ['Case Series']},
                    {'doi': 'different'}, {'identity': 'mismatch'},
                    {'scope_review_required': True}]:
        native, review = evidence()
        native.update(changes)
        assert decide([native, review], [])['status'] != 'excluded_case_report_main_scope'


def test_existing_date_integrity_and_duplicate_holds_take_precedence() -> None:
    native, review = evidence()
    other = {**native, 'id': 'other', 'online_date': '2024-01-12'}
    assert decide([native, review, other], [])['status'] == 'pending_conflicting_primary_online_years'
    other.update({'class': 'integrity_hold'})
    assert decide([native, review, other], [])['status'] == 'pending_integrity_review'
    assert decide([native, review], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
    assert decide([native, review], [], 'another-work')['status'].startswith('excluded_duplicate')


def test_explicit_springer_case_category_over_generic_originalpaper() -> None:
    native, review = evidence()
    native.update(family='publisher_springer', kind='OriginalPaper', categories=['OriginalPaper', 'Case Report'])
    assert decide([native,review],[])['status']=='excluded_case_report_main_scope'
    native.update(categories=['OriginalPaper'])
    assert decide([native,review],[])['status']!='excluded_case_report_main_scope'


def test_mdpi_case_set_with_source_link() -> None:
    native, review = evidence()
    native.update(family='publisher_mdpi_oai', kind='case-report', categories=[])
    assert decide([native,review],[])['status']=='excluded_case_report_main_scope'
    review['source_evidence_ids']=[]
    assert decide([native,review],[])['status']!='excluded_case_report_main_scope'
