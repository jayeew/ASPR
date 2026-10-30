from experiments.fig8_extension.fig8b_qualification_decide import decide, journal_research_format


def test_method_report_keeps_date_identity_and_conflict_guards() -> None:
    item = {'id': 'methods-report', 'identity': 'matched', 'family': 'archived_jats',
            'class': 'subtype_review_required', 'kind': 'brief-report',
            'journal_titles': ['Cell Reports Methods'], 'issns': ['2667-2375'],
            'doi': '10.1016/j.crmeth.2023.100001', 'categories': ['Report'],
            'title': 'A reproducible assay', 'online_date': '', 'date_precision': ''}
    assert journal_research_format(item)
    assert decide([item], [])['status'] == 'pending_primary_online_date'
    dated = {**item, 'online_date': '2023-01-01', 'date_precision': 'day'}
    assert decide([dated], [])['status'] == 'included_verified_research'
    for delta in [{'identity': 'mismatch'}, {'journal_titles': ['Cell Reports']},
                  {'issns': []}, {'categories': ['Protocol Review']},
                  {'kind': 'letter'}, {'scope_review_required': True},
                  {'class': 'integrity_hold'}]:
        assert not journal_research_format({**item, **delta})
    assert decide([dated], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'


def test_incomplete_duplicate_requires_same_doi_title_journal() -> None:
    precise = {'id': 'precise', 'identity': 'matched', 'family': 'archived_jats',
               'class': 'subtype_review_required', 'kind': 'brief-report',
               'journal_titles': ['Cell Reports Methods'], 'issns': ['2667-2375'],
               'doi': '10.1016/j.crmeth.2023.100001', 'categories': ['Report'],
               'title': 'A reproducible assay', 'online_date': '2023-01-01', 'date_precision': 'day'}
    partial = {**precise, 'id': 'partial', 'issns': [], 'categories': [], 'online_date': ''}
    assert decide([precise, partial], [])['status'] == 'included_verified_research'
    for delta in [{'doi': 'other'}, {'title': 'Other assay'}, {'journal_titles': ['Other']}]:
        assert decide([precise, {**partial, **delta}], [])['status'] != 'included_verified_research'
