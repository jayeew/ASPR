from experiments.fig8_extension.fig8b_qualification_decide import decide, nlm_indexed_research


def evidence(**changes: object) -> dict:
    item = dict(id='indexed', family='indexed_nlm_research', identity='matched',
                index_source='MED', review_rule='user_authorized_nlm_explicit_study_20261026',
                pmid='123', raw_sha256='raw', record_sha256='record', issns=['1234-5678'],
                categories=['Randomized Controlled Trial', 'Journal Article'],
                kind='Randomized Controlled Trial', title='Randomized treatment comparison',
                online_date='2024-03-02', date_precision='day', journal_titles=[],
                **{'class': 'unresolved_type'})
    item.update(changes)
    return item


def test_explicit_indexed_rule_retains_source_and_requires_specific_type() -> None:
    result = decide([evidence()], [])
    assert result['status'] == 'included_verified_research'
    assert 'secondary indexing evidence' in result['reason']
    assert not nlm_indexed_research(evidence(categories=['Journal Article']))
    assert not nlm_indexed_research(evidence(index_source='PMC'))
    assert not nlm_indexed_research(evidence(review_rule=''))


def test_exclusions_and_native_conflicts_cannot_be_overridden() -> None:
    for label in ['Case Reports', 'Review', 'Study Protocol', 'Data Note']:
        assert not nlm_indexed_research(evidence(categories=['Clinical Trial', label]))
        assert decide([evidence()], [{'types':[label]}])['status'] != 'included_verified_research'
    native = evidence(id='native', family='archived_jats', kind='review', **{'class':'nonprimary'})
    assert decide([evidence(), native], [])['status'] == 'pending_conflicting_primary_types'
    assert decide([evidence(), evidence(id='bad',identity='mismatch')], [])['status'] == 'pending_primary_identity_mismatch'
    assert decide([evidence(online_date='2022-03-02')], [])['status'] == 'excluded_primary_online_year_outside_window'
    assert not nlm_indexed_research(evidence(title='Study protocol for a randomized trial'))


def test_additional_explicit_study_types_are_not_topic_headings() -> None:
    for label in ['Validation Study', 'Multicenter Study']:
        assert nlm_indexed_research(evidence(categories=[label]))
    for label in ['Validation Studies as Topic', 'Multicenter Studies as Topic']:
        assert not nlm_indexed_research(evidence(categories=[label]))


def test_named_trial_variants_keep_protocol_and_case_exclusions() -> None:
    for label in ['Adaptive Clinical Trial', 'Pragmatic Clinical Trial', 'Equivalence Trial',
                  'Clinical Trial, Veterinary', 'Randomized Controlled Trial, Veterinary',
                  'Observational Study, Veterinary', 'Twin Study']:
        assert nlm_indexed_research(evidence(categories=[label]))
        assert not nlm_indexed_research(evidence(categories=[label, 'Clinical Trial Protocol']))
        assert not nlm_indexed_research(evidence(categories=[label, 'Case Reports']))
