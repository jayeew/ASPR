from experiments.fig8_extension.fig8b_qualification_decide import decide


def evidence(**updates: object) -> dict:
    return {'id': 'plos-format', 'identity': 'matched', 'family': 'publisher_plos',
            'issns': ['1553-7358'], 'kind': 'Methods', 'class': 'unresolved_type',
            'title': 'A new computational method', 'categories': ['Methods'],
            'online_date': '2024-02-01', **updates}


def test_plos_research_types_are_journal_and_source_specific() -> None:
    for issn, kinds in [('1553-7358', ['Methods', 'Software']),
                        ('1545-7885', ['Short Reports', 'Methods and Resources', 'Meta-Research Article', 'Update Article']),
                        ('1553-7404', ['Methods'])]:
        for kind in kinds:
            assert decide([evidence(issns=[issn], kind=kind, categories=[kind])], [])['status'] == 'included_verified_research'
    for changes in [{'issns': ['1932-6203']}, {'family': 'archived_jats'}, {'kind': 'Lab Protocol'}]:
        assert decide([evidence(**changes)], [])['status'] == 'pending_primary_type'


def test_plos_format_rules_preserve_all_existing_holds() -> None:
    for changes in [{'title': 'A systematic review of computational methods'}, {'categories': ['Study Protocol']},
                    {'title_scope_flag': True}, {'scope_review_required': True, 'class': 'subtype_review_required'}]:
        assert decide([evidence(**changes)], [])['status'] == 'pending_special_subtype_review'
    assert decide([evidence(online_date='')], [])['status'] == 'pending_primary_online_date'
    assert decide([evidence(online_date='2022-01-01')], [])['status'] == 'excluded_primary_online_year_outside_window'
    assert decide([evidence(identity='title_or_journal_mismatch')], [])['status'] == 'pending_primary_identity_mismatch'
    assert decide([evidence()], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
    integrity = evidence(id='integrity', kind='Expression of Concern', **{'class': 'integrity_hold'})
    assert decide([evidence(), integrity], [])['status'] == 'pending_integrity_review'


def test_plos_nonresearch_rule_does_not_override_conflicting_research() -> None:
    primer = evidence(issns=['1545-7885'], kind='Primer', categories=['Primer'])
    assert decide([primer], [])['status'] == 'excluded_primary_nonresearch'
    research = evidence(id='archive', family='archived_jats', kind='research-article', **{'class': 'research'})
    assert decide([primer, research], [])['status'] == 'pending_conflicting_primary_types'


def test_additional_plos_types_preserve_protocol_uncertainty() -> None:
    for issn, kind in [('1545-7885', 'Discovery Report'), ('1553-7374', 'Short Reports')]:
        item = evidence(issns=[issn], kind=kind, categories=[kind])
        assert decide([item], [])['status'] == 'included_verified_research'
        assert decide([dict(item, online_date='')], [])['status'] == 'pending_primary_online_date'
    for issn, kind in [('1553-7374', 'Pearls'), ('1935-2735', 'Viewpoints'),
                       ('1553-7404', 'Opinion Piece'), ('1932-6203', 'Registered Report Protocol'),
                       ('1545-7885', 'Unsolved Mystery')]:
        item = evidence(issns=[issn], kind=kind, categories=[kind])
        assert decide([item], [])['status'] == 'excluded_primary_nonresearch'
        research = evidence(id='archive', family='archived_jats', kind='research-article', **{'class': 'research'})
        assert decide([item, research], [])['status'] == 'pending_conflicting_primary_types'
    for issn, kind in [('1932-6203', 'Lab Protocol'), ('1553-7374', 'Opinion')]:
        assert decide([evidence(issns=[issn], kind=kind, categories=[kind])], [])['status'] == 'pending_primary_type'


def test_plos_one_planned_study_is_not_a_completed_research_report() -> None:
    protocol = evidence(issns=['1932-6203'], kind='Study Protocol', categories=['Study Protocol'],
                        **{'class': 'subtype_review_required'})
    assert decide([protocol], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([dict(protocol, scope_review_required=True)], [])['status'] == 'pending_special_subtype_review'
    assert decide([dict(protocol, family='archived_jats')], [])['status'] == 'pending_special_subtype_review'
    research = evidence(id='archive', family='archived_jats', kind='research-article', **{'class': 'research'})
    assert decide([protocol, research], [])['status'] == 'pending_conflicting_primary_types'
    assert decide([evidence(**{'class': 'subtype_review_required'})], [])['status'] == 'pending_special_subtype_review'


def test_notice_record_is_distinct_from_flagged_original_research() -> None:
    notice = evidence(kind='Expression of Concern', title='Expression of Concern: An original study',
                      **{'class': 'integrity_hold'})
    archive = dict(notice, id='archive', family='archived_jats', kind='expression-of-concern')
    assert decide([notice, archive], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([archive], [])['status'] == 'pending_integrity_review'
    original = dict(notice, title='An original study')
    assert decide([original], [])['status'] == 'pending_integrity_review'
    assert decide([notice, original], [])['status'] == 'pending_integrity_review'
    assert decide([dict(notice, scope_review_required=True)], [])['status'] == 'pending_integrity_review'
    assert decide([dict(notice, identity='title_or_journal_mismatch')], [])['status'] == 'pending_primary_identity_mismatch'
    research = evidence(id='research', family='archived_jats', kind='research-article', **{'class': 'research'})
    assert decide([notice, research], [])['status'] == 'pending_conflicting_primary_types'


def test_content_review_resolves_only_named_subtypes_and_keeps_conflicts() -> None:
    source = evidence(kind='Research Article', title='A new GWAS meta-analysis method',
                      **{'class': 'subtype_review_required'})
    review = evidence(id='review', family='reviewed_study_design', kind='Reviewed primary study design',
                      online_date='', abstract_sha256='a' * 64,
                      reviewed_primary_evidence_ids=[source['id']], source_evidence_ids=[source['id']],
                      **{'class': 'research'})
    assert decide([source, review], [])['status'] == 'included_verified_research'
    rebuilt = dict(review, study_design_class='research', **{'class': 'unresolved_type'})
    assert decide([source, rebuilt], [])['status'] == 'included_verified_research'
    assert decide([source, dict(review, **{'class': 'nonprimary'})], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([source, dict(review, source_evidence_ids=['absent'])], [])['status'] == 'pending_special_subtype_review'
    assert decide([dict(source, scope_review_required=True), review], [])['status'] == 'pending_special_subtype_review'
    assert decide([source, review], [{'types': ['Review']}])['status'] == 'pending_primary_secondary_type_conflict'
    integrity = evidence(id='integrity', kind='Retracted Publication', **{'class': 'integrity_hold'})
    assert decide([source, review, integrity], [])['status'] == 'pending_integrity_review'
    reviewed_meta = dict(review, reviewed_secondary_meta_analysis_receipts=['secondary.json'])
    meta = {'types': ['Meta-Analysis'], 'receipt': 'secondary.json'}
    assert decide([source, reviewed_meta], [meta])['status'] == 'included_verified_research'
    assert decide([source, reviewed_meta], [dict(meta, receipt='unreviewed.json')])['status'] == 'pending_primary_secondary_type_conflict'
    for flag in ['Review', 'Retracted Publication', 'Expression of Concern']:
        assert decide([source, reviewed_meta], [dict(meta, types=[flag])])['status'] == 'pending_primary_secondary_type_conflict'


def test_protocol_needs_individual_verified_content_review() -> None:
    protocol = evidence(kind='Lab Protocol', categories=['Lab Protocol'], **{'class': 'unresolved_type'})
    assert decide([protocol], [])['status'] == 'pending_primary_type'
    review = evidence(id='body-review', family='reviewed_study_design', kind='Lab Protocol',
                      study_design_class='research', content_sha256='b' * 64,
                      reviewed_primary_evidence_ids=[protocol['id']], source_evidence_ids=[protocol['id']])
    assert decide([protocol, review], [])['status'] == 'included_verified_research'
    assert decide([protocol, dict(review, content_sha256='')], [])['status'] == 'pending_primary_type'
    assert decide([protocol, dict(review, reviewed_primary_evidence_ids=['different-paper'])], [])['status'] == 'pending_primary_type'


def test_research_content_review_can_resolve_only_named_meta_label() -> None:
    source = evidence(kind='Research Article', **{'class': 'research'})
    review = evidence(id='review', family='reviewed_study_design',
                      study_design_class='research', abstract_sha256='a' * 64,
                      reviewed_primary_evidence_ids=[source['id']], source_evidence_ids=[source['id']],
                      reviewed_secondary_meta_analysis_receipts=['checked.json'])
    meta = {'types': ['Meta-Analysis'], 'receipt': 'checked.json'}
    assert decide([source], [meta])['status'] == 'pending_primary_secondary_type_conflict'
    assert decide([source, review], [meta])['status'] == 'included_verified_research'
    assert decide([source, review], [dict(meta, receipt='other.json')])['status'] == 'pending_primary_secondary_type_conflict'
    for kind in ['Review', 'Systematic Review', 'Retracted Publication', 'Expression of Concern']:
        assert decide([source, review], [dict(meta, types=[kind])])['status'] == 'pending_primary_secondary_type_conflict'
    assert decide([source, dict(review, study_design_class='nonprimary')], [meta])['status'] == 'pending_primary_secondary_type_conflict'


def test_original_component_review_is_receipt_and_label_specific() -> None:
    source = evidence(kind='Research Article', **{'class': 'research'})
    checked = {'receipt': 'checked.json', 'types': ['Review'], 'raw_sha256': 'b' * 64,
               'original_research_component': 'Observed cohort data with reported methods and results'}
    review = evidence(id='content-review', family='reviewed_study_design',
                      study_design_class='research', abstract_sha256='a' * 64,
                      reviewed_primary_evidence_ids=[source['id']], source_evidence_ids=[source['id']],
                      reviewed_secondary_content_labels=[checked])
    flag = {'receipt': 'checked.json', 'types': ['Review']}
    assert decide([source, review], [flag])['status'] == 'included_verified_research'
    for other in [dict(flag, receipt='other.json'), dict(flag, types=['Systematic Review']),
                  dict(flag, types=['Retracted Publication']), dict(flag, types=['Expression of Concern'])]:
        assert decide([source, review], [other])['status'] == 'pending_primary_secondary_type_conflict'
    for missing in ['original_research_component', 'raw_sha256']:
        invalid = dict(checked, **{missing: ''})
        assert decide([source, dict(review, reviewed_secondary_content_labels=[invalid])], [flag])['status'] == 'pending_primary_secondary_type_conflict'
    assert decide([source, dict(review, source_evidence_ids=['absent'])], [flag])['status'] == 'pending_primary_secondary_type_conflict'


def test_named_generic_archive_type_can_be_adjudicated_from_native_nonresearch() -> None:
    native = evidence(kind='Viewpoints', issns=['1935-2735'])
    archive = evidence(id='archive', family='archived_jats', kind='research-article', **{'class': 'research'})
    review = evidence(id='body', family='reviewed_study_design', study_design_class='nonprimary',
                      content_sha256='a'*64, content_locator='Introduction and recommendations',
                      generic_archive_nonresearch_adjudication=True,
                      reviewed_primary_evidence_ids=['archive'], source_evidence_ids=[native['id']])
    assert decide([native, archive, review], [])['status'] == 'excluded_primary_nonresearch'
    for invalid in [dict(review, content_sha256=''), dict(review, generic_archive_nonresearch_adjudication=False),
                    dict(review, source_evidence_ids=['absent'])]:
        assert decide([native, archive, invalid], [])['status'] == 'pending_conflicting_primary_types'
    other = dict(archive, family='publisher_springer')
    assert decide([native, other, review], [])['status'] == 'pending_conflicting_primary_types'
    other = dict(archive, kind='Original Research')
    assert decide([native, other, review], [])['status'] == 'pending_conflicting_primary_types'
