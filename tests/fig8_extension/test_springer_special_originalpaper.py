from experiments.fig8_extension.fig8b_qualification_decide import decide


def test_special_format_not_overridden_by_generic_originalpaper() -> None:
    item = dict(id='native', identity='matched', family='publisher_springer',
                kind='OriginalPaper', **{'class': 'research'}, title='Study results',
                online_date='2024-01-01', date_precision='day',
                categories=['Article', 'OriginalPaper'], journal_titles=['Example'], issns=[])
    assert decide([item], [])['status'] == 'included_verified_research'
    for category in ['Guideline', 'Images', 'Award', 'Highlight', 'Neuro-Images',
                     'Case image in cardiovascular ultrasound', 'Addendum']:
        value = {**item, 'categories': [category, 'OriginalPaper']}
        assert decide([value], [])['status'] == 'pending_special_subtype_review'
    value = {**item, 'title': 'Addendum to: Study results'}
    assert decide([value], [])['status'] == 'pending_special_subtype_review'
    assert decide([{**value, 'class': 'integrity_hold'}], [])['status'] == 'pending_integrity_review'
    assert decide([{**value, 'class': 'nonprimary'}], [])['status'] == 'excluded_primary_nonresearch'
    assert decide([{**value, 'online_date': '2022-01-01'}], [])['status'] == 'excluded_primary_online_year_outside_window'
    assert decide([{**item, 'title': 'Effects of an addendum to patient information'}], [])['status'] == 'included_verified_research'


def test_bound_content_review_resolves_only_reviewed_native_sources() -> None:
    native = dict(id='native', identity='matched', family='publisher_springer',
                  doi='10.1007/example', kind='OriginalPaper', **{'class': 'research'},
                  title='Review of prior studies', online_date='2024-01-01',
                  categories=['Award', 'OriginalPaper'], journal_titles=['Example'], issns=[])
    review = dict(native, id='review', family='reviewed_study_design',
                  study_design_class='nonprimary', review_protocol='springer_special_native_content_1',
                  reviewed_primary_evidence_ids=['native'], source_evidence_ids=['native'],
                  review_rule_receipt='supplement/springer_special_originalpaper/content_reviews.json',
                  raw_sha256='raw', content_sha256='body', content_locator='Introduction',
                  review_quotes=['This review summarizes'], study_design_reason='Previously published research synthesis')
    assert decide([native, review], [])['status'] == 'excluded_primary_nonresearch'
    for change in [{'doi': 'other'}, {'review_quotes': []}, {'source_evidence_ids': ['missing']},
                   {'study_design_class': 'research'}]:
        assert decide([native, {**review, **change}], [])['status'] == 'pending_special_subtype_review'
