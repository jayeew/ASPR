from copy import deepcopy

import pytest

from experiments.fig8_extension.fig8b_qualification_decide import decide, same_archive_missing_metadata


def records() -> tuple[dict, dict]:
    old = dict(id='old', identity='matched', family='archived_jats',
               doi='10.1016/j.gendis.example', title='Experimental regulation of tumor growth',
               journal_titles=['Genes & Diseases'], kind='brief-report',
               categories=[], issns=[], online_date='2024-02-03',
               date_precision='day', **{'class': 'subtype_review_required'},
               url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1234567/fullTextXML')
    precise = {**old, 'id': 'precise', 'issns': ['2352-3042'],
               'categories': ['Rapid Communication'],
               'url': 'https://pmc-oa-opendata.s3.amazonaws.com/PMC1234567.1/PMC1234567.1.xml'}
    return old, precise


def test_same_pmc_incomplete_record_no_longer_blocks_explicit_research() -> None:
    old, precise = records()
    original = deepcopy(old)
    assert same_archive_missing_metadata(old, precise)
    assert decide([old, precise], [])['status'] == 'included_verified_research'
    assert old == original


@pytest.mark.parametrize('change', [
    {'doi': '10.1016/j.gendis.other'}, {'title': 'Different study'},
    {'journal_titles': ['Other journal']}, {'kind': 'letter'},
    {'url': 'https://example.org/PMC7654321'}, {'url': ''},
    {'issns': ['9999-9999']}, {'categories': ['Correspondence']},
    {'scope_review_required': True}, {'title_scope_flag': True},
])
def test_incomplete_record_cannot_borrow_from_different_identity_or_hold(change: dict) -> None:
    old, precise = records()
    old.update(change)
    assert not same_archive_missing_metadata(old, precise)


@pytest.mark.parametrize('change', [
    {'categories': []}, {'categories': ['Correspondence']}, {'issns': []},
    {'identity': 'mismatch'}, {'title': 'A systematic review'},
    {'scope_review_required': True},
])
def test_precise_source_must_already_qualify_under_existing_rule(change: dict) -> None:
    old, precise = records()
    precise.update(change)
    assert not same_archive_missing_metadata(old, precise)


def test_matching_review_title_is_not_promoted() -> None:
    old, precise = records()
    old['title'] = precise['title'] = 'A meta-analysis of treatment outcomes'
    assert not same_archive_missing_metadata(old, precise)
    assert decide([old, precise], [])['status'] != 'included_verified_research'


def test_integrity_hold_survives_metadata_refinement() -> None:
    old, precise = records()
    precise['class'] = 'integrity_hold'
    assert decide([old, precise], [])['status'] == 'pending_integrity_review'
