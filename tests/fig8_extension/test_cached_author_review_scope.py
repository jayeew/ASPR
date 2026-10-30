"""Archived review exclusions must not replace native or mixed-study evidence."""
import hashlib

from experiments.fig8_extension.fig8b_qualification_decide import decide


def review() -> dict:
    abstract = ('This review summarizes published work on cellular metabolism and its regulation. '
                'The mechanisms reported in the existing literature are discussed alongside known '
                'limitations and potential future directions. The overview covers interactions between '
                'signaling pathways and describes therapeutic approaches reported by other researchers.')
    return {'id': 'review', 'family': 'cached_author_review_scope', 'identity': 'matched',
            'review_rule': 'med_author_review_no_primary_1', 'source': 'MED', 'pmid': '123',
            'raw_sha256': 'digest', 'content_locator': 'resultList.result', 'issns': ['1234-5678'],
            'doi': '10.1/example', 'indexed_types': ['Review', 'Journal Article'],
            'review_abstract': abstract, 'abstract_sha256': hashlib.sha256(abstract.encode()).hexdigest(),
            'secondary_receipt': 'raw.json.gz', 'title': 'Cellular metabolism'}


SECONDARY = [{'types': ['Review', 'Journal Article'], 'receipt': 'raw.json.gz'}]


def test_review_has_honest_separate_status_and_no_inferred_date() -> None:
    result = decide([review()], SECONDARY)
    assert result['status'] == 'excluded_archived_author_review'
    assert result['online_year'] == 0
    assert result['evidence_ids'] == ['review']
    assert decide([review()], SECONDARY, 'retained')['status'].startswith('excluded_duplicate')


def test_missing_mismatched_or_integrity_evidence_cannot_exclude() -> None:
    for update in [{'identity': 'mismatch'}, {'issns': []}, {'abstract_sha256': 'wrong'},
                   {'secondary_receipt': 'other'}, {'scope_review_required': True},
                   {'indexed_types': ['Journal Article']}, {'source': 'PPR'},
                   {'title': 'A systematic review'}, {'review_abstract': 'This review summarizes.'}]:
        assert decide([{**review(), **update}], SECONDARY)['status'].startswith('pending')
    for types in [['Review', 'Retracted Publication'], ['Review', 'Clinical Trial']]:
        assert decide([review()], [{'types': types, 'receipt': 'raw.json.gz'}])['status'].startswith('pending')


def test_mixed_original_results_and_native_evidence_take_precedence() -> None:
    item = review()
    item['review_abstract'] += ' We report new experimental results.'
    item['abstract_sha256'] = hashlib.sha256(item['review_abstract'].encode()).hexdigest()
    assert decide([item], SECONDARY)['status'].startswith('pending')
    native = {'id': 'native', 'identity': 'matched', 'family': 'publisher_example',
              'class': 'research', 'kind': 'research-article', 'online_date': '2024-02-01'}
    assert decide([review(), native], SECONDARY) == decide([native], SECONDARY)
    native['identity'] = 'mismatch'
    assert decide([review(), native], SECONDARY)['status'] == 'pending_primary_identity_mismatch'


def test_review_label_does_not_hide_original_computational_analyses() -> None:
    for addition in [
        ' We used computational methods to determine binding energies.',
        ' We found electric field strengths matching vestibular responses.',
        ' Our analysis revealed a new pathway.',
        ' We provide a first of its kind dosimetric analysis.',
        ' We identified molecular components through experimental assays.',
    ]:
        item = review()
        item['review_abstract'] += addition
        item['abstract_sha256'] = hashlib.sha256(item['review_abstract'].encode()).hexdigest()
        assert decide([item], SECONDARY)['status'].startswith('pending')
