from copy import deepcopy

import pytest

from experiments.fig8_extension.fig8b_qualification_decide import reviewed_pnas_letter_content


def evidence() -> list[dict]:
    return [dict(id='native', family='archived_jats', identity='matched', kind='letter',
                 doi='10.1073/pnas.example', **{'class': 'subtype_review_required'}),
            dict(id='review', family='reviewed_pnas_letter_content', identity='matched',
                 doi='10.1073/pnas.example', issns=['0027-8424'], study_design_class='nonprimary',
                 review_protocol='pnas_letter_full_body_individual_1', reviewed_primary_evidence_ids=['native'],
                 content_sha256='body', raw_sha256='raw', content_locator='body//p',
                 review_rule_receipt='review.json', content_review_reason='Individually read commentary')]


def test_only_individually_bound_letter_is_excluded() -> None:
    original = evidence()
    result = reviewed_pnas_letter_content(original)
    assert all(e['class'] == 'nonprimary' for e in result)
    assert original[0]['class'] == 'subtype_review_required'


@pytest.mark.parametrize('index,field,value', [
    (1, 'study_design_class', 'research'), (1, 'content_sha256', ''),
    (1, 'reviewed_primary_evidence_ids', ['missing']), (1, 'issns', ['0000-0000']),
    (1, 'doi', '10.9999/other'), (1, 'identity', 'title_or_journal_mismatch'),
    (0, 'doi', '10.1073/pnas.other'), (0, 'kind', 'research-article'),
    (0, 'class', 'integrity_hold'), (0, 'scope_review_required', True),
    (0, 'identity', 'title_or_journal_mismatch'),
])
def test_invalid_reviews_preserve_original(index: int, field: str, value: object) -> None:
    items = deepcopy(evidence())
    items[index][field] = value
    result = reviewed_pnas_letter_content(items)
    assert result == items[:1]
