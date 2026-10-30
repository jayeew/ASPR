from experiments.fig8_extension.fig8b_qualification_decide import decide
import pytest

from experiments.fig8_extension.mdpi_brief_content_apply import explicit_nested_abstract, native_article_xml
from experiments.fig8_extension.supplement_sources import jats_metadata


def test_oai_envelope_preserves_article_identity_and_formal_abstract() -> None:
    raw = b'<record xmlns="http://www.openarchives.org/OAI/2.0/"><metadata><article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/" xml:lang="en"><front><article-meta><article-id pub-id-type="doi">10.3390/test</article-id><abstract><sec><title>Simple Summary</title><p>Lay summary.</p></sec><sec><title>Abstract</title><p>Formal results.</p></sec></abstract></article-meta></front></article></metadata></record>'
    normalized = native_article_xml(raw)
    assert jats_metadata(normalized)['doi'] == ['10.3390/test']
    assert explicit_nested_abstract(normalized) == 'Formal results.'
    with pytest.raises(ValueError, match='Expected one'):
        native_article_xml(b'<record><metadata><article/><article/></metadata></record>')


def test_native_simple_summary_is_not_substituted_for_abstract() -> None:
    raw = b'<article><front><article-meta><abstract><sec><title>Simple Summary</title><p>Plain-language summary.</p></sec><sec><title>Abstract</title><p>Original abstract.</p></sec></abstract></article-meta></front></article>'
    assert explicit_nested_abstract(raw) == 'Original abstract.'


def test_mdpi_brief_requires_individual_content_review_and_preserves_conflicts() -> None:
    native = dict(id='native', doi='10.3390/test', identity='matched',
                  family='archived_jats', kind='brief-report', online_date='2024-01-01',
                  categories=[], **{'class': 'subtype_review_required'})
    review = dict(id='review', doi=native['doi'], identity='matched',
                  family='reviewed_study_design', kind='Individual completed study',
                  online_date='', study_design_class='research',
                  reviewed_primary_evidence_ids=['native'], source_evidence_ids=['native'],
                  review_protocol='native_mdpi_brief_report_design_1', raw_sha256='raw',
                  abstract_sha256='abstract', content_locator='article-meta/abstract',
                  review_rule_receipt='supplement/mdpi_brief_report_policy/content_reviews.json',
                  format_policy_receipt='supplement/mdpi_brief_report_policy/policy.json',
                  study_design_reason='Completed experiments with reported measurements')
    assert decide([native], [])['status'] == 'pending_special_subtype_review'
    assert decide([native, review], [])['status'] == 'included_verified_research'
    publisher = {**native, 'family': 'publisher_mdpi_oai'}
    assert decide([publisher, review], [])['status'] == 'included_verified_research'
    nonprimary = {**review, 'study_design_class': 'nonprimary',
                  'study_design_reason': 'Native abstract explicitly describes a literature overview'}
    assert decide([publisher, nonprimary], [])['status'] == 'excluded_primary_nonresearch'
    unrelated = {**native, 'family': 'publisher_other'}
    assert decide([unrelated, review], [])['status'] == 'pending_special_subtype_review'
    for field in ['raw_sha256', 'abstract_sha256', 'format_policy_receipt', 'study_design_reason']:
        invalid = {k: v for k, v in review.items() if k != field}
        assert decide([native, invalid], [])['status'] == 'pending_special_subtype_review'
    secondary = [{'types': ['Case Reports'], 'receipt': 'secondary'}]
    assert decide([native, review], secondary)['status'] == 'pending_primary_secondary_type_conflict'
    contrary = {**native, 'id': 'contrary', 'kind': 'review', 'class': 'nonprimary'}
    assert decide([native, review, contrary], [])['status'] == 'pending_conflicting_primary_types'
