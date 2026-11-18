from __future__ import annotations

from figure_pipeline.innovation_five_dimensions.runner import blind, replace_ids


def payload() -> dict:
    return {
        'source_passages': [{'source_id': 'GEAR:S1::SEG002', 'original_source_id': 'GEAR:S1',
                             'source_type': 'abstract', 'text': 'A occurs under B.'}],
        'report': {'status': 'supported', 'judgment': 'A occurs [GEAR:S1::SEG002].',
                   'evidence': [{'source_id': 'UNKNOWN_GRAPH:42', 'quote': 'A occurs.'}]},
        'citation_audit': {'citations': [{'source_id': 'UNKNOWN_GRAPH:42',
                                         'problem': 'unknown_source_id'}]},
    }


def test_blind_includes_unknown_citations_and_original_source_aliases() -> None:
    original = payload()
    anonymous, reverse = blind(original)
    assert set(reverse.values()) == {'GEAR:S1::SEG002', 'GEAR:S1', 'UNKNOWN_GRAPH:42'}
    assert anonymous['source_passages'][0]['source_id'].startswith('ANON_EVIDENCE_')
    assert anonymous['source_passages'][0]['original_source_id'].startswith('ANON_EVIDENCE_')
    unknown = anonymous['report']['evidence'][0]['source_id']
    assert unknown == anonymous['citation_audit']['citations'][0]['source_id']
    assert unknown not in {s['source_id'] for s in anonymous['source_passages']}
    assert replace_ids(anonymous, reverse) == original


def test_replacement_does_not_mutate_identifier_prefixes_or_substrings() -> None:
    text = 'S1 S10 XS1 S1::SEG002 [S1] S1@20:30 S1/extra S1-suffix'
    assert replace_ids(text, {'S1': 'E1'}) == 'E1 S10 XS1 S1::SEG002 [E1] E1@20:30 S1/extra S1-suffix'
    assert replace_ids('S1::SEG002 S1', {'S1': 'E1', 'S1::SEG002': 'E2'}) == 'E2 E1'


def test_alias_namespace_cannot_collide_with_original_scientific_text() -> None:
    original = payload()
    original['report']['rationale'] = 'The observed marker ANON_EVIDENCE_0_00001 is unrelated to citations.'
    anonymous, reverse = blind(original)
    assert all(key.startswith('ANON_EVIDENCE_1_') for key in reverse)
    assert 'ANON_EVIDENCE_0_00001' in anonymous['report']['rationale']
    assert replace_ids(anonymous, reverse) == original


def test_source_id_matching_an_enum_does_not_change_scientific_status() -> None:
    original = payload()
    original['report']['evidence'][0]['source_id'] = 'supported'
    original['report']['judgment'] = 'The finding is supported by the manuscript [supported].'
    anonymous, reverse = blind(original)
    assert anonymous['report']['status'] == 'supported'
    assert anonymous['report']['evidence'][0]['source_id'] != 'supported'
    assert 'The finding is supported by the manuscript' in anonymous['report']['judgment']
    assert '[supported]' not in anonymous['report']['judgment']
    assert replace_ids(anonymous, reverse) == original


def test_blinding_is_deterministic_and_empty_payload_stays_empty() -> None:
    assert blind(payload()) == blind(payload())
    assert blind({'source_passages': []}) == ({'source_passages': []}, {})


def test_generation_scope_uses_actual_report_packet_not_common_reference() -> None:
    from figure_pipeline.innovation_five_dimensions.runner import generation_material_scope
    generation = {'prior_analysis': 'Earlier report.', 'source_passages': [
        {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': 'Visible text.',
         'title': 'Target paper', 'start': 100, 'end': 113},
        {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': 'Other passage.',
         'title': 'Target paper', 'start': 1000, 'end': 1014}]}
    scope = generation_material_scope(generation)
    assert scope['source_counts_by_type'] == {'manuscript': 1}
    assert scope['sources'][0]['visible_text_ranges'] == [{'start': 100, 'end': 113}, {'start': 1000, 'end': 1014}]
    assert scope['earlier_analysis_provided'] is True
    assert 'text' not in scope['sources'][0] and 'title' in scope['sources'][0]['visible_metadata_fields']
    original = {'source_passages': [
        {'source_id': 'HISTORY_1', 'source_type': 'fulltext', 'text': 'Judge-only prior text.'}],
        'generation_material_scope': scope}
    anonymous, reverse = blind(original)
    assert anonymous['generation_material_scope']['source_counts_by_type'] == {'manuscript': 1}
    generation_id = anonymous['generation_material_scope']['sources'][0]['source_id']
    assert generation_id != 'MANUSCRIPT'
    assert generation_id != anonymous['source_passages'][0]['source_id']
    assert replace_ids(anonymous, reverse) == original


def test_generation_scope_keeps_disjoint_ranges_and_distinguishes_fulltext_from_metadata() -> None:
    from figure_pipeline.innovation_five_dimensions.runner import generation_material_scope
    packet = {'source_passages': [
        {'source_id': 'GEAR:1', 'source_type': 'fulltext', 'text': 'A', 'start': 1000, 'end': 1001},
        {'source_id': 'GEAR:1', 'source_type': 'fulltext', 'text': 'B', 'start': 10, 'end': 11},
        {'source_id': 'GEAR:1', 'source_type': 'fulltext', 'text': 'B', 'start': 10, 'end': 11},
        {'source_id': 'GEAR:2', 'source_type': 'metadata_only', 'title': 'A cited paper',
         'text': 'Bibliographic metadata', 'start': 0, 'end': 22}]}
    scope = generation_material_scope(packet)
    assert scope['source_counts_by_type'] == {'fulltext': 1, 'metadata_only': 1}
    assert scope['sources'][0]['visible_text_ranges'] == [{'start': 10, 'end': 11}, {'start': 1000, 'end': 1001}]
    assert scope['earlier_analysis_provided'] is False
    assert set(scope['sources'][0]) == {'source_id', 'source_type', 'visible_text_ranges', 'visible_metadata_fields'}


def test_material_scope_context_changes_only_evaluation_instruction() -> None:
    from figure_pipeline.innovation_five_dimensions import prompts
    from figure_pipeline.innovation_five_dimensions.io import fingerprint
    assert fingerprint(prompts.REPORT) == '039354ded513dd5995b76aa335ca6476bae2a67ab5a02f10ef8fc53aa1a5e5ae'
    assert fingerprint(prompts.REFERENCE) == 'd47a012cb805576c253de1475148cbd19fdd038bb0c850976c7702ca2fc6815d'
    assert fingerprint(prompts.EVALUATE) != '1493eba7cb070cbcd0860be8bccb4cd2f507b2024ffa49f0041a4dd2b7ea0647'
    assert 'generation_material_scope' in prompts.EVALUATE
    assert 'an omitted important prior-work comparison can still lower completeness' in prompts.EVALUATE
    assert 'Accurate statements of\nmissing generation material are not contradicted' in prompts.EVALUATE
