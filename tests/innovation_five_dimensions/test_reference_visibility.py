from __future__ import annotations

import copy

from figure_pipeline.innovation_five_dimensions.evaluation import evidence_audit, source_index
from figure_pipeline.innovation_five_dimensions.reference_visibility import reference_visible_sources


def test_visible_core_quote_is_not_an_unseen_fulltext_retrieval() -> None:
    text = 'Visible opening. Already supplied core finding. Unseen conclusion.'
    packet = {'source_passages': [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                                  'text': 'Visible opening.', 'start': 0, 'end': 16}],
              'core_contributions': [{'manuscript_quote': 'Already supplied core finding.'}]}
    original = copy.deepcopy(packet)
    result = reference_visible_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': text}])
    assert packet == original
    index = source_index(result['sources'])
    assert evidence_audit({'source_id': 'MANUSCRIPT', 'quote': 'Already supplied core finding.'}, index)['located']
    assert not evidence_audit({'source_id': 'MANUSCRIPT', 'quote': 'Unseen conclusion.'}, index)['located']
    assert result['audit']['request_changed'] is False and result['audit']['scientific_judgments_changed'] is False


def test_adjacent_visible_core_lines_join_only_original_whitespace() -> None:
    text = 'Both\nmRNA transfection and\nLNP delivery induced a response. Hidden result.'
    packet = {'source_passages': [], 'core_contributions': [{'manuscript_quotes': [
        'Both', 'mRNA transfection and', 'LNP delivery induced a response.']} ]}
    result = reference_visible_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': text}])
    assert len(result['sources']) == 1
    assert result['sources'][0]['text'] == 'Both\nmRNA transfection and\nLNP delivery induced a response.'
    assert len(result['audit']['additional_fragments'][0]['layout_only_gaps']) == 2
    assert all(not gap['text'].strip() for gap in result['audit']['additional_fragments'][0]['layout_only_gaps'])


def test_unseen_words_between_core_quotes_never_become_visible() -> None:
    text = 'First finding. UNSEEN QUALIFIER. Second finding.'
    packet = {'source_passages': [], 'core_contributions': [{'manuscript_quotes': ['First finding.', 'Second finding.']}]}
    result = reference_visible_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': text}])
    assert len(result['sources']) == 2
    assert all('UNSEEN' not in source['text'] for source in result['sources'])
    assert not evidence_audit({'source_id': 'MANUSCRIPT', 'quote': text}, source_index(result['sources']))['located']


def test_core_paraphrase_added_words_and_ambiguous_positions_are_not_admitted() -> None:
    text = 'A binds B. It phosphorylates ACLY. A binds B.'
    packet = {'source_passages': [], 'core_contributions': [{'manuscript_quotes': [
        'A binds B.', 'It phosphorylates ACLY at S455.', 'Other unseen finding.']}]}
    result = reference_visible_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': text}])
    assert result['sources'] == []
    assert [r['state'] for r in result['audit']['core_locations']] == [
        'ambiguous_original_position', 'not_located_in_original_manuscript', 'not_located_in_original_manuscript']


def test_core_description_cannot_masquerade_as_original_quote_or_history_source() -> None:
    text = 'An actual target-paper observation.'
    packet = {'source_passages': [], 'core_contributions': [{'description': text, 'manuscript_quote': text}]}
    result = reference_visible_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': text}])
    assert len(result['sources']) == 1
    assert not evidence_audit({'source_id': 'HISTORY', 'quote': text}, source_index(result['sources']))['located']
    assert result['audit']['core_locations'][0]['input_field'].endswith('.manuscript_quote')


def test_existing_packet_is_preserved_and_unverified_offsets_do_not_bridge_gaps() -> None:
    text = 'Visible first. Hidden middle. Visible last.'
    source = {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
              'text': 'Visible first.', 'start': 0, 'end': len(text)}  # Deliberately inconsistent.
    packet = {'source_passages': [source], 'core_contributions': [{'manuscript_quote': 'Visible last.'}]}
    result = reference_visible_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': text}])
    assert result['sources'][0] == source
    assert all('Hidden middle' not in row['text'] for row in result['sources'])
    assert result['audit']['core_locations'][0]['state'] == 'unverified_original_offsets'


def test_sufficient_original_packet_is_returned_without_additional_core_material() -> None:
    from figure_pipeline.innovation_five_dimensions.reference_visibility import required_reference_sources
    source = {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': 'Existing finding.', 'start': 0, 'end': 17}
    packet = {'source_passages': [source], 'core_contributions': [{'manuscript_quote': 'Other visible finding.'}]}
    checklist = {'dimensions': [{'evidence': [{'source_id': 'MANUSCRIPT', 'quote': 'Existing finding.'}]}]}
    result = required_reference_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                                                 'text': 'Existing finding. Other visible finding.'}], checklist)
    assert result['sources'] == packet['source_passages']
    assert result['audit']['state'] == 'original_packet_sufficient'


def test_only_missing_citation_spans_are_added_not_all_visible_core_text() -> None:
    from figure_pipeline.innovation_five_dimensions.reference_visibility import required_reference_sources
    text = 'Only this sentence is cited. This adjacent sentence is not needed.'
    packet = {'source_passages': [], 'core_contributions': [{'manuscript_quote': text}]}
    checklist = {'dimensions': [{'evidence': [{'source_id': 'MANUSCRIPT', 'quote': 'Only this sentence is cited.'},
                                            {'source_id': 'HISTORY', 'quote': 'Invented historical sentence.'}]}]}
    result = required_reference_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': text}], checklist)
    assert len(result['sources']) == 1
    assert result['sources'][0]['text'] == 'Only this sentence is cited.'
    assert len(result['audit']['remaining_failures']) == 1
    assert result['audit']['remaining_failures'][0]['source_id'] == 'HISTORY'


def test_only_pure_figure_citations_may_be_mechanically_restored() -> None:
    from figure_pipeline.innovation_five_dimensions.reference_visibility import figure_citation_location
    original = 'A induces cell death, whereas B has no effect (Fig. 4f–h\nand Supplementary Fig. 9h).'
    quote = 'A induces cell death, whereas B has no effect.'
    span = figure_citation_location(quote, original)
    assert span is not None and span['text'] == original
    assert len(span['figure_citations_restored']) == 1
    for forbidden in ['A induces cell death, whereas B has no effect (at 4 h).',
                      'A induces cell death, whereas B has no effect (Fig. 4, n=12).',
                      'A induces cell death, whereas B has an effect (Fig. 4).']:
        assert figure_citation_location(quote, forbidden) is None
    assert figure_citation_location('AKT1 phosphorylates ACLY at S455.',
                                     'AKT1 phosphorylates ACLY (Fig. 4).') is None


def test_figure_repair_copies_real_span_without_changing_scientific_content_fields() -> None:
    from figure_pipeline.innovation_five_dimensions.reference_visibility import repair_visible_reference_locations
    original = 'A induces cell death, whereas B has no effect (Fig. 4f–h and Supplementary Fig. 9h).'
    packet = {'source_passages': [], 'core_contributions': [{'manuscript_quote': original}]}
    checklist = {'dimensions': [{'dimension': 'evidential_basis', 'applicability': 'applicable',
                                'criteria': ['Compare A with B.'], 'scope': 'Cells only.', 'reason': 'Evidence question.',
                                'evidence': [{'source_id': 'MANUSCRIPT', 'quote': 'A induces cell death, whereas B has no effect.',
                                              'explanation': 'Control contrast.'}]}]}
    before = copy.deepcopy(checklist)
    result = repair_visible_reference_locations(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                                                          'text': original}], checklist)
    assert checklist == before
    assert result['value']['dimensions'][0]['evidence'][0]['quote'] == original
    result['value']['dimensions'][0]['evidence'][0]['quote'] = before['dimensions'][0]['evidence'][0]['quote']
    assert result['value'] == before
    assert result['sources'][0]['text'] == original
    assert result['audit']['scientific_judgments_changed'] is False


def test_overlapping_visible_packet_spans_allow_exact_cross_boundary_quote_without_core() -> None:
    from figure_pipeline.innovation_five_dimensions.reference_visibility import required_reference_sources
    text = 'First complete sentence. Second sentence continues the argument. Unseen final claim.'
    packet = {'source_passages': [
        {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'start': 0, 'end': 42, 'text': text[:42]},
        {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'start': 25, 'end': 63, 'text': text[25:63]}],
        'core_contributions': []}
    quote = text[:63]
    checklist = {'dimensions': [{'evidence': [{'source_id': 'MANUSCRIPT', 'quote': quote}]}]}
    result = required_reference_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                                                 'text': text}], checklist)
    assert result['sources'][:2] == packet['source_passages']
    assert result['sources'][2]['text'] == quote
    assert result['sources'][2]['start'] == 0 and result['sources'][2]['end'] == 63
    assert result['audit']['remaining_failures'] == []
    assert all('Unseen final claim' not in s['text'] for s in result['sources'])


def test_noncontinuous_packet_spans_cannot_be_joined_even_without_core() -> None:
    from figure_pipeline.innovation_five_dimensions.reference_visibility import required_reference_sources
    text = 'A causes B. UNSEEN CONDITION. C increases D.'
    last = text.index('C increases')
    packet = {'source_passages': [
        {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'start': 0, 'end': 11, 'text': text[:11]},
        {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'start': last, 'end': len(text), 'text': text[last:]}],
        'core_contributions': []}
    checklist = {'dimensions': [{'evidence': [{'source_id': 'MANUSCRIPT', 'quote': 'A causes B. C increases D.'}]}]}
    result = required_reference_sources(packet, [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript',
                                                 'text': text}], checklist)
    assert result['sources'] == packet['source_passages']
    assert len(result['audit']['remaining_failures']) == 1
