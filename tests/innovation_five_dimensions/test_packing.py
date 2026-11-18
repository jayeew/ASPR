from __future__ import annotations

from figure_pipeline.innovation_five_dimensions.evaluation import evidence_audit, source_index
from figure_pipeline.innovation_five_dimensions.io import encoded
from figure_pipeline.innovation_five_dimensions.packing import pack_sources, packet_sources, source_blocks


def test_packet_preserves_scope_metadata_and_original_offsets() -> None:
    text = 'start ' * 700 + 'A unique and verifiable observation.' + ' end' * 700
    source = {'source_id': 'H1', 'source_type': 'fulltext', 'text': text,
              'historical_use': 'after_cutoff_not_antecedent', 'source_notice': 'Keep this notice.',
              'publication_date': '2026-12-01'}
    packet = pack_sources([source], {'quote': 'A unique and verifiable observation.'}, 8000)
    for row in packet['source_passages']:
        assert row['text'] == text[row['start']:row['end']]
        assert row['historical_use'] == source['historical_use']
        assert row['source_notice'] == source['source_notice']
    audit = evidence_audit({'source_id': 'H1', 'quote': 'A unique and verifiable observation.'},
                           source_index(packet_sources(packet)))
    assert audit['located']
    assert audit['span']['source_start'] == text.index('A unique and verifiable observation.')


def test_manuscript_reservation_and_overall_budget() -> None:
    sources = [{'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': 'Manuscript original. ' * 5000}]
    sources.extend({'source_id': f'H{i}', 'source_type': 'fulltext', 'doi': f'10.1/{i}',
                    'text': 'Historically relevant material. ' * 200} for i in range(30))
    sources.extend({'source_id': f'GRAPH:{i}', 'source_type': 'graph_observation',
                    'source_notice': 'Generated node labels are not scientific evidence.',
                    'historical_use': 'not_an_independent_historical_original',
                    'text': '{"metrics":{"degree":3},"claim_text":"metadata only"}' * 120} for i in range(6))
    packet = pack_sources(sources, 'Historically relevant', 72000)
    assert len(encoded(packet['source_passages'])) <= 72000
    assert packet['material_coverage']['selected_characters'] == len(encoded(packet['source_passages']))
    assert packet['material_coverage']['selected_characters_by_group']['manuscript'] >= 21600
    assert all(packet['material_coverage']['selected_characters_by_group'][group] > 0
               for group in ('historical', 'graph'))
    for row in packet['source_passages']:
        if row['source_type'] == 'graph_observation':
            assert row['source_notice'] == 'Generated node labels are not scientific evidence.'


def test_unused_other_material_allowance_returns_to_manuscript() -> None:
    source = {'source_id': 'MANUSCRIPT', 'source_type': 'manuscript', 'text': 'Original manuscript. ' * 5000}
    packet = pack_sources([source], 'manuscript', 12000)
    assert packet['material_coverage']['selected_characters_by_group']['manuscript'] > 9000
    assert len(encoded(packet['source_passages'])) <= 12000


def test_graph_blocks_remain_literal_with_source_notice() -> None:
    text = '{"claim_text":"generated metadata",' + '"nodes":[' + ','.join(str(i) for i in range(3000)) + ']}'
    source = {'source_id': 'GRAPH:C1', 'source_type': 'graph_observation', 'text': text,
              'source_notice': 'Node text does not establish novelty.', 'historical_use': 'not_independent'}
    for block in source_blocks([source]):
        assert block['text'] == text[block['start']:block['end']]
        assert block['source_notice'] == source['source_notice']
        assert block['historical_use'] == source['historical_use']
