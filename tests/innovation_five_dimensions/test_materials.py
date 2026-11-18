from __future__ import annotations

import json
from pathlib import Path

from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS
from figure_pipeline.innovation_five_dimensions.io import read, write
from figure_pipeline.innovation_five_dimensions.materials import (
    SourceCollector,
    clean_structure,
    pool_text,
    prepare_generation,
    prepare_reference,
)
from figure_pipeline.innovation_five_dimensions.reporting import write_report


def paper() -> dict:
    return {'paper_id': 'P1', 'title': 'Example', 'doi': '10.1/target',
            'cutoff': '2026-01-01', 'cohort': 'main'}


def test_direct_generation_does_not_read_review_shared_claims_or_other_methods(tmp_path: Path) -> None:
    study, output = tmp_path / 'study', tmp_path / 'output'
    write(study / 'reports/direct_a/papers/P1.json', {'body': 'OWN_ANALYSIS', 'sources': [],
                                                    'analysis': {'text': 'DUPLICATE_NOT_NEEDED'}})
    write(study / 'reports/full/papers/P1.json', {'body': 'OTHER_METHOD_SECRET'})
    prepared = {'manuscript': 'ORIGINAL_MANUSCRIPT', 'manuscript_path': '/private/manuscript.md',
                'reviews': 'REVIEWER_SECRET', 'claims': ['SHARED_CLAIM_SECRET'],
                'reports': {'full': 'OTHER_METHOD_SECRET'}}
    prepare_generation(output, paper(), prepared, 'direct_a', study)
    payload = read(output / 'inputs/generation/direct_a/P1.json')
    text = json.dumps(payload)
    assert payload['analysis'] == 'OWN_ANALYSIS'
    assert len(payload['sources']) == 1
    assert payload['sources'][0]['text'] == 'ORIGINAL_MANUSCRIPT'
    assert all(term not in text for term in ('REVIEWER_SECRET', 'SHARED_CLAIM_SECRET',
                                           'OTHER_METHOD_SECRET', '/private/', 'DUPLICATE_NOT_NEEDED'))


def test_reference_filters_review_judgment_and_uses_native_graph(tmp_path: Path) -> None:
    study, output = tmp_path / 'study', tmp_path / 'output'
    write(study / 'evidence/evidence_pool/papers/P1.json', {
        'blocks': [{'block_id': 'b1', 'text': 'Prior work original.'},
                   {'block_id': 'b2', 'text': 'REVIEW_JUDGMENT'},
                   {'block_id': 'b3', 'text': 'GENERATED_GRAPH_INTERPRETATION'}],
        'sources': [
            {'source_id': 'H1', 'source_type': 'abstract', 'block_ids': [{'block_id': 'b1', 'start': 0, 'end': 20}]},
            {'source_id': 'R1', 'source_type': 'review', 'block_ids': [{'block_id': 'b2', 'start': 0, 'end': 15}]},
            {'source_id': 'G1', 'source_type': 'graph_observation', 'block_ids': [{'block_id': 'b3', 'start': 0, 'end': 30}]},
        ]})
    write(study / 'evidence/native_graph/P1.json', {'cards': [], 'joint': {'node_count': 3}})
    write(study / 'annotations/core/papers/P1.json', {'items': [
        {'core_id': 'C1', 'description': 'Core', 'manuscript_quote': 'Original core.',
         'selection_reason': 'GENERATED_SELECTION_JUDGMENT'}]})
    prepare_reference(output, paper(), {'manuscript': 'Manuscript', 'manuscript_path': '/private/paper'}, study)
    payload = read(output / 'inputs/reference/P1.json')
    text = json.dumps(payload)
    assert 'Prior work original.' in text
    assert any(s['source_type'] == 'graph_observation' for s in payload['sources'])
    assert all(term not in text for term in ('REVIEW_JUDGMENT', 'GENERATED_GRAPH_INTERPRETATION',
                                            'GENERATED_SELECTION_JUDGMENT', '/private/'))
    assert 'analysis' not in payload


def test_source_conflicts_preserve_both_original_passages() -> None:
    collector = SourceCollector(paper())
    for text in ('first', 'second', 'first'):
        collector.add({'source_id': 'H1', 'source_type': 'abstract'}, '/private/origin', text)
    assert [s['source_id'] for s in collector.sources] == ['H1', 'H1::SEG002']
    assert [s['text'] for s in collector.sources] == ['first', 'second']
    assert len(collector.provenance) == 3


def test_self_and_future_sources_are_not_marked_historical_antecedents() -> None:
    collector = SourceCollector(paper())
    collector.add({'source_id': 'SELF', 'source_type': 'fulltext', 'doi': '10.1/target'}, '', 'self')
    collector.add({'source_id': 'FUTURE', 'source_type': 'abstract', 'publication_date': '2026-05-01'}, '', 'future')
    assert collector.sources[0]['historical_use'] == 'target_paper_not_antecedent'
    assert collector.sources[1]['historical_use'] == 'after_cutoff_not_antecedent'


def test_paths_removed_without_removing_citation_path_counts() -> None:
    assert clean_structure({'original_path': '/private', 'two_hop_path_count': 2}) == {'two_hop_path_count': 2}
    assert pool_text({'block_ids': [{'block_id': 'b2', 'start': 3, 'end': 6},
                                     {'block_id': 'b1', 'start': 0, 'end': 3}]},
                     {'b1': 'abc', 'b2': 'def'}) == 'abcdef'


def test_original_attachment_is_preserved_verbatim(tmp_path: Path) -> None:
    prior = '  original paragraph\r\n\r\nsecond line without trailing newline'
    source = tmp_path / 'inputs/original_reports/direct_a/P1.md'
    source.parent.mkdir(parents=True)
    source.write_text(prior, newline='')
    report = {'contribution_overview': 'Overview', 'synthesis': 'Synthesis', 'dimensions': [
        {'dimension': key, 'judgment': 'Judgment', 'rationale': 'Reason', 'evidence': [],
         'scope': 'Scope', 'status': 'unresolved'} for key in DIMENSIONS]}
    write_report(tmp_path, 'P1', 'direct_a', report)
    path = tmp_path / 'reports/direct_a/P1.md'
    with path.open(newline='') as handle:
        assert handle.read().endswith(prior)
    assert path.read_text().index('贡献概述') < path.read_text().index('详细分析材料')
    assert read(path.with_suffix('.json')) == report
