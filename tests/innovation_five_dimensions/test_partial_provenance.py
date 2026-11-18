from __future__ import annotations

from pathlib import Path

import pytest

from gear.codex_cli import _strict_response_schema
from figure_pipeline.innovation_five_dimensions import prompts
from figure_pipeline.innovation_five_dimensions.aggregate import load_result
from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS, MODEL, PROTOCOL
from figure_pipeline.innovation_five_dimensions.io import fingerprint, read, write
from figure_pipeline.innovation_five_dimensions.models import Checklist
from figure_pipeline.innovation_five_dimensions.partial_reference import recover_reference
from figure_pipeline.innovation_five_dimensions.runner import blind, build_evaluation_input, replace_ids
from figure_pipeline.innovation_five_dimensions.selector_evaluation import (
    EVALUATION_PROTOCOL, prepare_selector_payload, resolve_selector_evaluation,
)
from test_aggregate_provenance import fixture


def partial_fixture(root: Path, all_failed: bool = False) -> dict:
    fixture(root)
    checklist = read(root / 'references/P.json')
    for row in checklist['dimensions'] if all_failed else checklist['dimensions'][:1]:
        row['evidence'][0]['quote'] = 'A was observed under B with an invented qualifier.'
    request = read(root / 'requests/reference/P.json')
    digest = fingerprint([PROTOCOL, MODEL, 'xhigh', prompts.REFERENCE, request,
                          _strict_response_schema(Checklist.model_json_schema())])
    write(root / 'checkpoints/reference/P' / f'{digest}.json', checklist)
    (root / 'references/P.json').unlink()
    (root / 'provenance/references/P.json').unlink()
    envelope = recover_reference(root, 'P')
    report = read(root / 'reports/full/P.json')
    inputs = build_evaluation_input(root, 'P', 'full', report, envelope['checklist'], envelope)
    anonymous, mapping = blind(inputs['payload'])
    payload, index = prepare_selector_payload(anonymous)
    write(root / 'requests/evaluation/full/P.json', payload)
    write(root / 'requests/evaluation_index/full/P.json', index)
    write(root / 'requests/evaluation_mapping/full/P.json', mapping)
    source = next(alias for alias, row in index['selector_sources'].items()
                  if index['fragments'][row['fragment_id']]['source_field'] == 'text')
    raw = {'dimensions': [{'dimension': dimension, 'status': 'assessed', 'score': 2,
                           'report_quotes': ['R01'], 'reason': 'Grounded fixture analysis.',
                           'checks': [{'report_quote': 'R01', 'support': 'supported', 'scope_correct': True,
                                       'reason': 'Original observation.', 'evidence': [
                                           {'source_quote_ids': [{'fragment_id': source, 'clause_index': 0}],
                                            'explanation': 'Direct observation.'}]}]}
                          for dimension in envelope['valid_dimensions']]}
    resolved = resolve_selector_evaluation(raw, index, anonymous['reference_technical_failures'])
    result = replace_ids(resolved['value'], mapping)
    write(root / 'evaluations/full/P.json', result)
    write(root / 'provenance/evaluations/full/P.json', {
        'request_hash': fingerprint([PROTOCOL, MODEL, EVALUATION_PROTOCOL, inputs['instruction'], payload]),
        'protocol': PROTOCOL, 'model': MODEL, 'effort': 'xhigh', 'provenance_protocol': EVALUATION_PROTOCOL,
        'partial_reference_fingerprint': envelope['artifact_fingerprint'],
        'reference_technical_failures': envelope['technical_failures'],
        'inference_performed': bool(envelope['valid_dimensions']),
        'model_evaluated_dimensions': envelope['valid_dimensions'],
        'programmatic_dimensions': [r['dimension'] for r in envelope['technical_failures']],
        'audit': {'pointer_resolution': replace_ids(resolved['audit'], mapping)},
    })
    return envelope


def test_partial_reference_is_accepted_without_inventing_a_successful_reference(tmp_path: Path) -> None:
    envelope = partial_fixture(tmp_path)
    before = Path(envelope['original_checkpoint']).read_bytes()
    result, error = load_result(tmp_path, 'P', 'full')
    assert error is None and result is not None
    rows = {r['dimension']: r for r in result['dimensions']}
    assert rows['historical_originality']['status'] == 'technical_failure'
    assert rows['historical_originality']['score'] is None
    assert len([r for r in rows.values() if r['status'] == 'assessed']) == 4
    assert all(rows[d]['score'] == 2 for d in DIMENSIONS if d != 'historical_originality')
    assert Path(envelope['original_checkpoint']).read_bytes() == before
    assert not (tmp_path / 'references/P.json').exists()


def test_partial_reference_does_not_turn_reference_failure_into_zero(tmp_path: Path) -> None:
    partial_fixture(tmp_path)
    path = tmp_path / 'evaluations/full/P.json'
    value = read(path)
    value['dimensions'][0] = {'dimension': 'historical_originality', 'status': 'assessed', 'score': 0,
                              'checks': [], 'report_quotes': [], 'reason': 'Incorrectly zeroed'}
    write(path, value)
    assert 'technical failure differs from reference availability audit' in load_result(tmp_path, 'P', 'full')[1]


@pytest.mark.parametrize('location', ['partial_references/P.json', 'provenance/evaluations/full/P.json'])
def test_partial_reference_tampering_blocks_stale_results(tmp_path: Path, location: str) -> None:
    partial_fixture(tmp_path)
    path = tmp_path / location
    value = read(path)
    field = 'artifact_fingerprint' if location.startswith('partial_') else 'partial_reference_fingerprint'
    value[field] = 'stale'
    write(path, value)
    result, error = load_result(tmp_path, 'P', 'full')
    assert result is None and error is not None
    assert 'partial reference' in error.lower() or 'stale_partial_evaluation' in error


def test_all_failed_reference_has_no_model_scored_dimensions(tmp_path: Path) -> None:
    envelope = partial_fixture(tmp_path, all_failed=True)
    assert envelope['valid_dimensions'] == []
    result, error = load_result(tmp_path, 'P', 'full')
    assert error is None
    assert all(row['status'] == 'technical_failure' and row['score'] is None for row in result['dimensions'])
    meta = read(tmp_path / 'provenance/evaluations/full/P.json')
    assert meta['inference_performed'] is False
    assert meta['model_evaluated_dimensions'] == []


def test_existing_successful_reference_ignores_an_unrelated_partial_sidecar(tmp_path: Path) -> None:
    fixture(tmp_path)
    paths = [tmp_path / name for name in ('requests/reference/P.json', 'requests/evaluation/full/P.json',
                                          'provenance/references/P.json', 'provenance/evaluations/full/P.json')]
    before = {str(p): p.read_bytes() for p in paths}
    write(tmp_path / 'partial_references/P.json', {'invalid_unrelated_sidecar': True})
    result, error = load_result(tmp_path, 'P', 'full')
    assert error is None and result is not None
    assert all(p.read_bytes() == before[str(p)] for p in paths)
