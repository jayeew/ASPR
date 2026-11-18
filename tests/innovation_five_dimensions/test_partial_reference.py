from __future__ import annotations

import asyncio
import copy
import fcntl
from pathlib import Path

import pytest
from pydantic import ValidationError

from gear.codex_cli import _strict_response_schema
from figure_pipeline.innovation_five_dimensions import prompts
from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS, MODEL, PROTOCOL
from figure_pipeline.innovation_five_dimensions.evaluation import validate_evaluation
from figure_pipeline.innovation_five_dimensions.io import fingerprint, read, write
from figure_pipeline.innovation_five_dimensions.models import Checklist, Evaluation
from figure_pipeline.innovation_five_dimensions.partial_reference import (
    build_partial_reference, recover_reference, validate_partial_checklist, verify_partial_reference,
)
from figure_pipeline.innovation_five_dimensions.recovery import recover
from figure_pipeline.innovation_five_dimensions.runner import Study, blind, build_evaluation_input
from figure_pipeline.innovation_five_dimensions.selector_evaluation import (
    prepare_selector_payload, resolve_selector_evaluation, selector_schema,
)
from test_aggregate_provenance import fixture


def failed_reference(root: Path, all_failed: bool = False) -> dict:
    fixture(root)
    checklist = read(root / 'references/P.json')
    for row in checklist['dimensions'] if all_failed else checklist['dimensions'][:1]:
        row['evidence'][0]['quote'] = 'A fabricated source quotation.'
    packet = read(root / 'requests/reference/P.json')
    shape = _strict_response_schema(Checklist.model_json_schema())
    digest = fingerprint([PROTOCOL, MODEL, 'xhigh', prompts.REFERENCE, packet, shape])
    write(root / 'checkpoints/reference/P' / f'{digest}.json', checklist)
    (root / 'references/P.json').unlink()
    return checklist


def partial_input(root: Path, all_failed: bool = False) -> tuple[dict, dict, dict, dict]:
    failed_reference(root, all_failed)
    envelope = recover_reference(root, 'P')
    report = read(root / 'reports/full/P.json')
    inputs = build_evaluation_input(root, 'P', 'full', report, envelope['checklist'], envelope)
    anonymous, _ = blind(inputs['payload'])
    payload, index = prepare_selector_payload(anonymous)
    return envelope, inputs, payload, index


def raw_partial(payload: dict, allowed: tuple[str, ...]) -> dict:
    return {'dimensions': [{'dimension': dim, 'status': 'assessed', 'score': 2,
                            'checks': [{'report_quote': 'R01', 'support': 'supported', 'scope_correct': True,
                                        'evidence': [{'source_quote_ids': [{'fragment_id': 'S001', 'clause_index': 0}],
                                                      'explanation': 'The manuscript observation.'}], 'reason': 'Specific.'}],
                            'report_quotes': ['R01'], 'reason': 'Grounded.'} for dim in allowed]}


def test_recovery_uses_original_checkpoint_and_preserves_scientific_fields(tmp_path: Path) -> None:
    original = failed_reference(tmp_path)
    request = (tmp_path / 'requests/reference/P.json').read_bytes()
    envelope = recover_reference(tmp_path, 'P')
    assert envelope['checklist'] == original
    assert envelope['model_calls_during_recovery'] == 0
    assert envelope['valid_dimensions'] == list(DIMENSIONS)[1:]
    assert envelope['technical_failures'][0]['dimension'] == 'historical_originality'
    assert envelope['availability']['historical_originality']['failed_citations'][0]['quote'] == 'A fabricated source quotation.'
    assert (tmp_path / 'requests/reference/P.json').read_bytes() == request
    assert not (tmp_path / 'references/P.json').exists()
    assert verify_partial_reference(tmp_path, 'P') == envelope
    validate_partial_checklist(envelope, read(tmp_path / 'requests/reference/P.json')['source_passages'])


def test_recovery_rejects_changed_request_or_forged_partial_artifact(tmp_path: Path) -> None:
    failed_reference(tmp_path)
    recover_reference(tmp_path, 'P')
    stored = read(tmp_path / 'partial_references/P.json')
    stored['technical_failures'] = []
    write(tmp_path / 'partial_references/P.json', stored)
    with pytest.raises(ValueError, match='Stale or modified'):
        verify_partial_reference(tmp_path, 'P')
    packet = read(tmp_path / 'requests/reference/P.json')
    packet['core_contributions'] = ['Different scientific input']
    write(tmp_path / 'requests/reference/P.json', packet)
    with pytest.raises(ValueError, match='differs from current'):
        build_partial_reference(tmp_path, 'P')


def test_bad_criteria_are_hidden_and_model_cannot_return_a_failed_dimension(tmp_path: Path) -> None:
    envelope, inputs, payload, index = partial_input(tmp_path)
    valid = tuple(envelope['valid_dimensions'])
    assert [row['dimension'] for row in payload['checklist']['dimensions']] == list(valid)
    assert 'A fabricated source quotation.' not in str(payload)
    assert inputs['technical_failures'] == envelope['technical_failures']
    schema = selector_schema(index, valid)
    raw = raw_partial(payload, valid)
    schema.model_validate(raw)
    raw['dimensions'][0]['dimension'] = 'historical_originality'
    with pytest.raises(ValidationError): schema.model_validate(raw)


def test_programmatic_technical_dimension_has_no_score_and_requires_matching_origin(tmp_path: Path) -> None:
    envelope, inputs, payload, index = partial_input(tmp_path)
    raw = raw_partial(payload, tuple(envelope['valid_dimensions']))
    before = copy.deepcopy(raw)
    result = resolve_selector_evaluation(raw, index, envelope['technical_failures'])
    assert raw == before
    technical = result['value']['dimensions'][0]
    assert technical['status'] == 'technical_failure' and technical['score'] is None
    assert technical['checks'] == [] and technical['report_quotes'] == []
    assert technical['origin']['stage'] == 'reference_citation_validation'
    from figure_pipeline.innovation_five_dimensions.runner import replace_ids
    _, mapping = blind(inputs['payload'])
    restored = replace_ids(result['value'], mapping)
    report = read(tmp_path / 'reports/full/P.json')
    validate_evaluation(restored, report, inputs['sources'], envelope['checklist'], envelope['technical_failures'])
    with pytest.raises(ValueError, match='unauthorised technical'):
        validate_evaluation(restored, report, inputs['sources'], envelope['checklist'])
    restored['dimensions'][0]['origin']['error_count'] = 99
    with pytest.raises(ValueError, match='differs from reference'):
        validate_evaluation(restored, report, inputs['sources'], envelope['checklist'], envelope['technical_failures'])


def test_all_failed_dimensions_never_launch_a_model_and_do_not_fake_scientific_evaluation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    envelope, _, _, _ = partial_input(tmp_path, all_failed=True)
    study = Study(tmp_path, 1, 1)
    async def forbidden(*args, **kwargs):
        raise AssertionError('All-technical reference must not invoke a model')
    monkeypatch.setattr(study.scheduler, 'ask', forbidden)
    asyncio.run(study.evaluate('P', 'full', read(tmp_path / 'reports/full/P.json'),
                               envelope['checklist'], envelope))
    result = Evaluation.model_validate(read(tmp_path / 'evaluations/full/P.json')).model_dump()
    assert all(row['status'] == 'technical_failure' and row['score'] is None for row in result['dimensions'])
    meta = read(tmp_path / 'provenance/evaluations/full/P.json')
    assert meta['inference_performed'] is False and meta['model_evaluated_dimensions'] == []
    assert meta['programmatic_dimensions'] == list(DIMENSIONS)
    assert study.records['evaluation/full/P']['state'] == 'technical_failure_only'


def test_full_request_and_schema_remain_identical_without_partial_reference(tmp_path: Path) -> None:
    fixture(tmp_path)
    report, checklist = read(tmp_path / 'reports/full/P.json'), read(tmp_path / 'references/P.json')
    expected = read(tmp_path / 'requests/evaluation/full/P.json')
    inputs = build_evaluation_input(tmp_path, 'P', 'full', report, checklist)
    anonymous, _ = blind(inputs['payload'])
    actual, index = prepare_selector_payload(anonymous)
    assert actual == expected
    assert inputs['instruction'] == prompts.EVALUATE + __import__(
        'figure_pipeline.innovation_five_dimensions.selector_evaluation', fromlist=['SELECTOR_INSTRUCTION']).SELECTOR_INSTRUCTION
    assert 'technical_failure' not in str(selector_schema(index).model_json_schema())


def test_recovery_preserves_all_completion_and_does_not_start_a_second_scheduler(tmp_path: Path) -> None:
    write(tmp_path / 'status/all_completion.json', {'tasks': {}})
    write(tmp_path / 'inputs/roster.json', [])
    before = (tmp_path / 'status/all_completion.json').read_bytes()
    with (tmp_path / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='no second scheduler'):
            recover(tmp_path, maximum=1, initial=1)
    result = recover(tmp_path, maximum=1, initial=1)
    assert result['scheduled_evaluations'] == 0
    assert (tmp_path / 'status/all_completion.json').read_bytes() == before
    assert list((tmp_path / 'recovery_runs').glob('*/previous_status/all_completion.json'))
    assert (tmp_path / 'status/recovery_completion.json').exists()
