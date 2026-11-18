from __future__ import annotations

from pathlib import Path

import pytest

from figure_pipeline.innovation_five_dimensions import prompts
from figure_pipeline.innovation_five_dimensions.aggregate import load_result
from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS, MODEL, PROTOCOL
from figure_pipeline.innovation_five_dimensions.evaluation import audit_report
from figure_pipeline.innovation_five_dimensions.io import fingerprint, read, write
from figure_pipeline.innovation_five_dimensions.packing import packet_sources
from figure_pipeline.innovation_five_dimensions.selector_evaluation import (
    EVALUATION_PROTOCOL, SELECTOR_INSTRUCTION, prepare_selector_payload, resolve_selector_evaluation,
)
from figure_pipeline.innovation_five_dimensions.runner import (
    JUDGE_PROVENANCE_PROTOCOL, blind, generation_material_scope, input_payload, reference_citation_context, replace_ids,
)


def save_provenance(root: Path, paper: str, method: str) -> None:
    """Supply real runner-compatible lineage to an already-written result fixture."""
    report = read(root / 'reports' / method / f'{paper}.json')
    checklist = read(root / 'references' / f'{paper}.json')
    source = {'source_id': 'S1', 'source_type': 'manuscript', 'text': 'A was observed under B.'}
    own = {'paper': {'paper_id': paper}, 'sources': [source], 'analysis': 'A was observed.'}
    neutral = {'paper': own['paper'], 'sources': [source], 'core_contributions': ['A']}
    write(root / 'inputs/generation' / method / f'{paper}.json', own)
    write(root / 'inputs/reference' / f'{paper}.json', neutral)
    generation, common = input_payload(own, 'report'), input_payload(neutral, 'reference')
    sources = common['source_passages'] + reference_citation_context(report, own['sources'])
    evaluation = {'paper': common['paper'], 'report': report, 'checklist': checklist,
                  'source_passages': sources, 'generation_material_scope': generation_material_scope(generation),
                  'citation_audit': audit_report(report, packet_sources(generation)),
                  'material_coverage': common['material_coverage']}
    anonymous, mapping = blind(evaluation)
    pointer_payload, pointer_index = prepare_selector_payload(anonymous)
    write(root / 'requests/evaluation_mapping' / method / f'{paper}.json', mapping)
    write(root / 'requests/evaluation_index' / method / f'{paper}.json', pointer_index)
    for stage, folder, payload, prompt in (
        ('generation', 'reports', generation, prompts.REPORT),
        ('reference', 'references', common, prompts.REFERENCE),
        ('evaluation', 'evaluations', pointer_payload, prompts.EVALUATE + SELECTOR_INSTRUCTION),
    ):
        suffix = Path(f'{paper}.json') if stage == 'reference' else Path(method) / f'{paper}.json'
        write(root / 'requests' / stage / suffix, payload)
        protocol = EVALUATION_PROTOCOL if stage == 'evaluation' else JUDGE_PROVENANCE_PROTOCOL
        prefix = [PROTOCOL, MODEL] + ([] if stage == 'generation' else [protocol])
        meta = {'protocol': PROTOCOL, 'model': MODEL, 'effort': 'high' if stage == 'generation' else 'xhigh',
                'request_hash': fingerprint(prefix + [prompt, payload])}
        if stage != 'generation':
            meta['provenance_protocol'] = protocol
        if stage == 'evaluation':
            meta['audit'] = {'pointer_resolution': fixture_pointer_audit(
                read(root / 'evaluations' / method / f'{paper}.json'), pointer_index, mapping)}
        write(root / 'provenance' / folder / suffix, meta)


def fixture_pointer_audit(evaluation: dict, index: dict, mapping: dict) -> dict:
    """Resolve legal v3 selectors instead of asserting an artificial success flag."""
    reverse = {v: k for k, v in mapping.items()}
    raw = replace_ids(evaluation, reverse)
    def report_selector(text: str) -> str:
        return next(alias for alias, fragment in index['selector_reports'].items()
                    if index['fragments'][fragment]['text'] == text)
    def source_selector(text: str) -> dict:
        for alias, source in index['selector_sources'].items():
            for position, clause_id in enumerate(source['clause_ids']):
                clause = index['clauses'][clause_id]
                fragment = index['fragments'][clause['fragment_id']]
                if fragment['text'][clause['start']:clause['end']] == text:
                    return {'fragment_id': alias, 'clause_index': position}
        raise ValueError('Fixture quotation is not an actual source clause')
    for dimension in raw['dimensions']:
        dimension['report_quotes'] = [report_selector(q) for q in dimension['report_quotes']]
        for check in dimension['checks']:
            check['report_quote'] = report_selector(check['report_quote'])
            check['evidence'] = [{'source_quote_ids': [source_selector(e['quote'])],
                                  'explanation': e['explanation']} for e in check['evidence']]
    return replace_ids(resolve_selector_evaluation(raw, index)['audit'], mapping)


def fixture(root: Path) -> None:
    evidence = {'source_id': 'S1', 'quote': 'A was observed under B.', 'explanation': 'Direct observation.'}
    report = {'contribution_overview': 'A was observed.', 'synthesis': 'A was observed.',
              'dimensions': [{'dimension': d, 'judgment': 'A was observed.', 'rationale': 'A was observed.',
                              'scope': 'Only under B.', 'status': 'supported', 'evidence': [evidence]}
                             for d in DIMENSIONS]}
    checklist = {'dimensions': [{'dimension': d, 'applicability': 'applicable', 'criteria': ['Discuss A.'],
                                 'evidence': [evidence], 'scope': 'Only under B.', 'reason': 'Assessable.'}
                                for d in DIMENSIONS]}
    evaluation = {'dimensions': [{'dimension': d, 'status': 'assessed', 'score': 2,
                                  'checks': [{'report_quote': 'A was observed.', 'support': 'supported',
                                              'scope_correct': True, 'evidence': [evidence], 'reason': 'Accurate.'}],
                                  'report_quotes': ['A was observed.'], 'reason': 'Grounded.'} for d in DIMENSIONS]}
    write(root / 'reports/full/P.json', report)
    write(root / 'references/P.json', checklist)
    write(root / 'evaluations/full/P.json', evaluation)
    save_provenance(root, 'P', 'full')


def test_current_result_is_included_with_restored_source_ids(tmp_path: Path) -> None:
    fixture(tmp_path)
    result, error = load_result(tmp_path, 'P', 'full')
    assert error is None and result is not None
    assert result['dimensions'][0]['checks'][0]['evidence'][0]['source_id'] == 'S1'


@pytest.mark.parametrize('stage', ['reports/full', 'references', 'evaluations/full'])
def test_stale_success_is_retained_but_excluded(tmp_path: Path, stage: str) -> None:
    fixture(tmp_path)
    artifact = tmp_path / stage / 'P.json'
    before = artifact.read_bytes()
    meta = tmp_path / 'provenance' / stage / 'P.json'
    value = read(meta)
    value['request_hash'] = 'earlier-success-before-protocol-change'
    write(meta, value)
    result, error = load_result(tmp_path, 'P', 'full')
    assert result is None and 'stale_' in error
    assert artifact.read_bytes() == before


def test_changed_evaluation_request_excludes_old_success(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / 'requests/evaluation/full/P.json'
    value = read(path)
    value['paper']['title'] = 'new request whose evaluation failed'
    write(path, value)
    assert 'stale_evaluation_request' in load_result(tmp_path, 'P', 'full')[1]


def test_missing_provenance_is_explicit(tmp_path: Path) -> None:
    fixture(tmp_path)
    (tmp_path / 'provenance/evaluations/full/P.json').unlink()
    result, error = load_result(tmp_path, 'P', 'full')
    assert result is None and 'missing_evaluation_request_or_provenance' in error


def test_judge_citation_locations_rechecked(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / 'evaluations/full/P.json'
    value = read(path)
    value['dimensions'][0]['checks'][0]['evidence'][0]['quote'] = 'A was never observed.'
    write(path, value)
    assert 'quotation_not_in_source' in load_result(tmp_path, 'P', 'full')[1]


def test_inverse_mapping_must_match_request(tmp_path: Path) -> None:
    fixture(tmp_path)
    write(tmp_path / 'requests/evaluation_mapping/full/P.json', {})
    assert 'stale_evaluation_mapping' in load_result(tmp_path, 'P', 'full')[1]


def test_changed_upstream_input_excludes_old_request_and_success(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / 'inputs/reference/P.json'
    value = read(path)
    value['core_contributions'] = ['B']
    write(path, value)
    assert 'stale_reference_request' in load_result(tmp_path, 'P', 'full')[1]


@pytest.mark.parametrize('old_protocol', [JUDGE_PROVENANCE_PROTOCOL, 'source_and_report_clause_pointers_v2'])
def test_old_judge_protocol_is_excluded(tmp_path: Path, old_protocol: str) -> None:
    fixture(tmp_path)
    path = tmp_path / 'provenance/evaluations/full/P.json'
    value = read(path)
    value['provenance_protocol'] = old_protocol
    write(path, value)
    result, error = load_result(tmp_path, 'P', 'full')
    assert result is None and 'citation protocol mismatch' in error


def test_clause_index_cannot_be_reused_after_mutation(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / 'requests/evaluation_index/full/P.json'
    value = read(path)
    next(iter(value['fragments'].values()))['text'] = 'Different report text.'
    write(path, value)
    assert 'stale_evaluation_index' in load_result(tmp_path, 'P', 'full')[1]


def test_pointer_resolution_provenance_is_required(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / 'provenance/evaluations/full/P.json'
    value = read(path)
    value.pop('audit')
    write(path, value)
    assert 'missing_evaluation_pointer_resolution' in load_result(tmp_path, 'P', 'full')[1]


def test_pointer_audit_must_restore_original_source_ids(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / 'provenance/evaluations/full/P.json'
    value = read(path)
    row = next(r for r in value['audit']['pointer_resolution']['quotations'] if r['kind'] == 'source')
    row['source_id'] = 'ANON_EVIDENCE_0_00001'
    write(path, value)
    assert 'copied quotations or restored IDs differ' in load_result(tmp_path, 'P', 'full')[1]


def test_full_expanded_input_cannot_reuse_old_evaluation(tmp_path: Path) -> None:
    fixture(tmp_path)
    path = tmp_path / 'inputs/generation/full/P.json'
    bundle = read(path)
    source = bundle['sources'][0]
    bundle['full_rerun_selection'] = [{'source_id': source['source_id'], 'start': 0,
                                      'end': len(source['text'])}]
    write(path, bundle)
    result, error = load_result(tmp_path, 'P', 'full')
    assert result is None and 'stale_generation_request' in error
