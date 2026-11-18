"""Deterministic recovery of already-generated references; never request a new judgment."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from gear.codex_cli import _strict_response_schema

from . import prompts
from .citation_repair import build_candidates, collect_targets
from .config import DIMENSIONS, MODEL, PROTOCOL
from .evaluation import evidence_audit, source_index
from .io import fingerprint, read, write
from .models import Checklist
from .reference_visibility import repair_visible_reference_locations

PARTIAL_REFERENCE_PROTOCOL = 'reference_dimension_technical_failure_v1'
PARTIAL_INSTRUCTION = '''
REFERENCE AVAILABILITY: Only independently source-verifiable checklist dimensions are included below.
Return judgments ONLY for those dimensions, exactly once each. The omitted dimensions have reference
citation-validation technical failures listed in reference_technical_failures. Do not infer or reconstruct
their missing criteria, score them, reinterpret their applicability, or judge their underlying science.
Software will add technical_failure/null rows for those dimensions. They are neither low scores nor
scientific uncertainty. Apply the existing scientific rubric unchanged to the available dimensions.
'''


def replay_saved_quote_repairs(output: Path, paper: str, value: dict[str, Any],
                               sources: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = collect_targets(value, sources, None)
    if not targets:
        return copy.deepcopy(value), {'state': 'no_repair_needed', 'applied': []}
    candidates, _ = build_candidates(targets, sources, None)
    digest = fingerprint({'targets': targets, 'candidate_fragments': candidates})
    path = output / 'audits/citation_repair/reference' / paper / f'{digest}.json'
    if not path.exists():
        return copy.deepcopy(value), {'state': 'no_saved_matching_repair', 'expected_audit': str(path), 'applied': []}
    saved, result, applied = read(path), copy.deepcopy(value), []
    if saved.get('request_fingerprint') != digest:
        raise ValueError('Saved quote-location audit does not match its original request')
    for target in saved['targets']:
        if not target['repaired']:
            continue
        parent = result
        for part in target['path'][:-1]:
            parent = parent[part]
        if parent['source_id'] != target['source_id'] or parent[target['path'][-1]] != target['quote']:
            raise ValueError('Saved quote repair does not match the original checklist quotation')
        quote = target['span']['text']
        if not evidence_audit({'source_id': target['source_id'], 'quote': quote}, source_index(sources))['located']:
            raise ValueError('Saved quote repair is not located in its original visible packet')
        pointer = target['pointer']
        if not (pointer['present'] and pointer['same_assertion_and_scope'] and pointer['confidence'] == 'high'):
            raise ValueError('Saved quotation was not accepted as the same complete assertion')
        parent[target['path'][-1]] = quote
        applied.append(target)
    return result, {'state': 'replayed_saved_quote_only_locations', 'audit_path': str(path),
                    'request_fingerprint': digest, 'applied': applied}


def reference_availability(checklist: dict[str, Any], sources: list[dict[str, Any]],
                           paper: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = Checklist.model_validate(checklist).model_dump(mode='json')
    index, availability, failures = source_index(sources), {}, []
    for row in value['dimensions']:
        quotes = [dict(evidence_index=i, **evidence_audit(evidence, index))
                  for i, evidence in enumerate(row['evidence'])]
        invalid = [quote for quote in quotes if not quote['located']]
        availability[row['dimension']] = {'status': 'technical_failure' if invalid else 'available',
                                          'citation_audit': quotes, 'failed_citations': invalid}
        if invalid:
            failures.append({'dimension': row['dimension'],
                             'reason': f'Reference citation provenance failed for {len(invalid)} quotation(s); no scientific score assigned.',
                             'origin': {'stage': 'reference_citation_validation', 'paper_id': paper,
                                        'dimension': row['dimension'],
                                        'audit_path': f'partial_references/{paper}.json#availability/{row["dimension"]}',
                                        'error_count': len(invalid)}})
    return availability, failures


def scientific_fields(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    for row in result['dimensions']:
        for evidence in row['evidence']:
            evidence.pop('quote')
    return result


def build_partial_reference(output: Path, paper: str) -> dict[str, Any]:
    from .runner import JUDGE_PROVENANCE_PROTOCOL, input_payload
    bundle = read(output / 'inputs/reference' / f'{paper}.json')
    packet = read(output / 'requests/reference' / f'{paper}.json')
    if packet != input_payload(bundle, 'reference'):
        raise ValueError('Reference recovery cannot use a request that differs from current original inputs')
    shape = _strict_response_schema(Checklist.model_json_schema())
    digest = fingerprint([PROTOCOL, MODEL, 'xhigh', prompts.REFERENCE, packet, shape])
    checkpoint = output / 'checkpoints/reference' / paper / f'{digest}.json'
    if not checkpoint.is_file():
        raise ValueError('No original schema-valid checklist checkpoint; recovery cannot create scientific judgments')
    original = Checklist.model_validate(read(checkpoint)).model_dump(mode='json')
    repaired, repair_audit = replay_saved_quote_repairs(output, paper, original, packet['source_passages'])
    visible = repair_visible_reference_locations(packet, bundle['sources'], repaired)
    if scientific_fields(original) != scientific_fields(visible['value']):
        raise ValueError('Deterministic reference recovery changed scientific checklist content')
    availability, failures = reference_availability(visible['value'], visible['sources'], paper)
    count = len(packet['source_passages'])
    envelope = {'paper_id': paper, 'protocol': PARTIAL_REFERENCE_PROTOCOL,
                'reference_request_hash': fingerprint([PROTOCOL, MODEL, JUDGE_PROVENANCE_PROTOCOL, prompts.REFERENCE, packet]),
                'checklist': visible['value'], 'valid_dimensions': [d for d in DIMENSIONS if availability[d]['status'] == 'available'],
                'technical_failures': failures, 'availability': availability,
                'supplemental_sources': visible['sources'][count:], 'visibility_audit': visible['audit'],
                'original_checkpoint': str(checkpoint), 'original_checkpoint_hash': fingerprint(original),
                'saved_quote_repair': repair_audit, 'scientific_fields_unchanged': True,
                'original_request_unchanged': True, 'model_calls_during_recovery': 0}
    return {**envelope, 'artifact_fingerprint': fingerprint(envelope)}


def recover_reference(output: Path, paper: str) -> dict[str, Any]:
    envelope = build_partial_reference(output, paper)
    write(output / 'partial_references' / f'{paper}.json', envelope)
    return envelope


def verify_partial_reference(output: Path, paper: str) -> dict[str, Any]:
    saved = read(output / 'partial_references' / f'{paper}.json')
    current = build_partial_reference(output, paper)
    if saved != current:
        raise ValueError('Stale or modified partial reference: original request/checkpoint/quotation audit differs')
    return saved


def validate_partial_checklist(envelope: dict[str, Any], sources: list[dict[str, Any]]) -> dict[str, Any]:
    availability, failures = reference_availability(envelope['checklist'], sources, envelope['paper_id'])
    if availability != envelope['availability'] or failures != envelope['technical_failures']:
        raise ValueError('Partial reference availability differs from current source citation validation')
    return availability
