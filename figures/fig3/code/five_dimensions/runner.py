"""Reuse old analyses; generate new reports and evaluate only their five-dimension body."""
from __future__ import annotations

import asyncio
import fcntl
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from . import prompts
from .client import Scheduler
from .citation_repair import repair_citations
from .config import DAMAGE_ID, MODEL, PILOT_IDS, PROTOCOL
from .evaluation import audit_report, locate_quote, validate_checklist, validate_evaluation
from .io import encoded, fingerprint, read, write
from .models import Checklist, Report
from .packing import full_source_packet, pack_sources, packet_sources
from .partial_reference import PARTIAL_INSTRUCTION, PARTIAL_REFERENCE_PROTOCOL
from .selector_evaluation import (EVALUATION_PROTOCOL, SELECTOR_INSTRUCTION,
                                  prepare_selector_payload, resolve_selector_evaluation, selector_schema)
from .reporting import write_report

JUDGE_PROVENANCE_PROTOCOL = 'pointer_only_citation_location_v1'

def input_payload(bundle: dict[str, Any], stage: str) -> dict[str, Any]:
    fixed = {'paper': bundle['paper']}
    if stage == 'report':
        fixed['prior_analysis'] = bundle['analysis']
    else:
        fixed['core_contributions'] = bundle['core_contributions']
    packet = (full_source_packet(bundle) if stage == 'report' and 'full_rerun_selection' in bundle
              else pack_sources(bundle['sources'], fixed, 72000))
    return {**fixed, **packet}


def generation_material_scope(generation_packet: dict[str, Any]) -> dict[str, Any]:
    """Describe only the report writer's actual visible source packet, never judge-only sources."""
    catalog: dict[str, dict[str, Any]] = {}
    for source in generation_packet['source_passages']:
        key = source['source_id']
        row = catalog.setdefault(key, {'source_id': key, 'source_type': source['source_type'],
                                       'visible_text_ranges': [], 'visible_metadata_fields': []})
        if row['source_type'] != source['source_type']:
            raise ValueError('One source ID has inconsistent source types in the generation packet')
        bounds = {'start': source['start'], 'end': source['end']}
        if bounds not in row['visible_text_ranges']:
            row['visible_text_ranges'].append(bounds)
        fields = set(row['visible_metadata_fields'])
        fields.update(key for key in source if key not in {'text', 'start', 'end'})
        row['visible_metadata_fields'] = sorted(fields)
    for row in catalog.values():
        row['visible_text_ranges'].sort(key=lambda bounds: (bounds['start'], bounds['end']))
    return {'scope_version': 'generation_access_v1', 'sources': list(catalog.values()),
            'source_counts_by_type': dict(Counter(row['source_type'] for row in catalog.values())),
            'earlier_analysis_provided': 'prior_analysis' in generation_packet,
            'meaning': ('Sources actually visible during report generation; listed text ranges only, plus listed '
                        'metadata fields and the earlier analysis. The evaluation may contain additional common '
                        'sources; those were not necessarily visible to the report writer.')}


def reference_citation_context(report: dict[str, Any], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    index = {s['source_id']: s for s in sources}
    for row in report['dimensions']:
        for citation in row['evidence']:
            source = index.get(citation['source_id'])
            if source is None:
                continue
            span = locate_quote(citation['quote'], source['text'])
            if span is None:
                continue
            start = max(0, span['start'] - 1000)
            end = min(len(source['text']), span['end'] + 1000)
            identity = (source['source_id'], start, end)
            if identity not in seen:
                result.append({k: v for k, v in source.items() if k != 'text'} |
                              {'start': start, 'end': end, 'text': source['text'][start:end]})
                seen.add(identity)
    return result


IDENTITY_FIELDS = {'source_id', 'original_source_id'}
IDENTITY_LIST_FIELDS = {'source_ids', 'cited_source_ids'}
CLASSIFICATION_FIELDS = {'dimension', 'status', 'support', 'applicability', 'source_type',
                         'historical_use', 'state', 'normalisation', 'problem'}


def replace_ids(value: Any, mapping: dict[str, str], field: str = '') -> Any:
    if isinstance(value, dict):
        return {key: replace_ids(item, mapping, key) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_ids(item, mapping, field) for item in value]
    if isinstance(value, str) and mapping:
        if field in CLASSIFICATION_FIELDS:
            return value
        if field in IDENTITY_FIELDS | IDENTITY_LIST_FIELDS:
            return mapping.get(value, value)
        # Complete identifiers only. In particular S1 must not alter S10 or S1::SEG002.
        alternatives = '|'.join(re.escape(key) for key in sorted(mapping, key=len, reverse=True) if key)
        if not alternatives:
            return value
        pattern = r'(?<![A-Za-z0-9_:/.-])(?:' + alternatives + r')(?![A-Za-z0-9_:/.-])'
        def replace(match: re.Match[str]) -> str:
            key = match[0]
            # A malformed citation ID such as "supported" must not rewrite ordinary prose.
            identifier_shaped = not key.isalpha() or key.isupper()
            bracketed = (match.start() > 0 and match.end() < len(value)
                         and value[match.start() - 1] in '[\"\'' and value[match.end()] in ']\"\'')
            return mapping[key] if identifier_shaped or bracketed else key
        return re.sub(pattern, replace, value)
    return value


def referenced_ids(value: Any) -> list[str]:
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in IDENTITY_FIELDS and isinstance(item, str) and item:
                result.append(item)
            elif key in IDENTITY_LIST_FIELDS and isinstance(item, list):
                result.extend(v for v in item if isinstance(v, str) and v)
            result.extend(referenced_ids(item))
    elif isinstance(value, list):
        result.extend(key for item in value for key in referenced_ids(item))
    return result


def blind(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    # Include invalid report citations and original aliases: their validity must not reveal the method.
    ids = list(dict.fromkeys(referenced_ids(payload)))
    text, namespace = encoded(payload), 0
    while True:
        prefix = f'ANON_EVIDENCE_{namespace}_'
        aliases = [f'{prefix}{i:05d}' for i in range(1, len(ids) + 1)]
        if all(alias not in text for alias in aliases):
            break
        namespace += 1
    mapping = dict(zip(ids, aliases))
    return replace_ids(payload, mapping), {value: key for key, value in mapping.items()}


def build_evaluation_input(output: Path, paper: str, method: str, report: dict[str, Any],
                           checklist: dict[str, Any], partial_reference: dict[str, Any] | None = None) -> dict[str, Any]:
    common = read(output / 'requests/reference' / f'{paper}.json')
    own = read(output / 'inputs/generation' / method / f'{paper}.json')
    generation = read(output / 'requests/generation' / method / f'{paper}.json')
    supplemental = partial_reference['supplemental_sources'] if partial_reference else []
    sources = common['source_passages'] + supplemental + reference_citation_context(report, own['sources'])
    failures = partial_reference['technical_failures'] if partial_reference else None
    allowed = tuple(partial_reference['valid_dimensions']) if partial_reference else None
    visible_checklist = ({'dimensions': [row for row in checklist['dimensions'] if row['dimension'] in allowed]}
                         if partial_reference else checklist)
    payload = {'paper': common['paper'], 'report': report, 'checklist': visible_checklist,
               'source_passages': sources, 'generation_material_scope': generation_material_scope(generation),
               'citation_audit': audit_report(report, packet_sources(generation)),
               'material_coverage': common['material_coverage']}
    instruction = prompts.EVALUATE + SELECTOR_INSTRUCTION
    if partial_reference:
        payload['reference_availability_protocol'] = PARTIAL_REFERENCE_PROTOCOL
        payload['reference_technical_failures'] = failures
        instruction += PARTIAL_INSTRUCTION
    return {'payload': payload, 'sources': sources, 'instruction': instruction,
            'technical_failures': failures, 'allowed_dimensions': allowed}


class Study:
    def __init__(self, output: Path, maximum: int, initial: int, timeout: int = 900) -> None:
        self.output = output
        self.scheduler = Scheduler(output, maximum, initial, timeout)
        self.references: dict[str, asyncio.Task[Any]] = {}
        self.records: dict[str, dict[str, Any]] = {}

    def paths(self, stage: str, paper: str, method: str = '') -> tuple[Path, Path]:
        parts = [stage] + ([method] if method else []) + [paper + '.json']
        result = self.output.joinpath(*parts)
        meta = self.output.joinpath('provenance', *parts)
        return result, meta

    def reusable(self, path: Path, meta: Path, request_hash: str) -> bool:
        return path.exists() and meta.exists() and read(meta).get('request_hash') == request_hash

    def status(self, key: str, state: str, **details: Any) -> None:
        self.records[key] = {'state': state, **details}
        write(self.output / 'status/tasks.json', self.records)

    async def reference(self, paper: str) -> dict[str, Any]:
        if paper not in self.references:
            self.references[paper] = asyncio.create_task(self.make_reference(paper))
        return await self.references[paper]

    async def make_reference(self, paper: str) -> dict[str, Any]:
        key = 'reference/' + paper
        self.status(key, 'running')
        bundle = read(self.output / 'inputs/reference' / f'{paper}.json')
        payload = input_payload(bundle, 'reference')
        digest = fingerprint([PROTOCOL, MODEL, JUDGE_PROVENANCE_PROTOCOL, prompts.REFERENCE, payload])
        path, meta = self.paths('references', paper)
        write(self.output / 'requests/reference' / f'{paper}.json', payload)
        sources = packet_sources(payload)
        if self.reusable(path, meta, digest):
            value = read(path)
            audit = validate_checklist(value, sources)
            write(meta, {**read(meta), 'audit': audit})
            self.status(key, 'reused')
            return value
        value = await self.scheduler.ask(key, prompts.REFERENCE, payload, Checklist, 'xhigh')
        value, audit = await self.repair_judge(key, payload, value,
                                             lambda row: validate_checklist(row, sources))
        write(path, value)
        write(meta, {'request_hash': digest, 'protocol': PROTOCOL, 'audit': audit,
                     'model': MODEL, 'effort': 'xhigh', 'provenance_protocol': JUDGE_PROVENANCE_PROTOCOL})
        self.status(key, 'completed')
        return value

    async def report(self, paper: str, method: str) -> dict[str, Any]:
        key = f'report/{method}/{paper}'
        self.status(key, 'running')
        bundle = read(self.output / 'inputs/generation' / method / f'{paper}.json')
        payload = input_payload(bundle, 'report')
        instruction = prompts.report_instruction(payload)
        digest = fingerprint([PROTOCOL, MODEL, instruction, payload])
        path, meta = self.paths('reports', paper, method)
        write(self.output / 'requests/generation' / method / f'{paper}.json', payload)
        if self.reusable(path, meta, digest):
            value = Report.model_validate(read(path)).model_dump(mode='json')
            audit = audit_report(value, packet_sources(payload))
            write(meta, {**read(meta), 'citation_audit': audit})
            self.status(key, 'reused')
            return value
        value = await self.scheduler.ask(key, instruction, payload, Report, 'high')
        audit = audit_report(value, packet_sources(payload))
        write_report(self.output, paper, method, value)
        write(meta, {'request_hash': digest, 'protocol': PROTOCOL, 'citation_audit': audit,
                     'model': MODEL, 'effort': 'high', 'original_analysis_unchanged': True})
        self.status(key, 'completed', unlocated_citations=audit['unlocated_count'])
        return value

    async def repair_judge(self, key: str, payload: dict[str, Any], value: dict[str, Any],
                           validator: Callable[[dict[str, Any]], dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return value, validator(value)
        except ValueError as exc:
            # Repair provenance pointers once, never regenerate scientific judgments or scores.
            # Inapplicability/unsupported-label errors without quote problems remain failures.
            fixed = await repair_citations(self.scheduler, key, value,
                                            payload['source_passages'], payload.get('report'))
            audit = validator(fixed['value'])
            audit['citation_location_repair'] = fixed['audit']
            audit['initial_validation_error'] = str(exc)
            return fixed['value'], audit

    async def evaluate(self, paper: str, method: str, report: dict[str, Any],
                       checklist: dict[str, Any], partial_reference: dict[str, Any] | None = None) -> None:
        key = f'evaluation/{method}/{paper}'
        self.status(key, 'running')
        inputs = build_evaluation_input(self.output, paper, method, report, checklist, partial_reference)
        sources, failures, allowed = inputs['sources'], inputs['technical_failures'], inputs['allowed_dimensions']
        anonymous, mapping = blind(inputs['payload'])
        pointer_payload, pointer_index = prepare_selector_payload(anonymous)
        instruction = inputs['instruction']
        digest = fingerprint([PROTOCOL, MODEL, EVALUATION_PROTOCOL, instruction, pointer_payload])
        path, meta = self.paths('evaluations', paper, method)
        write(self.output / 'requests/evaluation' / method / f'{paper}.json', pointer_payload)
        write(self.output / 'requests/evaluation_mapping' / method / f'{paper}.json', mapping)
        write(self.output / 'requests/evaluation_index' / method / f'{paper}.json', pointer_index)
        if self.reusable(path, meta, digest):
            validate_evaluation(read(path), report, sources, checklist, failures)
            self.status(key, 'reused')
            return
        raw = ({'dimensions': []} if allowed == () else await self.scheduler.ask(
            key, instruction, pointer_payload, selector_schema(pointer_index, allowed), 'xhigh'))
        # Failed dimensions are administrative rows, never emitted or scored by the judge.
        anonymous_failures = anonymous.get('reference_technical_failures') if partial_reference else None
        resolved = resolve_selector_evaluation(raw, pointer_index, anonymous_failures)
        restored = replace_ids(resolved['value'], mapping)
        audit = validate_evaluation(restored, report, sources, checklist, failures)
        audit['pointer_resolution'] = replace_ids(resolved['audit'], mapping)
        write(path, restored)
        provenance = {'request_hash': digest, 'protocol': PROTOCOL, 'audit': audit,
                      'model': MODEL, 'effort': 'xhigh', 'anonymous_request_saved': True,
                      'provenance_protocol': EVALUATION_PROTOCOL}
        if partial_reference:
            provenance.update(partial_reference_fingerprint=partial_reference['artifact_fingerprint'],
                              reference_technical_failures=failures, inference_performed=bool(allowed),
                              model_evaluated_dimensions=list(allowed),
                              programmatic_dimensions=[row['dimension'] for row in failures])
        write(meta, provenance)
        self.status(key, ('technical_failure_only' if allowed == () else 'partial_completed')
                    if failures else 'completed')

    async def one(self, paper: str, method: str) -> None:
        current = f'report/{method}/{paper}'
        try:
            report = await self.report(paper, method)
            current = f'reference/{paper}'
            reference = await self.reference(paper)
            current = f'evaluation/{method}/{paper}'
            await self.evaluate(paper, method, report, reference)
        except (ValueError, RuntimeError, OSError, TimeoutError, KeyError) as exc:
            self.status(current, 'failed', error_type=type(exc).__name__, error=str(exc))
            self.status(f'pipeline/{method}/{paper}', 'failed', blocked_by=current)

    async def progress(self) -> None:
        while True:
            counts: dict[str, int] = {}
            for row in self.records.values():
                counts[row['state']] = counts.get(row['state'], 0) + 1
            snapshot = {**self.scheduler.snapshot(), 'tasks': counts}
            write(self.output / 'status/progress.json', snapshot)
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            await asyncio.sleep(30)

    async def run(self, cohort: str, selected: list[str] | None = None) -> dict[str, Any]:
        roster = read(self.output / 'inputs/roster.json')
        if cohort == 'pilot':
            roster = [p for p in roster if p['paper_id'] in PILOT_IDS]
        elif cohort == 'damage':
            roster = [p for p in roster if p['paper_id'] == DAMAGE_ID]
        elif cohort == 'all':
            # Schedule the independent illustrative case early for concurrent figure preparation.
            roster = sorted(roster, key=lambda p: p['paper_id'] != DAMAGE_ID)
        if selected:
            roster = [p for p in roster if p['paper_id'] in selected]
        for paper in roster:
            self.references[paper['paper_id']] = asyncio.create_task(self.make_reference(paper['paper_id']))
        progress = asyncio.create_task(self.progress())
        try:
            await asyncio.gather(*(self.one(p['paper_id'], method) for p in roster for method in p['methods']))
            # Retrieve every checklist exception, including report-blocked papers.
            await asyncio.gather(*self.references.values(), return_exceptions=True)
        finally:
            progress.cancel()
            await asyncio.gather(progress, return_exceptions=True)
        summary = {'protocol': PROTOCOL, 'cohort': cohort, 'papers': len(roster),
                   'scheduler': self.scheduler.snapshot(), 'tasks': self.records,
                   'failed_tasks': sum(r['state'] == 'failed' for r in self.records.values()),
                   'failed_scientific_tasks': sum(r['state'] == 'failed' for key, r in self.records.items()
                                                  if not key.startswith('pipeline/')),
                   'blocked_pipelines': sum(r['state'] == 'failed' for key, r in self.records.items()
                                            if key.startswith('pipeline/'))}
        counts: dict[str, int] = {}
        for row in self.records.values():
            counts[row['state']] = counts.get(row['state'], 0) + 1
        write(self.output / 'status/progress.json', {**self.scheduler.snapshot(), 'tasks': counts,
                                                   'state': 'finished', 'cohort': cohort})
        write(self.output / 'status' / f'{cohort}_completion.json', summary)
        return summary


def run(output: Path, cohort: str, maximum: int, initial: int,
        selected: list[str] | None = None, timeout: int = 900) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another five-dimensional study runner is active') from exc
        return asyncio.run(Study(output, maximum, initial, timeout).run(cohort, selected))
