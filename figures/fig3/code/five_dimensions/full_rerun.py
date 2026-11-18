"""In-place, single-attempt replacement of exactly the main 100 Full reports."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken
from pydantic import BaseModel

from gear.codex_cli import _strict_response_schema

from . import prompts
from .config import DAMAGE_ID, MODEL, OUTPUT, PILOT_IDS, PROTOCOL, ROOT
from .io import encoded, fingerprint, read, write
from .runner import Study, input_payload


def main_roster(output: Path) -> list[dict[str, Any]]:
    roster = [r for r in read(output / 'inputs/roster.json') if r['cohort'] == 'main']
    ids = [r['paper_id'] for r in roster]
    if len(ids) != 100 or len(set(ids)) != 100 or DAMAGE_ID in ids:
        raise ValueError('Full rerun requires exactly the existing 100 unique main papers')
    return roster


def task_manifest(output: Path) -> list[str]:
    return [f'{stage}/full/{r["paper_id"]}' for r in main_roster(output)
            for stage in ('report', 'evaluation')]


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def select_union(output: Path, paper: str, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    sources = {s['source_id']: s for s in bundle['sources']}
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for method in ('gear', 'graph', 'full'):
        packet = read(output / 'requests/generation' / method / f'{paper}.json')
        for row in packet['source_passages']:
            source = sources[row['source_id']]
            if source['text'][row['start']:row['end']] != row['text']:
                raise ValueError(f'Original passage mismatch: {paper}/{method}/{row["source_id"]}')
            spans[row['source_id']].append((row['start'], row['end']))
    for source in bundle['sources']:
        if source['source_type'] == 'manuscript':
            spans[source['source_id']] = [(0, len(source['text']))]
    return [{'source_id': key, 'start': start, 'end': end}
            for key in sorted(spans) for start, end in merge_spans(spans[key])]


@lru_cache(maxsize=1)
def context_capacity() -> int:
    cache = Path.home() / '.codex/models_cache.json'
    models = read(cache)['models']
    model = next(m for m in models if m['slug'] == MODEL)
    return int(model['context_window'] * model.get('effective_context_window_percent', 95) / 100)


@lru_cache(maxsize=2)
def tokenizer(name: str) -> Any:
    return tiktoken.get_encoding(name)


def capacity_check(instruction: str, payload: dict[str, Any]) -> dict[str, int]:
    prompt = instruction + '\n\nINPUT:\n' + encoded(payload)
    # Conservative cross-tokenizer estimate, plus room for output/reasoning and CLI/schema.
    tokens = max(len(tokenizer(name).encode(prompt, disallowed_special=()))
                 for name in ('o200k_base', 'cl100k_base'))
    reserve = 65536 + 8192
    capacity = context_capacity()
    if tokens + reserve > capacity:
        raise ValueError(f'Capacity requires discussion: estimated {tokens} + reserve {reserve} > {capacity}')
    return {'estimated_input_tokens': tokens, 'reserved_tokens': reserve,
            'effective_context_tokens': capacity}


def protected_hashes(output: Path) -> dict[str, str]:
    paths: set[Path] = set()
    for folder in ('reports', 'evaluations', 'inputs/original_reports', 'inputs/generation',
                   'requests/generation', 'requests/evaluation', 'provenance/reports',
                   'provenance/evaluations'):
        for path in (output / folder).rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(output / folder)
            if folder == 'inputs/original_reports' or relative.parts[0] != 'full' or DAMAGE_ID in path.name:
                paths.add(path)
    for folder in ('references', 'partial_references', 'inputs/reference', 'requests/reference',
                   'provenance/references', 'framework'):
        paths.update(p for p in (output / folder).rglob('*') if p.is_file())
    paths.update(p for p in (ROOT / 'manuscript').rglob('*') if p.is_file())
    paths.update(p for p in (output / 'figures').glob('*') if p.is_file()
                 and not p.name.lower().startswith('fig3'))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def prepare(output: Path) -> dict[str, Any]:
    manifest_path = output / 'status/full_rerun.json'
    prior = read(manifest_path) if manifest_path.exists() else {}
    protected = prior.get('protected_hashes') or protected_hashes(output)
    rows, bundles = [], []
    for paper in main_roster(output):
        pid = paper['paper_id']
        path = output / 'inputs/generation/full' / f'{pid}.json'
        bundle = read(path)
        if 'full_rerun_selection' not in bundle:
            bundle['full_rerun_selection'] = select_union(output, pid, bundle)
        packet = input_payload(bundle, 'report')
        capacity = capacity_check(prompts.FULL_REPORT, packet)
        reference = output / 'references' / f'{pid}.json'
        partial = output / 'partial_references' / f'{pid}.json'
        if not reference.is_file() and not partial.is_file():
            raise ValueError(f'Existing reference missing; no generation authorised: {pid}')
        rows.append({'paper_id': pid, **capacity,
                     'source_characters': packet['material_coverage']['selected_characters']})
        bundles.append((path, bundle))
    for path, bundle in bundles:
        write(path, bundle)
    result = {**prior, 'tasks': task_manifest(output), 'capacity': rows,
              'protected_hashes': protected, 'evaluation_instruction_hash': fingerprint(prompts.EVALUATE),
              'report_instruction_hash': fingerprint(prompts.FULL_REPORT), 'state': 'prepared',
              'automatic_extra_calls': False, 'main_reports': 100, 'main_evaluations': 100}
    write(manifest_path, result)
    return {'reports': 100, 'evaluations': 100, 'max_estimated_input_tokens':
            max(r['estimated_input_tokens'] for r in rows), 'effective_context_tokens': context_capacity()}


async def ask_once(scheduler: Any, task: str, instruction: str, payload: dict[str, Any],
                   schema: type[BaseModel], effort: str) -> dict[str, Any]:
    if task not in task_manifest(scheduler.output):
        raise ValueError(f'Unauthorised model task: {task}')
    shape = _strict_response_schema(schema.model_json_schema())
    digest = fingerprint([PROTOCOL, MODEL, effort, instruction, payload, shape])
    checkpoint = scheduler.output / 'checkpoints' / task / f'{digest}.json'
    if checkpoint.exists():
        scheduler.counts['reused'] += 1
        return schema.model_validate(read(checkpoint)).model_dump(mode='json')
    ledger_path = scheduler.output / 'status/full_rerun_calls.json'
    ledger = read(ledger_path) if ledger_path.exists() else {}
    if task in ledger and ledger[task]['state'] not in {'scheduled', 'not_invoked'}:
        consume_retry_authorisation(scheduler.output, task, digest, ledger[task])
    capacity = capacity_check(instruction, payload)
    previous = ledger.get(task)
    ledger[task] = {'state': 'scheduled', 'request_hash': digest, **capacity}
    if previous and previous['state'] == 'needs_discussion':
        ledger[task]['previous_attempt'] = previous
    write(ledger_path, ledger)
    try:
        raw = await scheduler.invoke(task, instruction, payload, shape, effort, digest)
        result = schema.model_validate(raw).model_dump(mode='json')
        write(checkpoint, result)
    except (ValueError, RuntimeError, OSError, TimeoutError, asyncio.CancelledError) as exc:
        ledger = read(ledger_path)
        state = 'not_invoked' if ledger[task]['state'] == 'scheduled' else 'needs_discussion'
        ledger[task].update(state=state, error=str(exc))
        write(ledger_path, ledger)
        raise
    ledger = read(ledger_path)
    ledger[task]['state'] = 'completed'
    write(ledger_path, ledger)
    return result


def consume_retry_authorisation(output: Path, task: str, digest: str, previous: dict[str, Any]) -> None:
    path = output / 'status/full_rerun.json'
    manifest = read(path) if path.exists() else {}
    permission = manifest.get('extra_call_authorisations', {}).get(task, {})
    if (previous['state'] != 'needs_discussion' or previous['request_hash'] != digest
            or permission.get('allowed_additional_calls') != 1 or permission.get('used', 0) != 0
            or permission.get('reason') != 'user_approved_timeout_retry'
            or 'limited access to this content for safety reasons' in previous.get('error', '')):
        raise RuntimeError(f'Additional model call requires discussion: {task}; {previous["state"]}')
    permission['used'] = 1
    manifest['extra_calls'] = manifest.get('extra_calls', 0) + 1
    write(path, manifest)


class FullStudy(Study):
    def __init__(self, output: Path, maximum: int, initial: int, timeout: int) -> None:
        super().__init__(output, maximum, initial, timeout)
        self.scheduler.single_attempt = True
        state = output / 'status/tasks.json'
        self.records = read(state) if state.exists() else {}
        calls = output / 'status/full_rerun_calls.json'
        attempted = read(calls) if calls.exists() else {}
        self.records = {k: v for k, v in self.records.items() if k in attempted}

    async def one_full(self, paper: str) -> None:
        current = f'report/full/{paper}'
        try:
            report = await self.report(paper, 'full')
            current = f'evaluation/full/{paper}'
            path = self.output / 'references' / f'{paper}.json'
            partial = None if path.exists() else read(self.output / 'partial_references' / f'{paper}.json')
            checklist = read(path) if partial is None else partial['checklist']
            await self.evaluate(paper, 'full', report, checklist, partial)
        except (ValueError, RuntimeError, OSError, TimeoutError, KeyError) as exc:
            self.status(current, 'failed', error_type=type(exc).__name__, error=str(exc),
                        requires_discussion=True)
            if current.startswith('report/'):
                self.status(f'evaluation/full/{paper}', 'blocked', blocked_by=current,
                            model_call_performed=False)

    async def run_full(self, phase: str) -> dict[str, Any]:
        roster = main_roster(self.output)
        if phase == 'pilot':
            roster = [r for r in roster if r['paper_id'] in PILOT_IDS]
        progress = asyncio.create_task(self.progress())
        try:
            await asyncio.gather(*(self.one_full(r['paper_id']) for r in roster))
        finally:
            progress.cancel()
            await asyncio.gather(progress, return_exceptions=True)
        failed = [key for key, row in self.records.items() if row['state'] == 'failed']
        summary = {'phase': phase, 'papers': len(roster), 'scheduler': self.scheduler.snapshot(),
                   'failed_tasks': failed, 'state': 'needs_discussion' if failed else 'finished'}
        write(self.output / 'status/progress.json', summary)
        manifest = read(self.output / 'status/full_rerun.json')
        manifest.update(state=summary['state'], active_phase=None, last_phase=phase, last_completion=summary)
        write(self.output / 'status/full_rerun.json', manifest)
        return summary


def verify_protected(output: Path) -> dict[str, Any]:
    manifest = read(output / 'status/full_rerun.json')
    failures = [path for path, digest in manifest['protected_hashes'].items()
                if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    result = {'checked': len(manifest['protected_hashes']), 'changed': failures}
    if failures:
        raise ValueError(f'Protected files changed: {failures[:5]}')
    return result


def run_full(output: Path = OUTPUT, phase: str = 'prepare', timeout: int = 900) -> dict[str, Any]:
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == 'prepare':
            return prepare(output)
        verify_protected(output)
        if phase == 'check':
            return verify_protected(output)
        manifest = read(output / 'status/full_rerun.json')
        if manifest['evaluation_instruction_hash'] != fingerprint(prompts.EVALUATE):
            raise ValueError('Original evaluation instruction changed')
        if phase == 'run' and not manifest.get('last_completion', {}).get('state') == 'finished':
            raise ValueError('Pilot must finish before full execution')
        maximum, initial = (8, 8) if phase == 'pilot' else (64, 16)
        manifest.update(state='running', active_phase=phase)
        write(output / 'status/full_rerun.json', manifest)
        return asyncio.run(FullStudy(output, maximum, initial, timeout).run_full(phase))
