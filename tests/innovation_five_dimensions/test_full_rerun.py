from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from figure_pipeline.innovation_five_dimensions import full_rerun, prompts
from figure_pipeline.innovation_five_dimensions.client import Scheduler
from figure_pipeline.innovation_five_dimensions.io import read, write
from figure_pipeline.innovation_five_dimensions.runner import input_payload


def test_union_preserves_full_manuscript_and_merges_existing_evidence(tmp_path: Path) -> None:
    sources = [{'source_id': 'M', 'source_type': 'manuscript', 'text': 'whole manuscript'},
               {'source_id': 'H', 'source_type': 'abstract', 'text': '0123456789'}]
    for method, start, end in [('gear', 1, 5), ('graph', 4, 8), ('full', 7, 9)]:
        write(tmp_path / f'requests/generation/{method}/P.json', {'source_passages': [
            {**sources[1], 'start': start, 'end': end, 'text': sources[1]['text'][start:end]}]})
    bundle = {'paper': {'paper_id': 'P'}, 'sources': sources, 'analysis': 'whole prior analysis'}
    bundle['full_rerun_selection'] = full_rerun.select_union(tmp_path, 'P', bundle)
    packet = input_payload(bundle, 'report')
    assert [(r['source_id'], r['text']) for r in packet['source_passages']] == [
        ('H', '12345678'), ('M', 'whole manuscript')]
    assert packet['prior_analysis'] == 'whole prior analysis'
    assert prompts.report_instruction(packet) == prompts.FULL_REPORT
    del bundle['full_rerun_selection']
    assert prompts.report_instruction(input_payload(bundle, 'report')) == prompts.REPORT


def test_manifest_excludes_damage_and_other_methods(tmp_path: Path) -> None:
    write(tmp_path / 'inputs/roster.json', [{'paper_id': f'P{i}', 'cohort': 'main'} for i in range(100)]
          + [{'paper_id': full_rerun.DAMAGE_ID, 'cohort': 'damage'}])
    tasks = full_rerun.task_manifest(tmp_path)
    assert len(tasks) == len(set(tasks)) == 200
    assert sum(t.startswith('report/full/') for t in tasks) == 100
    assert sum(t.startswith('evaluation/full/') for t in tasks) == 100
    assert not any(full_rerun.DAMAGE_ID in t for t in tasks)


class Answer(BaseModel):
    value: str


def setup_scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scheduler:
    monkeypatch.setattr(full_rerun, 'task_manifest', lambda output: ['report/full/P'])
    monkeypatch.setattr(full_rerun, 'capacity_check', lambda *args: {})
    return Scheduler(tmp_path, single_attempt=True)


def test_success_resume_uses_checkpoint_without_second_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = setup_scheduler(tmp_path, monkeypatch)
    calls = []

    async def invoke(*args: object) -> dict:
        calls.append(args)
        return {'value': 'specific judgment'}

    monkeypatch.setattr(scheduler, 'invoke', invoke)
    first = asyncio.run(scheduler.ask('report/full/P', 'prompt', {}, Answer, 'high'))
    second = asyncio.run(scheduler.ask('report/full/P', 'prompt', {}, Answer, 'high'))
    assert first == second
    assert len(calls) == 1


def test_failure_does_not_retry_and_resume_requires_discussion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = setup_scheduler(tmp_path, monkeypatch)
    calls = []

    async def invoke(*args: object) -> dict:
        calls.append(args)
        path = tmp_path / 'status/full_rerun_calls.json'
        ledger = read(path)
        ledger['report/full/P']['state'] = 'started'
        write(path, ledger)
        raise TimeoutError('test timeout')

    monkeypatch.setattr(scheduler, 'invoke', invoke)
    with pytest.raises(TimeoutError):
        asyncio.run(scheduler.ask('report/full/P', 'prompt', {}, Answer, 'high'))
    with pytest.raises(RuntimeError, match='requires discussion'):
        asyncio.run(scheduler.ask('report/full/P', 'prompt', {}, Answer, 'high'))
    assert len(calls) == 1


def test_unapproved_stage_rejected_before_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = setup_scheduler(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='Unauthorised'):
        asyncio.run(scheduler.ask('reference/P', 'prompt', {}, Answer, 'xhigh'))


def test_explicit_retry_permission_is_consumed_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = setup_scheduler(tmp_path, monkeypatch)
    calls = []

    async def invoke(*args: object) -> dict:
        calls.append(args)
        path = tmp_path / 'status/full_rerun_calls.json'
        ledger = read(path)
        ledger['report/full/P']['state'] = 'started'
        write(path, ledger)
        raise TimeoutError('test timeout')

    monkeypatch.setattr(scheduler, 'invoke', invoke)
    with pytest.raises(TimeoutError):
        asyncio.run(scheduler.ask('report/full/P', 'prompt', {}, Answer, 'high'))
    write(tmp_path / 'status/full_rerun.json', {'extra_call_authorisations': {
        'report/full/P': {'allowed_additional_calls': 1, 'used': 0, 'reason': 'user_approved_timeout_retry'}}})
    with pytest.raises(TimeoutError):
        asyncio.run(scheduler.ask('report/full/P', 'prompt', {}, Answer, 'high'))
    with pytest.raises(RuntimeError, match='requires discussion'):
        asyncio.run(scheduler.ask('report/full/P', 'prompt', {}, Answer, 'high'))
    assert len(calls) == 2
    assert read(tmp_path / 'status/full_rerun.json')['extra_calls'] == 1


def test_full_one_reuses_reference_without_generating_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(full_rerun, 'task_manifest', lambda output: [])
    write(tmp_path / 'partial_references/P.json', {'checklist': {'dimensions': []}, 'valid_dimensions': []})
    study = full_rerun.FullStudy(tmp_path, 8, 8, 900)
    seen = []

    async def report(paper: str, method: str) -> dict:
        seen.append(('report', paper, method))
        return {'new': True}

    async def evaluate(*args: object) -> None:
        seen.append(('evaluation', *args))

    async def forbidden(*args: object) -> None:
        raise AssertionError('No reference generation')

    monkeypatch.setattr(study, 'report', report)
    monkeypatch.setattr(study, 'evaluate', evaluate)
    monkeypatch.setattr(study, 'reference', forbidden)
    monkeypatch.setattr(study, 'make_reference', forbidden)
    asyncio.run(study.one_full('P'))
    assert [x[0] for x in seen] == ['report', 'evaluation']
    assert seen[1][-1]['valid_dimensions'] == []
