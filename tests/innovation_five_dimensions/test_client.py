from __future__ import annotations

import asyncio
import errno
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from figure_pipeline.innovation_five_dimensions import client
from figure_pipeline.innovation_five_dimensions.client import (
    RefusalError,
    Scheduler,
    ToolUseError,
    inspect_events,
    transient_failure,
)


class Answer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: int


class ScientificAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    score: int
    judgment: str
    reasons: list[str]


@pytest.fixture
def scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scheduler:
    monkeypatch.setattr(client, 'available_memory_gib', lambda: 32.0)
    return Scheduler(tmp_path, maximum=4, initial=4)


def accelerate_sleep(monkeypatch: pytest.MonkeyPatch) -> Any:
    original = asyncio.sleep

    async def quick_sleep(seconds: float) -> None:
        await original(min(seconds, .001))

    monkeypatch.setattr(client.asyncio, 'sleep', quick_sleep)
    return original


def test_trailing_comma_is_repaired_locally_without_model_rewrite(scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = []
    broken = '{"answer": 2,}'

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        prompts.append(prompt)
        return broken if len(prompts) == 1 else '{"answer": 2}'

    monkeypatch.setattr(scheduler, 'process', process)
    answer = asyncio.run(scheduler.ask('report/P1', 'Answer', {'input': 1}, Answer, 'high'))
    assert answer == {'answer': 2}
    assert len(prompts) == 1 and scheduler.active == 0


@pytest.mark.parametrize('changes', [{'score': 3, 'judgment': 'Positive'},
                                     {'judgment': 'Weak support'}, {'reasons': ['B', 'A']}])
def test_format_repair_cannot_rewrite_science_or_reorder_lists(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch, changes: dict) -> None:
    original = {'score': 0, 'judgment': 'No', 'reasons': ['A', 'B'], 'extra': 'remove'}
    repaired = {'score': 0, 'judgment': 'No', 'reasons': ['A', 'B'], **changes}
    calls = []

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        calls.append(record['task'])
        return json.dumps(original if len(calls) == 1 else repaired)

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(ValueError, match='original scientific content'):
        asyncio.run(scheduler.ask('evaluation/P', 'Evaluate', {}, ScientificAnswer, 'xhigh'))
    assert calls == ['evaluation/P', 'evaluation/P/format_repair']
    assert not list((scheduler.output / 'checkpoints').rglob('*.json'))


def test_format_repair_can_delete_only_extra_and_preserves_low_score(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    values = [{'score': 0, 'judgment': 'No', 'reasons': ['A', 'B'], 'extra': 'remove'},
              {'score': 0, 'judgment': 'No', 'reasons': ['A', 'B']}]
    calls = []

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        calls.append(record['task'])
        return json.dumps(values[len(calls) - 1])

    monkeypatch.setattr(scheduler, 'process', process)
    assert asyncio.run(scheduler.ask('evaluation/P', 'Evaluate', {}, ScientificAnswer, 'xhigh')) == values[1]
    assert len(calls) == 2


@pytest.mark.parametrize('broken', ['{"answer":', '{"answer": unknown}', 'Here is an answer: {"answer":2}'])
def test_unparseable_scientific_value_never_goes_to_model_repair(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch, broken: str) -> None:
    calls = []

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        calls.append(record['task'])
        return broken

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(scheduler.ask('report/P', 'Answer', {}, Answer, 'high'))
    assert calls == ['report/P']
    assert not list((scheduler.output / 'checkpoints').rglob('*.json'))


def test_local_json_layout_repair_preserves_string_punctuation() -> None:
    assert client.parse_layout_json('```json\n{"quote":"literal ,} and ,]", "items":[0,],}\n```') == {
        'quote': 'literal ,} and ,]', 'items': [0]}
    client.verify_lossless_format({'value': '2.0'}, {'value': 2}, set())
    with pytest.raises(ValueError):
        client.verify_lossless_format({'value': False}, {'value': 0}, set())


def test_missing_required_content_does_not_enter_format_repair(scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        return '{"wrong_field": 2}'

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(ValidationError):
        asyncio.run(scheduler.ask('report/P1', 'Answer', {}, Answer, 'high'))
    assert count == 1 and scheduler.active == 0
    assert not list((scheduler.output / 'checkpoints').rglob('*.json'))


@pytest.mark.parametrize('response', [
    '{"refusal": "Cannot comply with this request."}',
    "I'm sorry, but I cannot assist with this request.",
    '抱歉，我无法协助这一请求。',
])
def test_explicit_refusal_never_enters_format_repair(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        return response

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(RefusalError):
        asyncio.run(scheduler.ask('report/P1', 'Answer', {}, Answer, 'high'))
    assert count == 1 and scheduler.active == 0


@pytest.mark.parametrize('kind', ['command_execution', 'file_change', 'mcp_tool_call',
                                 'web_search', 'dynamic_tool_call', 'function_call'])
def test_tool_execution_is_detected(kind: str) -> None:
    record: dict[str, Any] = {}
    with pytest.raises(ToolUseError):
        inspect_events(json.dumps({'type': 'item.completed', 'item': {'type': kind}}), record)
    assert record['tool_use_violations'] == [kind]


def test_tool_use_never_retries_even_if_message_mentions_connection(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        raise ToolUseError('connection tool attempted')

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(ToolUseError):
        asyncio.run(scheduler.ask('reference/P1', 'Answer', {}, Answer, 'xhigh'))
    assert count == 1 and scheduler.active == 0


def test_transient_timeout_retries_once_and_releases_slots(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        raise TimeoutError('temporary timeout')

    monkeypatch.setattr(scheduler, 'process', process)
    accelerate_sleep(monkeypatch)
    with pytest.raises(TimeoutError):
        asyncio.run(scheduler.ask('evaluate/P1', 'Answer', {}, Answer, 'xhigh'))
    assert count == 2 and scheduler.counts['timeout'] == 2 and scheduler.active == 0


@pytest.mark.parametrize('error', [PermissionError(errno.EACCES, 'Permission denied'),
                                  FileNotFoundError(errno.ENOENT, 'Missing executable'),
                                  OSError(errno.ENOSPC, 'No space left'),
                                  RuntimeError('schema is invalid'),
                                  RuntimeError('invalid source 142950')])
def test_permanent_errors_are_not_retried(scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch,
                                        error: BaseException) -> None:
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        raise error

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(type(error)):
        asyncio.run(scheduler.ask('evaluate/P1', 'Answer', {}, Answer, 'xhigh'))
    assert count == 1 and scheduler.active == 0


def test_retry_during_format_repair_does_not_restart_generation(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    records = []

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        records.append(record['task'])
        if len(records) == 1:
            return '{"answer": 3, "extra": "discard"}'
        if len(records) == 2:
            raise OSError(errno.ECONNRESET, 'Connection reset')
        return '{"answer":3}'

    monkeypatch.setattr(scheduler, 'process', process)
    accelerate_sleep(monkeypatch)
    assert asyncio.run(scheduler.ask('report/P1', 'Answer', {}, Answer, 'high')) == {'answer': 3}
    assert records == ['report/P1', 'report/P1/format_repair', 'report/P1/format_repair']


def test_all_stages_share_global_memory_bounded_concurrency(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    original_sleep = accelerate_sleep(monkeypatch)
    monkeypatch.setattr(scheduler, 'memory_ceiling', lambda: 2)
    observed = []

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        observed.append(scheduler.active)
        await original_sleep(.004)
        return '{"answer":1}'

    monkeypatch.setattr(scheduler, 'process', process)

    async def run() -> list[dict]:
        return await asyncio.gather(*[
            scheduler.ask(f'{stage}/P{i}', 'Answer', {'i': i}, Answer, 'high')
            for i in range(5) for stage in ('report', 'reference', 'evaluate')])

    results = asyncio.run(run())
    assert len(results) == 15 and max(observed) == 2
    assert scheduler.peak == 2 and scheduler.active == 0


def test_low_memory_pauses_new_tasks_without_killing_active(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    memory = [0]
    original_sleep = accelerate_sleep(monkeypatch)
    monkeypatch.setattr(scheduler, 'memory_ceiling', lambda: memory[0])

    async def run() -> None:
        pending = asyncio.create_task(scheduler.acquire())
        await original_sleep(.004)
        assert not pending.done() and scheduler.active == 0
        memory[0] = 1
        await asyncio.wait_for(pending, .1)
        memory[0] = 0
        assert scheduler.active == 1
        await scheduler.release('completed')

    asyncio.run(run())
    assert scheduler.active == 0


def test_adaptive_limits_respond_to_rate_limits_and_timeout_windows(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = Scheduler(tmp_path, maximum=64, initial=16)
    monkeypatch.setattr(scheduler, 'memory_ceiling', lambda: 64)

    async def window(states: list[str]) -> None:
        for state in states:
            await scheduler.acquire()
            await scheduler.release(state)

    asyncio.run(window(['completed'] * 32))
    assert scheduler.limit == 24
    asyncio.run(window(['timeout'] * 4 + ['completed'] * 28))
    assert scheduler.limit == 16
    asyncio.run(window(['rate_limit']))
    assert scheduler.limit == 8


def test_successful_call_is_reused_without_new_cli(scheduler: Scheduler,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        return '{"answer":1}'

    monkeypatch.setattr(scheduler, 'process', process)
    asyncio.run(scheduler.ask('report/P1', 'Answer', {}, Answer, 'high'))
    asyncio.run(scheduler.ask('report/P1', 'Answer', {}, Answer, 'high'))
    assert count == 1 and scheduler.counts['reused'] == 1


def test_timeout_terminates_cli_process_group(scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: Path) -> None:
    class Process:
        pid = 123456
        returncode = None

        async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
            if input is not None:
                await asyncio.sleep(10)
            return b'', b''

    process = Process()
    commands = []
    killed = []

    async def spawn(*args: str, **kwargs: Any) -> Process:
        commands.append((args, kwargs))
        return process

    def kill_group(pid: int, signal: int) -> None:
        killed.append(pid)
        process.returncode = -9

    scheduler.timeout = .001
    monkeypatch.setattr(client.shutil, 'which', lambda name: '/fake/codex')
    monkeypatch.setattr(client.asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setattr(client.os, 'killpg', kill_group)
    directory = tmp_path / 'call'
    directory.mkdir()
    with pytest.raises(TimeoutError):
        asyncio.run(scheduler.process(directory, 'Input', 'high', {}))
    assert killed == [123456]
    assert commands[0][1]['start_new_session'] is True
    assert '--ephemeral' in commands[0][0]
    assert '--output-schema' in commands[0][0]


def test_transient_classification_uses_network_error_codes() -> None:
    assert transient_failure(OSError(errno.ECONNRESET, 'Connection reset'))
    assert transient_failure(RuntimeError('HTTP 429 rate limit'))
    assert not transient_failure(PermissionError(errno.EACCES, 'Permission denied'))
    assert not transient_failure(RuntimeError('source 142950 was not found'))


def test_lossless_schema_format_repair_is_allowed_only_once(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        return '{"answer": "two"}'

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(ValidationError):
        asyncio.run(scheduler.ask('report/P1', 'Answer', {}, Answer, 'high'))
    assert count == 2 and scheduler.active == 0


def test_missing_dimension_does_not_trigger_scientific_completion(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    from figure_pipeline.innovation_five_dimensions.models import Checklist
    count = 0

    async def process(directory: Path, prompt: str, effort: str, record: dict) -> str:
        nonlocal count
        count += 1
        return '{"dimensions": []}'

    monkeypatch.setattr(scheduler, 'process', process)
    with pytest.raises(ValidationError, match='Exactly one'):
        asyncio.run(scheduler.ask('reference/P1', 'Answer', {}, Checklist, 'xhigh'))
    assert count == 1 and scheduler.active == 0


def test_ready_evaluation_is_not_queued_behind_all_reports_and_stages_do_not_starve(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    original_sleep = accelerate_sleep(monkeypatch)
    scheduler.limit = 1
    memory = [0]
    monkeypatch.setattr(scheduler, 'memory_ceiling', lambda: memory[0])
    admitted = []

    async def work(task: str) -> None:
        await scheduler.acquire(task)
        admitted.append(task.split('/')[0])
        assert scheduler.active == 1
        await original_sleep(.001)
        await scheduler.release('completed')

    async def run() -> None:
        # Arrival order deliberately reproduces Study.run: all references, then many reports.
        jobs = [asyncio.create_task(work(f'reference/{i}')) for i in range(8)]
        jobs += [asyncio.create_task(work(f'report/{i}')) for i in range(8)]
        await original_sleep(.003)
        jobs += [asyncio.create_task(work(f'evaluation/{i}')) for i in range(16)]
        await original_sleep(.003)
        memory[0] = 1
        await asyncio.wait_for(asyncio.gather(*jobs), 1)

    asyncio.run(run())
    assert admitted[:4] == ['evaluation', 'reference', 'evaluation', 'report']
    # With all queues supplied, every four admissions reserve one slot for each generation stage.
    assert all(admitted[i:i + 4].count('reference') == 1 for i in range(0, 32, 4))
    assert all(admitted[i:i + 4].count('report') == 1 for i in range(0, 32, 4))
    assert scheduler.peak == 1 and scheduler.active == 0
    assert all(not queue for queue in scheduler.waiting.values())


def test_cancelled_queued_task_does_not_block_its_stage(
        scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch) -> None:
    original_sleep = accelerate_sleep(monkeypatch)
    memory = [0]
    monkeypatch.setattr(scheduler, 'memory_ceiling', lambda: memory[0])

    async def run() -> None:
        cancelled = asyncio.create_task(scheduler.acquire('reference/cancelled'))
        remaining = asyncio.create_task(scheduler.acquire('reference/remaining'))
        await original_sleep(.003)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert len(scheduler.waiting['reference']) == 1
        memory[0] = 1
        await asyncio.wait_for(remaining, .1)
        await scheduler.release('completed')

    asyncio.run(run())
    assert scheduler.active == 0 and not scheduler.waiting['reference']
