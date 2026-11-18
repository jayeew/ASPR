from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from figure_pipeline.innovation_five_dimensions.config import DAMAGE_ID
from figure_pipeline.innovation_five_dimensions.io import read, write
from figure_pipeline.innovation_five_dimensions.usage import group_usage, summarise_usage


def iso(second: float) -> str:
    return datetime.fromtimestamp(second, timezone.utc).isoformat()


def test_full_rerun_usage_matches_request_not_reused_task_name(tmp_path: Path) -> None:
    for name in ('old', 'new', 'legacy'):
        call(tmp_path, name, 'report/full/P', 10, 12, 22)
        path = tmp_path / 'logs/calls' / name / 'record.json'
        value = read(path)
        if name != 'legacy':
            value['request_hash'] = name
        write(path, value)
    write(tmp_path / 'status/full_rerun_calls.json', {'report/full/P': {'request_hash': 'new'}})
    result = summarise_usage(tmp_path)
    assert result['phases']['full_rerun']['calls'] == 1
    assert result['phases']['development_or_unassigned']['calls'] == 2
    assert result['call_count'] == 3


def call(root: Path, identifier: str, task: str, start: float, admit: float, finish: float,
         usage: dict | None = None) -> None:
    write(root / 'logs/calls' / identifier / 'record.json', {
        'task': task, 'model': 'test-model', 'reasoning_effort': 'high', 'pid': 100,
        'state': 'completed', 'started_at': iso(start), 'admitted_at': iso(admit),
        'finished_at': iso(finish), 'seconds': finish - admit, 'usage': usage})


def complete(path: Path, elapsed: float, calls: int, tasks: list[str], end: float) -> None:
    write(path, {'scheduler': {'elapsed_seconds': elapsed, 'peak_active': 2, 'maximum': 64,
                                'active': 0, 'calls': {'completed': calls, 'reused': 20}},
                 'tasks': {task: {'state': 'completed'} for task in tasks}})
    os.utime(path, (end, end))


def batches(root: Path) -> str:
    reference = 'reference/' + DAMAGE_ID
    call(root, 'old', 'evaluation/direct_a/P', 10, 12, 22, {'input_tokens': 10, 'cached_input_tokens': 4})
    call(root, 'ref', reference, 100, 100, 160, {'input_tokens': 0})
    call(root, 'report', 'report/full/P', 100.2, 110, 150)
    complete(root / 'status/all_completion.json', 200, 2, [reference, 'report/full/P'], 300)
    stamp = datetime.fromtimestamp(400, timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    call(root, 'recover', 'evaluation/full/P', 410, 420, 450)
    call(root, 'repair', 'evaluation/full/P/format', 451, 460, 480, {'output_tokens': 20})
    complete(root / 'recovery_runs' / stamp / 'recovery_completion.json', 100, 2, ['evaluation/full/P'], 500)
    # Mutable status mirror must not be double-counted as a second recovery batch.
    complete(root / 'status/recovery_completion.json', 100, 2, ['evaluation/full/P'], 500)
    return stamp


def test_elapsed_concurrency_and_exclusive_recovery_assignment(tmp_path: Path) -> None:
    stamp = batches(tmp_path)
    result = summarise_usage(tmp_path)
    main = result['phases']['main_all']
    recovery = result['phases']['recovery_' + stamp]
    assert main['attribution_status'] == recovery['attribution_status'] == 'verified'
    assert main['actual_average_concurrency'] == .5  # (60 + 40) / 200, not configuration 64
    assert recovery['actual_average_concurrency'] == .5
    assert main['peak_active'] == 2
    assert sum(row['calls'] for row in result['phases'].values()) == result['call_count'] == 5
    assert result['phases']['development_or_unassigned']['calls'] == 1
    with (tmp_path / 'tables/model_calls.csv').open(encoding='utf-8-sig') as handle:
        rows = {r['call_id']: r for r in csv.DictReader(handle)}
    assert rows['report']['started_at'] == iso(100.2)
    assert rows['report']['admitted_at'] == iso(110)
    assert rows['report']['finished_at'] == iso(150)
    assert float(rows['report']['seconds']) == 40
    assert float(rows['report']['queue_seconds']) == pytest.approx(9.8)
    assert rows['repair']['phase'] == 'recovery_' + stamp


def test_missing_tokens_remain_null_and_real_zero_is_preserved(tmp_path: Path) -> None:
    batches(tmp_path)
    result = summarise_usage(tmp_path)
    main = result['phases']['main_all']
    assert main['input_tokens'] == 0
    assert main['input_tokens_reported_calls'] == 1
    assert main['input_tokens_missing_calls'] == 1
    assert main['cached_input_tokens'] is None
    assert main['cached_input_tokens_reported_calls'] == 0
    assert main['cached_input_tokens_missing_calls'] == 2
    assert result['totals']['input_tokens'] == 10  # Cached 4 not added again.
    assert result['totals']['cached_input_tokens'] == 4
    assert result['totals']['reasoning_output_tokens'] is None
    assert result['totals']['reasoning_output_tokens_missing_calls'] == 5
    assert group_usage([])['input_tokens'] is None


def test_unfinished_batch_has_no_average_concurrency(tmp_path: Path) -> None:
    call(tmp_path, 'ref', 'reference/' + DAMAGE_ID, 100, 110, 150)
    result = summarise_usage(tmp_path)
    main = result['phases']['main_all']
    assert main['actual_average_concurrency'] is None
    assert main['attribution_status'] == 'provisional'
    assert main['calls'] == 1


def test_inconsistent_scheduler_census_withholds_average(tmp_path: Path) -> None:
    batches(tmp_path)
    path = tmp_path / 'status/all_completion.json'
    complete(path, 200, 3, ['reference/' + DAMAGE_ID, 'report/full/P'], 300)
    main = summarise_usage(tmp_path)['phases']['main_all']
    assert main['actual_average_concurrency'] is None
    assert main['attribution_status'] == 'unreliable'
    assert any('census differs' in issue for issue in main['issues'])


def test_earlier_damage_call_prevents_main_start_inference(tmp_path: Path) -> None:
    batches(tmp_path)
    call(tmp_path, 'earlier_case', 'report/full/' + DAMAGE_ID, 50, 50, 60)
    result = summarise_usage(tmp_path)
    assert 'main_all' not in result['phases']
    assert any('earlier logged calls' in issue for issue in result['phase_assignment_issues'])


def test_unknown_start_keeps_logs_without_inventing_a_batch(tmp_path: Path) -> None:
    call(tmp_path, 'old', 'evaluation/full/P', 10, 10, 20)
    result = summarise_usage(tmp_path)
    assert result['call_count'] == 1
    assert set(result['phases']) == {'development_or_unassigned'}
    assert result['phase_assignment_issues']
