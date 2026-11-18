"""Read logged CLI costs and non-overlapping batch phases; never estimate missing usage."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .config import DAMAGE_ID, OUTPUT
from .io import read, write

TOKEN_FIELDS = ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_output_tokens')
UNASSIGNED = 'development_or_unassigned'


def instant(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return parsed.timestamp() if parsed.tzinfo is not None else None


def utc(value: float | None) -> str | None:
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def call_row(path: Path) -> dict[str, Any]:
    record = read(path)
    usage = record.get('usage') or {}
    protocol = record.get('quotation_protocol')
    if protocol is None and record['task'].startswith('evaluation/'):
        prompt = path.parent / 'prompt.txt'
        protocol = ('source_and_report_clause_pointers_v2' if prompt.exists()
                    and 'source_and_report_clause_pointers_v2' in prompt.read_text()
                    else 'literal_quote_development')
    started, admitted = instant(record.get('started_at')), instant(record.get('admitted_at'))
    queue = admitted - started if admitted is not None and started is not None else None
    return {'call_id': path.parent.name, 'task': record['task'], 'stage': record['task'].split('/')[0],
            'request_hash': record.get('request_hash'),
            'state': record['state'], 'cli_started': record.get('pid') is not None,
            'phase': UNASSIGNED, 'quotation_protocol': protocol,
            'model': record['model'], 'effort': record['reasoning_effort'],
            'started_at': record.get('started_at'), 'admitted_at': record.get('admitted_at'),
            'finished_at': record.get('finished_at'), 'seconds': record.get('seconds'),
            'queue_seconds': queue if queue is not None and queue >= 0 else None,
            'queue_timing_issue': 'negative_wall_clock_interval' if queue is not None and queue < 0 else None,
            **{field: usage.get(field) for field in TOKEN_FIELDS}, 'error': record.get('error', '')}


def group_usage(recorded: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [row for row in recorded if row['cli_started']]
    durations = [float(row['seconds']) for row in calls if row['seconds'] is not None]
    queues = [float(row['queue_seconds']) for row in calls if row['queue_seconds'] is not None]
    result = {'calls': len(calls), 'completed': sum(r['state'] == 'completed' for r in calls),
              'queued_not_started': len(recorded) - len(calls),
              'duration_reported_calls': len(durations), 'duration_missing_calls': len(calls) - len(durations),
              'cumulative_admitted_cli_seconds': sum(durations) if durations else None,
              'mean_seconds': mean(durations) if durations else None,
              'median_seconds': median(durations) if durations else None,
              'p90_seconds': sorted(durations)[min(len(durations)-1, int(.9*len(durations)))] if durations else None,
              'queue_reported_calls': len(queues), 'queue_missing_calls': len(calls) - len(queues),
              'mean_queue_seconds': mean(queues) if queues else None,
              'calls_without_token_usage': sum(r['input_tokens'] is None for r in calls)}
    for field in TOKEN_FIELDS:
        values = [r[field] for r in calls if r[field] is not None]
        result[field] = sum(values) if values else None
        result[field + '_reported_calls'] = len(values)
        result[field + '_missing_calls'] = len(calls) - len(values)
    return result


def completion_window(path: Path, phase: str, start: float, evidence: dict[str, Any]) -> dict[str, Any]:
    exists = path.is_file()
    completion = read(path) if exists else None
    return {'phase': phase, 'start': start, 'end': path.stat().st_mtime if exists else None,
            'completion': completion, 'completion_path': str(path),
            'boundary_evidence': evidence,
            'end_evidence': 'Preserved completion-file mtime, written after the final scheduler snapshot.'
                            if exists else 'No completion artifact; average concurrency withheld.'}


def phase_windows(output: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    windows, issues = [], []
    anchors = [r for r in rows if r['task'] == f'reference/{DAMAGE_ID}' and instant(r['started_at']) is not None]
    damage = [r for r in rows if DAMAGE_ID in r['task'] and instant(r['started_at']) is not None]
    if anchors:
        first = min(anchors, key=lambda row: instant(row['started_at']))
        start = instant(first['started_at'])
        if any(instant(row['started_at']) < start for row in damage):
            issues.append('DAMAGE has earlier logged calls; main-batch start cannot be inferred reliably.')
        else:
            windows.append(completion_window(output / 'status/all_completion.json', 'main_all', start, {
                'method': 'Earliest reference/DAMAGE started_at; runner schedules DAMAGE first in the all cohort.',
                'anchor_call_id': first['call_id'], 'anchor_task': first['task'],
                'no_earlier_damage_calls': True,
                'scope': 'Other batch membership is verified against completion task keys and scheduler call states.'}))
    else:
        issues.append('No reference/DAMAGE start anchor; main-batch phase remains unassigned.')
    for path in sorted((output / 'recovery_runs').glob('*/recovery_completion.json')):
        try:
            start = datetime.strptime(path.parent.name, '%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            issues.append(f'Unparseable recovery run stamp: {path.parent.name}')
            continue
        windows.append(completion_window(path, 'recovery_' + path.parent.name, start, {
            'method': 'UTC archive directory stamp created before this recovery scheduler starts.',
            'archive': str(path.parent)}))
    return windows, issues


def assign_phases(rows: list[dict[str, Any]], windows: list[dict[str, Any]]) -> list[str]:
    issues = []
    for row in rows:
        started = instant(row['started_at'])
        matches = [w for w in windows if started is not None and w['start'] <= started
                   and (w['end'] is None or started <= w['end'])]
        if len(matches) == 1:
            row['phase'] = matches[0]['phase']
        elif len(matches) > 1:
            issues.append(f'Overlapping batch windows for call {row["call_id"]}; call left unassigned.')
    return issues


def base_task(task: str) -> str:
    parts = task.split('/')
    return '/'.join(parts[:2] if parts[0] == 'reference' else parts[:3])


def batch_usage(window: dict[str, Any], rows: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {**group_usage(rows), 'start_utc': utc(window['start']), 'finish_utc': utc(window['end']),
              'completion_path': window['completion_path'], 'boundary_evidence': window['boundary_evidence'],
              'end_evidence': window['end_evidence'], 'actual_average_concurrency': None,
              'peak_active': None, 'wall_seconds': None, 'attribution_status': 'provisional', 'issues': []}
    stages = sorted({row['stage'] for row in rows})
    result['stages'] = {stage: group_usage([r for r in rows if r['stage'] == stage]) for stage in stages}
    completion = window['completion']
    if completion is None:
        result['issues'].append('Batch has not produced its completion artifact.')
        return result
    scheduler = completion.get('scheduler', {})
    result.update(peak_active=scheduler.get('peak_active'), wall_seconds=scheduler.get('elapsed_seconds'))
    issues = result['issues']
    if window['end'] < window['start']:
        issues.append('Completion mtime predates batch start.')
    expected = {k: v for k, v in scheduler.get('calls', {}).items() if k != 'reused' and v}
    admitted = [r for r in rows if r['admitted_at'] is not None]
    observed = dict(Counter(r['state'] for r in admitted))
    if observed != expected:
        issues.append(f'Call-state census differs from scheduler: observed={observed}, expected={expected}.')
    tasks = completion.get('tasks', {})
    if any(base_task(r['task']) not in tasks for r in rows):
        issues.append('At least one assigned call is absent from the completion task list.')
    if any(r['cli_started'] and (instant(r['finished_at']) is None or
                                instant(r['finished_at']) > window['end']) for r in rows):
        issues.append('At least one CLI call lacks a finish inside the recorded batch window.')
    if any(r['phase'] != window['phase'] and r['cli_started'] and
           instant(r['admitted_at']) is not None and instant(r['finished_at']) is not None and
           instant(r['admitted_at']) < window['start'] < instant(r['finished_at']) for r in all_rows):
        issues.append('A call assigned elsewhere overlaps this batch start.')
    elapsed = result['wall_seconds']
    if not isinstance(elapsed, (int, float)) or elapsed <= 0 or scheduler.get('active') != 0:
        issues.append('Final positive wall duration and inactive scheduler are required.')
    if result['duration_missing_calls']:
        issues.append('Some CLI calls lack admitted-duration records.')
    if not issues:
        active_seconds = result['cumulative_admitted_cli_seconds'] or 0.0
        average = active_seconds / elapsed
        peak = result['peak_active']
        if not isinstance(peak, (int, float)) or average > peak + .01:
            issues.append('Duration-based average exceeds, or lacks, the recorded scheduler peak.')
        else:
            result['actual_average_concurrency'] = average
    result['attribution_status'] = 'verified' if not issues else 'unreliable'
    return result


def usage_markdown(result: dict[str, Any]) -> str:
    lines = ['# 实际CLI资源用量', '',
             '仅统计有日志的实际CLI调用，按唯一call_id计数，包含开发、旧协议、重试、格式修复和引用定位。'
             '主代理与子代理交互消耗不可得，不混入或推算。每条日志只归入一个phase。', '',
             '`development_or_unassigned`包含正式全量之前已采纳的pilot任务及接口开发调用，'
             '也容纳无法可靠分批的记录，不表示这些科学结果全部废弃。', '',
             '`started_at`为排队记录创建时间，`admitted_at`为获准执行时间，`finished_at`为结束时间。'
             '`seconds`是准入后的单调时钟时长；`queue_seconds`另由前两项UTC时间相减。', '',
             '实际平均并发 = 本批累计准入后CLI秒数 / 本批scheduler.elapsed_seconds；峰值采用scheduler.peak_active。'
             '配置上限不作为实际并发。批次归属不能与结束记录逐项对上时，不给平均并发。', '',
             '| 批次 | CLI调用 | 墙钟秒数 | 实际平均并发 | 峰值 | 归属核查 |',
             '|---|---:|---:|---:|---:|---|']
    for name, row in result['phases'].items():
        avg = row.get('actual_average_concurrency')
        peak = row.get('peak_active')
        lines.append(f"| {name} | {row['calls']} | {row.get('wall_seconds') or '未定'} | "
                     f"{f'{avg:.3f}' if avg is not None else '未报告'} | {peak if peak is not None else '未定'} | "
                     f"{row.get('attribution_status', '未分批')} |")
    lines += ['', '各token字段分别记录reported/missing调用数；全缺记null，仅真实返回的0记0。'
              'cached_input_tokens已包含在input_tokens内，不额外相加；不推算价格。', '',
              '详细起止依据、归属核查问题及分阶段用量见model_usage.json；原始时间与phase见model_calls.csv。', '']
    return '\n'.join(lines)


def summarise_usage(output: Path = OUTPUT) -> dict[str, Any]:
    rows = [call_row(path) for path in sorted((output / 'logs/calls').glob('*/record.json'))]
    windows, issues = phase_windows(output, rows)
    issues.extend(assign_phases(rows, windows))
    ledger_path = output / 'status/full_rerun_calls.json'
    if ledger_path.exists():
        ledger = read(ledger_path)
        for row in rows:
            if row['task'] in ledger and row['request_hash'] == ledger[row['task']]['request_hash']:
                row['phase'] = 'full_rerun'
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row['stage']].append(row)
    phases = {w['phase']: batch_usage(w, [r for r in rows if r['phase'] == w['phase']], rows) for w in windows}
    phases[UNASSIGNED] = group_usage([r for r in rows if r['phase'] == UNASSIGNED])
    current = [r for r in rows if r['phase'] == 'full_rerun']
    if current:
        phases['full_rerun'] = {**group_usage(current), 'attribution_status': 'matched_request_hashes',
                               'stages': {stage: group_usage([r for r in current if r['stage'] == stage])
                                          for stage in ('report', 'evaluation')}}
    result = {'stages': {stage: group_usage(items) for stage, items in grouped.items()},
              'totals': group_usage(rows), 'phases': phases, 'phase_assignment_issues': issues,
              'phase_meanings': {UNASSIGNED: 'Pre-full includes adopted pilot and development, plus any '
                                 'unreliably assigned calls. This phase is not a discarded-results category.',
                                 'main_all': 'Actual CLI calls incurred during the main full-batch scheduler; '
                                 'reused adopted-pilot tasks do not incur another CLI call here.'},
              'coverage': {'included': 'Instrumented codex CLI calls only.',
                           'interactive_main_and_subagent_usage': None,
                           'interactive_usage_handling': 'Unavailable; neither included nor estimated.'},
              'call_count': sum(row['cli_started'] for row in rows),
              'queued_not_started': sum(not row['cli_started'] for row in rows),
              'interpretation': 'Instrumented CLI attempts only, including adopted pilot, development and repairs; '
                                'interactive main/sub-agent consumption is unavailable and not estimated. One phase per call. '
                                'seconds excludes queueing and uses the admitted monotonic duration. Tokens are '
                                'provider-reported; cached input is part of input, missing is not zero. No estimated prices.'}
    destination = output / 'tables'
    destination.mkdir(parents=True, exist_ok=True)
    if rows:
        with (destination / 'model_calls.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write(destination / 'model_usage.json', result)
    (destination / 'model_usage.md').write_text(usage_markdown(result), encoding='utf-8')
    return result
