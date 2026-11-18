"""Export the reduced six-panel Fig5 data without plotting or model calls."""
from __future__ import annotations

import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write

from .aggregate import interval, paired, parts_and_metrics
from .diagnostics import state as previous_state
from .models import Config
from .six_panel import ASPECTS

ROOT = Config().output.parent
OUT = ROOT / 'fig5_six_panel'
SOURCE = ROOT / 'fig5_revision'
FIG4 = ROOT / 'fig4_reference/experiment'


def state(row: dict[str, Any]) -> str:
    result = previous_state(row)
    if result == 'evaluation_unresolved' and row['scientific_unresolved'] == 0 and row['response_type'] == 'substantive_answer':
        return 'substantive_incomplete_or_ungrounded'
    return result


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def panel_a() -> None:
    samples = {r['paper_id']: r for r in csv_rows(SOURCE / 'samples.csv') if r['development'] == 'False'}
    metrics = [{**r, 'domain': samples[r['paper_id']]['group'], 'evaluation_protocol': 'Fig4_original_grounded_coverage'}
               for r in csv_rows(FIG4 / 'paper_aspect_metrics.csv') if r['paper_id'] in samples and r['condition'] in ('F', 'E')]
    pairs = [{**r, 'domain': samples[r['paper_id']]['group'], 'evaluation_protocol': 'Fig4_original_grounded_coverage'}
             for r in csv_rows(FIG4 / 'paired_effects.csv') if r['paper_id'] in samples and r['other'] == 'E']
    summaries = []
    for domain, aspect in sorted({(r['domain'], r['aspect']) for r in pairs}):
        selected = [r for r in pairs if r['domain'] == domain and r['aspect'] == aspect]
        values = [float(r['grounded_delta']) for r in selected if r['complete_pair'] == 'True' and r['grounded_delta']]
        full = [float(r['grounded_coverage']) for r in metrics if r['domain'] == domain and r['aspect'] == aspect and r['condition'] == 'F' and r['state'] == 'completed' and r['grounded_coverage']]
        summaries.append({'domain': domain, 'aspect': aspect, 'domain_roster_n': sum(s['group'] == domain for s in samples.values()),
                          'full_valid_n': len(full), 'full_grounded_coverage_mean': sum(full) / len(full) if full else None,
                          'contrast': 'F minus E', 'metric': 'original_grounded_coverage', **interval(values, 20261003)})
    for name, data in [('samples', list(samples.values())), ('paper_metrics', metrics), ('paired', pairs), ('domain_summary', summaries)]:
        write_csv(OUT / 'panel_a' / f'{name}.csv', data)


def core_panels() -> None:
    source = ROOT / 'fig5_initial'
    destination = OUT / 'core_data'
    shutil.copytree(source, destination, dirs_exist_ok=True)
    metrics = csv_rows(source / 'paper_metrics.csv')
    write_csv(OUT / 'panel_b/paper_metrics.csv', [r for r in metrics if r['condition'] in ('F', 'E50')])
    write_csv(OUT / 'panel_b/paired_answerable.csv', [r for r in csv_rows(source / 'paired_answerable.csv') if r['condition'] == 'E50'])
    write_csv(OUT / 'panel_b/conditions.csv', [r for r in csv_rows(source / 'conditions.csv') if r['condition'] in ('F', 'E50')])
    write_csv(OUT / 'panel_c/target_parts.csv', [r for r in csv_rows(source / 'critical_target_parts.csv') if r['condition'] != 'E50'])
    for name in ('paper_metrics', 'response_matrix'):
        shutil.copy2(source / f'{name}.csv', OUT / 'panel_d' / f'{name}.csv')
    config = Config(output=source)
    parts, _ = parts_and_metrics(config, read(source / 'tasks.json'))
    parts = [r for r in parts if r['aspect'] in ('historical_verification', 'knowledge_position')]
    transitions = []
    for paper in read(source / 'scope.json')['papers']:
        target = read(source / 'eligibility' / f'{paper}.json')['target']
        selected = {r['condition']: r for r in parts if r['paper_id'] == paper and (r['question_id'], r['part_id']) == (target['question_id'], target['part_id'])}
        full, control, critical = [selected[c] for c in ('F', 'NONCRITICAL', 'CRITICAL')]
        transitions.append({'paper_id': paper, 'question_id': target['question_id'], 'part_id': target['part_id'],
                            'F_state': state(full), 'NONCRITICAL_state': state(control), 'CRITICAL_state': state(critical),
                            'noncritical_harmful_flip': int(control['correct'] != 1) if full['correct'] == 1 else None,
                            'critical_target_abstention': critical['abstains'], 'critical_unsupported': critical['unsupported']})
    write_csv(OUT / 'panel_c/target_transitions.csv', transitions)
    counts = Counter((r['condition'], r['aspect'], r['answerability'], state(r)) for r in parts)
    write_csv(OUT / 'panel_d/decision_matrix.csv', [{'condition': k[0], 'aspect': k[1], 'answerability': k[2], 'response_state': k[3], 'parts': n} for k, n in sorted(counts.items())])


def panel_e() -> None:
    config = Config(output=OUT / 'panel_e')
    tasks = read(config.output / 'tasks.json')
    parts, metrics = parts_and_metrics(config, tasks)
    parts = [r for r in parts if r['aspect'] in ASPECTS]
    metrics = [r for r in metrics if r['aspect'] in ASPECTS]
    write_csv(config.output / 'answer_parts.csv', parts)
    write_csv(config.output / 'paper_metrics.csv', metrics)
    write_csv(config.output / 'paired_answerable.csv', paired(parts))
    index = {(r['paper_id'], r['aspect']): r for r in metrics if r['condition'] == 'F'}
    contrasts = []
    for row in metrics:
        if row['condition'] == 'F':
            continue
        base = index[(row['paper_id'], row['aspect'])]
        contrasts.append({'paper_id': row['paper_id'], 'condition': row['condition'], 'aspect': row['aspect'],
                          'baseline_q_fixed': base['q_fixed'], 'condition_q_fixed': row['q_fixed'],
                          'delta_q_fixed': row['q_fixed'] - base['q_fixed'] if row['q_fixed'] is not None and base['q_fixed'] is not None else None})
    write_csv(config.output / 'paired_fixed.csv', contrasts)
    ledger = config.output / 'call_ledger.jsonl'
    write(config.output / 'completion.json', {'metrics': len(metrics), 'metric_states': dict(Counter(r['state'] for r in metrics)),
          'parts': len(parts), 'part_states': dict(Counter(r['technical_state'] for r in parts)),
          'calls': len(ledger.read_text().splitlines()) if ledger.exists() else 0, 'hard_cap': 9, 'new_reports': 0})
    selected = {t['paper_id'] for t in tasks}
    write(config.output / 'report_adoption.json', [r for r in read(SOURCE / 'reuse.json') if r['paper_id'] in selected and r['condition'] in ('F', 'K5', 'LUNA')])


def panel_f() -> None:
    records = [(p, read(p)) for root in (FIG4, SOURCE) for p in (root / 'logs/calls').glob('*/record.json')]
    stages, totals = [], []
    metrics = csv_rows(ROOT / 'fig5_initial/paper_metrics.csv')
    for task in read(ROOT / 'fig5_initial/tasks.json'):
        paper, condition = task['paper_id'], task['condition']
        root = FIG4 if condition == 'F' else SOURCE
        specs = [('analysis', 'gear' if condition == 'F' else condition + '_gear'),
                 ('analysis', 'graph_F' if condition == 'F' else condition + '_graph'), ('reports', condition)]
        selected = []
        for stage, method in specs:
            artifact = root / stage / method / f'{paper}.json'
            matches = []
            for path, record in records:
                if not path.is_relative_to(root) or (record.get('paper_id'), record.get('stage'), record.get('method'), record.get('state')) != (paper, stage, method, 'completed'):
                    continue
                response = path.with_name('response.json')
                if artifact.exists() and response.exists() and read(response) == read(artifact):
                    matches.append((path, record))
            row = {'paper_id': paper, 'condition': condition, 'stage': stage, 'method': method, 'artifact_path': str(artifact)}
            if matches:
                path, record = max(matches, key=lambda x: x[1]['time'])
                usage = record.get('usage') or {}
                row.update(state='matched_saved_artifact', record_path=str(path), seconds=record.get('seconds'), started_at=record['time'], model=record['model'], effort=record['effort'], **usage)
                selected.append(row)
                evidence_path = OUT / 'panel_f/source_records' / path.parent.name / 'record.json'
                write(evidence_path, record)
            else:
                row['state'] = 'cost_attribution_missing'
            stages.append(row)
        complete = len(selected) == 3 and all(r.get('input_tokens') is not None and r.get('output_tokens') is not None for r in selected)
        timing = SOURCE / 'timing' / condition / f'{paper}.json'
        wall, timing_status = None, 'unavailable'
        if condition != 'F' and timing.exists():
            t = read(timing)
            if t.get('state') == 'completed' and t.get('wall_seconds') is not None:
                end = datetime.fromisoformat(t['ended_at']).timestamp()
                started = end - t['wall_seconds']
                if len(selected) == 3 and all(datetime.fromisoformat(r['started_at']).timestamp() >= started - 5 for r in selected):
                    wall, timing_status = t['wall_seconds'], 'complete_generation_wall_including_queue'
                else:
                    timing_status = 'resumed_partial_wall_excluded'
        for metric in [r for r in metrics if r['paper_id'] == paper and r['condition'] == condition]:
            totals.append({'paper_id': paper, 'condition': condition, 'aspect': metric['aspect'], 'q_fixed': metric['q_fixed'],
                           'answer_coverage': metric['answer_coverage'], 'answered_risk': metric['answered_risk'],
                           'cost_complete': complete, 'matched_stages': len(selected),
                           'input_tokens': sum(r['input_tokens'] for r in selected) if complete else None,
                           'output_tokens': sum(r['output_tokens'] for r in selected) if complete else None,
                           'call_seconds_sum': sum(r['seconds'] for r in selected) if len(selected) == 3 and all(r.get('seconds') is not None for r in selected) else None,
                           'wall_seconds': wall, 'wall_status': timing_status, 'cost_scope': 'saved successful inference stages; excludes evaluation/retrieval/build and failed attempts'})
    write_csv(OUT / 'panel_f/stage_costs.csv', stages)
    write_csv(OUT / 'panel_f/cost_quality.csv', totals)


def main() -> None:
    for panel in 'abcdef':
        (OUT / f'panel_{panel}').mkdir(parents=True, exist_ok=True)
    panel_a()
    core_panels()
    panel_e()
    panel_f()
    print('Exported six-panel tables (inspect panel_e/completion.json for pending evaluations).')


if __name__ == '__main__':
    main()
