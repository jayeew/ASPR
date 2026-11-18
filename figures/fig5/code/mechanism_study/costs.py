from __future__ import annotations

from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_robustness.aggregate import usage_summary

from .diagnostic import CONDITIONS, OLD
from .diagnostic import OUT as DIAGNOSTIC


def records(folder: Path) -> list[dict[str, Any]]:
    return [{**read(path), 'record_path': str(path)} for path in folder.glob('*/record.json')]


def diagnostic_costs() -> None:
    sources = {'baseline': records(OLD / 'baseline/calls'), 'replacement': records(OLD / 'logs/calls'),
               'new': records(DIAGNOSTIC / 'logs/calls')}
    rows, allocation = [], []
    for paper in read(DIAGNOSTIC / 'protocol.json')['papers']:
        for condition, (analysis, writer) in CONDITIONS.items():
            groups = [('baseline' if analysis == 'original' else 'replacement', 'analysis',
                       branch if analysis == 'original' else 'LUNA_' + ('gear' if branch == 'gear' else 'graph'))
                      for branch in ('gear', 'graph_F')]
            groups += [('baseline' if writer == 'original' else 'replacement', 'reports',
                        'F' if writer == 'original' else 'LUNA')] if analysis == writer else [('new', 'reports', condition)]
            chosen, missing = [], []
            for source, stage, method in groups:
                current = [r for r in sources[source] if r['paper_id'] == paper and r['stage'] == stage
                           and r['method'].removesuffix('__capacity_retry') == method]
                if not any(r['state'] == 'completed' for r in current):
                    missing.append(f'{source}/{stage}/{method}')
                chosen.extend(current)
                allocation.extend({'paper_id': paper, 'condition': condition, 'source_group': source,
                                   'record_path': r['record_path'], 'reused': source != 'new',
                                   'stage': stage, 'method': method} for r in current)
            usage = usage_summary(chosen)
            complete = not missing and usage['usage_complete']
            rows.append({'paper_id': paper, 'condition': condition, 'complete_cost': complete,
                         'missing_tasks': missing, **usage,
                         'time_estimated_calls': sum(bool(r.get('seconds_estimated_from_event_file_timestamp')) for r in chosen),
                         'total_input_and_output_tokens': usage['input_tokens'] + usage['output_tokens'] if complete else None,
                         'cost_scope': 'Branch analyses plus report synthesis; exclude reference/evaluation/retrieval/build.',
                         'wall_latency_claimed': False, 'monetary_cost_available': False})
    for name, values in [('generation_costs', rows), ('cost_allocation', allocation)]:
        write(DIAGNOSTIC / f'{name}.json', values)
        write_csv(DIAGNOSTIC / f'{name}.csv', values)
    print(f'Diagnostic cost coverage: {sum(r["complete_cost"] for r in rows)}/{len(rows)} configurations.')


def transfer_costs() -> None:
    folder = DIAGNOSTIC.parent / 'transfer_validation'
    baseline = {Path(r['record_path']).parent.name: r for r in records(OLD / 'baseline/calls')}
    for record in records(OLD.parent / 'fig4_reference/experiment/logs/calls'):
        baseline.setdefault(Path(record['record_path']).parent.name, record)
    sources = {'baseline': list(baseline.values()), 'new': records(folder / 'logs/calls'),
               'old_replacement': records(OLD / 'logs/calls'), 'diagnostic': records(DIAGNOSTIC / 'logs/calls'),
               'prerequisite': records(DIAGNOSTIC.parent / 'adapter_development/prerequisite_calls/logs/calls')}
    diagnostic_papers = read(DIAGNOSTIC / 'protocol.json')['papers']
    rows, allocation = [], []
    for paper in read(folder / 'protocol.json')['papers']:
        for condition in read(folder / 'protocol.json')['conditions']:
            original = condition == 'original_analysis_original_writer'
            if original:
                groups = [('baseline', 'analysis', 'gear'), ('baseline', 'analysis', 'graph_F'),
                          ('baseline', 'reports', 'F')]
            elif paper in diagnostic_papers:
                groups = [('old_replacement', 'analysis', 'LUNA_gear'), ('old_replacement', 'analysis', 'LUNA_graph'),
                          ('diagnostic', 'reports', condition)]
            else:
                groups = [('new', 'analysis', 'gear'), ('new', 'analysis', 'graph'), ('new', 'reports', condition)]
            chosen, missing = [], []
            for source, stage, method in groups:
                current = [r for r in sources[source] if r['paper_id'] == paper and r['stage'] == stage
                           and r['method'].removesuffix('__capacity_retry') == method]
                if original and not any(r['state'] == 'completed' for r in current):
                    fallback = {'graph_F': 'graph', 'F': 'original_model_free_report'}.get(method)
                    if fallback:
                        source = 'prerequisite'
                        current += [r for r in sources[source] if r['paper_id'] == paper
                                    and r['stage'] == stage and r['method'].removesuffix('__capacity_retry') == fallback]
                if not any(r['state'] == 'completed' for r in current):
                    missing.append(f'{source}/{stage}/{method}')
                chosen.extend(current)
                allocation.extend({'paper_id': paper, 'condition': condition, 'source_group': source,
                                   'record_path': r['record_path'], 'reused': source != 'new',
                                   'stage': stage, 'method': method} for r in current)
            usage = usage_summary(chosen)
            complete = not missing and usage['usage_complete']
            rows.append({'paper_id': paper, 'condition': condition, 'complete_cost': complete,
                         'missing_tasks': missing, **usage,
                         'time_estimated_calls': sum(bool(r.get('seconds_estimated_from_event_file_timestamp')) for r in chosen),
                         'total_input_and_output_tokens': usage['input_tokens'] + usage['output_tokens'] if complete else None,
                         'cost_scope': 'Branch analyses plus report synthesis; exclude reference/evaluation/retrieval/build.',
                         'wall_latency_claimed': False, 'monetary_cost_available': False})
    for name, values in [('generation_costs', rows), ('cost_allocation', allocation)]:
        write(folder / f'{name}.json', values)
        write_csv(folder / f'{name}.csv', values)
    print(f'Transfer cost coverage: {sum(r["complete_cost"] for r in rows)}/{len(rows)} configurations.')


if __name__ == '__main__':
    diagnostic_costs()
    transfer_costs()
