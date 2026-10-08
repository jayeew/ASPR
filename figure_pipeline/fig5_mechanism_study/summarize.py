from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_four_panel.aggregate import classify, interval

from .diagnostic import CONDITIONS
from .diagnostic import OUT as DIAGNOSTIC
from .interventions import OUT as INTERVENTIONS

OUT = DIAGNOSTIC.parent
METRICS = ('grounded_answer', 'reasonable_abstention', 'confirmed_unsupported')


def adapter_summary() -> None:
    folder = OUT / 'adapter_development'
    rows = []
    for path in (folder / 'aligned').glob('*.json'):
        mapping = read(folder / 'mapping' / path.name)
        for candidate in read(path)['candidates']:
            parts = [p for q in candidate['questions'] for p in q['parts']]
            values = [classify(p, 'answerable') for p in parts]
            complete = bool(parts) and all(p['technical_state'] == 'completed' for p in parts)
            rows.append({'paper_id': path.stem, 'condition': mapping[candidate['report_id']],
                         'aspect': 'all', 'fixed_denominator': len(parts), 'complete': complete,
                         **{m: sum(v[m] == 1 for v in values) / len(parts) if complete else None for m in METRICS}})
    write(folder / 'paper_metrics.json', rows)
    write_csv(folder / 'paper_metrics.csv', rows)
    contrasts(folder, 'original_model_free_report')
    paired = []
    index = {(r['paper_id'], r['condition']): r for r in rows}
    for paper in read(folder / 'protocol.json')['papers']:
        for model in ('original', 'replacement'):
            free = index.get((paper, model + '_model_free_report'))
            structured = index.get((paper, model + '_model_structured_report'))
            if not free or not structured or not free['complete'] or not structured['complete']:
                continue
            paired.extend({'paper_id': paper, 'model': model, 'metric': m,
                           'delta_percentage_points': 100 * (structured[m] - free[m])} for m in METRICS)
    write(folder / 'structured_paired.json', paired)
    write_csv(folder / 'structured_paired.csv', paired)
    summary = [{'model': model, 'metric': metric,
                **interval([r['delta_percentage_points'] for r in paired if r['model'] == model and r['metric'] == metric])}
               for model in ('original', 'replacement') for metric in METRICS]
    write(folder / 'structured_summary.json', summary)
    write_csv(folder / 'structured_summary.csv', summary)


def contrasts(folder: Path, baseline: str) -> list[dict[str, Any]]:
    if not (folder / 'paper_metrics.json').exists():
        return []
    rows = read(folder / 'paper_metrics.json')
    index = {(r['paper_id'], r['condition'], r['aspect']): r for r in rows}
    result = []
    for row in rows:
        base = index.get((row['paper_id'], baseline, row['aspect']))
        if not base or row['condition'] == baseline or not row['complete'] or not base['complete']:
            continue
        for metric in METRICS:
            result.append({'paper_id': row['paper_id'], 'condition': row['condition'],
                           'baseline': baseline, 'aspect': row['aspect'], 'metric': metric,
                           'delta_percentage_points': 100 * (row[metric] - base[metric])})
    write(folder / 'paired_changes.json', result)
    write_csv(folder / 'paired_changes.csv', result)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in result:
        grouped[row['condition'], row['aspect'], row['metric']].append(row['delta_percentage_points'])
    summary = [{'condition': c, 'aspect': a, 'metric': m, **interval(v)} for (c, a, m), v in grouped.items()]
    write(folder / 'paired_summary.json', summary)
    write_csv(folder / 'paired_summary.csv', summary)
    return summary


def factorial() -> None:
    path = DIAGNOSTIC / 'paper_metrics.json'
    if not path.exists():
        return
    rows = read(path)
    index = {(r['paper_id'], r['condition'], r['aspect']): r for r in rows if r['complete']}
    result = []
    for paper, aspect in sorted({(r['paper_id'], r['aspect']) for r in rows}):
        if not all((paper, c, aspect) in index for c in CONDITIONS):
            continue
        for metric in METRICS:
            a, b, c, d = (index[paper, condition, aspect][metric] for condition in CONDITIONS)
            values = {'writer_replacement_with_original_analysis': b - a,
                      'analysis_replacement_with_original_writer': c - a,
                      'writer_replacement_with_replacement_analysis': d - c,
                      'analysis_replacement_with_replacement_writer': d - b,
                      'analysis_writer_interaction': d - c - b + a}
            result.extend({'paper_id': paper, 'aspect': aspect, 'metric': metric,
                           'contrast': name, 'delta_percentage_points': 100 * value}
                          for name, value in values.items())
    write(DIAGNOSTIC / 'factorial_paired.json', result)
    write_csv(DIAGNOSTIC / 'factorial_paired.csv', result)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in result:
        grouped[row['contrast'], row['aspect'], row['metric']].append(row['delta_percentage_points'])
    summary = [{'contrast': c, 'aspect': a, 'metric': m, **interval(v)} for (c, a, m), v in grouped.items()]
    write(DIAGNOSTIC / 'factorial_summary.json', summary)
    write_csv(DIAGNOSTIC / 'factorial_summary.csv', summary)


def critical_transitions() -> None:
    path = INTERVENTIONS / 'answer_parts.json'
    if not path.exists():
        return
    index = {(r['paper_id'], r['condition'], r['question_id'], r['part_id']): r for r in read(path)}
    rows = []
    for decision_path in (INTERVENTIONS / 'eligibility').glob('*.json'):
        decision = read(decision_path)
        if not decision.get('eligible'):
            continue
        paper, target = decision_path.stem, decision['target']
        for condition in ('F', 'NONCRITICAL', 'CRITICAL', 'RESTORE'):
            key = (paper, condition, target['question_id'], target['part_id'])
            part = index.get(key)
            rows.append({'paper_id': paper, 'condition': condition, 'question_id': key[2], 'part_id': key[3],
                         'behavior': part['behavior'] if part else 'technical_missing',
                         'grounded_answer': part.get('grounded_answer') if part else None,
                         'reasonable_abstention': part.get('reasonable_abstention') if part else None})
    write(INTERVENTIONS / 'critical_transitions.json', rows)
    write_csv(INTERVENTIONS / 'critical_transitions.csv', rows)


def independent_errors() -> None:
    rows = []
    for path in (DIAGNOSTIC / 'independent_aligned').glob('*.json'):
        data = read(path)
        for candidate in data['candidates']:
            for question in candidate['questions']:
                for part in question['parts']:
                    rows.append({**part, 'paper_id': path.stem, 'condition': data['mapping'][candidate['report_id']],
                                 'question_id': question['question_id']})
    write(DIAGNOSTIC / 'independent_error_parts.json', rows)
    write_csv(DIAGNOSTIC / 'independent_error_parts.csv', rows)


def status() -> dict[str, Any]:
    report = {}
    for name, folder in [('diagnostic', DIAGNOSTIC), ('interventions', INTERVENTIONS),
                         ('independent_review', DIAGNOSTIC / 'independent_calls'),
                         ('adapter_development', OUT / 'adapter_development'),
                         ('transfer_validation', OUT / 'transfer_validation')]:
        records = [read(p) for p in (folder / 'logs/calls').glob('*/record.json')]
        evaluation_folder = folder / ('aligned_all' if name == 'interventions' else 'aligned')
        if name == 'independent_review':
            evaluation_folder = DIAGNOSTIC / 'independent_aligned'
        report[name] = {'call_states': dict(Counter(r['state'] for r in records)),
                        'available_reports': len(list((folder / 'reports').glob('*/*.json'))),
                        'evaluated_outputs': len(list(evaluation_folder.rglob('*.json'))),
                        'evaluation_unit': 'report' if name == 'interventions' else 'paper with paired candidates'}
    write(OUT / 'execution_status.json', report)
    return report


def main() -> None:
    contrasts(DIAGNOSTIC, 'original_analysis_original_writer')
    contrasts(INTERVENTIONS, 'F')
    contrasts(OUT / 'transfer_validation', 'original_analysis_original_writer')
    adapter_summary()
    factorial()
    critical_transitions()
    independent_errors()
    print(status(), flush=True)


if __name__ == '__main__':
    main()
