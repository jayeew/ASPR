from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from figure_pipeline.fig3_revision.aggregate import bootstrap, write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_study.aggregate import COMPARISONS, GROUNDED, paper_rows

from .models import ASPECTS, CONDITIONS, NAMES, Config
from .runtime import error_category


def parts_for(config: Config, paper: str, reference: dict[str, Any]) -> list[dict[str, Any]]:
    mapping_path = config.output / 'evaluation_mapping' / f'{paper}.json'
    mapping = read(mapping_path) if mapping_path.exists() else {}
    rows = []
    for aspect in ASPECTS:
        path = config.output / 'aligned_evaluation' / aspect / f'{paper}.json'
        observed = {}
        for candidate in read(path)['candidates'] if path.exists() else []:
            for question in candidate['questions']:
                for part in question['parts']:
                    observed[(mapping[candidate['report_id']], question['question_id'], part['part_id'])] = part
        for question in reference['questions']:
            if question['aspect'] != aspect or not question['applicable']:
                continue
            for expected in question['answer_parts']:
                for condition in CONDITIONS:
                    value = observed.get((condition, question['question_id'], expected['part_id']), {})
                    row = dict(report_set='new', paper_id=paper, condition=condition, configuration=NAMES[condition],
                               aspect=aspect, aspect_name=ASPECTS[aspect], question_id=question['question_id'],
                               part_id=expected['part_id'], question=question['question'],
                               expected_content=expected['expected_content'], reference_evidence=expected['evidence'],
                               status=value.get('status'), scope_correct=value.get('scope_correct'),
                               grounding=value.get('grounding'), technical_state=value.get('technical_state', 'missing_judgment'),
                               report_quotes=value.get('report_quotes', []), reason=value.get('reason', ''),
                               evidence_source_ids=value.get('evidence_source_ids', []))
                    row['correct'] = int(row['status'] == 'correct' and row['scope_correct'] is True) if row['technical_state'] == 'completed' else None
                    row['grounded_correct'] = int(row['correct'] and row['grounding'] in GROUNDED) if row['correct'] is not None else None
                    rows.append(row)
    return rows


def absolute(rows: list[dict[str, Any]], config: Config, excluded: set[str]) -> list[dict[str, Any]]:
    result = []
    for aspect in ASPECTS:
        current = [r for r in rows if r['aspect'] == aspect and r['paper_id'] not in excluded]
        complete = defaultdict(set)
        for row in current:
            if row['state'] == 'completed':
                complete[row['paper_id']].add(row['condition'])
        common = {p for p, cs in complete.items() if cs == set(CONDITIONS)}
        for condition in CONDITIONS:
            values = [r for r in current if r['paper_id'] in common and r['condition'] == condition]
            for metric in ('content_coverage', 'grounded_coverage'):
                result.append(dict(condition=condition, configuration=NAMES[condition], aspect=aspect,
                                   aspect_name=ASPECTS[aspect], metric=metric, paper_ids=sorted(common),
                                   **bootstrap([r[metric] for r in values], config)))
    return result


def effects(rows: list[dict[str, Any]], parts: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    index = {(r['paper_id'], r['condition'], r['aspect']): r for r in rows}
    part_index = {(r['paper_id'], r['condition'], r['part_id']): r for r in parts}
    results, transitions = [], []
    for full in [r for r in rows if r['condition'] == 'F']:
        for name, (other, focus) in COMPARISONS.items():
            if focus and full['aspect'] != focus:
                continue
            before = index[(full['paper_id'], other, full['aspect'])]
            valid = full['state'] == before['state'] == 'completed'
            results.append(dict(paper_id=full['paper_id'], contrast=name, other=other, aspect=full['aspect'],
                                complete_pair=valid, full_coverage=full['content_coverage'],
                                other_coverage=before['content_coverage'],
                                content_delta=full['content_coverage'] - before['content_coverage'] if valid else None,
                                grounded_delta=full['grounded_coverage'] - before['grounded_coverage'] if valid else None))
            for part in [p for p in parts if (p['paper_id'], p['condition'], p['aspect']) == (full['paper_id'], 'F', full['aspect'])]:
                old = part_index[(part['paper_id'], other, part['part_id'])]
                pair = (old['correct'], part['correct'])
                state = 'technical_missing' if None in pair else {
                    (1, 1): 'retained', (0, 1): 'gained', (1, 0): 'lost', (0, 0): 'neither_correct'}[pair]
                transitions.append(dict(paper_id=part['paper_id'], contrast=name, other=other, aspect=part['aspect'],
                                        part_id=part['part_id'], transition=state, question=part['question'],
                                        full=part, comparison=old))
    return results, transitions


def effect_summary(rows: list[dict[str, Any]], config: Config, excluded: set[str]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        if row['paper_id'] not in excluded:
            groups[(row['contrast'], row['other'], row['aspect'])].append(row)
    result = []
    for (contrast, other, aspect), all_rows in groups.items():
        paired = [r for r in all_rows if r['complete_pair']]
        for metric in ('content_delta', 'grounded_delta'):
            values = [r[metric] for r in paired]
            result.append(dict(contrast=contrast, other=other, aspect=aspect, metric=metric,
                               improved=sum(v > 0 for v in values), tied=sum(v == 0 for v in values),
                               reduced=sum(v < 0 for v in values), paper_ids=[r['paper_id'] for r in paired],
                               **bootstrap(values, config)))
    return result


def aggregate(config: Config) -> None:
    cohort = read(config.output / 'cohort.json')
    references = {p.stem: read(p) for p in (config.output / 'reference/papers').glob('*.json')
                  if (config.output / 'screening' / p.name).exists()}
    included = {p: r for p, r in references.items() if r['graph_useful']}
    parts = [row for p, r in included.items() for row in parts_for(config, p, r)]
    rows = [r for r in paper_rows(parts, included) if r['report_set'] == 'new']
    paired, transitions = effects(rows, parts)
    write(config.output / 'answer_parts.json', parts)
    write_csv(config.output / 'answer_parts.csv', parts)
    write_csv(config.output / 'paper_aspect_metrics.csv', rows)
    write(config.output / 'paper_aspect_metrics.json', rows)
    write_csv(config.output / 'paired_effects.csv', paired)
    write(config.output / 'paired_effects.json', paired)
    write(config.output / 'paired_changes.json', transitions)
    for suffix, excluded in [('', set()), ('_without_development', set(cohort['pilot_ids']))]:
        summaries = absolute(rows, config, excluded)
        contrasts = effect_summary(paired, config, excluded)
        write(config.output / f'configuration_summary{suffix}.json', summaries)
        write_csv(config.output / f'configuration_summary{suffix}.csv', summaries)
        write(config.output / f'paired_effects_summary{suffix}.json', contrasts)
        write_csv(config.output / f'paired_effects_summary{suffix}.csv', contrasts)
    screening = [dict(paper_id=p, state='not_screened' if p not in references else 'included' if p in included else 'scientifically_excluded',
                      reason=references.get(p, {}).get('eligibility_reason', ''), development=p in cohort['pilot_ids']) for p in cohort['paper_ids']]
    write_csv(config.output / 'graph_screening.csv', screening)
    write_csv(config.output / 'aspect_applicability.csv', [dict(paper_id=p, question_id=q['question_id'], aspect=q['aspect'],
              applicable=q['applicable'], reason=q['applicability_reason'], denominator=len(q['answer_parts']))
              for p, r in references.items() for q in r['questions']])
    records = [read(p) for p in (config.output / 'logs/calls').glob('*/record.json')]
    statuses = [read(p) for p in (config.output / 'task_status').rglob('*.json')]
    for status in statuses:
        matched = [r for r in records if (r['paper_id'], r['stage'], r['method']) ==
                   (status['paper_id'], status['stage'], status['condition']) and r['state'] == 'failed']
        if status['state'] == 'unresolved' and matched:
            status['category'] = error_category(str(matched[-1].get('cli_error', '')) + status.get('reason', ''))
            status['provider_error'] = matched[-1].get('cli_error', '')
    lengths = [dict(paper_id=p.stem, condition=p.parent.name, configuration=NAMES[p.parent.name], characters=len(read(p)['body']))
               for p in (config.output / 'reports').glob('*/*.json')]
    write_csv(config.output / 'report_lengths.csv', lengths)
    usage = Counter()
    for record in records:
        usage.update({k: v for k, v in (record.get('usage') or {}).items() if isinstance(v, (int, float))})
    ledger = [json.loads(s) for s in (config.output / 'call_ledger.jsonl').read_text().splitlines()]
    summary = dict(candidate_papers=len(cohort['paper_ids']), screened_papers=len(references), included_papers=list(included),
                   scientifically_excluded=[p for p in references if p not in included], reports=len(lengths),
                   calls_started=len(ledger), call_states=dict(Counter(r['state'] for r in records)),
                   calls_by_stage=dict(Counter(r['stage'] for r in records)), known_usage=dict(usage),
                   unknown_usage_calls=sum(r.get('usage') is None for r in records),
                   additional_calls=sum(r['additional'] for r in ledger),
                   unresolved_tasks=[r for r in statuses if r['state'] in ('blocked', 'unresolved')],
                   remaining_95_started=any(r['paper_id'] not in cohort['pilot_ids'] for r in ledger),
                   evaluation_label='model_evaluation', regression_tests_run=False)
    write(config.output / 'run_summary.json', summary)
    print(f'Aggregated {len(parts)} scientific-part slots; {len(lengths)} reports; {len(ledger)} calls.', flush=True)
