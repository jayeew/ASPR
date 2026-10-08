from __future__ import annotations

import argparse
import asyncio
import fcntl
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import Report
from figure_pipeline.fig4_explanation_study.evaluation import segments
from figure_pipeline.fig5_four_panel.aggregate import classify
from figure_pipeline.fig5_revision import pipeline, prompts
from figure_pipeline.fig5_revision.materials import (
    digest,
    equivalent,
    identities,
    original_blocks,
    retain,
    save_packet,
)
from figure_pipeline.fig5_revision.models import Config, Evaluation
from figure_pipeline.fig5_revision.reference_audit import audit_references
from figure_pipeline.fig5_revision.runtime import Runner, protected
from figure_pipeline.fig5_robustness.pipeline import align_evaluation

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'outputs/fig5_revision'
OUT = ROOT / 'outputs/fig5_mechanism_study/interventions'


def config() -> Config:
    return Config(output=OUT, cli_limit=32, ordinary_limit=1500, additional_limit=150, call_limit=1650)


def copy_once(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def source_input(paper: str, condition: str) -> Path:
    direct = OLD / 'report_inputs' / condition / f'{paper}.json'
    if direct.exists():
        return direct
    records = [r for r in read(OLD / 'reuse.json') if r['paper_id'] == paper and r['condition'] == condition]
    if records:
        report = Path(records[0]['source'])
        return report.parents[2] / 'report_inputs' / condition / report.name
    return direct


def adopt_branches(cfg: Config, paper: str, condition: str, data: dict[str, Any]) -> None:
    for branch in ('gear', 'graph'):
        key = f'{condition}_{branch}'
        content = {'manuscript': data['manuscript'], 'public_tasks': pipeline.public(cfg, paper)}
        if branch == 'gear':
            content.update(neutral_claims=[c['claim'] for c in data['graph']['cards']],
                           original_historical_passages=data['gear_evidence']['evidence_blocks'])
        else:
            content.update(native_graph=data['graph'], graph_metric_definitions=prompts.DEFINITIONS,
                           original_historical_passages=data['graph_evidence']['evidence_blocks'])
        input_path = OLD / 'analysis_inputs' / key / f'{paper}.json'
        if input_path.exists() and equivalent(read(input_path), content):
            for folder in ('analysis', 'analysis_inputs'):
                copy_once(OLD / folder / key / f'{paper}.json', OUT / folder / key / f'{paper}.json')


def prepare() -> None:
    cfg = config()
    cohort = read(OLD / 'cohort.json')
    papers = cohort['paper_ids']
    # Keep existing identities and allocation; the prior exploration is never relabeled held out.
    write(OUT / 'cohort.json', cohort)
    for folder in ('baseline', 'identities', 'support_results', 'supports', 'supports_v2', 'eligibility', 'order_mapping'):
        for source in (OLD / folder).rglob('*.json'):
            copy_once(source, OUT / source.relative_to(OLD))
    tasks, audit = [], []
    for paper in papers:
        full = read(OLD / 'inputs/F' / f'{paper}.json')
        aliases = identities(full)
        write(OUT / 'identities' / f'{paper}.json', aliases)
        selected = ['F', 'E50']
        if paper in cohort['exploration']:
            selected += ['E75', 'E25', 'G50', 'G25', 'E50_SEED2', 'E50_SEED3']
        if paper in cohort['small']:
            selected += ['ORDER', 'REPEAT2', 'REPEAT3']
        for condition in selected:
            view = 'F' if condition.startswith('REPEAT') else condition
            source = OLD / 'inputs' / view / f'{paper}.json'
            if source.exists():
                data = read(source)
            elif condition.startswith('E50'):
                works = sorted(set(aliases.values()))
                seed = f'20261004:{paper}:evidence-gradient'
                if condition != 'E50':
                    seed += ':' + condition
                random.Random(seed).shuffle(works)
                data = retain(full, aliases, set(works[:len(works) // 2]))
            else:
                raise ValueError(f'Missing established material: {paper}/{condition}')
            kept = sorted({aliases[p['source_id']] for b in original_blocks(data) for p in b['provenance']})
            save_packet(cfg, paper, view, data, retained=kept, unit='independent_original_work')
            reference = OLD / 'validated_reference' / view / f'{paper}.json'
            reference_input = OLD / 'reference_inputs' / view / f'{paper}.json'
            reference_match = (reference.exists() and reference_input.exists()
                               and read(reference_input).get('visible_packet') == pipeline.visible(data))
            if reference_match:
                pipeline.validate_reference(read(reference), pipeline.fixed(cfg, paper), data)
                copy_once(reference, OUT / 'validated_reference' / view / f'{paper}.json')
            adopt_branches(cfg, paper, condition, data)
            report = OLD / 'reports' / condition / f'{paper}.json'
            material = source_input(paper, condition)
            if condition == 'F' and paper in cohort['development']:
                report = OUT.parent / 'adapter_development/reports/original_model_free_report' / f'{paper}.json'
                material = OUT.parent / 'adapter_development/inputs' / f'{paper}.json'
            expected = {**pipeline.visible(data), 'public_tasks': pipeline.public(cfg, paper)}
            matches = material.exists() and all(equivalent(read(material).get(k), v) for k, v in expected.items())
            if report.exists() and matches:
                Report.model_validate(read(report))
                copy_once(report, OUT / 'reports' / condition / f'{paper}.json')
                copy_once(material, OUT / 'report_inputs' / condition / f'{paper}.json')
            tasks.append({'paper_id': paper, 'condition': condition, 'view': view,
                          'split': 'prior_exploration' if paper in cohort['exploration'] else 'expanded_intervention',
                          'kind': 'multistage'})
            audit.append({'paper_id': paper, 'condition': condition, 'packet_hash': digest(data),
                          'reference_reused': reference_match, 'report_reused': report.exists() and matches,
                          'original_report_path': str(report), 'original_input_path': str(material)})
    write(OUT / 'tasks.json', tasks)
    write(OUT / 'reuse_audit.json', audit)
    write_csv(OUT / 'reuse_audit.csv', audit)
    write(OUT / 'protocol.json', {
        'version': 1, 'core_paired_papers': len(papers), 'core_conditions': ['F', 'E50'],
        'gradient_papers': cohort['exploration'], 'critical_design_papers': cohort['exploration'] + cohort['confirmation'],
        'random_mask_repeats': 'Three prespecified half-source masks on all 20 exploration papers; average within paper.',
        'repetition_and_order_papers': cohort['small'], 'planned_static_reports': len(tasks),
        'critical_conditions': ['CRITICAL', 'NONCRITICAL', 'RESTORE'],
        'recovery': 'Fresh independent generation from restored original material; not conversational repair.',
        'critical_matching': 'Existing deletion controls match independent source counts; text-length imbalance reported.',
        'noninferiority_margin': None, 'inference': 'Descriptive paired paper-level intervals; no invented pass threshold.',
        'sample_scope': 'All 100 existing papers as one cohort; previous exploratory use is retained in provenance, not a separate main-analysis group.',
        'new_calls_cap': 1650, 'old_stop_markers': 'Untouched; no old runner or automation resumed.',
        'evaluation': 'All five aspects, fixed parts, model-assisted; no synthetic or desired outcomes.'})


async def critical_tasks(cfg: Config, runner: Runner, paper: str) -> list[dict[str, Any]]:
    # Existing support proposals are adopted only after a fresh material/identity validity check.
    for view in ('CRITICAL', 'NONCRITICAL'):
        for folder in ('inputs', 'masks', 'reference_inputs', 'validated_reference'):
            copy_once(OLD / folder / view / f'{paper}.json', OUT / folder / view / f'{paper}.json')
    additions = await pipeline.support_design(cfg, runner, paper)
    if not additions:
        return []
    additions.append({'paper_id': paper, 'condition': 'RESTORE', 'view': 'F',
                      'split': additions[0]['split'], 'kind': 'multistage'})
    for task in additions:
        condition = task['condition']
        data = pipeline.generation_packet(cfg, task)
        source_input = OLD / 'report_inputs' / condition / f'{paper}.json'
        expected = {**pipeline.visible(data), 'public_tasks': pipeline.public(cfg, paper)}
        if source_input.exists() and all(read(source_input).get(k) == v for k, v in expected.items()):
            copy_once(OLD / 'reports' / condition / f'{paper}.json', OUT / 'reports' / condition / f'{paper}.json')
            copy_once(source_input, OUT / 'report_inputs' / condition / f'{paper}.json')
        adopt_branches(cfg, paper, condition, data)
    return additions


async def evaluate(cfg: Config, runner: Runner, paper: str, task: dict[str, Any]) -> None:
    condition, view = task['condition'], task['view']
    data = pipeline.packet(cfg, paper, view)
    ref = read(OUT / 'validated_reference' / view / f'{paper}.json')
    material = {'visible_packet': pipeline.visible(data), 'independent_work_aliases': pipeline.visible_aliases(cfg, paper, data),
                'reference_questions': pipeline.fixed(cfg, paper)['questions'], 'public_tasks': pipeline.public(cfg, paper),
                'candidates': [{'report_id': 'R01', 'conditional_reference': ref['parts'],
                                'report_segments': segments(read(OUT / 'reports' / condition / f'{paper}.json')['body'])}]}
    write(OUT / 'evaluation_inputs' / condition / f'{paper}.json', material)
    result = await runner.call(paper, 'evaluation', condition, prompts.EVALUATE, material, Evaluation)
    value = align_evaluation(result, material)['candidates'][0]
    for question in value['questions']:
        for part in question['parts']:
            expected = {'substantive_answer': False, 'omission': False, 'target_abstention': True,
                        'unresolved': None}[part['response_type']]
            if part['explicitly_abstains'] is not expected:
                part['technical_state'] = 'invalid_response_taxonomy'
    write(OUT / 'aligned_all' / condition / f'{paper}.json', value)


async def run(selected_papers: list[str] | None = None) -> None:
    cfg, runner = config(), Runner(config())
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=cfg.cli_limit))
    cohort = read(OUT / 'cohort.json')
    tasks = read(OUT / 'tasks.json')
    design = cohort['exploration'] + cohort['confirmation']
    papers = selected_papers if selected_papers is not None else design + [p for p in cohort['paper_ids'] if p not in design]
    async def one(paper: str) -> None:
        restrictions = OUT / 'nonretryable_failures.json'
        if restrictions.exists() and paper in {r['paper_id'] for r in read(restrictions)}:
            runner.status(paper, 'study', 'all', 'unresolved', reason='Provider content restriction; no further calls for this paper. Retained in the 100-paper denominator.')
            return
        current = [t for t in tasks if t['paper_id'] == paper]
        for view in dict.fromkeys(t['view'] for t in current):
            await protected(runner, paper, 'reference', view, pipeline.reference(cfg, runner, paper, view))
        audit = await protected(runner, paper, 'reference_consistency', 'nested_originals',
                                audit_references(cfg, runner, paper))
        if audit is not True:
            return
        if paper in design:
            extra = await protected(runner, paper, 'support_design', 'critical', critical_tasks(cfg, runner, paper))
            if extra:
                current += extra
        write(OUT / 'paper_tasks' / f'{paper}.json', current)
        for task in current:
            condition = task['condition']
            if not (OUT / 'validated_reference' / task['view'] / f'{paper}.json').exists():
                continue
            await protected(runner, paper, 'generation', condition, pipeline.generate(cfg, runner, task))
            if (OUT / 'reports' / condition / f'{paper}.json').exists():
                await protected(runner, paper, 'grading', condition, evaluate(cfg, runner, paper, task))
    await asyncio.gather(*(one(p) for p in papers))
    aggregate()


def all_tasks() -> list[dict[str, Any]]:
    tasks = {(t['paper_id'], t['condition']): t for t in read(OUT / 'tasks.json')}
    for path in (OUT / 'paper_tasks').glob('*.json'):
        for task in read(path):
            tasks.setdefault((task['paper_id'], task['condition']), task)
    return list(tasks.values())


def aggregate() -> None:
    rows, metrics, inventory = [], [], []
    for task in all_tasks():
        paper, condition, view = (task[k] for k in ('paper_id', 'condition', 'view'))
        path = OUT / 'aligned_all' / condition / f'{paper}.json'
        reference_path = OUT / 'validated_reference' / view / path.name
        refs = {(r['question_id'], r['part_id']): r for r in read(reference_path)['parts']} if reference_path.exists() else {}
        fixed = pipeline.fixed(config(), paper)
        observed = {(q['question_id'], p['part_id']): p for q in read(path)['questions'] for p in q['parts']} if path.exists() else {}
        current = []
        for question in fixed['questions']:
            if not question['applicable']:
                continue
            for expected in question['answer_parts']:
                key = question['question_id'], expected['part_id']
                part = observed.get(key, {'technical_state': 'missing'})
                answerability = refs.get(key, {}).get('answerability', 'unresolved')
                current.append({**part, **classify(part, answerability), 'paper_id': paper,
                                'condition': condition, 'aspect': question['aspect'],
                                'question_id': key[0], 'part_id': key[1], 'answerability': answerability})
        rows.extend(current)
        inventory.append({**task, 'report_available': (OUT / 'reports' / condition / path.name).exists(),
                          'reference_available': reference_path.exists(), 'evaluation_available': path.exists(),
                          'complete': bool(current) and all(r['technical_state'] == 'completed' for r in current)})
        for aspect in ['all', *sorted({q['aspect'] for q in fixed['questions']})]:
            selected = [r for r in current if aspect == 'all' or r['aspect'] == aspect]
            complete = bool(selected) and all(r['technical_state'] == 'completed' for r in selected)
            item = {'paper_id': paper, 'condition': condition, 'aspect': aspect, 'complete': complete,
                    'fixed_denominator': len(selected),
                    'technical_missing': sum(r['technical_state'] != 'completed' for r in selected)}
            for name in ('grounded_answer', 'reasonable_abstention', 'confirmed_unsupported', 'scientific_unresolved'):
                item[name] = sum(r[name] == 1 for r in selected) / len(selected) if complete else None
            metrics.append(item)
    for name, data in [('answer_parts', rows), ('paper_metrics', metrics), ('task_inventory', inventory)]:
        write(OUT / f'{name}.json', data)
        write_csv(OUT / f'{name}.csv', data)
    print(f'Interventions: {sum(r["complete"] for r in inventory)}/{len(inventory)} complete evaluated reports', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'aggregate'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            prepare()
        elif args.command == 'run':
            asyncio.run(run())
        else:
            aggregate()


if __name__ == '__main__':
    main()
