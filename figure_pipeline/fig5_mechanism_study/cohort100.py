from __future__ import annotations

import argparse
import asyncio
import fcntl
import random

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_revision import pipeline
from figure_pipeline.fig5_revision.materials import (
    identities,
    original_blocks,
    retain,
    save_packet,
)
from figure_pipeline.fig5_robustness import prompts

from . import interventions, transfer
from .diagnostic import OLD
from .diagnostic import OUT as DIAGNOSTIC

ROOT = DIAGNOSTIC.parent


def prepare() -> None:
    cohort = read(interventions.OLD / 'cohort.json')
    additions = cohort['development']
    write(ROOT / 'cohort_100_amendment.json', {
        'authorization': 'User requested treating the five previously separated papers and the other 95 as one 100-paper cohort.',
        'primary_cohort': cohort['paper_ids'], 'n': 100,
        'reporting': 'Unified 100-paper cohort, with each main comparison reporting actual availability out of 100.',
        'history': 'Previous exploratory uses retained in provenance; no claim that this is an untouched independent test set.',
        'extension': 'Fill the same full/half-source and original/hybrid-model conditions for all papers.',
        'subsets': 'Dose, repeated-mask, stage-diagnostic and targeted-deletion subsets remain identified by their actual sizes; no extrapolation to 100.',
        'no_new_prompt_selected': True})
    tasks = [t for t in read(interventions.OUT / 'tasks.json')
             if t['condition'] not in ('CRITICAL', 'NONCRITICAL', 'RESTORE')]
    for paper in additions:
        name = f'{paper}.json'
        full = read(interventions.OLD / 'inputs/F' / name)
        aliases = identities(full)
        write(interventions.OUT / 'identities' / name, aliases)
        for condition in ('F', 'E50'):
            works = sorted(set(aliases.values()))
            random.Random(f'20261004:{paper}:evidence-gradient').shuffle(works)
            data = full if condition == 'F' else retain(full, aliases, set(works[:len(works) // 2]))
            kept = sorted({aliases[p['source_id']] for b in original_blocks(data) for p in b['provenance']})
            save_packet(interventions.config(), paper, condition, data, retained=kept, unit='independent_original_work')
            task = {'paper_id': paper, 'condition': condition, 'view': condition,
                    'split': 'unified_100_paper_cohort', 'kind': 'multistage'}
            if not any(t['paper_id'] == paper and t['condition'] == condition for t in tasks):
                tasks.append(task)
        material = read(ROOT / 'adapter_development/inputs' / name)
        full_report = read(ROOT / 'adapter_development/reports/original_model_free_report' / name)
        expected = {**pipeline.visible(full), 'public_tasks': pipeline.public(interventions.config(), paper)}
        if not all(interventions.equivalent(material.get(k), value) for k, value in expected.items()):
            raise ValueError(f'Additional full report material differs: {paper}')
        interventions.copy_once(ROOT / 'adapter_development/inputs' / name, interventions.OUT / 'report_inputs/F' / name)
        write(interventions.OUT / 'reports/F' / name, full_report)
    write(interventions.OUT / 'tasks.json', tasks)
    protocol = read(interventions.OUT / 'protocol.json')
    interventions.copy_once(interventions.OUT / 'protocol.json', interventions.OUT / 'protocol_before_unified100.json')
    protocol.update(core_paired_papers=100, sample_scope='All 100 existing papers analyzed as one cohort; exploratory history retained.',
                    planned_static_reports=sum(t['condition'] not in ('CRITICAL', 'NONCRITICAL', 'RESTORE') for t in tasks))
    write(interventions.OUT / 'protocol.json', protocol)
    extend_models(cohort)


def extend_models(cohort: dict) -> None:
    additions = cohort['development'] + cohort['exploration']
    for paper in additions:
        name = f'{paper}.json'
        if paper in cohort['development']:
            material = read(ROOT / 'adapter_development/inputs' / name)
            report = read(ROOT / 'adapter_development/reports/original_model_free_report' / name)
        else:
            material = read(OLD / 'baseline/report_inputs/F' / name)
            report = read(DIAGNOSTIC / 'reports/original_analysis_original_writer' / name)
            interventions.copy_once(DIAGNOSTIC / 'reports/replacement_analysis_original_writer' / name,
                                    transfer.OUT / 'reports/replacement_analysis_original_writer' / name)
            interventions.copy_once(DIAGNOSTIC / 'report_inputs/replacement_analysis_original_writer' / name,
                                    transfer.OUT / 'report_inputs/replacement_analysis_original_writer' / name)
        write(transfer.OUT / 'original_inputs' / name, material)
        write(transfer.OUT / 'reports/original_analysis_original_writer' / name, report)
        if paper in cohort['development']:
            data = read(interventions.OLD / 'inputs/F' / name)
            common = {'manuscript': data['manuscript'], 'public_tasks': material['public_tasks']}
            gear = {**common, 'neutral_claims': [c['claim'] for c in data['graph']['cards']],
                    'original_historical_passages': data['gear_evidence']['evidence_blocks']}
            graph = {**common, 'native_graph': data['graph'], 'graph_metric_definitions': prompts.GRAPH_DEFINITIONS,
                     'original_historical_passages': data['graph_evidence']['evidence_blocks']}
            for branch, content in [('gear', gear), ('graph', graph)]:
                original = OLD / 'baseline/analysis_inputs' / ('gear' if branch == 'gear' else 'graph_F') / name
                if original.exists() and read(original) != content:
                    raise ValueError(f'Original branch material mismatch: {paper}/{branch}')
                write(transfer.OUT / 'analysis_inputs' / branch / name, content)
    protocol = read(transfer.OUT / 'protocol.json')
    interventions.copy_once(transfer.OUT / 'protocol.json', transfer.OUT / 'protocol_before_unified100.json')
    protocol.update(papers=cohort['paper_ids'], n=100,
                    call_cap=400, ordinary_limit=350, additional_limit=50,
                    selection='All 100 existing papers; no exclusion for previous exploratory use.',
                    sample_limit='Unified existing cohort, not an untouched independent test set.',
                    basis='User requested a unified 100-paper analysis; compare the original and analysis-stage replacement configurations on every paper.')
    write(transfer.OUT / 'protocol.json', protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'interventions', 'models'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
        return
    cohort = read(interventions.OLD / 'cohort.json')
    with (ROOT / f'extension100_{args.command}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'interventions':
            asyncio.run(interventions.run(cohort['development']))
        else:
            asyncio.run(transfer.run(cohort['development'] + cohort['exploration']))


if __name__ == '__main__':
    main()
