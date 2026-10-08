from __future__ import annotations

import copy
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.materials import merged_blocks

from .models import Config

GROUPS = {
    'life': {'Immunology and Microbiology', 'Biochemistry, Genetics and Molecular Biology',
             'Neuroscience', 'Agricultural and Biological Sciences'},
    'physical_engineering': {'Physics and Astronomy', 'Chemistry', 'Materials Science',
                            'Energy', 'Engineering', 'Chemical Engineering', 'Computer Science'},
    'medicine': {'Medicine'},
    'earth_environment': {'Earth and Planetary Sciences', 'Environmental Science'},
}
QUOTAS = {'life': (4, 4), 'physical_engineering': (4, 3), 'medicine': (1, 2), 'earth_environment': (1, 1)}
REPEAT_QUOTAS = {'life': (1, 1), 'physical_engineering': (1, 0), 'medicine': (0, 1), 'earth_environment': (1, 0)}


def group(field: str) -> str:
    return next(name for name, fields in GROUPS.items() if field in fields)


def source_key(provenance: dict[str, Any]) -> str:
    doi = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', str(provenance.get('doi') or '').strip(), flags=re.IGNORECASE).lower()
    if doi:
        return 'doi:' + doi
    for field in ('work_id', 'parent_openalex_work_id', 'parent_paper_id'):
        if provenance.get(field):
            return 'work:' + str(provenance[field]).rstrip('/').rsplit('/', 1)[-1].lower()
    title = re.sub(r'\W+', '', str(provenance.get('title') or '').casefold())
    if not title:
        raise ValueError('Historical source has no recoverable identity')
    return 'title:' + title


def source_catalog(data: dict[str, Any]) -> dict[str, str]:
    """Resolve aliases using local provenance, including title matches to a known DOI."""
    items = [p for key in ('gear_evidence', 'graph_evidence')
             for b in data[key]['evidence_blocks'] for p in b['provenance']]
    titles: dict[str, set[str]] = defaultdict(set)
    for p in items:
        title = re.sub(r'\W+', '', str(p.get('title') or '').casefold())
        if p.get('doi'):
            titles[title].add(source_key(p))
    result = {}
    for p in items:
        key = source_key(p)
        title = re.sub(r'\W+', '', str(p.get('title') or '').casefold())
        if not p.get('doi') and len(titles[title]) == 1:
            key = next(iter(titles[title]))
        if p['source_id'] in result and result[p['source_id']] != key:
            raise ValueError(f"Conflicting local identity: {p['source_id']}")
        result[p['source_id']] = key
    return result


def reduce_sources(data: dict[str, Any], paper: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    aliases = source_catalog(data)
    works = sorted(set(aliases.values()))
    kept = set(random.Random(f'{seed}:{paper}:E50').sample(works, len(works) // 2))
    result = copy.deepcopy(data)
    for key in ('gear_evidence', 'graph_evidence'):
        blocks = []
        for block in result[key]['evidence_blocks']:
            block['provenance'] = [p for p in block['provenance'] if aliases[p['source_id']] in kept]
            if block['provenance']:
                blocks.append(block)
        result[key] = {'evidence_blocks': blocks}
    return result, {'paper_id': paper, 'source_aliases': aliases, 'retained': sorted(kept),
                        'removed': sorted(set(works) - kept), 'total': len(works),
                        'retained_fraction': len(kept) / len(works) if works else None,
                        'identity_policy': 'normalized DOI; local title/work identity when DOI unavailable'}


def reduced_graph(graph: dict[str, Any], runtime: Any, limit: int = 5) -> dict[str, Any]:
    from gear.innovation.joint_graph import joint_structure
    from gear.review_contracts import GraphFactCard
    cards = []
    runtime._connections()
    for saved in graph['cards']:
        card = GraphFactCard.model_validate(saved)
        neighbors = sorted(card.neighbors, key=lambda n: n.semantic_rank)[:limit]
        identities = {n.claim_id for n in neighbors}
        card = card.model_copy(update={
            'neighbors': neighbors, 'metrics': runtime._metrics(neighbors, card.claim.claim_type),
            'neighbor_edges': [e for e in card.neighbor_edges if set(e) <= identities],
            'community_ids': sorted({n.community_id for n in neighbors if n.community_id is not None}),
            'insertion_policy': f'threshold_parent_path:k={limit}:cosine>0.5',
        })
        cards.append(card)
    union = {n.claim_id for c in cards for n in c.neighbors}
    edges = [e for e in graph['joint']['historical_edges'] if set(e) <= union]
    return {'cards': [c.model_dump(mode='json') for c in cards], 'joint': joint_structure(cards, edges)}


def coverage(paper: dict[str, Any], data: dict[str, Any], development: bool) -> dict[str, Any]:
    cards = data['graph']['cards']
    counts = [len(c['neighbors']) for c in cards]
    nearest = [max(n['cosine_similarity'] for n in c['neighbors']) for c in cards if c['neighbors']]
    aliases = source_catalog(data)
    fulltext = {aliases[p['source_id']] for k in ('gear_evidence', 'graph_evidence')
                for b in data[k]['evidence_blocks'] for p in b['provenance'] if p['source_type'] == 'fulltext'}
    return {'paper_id': paper['paper_id'], 'field_name': paper['field_name'], 'group': group(paper['field_name']),
                'development': development, 'claims': len(cards), 'sparse': any(n < 10 for n in counts),
                'sparse_claim_fraction': sum(n < 10 for n in counts) / len(cards) if cards else None,
                'zero_neighbor_claims': counts.count(0), 'mean_neighbor_count': sum(counts) / len(cards) if cards else None,
                'mean_nearest_similarity': sum(nearest) / len(nearest) if nearest else None,
                'readable_sources': len(set(aliases.values())),
                'fulltext_fraction': len(fulltext) / len(set(aliases.values())) if aliases else None}


def select(rows: list[dict[str, Any]], quotas: dict[str, tuple[int, int]], seed: str) -> list[str]:
    rng = random.Random(seed)
    ids = []
    for name, numbers in quotas.items():
        for sparse, number in zip((True, False), numbers):
            candidates = sorted(r['paper_id'] for r in rows if r['group'] == name and r['sparse'] == sparse and not r['development'])
            ids.extend(rng.sample(candidates, number))
    return sorted(ids)


def copy_once(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise ValueError(f'Adopted input changed; explicit review required: {source}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return True


def prepare(config: Config) -> None:
    from gear.claim_attribution import ClaimGraphRuntime
    roster = read(config.source / 'papers.json')
    development = set(read(config.source / 'cohort.json')['pilot_ids'])
    rows, inputs = [], {}
    for paper in roster:
        ident = paper['paper_id']
        path = config.output / 'inputs/F' / f'{ident}.json'
        copy_once(config.source / 'inputs/papers' / f'{ident}.json', path)
        inputs[ident] = read(path)
        for card in inputs[ident]['graph']['cards']:
            if card['claim']['paper_id'] != ident:
                raise ValueError('Shared claim paper identity mismatch')
            if any(n['cosine_similarity'] <= .5 or n['parent_paper_id'] == ident
                   or n['publication_date'] >= paper['publication_date'] for n in card['neighbors']):
                raise ValueError('Saved neighborhood violates eligibility rules')
        rows.append(coverage(paper, inputs[ident], ident in development))
    claim_ids = [c['claim']['claim_id'] for data in inputs.values() for c in data['graph']['cards']]
    if len(rows) != 100 or len(claim_ids) != 717 or len(set(claim_ids)) != 717:
        raise ValueError('Expected accepted 100-paper, 717-claim cohort')
    selected = select(rows, QUOTAS, str(config.seed))
    repeated = select([r for r in rows if r['paper_id'] in selected], REPEAT_QUOTAS, f'{config.seed}:repeat')
    cohort = {'paper_ids': [r['paper_id'] for r in rows], 'selected': selected, 'repeated': repeated,
                  'development': sorted(development), 'seed': config.seed}
    previous = config.output / 'cohort.json'
    if previous.exists() and read(previous) != cohort:
        raise ValueError('Refusing to change an established cohort')
    write(previous, cohort)
    write(config.output / 'papers.json', roster)
    write_csv(config.output / 'paper_coverage.csv', rows)
    write(config.output / 'paper_coverage.json', rows)
    runtime = ClaimGraphRuntime(config.graph_assets, config.embedding_model, 5, .5)
    try:
        for ident in selected:
            data = inputs[ident]
            reduced, mask = reduce_sources(data, ident, config.seed)
            write(config.output / 'inputs/E50' / f'{ident}.json', reduced)
            write(config.output / 'source_masks' / f'{ident}.json', mask)
            changed = {**data, 'graph': reduced_graph(data['graph'], runtime)}
            write(config.output / 'inputs/K5' / f'{ident}.json', changed)
    finally:
        runtime.close()
    write(config.output / 'protocol.json', {'version': 'fig5_v1', 'seed': config.seed,
          'source': str(config.source), 'generation_model': config.model, 'alternate_model': 'gpt-5.6-luna',
          'generation_effort': 'medium', 'evaluation_effort': 'high', 'ordinary_limit': 295, 'additional_limit': 30,
          'total_limit': 325, 'papers': 100, 'paired_papers': 20, 'repeated_papers': 5,
          'planned_reports': 85, 'new_reports': 65, 'render': False, 'scope': 'fixed_input_analysis_and_writing',
          'bootstrap_repeats': 10000, 'sampling': 'field_and_sparse_neighborhood_stress_sample'})
    sync_baselines(config)


def sync_baselines(config: Config) -> dict[str, str]:
    protocol = read(config.source / 'protocol.json')
    if (protocol.get('version') != 'accepted_J_S_v2_expansion' or protocol.get('generation_model') != config.model
            or protocol.get('generation_effort') != 'medium' or protocol.get('evaluation_effort') != 'high'):
        raise ValueError('Fig4 source protocol does not match approved baseline')
    copy_once(config.source / 'protocol.json', config.output / 'baseline/protocol.json')
    cohort = read(config.output / 'cohort.json')
    states = {}
    for paper in cohort['paper_ids']:
        required = ['public_tasks', 'reference/papers', 'analysis/gear', 'analysis/graph_F', 'reports/F']
        present = [copy_once(config.source / folder / f'{paper}.json',
                            config.output / 'baseline' / folder / f'{paper}.json') for folder in required]
        for folder in ('analysis_inputs/gear', 'analysis_inputs/graph_F', 'report_inputs/F', 'screening', 'evaluation_mapping'):
            copy_once(config.source / folder / f'{paper}.json', config.output / 'baseline' / folder / f'{paper}.json')
        from figure_pipeline.fig4_explanation_100.models import ASPECTS
        for aspect in ASPECTS:
            copy_once(config.source / 'aligned_evaluation' / aspect / f'{paper}.json',
                      config.output / 'baseline/aligned_evaluation' / aspect / f'{paper}.json')
        if all(present):
            for stage, method in (('analysis', 'gear'), ('analysis', 'graph_F'), ('reports', 'F')):
                state_file = config.source / 'task_status' / stage / method / f'{paper}.json'
                if not state_file.exists() or read(state_file)['state'] != 'completed':
                    present.append(False)
        states[paper] = 'ready' if all(present) else 'waiting_upstream'
        if states[paper] == 'ready' and paper in cohort['selected']:
            copy_once(config.output / 'baseline/reports/F' / f'{paper}.json',
                      config.output / 'reports/F' / f'{paper}.json')
    for path in (config.source / 'logs/calls').glob('*/record.json'):
        record = read(path)
        if (record.get('state') != 'running' and record.get('paper_id') in cohort['selected']
                and (record.get('stage'), record.get('method')) in {('analysis', 'gear'), ('analysis', 'graph_F'), ('reports', 'F')}):
            copy_once(path, config.output / 'baseline/calls' / path.parent.name / 'record.json')
    records = [(p, read(p)) for p in (config.output / 'baseline/calls').glob('*/record.json')]
    adoption = []
    for paper in cohort['selected']:
        for stage, method in (('analysis', 'gear'), ('analysis', 'graph_F'), ('reports', 'F')):
            matching = [(p, r) for p, r in records if (r['paper_id'], r['stage'], r['method'], r['state']) ==
                        (paper, stage, method, 'completed')]
            if not matching:
                states[paper] = 'waiting_upstream'
                continue
            path, record = max(matching, key=lambda pair: pair[1]['time'])
            if record['model'] != config.model or record['effort'] != 'medium':
                raise ValueError(f'Actual Fig4 baseline configuration mismatch: {paper}/{stage}/{method}')
            adoption.append({'paper_id': paper, 'stage': stage, 'method': method, 'record': str(path),
                             'source_file': str(config.source / stage / method / f'{paper}.json'),
                             'model': record['model'], 'effort': record['effort']})
    write(config.output / 'baseline_adoption.json', adoption)
    write(config.output / 'baseline_status.json', states)
    return states


def view(config: Config, paper: str, condition: str) -> dict[str, Any]:
    name = condition if condition in ('E50', 'K5') else 'F'
    return read(config.output / 'inputs' / name / f'{paper}.json')


def original_blocks(data: dict[str, Any]) -> list[dict[str, Any]]:
    return merged_blocks(data['gear_evidence']['evidence_blocks'] + data['graph_evidence']['evidence_blocks'])
