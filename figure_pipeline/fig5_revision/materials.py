from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_robustness.materials import (
    copy_once,
    original_blocks,
    reduced_graph,
)

from .models import Config


def digest(value: Any) -> str:
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def equivalent(a: Any, b: Any) -> bool:
    """Ignore only floating-point roundoff, never changed evidence or topology."""
    if isinstance(a, float) and isinstance(b, (float, int)):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    return a == b


def effective_material(data: dict[str, Any]) -> dict[str, Any]:
    return {**data, **{k: {'evidence_blocks': data[k]['evidence_blocks']} for k in ('gear_evidence', 'graph_evidence')}}


def identities(data: dict[str, Any]) -> dict[str, str]:
    """Union explicit identifiers and unambiguous titles, rejecting conflicting DOIs."""
    versions = {alias: group['canonical_doi'] for group in read(Path(__file__).with_name('source_versions.json')) for alias in group['aliases']}
    entries = [p for k in ('gear_evidence', 'graph_evidence') for b in data[k]['evidence_blocks'] for p in b['provenance']]
    parents: dict[str, str] = {}
    def root(key: str) -> str:
        parents.setdefault(key, key)
        while parents[key] != key:
            key = parents[key]
        return key
    tokens = []
    titles: dict[str, set[str]] = defaultdict(set)
    for p in entries:
        doi = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)', '', str(p.get('doi') or '').strip(), flags=re.IGNORECASE).lower()
        doi = versions.get(doi, doi)
        title = re.sub(r'\W+', '', str(p.get('title') or '').casefold())
        keys = ['source:' + p['source_id']]
        if doi:
            keys.append('doi:' + doi)
            if title:
                titles[title].add('doi:' + doi)
        for field in ('work_id', 'parent_openalex_work_id', 'parent_paper_id'):
            if p.get(field):
                keys.append('work:' + str(p[field]).rstrip('/').rsplit('/', 1)[-1].lower())
        tokens.append((p, title, keys))
    for p, title, keys in tokens:
        if title and len(titles[title]) <= 1:
            keys += ['title:' + title, *titles[title]]
        for key in keys[1:]:
            parents[root(key)] = root(keys[0])
    groups: dict[str, set[str]] = defaultdict(set)
    for key in parents:
        groups[root(key)].add(key)
    canonical = {}
    for key, values in groups.items():
        dois = sorted(v for v in values if v.startswith('doi:'))
        if len(dois) > 1:
            raise ValueError(f'Conflicting independent work DOIs: {dois}')
        candidates = dois or sorted(v for v in values if v.startswith(('work:', 'title:')))
        if not candidates:
            raise ValueError('Source lacks scientific identity')
        canonical[key] = candidates[0]
    return {p['source_id']: canonical[root('source:' + p['source_id'])] for p in entries}


def retain(data: dict[str, Any], aliases: dict[str, str], kept: set[str]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    for key in ('gear_evidence', 'graph_evidence'):
        blocks = []
        for b in result[key]['evidence_blocks']:
            b['provenance'] = [p for p in b['provenance'] if aliases[p['source_id']] in kept]
            if b['provenance']:
                blocks.append(b)
        result[key]['evidence_blocks'] = blocks
    return result


def graph_mask(graph: dict[str, Any], runtime: Any, kept: set[str]) -> dict[str, Any]:
    masked = copy.deepcopy(graph)
    for card in masked['cards']:
        card['neighbors'] = [n for n in card['neighbors'] if n['parent_paper_id'] in kept]
    result = reduced_graph(masked, runtime, 10)
    for card in result['cards']:
        card['insertion_policy'] = 'local_parent_availability_mask:k<=10:cosine>0.5:no_refill'
    return result


def packet(config: Config, paper: str, condition: str) -> dict[str, Any]:
    return read(config.output / 'inputs' / condition / f'{paper}.json')


def save_packet(config: Config, paper: str, condition: str, data: dict[str, Any], **meta: Any) -> None:
    path = config.output / 'inputs' / condition / f'{paper}.json'
    if path.exists() and read(path) != data:
        raise ValueError(f'Established packet changed: {paper}/{condition}')
    write(path, data)
    write(config.output / 'masks' / condition / f'{paper}.json', {
        'paper_id': paper, 'condition': condition, 'packet_hash': digest(data), **meta})


def ordered(data: dict[str, Any], seed: int, paper: str) -> tuple[dict[str, Any], dict[str, str]]:
    result = copy.deepcopy(data)
    for key in ('gear_evidence', 'graph_evidence'):
        random.Random(f'{seed}:{paper}:{key}').shuffle(result[key]['evidence_blocks'])
    mapping = {b['block_id']: f'B{i:05d}' for i, b in enumerate(original_blocks(result), 1)}
    for key in ('gear_evidence', 'graph_evidence'):
        for block in result[key]['evidence_blocks']:
            block['block_id'] = mapping[block['block_id']]
    return result, mapping


def choose(rows: list[dict[str, Any]], quotas: dict[str, int], seed: str) -> list[str]:
    rng, chosen = random.Random(seed), []
    for group, count in quotas.items():
        available = sorted((r for r in rows if r['group'] == group), key=lambda r: r['paper_id'])
        rng.shuffle(available)
        sparse = [r for r in available if r['sparse']]
        dense = [r for r in available if not r['sparse']]
        picks = sparse[:(count + 1) // 2] + dense[:count // 2]
        picks += [r for r in available if r not in picks][:count - len(picks)]
        if len(picks) != count:
            raise ValueError(f'Insufficient {group} candidates')
        chosen += [r['paper_id'] for r in picks]
    return sorted(chosen)


def prepare(config: Config) -> None:
    from gear.claim_attribution import ClaimGraphRuntime
    old_cohort = read(config.old / 'cohort.json')
    rows = read(config.old / 'paper_coverage.json')
    exploration = old_cohort['selected']
    remaining = [r for r in rows if not r['development'] and r['paper_id'] not in exploration]
    confirm = choose(remaining, {'life': 6, 'physical_engineering': 6, 'medicine': 4, 'earth_environment': 4}, f'{config.seed}:confirm')
    small = choose([r for r in rows if r['paper_id'] in exploration], {'life': 4, 'physical_engineering': 3, 'medicine': 2, 'earth_environment': 1}, f'{config.seed}:small')
    cohort = {'exploration': exploration, 'confirmation': confirm, 'small': small,
              'development': old_cohort['development'], 'paper_ids': old_cohort['paper_ids'], 'seed': config.seed}
    existing = config.output / 'cohort.json'
    if existing.exists() and read(existing) != cohort:
        raise ValueError('Cohort changed')
    write(existing, cohort)
    for r in rows:
        r['split'] = 'development' if r['development'] else 'exploration' if r['paper_id'] in exploration else 'confirmation' if r['paper_id'] in confirm else 'background'
    write(config.output / 'samples.json', rows)
    write_csv(config.output / 'samples.csv', rows)
    tasks, reuse, visible = [], [], []
    runtime = ClaimGraphRuntime(config.graph_assets, config.embedding_model, 10, .5)
    try:
        for row in rows:
            paper = row['paper_id']
            data = read(config.old / 'inputs/F' / f'{paper}.json')
            aliases = identities(data)
            row['previous_readable_sources'] = row['readable_sources']
            row['readable_sources'] = len(set(aliases.values()))
            fulltext = {aliases[p['source_id']] for b in original_blocks(data) for p in b['provenance'] if p['source_type'] == 'fulltext'}
            row['fulltext_fraction'] = len(fulltext) / row['readable_sources'] if row['readable_sources'] else None
            write(config.output / 'identities' / f'{paper}.json', aliases)
            for folder in ('reference/papers', 'public_tasks', 'analysis/gear', 'analysis/graph_F'):
                copy_once(config.source / folder / f'{paper}.json', config.output / 'baseline' / folder / f'{paper}.json')
            save_packet(config, paper, 'F', data, retained_fraction=1)
            save_packet(config, paper, 'E', {**data, 'graph': None, 'graph_evidence': {'evidence_blocks': []}}, interpretation='existing GEAR-only comparator')
            for method in ('F', 'E'):
                source = config.source / 'reports' / method / f'{paper}.json'
                if not row['development']:
                    tasks.append({'paper_id': paper, 'condition': method, 'view': method, 'split': row['split'], 'kind': 'adopted_fig4'})
                    if copy_once(source, config.output / 'reports' / method / f'{paper}.json'):
                        reuse.append({'paper_id': paper, 'condition': method, 'source': str(source), 'reason': 'existing complete report; fresh corrected evaluation'})
            if paper not in exploration + confirm:
                continue
            works = sorted(set(aliases.values()))
            rng = random.Random(f'{config.seed}:{paper}:originals')
            rng.shuffle(works)
            # Preserve old E50 only if canonical work aliases and exact packets still match.
            old_mask = config.old / 'source_masks' / f'{paper}.json'
            if old_mask.exists() and read(old_mask)['source_aliases'] == aliases:
                kept_old = set(read(old_mask)['retained'])
                if len(kept_old) == len(works) // 2:
                    works = [w for w in works if w in kept_old] + [w for w in works if w not in kept_old]
            conditions = ['E75', 'E50', 'E25', 'G50', 'G25', 'K3', 'K5', 'LUNA'] if paper in exploration else ['E50']
            parents = sorted({n['parent_paper_id'] for c in data['graph']['cards'] for n in c['neighbors']})
            random.Random(f'{config.seed}:{paper}:parents').shuffle(parents)
            for condition in conditions:
                changed = data
                meta: dict[str, Any] = {}
                if condition.startswith('E'):
                    fraction = int(condition[1:]) / 100
                    kept = set(works[:int(len(works) * fraction)])
                    changed = retain(data, aliases, kept)
                    meta = {'retained': sorted(kept), 'removed': sorted(set(works) - kept), 'nominal_fraction': fraction,
                            'retained_fraction': len(kept) / len(works) if works else None, 'unit': 'independent_original_work'}
                elif condition.startswith('G'):
                    fraction = int(condition[1:]) / 100
                    kept = set(parents[:int(len(parents) * fraction)])
                    changed = {**data, 'graph': graph_mask(data['graph'], runtime, kept)}
                    meta = {'retained': sorted(kept), 'removed': sorted(set(parents) - kept), 'nominal_fraction': fraction,
                            'retained_fraction': len(kept) / len(parents) if parents else None, 'unit': 'local_historical_parent_paper'}
                elif condition.startswith('K'):
                    changed = {**data, 'graph': reduced_graph(data['graph'], runtime, int(condition[1:]))}
                view = 'F' if condition == 'LUNA' else condition
                if view != 'F':
                    save_packet(config, paper, view, changed, **meta)
                tasks.append({'paper_id': paper, 'condition': condition, 'view': view, 'split': row['split'], 'kind': 'multistage'})
                old_input = config.old / 'inputs' / view / f'{paper}.json'
                old_report = config.old / 'reports' / condition / f'{paper}.json'
                if old_input.exists() and equivalent(effective_material(read(old_input)), effective_material(changed)) and old_report.exists():
                    copy_once(old_report, config.output / 'reports' / condition / f'{paper}.json')
                    reuse.append({'paper_id': paper, 'condition': condition, 'source': str(old_report), 'reason': 'actual branch/writer evidence equality; graph float tolerance 1e-12; unused coverage metadata excluded'})
            if paper in small:
                for condition in ('DW_F', 'DW_E50', 'DW_LUNA', 'LOW', 'HIGH', 'REPEAT2', 'REPEAT3', 'ORDER'):
                    view = 'E50' if condition == 'DW_E50' else 'ORDER' if condition == 'ORDER' else 'F'
                    if condition == 'ORDER':
                        reordered, mapping = ordered(data, config.seed, paper)
                        save_packet(config, paper, 'ORDER', reordered, transformation='evidence order and block labels only')
                        write(config.output / 'order_mapping' / f'{paper}.json', mapping)
                    tasks.append({'paper_id': paper, 'condition': condition, 'view': view, 'split': row['split'], 'kind': 'direct' if condition.startswith('DW') else 'multistage'})
            for path in (config.output / 'inputs').glob(f'*/{paper}.json'):
                current = read(path)
                for b in original_blocks(data):
                    retained_ids = {p['source_id'] for bb in original_blocks(current) for p in bb['provenance']}
                    for p in b['provenance']:
                        visible.append({'paper_id': paper, 'view': path.parent.name, 'block_id': b['block_id'], 'work_id': aliases[p['source_id']],
                                        'source_id': p['source_id'], 'source_type': p['source_type'], 'characters': len(b['text']), 'retained': p['source_id'] in retained_ids})
    finally:
        runtime.close()
    write(config.output / 'samples.json', rows)
    write_csv(config.output / 'samples.csv', rows)
    write(config.output / 'tasks.json', tasks)
    write(config.output / 'reuse.json', reuse)
    write_csv(config.output / 'visible_materials.csv', visible)
    write(config.output / 'protocol.json', {'version': 'fig5_revision_v2', 'seed': config.seed, 'render': False,
          'planned_static_reports': len(tasks), 'support_design_papers': len(exploration + confirm),
          'conditional_critical_reports': 'CRITICAL/NONCRITICAL for eligible; RESTORE first 5 eligible exploration; DW_CRITICAL eligible small',
          'ordinary_limit': config.ordinary_limit, 'additional_limit': config.additional_limit, 'call_limit': config.call_limit,
          'generation_model': config.model, 'alternate_model': 'gpt-5.6-luna', 'evaluation_model': config.model,
          'budgets': {'LOW': 'low', 'F': 'medium', 'HIGH': 'high'}, 'noninferiority_margins': None,
          'inference': 'descriptive paired estimates; no robustness pass/fail without justified prospective margins',
          'cost_scope': 'fixed_material_inference_excludes_retrieval_build_embedding', 'domain_comparator': 'E',
          'confirmation_conditions': ['F', 'E50', 'CRITICAL', 'NONCRITICAL'],
          'primary_aspects': {'E_gradient': ['historical_verification', 'knowledge_position'], 'G_gradient': ['joint_contribution', 'structural_resolution', 'citation_contact']}})
    print(f'PREPARED {len(tasks)} static reports; {len(reuse)} adopted; 20 exploration + 20 confirmation', flush=True)
