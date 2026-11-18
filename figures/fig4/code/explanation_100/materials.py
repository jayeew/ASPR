from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.config import ROOT
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_study.input_information import components
from figure_pipeline.fig4_mechanisms.generation import project_facts

from .models import ASPECTS, CONDITIONS, NAMES, Config


def base(config: Config, paper: str) -> dict[str, Any]:
    return read(config.output / 'inputs/papers' / f'{paper}.json')


def merged_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for block in blocks:
        key = block['block_id']
        if key not in result:
            result[key] = copy.deepcopy(block)
        else:
            if result[key]['text'] != block['text']:
                raise ValueError(f'Conflicting original passages for block {key}')
            for provenance in block['provenance']:
                if provenance not in result[key]['provenance']:
                    result[key]['provenance'].append(copy.deepcopy(provenance))
    return list(result.values())


def view(config: Config, paper: str, condition: str) -> dict[str, Any]:
    data = base(config, paper)
    result = {'manuscript': data['manuscript']}
    if condition == 'T':
        return result
    evidence = []
    if condition != 'G':
        evidence.extend(data['gear_evidence']['evidence_blocks'])
    if condition != 'E':
        graph = project_facts(data['graph'], condition)
        if condition == 'F_noP':
            for card in graph['cards']:
                for neighbor in card['neighbors']:
                    neighbor.pop('edge_type', None)
        result['graph'] = graph
        evidence.extend(data['graph_evidence']['evidence_blocks'])
    result['original_evidence'] = merged_blocks(evidence)
    return result


def public_tasks(config: Config, paper: str) -> dict[str, Any]:
    return read(config.output / 'public_tasks' / f'{paper}.json')


def information(paper: str, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cards, joint = data['graph']['cards'], data['graph']['joint']
    nodes = {n['claim_id'] for c in cards for n in c['neighbors']}
    targets = {c['claim']['claim_id'] for c in cards}
    local = {tuple(sorted(e)) for c in cards for e in c['neighbor_edges']}
    union = {tuple(sorted(e)) for e in joint['historical_edges']}
    inserted = {tuple(sorted((c['claim']['claim_id'], n['claim_id']))) for c in cards for n in c['neighbors']}
    contacts = sum(any(n.get(k, 0) for k in ('direct_citation', 'two_hop_path_count', 'shared_reference_count'))
                   for c in cards for n in c['neighbors'])
    gear_blocks = {b['block_id'] for b in data['gear_evidence']['evidence_blocks']}
    graph_blocks = {b['block_id'] for b in data['graph_evidence']['evidence_blocks']}
    row = {'paper_id': paper, 'target_contributions': len(cards), 'historical_nodes': len(nodes),
           'gear_original_blocks': len(gear_blocks), 'graph_original_blocks': len(graph_blocks),
           'shared_original_blocks': len(gear_blocks & graph_blocks),
           'historical_edges_in_single_cards': len(local), 'historical_edges_in_joint': len(union),
           'historical_edges_only_in_joint': len(union - local), 'positive_contact_edges': contacts,
           'citation_opportunity': bool(contacts),
           'components_from_single_cards': len(components(nodes | targets, local | inserted)),
           'components_with_joint': len(components(nodes | targets, union | inserted))}
    return row, {'paper_id': paper, 'joint_only_historical_edges': sorted(union - local),
                 'reconstructible': ['density', 'component_merges', 'reachable_pairs', 'effective_communities',
                                     'cross_boundary_weight', 'cross_type_counts'],
                 'not_reconstructible_from_cards_alone': ['centroid_distance_diversity', 'global_pair_frequency']}


def prepare(config: Config) -> None:
    roster = read(config.source / 'papers.json')
    pilot = read(ROOT / 'outputs/fig4_capabilities_pilot/pilot.json')['paper_ids']
    write(config.output / 'cohort.json', {'paper_ids': [p['paper_id'] for p in roster], 'pilot_ids': pilot})
    write(config.output / 'papers.json', roster)
    inventory, sizes, information_rows, details = [], [], [], []
    for paper in roster:
        ident = paper['paper_id']
        paths = {'original_inputs': config.source / 'inputs/papers' / f'{ident}.json',
                 'fixed_reference': config.source / 'inputs/evaluation' / f'{ident}.json',
                 'fig3_criteria': ROOT / 'outputs/fig3_reference/study/annotations/quality_checklist/papers' / f'{ident}.json'}
        data, fixed, checklist = (read(paths[key]) for key in paths)
        data = {k: v for k, v in data.items() if k != 'gear'}
        write(config.output / 'inputs/papers' / f'{ident}.json', data)
        full = view(config, ident, 'F')
        write(config.output / 'inputs/reference' / f'{ident}.json', {
            'manuscript': data['manuscript'], 'native_graph': data['graph'],
            'evidence_blocks': full['original_evidence'], 'core_contribution_identities': fixed['cores'],
            'fig3_criteria': [{'dimension': d['dimension'], 'criteria': d['criteria']} for d in checklist['dimensions']]})
        row, detail = information(ident, data)
        information_rows.append(row); details.append(detail)
        for kind, path in paths.items():
            inventory.append({'paper_id': ident, 'material': kind, 'source_file': str(path), 'state': 'reused'})
        for condition in CONDITIONS:
            material = view(config, ident, condition)
            sizes.append({'paper_id': ident, 'condition': condition, 'configuration': NAMES[condition],
                          'characters': len(json.dumps(material, ensure_ascii=False)), 'fields': list(material)})
    write_csv(config.output / 'reuse_inventory.csv', inventory)
    write_csv(config.output / 'input_inventory.csv', sizes)
    write_csv(config.output / 'input_information.csv', information_rows)
    write(config.output / 'input_information_details.json', details)
    write(config.output / 'conditions.json', NAMES)
    write(config.output / 'protocol.json', {
        'version': 'sol_medium_high_expansion_v1', 'model': config.model,
        'generation_effort': config.generation_effort, 'evaluation_effort': config.evaluation_effort,
        'aspects': ASPECTS, 'ordinary_limit': config.ordinary_limit, 'additional_limit': config.additional_limit,
        'total_limit': config.call_limit, 'pilot_review_required': True, 'length_target': None,
        'reference_is_private': True, 'old_generated_judgments_reused': False,
        'transport_observation': 'WebSocket support configured; negotiated transport not exposed by current event metadata',
        'scope': 'controlled_information_and_interpretation_under_common_scientific_tasks',
        'bootstrap_repeats': config.bootstrap_repeats, 'seed': config.seed})
    restrictions = read(ROOT / 'outputs/fig4_explanation_study/unresolved_tasks.json')
    write(config.output / 'known_restrictions.json', restrictions)
    print(f'Prepared {len(roster)} papers, {sum(r["target_contributions"] for r in information_rows)} claims; no model calls.', flush=True)
