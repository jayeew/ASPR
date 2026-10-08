"""Only local reuse and field deletion; no model calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.materials import PacketIndex, encoded, packet
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_mechanisms.generation import project_facts

from .config import CONDITIONS, METRICS, NAMES, Config


def source_file(config: Config, stage: str, paper: str) -> Path:
    return config.source / stage / f'{paper}.json'


def papers(config: Config) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (config.source / 'papers.jsonl').read_text().splitlines() if line]


def matching_sources(pool: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    keys = {key for key in keys if key}
    sources = []
    for source in pool['sources']:
        if source['source_type'] not in {'fulltext', 'abstract'}:
            continue
        identities = {source['source_id'], source.get('work_id') or '',
                      (source.get('doi') or '').split('/')[-1], Path(source.get('original_path') or '').stem}
        if identities & keys or any(p['block_id'] in keys for p in source['block_ids']):
            sources.append(source)
    used = {p['block_id'] for source in sources for p in source['block_ids']}
    return {'sources': sources, 'blocks': [b for b in pool['blocks'] if b['block_id'] in used]}


def evidence_keys(gear: dict[str, Any]) -> set[str]:
    keys = set(gear.get('cited_source_ids', []))
    for assessment in gear['analyses']:
        for finding in assessment['findings']:
            keys.update(finding['evidence_keys'])
    return {k.split('@')[0] for k in keys}


def original_packet(pool: dict[str, Any], query: Any) -> dict[str, Any]:
    return packet(pool, query, 24000, True, PacketIndex(pool))


def payload(config: Config, paper: str, condition: str) -> dict[str, Any]:
    base = read(config.output / 'inputs/papers' / f'{paper}.json')
    result = {'manuscript': base['manuscript']}
    if condition == 'T':
        return result
    evidence = []
    if condition != 'G':
        result['gear'] = base['gear']
        evidence.extend(base['gear_evidence']['evidence_blocks'])
    if condition != 'E':
        result['graph'] = project_facts(base['graph'], condition)
        evidence.extend(base['graph_evidence']['evidence_blocks'])
    if evidence:
        result['original_evidence'] = list({b['block_id']: b for b in evidence}.values())
    return result


def prepare(config: Config) -> None:
    inventory, sizes = [], []
    all_papers = papers(config)
    for paper in all_papers:
        ident = paper['paper_id']
        paths = {
            'manuscript': source_file(config, 'inputs/prepare/papers', ident),
            'fixed': source_file(config, 'inputs/fixed_analysis', ident),
            'graph': source_file(config, 'evidence/native_graph', ident),
            'core': source_file(config, 'annotations/core/papers', ident),
            'reference': source_file(config, 'evidence/reference/papers', ident),
            'pool': source_file(config, 'evidence/evidence_pool/papers', ident),
            'joint_structure': source_file(config, 'inputs/joint_structure', ident),
        }
        data = {name: read(path) for name, path in paths.items()}
        graph = copy.deepcopy(data['graph'])
        for card in graph['cards']:
            card.pop('notes', None)
        graph['joint'].pop('notes', None)
        gear = data['fixed']['gear']
        gear_pool = matching_sources(data['pool'], evidence_keys(gear))
        graph_ids = {value for card in graph['cards'] for n in card['neighbors']
                     for value in (n['parent_paper_id'], n.get('parent_openalex_work_id', ''))}
        graph_pool = matching_sources(data['pool'], graph_ids)
        query = [c['claim']['claim_text'] for c in graph['cards']]
        common = {'manuscript': data['manuscript']['manuscript'], 'gear': gear['analyses'],
                  'graph': graph, 'gear_evidence': original_packet(gear_pool, gear['analyses']),
                  'graph_evidence': original_packet(graph_pool, query)}
        write(config.output / 'inputs/papers' / f'{ident}.json', common)
        eval_evidence = original_packet(data['pool'], {'core': data['core']['items'],
                                                     'reference': data['reference']['items']})
        write(config.output / 'inputs/evaluation' / f'{ident}.json', {
            'manuscript': common['manuscript'], 'cores': data['core']['items'],
            'reference': data['reference']['items'], **eval_evidence})
        write(config.output / 'inputs/joint_structure' / f'{ident}.json', data['joint_structure'])
        for name, path in paths.items():
            inventory.append({'paper_id': ident, 'material': name, 'source_file': str(path), 'state': 'reused'})
        for condition in CONDITIONS:
            material = payload(config, ident, condition)
            sizes.append({'paper_id': ident, 'condition': condition, 'input_chars': len(encoded(material)),
                          'input_fields': list(material), 'fits_material_limit': len(encoded(material)) <= config.material_max_chars})
    write(config.output / 'papers.json', all_papers)
    write(config.output / 'pilot.json', read(config.source / 'pilot.json'))
    write(config.output / 'conditions.json', {'names': NAMES, 'metrics': METRICS, 'model': config.model,
          'generation_effort': 'high', 'evaluation_effort': 'xhigh', 'report_length_target': None,
          'call_limit': 1400, 'pilot_call_limit': 70, 'automatic_retries': False,
          'scope': 'existing_GEAR_and_structured_graph_inputs_direct_single_writer',
          'graph_interpretation_text': False, 'extra_analysis_stages': False})
    write_csv(config.output / 'reuse_inventory.csv', inventory)
    write_csv(config.output / 'input_inventory.csv', sizes)
    print(f'Prepared {len(all_papers)} papers; largest generation input {max(r["input_chars"] for r in sizes):,} chars.', flush=True)
