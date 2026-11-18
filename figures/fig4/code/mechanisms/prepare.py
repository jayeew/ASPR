"""Reuse source-bound inputs and derive joint-exclusive connectivity without models."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import networkx as nx

from figure_pipeline.fig3_revision.materials import PacketIndex, packet
from figure_pipeline.fig3_revision.retrieval import doi, eligible
from figure_pipeline.fig3_revision.storage import Store, roster

from .config import CONDITIONS, Config, condition_rows
from .io import artifact, jsonl, read, record, write, write_csv

SAFE_FIELDS = ('source_id', 'source_type', 'title', 'doi', 'publication_date', 'year',
               'original_path', 'fulltext_origin', 'char_range', 'work_id')


def historical_pool(pool: dict[str, Any], target: dict[str, Any]) -> tuple[dict, list[dict]]:
    blocks = {b['block_id']: b for b in pool['blocks']}
    sources, groups, excluded = [], {}, []
    for source in pool['sources']:
        if source['source_type'] not in {'fulltext', 'abstract', 'manuscript'}:
            if source['source_type'] == 'metadata_only':
                excluded.append({'source_id': source['source_id'], 'reason': 'no_readable_original_passage'})
            continue
        text = ''.join(blocks[b['block_id']]['text'] for b in source['block_ids'])
        if source['source_type'] != 'manuscript' and not eligible({**source, 'passage': text}, target):
            excluded.append({'source_id': source['source_id'], 'reason': 'target_version_or_time_not_eligible'})
            continue
        row = {k: source[k] for k in SAFE_FIELDS if k in source}
        row['block_ids'] = source['block_ids']
        sources.append(row)
        if source['source_type'] == 'manuscript':
            continue
        key = doi(source.get('doi')) or source['source_id']
        group = groups.setdefault(key, {'canonical_source_id': key, 'aliases': [], 'passages': []})
        if source['source_id'] not in group['aliases']:
            group['aliases'].append(source['source_id'])
        for part in source['block_ids']:
            passage = {**row, **part, 'text': blocks[part['block_id']]['text']}
            passage.pop('block_ids')
            if passage not in group['passages']:
                group['passages'].append(passage)
    used = {b['block_id'] for s in sources for b in s['block_ids']}
    return {'blocks': [b for b in pool['blocks'] if b['block_id'] in used], 'sources': sources,
            'excluded': excluded, 'limitations': pool.get('limitations', [])}, list(groups.values())


def connected_gain(history: nx.Graph, insertion: list[list[str]]) -> set[tuple[str, str]]:
    before = {n: i for i, nodes in enumerate(nx.connected_components(history)) for n in nodes}
    graph = history.copy()
    graph.add_edges_from(insertion)
    after = {n: i for i, nodes in enumerate(nx.connected_components(graph)) for n in nodes}
    return {(a, b) for a, b in combinations(sorted(history), 2)
            if before[a] != before[b] and after[a] == after[b]}


def joint_record(facts: dict[str, Any]) -> dict[str, Any]:
    joint = facts['joint']
    history = nx.Graph()
    history.add_nodes_from(n['claim_id'] for c in facts['cards'] for n in c['neighbors'])
    history.add_edges_from(joint['historical_edges'])
    all_pairs = connected_gain(history, joint['insertion_edges'])
    single = set()
    for card in facts['cards']:
        edges = [[card['claim']['claim_id'], n['claim_id']] for n in card['neighbors']]
        single.update(connected_gain(history, edges))
    exclusive = all_pairs-single
    denominator = math.comb(len(history), 2)
    return {'history_node_count': len(history), 'history_pair_denominator': denominator,
            'new_pairs_all': len(all_pairs), 'new_pairs_single_union': len(single),
            'joint_exclusive_pairs': len(exclusive),
            'J_topo': len(exclusive)/denominator if denominator else None,
            'exclusive_pair_ids': sorted(exclusive), 'historical_nodes': sorted(history),
            'historical_edges': joint['historical_edges'], 'insertion_edges': joint['insertion_edges'],
            'claim_neighbors': {c['claim']['claim_id']: [n['claim_id'] for n in c['neighbors']] for c in facts['cards']}}


def pilot_ids(papers: list[dict[str, Any]]) -> list[str]:
    fields: dict[str, list[str]] = defaultdict(list)
    for paper in papers:
        fields[paper.get('field_name') or 'Unknown'].append(paper['paper_id'])
    selected = sorted(fields, key=lambda f: (-len(fields[f]), f))[:5]
    return [min(fields[field]) for field in selected]


def prepare(config: Config) -> None:
    old = Store(config.model_copy(update={'output': config.source}))
    new = Store(config)
    papers = roster(config)
    claims_out, cores_out, sources_out, inventory, graph_out = [], [], [], [], []
    for paper in papers:
        ident = paper['paper_id']
        try:
            data, claims = old.get('prepare', ident), old.get('claims', ident)
            core, reference = old.get('core', ident), old.get('reference', ident)
            bundle = {method: old.get(method, ident) for method in ('gear', 'graph', 'full')}
            bundle = {method: {k: v for k, v in value.items() if k not in {'body', 'sources', 'finding_retention'}}
                      for method, value in bundle.items()}
            facts_path = config.source/'evidence/native_graph'/f'{ident}.json'
            facts = read(facts_path)
            pool, history = historical_pool(old.get('evidence_pool', ident),
                                            {'title': paper['title'], 'doi': paper.get('doi'), 'cutoff': data['cutoff']})
            for stage, value in [('prepare', {k: v for k, v in data.items() if k not in
                                               {'reviews', 'review_blocks', 'reports', 'review_attachment_spans'}}),
                                 ('claims', claims), ('core', core), ('reference', reference), ('evidence_pool', pool)]:
                new.put(stage, ident, value)
                inventory.append({'paper_id': ident, 'material': stage, 'source_file': str(old.path(stage, ident)),
                                  'state': 'reused', 'missing_reason': ''})
            inventory.append({'paper_id': ident, 'material': 'historical_source_coverage',
                'source_file': str(old.path('evidence_pool', ident)),
                'state': 'limited' if pool['excluded'] or pool['limitations'] else 'reused',
                'missing_reason': {'excluded_sources': pool['excluded'], 'retrieval_limitations': pool['limitations']}})
            index = PacketIndex(pool)
            packets = {c['claim_id']: packet(pool, c['normalized_claim_text'], 12000, True, index) for c in claims['claims']}
            write(config.output/'inputs/common'/f'{ident}.json', {'paper': paper, 'claims': claims['claims'],
                                                                 'claim_materials': packets})
            write(config.output/'inputs/fixed_analysis'/f'{ident}.json', bundle)
            write(config.output/'evidence/native_graph'/f'{ident}.json', facts)
            graph = {'paper_id': ident, **joint_record(facts)}
            write(config.output/'inputs/joint_structure'/f'{ident}.json', graph)
            graph_out.append(graph)
            claims_out.extend({'paper_id': ident, **c} for c in claims['claims'])
            cores_out.extend({'paper_id': ident, **c, 'claim_mapping': next(
                (m for m in core.get('mapping', []) if m['core_id'] == c['core_id']), None)} for c in core['items'])
            sources_out.extend({'paper_id': ident, **h} for h in history)
            inventory.extend({'paper_id': ident, 'material': method+'_analysis',
                              'source_file': str(old.path(method, ident)), 'state': 'reused', 'missing_reason': ''}
                             for method in bundle)
            inventory.append({'paper_id': ident, 'material': 'native_graph', 'source_file': str(facts_path),
                              'state': 'reused', 'missing_reason': ''})
            record(config.output, 'prepare', ident, '', 'completed')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            inventory.append({'paper_id': ident, 'material': 'input_bundle', 'source_file': '',
                              'state': 'failed', 'missing_reason': str(exc)})
            record(config.output, 'prepare', ident, '', 'failed', error=str(exc))
    jsonl(config.output/'papers.jsonl', papers)
    jsonl(config.output/'claims.jsonl', claims_out)
    jsonl(config.output/'cores.jsonl', cores_out)
    jsonl(config.output/'history_sources.jsonl', sources_out)
    jsonl(config.output/'joint_graph_records.jsonl', graph_out)
    write_csv(config.output/'reuse_inventory.csv', inventory)
    write(config.output/'pilot.json', {'paper_ids': pilot_ids(papers), 'selection': 'largest_5_fields_then_min_paper_id'})
    write(config.output/'conditions.json', {'conditions': condition_rows(), 'model': config.model,
          'efforts': config.efforts, 'report_length_target': None, 'budget': 'matched_downstream_tasks_actual_usage_reported',
          'repeat_evaluation': False, 'source': str(config.source)})
    write_csv(config.output/'contrasts.csv', [
        {'from_condition': a, 'to_condition': b, 'operation': op, 'edge_label': op}
        for a, b, op in [('T', 'E', '+GEAR'), ('T', 'G', '+Graph'), ('E', 'F', '+Graph'),
                          ('G', 'F', '+GEAR'), ('F', 'F_noJ', 'Joint removed'),
                          ('F', 'F_noM', 'Metrics hidden'), ('F', 'F_noP', 'Paths hidden')]])
    print(f'Prepared {len(graph_out)}/{len(papers)} papers, {len(claims_out)} claims, {len(cores_out)} cores.', flush=True)
