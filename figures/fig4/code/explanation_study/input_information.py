from __future__ import annotations

from collections import defaultdict
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write

from .models import Config


def components(nodes: set[str], edges: set[tuple[str, str]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining, groups = set(nodes), []
    while remaining:
        group, pending = set(), [next(iter(remaining))]
        while pending:
            node = pending.pop()
            if node not in group:
                group.add(node)
                pending.extend(adjacency[node] - group)
        remaining -= group
        groups.append(group)
    return groups


def input_information(config: Config) -> None:
    rows, details = [], []
    for paper in read(config.source / 'pilot.json')['paper_ids']:
        material = read(config.source / 'inputs/papers' / f'{paper}.json')
        cards, joint = material['graph']['cards'], material['graph']['joint']
        nodes = {n['claim_id'] for c in cards for n in c['neighbors']}
        targets = {c['claim']['claim_id'] for c in cards}
        local = {tuple(sorted(e)) for c in cards for e in c['neighbor_edges']}
        full = {tuple(sorted(e)) for e in joint['historical_edges']}
        insertions = {tuple(sorted((c['claim']['claim_id'], n['claim_id'])))
                      for c in cards for n in c['neighbors']}
        missing_edges = full - local
        source_sets = {branch: {p['source_id'] for b in material[branch + '_evidence']['evidence_blocks']
                                for p in b['provenance']} for branch in ('gear', 'graph')}
        row: dict[str, Any] = {
            'paper_id': paper, 'target_contributions': len(cards), 'historical_nodes': len(nodes),
            'historical_edges_in_single_cards': len(local), 'historical_edges_in_joint': len(full),
            'historical_edges_only_in_joint': len(missing_edges),
            'components_after_reconstructing_single_cards': len(components(nodes | targets, local | insertions)),
            'components_after_full_joint': len(components(nodes | targets, full | insertions)),
            'positive_contact_edges': sum(any(n.get(k, 0) for k in
                                            ('direct_citation', 'two_hop_path_count', 'shared_reference_count'))
                                          for c in cards for n in c['neighbors']),
            'gear_source_aliases': len(source_sets['gear']), 'graph_source_aliases': len(source_sets['graph']),
            'shared_source_aliases': len(source_sets['gear'] & source_sets['graph']),
            'reconstructible_removed_summary_families': 'effective community count; density; component merges; reachable pairs; cross-boundary share; cross-type count; joint counts from retained union edges',
            'not_reconstructible_from_cards_alone': 'Rao–Stirling centroid distances; global community-pair frequency/first-observed statistics',
        }
        rows.append(row)
        details.append({'paper_id': paper, 'joint_only_historical_edges': sorted(missing_edges),
                        'shared_source_aliases': sorted(source_sets['gear'] & source_sets['graph']),
                        'source_count_note': 'Source aliases, not deduplicated independent publications.'})
    write_csv(config.output / 'input_information.csv', rows)
    write(config.output / 'input_information_details.json', details)
    print(f'Wrote factual input comparison for the original {len(rows)} papers.', flush=True)
