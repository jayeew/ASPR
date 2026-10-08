"""Whole-corpus graph summaries and bounded, source-linked prediction packets."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'data/claim_graph'
OUT = ROOT / 'outputs/fig7_reference/forecast'
YEARS = ('2023', '2024', '2025')


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def load_graph() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_parquet(ASSETS / 'claim_nodes.parquet')
    communities = pd.read_parquet(ASSETS / 'claim_communities.parquet')
    papers = pd.read_parquet(ASSETS / 'canonical_target_works.parquet')
    nodes = nodes.merge(communities, on='claim_id', validate='one_to_one')
    nodes = nodes.merge(papers[['nature_article_id', 'primary_topic_name', 'field_name', 'doi', 'source_name']],
                        left_on='parent_paper_id', right_on='nature_article_id', validate='many_to_one')
    nodes['date'] = pd.to_datetime(nodes.publication_date).dt.strftime('%Y-%m-%d')
    nodes['year'] = nodes.date.str[:4]
    assert nodes.claim_id.is_unique and nodes.year.isin(YEARS).all()
    topics = {name: f'U{i:04d}' for i, name in enumerate(sorted(nodes.loc[
        nodes.community_id.isna(), 'primary_topic_name'].fillna('Unknown topic').unique()))}
    nodes['group'] = [f'C{int(c):04d}' if pd.notna(c) else topics[t if pd.notna(t) else 'Unknown topic']
                      for c, t in zip(nodes.community_id, nodes.primary_topic_name)]
    nodes['nid'] = [f'N{i:05d}' for i in range(len(nodes))]
    edges = pd.read_parquet(ASSETS / 'semantic_claim_edges.parquet')
    edges = edges.loc[edges.cosine_similarity > .5].copy().reset_index(drop=True)
    index = nodes.set_index('claim_id')
    for side in ('earlier', 'later'):
        edges[side + '_group'] = edges[side + '_claim_id'].map(index['group'])
        edges[side + '_nid'] = edges[side + '_claim_id'].map(index.nid)
        assert edges[side + '_nid'].notna().all()
    assert (edges.earlier_publication_date < edges.later_publication_date).all()
    assert (edges.earlier_paper_id != edges.later_paper_id).all()
    assert edges.groupby('later_claim_id').size().max() <= 10
    edges['year'] = edges.later_publication_date.str[:4]
    edges['cross_group'] = edges.earlier_group != edges.later_group
    edges['eid'] = [f'E{i:06d}' for i in range(len(edges))]
    return nodes, edges


def node_record(row: Any) -> dict[str, Any]:
    return {'id': row.nid, 'claim_id': row.claim_id, 'paper_id': row.parent_paper_id,
            'date': row.date, 'type': row.claim_type, 'group': row.group,
            'topic': row.primary_topic_name, 'claim': row.claim_text, 'title': row.title,
            'doi': row.doi, 'source_fragments': list(row.source_fragments),
            'source_sentence_ids': list(row.source_sentence_ids)}


def representative_ids(frame: pd.DataFrame, cross_degree: pd.Series) -> list[str]:
    """Recent, cross-connected and method examples; preserve paper diversity."""
    frame = frame.assign(bridge=frame.claim_id.map(cross_degree).fillna(0))
    limit = 6 if len(frame) >= 100 else (3 if len(frame) >= 10 else 1)
    order = pd.concat([
        frame.sort_values(['date', 'bridge', 'claim_id'], ascending=[False, False, True]).head(1),
        frame.sort_values(['bridge', 'date', 'claim_id'], ascending=[False, False, True]).head(1),
        frame.loc[frame.claim_type == 'METHOD'].sort_values(['date', 'claim_id'], ascending=[False, True]).head(1),
        frame.sort_values(['date', 'bridge', 'claim_id'], ascending=[False, False, True]),
    ]).drop_duplicates('parent_paper_id')
    return order.head(limit).nid.tolist()


def summarize(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    totals = nodes.groupby('year').parent_paper_id.nunique().to_dict()
    outgoing = edges.groupby(['later_group', 'year']).size()
    crossed = edges.loc[edges.cross_group].groupby(['later_group', 'year']).size()
    cross_degree = pd.concat([edges.loc[edges.cross_group, 'earlier_claim_id'],
                              edges.loc[edges.cross_group, 'later_claim_id']]).value_counts()
    incident: dict[str, list[int]] = defaultdict(list)
    for row in edges.itertuples():
        incident[row.later_group].append(row.Index)
        if row.earlier_group != row.later_group:
            incident[row.earlier_group].append(row.Index)
    records = {row.nid: node_record(row) for row in nodes.itertuples()}
    profiles = []
    for group, frame in nodes.groupby('group', sort=True):
        edge_frame = edges.loc[incident[group]]
        reps = representative_ids(frame, cross_degree)
        selected_edges = edge_frame.loc[edge_frame.earlier_nid.isin(reps) | edge_frame.later_nid.isin(reps)]
        selected_edges = selected_edges.sort_values(
            ['cross_group', 'later_publication_date', 'cosine_similarity'], ascending=False)
        examples = []
        papers_seen: set[tuple[str, str]] = set()
        for row in selected_edges.itertuples():
            pair = (row.earlier_paper_id, row.later_paper_id)
            if pair in papers_seen:
                continue
            papers_seen.add(pair)
            examples.append({'id': row.eid, 'earlier': row.earlier_nid, 'later': row.later_nid,
                             'cosine': round(row.cosine_similarity, 6),
                             'groups': [row.earlier_group, row.later_group]})
            reps.extend([row.earlier_nid, row.later_nid])
            if len(examples) >= (3 if len(frame) >= 100 else 1):
                break
        annual = []
        for year in YEARS:
            part = frame.loc[frame.year == year]
            n_out = int(outgoing.get((group, year), 0))
            n_cross = int(crossed.get((group, year), 0))
            annual.append([len(part), part.parent_paper_id.nunique(),
                           round(part.parent_paper_id.nunique() / totals[year], 6),
                           n_out, n_cross, round(n_cross / n_out, 6) if n_out else None])
        other = np.where(edge_frame.earlier_group == group, edge_frame.later_group, edge_frame.earlier_group)
        links = Counter(str(g) for g in other if g != group)
        method_edges = edge_frame.loc[(edge_frame.earlier_claim_type == 'METHOD') |
                                      (edge_frame.later_claim_type == 'METHOD')]
        method_other = set(method_edges.earlier_group) | set(method_edges.later_group)
        method_other.discard(group)
        profiles.append({'id': group, 'kind': 'community' if group.startswith('C') else 'unassigned_topic',
                         'claims': len(frame), 'papers': frame.parent_paper_id.nunique(),
                         'topics': frame.primary_topic_name.value_counts().head(3).index.tolist(),
                         'annual': annual, 'types': frame.claim_type.value_counts().to_dict(),
                         'neighbors': links.most_common(3), 'method_incident_edges': len(method_edges),
                         'method_neighbor_groups': len(method_other),
                         'evidence': list(dict.fromkeys(reps)), 'edge_examples': examples})
    return profiles, records


def pack(profiles: list[dict[str, Any]], records: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    packets: list[list[dict[str, Any]]] = [[] for _ in range(8)]
    sizes = [0] * 8
    lookup = {p['id']: p for p in profiles}
    def weight(p: dict[str, Any]) -> int:
        return len(compact(p)) + sum(len(records[k]['claim']) + 125 for k in p['evidence'])
    for profile in sorted(profiles, key=lambda p: (-weight(p), p['id'])):
        target = min(range(8), key=lambda i: sizes[i])
        packets[target].append(profile)
        sizes[target] += weight(profile)
    result = []
    for i, group_profiles in enumerate(packets):
        ids = sorted({k for p in group_profiles for k in p['evidence']})
        neighbors = sorted({n for p in group_profiles for n, _ in p['neighbors']})
        evidence = [{k: records[n][k] for k in ('id', 'paper_id', 'date', 'type', 'group', 'claim')}
                    for n in ids]
        packet = {'packet_id': f'P{i+1:02d}', 'corpus': summary,
                  'annual_columns': ['claims', 'papers', 'share_of_corpus_papers', 'outgoing_edges',
                                     'cross_group_outgoing_edges', 'cross_group_fraction'],
                  'years': list(YEARS), 'group_columns': list(group_profiles[0]),
                  'edge_columns': ['id', 'earlier', 'later', 'cosine', 'groups'],
                  'groups': [list({**g, 'edge_examples': [list(e.values()) for e in g['edge_examples']]}.values()) for g in group_profiles],
                  'neighbor_labels': {k: lookup[k]['topics'][:1] for k in neighbors},
                  'evidence_columns': list(evidence[0]), 'evidence': [list(e.values()) for e in evidence]}
        assert len(compact(packet)) < 448000, f'Packet {i+1} {len(compact(packet))} chars; compress before calls'
        result.append(packet)
    return result


def prepare() -> None:
    nodes, edges = load_graph()
    profiles, records = summarize(nodes, edges)
    annual = {y: {'claims': int((nodes.year == y).sum()),
                  'papers': int(nodes.loc[nodes.year == y, 'parent_paper_id'].nunique()),
                  'edges': int((edges.year == y).sum())} for y in YEARS}
    summary = {'claims': len(nodes), 'papers': nodes.parent_paper_id.nunique(), 'edges': len(edges),
               'communities': nodes.community_id.nunique(), 'unassigned_claims': int(nodes.community_id.isna().sum()),
               'groups': len(profiles), 'cutoff': '2025-12-31', 'annual': annual}
    packets = pack(profiles, records, summary)
    seen = [g[0] for p in packets for g in p['groups']]
    assert len(seen) == len(set(seen)) == len(profiles)
    assert sum(g[2] for p in packets for g in p['groups']) == len(nodes)
    write(OUT / 'data/summary.json', summary)
    write(OUT / 'data/group_profiles.json', profiles)
    write(OUT / 'data/evidence.json', records)
    nodes[['nid', 'claim_id', 'parent_paper_id', 'group', 'year']].to_parquet(OUT / 'data/node_assignments.parquet', index=False)
    write(OUT / 'data/coverage.json', {'all_nodes_accounted_for': True, 'all_groups_accounted_for': True,
          'packets': [{'id': p['packet_id'], 'groups': len(p['groups']), 'claims': sum(g[2] for g in p['groups']),
                       'evidence_claims': len(p['evidence']), 'input_chars': len(compact(p))} for p in packets]})
    for packet in packets:
        write(OUT / 'packets' / (packet['packet_id'] + '.json'), packet)
    print(compact(summary), flush=True)
    print(compact(read(OUT / 'data/coverage.json')), flush=True)
