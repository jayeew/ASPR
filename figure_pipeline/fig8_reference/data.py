"""Derive paper-level statistics without model inference or graph mutation."""
from __future__ import annotations

import hashlib
import json
import gzip
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .helpers import (
    effective_communities, neighborhood_overlap, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'data/claim_graph'
OUT = ROOT / 'outputs/fig8_reference'
YEARS = [2023, 2024, 2025]
JOURNALS = ['Nature Communications', 'Nature Biomedical Engineering',
            'Nature Cell Biology', 'Nature Ecology & Evolution', 'Nature Human Behaviour',
            'Nature Microbiology', 'Nature Immunology', 'Nature Structural & Molecular Biology']
SHORT = ['Nature Commun.', 'Nat. Biomed. Eng.', 'Nat. Cell Biol.', 'Nat. Ecol. Evol.',
         'Nat. Hum. Behav.', 'Nat. Microbiol.', 'Nat. Immunol.', 'Nat. Struct. Mol. Biol.']
TYPES = ['METHOD', 'FINDING', 'MECHANISM', 'RESOURCE', 'THEORY']
SOURCES: dict[str, dict] = {}


def read_table(name: str, columns: list[str] | None = None) -> pa.Table:
    path = ASSETS / name
    SOURCES[str(path.relative_to(ROOT))] = {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': pq.ParquetFile(path).metadata.num_rows,
    }
    return pq.read_table(path, columns=columns)


def paper_metrics(meta: dict, claims: list[dict], neighbors: dict[str, set[str]],
                  communities: dict[str, int | None]) -> dict:
    sets = [neighbors[c['claim_id']] for c in claims]
    union = set().union(*sets) if sets else set()
    assigned = [communities[n] for n in union if communities[n] is not None]
    overlap, pairs = neighborhood_overlap(sets)
    multiplicity = Counter(n for subset in sets for n in subset)
    row = dict(paper_id=meta['nature_article_id'], title=meta['title'],
               journal=meta['source_name'], year=meta['publication_year'],
               publication_date=meta['publication_date'], field=meta['field_name'],
               topic=meta['primary_topic_name'], work_type=meta['work_type'],
               n_claims=len(claims), n_nonempty_claims=sum(bool(s) for s in sets),
               n_historical_neighbors=len(union), n_assigned_neighbors=len(assigned),
               community_coverage=len(assigned) / len(union) if union else None,
               effective_communities=effective_communities(assigned),
               mean_jaccard=overlap, n_valid_pairs=pairs,
               shared_neighbor_fraction=sum(n >= 2 for n in multiplicity.values()) / len(union) if pairs else None,
               claim_ids=[c['claim_id'] for c in claims])
    row['scatter_eligible'] = row['effective_communities'] is not None and overlap is not None
    reasons = []
    if not claims:
        reasons.append('no_claim_records')
    elif not union:
        reasons.append('no_eligible_prior_neighbors')
    else:
        if not assigned:
            reasons.append('no_assigned_neighbor_community')
        if not pairs:
            reasons.append('fewer_than_two_nonempty_claim_neighborhoods')
    row['scatter_missing_reasons'] = reasons
    row.update({f'type_{t}': sum(c['claim_type'] == t for c in claims) / len(claims)
                if claims else None for t in TYPES})
    return row


def summarize(papers: list[dict]) -> list[dict]:
    summaries = []
    for journal in JOURNALS:
        for year in [0, *YEARS]:
            subset = [p for p in papers if p['journal'] == journal and (not year or p['year'] == year)]
            claimed = [p for p in subset if p['n_claims']]
            plotted = [p for p in subset if p['scatter_eligible']]
            c_rows = [p for p in subset if p['shared_neighbor_fraction'] is not None]
            row = dict(journal=journal, year=year, n_papers=len(subset), n_claimed_papers=len(claimed),
                       n_claims=sum(p['n_claims'] for p in subset), n_scatter=len(plotted),
                       coordinate_coverage=len(plotted) / len(subset), n_distribution=len(c_rows),
                       median_shared_fraction=median(p['shared_neighbor_fraction'] for p in c_rows),
                       median_community_coverage=median(p['community_coverage'] for p in subset if p['community_coverage'] is not None))
            row.update({f'type_{t}': mean(p[f'type_{t}'] for p in claimed) for t in TYPES})
            summaries.append(row)
    return summaries


def prepare() -> dict:
    for folder in ['data', 'sources', 'qa', 'final', 'panels']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    metadata = read_table('canonical_target_works.parquet').to_pylist()
    claims = read_table('claim_nodes.parquet').to_pylist()
    communities = {r['claim_id']: r['community_id'] for r in read_table('claim_communities.parquet').to_pylist()}
    raw_edges = read_table('semantic_claim_edges.parquet', [
        'earlier_claim_id', 'later_claim_id', 'earlier_paper_id', 'later_paper_id',
        'earlier_publication_date', 'later_publication_date', 'cosine_similarity', 'semantic_rank'])
    kept = raw_edges.filter(pc.greater(raw_edges['cosine_similarity'], .5))
    edges = kept.to_pylist()
    meta_by_id = {m['nature_article_id']: m for m in metadata}
    claim_by_id = {c['claim_id']: c for c in claims}
    assert len(metadata) == len(meta_by_id) == 24919
    assert len(claims) == len(claim_by_id) == 70034
    assert set(claim_by_id) == set(communities)
    assert {m['source_name'] for m in metadata} == set(JOURNALS)
    assert {m['publication_year'] for m in metadata} == set(YEARS)
    neighbors: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        a, b = e['earlier_claim_id'], e['later_claim_id']
        assert a in claim_by_id and b in claim_by_id
        assert e['earlier_paper_id'] != e['later_paper_id']
        assert e['earlier_publication_date'] < e['later_publication_date']
        assert e['earlier_publication_date'] == meta_by_id[e['earlier_paper_id']]['publication_date']
        assert e['later_publication_date'] == meta_by_id[e['later_paper_id']]['publication_date']
        assert 1 <= e['semantic_rank'] <= 10
        neighbors[b].add(a)
    assert sum(map(len, neighbors.values())) == len(edges)
    assert max(map(len, neighbors.values())) <= 10
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in claims:
        by_paper[c['parent_paper_id']].append(c)
    papers = [paper_metrics(m, by_paper[m['nature_article_id']], neighbors, communities)
              for m in sorted(metadata, key=lambda m: m['nature_article_id'])]
    summaries = summarize(papers)
    write_csv(OUT / 'data/journal_year_summary.csv', summaries)
    write_json(OUT / 'sources/source_manifest.json', SOURCES)
    qa = dict(papers=len(papers), claim_bearing_papers=sum(bool(p['n_claims']) for p in papers),
              claims=len(claims), raw_temporal_edges=raw_edges.num_rows, retained_edges=len(edges),
              graph_communities=len({c for c in communities.values() if c is not None}),
              unassigned_claims=sum(c is None for c in communities.values()),
              scatter_papers=sum(p['scatter_eligible'] for p in papers),
              distribution_papers=sum(p['shared_neighbor_fraction'] is not None for p in papers),
              no_claim_papers=[p['paper_id'] for p in papers if not p['n_claims']],
              scatter_missing_reasons=dict(Counter(r for p in papers for r in p['scatter_missing_reasons'])),
              model_calls=0, embedding_calls=0, gear_inputs_used=False,
              source_claims='abstract, one to three per paper',
              temporal_metadata='canonical_target_works; not synthetic dates in abstract claim records',
              community_scope='2023–2025 retrospective partition; assigned neighbors only')
    write_json(OUT / 'qa/data_validation.json', qa)
    result = dict(papers=papers, summaries=summaries, qa=qa)
    with gzip.open(OUT / 'data/plot_data.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',', ':'))
    print(json.dumps(qa, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    prepare()
