"""2026 manuscript claims linked to the read-only 2023–2025 graph."""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .data import ASSETS, ROOT, compact, read, write

OUT = ROOT / 'outputs/fig7_reference/frontier_2026'
STUDIES = [ROOT/'outputs'/s for s in ('innovation_200_20260907', 'innovation_candidates_20260910')]
MODEL = ROOT/'data/models/Qwen3-Embedding-4B'


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inventory() -> list[dict[str, Any]]:
    existing = {r['paper_id']: (r, p/'papers'/r['paper_id']) for p in STUDIES
                for r in jsonl(p/'papers.jsonl')}
    rows = []
    for m in jsonl(ROOT/'data/nature_2026_testset/manifest.jsonl'):
        ident = m['article_id']
        metadata, source = existing.get(ident, ({}, OUT/'papers'/ident))
        own = OUT/'papers'/ident
        shared = source/'shared'
        if (own/'shared/claims.json').exists():
            shared = own/'shared'
        claims = read(shared/'claims.json')['claims'] if (shared/'claims.json').exists() else []
        graph = source/'graph'
        if (own/'graph').exists():
            graph = own/'graph'
        facts = sum((graph/c['claim_id'].rsplit('::', 1)[-1]/'evidence_trace.jsonl').exists() for c in claims)
        rows.append({'paper_id': ident, 'title': m['title'], 'doi': m['doi'],
                     'date': m['publication_date'], 'paper_path': m['paper_markdown_path'],
                     'venue': m['journal_name'], 'field': metadata.get('field_name', 'Unknown'),
                     'metadata': metadata, 'shared': str(shared), 'graph': str(graph),
                     'claim_count': len(claims), 'graph_fact_count': facts,
                     'unsupported_claim_count': sum(c['internal_support']=='internally_unsupported' for c in claims)})
    if len({r['paper_id'] for r in rows}) != 1000:
        raise ValueError('Expected the original 1,000 unique paper IDs')
    write(OUT/'inventory.json', rows)
    return rows


def item_for(row: dict[str, Any]) -> Any:
    from gear.review_contracts import InnovationPaperInput
    m = row['metadata']
    published = date.fromisoformat(row['date'])
    return InnovationPaperInput(paper_id=row['paper_id'], title=row['title'],
        paper_path=Path(row['paper_path']), doi=row['doi'], venue=row['venue'],
        publication_date=published, cutoff_date=published,
        abstract_text=m.get('abstract_text', ''), abstract_source=m.get('field_source', 'not_available'),
        authors=m.get('authors', []), openalex_work_id=m.get('openalex_work_id'),
        reference_work_ids=m.get('reference_work_ids', []))


def pilot_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Span claim counts and fields; reused papers fill one production-sized packet."""
    path = OUT/'pilot_selection.json'
    if path.exists():
        return read(path)
    pending = sorted([r for r in rows if r['claim_count'] and not r['graph_fact_count']],
                     key=lambda r: (r['claim_count'], r['field'], r['paper_id']))
    new = [pending[round(i*(len(pending)-1)/11)]['paper_id'] for i in range(12)]
    reused = [r['paper_id'] for r in rows if r['graph_fact_count']==r['claim_count'] and r['claim_count']]
    random.Random(20261003).shuffle(reused)
    missing = [r['paper_id'] for r in rows if not r['claim_count']]
    result = {'new_graph_paper_ids': new, 'extract_paper_ids': missing[:1],
              'forecast_paper_ids': new + reused[:112] + missing[:1],
              'description': '12 new-graph papers + 1 missing-claims paper + 112 reused papers; one 125-paper packet'}
    write(path, result)
    return result


def extract_missing(rows: list[dict[str, Any]]) -> None:
    from experiments.innovation_200.common import experiment_config, setup_stage_logging
    from gear.innovation.shared import prepare_shared
    from gear.innovation.usage import progress_logging, usage_log
    config = experiment_config()
    config.role_effort_overrides = {key: 'low' for key in config.role_effort_overrides}
    config.codex_cli.executable = str(ROOT/'figure_pipeline/fig4_explanation_study/codex_session')
    logger = setup_stage_logging(OUT, 'extract_missing')
    for row in rows:
        if row['claim_count']:
            continue
        started = time.monotonic()
        path = OUT/'timings'/f'extract_{row["paper_id"]}.json'
        with usage_log(OUT/'logs/claim_usage'/f'{row["paper_id"]}.jsonl'), progress_logging(logger, row['paper_id']):
            _, claims = prepare_shared(item_for(row), OUT/'papers'/row['paper_id'], config)
        write(path, {'seconds': time.monotonic()-started, 'paper_id': row['paper_id'],
                     'claims': len(claims.claims), 'model': config.role_model_override, 'effort': 'low'})


def claim_objects(row: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from gear.contracts import PaperIR
    from gear.innovation.contracts import ClaimSet
    shared = Path(row['shared'])
    claims = ClaimSet.model_validate(read(shared/'claims.json'))
    paper = PaperIR.model_validate(read(shared/'paper_ir.json'))
    return claims, {s.span_id: s for s in paper.spans}


def fact_card(graph: Path, claim_id: str) -> Any | None:
    from gear.review_contracts import GraphFactCard
    from gear.trace import EvidenceStore
    directory = graph/claim_id.rsplit('::', 1)[-1]
    if not (directory/'evidence_trace.jsonl').exists():
        return None
    record = EvidenceStore(directory)._evidence.get('GRAPH:'+claim_id)
    return GraphFactCard.model_validate(record.payload) if record else None


def graph_prepare(rows: list[dict[str, Any]], batch_size: int = 16) -> None:
    from gear.claim_attribution import ClaimGraphRuntime
    from gear.innovation.joint_graph import joint_structure
    from gear.review_contracts import GraphClaim
    from gear.trace import EvidenceStore
    runtime = ClaimGraphRuntime(ASSETS, MODEL)
    pending, cards_by_paper = [], defaultdict(list)
    started = time.monotonic()
    for row in rows:
        claims, spans = claim_objects(row)
        for c in claims.claims:
            target = GraphClaim(claim_id=c.claim_id, paper_id=row['paper_id'], claim_type=c.claim_type,
                claim_text=c.normalized_claim_text, source_sentence_ids=c.source_span_ids,
                source_sentence_texts=[spans[k].text for k in c.source_span_ids])
            card = fact_card(Path(row['graph']), c.claim_id)
            if card:
                if card.claim != target or card.insertion_policy != runtime.insertion_policy:
                    raise ValueError('Saved graph claim/policy mismatch: '+c.claim_id)
                cards_by_paper[row['paper_id']].append(card)
            else:
                pending.append((row, target))
    timings = {'requested_papers': len(rows), 'new_claims': len(pending), 'batches': [], 'joint_seconds': 0.0}
    try:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset+batch_size]
            start = time.monotonic()
            vectors = runtime.encode_batch([c.claim_text for _, c in batch], batch_size)
            embedding_seconds = time.monotonic()-start
            start = time.monotonic()
            for (row, claim), vector in zip(batch, vectors, strict=True):
                card = runtime.insert_vector(claim, item_for(row), vector)
                cards_by_paper[row['paper_id']].append(card)
                target = OUT/'papers'/row['paper_id']/'graph'/claim.claim_id.rsplit('::', 1)[-1]
                EvidenceStore(target).add_evidence('GRAPH:'+claim.claim_id, 'graph_fact', card)
            result = {'claims': len(batch), 'characters': sum(len(c.claim_text) for _, c in batch),
                      'embedding_seconds': embedding_seconds, 'facts_seconds': time.monotonic()-start}
            timings['batches'].append(result)
            write(OUT/'timings/graph_latest.json', timings)
            print('Graph batch', offset+len(batch), '/', len(pending), compact(result), flush=True)
        start = time.monotonic()
        for row in rows:
            target = OUT/'papers'/row['paper_id']/'graph/joint/facts.json'
            if target.exists() or (Path(row['graph'])/'joint/facts.json').exists():
                continue
            cards = cards_by_paper[row['paper_id']]
            neighbors = {n.claim_id: n for card in cards for n in card.neighbors}
            edges = []
            if neighbors:
                runtime._connections()
                mapping = {int(runtime._claim_db.execute('SELECT claim_row FROM claim_nodes WHERE claim_id=?',
                           (n,)).fetchone()[0]): n for n in neighbors}
                edges = [[mapping[a], mapping[b]] for a,b in sorted(runtime._neighbor_edges(list(neighbors.values())))]
            fact = joint_structure(cards, edges)
            EvidenceStore(target.parent).add_evidence('JOINT_GRAPH:'+row['paper_id'], 'joint_graph_fact', fact)
            write(target, fact)
        timings['joint_seconds'] = time.monotonic()-start
    finally:
        runtime.close()
        timings['total_seconds'] = time.monotonic()-started
        stamp = time.time_ns()
        write(OUT/f'timings/graph_{stamp}.json', timings)
        write(OUT/'timings/graph_latest.json', timings)


def build_material(row: dict[str, Any]) -> dict[str, Any]:
    claims, spans = claim_objects(row)
    evidence, graph, omitted = {}, {}, []
    for claim in claims.claims:
        if claim.internal_support.value == 'internally_unsupported':
            omitted.append(claim.claim_id)
            continue
        card = fact_card(Path(row['graph']), claim.claim_id)
        if not card or card.claim.claim_text != claim.normalized_claim_text:
            raise ValueError('Missing or mismatched Graph fact: '+claim.claim_id)
        target = claim.claim_id
        evidence[target] = {'paper_id': row['paper_id'], 'claim': claim.normalized_claim_text,
            'type': claim.claim_type.value, 'date': row['date'], 'doi': row['doi'],
            'provenance': '2026_fulltext', 'internal_support': claim.internal_support.value,
            'narrowing_reason': claim.narrowing_reason,
            'source_spans': [{'id': k, 'text': spans[k].text} for k in claim.support_span_ids]}
        edges = []
        for neighbor in card.neighbors:
            evidence[neighbor.claim_id] = {'paper_id': neighbor.parent_paper_id, 'claim': neighbor.claim_text,
                'type': neighbor.claim_type.value, 'date': neighbor.publication_date.isoformat(),
                'provenance': '2023_2025_abstract', 'community_id': neighbor.community_id}
            edges.append({'target': target, 'historical': neighbor.claim_id,
                          'cosine': neighbor.cosine_similarity, 'community_id': neighbor.community_id,
                          'direct_citation': neighbor.direct_citation,
                          'two_hop_paths': neighbor.two_hop_path_count,
                          'shared_references': neighbor.shared_reference_count})
        graph['GRAPH:'+target] = {'target': target, 'edges': edges,
            'neighbor_edges': card.neighbor_edges, 'metrics': [m.model_dump(mode='json') for m in card.metrics]}
    # Full evidence remains on disk; packet text is compacted separately, never truncated mid-claim.
    return {'paper_id': row['paper_id'], 'title': row['title'], 'date': row['date'], 'field': row['field'],
            'evidence': evidence, 'graph': graph, 'unsupported_claim_ids': omitted,
            'citation_metadata_available': bool(row['metadata'])}


def compact_material(material: dict[str, Any], representatives: int = 1) -> dict[str, Any]:
    """All target claims and neighbor IDs participate; representative historical originals are included."""
    graph = material['graph']
    kept = {v['target'] for v in graph.values()}
    for fact in graph.values():
        # Top similarity plus a different community where available, then diversity in source papers.
        edges = sorted(fact['edges'], key=lambda e: -e['cosine'])
        choices = []
        for e in edges:
            if len(choices) < representatives and (not choices or e['community_id'] != choices[0]['community_id']):
                choices.append(e)
        for e in edges:
            if len(choices) >= representatives:
                break
            if e not in choices:
                choices.append(e)
        kept.update(e['historical'] for e in choices)
    evidence = {k: {a:b for a,b in v.items() if a not in ('source_spans', 'narrowing_reason')}
                for k,v in material['evidence'].items() if k in kept}
    facts = {k: {'target': v['target'], 'neighbor_count': len(v['edges']),
                'edges': v['edges'], 'neighbor_edges': v['neighbor_edges'],
                'metrics': {m['name']:m['value'] for m in v['metrics'] if m['value'] is not None}}
             for k,v in graph.items()}
    return {k: material[k] for k in ('paper_id','title','date','field','unsupported_claim_ids','citation_metadata_available')} | {
        'evidence': evidence, 'graph': facts}


def prepare_packets(rows: list[dict[str, Any]], destination: Path, count: int) -> list[dict[str, Any]]:
    materials = [build_material(r) for r in rows]
    for m in materials:
        write(OUT/'materials'/f'{m["paper_id"]}.json', m)
    compacted = [compact_material(m) for m in materials]
    bins: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    sizes = [0]*count
    # Deterministic field ordering with length-balanced allocation across packets.
    for m in sorted(compacted, key=lambda m: (m['field'], m['paper_id'])):
        index = min(range(count), key=lambda i: sizes[i])
        bins[index].append(m)
        sizes[index] += len(compact(m))
    packets = []
    for i, group in enumerate(bins, 1):
        packet = {'packet_id': f'P{i:02d}', 'scope': '2026 papers through 2026-05-08 with 2023–2025 historical graph',
            'target_papers': [{k:v for k,v in m.items() if k not in ('evidence','graph')} for m in group],
            'evidence': {k:v for m in group for k,v in m['evidence'].items()},
            'graph': {k:v for m in group for k,v in m['graph'].items()},
            'limitations': ['Historical neighbors are abstract claims; target claims are full-text grounded.',
                'No 2026-to-2026 semantic edges have been computed. Cross-paper combinations are hypotheses.',
                'All target claims represented except explicitly internally unsupported claims.',
                'All eligible neighbor IDs/edges retained; one representative historical original per target; full originals retained on disk.']}
        packet = encode_packet(packet, destination)
        if len(compact(packet)) > 900000:
            raise ValueError('Packet exceeds configured input size; discuss compression before more calls')
        write(destination/'packets'/f'P{i:02d}.json', packet)
        packets.append(packet)
    covered = [p['paper_id'] for x in packets for p in x['target_papers']]
    assert len(covered)==len(set(covered))==len(rows)
    write(destination/'coverage.json', {'paper_ids': sorted(covered), 'paper_count':len(rows),
        'full_1000':len(rows)==1000,'packet_characters':[len(compact(p)) for p in packets],
        'unsupported_claim_ids':[c for m in materials for c in m['unsupported_claim_ids']],
        'target_claim_count':sum(len(m['graph']) for m in materials)})
    return packets


def encode_packet(packet: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Dictionary-code repeated IDs/column names; retain every target and actual edge."""
    all_ids = set(packet['evidence'])
    for fact in packet['graph'].values():
        all_ids.update(e['historical'] for e in fact['edges'])
    codes = {key: f'S{i:05d}' for i,key in enumerate(sorted(all_ids))}
    originals = {codes[k]: {'claim_id': k, **v} for k,v in packet['evidence'].items()}
    write(destination/'sources'/f'{packet["packet_id"]}.json', originals)
    columns = ['id','paper_id','date','type','claim','provenance','internal_support','community_id']
    evidence = [[code]+[v.get(k) for k in columns[1:]] for code,v in originals.items()]
    metric_columns = sorted({k for f in packet['graph'].values() for k in f['metrics']})
    graph = []
    for key, fact in packet['graph'].items():
        graph.append([f'G{codes[fact["target"]][1:]}', codes[fact['target']],
            [[codes[e['historical']], round(e['cosine'],4), e['community_id'], e['direct_citation'],
              e['two_hop_paths'],e['shared_references']] for e in fact['edges']],
            len(fact['neighbor_edges']),
            [round(fact['metrics'][k],4) if isinstance(fact['metrics'].get(k),float) else fact['metrics'].get(k) for k in metric_columns]])
    return {k:v for k,v in packet.items() if k not in ('evidence','graph')} | {
        'evidence_columns':columns, 'evidence':evidence,
        'graph_columns':['id','target','neighbors','historical_neighbor_edge_count','metrics'],
        'metric_columns':metric_columns,
        'neighbor_columns':['id','cosine','community_id','direct_citation','two_hop_paths','shared_references'],
        'graph':graph}
