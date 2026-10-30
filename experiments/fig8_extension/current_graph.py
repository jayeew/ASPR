"""Fixed-reference, paper-joint local graph facts; no interpretation model calls."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from datetime import date
from itertools import combinations
import json
import math
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gear.claim_attribution import ClaimGraphRuntime
from gear.innovation.joint_graph import joint_structure
from gear.review_contracts import GraphClaim, GraphFactCard, GraphNeighbor
from gear.trace import EvidenceStore
from .current_extract import OUT as CLAIMS, PROTOCOL, SUCCESS, write_json
from .current_metadata import write_csv
from .supplement_queue import ROOT

OUT=ROOT/'analysis/graph'
HISTORY=Path('data/claim_graph')
MODEL=Path('data/models/Qwen3-Embedding-4B')


def efficiency(nodes: set[str], edges: list[list[str]], historical: set[str]) -> float|None:
    if len(historical)<2:
        return None
    adj={n:set() for n in nodes}
    for a,b in edges:
        adj[a].add(b);adj[b].add(a)
    total=0.
    for start in historical:
        distances={start:0};queue=deque([start])
        while queue:
            node=queue.popleft()
            for other in adj[node]-distances.keys():
                distances[other]=distances[node]+1;queue.append(other)
        total+=sum(1/d for n,d in distances.items() if n in historical and d>0)
    return total/(len(historical)*(len(historical)-1))


def joint_metrics(cards: list[GraphFactCard], edges: list[list[str]]) -> tuple[dict,dict]:
    facts=joint_structure(cards,edges)
    neighborhoods=[{n.claim_id for n in c.neighbors} for c in cards]
    historical=set().union(*neighborhoods) if neighborhoods else set()
    target={c.claim.claim_id for c in cards}
    n=len(historical);matched=sum(bool(s) for s in neighborhoods)
    before=efficiency(historical,edges,historical)
    after=efficiency(historical|target,edges+facts['insertion_edges'],historical)
    nonempty=[s for s in neighborhoods if s]
    overlaps=[len(a&b)/len(a|b) for a,b in combinations(nonempty,2)]
    communities={n.community_id for c in cards for n in c.neighbors if n.community_id is not None}
    result={'claim_count':len(cards),'matched_claim_count':matched,
        'neighbor_match_coverage':matched/len(cards) if cards else None,
        'historical_neighbor_count':n,'insertion_edge_count':len(facts['insertion_edges']),
        'historical_edge_count':len(edges),'component_merge_count':facts['joint_component_merge_count'],
        'newly_connected_pairs':facts['joint_newly_connected_historical_pairs'],
        'newly_connected_pair_ratio':facts['joint_newly_connected_historical_pairs']/math.comb(n,2) if n>=2 else None,
        'efficiency_before':before,'efficiency_after':after,
        'efficiency_change':after-before if before is not None else None,
        'community_count':len(communities),'neighbor_jaccard_mean':float(np.mean(overlaps)) if overlaps else None,
        'matched_claim_bin':'0' if not matched else '1' if matched==1 else '2' if matched==2 else '3-4' if matched<=4 else '>=5',
        'neighbor_bin':'0' if not n else '1-10' if n<=10 else '11-20' if n<=20 else '>20',
        'graph_status':'no_contributions' if not cards else 'no_neighbors' if not n else 'insufficient_pairs' if n<2 else 'computed'}
    return result,facts


def embeddings(runtime: ClaimGraphRuntime, claims: list[dict]) -> tuple[np.ndarray,dict[str,int]]:
    path=OUT/'claim_vectors.npy';index_path=OUT/'claim_vector_index.parquet'
    old=pq.read_table(index_path).to_pylist() if index_path.exists() else []
    index={r['claim_id']:r['row'] for r in old}
    missing=[c for c in claims if c['claim_id'] not in index]
    if missing:
        import torch
        try:
            new=runtime.encode_batch([c['claim_text'] for c in missing],batch_size=16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new=runtime.encode_batch([c['claim_text'] for c in missing],batch_size=8)
        existing=np.load(path) if old else np.empty((0,new.shape[1]),np.float32)
        matrix=np.concatenate([existing,new],axis=0)
        old.extend({'claim_id':c['claim_id'],'row':len(index)+i} for i,c in enumerate(missing))
        np.save(path,matrix)
        pq.write_table(pa.Table.from_pylist(old),index_path,compression='zstd')
        index={r['claim_id']:r['row'] for r in old}
    return np.load(path,mmap_mode='r'),index


def batch_search(vectors: np.ndarray, vector_index: dict[str,int], claims: list[dict]) -> dict:
    """Batch exact cosine search on GPU; the historical matrix is reused unchanged."""
    import torch

    device='cuda' if torch.cuda.is_available() else 'cpu'
    history=torch.tensor(np.load(HISTORY/'claim_embedding_matrix.npy',mmap_mode='r'),device=device)
    output={}
    for start in range(0,len(claims),128):
        batch=claims[start:start+128]
        query=torch.tensor(np.asarray([vectors[vector_index[c['claim_id']]] for c in batch]),device=device)
        scores,indices=torch.topk(query@history.T,min(64,len(history)),dim=1)
        scores=scores.cpu().numpy();indices=indices.cpu().numpy()
        for i,c in enumerate(batch):
            output[c['claim_id']]=(scores[i],indices[i])
    del history
    if device=='cuda':
        torch.cuda.empty_cache()
    return output


def neighbors(runtime: ClaimGraphRuntime, vector: np.ndarray, own: str,
              history: dict[str,dict], row_ids: list[str], communities: dict,
              search_result: tuple[np.ndarray,np.ndarray]) -> list[GraphNeighbor]:
    scores,indices=search_result
    requested=len(indices)
    while True:
        eligible=[(float(s),int(i)) for s,i in zip(scores,indices) if i>=0 and float(s)>.5
                  and history[row_ids[int(i)]]['parent_paper_id']!=own]
        if len(eligible)>=10 or requested>=len(row_ids) or float(scores[-1])<=.5:
            break
        requested=min(requested*2,len(row_ids))
        scores,indices=runtime._semantic_search(vector,requested)
    output=[]
    for rank,(score,i) in enumerate(eligible[:10],1):
        node=history[row_ids[i]]
        output.append(GraphNeighbor(claim_id=node['claim_id'],parent_paper_id=node['parent_paper_id'],
            claim_text=node['claim_text'],claim_type=node['claim_type'],
            publication_date=date.fromisoformat(str(node['publication_date'])[:10]),
            cosine_similarity=score,semantic_rank=rank,community_id=communities.get(node['claim_id'])))
    return output


def distribution_rows(papers: list[dict]) -> list[dict]:
    groups=defaultdict(list)
    for p in papers:
        groups[(p['journal'],p['field'],p['online_year'],'all','all')].append(p)
        groups[(p['journal'],p['field'],'all','all','all')].append(p)
        groups[(p['journal'],p['field'],p['online_year'],p['matched_claim_bin'],p['neighbor_bin'])].append(p)
    output=[]
    for (journal,field,year,cb,hb),rows in sorted(groups.items(),key=lambda x:str(x[0])):
        done=sum(r['extraction_status'] in SUCCESS for r in rows)
        valid=[r for r in rows if r['graph_status']=='computed']
        for metric in ['newly_connected_pair_ratio','efficiency_change','component_merge_count','community_count','neighbor_jaccard_mean']:
            values=[r[metric] for r in valid if r.get(metric) is not None]
            output.append({'journal':journal,'field':field,'online_year':year,
                'matched_claim_bin':cb,'neighbor_bin':hb,'metric':metric,
                'paper_n':len(rows),'extracted_n':done,'extraction_completion_rate':done/len(rows),
                'computable_n':len(values),'median':float(np.median(values)) if values else None,
                'q25':float(np.quantile(values,.25)) if values else None,
                'q75':float(np.quantile(values,.75)) if values else None,
                'distribution_ready':done/len(rows)>=.95 and len(values)>=30,
                'scope':'observed_current_cohort_fixed_reference'})
    return output


def run(exported_dois: set[str]|None = None) -> dict[str,Any]:
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    claims=pq.read_table(CLAIMS/'claims.parquet').to_pylist()
    selection=pq.read_table(ROOT/'analysis/cohort/selection.parquet',filters=[('included','=',True)]).to_pylist()
    from .current_extract import JOURNALS
    targets={r['work_id']:r for r in selection if r['source_id'] in JOURNALS}
    db=sqlite3.connect(CLAIMS/'contributions.sqlite')
    results={r[0]:{'status':r[1],'error':r[2]} for r in db.execute('SELECT doi,status,error FROM results WHERE protocol_hash=?',(PROTOCOL,))}
    db.close()
    if exported_dois is not None:
        results={doi:r for doi,r in results.items() if doi in exported_dois or r['status'] not in SUCCESS}
    previous={}
    metric_path=OUT/'paper_metrics.parquet'
    if metric_path.exists():
        previous={r['work_id']:r for r in pq.read_table(metric_path).to_pylist()
                  if r['extraction_status'] in SUCCESS}
    by_paper=defaultdict(list)
    for c in claims:
        by_paper[c['work_id']].append(c)
    runtime=ClaimGraphRuntime(HISTORY,MODEL,10,.5)
    pending_claims=[c for c in claims if c['work_id'] not in previous]
    if pending_claims:
        vectors,vector_index=embeddings(runtime,claims)
        runtime.close()
        search_results=batch_search(vectors,vector_index,pending_claims)
        history={r['claim_id']:r for r in pq.read_table(HISTORY/'claim_nodes.parquet').to_pylist()}
        row_ids=[r['claim_id'] for r in pq.read_table(HISTORY/'claim_embedding_index.parquet').sort_by('claim_row').to_pylist()]
        communities={r['claim_id']:r['community_id'] for r in pq.read_table(HISTORY/'claim_communities.parquet').to_pylist()}
        canonical={r['doi'].lower():r['nature_article_id'] for r in pq.read_table(HISTORY/'canonical_target_works.parquet',columns=['doi','nature_article_id']).to_pylist() if r['doi']}
        adj=defaultdict(set)
        for e in pq.read_table(HISTORY/'claim_backbone_edges.parquet').to_pylist():
            a,b=e['claim_id_a'],e['claim_id_b'];adj[a].add(b);adj[b].add(a)
    store=EvidenceStore(OUT)
    papers=[];edge_rows=[];neighborhoods=[]
    if previous:
        edge_rows=[r for r in pq.read_table(OUT/'claim_neighbors.parquet').to_pylist() if r['work_id'] in previous]
        neighborhoods=[r for r in pq.read_table(OUT/'paper_neighborhoods.parquet').to_pylist() if r['work_id'] in previous]
    for work_id,p in sorted(targets.items()):
        if work_id in previous:
            papers.append(previous[work_id])
            continue
        cards=[]
        for c in by_paper[work_id]:
            ns=neighbors(runtime,vectors[vector_index[c['claim_id']]],canonical.get(p['doi'].lower(),work_id),
                         history,row_ids,communities,search_results[c['claim_id']])
            cards.append(GraphFactCard(claim=GraphClaim(claim_id=c['claim_id'],paper_id=work_id,
                claim_type=c['claim_type'],claim_text=c['claim_text'],source_sentence_ids=c['source_sentence_ids'],
                source_sentence_texts=c['source_fragments']),neighbors=ns,metrics=[],
                insertion_policy='fixed_reference:k=10:cosine>0.5'))
            edge_rows.extend({'work_id':work_id,'claim_id':c['claim_id'],'neighbor_id':n.claim_id,
                'cosine_similarity':n.cosine_similarity,'community_id':n.community_id,'semantic_rank':n.semantic_rank} for n in ns)
        union={n.claim_id for c in cards for n in c.neighbors}
        edges=sorted([a,b] for a in union for b in adj[a]&union if a<b) if cards else []
        metrics,facts=joint_metrics(cards,edges)
        extraction=results.get(p['doi'],{'status':'pending','error':''})
        if extraction['status'] not in SUCCESS:
            metrics['graph_status']='extraction_unresolved'
        evidence_key='joint_graph:'+work_id
        if extraction['status'] in SUCCESS:
            store.add_evidence(evidence_key,'joint_graph_structure',{'work_id':work_id,
                'claims':[c.model_dump(mode='json') for c in cards],'facts':facts,'metrics':metrics,
                'reference':'data/claim_graph','time_scope':'fixed_2023_2025_retrospective'})
        papers.append({'work_id':work_id,'doi':p['doi'],'journal':JOURNALS[p['source_id']],
            'field':p['field'],'online_year':p['online_year'],'extraction_status':extraction['status'],
            'extraction_error':extraction['error'],'evidence_key':evidence_key if extraction['status'] in SUCCESS else None,**metrics})
        neighborhoods.append({'work_id':work_id,'historical_edges':edges,'insertion_edges':facts['insertion_edges']})
    runtime.close()
    pq.write_table(pa.Table.from_pylist(papers),OUT/'paper_metrics.parquet',compression='zstd')
    es=pa.schema([('work_id',pa.string()),('claim_id',pa.string()),('neighbor_id',pa.string()),
                 ('cosine_similarity',pa.float64()),('community_id',pa.int64()),('semantic_rank',pa.int64())])
    pq.write_table(pa.Table.from_pylist(edge_rows,schema=es),OUT/'claim_neighbors.parquet',compression='zstd')
    ns=pa.schema([('work_id',pa.string()),('historical_edges',pa.list_(pa.list_(pa.string()))),
                  ('insertion_edges',pa.list_(pa.list_(pa.string())))])
    pq.write_table(pa.Table.from_pylist(neighborhoods,schema=ns),OUT/'paper_neighborhoods.parquet',compression='zstd')
    tables=ROOT/'analysis/tables';tables.mkdir(parents=True,exist_ok=True)
    write_csv(tables/'panel_a_papers.csv',[p for p in papers if p['journal']=='Nature'])
    write_csv(tables/'panel_b_four_journal_papers.csv',papers)
    write_csv(tables/'panel_b_graph_distributions.csv',distribution_rows(papers))
    available=sorted([p for p in papers if p['journal']=='Nature' and p['graph_status']=='computed'],
                     key=lambda p:(p['newly_connected_pair_ratio'],p['efficiency_change'],p['work_id']))
    examples=[]
    if available:
        for label,q in [('low',.1),('middle',.5),('high',.9)]:
            p=available[round(q*(len(available)-1))]
            examples.append({'position':label,**p})
    write_json(OUT/'nature_network_examples.json',examples)
    summary={'target_papers':len(papers),'new_claims':len(claims),
        'new_papers_computed_this_pass':sum(p['extraction_status'] in SUCCESS and p['work_id'] not in previous for p in papers),
        'current_run_papers_reused':len(previous),
        'graph_status':dict(Counter(p['graph_status'] for p in papers)),
        'journal_status':[{'journal':j,'papers':sum(p['journal']==j for p in papers),
            'extracted':sum(p['journal']==j and p['extraction_status'] in SUCCESS for p in papers),
            'computable':sum(p['journal']==j and p['graph_status']=='computed' for p in papers)} for j in JOURNALS.values()],
        'reference':'fixed 2023–2025 graph, retrospective local structure; not historical novelty',
        'reference_source_limitation':'Approximately 97% of reference papers are Nature Communications.',
        'extra_generation_calls':0,'elapsed_seconds':round(time.monotonic()-started,1)}
    write_json(OUT/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    run()
