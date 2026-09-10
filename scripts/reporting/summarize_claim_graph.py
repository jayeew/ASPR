"""Export aggregate-only native historical Claim Graph statistics."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/claim_graph'
OUT = ROOT / 'outputs/GRAPH_STATISTICS'


def numeric(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    v = x.dropna()
    return {'total': len(x), 'valid': len(v), 'missing_or_nonfinite': int(x.isna().sum()),
            'zero_count': int((v == 0).sum()), 'one_count': int((v == 1).sum()),
            **({k: float(z) for k, z in zip(['min','p01','p05','p25','median','p75','p95','p99','max'], v.quantile([0,.01,.05,.25,.5,.75,.95,.99,1]))} if len(v) else {}),
            'mean': float(v.mean()) if len(v) else None, 'std_population': float(v.std(ddof=0)) if len(v) else None}


def counts(values: pd.Series) -> dict:
    return {str(k): int(v) for k, v in values.fillna('MISSING').astype(str).value_counts().sort_index().items()}


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(SOURCE / (name + '.parquet'))


def metric_group(name: str) -> str:
    if 'paper_' in name or 'agreement' in name: return 'paper_path_annotation'
    if any(x in name for x in ['density','components','component_merge','connected','boundary']): return 'local_connectivity'
    if any(x in name for x in ['surprisal','pair','recent_nature']): return 'community_pair_history'
    if 'community' in name: return 'community_composition'
    if 'type' in name: return 'contribution_role_composition'
    return 'semantic_neighbor_selection'


def edge_stats(name: str, nodes: pd.DataFrame) -> dict:
    d = read(name)
    backbone = name == 'claim_backbone_edges'
    a,b = ('claim_id_a','claim_id_b') if backbone else ('earlier_claim_id','later_claim_id')
    known = set(nodes.claim_id)
    pairs = d[[a,b]].copy()
    g = nx.Graph(); g.add_nodes_from(known); g.add_edges_from(pairs.itertuples(index=False, name=None))
    component_sizes = pd.Series([len(c) for c in nx.connected_components(g)])
    result = {'row_count':len(d), 'orientation':'undirected semantic backbone' if backbone else 'earlier claim → later claim',
              'duplicate_ordered_pairs':int(pairs.duplicated().sum()), 'self_loops':int((d[a]==d[b]).sum()),
              'unknown_endpoint_rows':int((~d[a].isin(known)|~d[b].isin(known)).sum()),
              'unique_undirected_edges':g.number_of_edges(), 'undirected_density':nx.density(g),
              'undirected_degree':numeric(pd.Series(dict(g.degree()).values())),
              'isolates':nx.number_of_isolates(g),'connected_components':len(component_sizes),
              'component_size_summary':numeric(component_sizes),'component_size_histogram':counts(component_sizes),
              'cosine_similarity':numeric(d.cosine_similarity)}
    if backbone:
        result['transitivity'] = nx.transitivity(g)
        result['mean_local_clustering_including_isolates'] = nx.average_clustering(g)
    else:
        result['same_parent_paper_rows'] = int((d.earlier_paper_id==d.later_paper_id).sum())
        result['non_strict_temporal_rows'] = int((pd.to_datetime(d.earlier_publication_date)>=pd.to_datetime(d.later_publication_date)).sum())
        result['role_pairs'] = d.groupby(['earlier_claim_type','later_claim_type']).size().rename('count').reset_index().to_dict('records')
        result['flag_counts'] = {c:counts(d[c]) for c in ['from_semantic','from_paper_path','is_cross_type','paper_direct_citation']}
        result['numeric_annotations'] = {c:numeric(d[c]) for c in d.select_dtypes(include='number')}
        result['earlier_endpoint_out_degree'] = numeric(d[a].value_counts().reindex(nodes.claim_id, fill_value=0))
        result['later_endpoint_in_degree'] = numeric(d[b].value_counts().reindex(nodes.claim_id, fill_value=0))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = read('claim_nodes'); communities = read('claim_communities'); profiles = read('historical_insertion_profiles')
    dates = pd.to_datetime(nodes.publication_date, errors='coerce')
    node_stats = {'rows':len(nodes),'unique_claims':nodes.claim_id.nunique(),'duplicate_claim_keys':int(nodes.claim_id.duplicated().sum()),
                  'parent_papers':nodes.parent_paper_id.nunique(),'role_type_count':nodes.claim_type.nunique(), 'role_counts':counts(nodes.claim_type),
                  'year_counts':counts(dates.dt.year),'month_counts':counts(dates.dt.strftime('%Y-%m')),
                  'min_publication_date':str(dates.min().date()),'max_publication_date':str(dates.max().date()),
                  'claims_per_parent_paper':numeric(nodes.groupby('parent_paper_id').size()),
                  'claims_per_parent_paper_histogram':counts(nodes.groupby('parent_paper_id').size()),
                  'null_counts':{c:int(nodes[c].isna().sum()) for c in nodes},
                  'claim_text_character_lengths':numeric(nodes.claim_text.str.len()),
                  'source_sentence_count':numeric(nodes.source_sentence_ids.map(len))}
    cs = communities.community_id.value_counts()
    community_stats = {'assignment_rows':len(communities),'assigned_claims':int(communities.community_id.notna().sum()),
                       'unassigned_claims':int(communities.community_id.isna().sum()),'community_count':len(cs),
                       'duplicate_claim_keys':int(communities.claim_id.duplicated().sum()),
                       'community_size':numeric(cs), 'community_size_histogram':counts(cs)}
    numeric_cols = [c for c in profiles.select_dtypes(include='number') if c!='dominant_community_id']
    metrics = {c:{'group':metric_group(c),'overall':numeric(profiles[c]),
                  'by_role':{str(k):numeric(v[c]) for k,v in profiles.groupby('claim_type')}} for c in numeric_cols}
    inventory = []
    for path in sorted(SOURCE.iterdir()):
        if not path.is_file(): continue
        item = {'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size,
                'mtime_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()}
        if path.suffix=='.parquet':
            pf = pq.ParquetFile(path); item.update(rows=pf.metadata.num_rows, fields=[{'name':f.name,'type':str(f.type)} for f in pf.schema_arrow])
        inventory.append(item)
    arrays = {}
    for name in ['claim_embedding_matrix.npy','community_centroid_matrix.npy']:
        arr=np.load(SOURCE/name,mmap_mode='r'); norms=np.linalg.norm(arr,axis=1)
        arrays[name]={'shape':list(arr.shape),'dtype':str(arr.dtype),'nonfinite_value_count':int((~np.isfinite(arr)).sum()),'row_l2_norm':numeric(pd.Series(norms))}
    supplementary = {}
    for name in ['community_history_counts','community_pair_history','community_profiles']:
        d=read(name); supplementary[name]={'rows':len(d),'metrics':{c:numeric(d[c]) for c in d.select_dtypes(include='number') if c not in ['community_id','community_a','community_b']}}
    summaries=read('historical_metric_summary'); pct=read('historical_metric_percentiles')
    result={'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'scope':'native historical Claim Graph assets; aggregates only; no target insertions added',
            'statistical_conventions':{'quantiles':'pandas linear interpolation','std':'population ddof=0','topology':'all historical nodes included; undirected projection for components and density; no sampled topology',
            'historical_calibration_warning':'Historical insertion profiles and percentiles were generated under an older neighbor-selection policy. They are archived structural descriptions, not valid calibration for current cosine >0.5, top-k<=10 runtime. Fixed full-graph communities are unsuitable for strict earlier-cutoff validation.'},
            'nodes':node_stats,'communities':community_stats,'edges':{},'historical_insertion_metrics':metrics,
            'historical_metric_groups':{k:[c for c in numeric_cols if metric_group(c)==k] for k in sorted({metric_group(c) for c in numeric_cols})},
            'historical_calibration':{'summary_rows':len(summaries),'percentile_rows':len(pct),'metric_names':sorted(summaries.metric_name.unique()),'reference_scopes':counts(summaries.reference_scope),'saved_summary_records':json.loads(summaries.to_json(orient='records'))},
            'community_history':supplementary,'embedding_assets':arrays,'source_inventory':inventory}
    for name in ['claim_edges','semantic_claim_edges','paper_path_claim_edges','claim_backbone_edges']:
        print('Computing', name, flush=True); result['edges'][name]=edge_stats(name,nodes)
    semantic=read('semantic_claim_edges'); combined=read('claim_edges'); path=read('paper_path_claim_edges')
    key=['earlier_claim_id','later_claim_id']
    s=set(semantic[key].itertuples(index=False,name=None)); p=set(path[key].itertuples(index=False,name=None)); c=set(combined[key].itertuples(index=False,name=None))
    result['edge_family_overlap']={'semantic_and_path':len(s&p),'path_not_semantic':len(p-s),'combined_not_semantic':len(c-s),'semantic_not_combined':len(s-c),'warning':'Edge-family files overlap and must not be added as independent edges.'}
    (OUT/'claim_graph_statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 历史 Claim Graph 全量统计','',f'统计时间：{result["generated_at_utc"]}。仅导出聚合量，不含主张原文、节点 ID、逐边记录、嵌入向量或坐标。','',
           f'历史节点 **{len(nodes):,}**，来自 **{node_stats["parent_papers"]:,}** 篇论文，贡献角色 **{node_stats["role_type_count"]}** 类；社区 **{len(cs):,}** 个，未分配社区 **{community_stats["unassigned_claims"]:,}** 条。', '', '|贡献角色|节点数|','|---|---:|']
    lines += [f'|{k}|{v:,}|' for k,v in node_stats['role_counts'].items()]
    lines += ['', '|边族|保存行数|无向唯一边|孤立节点|连通分量|','|---|---:|---:|---:|---:|']
    for k,v in result['edges'].items(): lines.append(f'|{k}|{v["row_count"]:,}|{v["unique_undirected_edges"]:,}|{v["isolates"]:,}|{v["connected_components"]:,}|')
    lines += ['', '边族存在包含和重叠，不可相加。Paper-path 是主张边的论文关系注释，不表示已核验的支持或先行关系。社区是算法簇，不是已验证学科。','', '## 当前资产中的历史指标分组','']
    for k,v in result['historical_metric_groups'].items(): lines += [f'- **{k}**：'+', '.join(v)]
    lines += ['', '## 全指标统计（旧规则历史画像）','', '|指标|有效/缺失|均值|中位数|P05|P95|','|---|---:|---:|---:|---:|---:|']
    for k,v in metrics.items():
        st=v['overall']; lines.append(f'|{k}|{st["valid"]}/{st["missing_or_nonfinite"]}|{st["mean"]:.5g}|{st["median"]:.5g}|{st["p05"]:.5g}|{st["p95"]:.5g}|')
    lines += ['', '**口径限制：** 上表来自历史 insertion profiles，不能把旧规则的分位数用于当前阈值检索。完整 JSON 同时包含每种贡献角色的指标分布、时间分布、类型边矩阵、权重分布、来源字段/文件体积/时间和嵌入维度。全局拓扑以完整节点集计算，保留无边节点；没有计算或宣称科学创新正确率。']
    (OUT/'claim_graph_summary.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
