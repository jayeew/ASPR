"""Aggregate live study artifacts; export no claim/paper identifiers or source text."""
from __future__ import annotations
import json
import math
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/innovation_200_20260907'
OUT = ROOT / 'outputs/GRAPH_STATISTICS'


def describe(values: list[Any]) -> dict[str, Any]:
    """Population descriptions, with explicit null counts and discrete masses."""
    x = np.array([float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)], dtype=float)
    out: dict[str, Any] = {'total': len(values), 'valid_n': len(x), 'missing_n': len(values)-len(x)}
    if not len(x):
        return out
    out.update(dict(zip(['min','p05','p25','median','p75','p95','max'], map(float,np.quantile(x,[0,.05,.25,.5,.75,.95,1])))))
    out.update(mean=float(x.mean()), population_std=float(x.std()), zero_n=int(sum(x==0)), one_n=int(sum(x==1)), negative_n=int(sum(x<0)))
    if len(set(x)) <= 30:
        out['value_counts'] = dict(sorted(Counter(map(str,x)).items(), key=lambda p: float(p[0])))
    return out


def dictionary() -> dict[str, dict[str, str]]:
    rows = [
        ('nearest_prior_similarity','最近历史主张相似度','语义接近','max cosine','[-1,1]'),
        ('mean_top5_similarity','前五邻居平均相似度','语义接近','mean 前 min(5,n) 个相似度','[-1,1]'),
        ('effective_community_count','有效社区数','社区组合','1 / sum(p_c²)，p 为社区正相似度权重占比','[1,n]；无社区可能保存0'),
        ('community_rao_stirling','社区Rao–Stirling差异','社区组合','sum 2*p_a*p_b*clip(1-centroid_cosine,0,1)，a<b','[0,1]'),
        ('first_observed_recent_nature_pair_share','历史表未出现社区对占比','社区组合','未命中community_pair_history的无序社区对数 / 全部社区对数','[0,1]'),
        ('community_pair_mean_surprisal','社区对平均负对数相对常见度','社区组合','mean[-ln((pair_connector_count+0.5)*historical_claim_count/(N_a*N_b))]；为lift式量而非概率，允许负数','实数'),
        ('neighbor_induced_density','历史邻域诱导密度','局部连通','2m/[n(n-1)]；仅历史邻居间边','[0,1]'),
        ('component_merge_count','局部分量合并数','局部连通','max(历史邻域分量数-1,0)','[0,n-1]'),
        ('newly_connected_neighbor_pair_count','新连通历史邻居对数','局部连通','choose(n,2)-sum choose(component_size,2)','[0,choose(n,2)]'),
        ('cross_boundary_weight_share','非主导社区权重占比','社区组合','1-max(p_c)，不是全局跨学科影响','[0,1]'),
        ('direct_citation_neighbor_count','直接引用邻居数','论文路径注释','sum(direct_citation)','[0,n]'),
        ('two_hop_neighbor_count','有二跳引用路径邻居数','论文路径注释','sum(two_hop_path_count>0)','[0,n]'),
        ('co_citation_neighbor_count','有共同参考文献邻居数','论文路径注释','sum(shared_reference_count>0)；实际是共同参考/书目耦合，不是被同一第三方共引','[0,n]'),
        ('cross_type_neighbor_count','异主张类型邻居数','主张类型组合','sum(neighbor.claim_type != target.claim_type)','[0,n]'),
        ('neighbor_count','保留邻居数','检索覆盖','len(neighbors)；触及10不证明候选充足','[0,10]'),
        ('historical_edge_count','历史局部边数','局部连通','去重无向非自环neighbor_edges数','[0,choose(n,2)]'),
        ('community_count','观测社区数','社区组合','邻居非空community_id不同值数量','[0,n]'),
        ('connected_pair_share','局部新连通对占比','局部连通','newly_connected_neighbor_pair_count / choose(n,2)','[0,1]'),
    ]
    result = {key: {'name_zh':name,'group':group,'unit':'目标claim','definition':formula,'range':bounds,'na_rule':'按保存值；原始14项无邻居为NA；社区对无配对/统计表缺失为NA。'} for key,name,group,formula,bounds in rows}
    result['connected_pair_share']['na_rule']='n<2为NA；禁止用0填补'
    result['neighbor_induced_density']['na_rule']='保存实现：n=1为0；n=0整组指标为NA。n<2不能解释密度为可测比例。'
    joints = {'historical_neighbor_count':'联合历史邻居并集节点数','target_claim_count':'参与联合插入目标主张数','historical_components_before':'插入前历史分量数','historical_components_after':'联合插入后历史节点分量数','joint_component_merge_count':'联合插入分量减少数','joint_newly_connected_historical_pairs':'联合插入后新可达历史节点对数','requested_claim_count':'请求主张数','historical_edge_count':'联合历史边数','insertion_edge_count':'目标到历史插入边数','shared_neighbor_node_count':'被至少两个目标共享的历史邻居节点数','claims_without_neighbors_count':'无邻居目标数','missing_fact_claim_count':'缺少事实的请求主张数','joint_connected_pair_share':'联合新连通对数 / choose(历史邻居数,2)','neighbor_reuse_share':'1-历史邻居并集数/插入边数，非Jaccard'}
    for key,name in joints.items():
        result['joint.'+key]={'name_zh':name,'group':'整篇联合局部结构','unit':'论文','definition':name,'range':'非负计数；share为[0,1]','na_rule':'无保存joint工件为NA；比例分母为0或节点不足2为NA'}
    for key,name,unit in [('_pair_shared','两个目标claim共享的历史邻居数','论文内无序目标claim对'),('_pair_connected','两个目标claim是否经历史图连接','论文内无序目标claim对'),('_reuse_multiplicity','共享历史节点被多少目标claim使用','至少被两个目标使用的历史节点（论文内）')]:
        result['joint.'+key]={'name_zh':name,'group':'联合重叠分布','unit':unit,'definition':name,'range':'计数；connected为0或1','na_rule':'无joint或无符合条件单位时不纳入；不是独立论文单位'}
    return result


def main() -> None:
    start = datetime.now(timezone.utc).isoformat()
    paths: dict[Path, tuple[int,int]] = {}
    errors: Counter[str] = Counter()
    def read(path: Path, lines: bool=False) -> Any:
        if not path.exists():
            return [] if lines else {}
        stat=path.stat(); paths[path]=(stat.st_size,stat.st_mtime_ns)
        try:
            raw=path.read_text()
            return [json.loads(s) for s in raw.splitlines() if s.strip()] if lines else json.loads(raw)
        except (json.JSONDecodeError,UnicodeDecodeError):
            errors['unreadable_json_files']+=1
            return [] if lines else {}
    roster=read(RUN/'papers.jsonl',True)
    original=set(read(ROOT/'outputs/FROM_WEB/data/comparison_cohort.json')['paper_ids'])
    observations: list[dict[str, Any]]=[]; joints: list[dict[str,Any]]=[]; paperrows=[]; neighborrows=[]; sharedrows=[]
    trace_counts=Counter(); policy=Counter(); all_claim_ids=set(); graph_claim_ids=set()
    for paper in roster:
        pid=paper['paper_id']; base=RUN/'papers'/pid
        group={'cohort':'original200' if pid in original else 'extension800','journal':paper.get('journal_name','unknown'),'field':paper.get('field_name','unknown'),'field_source':paper.get('field_source','unknown')}
        claims=read(base/'shared/claims.json').get('claims',[])
        for c in claims:
            all_claim_ids.add(c['claim_id'])
            sharedrows.append({**group,'claim_type':c.get('claim_type','unknown'),'internal_support':c.get('internal_support','unknown'),'source_span_count':len(c.get('source_span_ids',[])),'support_span_count':len(c.get('support_span_ids',[]))})
        pr={**group,'shared_claims':len(claims),'has_shared_claims':bool(claims),'graph_facts':0,'gear_cards':0,'graph_analysis':int((base/'graph/analysis.json').exists()),'gear_analysis':int((base/'gear/analysis.json').exists()),'joint_analysis':int((base/'graph/joint/analysis.json').exists())}
        for system in ['graph','direct_llm','fusion']:
            pr[system+'_report']=int((RUN/'reports'/system/(pid+'.json')).exists())
        for path in sorted((base/'graph').glob('[0-9]*/evidence_trace.jsonl')):
            records=read(path,True); selected={}
            for record in records:
                if record.get('kind')!='graph_fact': continue
                trace_counts['graph_fact_records']+=1; fact=record.get('payload',{})
                if not {'claim','neighbors','metrics'} <= fact.keys():
                    trace_counts['incomplete_graph_fact_records']+=1;continue
                selected[fact['claim']['claim_id']]=fact
            trace_counts['selected_records']+=len(selected)
            for cid,fact in selected.items():
                if cid in graph_claim_ids: trace_counts['duplicate_claims_across_trace_files']+=1;continue
                graph_claim_ids.add(cid);pr['graph_facts']+=1
                n=len(fact['neighbors']); metrics={m['name']:m.get('value') for m in fact['metrics']}
                metrics.update(neighbor_count=n,historical_edge_count=len({tuple(sorted(edge)) for edge in fact.get('neighbor_edges',[]) if edge[0]!=edge[1]}),community_count=len({x['community_id'] for x in fact['neighbors'] if x.get('community_id') is not None}))
                metrics['connected_pair_share']=metrics.get('newly_connected_neighbor_pair_count',0)/(n*(n-1)/2) if n>=2 and metrics.get('newly_connected_neighbor_pair_count') is not None else None
                p=fact.get('insertion_policy','unknown');policy[p]+=1
                observations.append({**group,'claim_type':fact['claim']['claim_type'],'policy':p,**metrics})
                for neighbor in fact['neighbors']:
                    neighborrows.append({k:neighbor.get(k) for k in ['edge_type','claim_type','cosine_similarity','semantic_rank','direct_citation','two_hop_path_count','shared_reference_count','shared_reference_salton','parent_id_cache_hit','community_id']})
        cards=[]
        for path in sorted((base/'gear').glob('[0-9]*/gear_card.json')):
            card=read(path)
            if card: cards.append(card)
        pr['gear_cards']=len(cards)
        pr['gear_execution_status']=dict(Counter(str(read(path).get('status','unknown')) for path in (base/'gear').glob('[0-9]*/execution.json')))
        paperrows.append(pr)
        fact=read(base/'graph/joint/facts.json')
        if fact:
            row={k:v for k,v in fact.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
            row.update(historical_edge_count=len(fact.get('historical_edges',[])),insertion_edge_count=len(fact.get('insertion_edges',[])),shared_neighbor_node_count=len(fact.get('shared_historical_neighbors',{})),claims_without_neighbors_count=len(fact.get('claims_without_neighbors',[])),missing_fact_claim_count=len(fact.get('missing_fact_claim_ids',[])))
            n=fact.get('historical_neighbor_count',0); e=row['insertion_edge_count']
            row['joint_connected_pair_share']=fact.get('joint_newly_connected_historical_pairs',0)/(n*(n-1)/2) if n>=2 else None
            row['neighbor_reuse_share']=1-n/e if e else None
            row['_pair_shared']=[x['shared_neighbor_count'] for x in fact.get('claim_pairs',[])];row['_pair_connected']=[x['connected_via_historical_graph'] for x in fact.get('claim_pairs',[])]
            row['_reuse_multiplicity']=list(fact.get('shared_historical_neighbors',{}).values());joints.append({**group,**row})
    metricnames=sorted({k for row in observations for k in row if k not in ['cohort','journal','field','field_source','claim_type','policy']})
    summaries={k:describe([r.get(k) for r in observations]) for k in metricnames}
    bygroup={key:{str(value):{'n':sum(r[key]==value for r in observations),'metrics':{k:describe([r.get(k) for r in observations if r[key]==value]) for k in metricnames}} for value in sorted({r[key] for r in observations})} for key in ['cohort','journal','field','claim_type','policy']}
    cohorts={}
    for co in ['all1000','original200','extension800']:
        rows=[r for r in paperrows if co=='all1000' or r['cohort']==co]
        keys=['shared_claims','has_shared_claims','graph_facts','gear_cards','graph_analysis','gear_analysis','joint_analysis','graph_report','direct_llm_report','fusion_report']
        cohorts[co]={'roster_papers':len(rows),**{k:sum(r[k] for r in rows) for k in keys},'journal_counts':dict(Counter(r['journal'] for r in rows)),'field_counts':dict(Counter(r['field'] for r in rows)),'field_source_counts':dict(Counter(r['field_source'] for r in rows)),'claims_per_paper':describe([r['shared_claims'] for r in rows]),'shared_claim_type_counts':dict(Counter(r['claim_type'] for r in sharedrows if co=='all1000' or r['cohort']==co)),'shared_claim_internal_support_counts':dict(Counter(r['internal_support'] for r in sharedrows if co=='all1000' or r['cohort']==co)),'gear_execution_status':dict(sum((Counter(r['gear_execution_status']) for r in rows),Counter()))}
        cohorts[co]['missing_graph_fact_claims_unknown_execution']=cohorts[co]['shared_claims']-cohorts[co]['graph_facts']
        cohorts[co]['missing_gear_cards_unknown_execution']=cohorts[co]['shared_claims']-cohorts[co]['gear_cards']
    correlations=[]
    for i,left in enumerate(metricnames):
        for right in metricnames[i+1:]:
            pairs=[(r[left],r[right]) for r in observations if r.get(left) is not None and r.get(right) is not None]
            if len(pairs)>2 and len({p[0] for p in pairs})>1 and len({p[1] for p in pairs})>1:
                rho=float(spearmanr(np.array(pairs)[:,0],np.array(pairs)[:,1]).statistic)
                correlations.append({'metric_a':left,'metric_b':right,'n':len(pairs),'spearman_rho':rho})
    jointkeys=sorted({k for r in joints for k in r if k not in ['cohort','journal','field','field_source'] and not k.startswith('_')})
    jointstats={k:describe([r.get(k) for r in joints]) for k in jointkeys}
    for key in ['_pair_shared','_pair_connected','_reuse_multiplicity']:
        jointstats[key]=describe([v for r in joints for v in r[key]])
    changed=sum(not p.exists() or (p.stat().st_size,p.stat().st_mtime_ns)!=state for p,state in paths.items())
    data={'source_definitions':{'current_policy_code':'threshold_parent_path_v2:k=10:cosine>0.5','metric_implementation':'gear/claim_attribution.py::_metrics / _pair_surprisal / _new_pair_share','metric_code_sha256':hashlib.sha256((ROOT/'gear/claim_attribution.py').read_bytes()).hexdigest(),'roster_sha256':hashlib.sha256((RUN/'papers.jsonl').read_bytes()).hexdigest(),'original_cohort_sha256':hashlib.sha256((ROOT/'outputs/FROM_WEB/data/comparison_cohort.json').read_bytes()).hexdigest(),'append_only_selection':'last complete graph_fact per claim in each trace; deduplicate across files; incomplete and duplicate counts disclosed','operational_status':'gear execution statuses are read separately from existence of saved cards; saved card does not establish successful evidence or independent correctness'},'schema_version':'aggregate_graph_statistics_v1','source':'outputs/innovation_200_20260907','scan':{'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'files_read':len(paths),'files_changed_during_scan':changed,'parse_errors':dict(errors),'mtime_min_utc':datetime.fromtimestamp(min(v[1] for v in paths.values())/1e9,timezone.utc).isoformat(),'mtime_max_utc':datetime.fromtimestamp(max(v[1] for v in paths.values())/1e9,timezone.utc).isoformat()},'boundaries':['仅聚合统计，无paper/claim逐条标识、原文、节点边清单。','实时扫描非事务快照；GEAR可能继续运行。缺工件不等于科学无证据。','保存运行policy v1；当前代码已修复同work自环并有v2，不代表保存结果已经按v2重跑。','主张、语义关系、社区和解释包含模型/算法输出；不是独立人工创新真值。','指标分位数为当前样本描述，不是历史创新百分位；相关性不构成独立科学验证。','n=10是保留上限；旧工件无候选总数，不能判断实际被截断候选数。'], 'cohorts':cohorts,'trace_selection':dict(trace_counts),'saved_policies':dict(policy),'metric_dictionary':dictionary(),'metric_dictionary_count':len(dictionary()),'observed_metric_count':len(metricnames),'claim_metric_statistics':summaries,'grouped_claim_metrics':bygroup,'shared_claim_source_counts':{k:describe([r[k] for r in sharedrows]) for k in ['source_span_count','support_span_count']},'neighbor_annotations':{'records':len(neighborrows),'edge_type_counts':dict(Counter(str(r['edge_type']) for r in neighborrows)),'neighbor_claim_type_counts':dict(Counter(str(r['claim_type']) for r in neighborrows)),'community_id_counts':dict(Counter(str(r['community_id']) for r in neighborrows)),'statistics':{k:describe([r[k] for r in neighborrows]) for k in ['cosine_similarity','semantic_rank','direct_citation','two_hop_path_count','shared_reference_count','shared_reference_salton','parent_id_cache_hit']}},'joint_statistics':jointstats,'joint_papers':len(joints),'spearman_correlations_descriptive':sorted(correlations,key=lambda r:-abs(r['spearman_rho']))}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'runtime_graph_statistics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 当前运行 Graph 聚合统计',f'扫描时间：{start} 至 {data["scan"]["end_utc"]}（UTC）。','全部为聚合描述统计，不含原文或逐条数据。','\n## 覆盖和边界']
    for co,summary in cohorts.items(): lines.extend([f'\n### {co}', '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```'])
    lines.extend(['\n## 指标定义','| 字段 | 中文 | 分组 | 单位 | 定义 | NA规则 |','|---|---|---|---|---|---|'])
    for k,d in dictionary().items():lines.append('| '+' | '.join([k,d['name_zh'],d['group'],d['unit'],d['definition'],d['na_rule']])+' |')
    lines.extend(['\n## 当前主张指标分布','| 指标 | 有效n | NA | 最小 | P25 | 中位 | P75 | 最大 | 均值 | 标准差 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for k,s in summaries.items():lines.append('| '+k+' | '+' | '.join(str(round(s[a],6)) if a in s else 'NA' for a in ['valid_n','missing_n','min','p25','median','p75','max','mean','population_std'])+' |')
    lines.extend(['\n## 邻居路径和联合结构统计','```json',json.dumps({'neighbor_annotations':data['neighbor_annotations'],'joint_statistics':jointstats},ensure_ascii=False,indent=2),'```','\n## 最强描述性相关（前20）','```json',json.dumps(data['spearman_correlations_descriptive'][:20],ensure_ascii=False,indent=2),'```','\n## 范围限制',*['- '+x for x in data['boundaries']], '\n完整期刊、领域、主张类型、队列与政策版本分组统计见同目录 runtime_graph_statistics.json（聚合单元，无原始行）。'])
    (OUT/'runtime_graph_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cohorts':cohorts,'scan':data['scan'],'metric_count':len(metricnames),'policies':dict(policy)},ensure_ascii=False))

if __name__=='__main__':
    main()
