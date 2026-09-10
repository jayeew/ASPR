"""Read-only aggregate census of local Paper Graph; emits no individual records."""
from __future__ import annotations
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/claim_graph'
OUT = ROOT / 'outputs/GRAPH_STATISTICS'


def describe(values: np.ndarray) -> dict:
    return {'n': int(values.size), 'min': float(values.min()) if values.size else None,
            'max': float(values.max()) if values.size else None,
            'mean': float(values.mean()) if values.size else None,
            'quantiles': dict(zip(['p0','p25','p50','p75','p90','p95','p99','p100'],
                                 np.quantile(values, [0,.25,.5,.75,.9,.95,.99,1]).tolist())) if values.size else {},
            'histogram': {f'{lo}–{hi-1}': int(((values >= lo) & (values < hi)).sum())
                          for lo, hi in zip([0,1,2,5,10,20,50,100,500,1000,10000],
                                            [1,2,5,10,20,50,100,500,1000,10000,2**31])}}


def source(path: Path) -> dict:
    s = path.stat()
    return {'path': str(path.relative_to(ROOT)), 'bytes': s.st_size,
            'mtime_utc': datetime.fromtimestamp(s.st_mtime, timezone.utc).isoformat()}


def count_column(array: object, counter: Counter) -> None:
    for row in pc.value_counts(array).to_pylist():
        counter[str(row['values']) if row['values'] is not None else '<MISSING>'] += row['counts']


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    npf = pq.ParquetFile(DATA / 'paper_nodes.parquet')
    epf = pq.ParquetFile(DATA / 'paper_edges.parquet')
    n = npf.metadata.num_rows
    ids = np.empty(n, dtype=np.uint64)
    years = np.empty(n, dtype=np.int32)
    dates = np.empty(n, dtype='datetime64[D]')
    refs = np.empty(n, dtype=np.int32)
    categories = {k: Counter() for k in ['work_type','publication_year','source_name',
                  'field_name','primary_topic_name','hop_min','is_nature_target']}
    nulls = Counter()
    claims = pq.read_table(DATA / 'claim_nodes.parquet', columns=['parent_paper_id'])['parent_paper_id'].to_pylist()
    claim_parents = set(claims)
    covered = set()
    position = 0
    for batch in npf.iter_batches(batch_size=131072):
        table = batch.to_pydict()
        size = batch.num_rows
        raw = table['work_id']
        ids[position:position+size] = [int(v[1:]) for v in raw]
        years[position:position+size] = [v if v is not None else -1 for v in table['publication_year']]
        dates[position:position+size] = np.asarray([v or 'NaT' for v in table['publication_date']], dtype='datetime64[D]')
        refs[position:position+size] = table['referenced_works_count']
        covered.update(v for v in table['nature_article_id'] if v in claim_parents)
        for name in batch.schema.names:
            nulls[name] += int(pc.sum(pc.or_(pc.is_null(batch.column(name)), pc.fill_null(pc.equal(pc.cast(batch.column(name), 'string'), ''), False))).as_py() or 0)
        for name, counts in categories.items():
            count_column(batch.column(name), counts)
        position += size
        if position % (131072 * 20) == 0:
            print(f'Nodes {position:,}/{n:,}', flush=True)
    print('Sorting node index', flush=True)
    order = np.argsort(ids)
    ids = ids[order]
    dates = dates[order]
    duplicate_nodes = int((ids[1:] == ids[:-1]).sum())
    del order
    encoded = np.empty(epf.metadata.num_rows, dtype=np.uint64)
    written = 0
    edge_stats = Counter()
    hop_pairs = Counter()
    for batch in epf.iter_batches(batch_size=262144):
        a = np.asarray([int(v[1:]) for v in batch.column('citing_work_id').to_pylist()], dtype=np.uint64)
        b = np.asarray([int(v[1:]) for v in batch.column('cited_work_id').to_pylist()], dtype=np.uint64)
        ia, ib = np.searchsorted(ids, a), np.searchsorted(ids, b)
        va = (ia < n) & (ids[np.minimum(ia,n-1)] == a)
        vb = (ib < n) & (ids[np.minimum(ib,n-1)] == b)
        valid = va & vb
        edge_stats['missing_citing_endpoint_rows'] += int((~va).sum())
        edge_stats['missing_cited_endpoint_rows'] += int((~vb).sum())
        edge_stats['missing_either_endpoint_rows'] += int((~valid).sum())
        edge_stats['self_loop_rows'] += int((a == b).sum())
        ia, ib = ia[valid], ib[valid]
        codes = ia.astype(np.uint64) * np.uint64(n) + ib.astype(np.uint64)
        encoded[written:written+len(codes)] = codes
        written += len(codes)
        da, db = dates[ia], dates[ib]
        usable = (~np.isnat(da)) & (~np.isnat(db)) & (ia != ib)
        edge_stats['nonself_date_comparable_rows'] += int(usable.sum())
        edge_stats['nonself_citing_date_before_cited_date_rows'] += int((usable & (da < db)).sum())
        edge_stats['nonself_same_date_rows'] += int((usable & (da == db)).sum())
        for pair,count in Counter(zip(batch.column('citing_hop_min').to_pylist(), batch.column('cited_hop_min').to_pylist())).items():
            hop_pairs[f'{pair[0]}→{pair[1]}'] += count
        if written % (262144 * 20) < 262144:
            print(f'Edges {written:,}/{epf.metadata.num_rows:,}', flush=True)
    encoded = encoded[:written]
    print('Deduplicating endpoint-valid edge pairs', flush=True)
    encoded.sort()
    keep = np.empty(len(encoded), dtype=bool)
    keep[0] = True
    keep[1:] = encoded[1:] != encoded[:-1]
    edge_stats['duplicate_endpoint_valid_edge_rows'] = int((~keep).sum())
    encoded = encoded[keep]
    del keep, dates, ids
    a = (encoded // np.uint64(n)).astype(np.int32)
    b = (encoded % np.uint64(n)).astype(np.int32)
    del encoded
    nonself = a != b
    edge_stats['distinct_self_loops'] = int((~nonself).sum())
    a, b = a[nonself], b[nonself]
    del nonself
    edge_stats['clean_distinct_nonself_edges'] = int(len(a))
    outgoing = np.bincount(a, minlength=n)
    incoming = np.bincount(b, minlength=n)
    degree_stats = {'incoming': describe(incoming), 'outgoing': describe(outgoing),
                    'incident_in_plus_out': describe(incoming + outgoing),
                    'isolated_nodes': int(((incoming == 0) & (outgoing == 0)).sum()),
                    'zero_indegree_nodes': int((incoming == 0).sum()),
                    'zero_outdegree_nodes': int((outgoing == 0).sum())}
    print('Computing exact weak components', flush=True)
    adjacency = coo_matrix((np.ones(len(a), dtype=np.uint8), (a,b)), shape=(n,n)).tocsr()
    del a,b
    components, labels = connected_components(adjacency, directed=True, connection='weak')
    sizes = np.bincount(labels)
    component_stats = {'weak_component_count': int(components), 'weak_component_sizes': describe(sizes),
                       'largest_weak_component_nodes': int(sizes.max()),
                       'singleton_components': int((sizes == 1).sum()),
                       'strong_components': {'status': 'not_computed', 'reason': 'Outside bounded census; weak components computed exactly.'}}
    del adjacency, labels, sizes
    stats = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Complete local paper parquet graph, not Fig.1 sampled atlas. No raw paper identifiers, titles or records exported.',
             'method': 'Exact batch scan; numeric OpenAlex W identifiers indexed by sorted uint64; sorted endpoint-pair deduplication; NumPy degrees; scipy weak components. Source files read-only.',
             'sources': [source(DATA / name) for name in ['paper_nodes.parquet','paper_edges.parquet','claim_nodes.parquet']],
             'nodes': {'raw_rows': n, 'distinct_work_ids': n-duplicate_nodes, 'duplicate_work_id_rows': duplicate_nodes,
                       'missing_or_empty_fields': dict(nulls), 'category_counts': {k: dict(v.most_common()) for k,v in categories.items()},
                       'category_cardinalities_nonmissing': {k: len(set(v)-{'<MISSING>'}) for k,v in categories.items()},
                       'referenced_works_count_metadata': describe(refs)},
             'edges': {'raw_rows': epf.metadata.num_rows, 'relation_type': 'paper citation (citing → cited)',
                       'relation_type_count': 1, **dict(edge_stats), 'hop_pair_counts': dict(hop_pairs),
                       'directed_density_excluding_loops': edge_stats['clean_distinct_nonself_edges'] / (n*(n-1)),
                       'date_inversion_interpretation': 'Metadata ordering anomaly only; versions, online-first and date fields require verification; not automatically future leakage.'},
             'degree': degree_stats, 'components': component_stats,
             'claim_parent_coverage': {'distinct_historical_claim_parent_articles': len(claim_parents),
                                       'matched_to_paper_nature_article_id': len(covered),
                                       'unmatched': len(claim_parents - covered)},
             'metric_groups': {'inventory': ['nodes','edges','relation types','category cardinalities'],
                               'composition': list(categories),
                               'integrity': ['duplicate identifiers','duplicate edges','self-loops','missing endpoints','missing metadata','date ordering'],
                               'topology': ['in-degree','out-degree','incident degree','isolation','density','weak components'],
                               'cross_layer_coverage': ['claim parent article mapping'],
                               'bibliographic_metadata': ['referenced_works_count (not induced-graph outdegree)']},
             'not_computed': ['global shortest paths / diameter / betweenness','PageRank','strong components','citation semantic validity','version-normalized corrected publication dates'],
             'limitations': ['Parquet is an existing local acquisition, not a complete global citation graph.',
                             'Hop-2 acquisition boundaries explain many zero-outdegree nodes; zero observed outdegree does not mean no references in the original paper.',
                             'Full graph self-loops are counted and excluded in derived clean statistics; original datasets are not modified.',
                             'Fields and topics are provider metadata; counts do not independently validate classifications.',
                             'Source modification times and sizes are recorded; files are not cryptographically frozen during scan.']}
    target = OUT / 'paper_graph_statistics.json'
    target.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Paper Graph 全量统计汇总', '', f'统计时间：{stats["generated_at_utc"]}。全部为本地 parquet 实测统计；不含逐论文原始记录。', '',
             '| 统计项 | 数值 |','|---|---:|', f'| 论文节点原始行数 | {n:,} |',
             f'| 原始引用边 | {epf.metadata.num_rows:,} |', f'| 自环行 | {edge_stats["self_loop_rows"]:,} |',
             f'| 端点有效边重复行 | {edge_stats["duplicate_endpoint_valid_edge_rows"]:,} |',
             f'| 端点缺失边 | {edge_stats["missing_either_endpoint_rows"]:,} |',
             f'| 清洁去重非自环边 | {edge_stats["clean_distinct_nonself_edges"]:,} |',
             f'| 孤立节点 | {degree_stats["isolated_nodes"]:,} |',
             f'| 弱连通分量 | {components:,} |',
             f'| 最大弱分量节点 | {component_stats["largest_weak_component_nodes"]:,} |',
             f'| 引用日期逆序（待核） | {edge_stats["nonself_citing_date_before_cited_date_rows"]:,} |', '',
             '完整年份、期刊、主题、领域、类型、hop、元数据缺失分布及度数分位数见同目录 paper_graph_statistics.json。', '',
             '引用日期逆序仅是日期元数据异常，不能直接认定未来泄漏；本次只生成统计，未修改原图。引用边只表示文献引用，不直接证明 claim 支持、因果或创新。', '',
             '统计分组：规模；节点组成；数据完整性；度数/连通拓扑；跨层归属覆盖；参考文献元数据。', '',
             '未计算：全局最短路/直径/介数、PageRank、强连通分量；未进行逐篇引用真实性或出版版本核验。']
    for group in ['work_type', 'hop_min', 'is_nature_target']:
        lines.extend(['', f'## {group}', '', '| 类别 | 节点数 | 占全部节点 |', '|---|---:|---:|'])
        lines.extend(f'| {key} | {count:,} | {count/n:.3%} |' for key, count in categories[group].most_common())
    lines.extend(['', '## 解释边界', '', 'hop=2 是采集边界层；大量零出度反映当前数据库未继续收集这些文献的出边，不能理解为原论文没有参考文献。source_id、field_id、primary_topic_id 全部缺失；类别名称统计仍来自保存的名称字段，不能视为经 ID 规范化后的分类。'])
    (OUT / 'paper_graph_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'nodes': n, 'edges': dict(edge_stats), 'components': component_stats['weak_component_count']}), flush=True)

if __name__ == '__main__':
    main()
