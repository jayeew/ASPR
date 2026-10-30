"""Summarize inspected Fig.8 metadata and conditional, non-inference budgets."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import Counter
from statistics import median
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .stage0 import COHORTS, DEFAULT_OUT, FIELDS, ROOT, Client, write_csv, write_json
from .stage0_roster import abstract_text


def read(name: str) -> Any:
    return json.loads((DEFAULT_OUT / name).read_text())


def reconcile_missing_groups() -> None:
    """Resolve omitted null taxonomy buckets instead of silently losing records."""
    client = Client(DEFAULT_OUT)
    result = []
    for item in read('aggregates.json'):
        if item['total'] == item['group_sum']:
            continue
        query = item['filter'] + ',' + item['group_by'] + ':null'
        response = client.get('works', {'filter': query, 'per_page': 1, 'select': 'id'})
        missing = response['meta']['count']
        result.append({'cohort': item['cohort'], 'metric': item['metric'],
            'difference': item['total'] - item['group_sum'], 'verified_null_count': missing,
            'resolved_as_missing_taxonomy': item['total'] - item['group_sum'] == missing})
    write_json(DEFAULT_OUT / 'group_null_reconciliation.json', result)
    if not all(row['resolved_as_missing_taxonomy'] for row in result):
        raise RuntimeError('Unexplained aggregate denominator difference')


def api_receipt() -> dict[str, Any]:
    times, costs = [], []
    for path in (DEFAULT_OUT / 'api_raw').glob('*.json'):
        row = json.loads(path.read_text())
        times.append(row['retrieved_at'])
        costs.append((row['response'].get('meta') or {}).get('cost_usd', 0))
    return {'api_cached_queries': len(times), 'api_retrieved_at_min': min(times),
            'api_retrieved_at_max': max(times), 'api_reported_cost_usd_sum': sum(costs)}


def sampled_coverage() -> list[dict[str, Any]]:
    rows = []
    for cohort in COHORTS:
        records = read(f'sample_{cohort}.json')['results']
        authors = [a for r in records for a in r.get('authorships') or []]
        lengths = [len(text) for r in records if (text := abstract_text(r.get('abstract_inverted_index')))]
        rows.append({'cohort': cohort, 'sample_n': len(records),
            'sample_has_abstract_n': len(lengths), 'observed_authorship_n': len(authors),
            'authorship_with_institution_n': sum(bool(a.get('institutions')) for a in authors),
            'work_all_observed_authors_have_institution_n': sum(bool(r.get('authorships')) and
                all(a.get('institutions') for a in r['authorships']) for r in records),
            'work_truncated_authors_n': sum(bool(r.get('is_authors_truncated')) or
                (r.get('authors_count') or 0) > len(r.get('authorships') or []) for r in records),
            'median_abstract_characters': median(lengths) if lengths else None,
            'scope': '100 API random works per cohort; author-completeness is sample-only, not census'})
    write_csv(DEFAULT_OUT / 'sample_author_coverage.csv', rows)
    return rows


def id_spotcheck() -> dict[str, Any]:
    """Check recovered labels against a deterministic 200-work current API sample."""
    path = DEFAULT_OUT / 'id_spotcheck.json'
    if path.exists():
        return read(path.name)
    rows = pq.read_table(DEFAULT_OUT / 'canonical_target_works_ids_restored.parquet').to_pylist()
    selected = []
    for source in sorted({row['source_id'] for row in rows}):
        selected.extend(sorted((row for row in rows if row['source_id'] == source),
                               key=lambda r: r['work_id'])[:25])
    client, records = Client(DEFAULT_OUT), []
    for start in range(0, len(selected), 100):
        ids = '|'.join(row['work_id'] for row in selected[start:start + 100])
        response = client.get('works', {'filter': 'openalex:' + ids, 'per_page': 100,
            'select': 'id,doi,primary_topic,primary_location,updated_date'})
        records.extend(response['results'])
    mapping = {row['id'].rsplit('/', 1)[-1]: row for row in records}
    details = []
    for old in selected:
        raw = mapping.get(old['work_id'])
        if raw is None:
            details.append({'work_id': old['work_id'], 'issue': 'not_returned'}); continue
        topic = raw.get('primary_topic') or {}
        live = {'source_id': ((raw.get('primary_location') or {}).get('source') or {}).get('id'),
                'field_id': (topic.get('field') or {}).get('id'), 'primary_topic_id': topic.get('id')}
        for key, value in live.items():
            if old[key] != value:
                details.append({'work_id': old['work_id'], 'issue': key + '_snapshot_live_difference',
                                'restored': old[key], 'live': value})
    result = {'selected_n': len(selected), 'returned_n': len(records), 'discrepancies': details,
              'scope': 'First 25 sorted IDs per source; current API versus snapshot labels, not a random error-rate estimate.'}
    write_json(path, result)
    return result


def workload(items: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(item['cohort'], item['metric']): item for item in items}
    nature = pq.read_table(DEFAULT_OUT / 'nature_candidate_roster.parquet').to_pylist()
    extra = [r for r in nature if not r['panel_b_overlap']]
    n_target = sum(lookup[c, 'candidate']['total'] for c in FIELDS) + len(extra)
    n_abstract = sum(lookup[c, 'abstract']['total'] for c in FIELDS) + sum(bool(r['abstract_text']) for r in extra)
    nature_fields = {r['field_id'] for r in nature if r['field_id']}
    history_rows = []
    for item in items:
        if item['cohort'] == 'history' and item['metric'].endswith('_abstract'):
            for row in item['rows']:
                if row['key'] in nature_fields:
                    history_rows.append({'field_id': row['key'], 'year': int(item['metric'][:4]),
                        'abstract_available_api_n': row['count'], 'planned_sample_n': min(2000, row['count'])})
    write_csv(DEFAULT_OUT / 'history_sampling_capacity.csv', history_rows)
    n_history = sum(r['planned_sample_n'] for r in history_rows)
    old = pq.read_table(DEFAULT_OUT / 'canonical_target_works_ids_restored.parquet').to_pylist()
    claim_papers = set(pq.read_table(ROOT / 'data/claim_graph/claim_nodes.parquet', columns=['parent_paper_id'])['parent_paper_id'].to_pylist())
    snapshot_reuse = sum(r['field_id'].rsplit('/', 1)[-1] in FIELDS.values() and r['work_type'] in ['article', 'letter']
                        and r['nature_article_id'] in claim_papers for r in old)
    current_audit = read('current_id_recovery_audit.json')
    reuse = current_audit['current_b_candidate_with_legacy_claim_n']
    claim_n = 3 * (n_abstract + n_history)
    matrix = np.load(ROOT / 'data/claim_graph/claim_embedding_matrix.npy', mmap_mode='r')
    budget = {'target_candidate_n': n_target, 'target_with_abstract_n': n_abstract,
        'target_missing_abstract_n': n_target - n_abstract, 'history_sample_n': n_history,
        'nature_fields_n': len(nature_fields), 'history_cells_n': len(history_rows),
        'new_extractions_no_verified_reuse': n_abstract + n_history,
        'potential_legacy_overlap_unverified': reuse,
        'snapshot_label_potential_overlap': snapshot_reuse,
        'new_extractions_if_all_potential_reuse_verified': n_abstract + n_history - reuse,
        'new_extractions_if_all_candidate_abstracts_recovered': n_target + n_history,
        'max_claims_current_abstract_scope': claim_n, 'embedding_dimensions': matrix.shape[1],
        'embedding_float32_gib_upper': claim_n * matrix.shape[1] * 4 / 2**30,
        'target_neighbor_rows_upper': 30 * n_abstract,
        'metadata_requests_full_abstract_scope_200_per_page': math.ceil((n_abstract + n_history) / 200),
        'candidate_semantics': 'OpenAlex Work counts; publisher original-research and cross-Work DOI/version audit pending'}
    write_json(DEFAULT_OUT / 'workload_budget.json', budget)
    return budget


def budget_scenarios(budget: dict[str, Any]) -> list[dict[str, Any]]:
    tree = ast.parse((ROOT / 'scripts/claim_graph/04_extract_claims.py').read_text())
    prompt = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'PROMPT_TEMPLATE' for target in node.targets))
    lengths = [len(text) for cohort in COHORTS for row in read(f'sample_{cohort}.json')['results']
               if (text := abstract_text(row.get('abstract_inverted_index')))]
    middle = math.ceil((len(prompt) + median(lengths)) / 4 + 100)
    settings = [('low', .02, .8, 300, 15), ('central', .05, 1., 600, 30), ('high', .10, 1.5, 1200, 60)]
    rows = []
    for name, retries, multiplier, output, seconds in settings:
        calls = math.ceil(budget['new_extractions_no_verified_reuse'] * (1 + retries))
        input_tokens = math.ceil(middle * multiplier)
        rows.append({'scenario': name, 'attempts': calls, 'retry_overhead_assumed': retries,
            'input_tokens_per_attempt_assumed': input_tokens, 'output_tokens_per_attempt_assumed': output,
            'input_million_tokens': calls * input_tokens / 1e6,
            'output_million_tokens': calls * output / 1e6,
            'seconds_per_attempt_assumed': seconds, 'concurrency_assumed': 16,
            'wall_days_idealized': calls * seconds / 16 / 86400,
            'currency_cost': 'input_million_tokens*P_in + output_million_tokens*P_out',
            'limitations': 'No model benchmark; chars/4 heuristic; reasoning tokens, quotas and GPU work not priced.'})
    write_csv(DEFAULT_OUT / 'budget_scenarios.csv', rows)
    return rows


def supplemental_counts(items: list[dict[str, Any]], budget: dict[str, Any]) -> None:
    lookup = {(item['cohort'], item['metric']): item for item in items}
    rows = []
    for cohort in COHORTS:
        n = lookup[cohort, 'candidate']['total']
        abs_n = lookup[cohort, 'abstract']['total']
        both = lookup[cohort, 'abstract_institution']['total']
        rows.extend([
            {'cohort': cohort, 'task': 'abstract_unavailable_in_API', 'n': n - abs_n,
             'meaning': 'Candidate-record gap; may include non-research, do not equate with recoverable research abstracts'},
            {'cohort': cohort, 'task': 'no_linked_institution_any_author', 'n': n - lookup[cohort, 'institution']['total'],
             'meaning': 'Does not count partially mapped authorships'},
            {'cohort': cohort, 'task': 'has_abstract_but_no_institution', 'n': abs_n - both,
             'meaning': 'Direct metadata bottleneck for c'},
            {'cohort': cohort, 'task': 'doi_missing', 'n': n - lookup[cohort, 'doi']['total'],
             'meaning': 'Requires version/identity handling before deduplicated final roster'},
            {'cohort': cohort, 'task': 'publisher_type_date_verification_pending', 'n': n,
             'meaning': 'No automatic equivalence between API article and original research'},
        ])
    rows.append({'cohort': 'a_union_b', 'task': 'new_historical_abstract_claim_extraction',
                 'n': budget['history_sample_n'], 'meaning': 'Background sample capacity, not already downloaded abstracts'})
    write_csv(DEFAULT_OUT / 'supplement_requirements.csv', rows)


def report(budget: dict[str, Any], scenarios: list[dict[str, Any]], check: dict[str, Any]) -> None:
    with (DEFAULT_OUT / 'coverage_by_cohort_year.csv').open(encoding='utf-8-sig') as stream:
        coverage = [r for r in csv.DictReader(stream) if r['year'] == '0']
    with (DEFAULT_OUT / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = list(csv.DictReader(stream))
    counts = Counter(r['cohort'] for r in journals)
    nature_unknown = sum(r['nature_portfolio_current'] == '' for r in journals)
    provenance = {'snapshot': read('snapshot_audit.json'), **api_receipt(),
        'source_entities': read('source_entity_coverage.json'), 'group_mismatches': read('group_reconciliation.json'),
        'group_null_reconciliation': read('group_null_reconciliation.json'),
        'id_recovery': {k: v for k, v in read('id_recovery_audit.json').items() if k != 'sources'},
        'current_id_recovery': read('current_id_recovery_audit.json'),
        'model_calls': 0, 'embedding_calls': 0, 'historical_graph_modified': False,
        'original_canonical_sha256_after': hashlib.sha256((ROOT / 'data/claim_graph/canonical_target_works.parquet').read_bytes()).hexdigest()}
    write_json(DEFAULT_OUT / 'source_manifest.json', provenance)
    current = read('current_id_recovery_audit.json')
    lines = ['# Fig.8 第一阶段：数据名录、覆盖与预算', '',
        '执行日期：2026-10-21。已完成本地 snapshot 清单核查、旧表 ID 恢复、当前 API 候选期刊名录与覆盖统计、Nature 候选明细和预算。未执行 LLM／嵌入／建图。', '',
        '## 数据版本与完成边界', '',
        '本地 snapshot 为 updated_date 最晚 2026-03-30 的版本；2,127 个分片共 639,189,333,248 字节，manifest 记录 492,361,307 条。所有文件均存在且大小一致，没有清单外 gzip 分片。另读了 12 个分片的前 3 条记录核对字段。没有重新解压扫描全库或重新计算每片 CRC，因此不能把文件清单一致性称为全记录完整性认证。', '',
        '本地没有 sources、institutions 独立实体快照，旧索引只存 Work ID、不含领域或记录偏移。为建立可用名录，覆盖统计和期刊实体采用当前 OpenAlex API，并逐请求保存原始响应。它们不是 2026-03 snapshot 的精确队列统计，不混合解释成一个冻结版本。', '',
        '## 候选范围与覆盖', '',
        '范围：2023–2025，期刊来源，article 或 letter，非撤稿、非 paratext，language=en。b 按论文 primary field 13／25／31 纳入。原始研究类型及在线发表日期仍需出版社核验；英文标签也不代表逐条检测了摘要语言。下表为 Work 级候选统计，不是已验证原创研究总量。', '',
        '| 队列 | 候选记录 | 期刊数 | 有摘要 | 摘要率 | 至少一个机构 | 机构率 | 摘要及机构均有 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in coverage:
        lines.append(f"| {r['cohort']} | {int(r['candidate_n']):,} | {counts[r['cohort']]:,} | {int(r['abstract_n']):,} | {float(r['abstract_coverage']):.2%} | {int(r['institution_n']):,} | {float(r['institution_coverage']):.2%} | {int(r['abstract_institution_n']):,} |")
    lines.extend(['', '机构率只要求至少一个作者关联机构，不能用来证明全部作者／国家归属完整。另有每队列 100 篇随机记录的作者级抽查，结果见 sample_author_coverage.csv，不能冒充总体覆盖率。', '',
        f"期刊名录合并后有 {len({r['source_id'] for r in journals}):,} 个唯一来源。品牌列是当前出版组织谱系；发表当年的品牌归属全部标记为未验证，不能按名称包含 nature 推断。按队列的 {nature_unknown:,} 行没有出版组织谱系，保留 unknown。", '',
        '## ID 恢复与本地复用', '',
        '24,919 篇旧论文的 source_id、field_id、primary_topic_id 已恢复，无缺失和 topic-field 冲突；保存在独立 parquet／CSV。领域、主题来自本地 taxonomy 的唯一精确名称匹配，期刊来自当前 API 唯一精确名称匹配。此方法恢复实体身份，不代表原始论文主题获得重新验证。', '',
        f"另对每刊按 Work ID 排序的前 25 篇、共 {check['selected_n']} 篇检查当前 API；返回 {check['returned_n']} 篇，记录到 {len(check['discrepancies'])} 项差异。详见 id_spotcheck.json；如有差异，区分元数据更新和恢复错误，不自动覆盖历史字段。", '',
        f"鉴于抽查发现分类漂移，进一步对全部 24,919 篇获取当前元数据：已映射 {current['matched_requested']:,} 篇，未解决 ID 数 {len(current['missing_requested_ids'])}。当前记录保存在 legacy_papers_current_metadata.parquet，历史名称恢复副本单独保留。差异按字段计数为 {json.dumps(current['difference_counts'], ensure_ascii=False)}，并非恢复算法错误率；它说明不能混用旧标签与当前 API 分组。", '',
        '期刊分组与年份分组均已核对加总；Nature 领域分组少的 30 条及历史领域分组缺口，已用独立 null-field 查询验证为未分类记录，见 group_null_reconciliation.json。未分类论文保留在总体分母，历史分层采样不将其虚构为一个领域。', '',
        f"当前三个领域中，旧 claims 可能复用的论文最多 {budget['potential_legacy_overlap_unverified']:,} 篇；尚未核实新纳入条件、语言和模型／提示词兼容性，因此主预算按 R=0 计算。", '',
        '## 补充量与工作量', '',
        f"Nature 与 b 按 primary field 去重后，共 {budget['target_candidate_n']:,} 条候选，其中 {budget['target_with_abstract_n']:,} 条已有 API 摘要，{budget['target_missing_abstract_n']:,} 条没有。无摘要缺口包含待排除的非研究内容，不能直接解读成需要下载这么多篇研究全文。", '',
        f"2018–2022 历史参考采样容量按 Nature 所涉 {budget['nature_fields_n']} 个已识别领域、每领域年份最多 2,000 篇计算，共 {budget['history_cells_n']} 个单元、{budget['history_sample_n']:,} 篇。计数说明 API 中有候选，不意味着这些摘要已全部落地；下一阶段仍需从 snapshot 定向抽取或通过 API 补取，再统一提取 claim。", '',
        f"按已有摘要目标＋历史背景、暂不认证旧提取复用：首轮最多需要 {budget['new_extractions_no_verified_reuse']:,} 次论文级提取。若全部潜在旧记录满足复用条件，可降至 {budget['new_extractions_if_all_potential_reuse_verified']:,} 次。若所有候选摘要缺口均补齐，上界为 {budget['new_extractions_if_all_candidate_abstracts_recovered']:,} 次；最终原创研究类型筛选可能降低上述数量。", '',
        f"现有摘要范围最多产生 {budget['max_claims_current_abstract_scope']:,} 条 claim。按现有 {budget['embedding_dimensions']} 维 float32 向量，仅向量矩阵上限约 {budget['embedding_float32_gib_upper']:.1f} GiB，另需索引、原文和中间文件空间；目标邻域边上限 {budget['target_neighbor_rows_upper']:,}。禁止为全量运行默认一次性装入 GPU。", '',
        '## 条件预算（不是实测报价）', '',
        '输入 token 按现有摘要提取 prompt 和抽查摘要长度，以字符数／4近似；输出、重试及耗时为显式情景假设。尚未运行模型基准，不含未计价的推理 token、GPU 编码／建图、限流和人工成本。', '',
        '| 情景 | 尝试次数 | 输入百万 token | 输出百万 token | 假设单次秒数 | 16 并发理想天数 |',
        '|---|---:|---:|---:|---:|---:|'])
    for row in scenarios:
        lines.append(f"| {row['scenario']} | {row['attempts']:,} | {row['input_million_tokens']:,.1f} | {row['output_million_tokens']:,.1f} | {row['seconds_per_attempt_assumed']} | {row['wall_days_idealized']:.1f} |")
    lines.extend(['', '货币预算公式：输入百万 token × P_in + 输出百万 token × P_out。P_in／P_out 为最终模型实际计费单价；当前采用 CLI 的模型路由不能直接等同为已确认 API 单价。先做 300 篇统一试运行，再给出实测耗时和所选模型报价；本阶段不伪造金额。', '',
        '人工核验：300 篇×2 名审阅者；若每篇每人 5–10 分钟，约 50–100 人时，另加争议裁决（假设预算，非实测）。', '',
        '## 后续处理优先级', '',
        '1. 先处理 Nature 的文献类型：抽查中同时存在新闻／职业内容和无摘要的真实研究，不能“有摘要即研究、无摘要即新闻”。全部 11,089 条候选已落地，逐出版社类型核验后锁定主刊研究论文分母。',
        '2. 三领域先落地元数据清单、做 DOI／版本去重和类型核验；现阶段是全期刊聚合名录，不是三领域每篇论文已下载。各队列的缺摘要、缺机构、缺 DOI 数量见 supplement_requirements.csv。',
        '3. 对有摘要但没有机构的论文优先补全机构信息，随后处理作者部分映射、大学层级和历史品牌归属。',
        '4. 按当前规模，直接全量提取是百万级调用。先以 300 篇试运行核实质量和吞吐；再明确选择全量版或事先定义抽样概率的全期刊分层样本版，二者不混用。', '',
        '## 文件与复现', '',
        '- snapshot_audit.json／snapshot_files.csv／snapshot_schema_probes.json：本地文件清单、版本与字段探查。',
        '- canonical_target_works_ids_restored.parquet／restored_id_mapping.csv：旧表 ID 恢复副本。',
        '- legacy_papers_current_metadata.parquet／metadata_version_differences.csv：全部旧论文的当前元数据及版本差异。',
        '- journal_registry.csv／field_registry.csv：逐队列期刊全集、字段及当前品牌信息。',
        '- nature_candidate_roster.parquet／CSV：Nature 候选逐论文清单；parquet 含已取得摘要及作者机构明细。',
        '- coverage_by_cohort_year.csv／cohort_flow.csv／supplement_requirements.csv：分年度覆盖、类型分布与补充量。',
        '- history_sampling_capacity.csv／workload_budget.json／budget_scenarios.csv：历史采样容量及预算。',
        '- source_manifest.json／api_raw/：查询来源、时间、原始响应及完整性核对。', '',
        '```bash', 'python3 -m experiments.fig8_extension.stage0 --workers 6',
        'python3 -m experiments.fig8_extension.stage0_roster', 'python3 -m experiments.fig8_extension.stage0_current_ids',
        'python3 -m experiments.fig8_extension.stage0_report', '```', '',
        'API 分组使用 cursor 直至终止，并检查各分组加总。查询缓存按参数保存，重复运行复用同一响应；更新数据版本应使用新目录。原始 data/claim_graph 保持只读。'])
    (DEFAULT_OUT / '阶段一结果与预算.md').write_text('\n'.join(lines) + '\n')


def main() -> None:
    items = read('aggregates.json')
    reconcile_missing_groups()
    sampled_coverage()
    check = id_spotcheck()
    budget = workload(items)
    scenarios = budget_scenarios(budget)
    supplemental_counts(items, budget)
    report(budget, scenarios, check)
    print(json.dumps(budget, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
