"""Prepare C/D source-backed tables and approval packets, entirely offline.

Run: python -m figure_pipeline.fig3_extensions.prepare_cd
No model client, subprocess, network, or figure renderer is imported.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .cd_approval import finalize_source_readiness
from .cd_contracts import EFFORTS, PROMPTS, SCHEMAS
from .cd_sources import (METHODS, OUTPUT, SOURCE, PaperSources, chunks, read,
                         stage_path, write_json, write_rows)


def keyed(rows: list[dict], key: str) -> dict[str, dict]:
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key}')
    return result


def method_material(paper: str) -> dict:
    material = {}
    for method in METHODS:
        report = read(stage_path(paper, 'full' if method == 'fusion' else method))
        extracted = read(stage_path(paper, 'extract', method))
        material[method] = {'report': report, 'units': keyed(extracted['units'], 'unit_id'),
                            'predictions': keyed(extracted['predictions'], 'core_id'),
                            'support': keyed(read(stage_path(paper, 'support', method))['units'], 'unit_id'),
                            'novelty': keyed(read(stage_path(paper, 'novelty', method))['items'], 'core_id'),
                            'concerns': keyed(read(stage_path(paper, 'concerns', method))['matches'], 'concern_id')}
    return material


class Preparation:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = defaultdict(list)
        self.packets: list[dict] = []
        self.seen_reviews: dict[tuple, str] = {}
        self.config = read(Path(__file__).parents[1] / 'fig3_revision/config.json')

    def run_paper(self, paper: str) -> None:
        sources = PaperSources(paper, self.tables['evidence_index'])
        methods = method_material(paper)
        cores = read(stage_path(paper, 'core'))
        references = keyed(read(stage_path(paper, 'reference'))['items'], 'core_id')
        checklist = read(stage_path(paper, 'checklist'))
        sections = keyed(read(stage_path(paper, 'review_sections'))['sections'], 'section_id')
        concerns = [c for c in checklist['concerns'] if c['dimension'] == 'novelty']
        mapping = keyed(cores['mapping'], 'core_id')
        self.tables['papers'].append({'id': paper, 'paper_id': paper,
            'core_count': len(cores['items']), 'novelty_concern_count': len(concerns),
            'review_availability': 'candidates_present' if concerns else 'no_explicit_novelty_in_existing_checklist',
            'unresolved_source_sections': checklist.get('unresolved_source_sections', []),
            'manuscript_path': sources.data['manuscript_path'], 'review_path': sources.data['review_path']})
        for core in cores['items']:
            cid = core['core_id']
            self.tables['c_cores'].append({'id': f'{paper}/{cid}', 'paper_id': paper, **core,
                'claim_mapping': mapping.get(cid), 'manuscript_evidence_id': sources.original('manuscript', core['manuscript_quote']),
                'reference': references[cid], 'reference_evidence_ids': sources.refs(references[cid]['evidence']),
                'review_summary': {'state': 'pending' if concerns else 'no_explicit_novelty_in_existing_checklist',
                                   'not_equivalent_to_review_disapproval': True},
                'method_ids': [f'{paper}/{cid}/{m}' for m in METHODS]})
            self.core_methods(sources, core, methods)
        for concern in concerns:
            canonical = self.review_record(sources, concern, sections)
            for core in cores['items']:
                self.tables['c_review_links'].append({'id': f'{paper}/{core["core_id"]}/{concern["concern_id"]}',
                    'paper_id': paper, 'core_id': core['core_id'], 'concern_id': concern['concern_id'],
                    'canonical_concern_id': canonical, 'match': None, 'status': 'pending',
                    'strict_applicability_from_existing_record': concern['applies_to_input'] == 'yes'})
            self.concern_reuse(sources, concern, methods)
            if canonical == concern['concern_id']:
                self.c_packets(sources, cores['items'], concern, sections)
        self.d_candidates(sources, methods)

    def core_methods(self, sources: PaperSources, core: dict, methods: dict) -> None:
        paper, cid = sources.paper, core['core_id']
        for method, data in methods.items():
            prediction, novelty = data['predictions'][cid], data['novelty'][cid]
            quotes = prediction['quotes'] + prediction.get('conflicting_quotes', [])
            self.tables['c_methods'].append({'id': f'{paper}/{cid}/{method}', 'paper_id': paper,
                'core_id': cid, 'method': method, 'prediction': prediction,
                'report_evidence_ids': [sources.report(method, q, data['report']['body']) for q in quotes],
                'existing_novelty_evaluation': novelty, 'source_evidence_ids': sources.refs(novelty['evidence']),
                'existing_effective_increment': novelty.get('effective_increment'),
                'recognized_contribution_identified': None, 'review_reason_coverage': None,
                'supplementary_status': 'pending_core_review_alignment'})

    def review_record(self, sources: PaperSources, concern: dict, sections: dict) -> str:
        paper, cid = sources.paper, concern['concern_id']
        section_ids = [s for s in sections if s in concern['section_id'].replace(',', ' ').split()]
        # IDs can also be joined using slashes; never infer reviewer identity from text.
        if not section_ids:
            import re
            section_ids = [s for s in re.findall(r'R\d{4}', concern['section_id']) if s in sections]
        origins = [{k: sections[s].get(k) for k in ('section_id', 'role', 'reviewer_id', 'round_number',
                                                   'manuscript_scope', 'identity_explicit', 'source_region')} for s in section_ids]
        key = (paper, tuple(sorted(section_ids)), concern['reviewer_id'], concern['round_number'],
               concern['quote'], concern['scope'], concern['stance'], tuple(concern['reason_quotes']))
        canonical = self.seen_reviews.setdefault(key, cid)
        self.tables['c_review_records'].append({'id': f'{paper}/{cid}', 'paper_id': paper, **concern,
            'canonical_concern_id': canonical, 'source_origins': origins,
            'round_status': 'unknown' if concern['round_number'] == 0 else 'recorded',
            'review_evidence_id': sources.original('review', concern['quote']),
            'reason_evidence_ids': [sources.original('review', q) for q in concern['reason_quotes']],
            'source_role_ready': bool(origins) and all(s['role'] == 'reviewer' for s in origins),
            'applicability_for_strict_analysis': concern['applies_to_input'] == 'yes'})
        return canonical

    def concern_reuse(self, sources: PaperSources, concern: dict, methods: dict) -> None:
        for method, data in methods.items():
            match = data['concerns'].get(concern['concern_id'])
            if match is None:
                evidence_id, location = None, {'state': 'missing_record'}
            else:
                evidence_id = sources.report(method, match['report_quote'], data['report']['body'])
                location = sources.locator.locate(data['report']['body'], match['report_quote'])
            self.tables['c_existing_concern_matches'].append({
                'id': f'{sources.paper}/{concern["concern_id"]}/{method}', 'paper_id': sources.paper,
                'concern_id': concern['concern_id'], 'method': method, 'existing_match': match,
                'report_evidence_id': evidence_id, 'report_quote_location': location['state'],
                'reuse_state': 'pending_core_and_reason_scope',
                'reuse_blockers': ['core_alignment_not_evaluated', 'reason_scope_not_evaluated'] +
                    ([] if location['state'] in {'exact', 'normalized_whitespace_unicode'} else ['report_quote_not_located'])})

    def c_packets(self, sources: PaperSources, cores: list[dict], concern: dict, sections: dict) -> None:
        selected_cores = [{k: c[k] for k in ('core_id', 'description', 'manuscript_quote')} for c in cores]
        review = {k: concern[k] for k in ('concern_id', 'reviewer_id', 'round_number', 'quote',
                                         'object_description', 'scope', 'manuscript_scope')}
        origin = self.tables['c_review_records'][-1]['source_origins']
        payload = {'paper_id': sources.paper, 'cores': selected_cores, 'review': review, 'source_origins': origin}
        if len(json.dumps(payload, ensure_ascii=False)) <= 10000:
            self.packet('c_match', f'{sources.paper}/{concern["concern_id"]}', payload)
            return
        # Preserve every character; a fragment cannot establish a whole-record negative match.
        for core in selected_cores:
            for ci, core_part in enumerate(chunks(core['manuscript_quote'])):
                for ri, review_part in enumerate(chunks(concern['quote'])):
                    fragment = {'paper_id': sources.paper, 'cores': [{**core, 'manuscript_quote': core_part['text']}],
                                'review': {**review, 'quote': review_part['text']}, 'source_origins': origin,
                                'fragment': {'core_range': [core_part['start'], core_part['end']],
                                             'review_range': [review_part['start'], review_part['end']],
                                             'requires_reconciliation': True}}
                    self.packet('c_match', f'{sources.paper}/{concern["concern_id"]}/{core["core_id"]}/{ci}-{ri}', fragment)
        self.defer(f'C/reconcile/{sources.paper}/{concern["concern_id"]}', 'c_fragment_reconciliation',
                   'all_fragments_complete', 'Complete evidence across fragments required before a final match.')

    def packet(self, task: str, suffix: str, payload: dict) -> None:
        ident = task+'/'+suffix
        schema = SCHEMAS[task].model_json_schema()
        material = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        request = PROMPTS[task]+'\nINPUT:\n'+material+json.dumps(schema, ensure_ascii=False)
        if len(material) > 12000 or len(request) > 24000 or len(request.encode()) > 64000:
            raise ValueError(f'Further offline splitting required: {ident}')
        self.packets.append({'id': ident, 'task': task, 'model': 'gpt-5.6-luna',
            'effort': EFFORTS[task], 'status': 'awaiting_user_approval', 'payload': payload,
            'prompt_file': f'prompts/{task}.txt', 'schema_file': f'schemas/{task}.json',
            'material_chars': len(material), 'request_chars': len(request),
            'request_bytes': len(request.encode()), 'estimated_visible_input_tokens': None})

    def defer(self, ident: str, task: str, dependency: str, reason: str) -> None:
        self.tables['deferred_tasks'].append({'id': ident, 'task': task, 'status': 'not_ready',
                                             'dependency': dependency, 'reason': reason})

    def d_candidates(self, sources: PaperSources, methods: dict) -> None:
        data = read(stage_path(sources.paper, 'clusters'))
        eligible = set(data['eligible_unit_keys'])
        importance = keyed(data['importance'], 'cluster_id')
        for cluster in data['clusters']:
            members = {data['mapping'][u].split('/')[0] for u in cluster['unit_keys'] if u in eligible}
            if 'graph' not in members or 'gear' in members:
                continue
            ident = f'{sources.paper}/{cluster["cluster_id"]}'
            units, evidence_ids = [], []
            for key in cluster['unit_keys']:
                method, uid = data['mapping'][key].split('/')
                if method not in METHODS:
                    continue
                unit = methods[method]['units'][uid]
                support = methods[method]['support'][uid]
                source_ids = sources.refs(support['evidence'])
                evidence_ids.extend(source_ids)
                units.append({'method': method, 'cluster_unit_key': key, 'unit': unit,
                    'existing_support': support, 'eligible': key in eligible,
                    'report_evidence_id': sources.report(method, unit['quote'], methods[method]['report']['body']),
                    'source_evidence_ids': source_ids})
            self.tables['d_candidates'].append({'id': ident, 'paper_id': sources.paper, **cluster,
                'eligible_methods': sorted(members & set(METHODS)), 'units': units,
                'existing_importance': importance[cluster['cluster_id']],
                'gear_mentioned_in_cluster': any(u['method'] == 'gear' for u in units),
                'source_evidence_ids': list(dict.fromkeys(evidence_ids)),
                'classification': None, 'status': 'pending_semantic_triage'})
            self.d_trace_sources(sources, cluster, units, methods)
            statements = [{'unit_id': u['unit']['unit_id'], 'quote': u['unit']['quote'],
                           'claim_ids': u['unit']['claim_ids']} for u in units if u['method'] == 'graph']
            groups, group = [], []
            for statement in statements:
                for part in chunks(statement['quote'], 4000):
                    item = {**statement, 'quote': part['text'], 'quote_range': [part['start'], part['end']]}
                    if group and len(json.dumps(group+[item], ensure_ascii=False)) > 7500:
                        groups.append(group)
                        group = []
                    group.append(item)
            if group:
                groups.append(group)
            for i, group in enumerate(groups):
                self.packet('d_triage', f'{ident}/{i}', {'paper_id': sources.paper,
                    'cluster_id': cluster['cluster_id'], 'statements': group,
                    'fragment_index': i, 'fragment_count': len(groups),
                    'method_labels_hidden': True, 'previous_evaluations_hidden': True})
            if len(groups) > 1:
                self.defer('D/reconcile/'+ident, 'd_fragment_reconciliation', 'all_cluster_fragments_complete',
                           'Reconcile duplicate or mixed relations across fragments before verification.')
            for task, dependency in [('d_verify', 'd_triage_complete'), ('d_trace', 'd_verify_complete')]:
                self.defer(task+'/'+ident, task, dependency,
                           'Relation count and scoped evidence packets depend on triage; not executable yet.')

    def d_trace_sources(self, sources: PaperSources, cluster: dict, units: list[dict], methods: dict) -> None:
        claims = sorted({c for u in units if u['method'] == 'graph' for c in u['unit']['claim_ids']})
        full = methods['fusion']['report']
        retention = [r for r in full.get('finding_retention', []) if r.get('claim_id') in claims]
        details = full.get('fusion_details', [])
        if isinstance(details, list):
            details = [r for r in details if r.get('claim_id') in claims]
        for record in retention:
            for item in record.get('outcomes', []):
                if item.get('report_quote'):
                    sources.report('fusion', item['report_quote'], full['body'])
        self.tables['d_processing_trace'].append({'id': f'{sources.paper}/{cluster["cluster_id"]}',
            'paper_id': sources.paper, 'cluster_id': cluster['cluster_id'], 'claim_ids': claims,
            'candidate_finding_retention': retention, 'candidate_fusion_details': details,
            'candidate_scope_only': True, 'finding_and_relation_match': 'pending',
            'gear_state': None, 'full_state': None, 'recorded_reason': None,
            'report_paths': {m: str(stage_path(sources.paper, 'full' if m == 'fusion' else m)) for m in METHODS},
            'graph_analysis_path': str(stage_path(sources.paper, 'graph')),
            'status': 'pending_relation_verification_and_scope_alignment'})


def historical_call_stats() -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    path = SOURCE / 'derived/model_calls.csv'
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row['state'] == 'completed' and row['task'] in {'concerns', 'clusters', 'support'}:
                grouped[row['task']].append(row)
    result = {}
    for task, rows in grouped.items():
        seconds = sorted(float(r['seconds']) for r in rows)
        usage = [json.loads(r['usage']) for r in rows if r.get('usage') and r['usage'] != 'null']
        entry = {'n_completed': len(rows), 'median_seconds': seconds[len(seconds)//2],
                 'p90_seconds': seconds[min(len(seconds)-1, int(len(seconds)*.9))]}
        for key in ('input_tokens', 'output_tokens', 'reasoning_output_tokens'):
            values = sorted(u[key] for u in usage if u.get(key) is not None)
            entry['median_'+key] = values[len(values)//2] if values else None
        result[task] = entry
    return result


def visible_tokens(text: str) -> tuple[int, int]:
    """Heuristic range, explicitly not an exact tokenizer count or bill."""
    ascii_count = sum(ord(c) < 128 for c in text)
    other = len(text)-ascii_count
    return math.ceil(ascii_count/5+other*.7), math.ceil(ascii_count/3+other*1.8)


def budget(preparation: Preparation) -> dict:
    historical = historical_call_stats()
    groups = {}
    for task in ('c_match', 'd_triage'):
        rows = [r for r in preparation.packets if r['task'] == task]
        low = high = 0
        for row in rows:
            text = PROMPTS[task]+json.dumps(row['payload'], ensure_ascii=False)+json.dumps(SCHEMAS[task].model_json_schema())
            token_range = visible_tokens(text)
            row['estimated_visible_input_tokens'] = {'low': token_range[0], 'high': token_range[1], 'method': 'character_heuristic'}
            low += token_range[0]
            high += token_range[1]
        proxy = historical['concerns' if task == 'c_match' else 'clusters']
        groups[task] = {'ready_calls': len(rows), 'max_calls_including_one_format_repair': len(rows)*2,
            'visible_input_tokens_range': [low, high], 'historical_task_proxy': proxy,
            'historical_usage_projection_input_tokens': len(rows)*(proxy['median_input_tokens'] or 0),
            'historical_usage_projection_output_tokens': len(rows)*(proxy['median_output_tokens'] or 0),
            'ideal_minutes_at_16_workers_median_to_p90': [len(rows)*proxy[k]/16/60 for k in ('median_seconds', 'p90_seconds')],
            'ideal_minutes_at_64_workers_median_to_p90': [len(rows)*proxy[k]/64/60 for k in ('median_seconds', 'p90_seconds')]}
    return {'model': 'gpt-5.6-luna', 'timeout_seconds': 900, 'initial_concurrency': 16,
        'worker_and_cli_ceiling': 64, 'resource_limits': {k: preparation.config[k] for k in
          ('memory_reserve_gib', 'task_memory_gib', 'workers_per_cpu')},
        'network_retrieval': False, 'automatic_failed_call_retries': 0, 'format_repair_limit': 1,
        'authorization': 'not_authorized_no_calls_made', 'ready_stages': groups,
        'ready_calls_total': len(preparation.packets), 'maximum_initial_calls': len(preparation.packets)*2,
        'deferred_task_records': len(preparation.tables['deferred_tasks']),
        'deferred_call_count': None,
        'deferred_count_reason': 'Relation multiplicity, valid review matches, fragmented reconciliation and missing quotes are not yet judged; no defensible fixed total.',
        'c_reason_object_upper_bound': len(preparation.tables['c_review_links'])*3,
        'd_source_candidate_count': len(preparation.tables['d_candidates']),
        'cost_currency': None, 'cost_reason': 'No verified unit price; visible tokens exclude CLI overhead and reasoning. Historical projections are workload proxies, not exact cost.',
        'time_caveat': 'Ideal throughput only; excludes queueing, rate limits, repairs, stragglers and downstream stages. No promised completion time.'}


def stats(preparation: Preparation) -> dict:
    tables = preparation.tables
    evidence = tables['evidence_index']
    return {'status': 'offline_preparation_complete_semantic_evaluation_pending',
        'counts': {k: len(v) for k, v in tables.items()},
        'C': {'papers': len(tables['papers']), 'core_count': len(tables['c_cores']),
              'reference_states': dict(Counter(r['reference']['state'] for r in tables['c_cores'])),
              'review_stances': dict(Counter(r['stance'] for r in tables['c_review_records'])),
              'review_applicability': dict(Counter(r['applies_to_input'] for r in tables['c_review_records'])),
              'unknown_review_round': sum(r['round_number'] == 0 for r in tables['c_review_records']),
              'papers_without_novelty_records': [r['paper_id'] for r in tables['papers'] if not r['novelty_concern_count']],
              'unresolved_source_sections': sum(len(r['unresolved_source_sections']) for r in tables['papers']),
              'existing_concern_quote_locations': dict(Counter(r['report_quote_location'] for r in tables['c_existing_concern_matches'])),
              'semantic_matches_completed': 0},
        'D': {'candidate_count': len(tables['d_candidates']),
              'candidate_papers': len({r['paper_id'] for r in tables['d_candidates']}),
              'original_types': dict(Counter(r['kind'] for r in tables['d_candidates'])),
              'full_eligible': sum('fusion' in r['eligible_methods'] for r in tables['d_candidates']),
              'gear_mentioned': sum(r['gear_mentioned_in_cluster'] for r in tables['d_candidates']),
              'scientific_relation_count': None, 'verified_relation_count': None},
        'evidence_location_states': dict(Counter(r['location']['state'] for r in evidence)),
        'pending_metrics': ['C reference x prediction x review stance', 'C recognized contribution coverage',
                            'C reviewer reason coverage', 'D scientific relation support', 'D Full verified relation retention']}


def checks(preparation: Preparation) -> None:
    for name, rows in preparation.tables.items():
        keyed(rows, 'id')
    papers = preparation.tables['papers']
    if len(papers) != 100 or sum(r['core_count'] for r in papers) != 200:
        raise ValueError('Unexpected roster/core population; do not silently change denominators')
    if len(preparation.tables['c_review_links']) != 870 or len(preparation.tables['d_candidates']) != 261:
        raise ValueError('Candidate population differs from agreed scope')
    if len(preparation.tables['c_methods']) != 600:
        raise ValueError('Missing method/core record')
    if not all(r['match'] is None for r in preparation.tables['c_review_links']):
        raise ValueError('Offline preparation cannot manufacture semantic matches')
    for packet in preparation.packets:
        if packet['task'] == 'c_match' and set(packet['payload']) & {'reference', 'methods', 'predictions'}:
            raise ValueError('Method/reference leakage in review matching')


def write_report(summary: dict, estimate: dict) -> None:
    c, d = summary['C'], summary['D']
    lines = ['# C、D 离线准备与模型调用商议材料', '',
      '**模型调用数：0。所有语义匹配、关系分类、关系核验和Full处理结论尚未执行。**', '',
      '## 已准备数据', '',
      f'- C：{c["core_count"]}项贡献、600条方法记录、435条审稿候选、870个贡献—审稿候选对。',
      f'- D：{d["candidate_count"]}个候选簇，来自{d["candidate_papers"]}篇；Full原有有效簇141个，GEAR有相关表述81个。',
      f'- 引文定位状态：{json.dumps(summary["evidence_location_states"], ensure_ascii=False)}。',
      f'- 既有问题匹配报告引文：{json.dumps(c["existing_concern_quote_locations"], ensure_ascii=False)}。',
      '- 未定位到不等于引文虚假；换行、摘录形式、来源粒度或实际错误均可能造成缺口，未自动改写。',
      '- 仅空白/Unicode规范化匹配保留原文偏移；没有模糊语义匹配，也没有自动恢复引文。', '',
      '## 可提交的第一阶段调用（目前尚未授权）', '',
      '| 任务 | 初始调用数 | 含一次格式修复的上限 | 可见输入Token估算 |',
      '|---|---:|---:|---:|']
    for task, row in estimate['ready_stages'].items():
        low, high = row['visible_input_tokens_range']
        lines.append(f'| {task} | {row["ready_calls"]} | {row["max_calls_including_one_format_repair"]} | {low:,}–{high:,} |')
    lines += ['', '- c_match：每条审稿意见与同篇两项贡献逐项对应。隐藏系统和历史参考结论；输出不是新收集的人工标注。',
      '- d_triage：从原始报告断言区分具体关系、图结构和边界提醒。旧类型、支持和重要性标签不进入输入。',
      '- 模型固定gpt-5.6-luna；c_match为xhigh，d_triage为high。900秒超时、初始16并发、资源允许时上限64。',
      '- 不自动重试失败调用；至多一次格式修复计入总上限。预算不包括后续阶段；超出范围重新商议。',
      '- 估算Token采用字符启发式，不是精确分词；CLI上下文和推理开销另见历史调用统计，不能直接据此推算金额。', '',
      '## 耗时依据', '']
    for task, row in estimate['ready_stages'].items():
        times = row['ideal_minutes_at_16_workers_median_to_p90']
        fast = row['ideal_minutes_at_64_workers_median_to_p90']
        lines.append(f'- {task}：按同类历史单次耗时估算，16并发理想吞吐约{times[0]:.0f}–{times[1]:.0f}分钟；64并发约{fast[0]:.0f}–{fast[1]:.0f}分钟。')
    lines += ['- 以上仅吞吐估算，不含排队、限流、长尾、修复及后续任务，不能当作完成承诺。', '',
      '## 后续阶段与边界', '',
      '- 分片任务须合并核对全部片段后才可生成整体判断；片段中没找到不能判整段不存在。',
      '- C理由复用须先完成贡献对应和理由范围判断。最多2610个方法—候选对需考虑，不等于2610次调用；不相关候选不进入后续评价。',
      '- D关系数量由分类产生，不能预先把261簇当作261条科学关系。关系核验和Full追踪任务当前仅有依赖记录。',
      '- 新关系核验只能采用原始稿件/历史证据；图结构和旧模型判断只能作为定位背景。只有摘要不能推断全文不存在。',
      '- 具体关系数、证据有效数和保留率目前为null，不是0；主要结果表需获准评价后才生成。',
      '- 归因仅限现有候选，不能证明Graph因果贡献，也不提供全体关系召回率。', '',
      '## 文件与运行', '',
      '- tables/：底表、引用位置、候选对应、旧结果复用条件和待执行依赖；JSONL按论文范围ID保存。',
      '- packets/：完整待调用输入；prompts/与schemas/：五类任务完整指令与响应格式。',
      '- examples/：每种可执行任务一个真实输入样例。',
      '- statistics.json、call_budget.json、descriptive_statistics.csv：当前可计算统计与预算。',
      '- 后续关系与方法理由提示词已准备，完整输入须等待前序结果；未提前伪造答案。',
      '- 命令：`/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig3_extensions.prepare_cd`。',
      '- 程序仅离线整理；不包含执行模型的入口，不导入当前Fig3绘图程序。已存在新增记录按ID复用，不覆盖。',
      '- 证据索引中的json_pointer为来源定位描述；blocks/source ID需结合所属论文解析，不能跨论文连接。',
      '- 当前只建议商议第一阶段调用。批准第一阶段不授权未知数量的后续关系核验、引文恢复或报告追踪。', '']
    path = OUTPUT / 'README.md'
    if not path.exists():
        with path.open('x') as stream:
            stream.write('\n'.join(lines))


def save(preparation: Preparation, summary: dict, estimate: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, rows in preparation.tables.items():
        write_rows(OUTPUT / 'tables' / f'{name}.jsonl', rows)
    write_rows(OUTPUT / 'tables/d_relations.jsonl', [])
    for task, prompt in PROMPTS.items():
        path = OUTPUT / 'prompts' / f'{task}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('x') as stream:
                stream.write(prompt+'\n')
        write_json(OUTPUT / 'schemas' / f'{task}.json', SCHEMAS[task].model_json_schema())
    for task in ('c_match', 'd_triage'):
        rows = [r for r in preparation.packets if r['task'] == task]
        write_rows(OUTPUT / 'packets' / f'{task}.jsonl', rows)
        write_json(OUTPUT / 'examples' / f'{task}.json', rows[0])
    write_json(OUTPUT / 'statistics.json', summary)
    write_json(OUTPUT / 'call_budget.json', estimate)
    path = OUTPUT / 'descriptive_statistics.csv'
    if not path.exists():
        with path.open('x') as stream:
            writer = csv.writer(stream)
            writer.writerow(['group', 'category', 'count', 'denominator', 'unit'])
            for kind, count in summary['C']['reference_states'].items():
                writer.writerow(['C_reference', kind, count, 200, 'core'])
            for kind, count in summary['C']['review_stances'].items():
                writer.writerow(['C_review_candidates', kind, count, 435, 'concern'])
            for kind, count in summary['D']['original_types'].items():
                writer.writerow(['D_original_type', kind, count, 261, 'cluster'])
    write_report(summary, estimate)


def main() -> None:
    preparation = Preparation()
    papers = sorted(p.stem for p in (SOURCE/'annotations/core/papers').glob('*.json'))
    for i, paper in enumerate(papers, 1):
        preparation.run_paper(paper)
        if i % 10 == 0:
            print(f'Prepared {i}/{len(papers)} papers', flush=True)
    # Subsequent semantic work is conditional, never submitted by this program.
    for link in preparation.tables['c_review_links']:
        preparation.defer('c_reason/'+link['id'], 'c_reason', 'c_match_same_scope',
                          'Three method records considered only after a valid core/review alignment; quotes alone do not establish reason scope.')
    checks(preparation)
    estimate = budget(preparation)
    summary = stats(preparation)
    save(preparation, summary, estimate)
    approval = finalize_source_readiness(preparation.tables, preparation.packets, estimate)
    print(json.dumps({'statistics': summary, 'calls': estimate['ready_calls_total'],
                      'call_cap': estimate['maximum_initial_calls'], 'source_ready_approval': approval}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
