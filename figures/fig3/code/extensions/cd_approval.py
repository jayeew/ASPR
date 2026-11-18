"""Source readiness narrows prepared requests before any user approval."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .cd_sources import OUTPUT, write_json, write_rows


def finalize_source_readiness(tables: dict, packets: list[dict], estimate: dict) -> dict:
    evidence = {r['id']: r for r in tables['evidence_index']}
    reviews = {r['id']: r for r in tables['c_review_records']}
    candidates = {r['id']: r for r in tables['d_candidates']}
    readiness, gaps = [], []
    for packet in packets:
        payload = packet['payload']
        blockers, warnings, source_ids = [], [], []
        if packet['task'] == 'c_match':
            review = reviews[f'{payload["paper_id"]}/{payload["review"]["concern_id"]}']
            source_ids = [review['review_evidence_id']]
            if evidence[source_ids[0]]['location']['state'] not in {'exact', 'normalized_whitespace_unicode'}:
                blockers.append('review_quote_not_fully_located')
                gaps.append({'id': 'review_source/'+review['id'], 'paper_id': review['paper_id'],
                             'concern_id': review['concern_id'], 'evidence_id': source_ids[0],
                             'status': 'pending_source_recovery_discussion',
                             'source_origins': review['source_origins'], 'automatic_repair_performed': False})
            if not review['source_role_ready']:
                blockers.append('reviewer_role_unresolved')
            if review['applies_to_input'] != 'yes':
                warnings.append('exclude_from_strict_analysis_until_applicability_resolved')
            if not all(r['identity_explicit'] for r in review['source_origins']):
                warnings.append('reviewer_identity_not_explicit_keep_uncertainty')
            if review['round_number'] == 0:
                warnings.append('round_unknown_do_not_infer_first_round')
        else:
            candidate = candidates[f'{payload["paper_id"]}/{payload["cluster_id"]}']
            source_ids = [u['report_evidence_id'] for u in candidate['units'] if u['method'] == 'graph']
            if any(evidence[e]['location']['state'] not in {'exact', 'normalized_whitespace_unicode'} for e in source_ids):
                blockers.append('report_statement_not_located')
        readiness.append({'id': packet['id'], 'task': packet['task'],
                          'ready_for_approval': not blockers, 'blockers': blockers, 'warnings': warnings,
                          'source_evidence_ids': source_ids,
                          'execution_authorized': False})
    counts = Counter(r['task'] for r in readiness if r['ready_for_approval'])
    rows_by_id = {r['id']: r for r in packets}
    approval = {'authorization': 'awaiting_explicit_user_confirmation', 'model_calls_made': 0,
                'enumerated_packets': len(packets), 'source_ready_calls': sum(counts.values()),
                'held_packets': sum(not r['ready_for_approval'] for r in readiness),
                'maximum_calls_with_format_repairs': sum(counts.values())*2,
                'automatic_failure_retries': 0, 'format_repair_limit_per_request': 1,
                'model': 'gpt-5.6-luna', 'initial_concurrency': 16, 'max_concurrency': 64,
                'timeout_seconds': 900, 'stages': {}}
    for task, count in counts.items():
        packets_for_task = [rows_by_id[r['id']] for r in readiness if r['task'] == task and r['ready_for_approval']]
        proxy = estimate['ready_stages'][task]['historical_task_proxy']
        approval['stages'][task] = {'calls': count, 'max_including_format_repairs': count*2,
            'visible_input_tokens_range': [sum(r['estimated_visible_input_tokens'][k] for r in packets_for_task) for k in ('low', 'high')],
            'ideal_minutes_16_workers': [count*proxy[k]/16/60 for k in ('median_seconds', 'p90_seconds')],
            'ideal_minutes_64_workers': [count*proxy[k]/64/60 for k in ('median_seconds', 'p90_seconds')],
            'historical_input_token_projection': count*(proxy['median_input_tokens'] or 0),
            'historical_output_token_projection': count*(proxy['median_output_tokens'] or 0)}
    approval['exclusions'] = ['15 unlocated review excerpts', 'C method/reviewer reason judgment',
                              'D original-evidence relation verification', 'D method trace',
                              'any source recovery, external retrieval, or plotting']
    write_rows(OUTPUT/'tables/source_readiness.jsonl', readiness)
    write_rows(OUTPUT/'tables/source_gaps.jsonl', gaps)
    write_json(OUTPUT/'approval_request.json', approval)
    write_approval_note(approval)
    return approval


def write_approval_note(approval: dict) -> None:
    lines = ['# 本次可商议的调用范围', '',
        '**尚未调用模型。此文件以原文定位结果筛选候选请求；实际提交必须同时满足 source_readiness.jsonl。**', '',
        f'已枚举{approval["enumerated_packets"]}个请求，其中{approval["source_ready_calls"]}个具备来源定位条件；{approval["held_packets"]}个暂缓。', '',
        '| 阶段 | 可提交调用 | 包含一次格式修复的上限 |', '|---|---:|---:|']
    for task, row in approval['stages'].items():
        lines.append(f'| {task} | {row["calls"]} | {row["max_including_format_repairs"]} |')
    lines += ['', f'总初始调用{approval["source_ready_calls"]}次，格式修复最多每请求1次，总上限{approval["maximum_calls_with_format_repairs"]}次；失败不自动重试。',
      '', '模型为gpt-5.6-luna。C对应判断使用xhigh，D内容分类使用high；初始16并发，资源允许时上限64，900秒超时。',
      '', '15条未完整定位的审稿摘录保留在底表和source_gaps中；相关30个贡献—审稿候选对继续pending。不会把它们当作未匹配、未认可或错误。',
      '', '4条身份未明确的审稿记录及未知轮次记录保留身份/轮次不确定性，不用未知身份进行去重或推断轮次；适用性不确定的3条不能直接进入严格统计。',
      '', '当前scope不包括关系支持核验、Full处理判断、C理由覆盖、引文恢复或任何检索。这些需要前序结果才能确定输入和实际预算，届时另行商议。',
      '', '时间估计为理想吞吐：']
    for task, row in approval['stages'].items():
        a,b = row['ideal_minutes_16_workers']; c,d = row['ideal_minutes_64_workers']
        lines.append(f'- {task}：16并发约{a:.0f}–{b:.0f}分钟；64并发约{c:.0f}–{d:.0f}分钟。')
    lines += ['', '不含排队、限流、长尾和修复时间。可见输入Token是字符启发式范围；CLI额外上下文和推理输出开销可很大，历史用量推估保存在approval_request.json，均不是精确账单。',
      '', '提交前必须得到用户对以上范围的明确确认，含试跑。未得到确认不得调用客户端。', '']
    path = OUTPUT/'APPROVAL.md'
    if not path.exists():
        with path.open('x') as stream:
            stream.write('\n'.join(lines))
