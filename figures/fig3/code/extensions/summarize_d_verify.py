"""Publish source-bound D verification results without downstream model calls."""
from __future__ import annotations

import csv
import json
from collections import Counter

from .cd_sources import OUTPUT, read, write_json, write_rows
from .run_cd_first import rows
from .run_d_verify import RUN, packets, path_for


def usage() -> dict:
    states,tokens = Counter(),Counter()
    repairs=0
    for p in (RUN/'logs/calls').glob('*/record.json'):
        r=read(p);states[r['state']]+=1;repairs+=int(r.get('repair',0)>0)
        for k,v in (r.get('usage') or {}).items():
            if isinstance(v,(int,float)):tokens[k]+=v
    return {'actual_calls':sum(states.values()),'states':dict(states),'format_repairs':repairs,'usage':dict(tokens)}


def main() -> None:
    selected=packets()
    if any(not path_for(p).exists() for p in selected):
        raise ValueError('Approved requests are still running')
    original={r['id']:r for r in rows(OUTPUT/'first_stage/d_relation_candidates.jsonl')}
    trace={r['id']:r for r in rows(OUTPUT/'tables/d_processing_trace.jsonl')}
    verified,followup=[],[]
    for packet in selected:
        record=read(path_for(packet));candidate=original[record['relation_id']]
        answer=record['answer']
        verified.append({'id':record['relation_id'],'paper_id':candidate['paper_id'],
            'cluster_id':candidate['cluster_id'],'request_status':record['status'],
            'verified_state':answer['state'] if record['status']=='completed' else None,
            'assessment':answer,'integrity_issues':record.get('integrity_issues',[]),
            'original_relation':candidate['triage_part'],'input_evidence_ids':[e['evidence_id'] for e in packet['payload']['original_evidence']],
            'evaluation_source':'source_bound_luna_xhigh','new_human_annotation':False})
        if record['status']!='completed':continue
        source=trace[f'{candidate["paper_id"]}/{candidate["cluster_id"]}']
        followup.append({'id':record['relation_id'],'paper_id':candidate['paper_id'],'cluster_id':candidate['cluster_id'],
            'evidence_verdict':answer['state'],'scope_for_trace':answer['scope'],
            'supported_statement':answer['supported_statement'],
            'original_statement':candidate['triage_part']['relation_statement'],
            'candidate_claim_ids':source['claim_ids'],'report_paths':source['report_paths'],
            'candidate_finding_retention':source['candidate_finding_retention'],
            'candidate_fusion_details':source['candidate_fusion_details'],
            'full_state':None,'gear_state':None,'recorded_reason':None,
            'status':'pending_scoped_report_trace','model_execution_authorized':False,
            'candidate_logs_not_yet_scientifically_aligned':True})
    write_rows(RUN/'verification_results.jsonl',verified)
    write_rows(RUN/'full_trace_preparation.jsonl',followup)
    accepted=[r for r in verified if r['request_status']=='completed']
    states=Counter(r['verified_state'] for r in accepted)
    positive=[r for r in accepted if r['verified_state'] in {'supported','supported_after_narrowing'}]
    summary={'approved_requests':52,'request_statuses':dict(Counter(r['request_status'] for r in verified)),
        'usable_verdicts':dict(states),'raw_model_verdicts':dict(Counter(r['assessment']['state'] for r in verified if r['assessment'])),
        'supported_or_narrowed_relations':len(positive),
        'supported_or_narrowed_clusters':len({(r['paper_id'],r['cluster_id']) for r in positive}),
        'supported_or_narrowed_papers':len({r['paper_id'] for r in positive}),
        'original_candidate_population':76,'not_submitted_source_gap_relations':24,
        'followup_trace_records':len(followup),'full_retention_rate':None,
        'call_usage':usage(),'limitations':['Evidence excerpts are source-bound and incomplete, not exhaustive literature.',
         'Relations are drawn from Graph-valid, GEAR-ineligible cluster candidates; global relation recall is unavailable.',
         'Verified relations do not automatically establish innovation, firstness, causal benefit or Full superiority.',
         'No source recovery, new retrieval, C calls or Full trace evaluation performed.']}
    write_json(RUN/'summary.json',summary)
    output=RUN/'statistics.csv'
    if not output.exists():
        with output.open('x') as stream:
            writer=csv.writer(stream);writer.writerow(['state','count','denominator','unit'])
            for state in ('supported','supported_after_narrowing','contradicted','insufficient_material'):
                writer.writerow([state,states[state],len(accepted),'relation_with_usable_verdict'])
    lines=['# D 原文核验结果','',f'批准的52个请求已全部尝试，模型固定gpt-5.6-luna、xhigh。实际调用{summary["call_usage"]["actual_calls"]}次，格式修复{summary["call_usage"]["format_repairs"]}次。',
        '',f'请求状态：{json.dumps(summary["request_statuses"],ensure_ascii=False)}。',
        '',f'可用原文核验结论：{json.dumps(summary["usable_verdicts"],ensure_ascii=False)}。',
        '',f'得到支持或收窄后支持的关系共{len(positive)}条，涉及{summary["supported_or_narrowed_clusters"]}个原信息簇、{summary["supported_or_narrowed_papers"]}篇论文。三种单位不能混为新增创新数。',
        '', 'supported：原关系在所给原文及限定条件下得到支持；supported_after_narrowing：只能支持更窄的关系，后续Full追踪须采用收窄后的表述；contradicted：原文明确反驳；insufficient_material：现有材料不足，不能归为错误。',
        '', '原76个候选中的24个没有合格现有原文片段，本次未提交；另有输出引文或标识问题的请求保留pending，原模型回答仍完整保存。',
        '', 'verification_results.jsonl保存原关系、核验判断、引用和问题；full_trace_preparation.jsonl保存核验后的范围及已有融合记录索引。后者尚未进行关系范围对应或Full处理判断，不能读取为已保留/已遗漏。',
        '', 'Full保留率尚未计算。下一步需要确认三方法报告是否表达同一关系和限定条件；收窄/修正、明确未决与真正遗漏必须分开。取舍理由只能来自已有记录，不能事后编造。',
        '', '本次没有调用后续模型，没有新增检索或绘图；旧数据和Fig3保持原样。','']
    report=RUN/'README.md'
    if not report.exists():
        with report.open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
