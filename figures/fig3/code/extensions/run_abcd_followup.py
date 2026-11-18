"""Run approved final supplementary evaluations; never regenerate reports."""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from .abcd_followup import (DEST,PREP,SCREEN_PROMPT,FINAL_PROMPT,ScreenResult,FinalJudgment,fit)
from .cd_sources import Locator,read,write_json,write_rows
from .run_cd_first import rows
from ..fig3_revision.config import Config,load_config
from ..fig3_revision.resources import admission

RUN=DEST/'evaluation'


def result_path(packet: dict) -> Path:
    return RUN/'results'/f'{packet["id"]}.json'


def integrity(packet: dict,answer: dict) -> list[str]:
    locator=Locator();issues=[];payload=packet['payload']
    if packet['task']=='abcd_screen':
        expected={r['target_id'] for r in payload['targets']};returned=[r['target_id'] for r in answer['targets']]
        if set(returned)!=expected or len(returned)!=len(expected):issues.append('target_id_or_completeness')
        for r in answer['targets']:
            if r['status']=='found' and not r['passages']:issues.append(r['target_id']+'/no_passage')
            for passage in r['passages']:
                if locator.locate(payload['report_chunk']['text'],passage['quote'])['state'] not in {'exact','normalized_whitespace_unicode'}:
                    issues.append(r['target_id']+'/quote_not_located')
    else:
        if answer['target_id']!=payload['target']['target_id']:issues.append('target_id')
        quoted=[r['quote'] for r in payload['located_report_passages']]
        if any(not any(locator.locate(text,q)['state'] in {'exact','normalized_whitespace_unicode'} for text in quoted) for q in answer['report_quotes']):
            issues.append('final_quote_not_located')
        if payload['target']['kind']=='review_reason':
            if answer['reason_coverage'] in {'complete','partial'} and not answer['report_quotes']:issues.append('coverage_without_quote')
        elif answer['relation_treatment'] not in {'not_located','uncertain'} and not answer['report_quotes']:
            issues.append('relation_treatment_without_quote')
        if not payload['screening_complete'] and (answer['reason_coverage']=='none' or answer['relation_treatment']=='not_located'):
            issues.append('absence_without_complete_report_screening')
    return issues


def execute(packet: dict,config: Config) -> dict:
    from ..fig3_revision.client import Client
    path=result_path(packet)
    if path.exists():return read(path)
    started=time.monotonic();task=packet['task']
    record={'id':packet['id'],'task':task,'status':'failed','answer':None,'model':'gpt-5.6-luna','effort':'xhigh'}
    try:
        client=Client(config,context={'paper_id':packet['paper_id'],'stage':'abcd_followup','step':packet['id']})
        answer=client.call(task,SCREEN_PROMPT if task=='abcd_screen' else FINAL_PROMPT,packet['payload'],
                           ScreenResult if task=='abcd_screen' else FinalJudgment)
        problems=integrity(packet,answer)
        record.update(answer=answer,status='completed' if not problems else 'pending_integrity',integrity_issues=problems)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        record.update(error_type=type(exc).__name__,error=str(exc))
    finally:
        record['seconds']=time.monotonic()-started;write_json(path,record)
    return record


def run_batch(packets: list[dict],config: Config,label: str) -> None:
    pending=[p for p in packets if not result_path(p).exists()]
    counts=Counter(read(result_path(p))['status'] for p in packets if result_path(p).exists())
    active: dict[Future,str]={};position=0;limit=16;successes=0;last_print=0.0
    with ThreadPoolExecutor(max_workers=64) as executor:
        while position<len(pending) or active:
            slots,memory,_=admission(config,limit,len(active))
            for _ in range(min(slots,len(pending)-position)):
                p=pending[position];active[executor.submit(execute,p,config)]=p['id'];position+=1
            if active:
                done,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                for f in done:
                    active.pop(f);record=f.result();counts[record['status']]+=1
                    if record['status']!='failed':successes+=1
                    if successes>=8 and limit<64:limit=min(64,limit+16);successes=0
            else:time.sleep(2)
            now=time.monotonic()
            if now-last_print>=30 or (position==len(pending) and not active):
                print(json.dumps({'phase':label,'finished':sum(counts.values()),'total':len(packets),
                    'status':dict(counts),'active':len(active),'concurrency_target':limit,'available_memory_gib':round(memory,1)}),flush=True);last_print=now


def final_packets(targets: list[dict],screen: list[dict]) -> tuple[list[dict],list[dict]]:
    screen_index={p['id']:p for p in screen};packets=[];deferred=[];locator=Locator()
    for target in targets:
        passages=[];states=[];issues=[]
        for ident in target['screen_request_ids']:
            record=read(result_path(screen_index[ident]));answer=record['answer']
            if answer is None or 'target_id_or_completeness' in record.get('integrity_issues',[]):
                states.append('uncertain');issues.append(ident);continue
            output=next(r for r in answer['targets'] if r['target_id']==target['blind_id'])
            if any(i.startswith(target['blind_id']+'/') for i in record.get('integrity_issues',[])):
                states.append('uncertain');issues.append(ident);continue
            states.append(output['status'])
            text=screen_index[ident]['payload']['report_chunk']['text']
            for passage in output['passages']:
                located=locator.locate(text,passage['quote'])
                if located['state'] in {'exact','normalized_whitespace_unicode'}:
                    passages.append({'quote':located['source_quote']})
        # Preserve shorter excerpts and longer context; remove exact duplicates only.
        unique={r['quote']:r for r in passages}
        payload={'target':{'target_id':target['blind_id'],'kind':target['kind'],'criterion':target['criterion']},
            'located_report_passages':list(unique.values()),'screening_complete':not issues,
            'screened_chunk_count':len(target['screen_request_ids']),
            'uncertain_chunk_count':sum(s=='uncertain' for s in states)}
        packet={'id':'final/'+target['id'],'task':'abcd_final','paper_id':target['paper_id'],
                'target_id':target['id'],'payload':payload,'model':'gpt-5.6-luna','effort':'xhigh'}
        if not fit(payload,FINAL_PROMPT,FinalJudgment):
            deferred.append({'id':target['id'],'reason':'final_material_exceeds_limits','status':'pending','model_call_made':False})
        else:packets.append(packet)
    return packets,deferred


def main() -> None:
    screen=rows(PREP/'screen_packets.jsonl');targets=rows(PREP/'targets.jsonl')
    if len(screen)!=311 or len(targets)!=230:raise ValueError('Approved scope differs')
    if any(p['model']!='gpt-5.6-luna' or p['effort']!='xhigh' for p in screen):raise ValueError('Approved model differs')
    write_json(RUN/'authorization.json',{'source':'Human approved all remaining supplementary calls to complete ABCD in this chat.',
        'screen_requests':311,'final_request_ceiling':230,'initial_request_ceiling':541,
        'maximum_calls_including_format_repairs':1082,'model':'gpt-5.6-luna','effort':'xhigh',
        'no_failed_retries':True,'excluded':['retrieval','source_recovery','report_regeneration','old_results_overwrite']})
    config=load_config().model_copy(update={'output':RUN,'model':'gpt-5.6-luna',
        'efforts':{'abcd_screen':'xhigh','abcd_final':'xhigh'},'workers':64,'cli_limit':64,
        'cli_initial':16,'timeout_seconds':900,'synthesis_timeout_seconds':900,
        'repair_attempts':1,'network_retries':0,'download_fulltext':False})
    run_batch(screen,config,'source_location')
    final,deferred=final_packets(targets,screen)
    write_rows(RUN/'final_packets.jsonl',final);write_rows(RUN/'deferred_final_targets.jsonl',deferred)
    if len(final)>230:raise ValueError('Final call ceiling exceeded')
    print(json.dumps({'final_requests_ready':len(final),'deferred_for_material_limit':len(deferred)}),flush=True)
    run_batch(final,config,'final_judgment')
    write_json(RUN/'completion.json',{'screen_requests':311,'final_requests':len(final),'deferred_final_targets':len(deferred),
        'request_statuses':dict(Counter(read(result_path(p))['status'] for p in screen+final))})


if __name__=='__main__':main()
