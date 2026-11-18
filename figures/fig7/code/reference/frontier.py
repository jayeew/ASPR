"""2026 Fig7 workflow: explicit pilot first; full processing is a separate command."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .data import compact, read, write
from .frontier_data import OUT, extract_missing, graph_prepare, inventory, pilot_selection, prepare_packets


def estimate() -> dict[str, Any]:
    rows = inventory()
    trials = [read(p) for p in (OUT/'timings').glob('graph_[0-9]*.json')]
    graph = max(trials,key=lambda t:t['new_claims'])
    batches = graph['batches']
    warm = batches[1:]
    warm_count = sum(b['claims'] for b in warm)
    warm_seconds = sum(b['embedding_seconds']+b['facts_seconds'] for b in warm)
    rate = warm_seconds/warm_count
    missing = [r for r in rows if not r['claim_count']]
    known_remaining = sum(r['claim_count']-r['graph_fact_count'] for r in rows)
    estimated_remaining = known_remaining+8*len(missing)
    cold = max(0,batches[0]['embedding_seconds']+batches[0]['facts_seconds']-batches[0]['claims']*rate)
    graph_seconds = cold+estimated_remaining*rate+graph['joint_seconds']/graph['requested_papers']*len(rows)
    extracts = [read(p) for p in (OUT/'timings').glob('extract_*.json')]
    extract_seconds = statistics.mean(x['seconds'] for x in extracts)*len(missing) if extracts else None
    calls = [read(p) for p in (OUT/'pilot/logs/calls').glob('*/record.json')]
    forecast = [r for r in calls if r['task']=='forecast' and r['state']=='completed']
    merge = [r for r in calls if r['task']=='merge' and r['state']=='completed']
    result: dict[str, Any] = {
        'scope':'Remaining work after the pilot, sequential model calls; not an execution trigger',
        'remaining_missing_claim_papers':[r['paper_id'] for r in missing],
        'remaining_known_graph_claims':known_remaining,'remaining_graph_claims_upper_estimate':estimated_remaining,
        'graph_pilot':graph,'warm_graph_seconds_per_claim':rate,'graph_remaining_seconds_point':graph_seconds,
        'claim_extraction_pilot':extracts,'claim_extraction_remaining_seconds_point':extract_seconds,
        'forecast_pilot_records':calls,'full_forecast_calls':8,'full_merge_calls':1,'full_retry_reserve':1,
        'pilot_forecast_calls_used':len(calls),
        'pilot_claim_calls_used':sum(len(p.read_text().splitlines()) for p in (OUT/'logs/claim_usage').glob('*.jsonl')),
        'full_run_started':False,
        'uncertainties':['One extraction paper and one forecast packet are timing samples, not confidence intervals.',
            'Final merge processes up to 96 candidates, versus at most 12 in the pilot.',
            'Full packet sizes depend on uncomputed neighborhoods; oversized packets stop before a call.',
            'Queueing, GPU contention and failed attempts may increase elapsed time.']}
    if forecast and merge:
        f,m = forecast[-1]['seconds'],merge[-1]['seconds']
        prior_merges = [read(p) for p in (OUT.parent/'forecast/logs/calls').glob('*/record.json')
                        if read(p).get('task')=='merge' and read(p).get('state')=='completed']
        historical_merge = statistics.median(r['seconds'] for r in prior_merges) if prior_merges else m*4
        merge_lo, merge_hi = max(m,historical_merge), max(m*4,historical_merge*2)
        # Forecast packets are production-sized; merge is smaller and must not be naively scaled as 8 calls.
        fixed_lo = graph_seconds*.8+(extract_seconds or 0)*.7+120
        fixed_hi = graph_seconds*1.8+(extract_seconds or 0)*1.5+600
        low = fixed_lo+8*f*.75+merge_lo
        high = fixed_hi+8*f*1.6+merge_hi+max(f,merge_hi)
        result['remaining_minutes_planning_range']=[round(low/60),round(high/60)]
        result['forecast_seconds_per_packet']=f
        result['merge_seconds_planning_range']=[merge_lo,merge_hi]
        result['prior_full_merge_seconds']=historical_merge
        result['estimation_formula']='low=fixed_low+8*0.75*pilot_forecast+merge_low; high=fixed_high+8*1.6*pilot_forecast+merge_high+one_retry; merge range also uses prior full-size merges'
    write(OUT/'timing_estimate.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('graph_pilot','forecast_pilot_records','claim_extraction_pilot')},indent=2,ensure_ascii=False))
    return result


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['inventory','pilot','estimate','extract','graph','packets','predict','render','all'])
    parser.add_argument('--full',action='store_true',help='Explicit full-corpus execution; omit for pilot')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.stage=='estimate':
        estimate();return
    rows=inventory()
    selection=pilot_selection(rows)
    destination=OUT/('full' if args.full else 'pilot')
    if args.stage=='inventory':
        print(compact({'papers':len(rows),'claims':sum(r['claim_count'] for r in rows),
                       'graph_claims':sum(r['graph_fact_count'] for r in rows),'pilot':selection}));return
    if args.stage=='pilot' and args.full:
        parser.error('pilot does not accept --full')
    scope=rows if args.full else [r for r in rows if r['paper_id'] in selection['forecast_paper_ids']]
    if args.stage in ('extract','pilot','all'):
        extract_missing(scope)
        rows=inventory()
        scope=rows if args.full else [r for r in rows if r['paper_id'] in selection['forecast_paper_ids']]
    if args.stage in ('graph','pilot','all'):
        graph_prepare(scope,args.batch_size)
        rows=inventory()
        scope=rows if args.full else [r for r in rows if r['paper_id'] in selection['forecast_paper_ids']]
    if args.stage in ('packets','pilot','all'):
        prepare_packets(scope,destination,8 if args.full else 1)
    if args.stage in ('predict','pilot','all'):
        from .frontier_predict import predict
        predict(destination,10 if args.full else 2)
    if args.stage in ('render','pilot','all'):
        from .frontier_render import render
        render(destination)
    if args.stage=='pilot':
        estimate()


if __name__=='__main__':
    main()
