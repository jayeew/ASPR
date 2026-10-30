"""Apply a combined fixed cohort of reviewed dated publisher permission statements."""
from __future__ import annotations

import json

from .cell_press_annual_cache import apply_proposals
from .jats_annual_fixed_scope import prepare
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder=ROOT/'supplement/remaining_named_annual'
    proposals,reviews=prepare(folder)
    saved=json.loads((folder/'proposals.json').read_text())
    if proposals!=saved:
        raise ValueError('Prepared annual evidence changed; review before applying')
    atomic_json(folder/'application_status.json',{'status':'waiting_for_writer','proposed_pairs':len(proposals)})
    result=apply_proposals(proposals,folder,[r for r in reviews if r['status']!='dated_publisher_identity_confirmed'],
        rule_status='dated_native_jats_publication_statement',
        reason='Individually matched native publication, agreeing electronic/copyright year and explicit named publication statement; society/imprint header handled only by reviewed literal journal rule',
        route='native_jats_copyright',apply_pairs={(r['source_id'],r['year']) for r in proposals})
    atomic_json(folder/'application_status.json',{'status':'applied','result':result,'network_requests':0,'final_task_closure':False})
    return result


if __name__=='__main__':
    print(json.dumps(run()),flush=True)
