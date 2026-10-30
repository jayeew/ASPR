"""Check declared PMC metadata availability on the existing EDP journal sets."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter

from lxml import etree

from . import ojs_metadata_harvest as transport
from .edp_oai_harvest import FOLDER, NS, URL
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = FOLDER / 'rich_set_review'
    folder.mkdir(exist_ok=True)
    with (folder / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.FOLDER, transport.RAW = FOLDER, FOLDER / 'raw'
        plans = json.loads((FOLDER / 'plan.json').read_text())['sets']
        results_path = folder / 'results.json'
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
        summary = json.loads((folder / 'status.json').read_text()) if (folder / 'status.json').exists() else {}
        for plan in plans:
            spec = plan['set']
            if spec in results:
                continue
            params = {'verb': 'ListRecords', 'metadataPrefix': 'pmc', 'set': spec,
                      'from': '2023-01-01', 'until': '2026-10-23'}
            old = json.loads((FOLDER / 'aa_first.json').read_text())
            receipt = old if old.get('params') == params else transport.fetch(URL, params)
            result = {'set': spec, 'target_candidates': len(plan['targets']), 'receipt': receipt}
            if receipt.get('http_status') == 200:
                raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
                result['errors'] = [{'code': e.get('code'), 'text': e.text}
                                    for e in tree.findall('o:error', NS)]
                result['records'] = len(tree.findall('o:ListRecords/o:record', NS))
                result['next_token'] = tree.findtext('o:ListRecords/o:resumptionToken', namespaces=NS)
                result['status'] = ('rich_records_identity_type_date_review_required' if result['records']
                                    else 'no_rich_records_in_requested_set_window')
            else:
                result['status'] = 'source_failure_unattempted_retained'
            results[spec] = result
            atomic_json(results_path, results)
            summary = {'sets_reviewed': len(results), 'fixed_sets': len(plans),
                       'outcomes': dict(Counter(r['status'] for r in results.values())),
                       'qualification_changes': 0, 'final_task_closure': False}
            atomic_json(folder / 'status.json', summary)
            if receipt.get('http_status') != 200:
                break
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    run()
