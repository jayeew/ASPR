"""Apply the official BDJ template rule after the fixed acquisition consumers finish."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .pensoft_bulk import FOLDER
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_bytes, atomic_json


def apply() -> dict:
    folder = FOLDER / 'taxonomy_rule'
    staged = folder / 'staged'
    staged.mkdir(exist_ok=True)
    sources = {}
    for relative in ['staged', 'type_repair/staged', 'native/staged', 'alternate_hosts/staged']:
        for path in (FOLDER / relative).glob('*.json'):
            record = json.loads(path.read_text())
            p = record['parsed']
            if not set(p['issns']) & {'1314-2836', '1314-2828'}:
                continue
            if 'Taxonomy & Inventories' not in p['article_types']:
                continue
            sources[record['doi']] = (path, record)
    jobs = []
    for doi, (path, record) in sources.items():
        target = staged / path.name
        atomic_json(target, {**record, 'type_policy_receipt': str((folder / 'rule.json').relative_to(ROOT)),
                            'identity_source_receipt': str(path.relative_to(ROOT))})
        jobs.append((doi, str(target.relative_to(ROOT))))
    result = apply_available(supplied_jobs=jobs) if jobs else {}
    db = connect()
    author_jobs = [(wid, shard, receipt) for doi, receipt in jobs
                   for wid, shard in db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
    db.close()
    authors = apply_authors(selected_jobs=author_jobs, workers=4, fill_abstract=True) if author_jobs else {}
    atomic_bytes(ROOT / 'fields/qualification/decide.py', Path(__file__).with_name('fig8b_qualification_decide.py').read_bytes())
    output = {'status':'reviewed_taxonomy_rule_applied_other_types_retained','source_dois':len(sources),'application':result,'authors':authors,'network_requests':0}
    atomic_json(folder / 'status.json', output)
    return output


def run() -> None:
    folder = FOLDER / 'taxonomy_rule'
    with (folder / 'dispatcher.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(folder / 'status.json', {'status':'waiting_for_fixed_native_consumers','network_requests':0})
        with (FOLDER / 'native/collector.lock').open('a') as native:
            fcntl.flock(native, fcntl.LOCK_EX)
            with (FOLDER / 'alternate_hosts/collector.lock').open('a') as alternate:
                fcntl.flock(alternate, fcntl.LOCK_EX)
                print(json.dumps(apply()), flush=True)


if __name__ == '__main__':
    run()
