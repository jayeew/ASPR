"""Reclassify cached literal formats for reviewed journal-specific formats."""
from __future__ import annotations

import fcntl
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .fig8b_qualification_build import write_decisions
from .publisher_delta_apply import evidence_index, updated_decision
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(root: Path = ROOT, selected_source_ids: set[str] | None = None) -> dict:
    folder = root / 'supplement/journal_type_policies'
    q = root / 'fields/qualification'
    stats = Counter()
    transitions = Counter()
    with (root / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect(root)
        evidence_index(db, q)
        selected = []
        from .fig8b_qualification_decide import journal_research_format, science_china_chemistry_format, science_china_materials_format
        source_ids = sorted(selected_source_ids if selected_source_ids is not None else {
            'https://openalex.org/S106963461', 'https://openalex.org/S127827428',
            'https://openalex.org/S125754415', 'https://openalex.org/S143846845',
            'https://openalex.org/S123480357', 'https://openalex.org/S137905309',
            'https://openalex.org/S4210214063', 'https://openalex.org/S4210232714',
            'https://openalex.org/S151741590', 'https://openalex.org/S2764671299',
            'https://openalex.org/S2764926557', 'https://openalex.org/S139253143',
            'https://openalex.org/S103895331', 'https://openalex.org/S41063453',
            'https://openalex.org/S4210228084', 'https://openalex.org/S2298632',
            'https://openalex.org/S203256638', 'https://openalex.org/S197962172',
            'https://openalex.org/S165830345', 'https://openalex.org/S2248630',
            'https://openalex.org/S147836164', 'https://openalex.org/S2764782236'})
        placeholders = ','.join('?' for _ in source_ids)
        for wid, shard, payload in db.execute(f"""SELECT p.work_id,p.shard,e.payload
            FROM papers p JOIN current_evidence_index e USING(work_id)
            WHERE p.source_id IN ({placeholders})
            AND e.kind='primary'""", source_ids):
            if journal_research_format(json.loads(payload)) or science_china_chemistry_format(json.loads(payload)) or science_china_materials_format(json.loads(payload)):
                selected.append((wid, shard))
        grouped = defaultdict(set)
        for wid, shard in selected:
            grouped[shard].add(wid)
        annual = {(r['source_id'], int(r['year'])): r['nature_portfolio_year']
                  for r in read_csv(root / 'fields/journal_publisher_by_year.csv')}
        audit = []
        for shard, ids in grouped.items():
            path = q / 'decisions' / shard
            rows = pq.read_table(path).to_pylist()
            changed = False
            for old in rows:
                if old['work_id'] not in ids:
                    continue
                primary = []
                secondary = []
                for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (old['work_id'],)):
                    if kind == 'primary':
                        primary.append(json.loads(payload))
                    else:
                        secondary = json.loads(payload)
                updated = updated_decision(old, primary, secondary, annual)
                stats['assessed'] += 1
                audit.append({'work_id': old['work_id'], 'before': dict(old), 'after': updated,
                              'rule': 'supplement/journal_type_policies/research_formats.json'})
                if updated == old:
                    continue
                transitions[old['eligibility_status'] + ' -> ' + updated['eligibility_status']] += 1
                old.update(updated)
                changed = True
                db.execute('UPDATE papers SET status=? WHERE work_id=?', (old['eligibility_status'], old['work_id']))
                resolved = old['eligibility_status'].startswith(('included', 'excluded'))
                db.execute("UPDATE tasks SET status=?,reason=? WHERE kind='qualification' AND entity_key=?",
                           ('resolved' if resolved else 'pending', old['reason'], old['doi']))
                if old['eligibility_status'] == 'included_verified_research':
                    task(db, 'metadata', old['work_id'], 'pmc_or_publisher',
                         {'doi': old['doi'], 'source_id': old['source_id'],
                          'reason': 'Newly qualified journal-specific research format; assess current abstract/byline/affiliation gaps'}, old['work_id'])
                    stats['newly_qualified'] += 1
            if changed:
                write_decisions(path, rows, q / 'decision_changes.jsonl.gz')
            db.commit()
        db.close()
        shutil.copy2(Path(__file__).with_name('fig8b_qualification_decide.py'), q / 'decide.py')
        audit_path = folder / 'application_audit.json'
        previous = json.loads(audit_path.read_text()) if audit_path.exists() else []
        atomic_json(audit_path, previous + audit)
        result = {'status': 'applied_derived_refresh_required', 'network_requests': 0,
                  'transitions': dict(transitions), **stats}
        atomic_json(folder / 'application_status.json', result)
        print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    from .derived_refresh import refresh
    run()
    refresh()
