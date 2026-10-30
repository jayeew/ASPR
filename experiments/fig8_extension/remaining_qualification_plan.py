"""Route every remaining qualification task using existing evidence, without fetching."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def next_action(status: str, matched: int, urls: set[str], archive: bool) -> str:
    if any(word in status for word in ('conflict', 'identity_mismatch', 'integrity')):
        return 'review_identity_type_or_integrity_conflict'
    if matched:
        return 'interpret_cached_primary_fields'
    if 'bulk_transport_reviewed_alternatives_pending' in urls:
        return 'assess_alternative_source_after_retained_bulk_transport_failure'
    if 'source_path_review_pending' in urls:
        return 'assess_unreviewed_publication_path'
    if archive:
        return 'review_existing_archive_outcome_and_alternatives'
    if any('hold' in value for value in urls):
        return 'assess_documented_source_hold_and_alternatives'
    return 'assess_bulk_source_coverage_and_alternatives'


def run() -> None:
    base = ROOT / 'supplement'
    source = sqlite3.connect(f'file:{base / "tasks.sqlite"}?mode=ro', uri=True)
    links = sqlite3.connect(f'file:{base / "cached_primary_url_leads/leads.sqlite"}?mode=ro', uri=True)
    folder = base / 'remaining_qualification_plan'
    folder.mkdir(exist_ok=True)
    out = sqlite3.connect(folder / 'queue.sqlite')
    out.execute('CREATE TABLE IF NOT EXISTS current_tasks(work_id TEXT PRIMARY KEY, qualification_status TEXT, route TEXT, next_action TEXT, matched_primary_records INTEGER, evidence_keys_json TEXT, url_states_json TEXT, archive_lead INTEGER, qualification_task_states_json TEXT, closure_status TEXT)')
    url_states: dict[str, set[str]] = defaultdict(set)
    for wid, state in links.execute('SELECT l.work_id,d.status FROM leads l JOIN url_dispatch d USING(url)'):
        url_states[wid].add(state)
    links.close()
    source.execute('BEGIN')
    evidence: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for wid, key, payload in source.execute("SELECT e.work_id,e.evidence_key,e.payload FROM current_evidence_index e JOIN papers p USING(work_id) WHERE p.status LIKE 'pending%' AND e.kind='primary'"):
        item = json.loads(payload)
        evidence[wid].append((key, item.get('identity') == 'matched'))
    task_states: dict[str, set[str]] = defaultdict(set)
    for wid, state in source.execute("SELECT l.work_id,t.status FROM tasks t JOIN task_papers l USING(task_id) WHERE t.kind='qualification' AND t.active=1"):
        task_states[wid].add(state)
    counts = Counter(); routes = Counter(); statuses = Counter()
    out.execute('BEGIN'); out.execute('DELETE FROM current_tasks')
    for wid, status, route, pmcids in source.execute("SELECT work_id,status,route,pmcids FROM papers WHERE status LIKE 'pending%'"):
        items = evidence.get(wid, [])
        matched = sum(value for _, value in items)
        archive = bool(json.loads(pmcids or '[]'))
        states = url_states.get(wid, set())
        action = next_action(status, matched, states, archive)
        out.execute('INSERT INTO current_tasks VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (wid, status, route, action, matched, json.dumps([key for key, _ in items]),
                     json.dumps(sorted(states)), int(archive), json.dumps(sorted(task_states.get(wid, set()))), 'not_finally_closed'))
        counts[action] += 1; routes[(route, action)] += 1; statuses[status] += 1
    expected = source.execute("SELECT count(*) FROM papers WHERE status LIKE 'pending%'").fetchone()[0]
    assert sum(counts.values()) == expected == out.execute('SELECT count(*) FROM current_tasks').fetchone()[0]
    source.close(); out.commit(); out.close()
    result = {'status': 'all_remaining_candidates_routed_not_source_closure',
        'updated_at': datetime.now(timezone.utc).isoformat(), 'pending_candidates': expected,
        'actions': dict(counts), 'qualification_statuses': dict(statuses),
        'route_actions': [{'route': route, 'action': action, 'papers': n} for (route, action), n in routes.most_common()],
        'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False,
        'evidence_locator': 'tasks.sqlite:current_evidence_index(work_id,evidence_key)',
        'url_evidence_locator': 'cached_primary_url_leads/leads.sqlite:leads JOIN url_dispatch',
        'limitation': 'A queued public URL is not proof of access. Archive identifiers are leads, not unattempted requests. Routing does not imply that source alternatives have been exhausted.'}
    atomic_json(folder / 'summary.json', result)
    print(json.dumps({'pending_candidates': expected, 'actions': dict(counts)}), flush=True)


if __name__ == '__main__':
    run()
