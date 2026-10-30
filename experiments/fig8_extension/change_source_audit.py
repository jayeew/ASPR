"""Incrementally audit change-ledger source paths without rewriting paper data."""
from __future__ import annotations

import fcntl
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def source_paths(receipt: str, root: Path) -> dict:
    """Resolve explicit receipt paths; remote URLs alone remain unverified."""
    if not receipt:
        return {'status': 'missing_receipt', 'unresolved': []}
    try:
        values = json.loads(receipt)
    except json.JSONDecodeError:
        values = receipt
    if isinstance(values, str):
        values = [values]
    accepted_keys = ({'identifier_receipt', 'publisher_receipts'},
                     {'identifier_receipt', 'publisher_receipts', 'review'})
    if isinstance(values, dict) and set(values) in accepted_keys:
        review = [values['review']] if 'review' in values else []
        if (isinstance(values['identifier_receipt'], str)
                and isinstance(values['publisher_receipts'], list)
                and all(isinstance(value, str) for value in review)):
            values = [values['identifier_receipt'], *values['publisher_receipts'], *review]
    if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
        return {'status': 'unsupported_receipt_structure', 'unresolved': [receipt]}
    unresolved = []
    resolved = []
    for value in values:
        if value.startswith(('https://', 'http://')):
            index = root / 'fields/official_capture_index.json'
            captures = json.loads(index.read_text()) if index.is_file() else []
            matches = [r for r in captures if r.get('url') == value and r.get('http_status') == 200
                       and r.get('raw_file') and (root / 'fields' / r['raw_file']).is_file()]
            if matches:
                resolved.extend('fields/' + r['raw_file'] for r in matches)
            else:
                unresolved.append(value)
            continue
        if not value:
            unresolved.append(value)
            continue
        path = Path(value.split('#', 1)[0])
        choices = [path] if path.is_absolute() else [root / path, root / 'fields/qualification' / path]
        existing = next((p for p in choices if p.is_file()), None)
        if existing is None:
            unresolved.append(value)
        else:
            resolved.append(str(existing))
    return {'status': 'source_paths_resolved' if not unresolved else 'unresolved_source_paths',
            'unresolved': unresolved, 'resolved_files': resolved}


def run(root: Path = ROOT) -> dict:
    folder = root / 'supplement/change_source_audit'
    folder.mkdir(exist_ok=True)
    with (folder / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = sqlite3.connect(folder / 'audit.sqlite')
        audit.executescript('''
            CREATE TABLE IF NOT EXISTS receipts(receipt TEXT PRIMARY KEY, result TEXT);
            CREATE TABLE IF NOT EXISTS issues(change_id INTEGER PRIMARY KEY, work_id TEXT,
                field TEXT, receipt TEXT, issues TEXT);
            CREATE INDEX IF NOT EXISTS issues_receipt ON issues(receipt);
            CREATE TABLE IF NOT EXISTS progress(id INTEGER PRIMARY KEY CHECK(id=1),
                cursor INTEGER, counts TEXT);
        ''')
        previous = audit.execute('SELECT cursor,counts FROM progress WHERE id=1').fetchone()
        cursor, counts = (previous[0], Counter(json.loads(previous[1]))) if previous else (0, Counter())
        db = sqlite3.connect(f'file:{root / "supplement/tasks.sqlite"}?mode=ro', uri=True)
        maximum = db.execute('SELECT COALESCE(max(id),0) FROM changes').fetchone()[0]
        # Revisit unresolved references only; never rescan already resolved ledger rows.
        for receipt, encoded in audit.execute('SELECT receipt,result FROM receipts').fetchall():
            old_status = json.loads(encoded)['status']
            if old_status not in {'unsupported_receipt_structure', 'unresolved_source_paths'}:
                continue
            result = source_paths(receipt, root)
            if result['status'] == old_status:
                continue
            audit.execute('UPDATE receipts SET result=? WHERE receipt=?', (json.dumps(result), receipt))
            for ident, problems in audit.execute('SELECT change_id,issues FROM issues WHERE receipt=?', (receipt,)).fetchall():
                remaining = [v for v in json.loads(problems) if v != old_status]
                if result['status'] != 'source_paths_resolved':
                    remaining.append(result['status'])
                if remaining:
                    audit.execute('UPDATE issues SET issues=? WHERE change_id=?', (json.dumps(remaining), ident))
                else:
                    audit.execute('DELETE FROM issues WHERE change_id=?', (ident,))
                    counts['changes_with_issues'] -= 1
                    counts['structurally_resolved_changes'] += 1
        audit.execute('INSERT OR REPLACE INTO progress VALUES(1,?,?)', (cursor, json.dumps(counts)))
        audit.commit()
        state = {'status': 'no_new_changes', 'through_change_id': cursor,
                 'fixed_end_change_id': maximum, **counts, 'network_requests': 0,
                 'goal_complete': False}
        cache = dict(audit.execute('SELECT receipt,result FROM receipts'))
        while cursor < maximum:
            rows = db.execute('''SELECT id,work_id,field,receipt,reason,
                before_value IS NOT NULL,after_value IS NOT NULL FROM changes
                WHERE id>? AND id<=? ORDER BY id LIMIT 2000''', (cursor, maximum)).fetchall()
            if not rows:
                break
            for ident, work, field, receipt, reason, before, after in rows:
                receipt = receipt or ''
                if receipt not in cache:
                    cache[receipt] = json.dumps(source_paths(receipt, root))
                    audit.execute('INSERT INTO receipts VALUES(?,?)', (receipt, cache[receipt]))
                result = json.loads(cache[receipt])
                problems = []
                if result['status'] != 'source_paths_resolved':
                    problems.append(result['status'])
                if not reason:
                    problems.append('missing_change_reason')
                if not before or not after:
                    problems.append('missing_before_or_after_value')
                counts['reviewed_changes'] += 1
                counts['changes_with_issues' if problems else 'structurally_resolved_changes'] += 1
                if problems:
                    audit.execute('INSERT OR REPLACE INTO issues VALUES(?,?,?,?,?)',
                                  (ident, work, field, receipt, json.dumps(problems)))
                cursor = ident
            audit.execute('INSERT OR REPLACE INTO progress VALUES(1,?,?)', (cursor, json.dumps(counts)))
            audit.commit()
            state = {'at': datetime.now(timezone.utc).isoformat(), 'through_change_id': cursor,
                     'fixed_end_change_id': maximum, **counts,
                     'status': 'fixed_ledger_prefix_checked' if cursor == maximum else 'running',
                     'limitations': ['Path existence does not establish source content validity or applicability.',
                                     'Fragment identifiers are retained but only the referenced file is checked.',
                                     'Qualification evidence and decision-change logs require separate checks.',
                                     'Later changes are outside this fixed ledger prefix.'],
                     'network_requests': 0, 'goal_complete': False}
            atomic_json(folder / 'status.json', state)
        db.close()
        audit.close()
        if maximum:
            atomic_json(folder / 'status.json', state)
        return state if maximum else {'status': 'empty_change_ledger'}


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
