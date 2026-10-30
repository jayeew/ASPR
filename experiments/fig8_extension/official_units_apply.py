"""Apply reviewed independent-unit aliases to exact primary signature slots."""
from __future__ import annotations

import fcntl
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import nullcontext

import pyarrow.parquet as pq

from .arrow_row_patch import patch_rows, selected_rows
from .fig8b_qualification_recompute import metrics
from . import metadata_task_delta
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

RULES = 'supplement/official_entities/infn_exact_rules.json'


def has_rule(evidence: dict, receipt: str) -> bool:
    """Recognize prior application even when another shared rule was applied later."""
    while isinstance(evidence, dict):
        if evidence.get('official_rule') == receipt:
            return True
        evidence = evidence.get('previous_identity_evidence')
    return False


def affected_papers(db: sqlite3.Connection, aliases: dict) -> dict[str, set[str]]:
    """Discover both missing identities and known identities lacking ownership.

    Entity links select candidate papers only; resolve still requires an exact
    whole signature and rejects conflicting institution identities.
    """
    affected = defaultdict(set)
    keys = [('institution', name) for name in aliases]
    keys += [('university_ownership', entity) for entity in
             sorted({rule['entity_id'] for rule in aliases.values()})]
    for kind, key in keys:
        for wid, shard in db.execute(
            "SELECT DISTINCT p.work_id,p.shard FROM tasks t "
            "JOIN task_papers l USING(task_id) JOIN papers p ON p.work_id=l.work_id "
            "WHERE t.kind=? AND t.entity_key=? AND p.status='included_verified_research'",
            (kind, key),
        ):
            affected[shard].add(wid)
    return dict(affected)


def resolve(author: dict, aliases: dict, rule_receipt: str = RULES) -> tuple[dict, int]:
    if author.get('nature_v1_slots') is not None:
        return author, 0
    applicable = False
    for slot in author.get('publisher_affiliation_slots') or []:
        candidate = aliases.get(' '.join(slot.get('text', '').casefold().split()))
        if candidate and not (set(slot.get('institution_ids', [])) - {candidate['entity_id']}):
            applicable = True
            break
    if not applicable:
        return author, 0
    new = json.loads(json.dumps(author))
    changed = 0
    for slot in new.get('publisher_affiliation_slots') or []:
        rule = aliases.get(' '.join(slot.get('text', '').casefold().split()))
        if not rule or set(slot.get('institution_ids', [])) - {rule['entity_id']}:
            continue
        parent_targets = rule.get('university_targets', {})
        unit_rule = rule['ownership_status'] == 'officially_confirmed_university_unit'
        if unit_rule:
            if len(parent_targets) != 1 or list(parent_targets.values()) != [1.0]:
                raise ValueError('University-unit rule requires one officially confirmed parent')
            prior_targets = slot.get('official_university_targets')
            if prior_targets and prior_targets != parent_targets:
                continue
        before = json.dumps(slot, sort_keys=True)
        fields = dict(institution_ids=[rule['entity_id']],
                      identity_status='official_exact_declared_name',
                      entity_type=rule['entity_type'],
                      university_ownership_status=rule['ownership_status'])
        if unit_rule:
            fields['official_university_targets'] = parent_targets
        if not slot.get('official_identity_evidence') or any(slot.get(k) != v for k, v in fields.items()):
            slot['official_identity_evidence'] = rule.get('_rule_receipt', rule_receipt)
        slot.update(fields)
        if before != json.dumps(slot, sort_keys=True):
            changed += 1
        if rule['entity_id'] not in new['institution_ids_or_verified_local_entities']:
            new['institution_ids_or_verified_local_entities'].append(rule['entity_id'])
            changed += 1
    return new, changed


def selected_papers(db: sqlite3.Connection, work_ids: set[str]) -> dict[str, set[str]]:
    """Read explicit incremental targets without depending on a refreshed gap index."""
    affected = defaultdict(set)
    for wid in sorted(work_ids):
        row = db.execute(
            "SELECT shard FROM papers WHERE work_id=? AND status='included_verified_research'",
            (wid,),
        ).fetchone()
        if row:
            affected[row[0]].add(wid)
    return dict(affected)


def run(rules_path: str = RULES, status_name: str = 'infn_apply_status.json',
        rule_paths: list[str] | None = None, selected_work_ids: set[str] | None = None,
        lock_held: bool = False) -> dict:
    rules = []
    aliases = {}
    for receipt in rule_paths if rule_paths is not None else [rules_path]:
        document = json.loads((ROOT / receipt).read_text())
        if document['status'] != 'reviewed':
            raise ValueError('Rules require evidence review')
        for original in document['rules']:
            rule = {**original, '_rule_receipt': receipt}
            rule.setdefault('university_targets', {})
            rules.append(rule)
            for name in rule['exact_names']:
                old = aliases.get(name)
                if old and any(old.get(k) != rule.get(k) for k in
                               ['entity_id', 'entity_type', 'ownership_status', 'university_targets']):
                    raise ValueError(f'Conflicting reviewed rules for exact signature: {name}')
                aliases.setdefault(name, rule)
    stats = Counter()
    context = nullcontext() if lock_held else (ROOT / 'supplement/checkpoint.lock').open('a')
    with context as lock:
        if not lock_held:
            fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        affected = (affected_papers(db, aliases) if selected_work_ids is None
                    else selected_papers(db, selected_work_ids))
        semantics_unchanged = True
        for rule in rules:
            receipt = rule['_rule_receipt']
            old = db.execute('SELECT evidence,entity_type,ownership_status,university_targets FROM identities WHERE entity_id=?', (rule['entity_id'],)).fetchone()
            semantics_unchanged = semantics_unchanged and (
                old[1] == rule['entity_type'] and old[2] == rule['ownership_status']
                and json.loads(old[3]) == rule.get('university_targets', {}))
            evidence = {'previous_identity_evidence': json.loads(old[0]),
                        'previous_identity_state': {'entity_type': old[1], 'ownership_status': old[2],
                                                    'university_targets': json.loads(old[3])},
                        'official_rule': receipt, 'source': rule['source'],
                        'official_url': rule['official_url']}
            # Avoid nesting identical evidence on repeat runs.
            if not (has_rule(evidence['previous_identity_evidence'], receipt)
                    and old[1] == rule['entity_type'] and old[2] == rule['ownership_status']
                    and json.loads(old[3]) == rule.get('university_targets', {})):
                db.execute('UPDATE identities SET evidence=?,entity_type=?,ownership_status=?,university_targets=? WHERE entity_id=?',
                           (json.dumps(evidence), rule['entity_type'], rule['ownership_status'], json.dumps(rule.get('university_targets',{})), rule['entity_id']))
        if not semantics_unchanged:
            metadata_task_delta.ensure_table(db)
            db.execute('DELETE FROM metadata_task_deltas')
        db.commit()
        for shard, ids in affected.items():
            path = ROOT / 'fields/papers' / shard
            decision_path = ROOT / 'fields/qualification/decisions' / shard
            before_signature = metadata_task_delta.signature(path, decision_path)
            table = pq.read_table(path)
            rows = selected_rows(table, ids)
            updates = {}
            for paper in rows:
                if paper['work_id'] not in ids:
                    continue
                before = paper['authors_json']
                revised = [resolve(a, aliases, rules_path) for a in json.loads(before)]
                count = sum(n for _, n in revised)
                if not count:
                    continue
                paper['authors_json'] = json.dumps([a for a, _ in revised], ensure_ascii=False)
                updates[paper['work_id']] = {'authors_json': paper['authors_json']}
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (paper['work_id'], 'authors_json', before, paper['authors_json'], json.dumps(rule_paths) if rule_paths is not None else rules_path,
                            'Exact whole declared signature plus applicable reviewed official entity rule; country unchanged'))
                values = metrics(paper)
                for metric, value in values.items():
                    field = 'after_' + metric
                    if field in paper and paper[field] != value:
                        db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                                   (paper['work_id'], field, json.dumps(paper[field]), json.dumps(value), json.dumps(rule_paths) if rule_paths is not None else rules_path, 'Recompute after official unit identity assignment'))
                        paper[field] = value
                        updates[paper['work_id']][field] = value
                db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?',
                           (not values['every_author_institution'], not values['every_author_country'], paper['work_id']))
                stats['papers_changed'] += 1
                stats['slot_and_author_identity_updates'] += count
            if updates:
                tmp = path.with_suffix('.parquet.tmp')
                pq.write_table(patch_rows(table, updates), tmp, compression='zstd')
                tmp.replace(path)
                metadata_task_delta.record(db, shard, before_signature,
                    metadata_task_delta.signature(path, decision_path), set(updates), semantics_unchanged)
            db.commit()
        db.close()
    result = {'status': 'applied_derived_refresh_required', **stats}
    atomic_json(ROOT / 'supplement/official_entities' / status_name, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
