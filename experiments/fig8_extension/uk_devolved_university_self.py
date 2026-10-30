"""Reuse devolved-government university lists for exact university-self signatures."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .england_university_self import normalized, record_attempts
from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

SOURCES = {
    'scotland': ('https://www.gov.scot/policies/universities/',
        'University of Aberdeen|Abertay University|University of Dundee|University of Edinburgh|Edinburgh Napier University|University of Glasgow|Glasgow Caledonian University|Heriot-Watt University|University of the Highlands and Islands|Queen Margaret University|Robert Gordon University|University of St Andrews|University of Stirling|University of Strathclyde|University of the West of Scotland'),
    'wales': ('https://www.gov.uk/check-university-award-degree/recognised-bodies-wales',
        'Aberystwyth University|Bangor University|Cardiff Metropolitan University|Cardiff University|University of South Wales|Swansea University|University of Wales|University of Wales Trinity Saint David'),
}


def run(sources: dict[str, tuple[str, str]] | None = None, folder_name: str = 'uk_devolved_self',
        country_by_area: dict[str, str] | None = None, exclusions: str | None = None,
        evidence_access: str = 'Web tool rendered text',
        evidence_locator: str = 'Recognised bodies list',
        source_receipts: dict[str, str] | None = None) -> None:
    sources = SOURCES if sources is None else sources
    country_by_area = country_by_area or {}
    folder = ROOT / 'supplement/official_entities' / folder_name
    folder.mkdir(exist_ok=True)
    receipt = str((folder / 'rules.json').relative_to(ROOT))
    rules = json.loads((folder / 'rules.json').read_text())['rules'] if (folder / 'rules.json').exists() else []
    known = {r['entity_id'] for r in rules}; held = []
    db = connect()
    entities = db.execute('SELECT entity_id,name,entity_type,countries,ownership_status FROM identities').fetchall()
    for area, (url, names) in sources.items():
        country = country_by_area.get(area, 'GB')
        proof = folder / (area + '_review.json')
        atomic_json(proof, {'url': url, 'reviewed_at': datetime.now(timezone.utc).isoformat(),
            'reviewed_names': names.split('|'), 'locator': evidence_locator,
            'access': evidence_access, 'raw_http_retained': bool(source_receipts and area in source_receipts),
            **({'receipt': source_receipts[area]} if source_receipts and area in source_receipts else {}),
            'scope': 'Official university-self identity corroboration for exact primary 2023–2025 signed names; not historical subsidiary ownership evidence.',
            'excluded': exclusions or 'Colleges, Open University regional designation, Wrexham rename, and all hospitals/centers/campuses.'})
        for name in names.split('|'):
            same_name = [e for e in entities if normalized(e[1]) == normalized(name)]
            found = [e for e in same_name if e[2] == 'education' and json.loads(e[3]) == [country]]
            if len(same_name) != 1:
                held.append({'name': name, 'reason': 'globally_nonunique_or_missing_registry_name'}); continue
            if len(found) != 1:
                held.append({'name': name, 'reason': 'no_unique_existing_country_education_entity'}); continue
            entity, label, _, _, status = found[0]
            if entity in known:
                continue
            if status != 'unverified':
                held.append({'name': name, 'reason': 'reviewed_status_preserved', 'status': status}); continue
            rules.append({'entity_id': entity, 'name': label,
                'exact_names': sorted({normalized(name), label.casefold(), 'the '+normalized(name)}),
                'entity_type': 'education', 'ownership_status': 'officially_confirmed_university_self',
                'university_targets': {entity: 1.0}, 'source': str(proof.relative_to(ROOT)),
                'official_url': url, 'locator': evidence_locator,
                'scope': 'Exact whole university self name only; no subsidiaries, location or country inference.'})
    atomic_json(folder / 'rules.json', {'status': 'reviewed', 'rules': rules})
    aliases = {n:r for r in rules for n in r['exact_names']}
    ids = set().union(*affected_papers(db, aliases).values()) if aliases else set()
    low = db.execute('SELECT coalesce(max(id),0) FROM changes').fetchone()[0]; db.close()
    atomic_json(folder / 'scope.json', {'selected_work_ids': sorted(ids), 'held': held, 'rules': len(rules)})
    result = apply_rules(rule_paths=[receipt], selected_work_ids=ids, status_name=folder_name+'_status.json')
    checked = verify_changes(low, ids, [receipt]); assert checked == result.get('papers_changed', 0)
    with (folder.parent / 'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = folder.parent / 'current_rule_reuse_scope.json'; state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']:
            state['next_rule_paths'].append(receipt)
        atomic_json(path, state)
    record_attempts(rules, receipt, folder_name+'_official_university_self')
    atomic_json(folder / 'validation.json', {'result': result, 'changed_papers_reproduced': checked,
        'idempotent_slot_updates': True, 'country_and_nature_preserved': True, 'final_task_closure': False})
    if rules:
        refresh()


if __name__ == '__main__':
    run()
