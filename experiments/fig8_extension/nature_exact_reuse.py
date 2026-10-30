"""Reuse reviewed Nature signatures with intact official cached bodies."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from pathlib import Path

from .official_units_apply import run as apply_rules
from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json


APPROVED = {
    'reviewed-local:e5031928a984490a5457': 'Official laboratory name, research activity and Shanghai address.',
    'reviewed-local:111203a3ca74b60d8d73': 'Official museum description names Holbæk and its 2012 merger.',
    'reviewed-local:2e990d088570a56941f1': 'Corporate Cambridge laboratories explicitly opened in 2003.',
    'reviewed-local:273ccbd9fd8b3f34e8a5': 'DLR mission site names Institute of Planetary Research and Berlin address.',
    'reviewed-local:bf25f3dbdf29ef736036': 'Official museum name and Museumsgade 3, Hjørring address.',
    'https://openalex.org/I919571938': 'Official department page identifies McGovern Medical School, UTHealth Houston and full university legal name in footer.',
}
RULE_PATH = 'supplement/official_entities/nature_reused_exact_rules.json'


def prepare() -> dict:
    folder = ROOT / 'supplement/official_entities'
    candidates = json.loads((folder / 'nature_exact_reuse_candidates.json').read_text())
    rules = []
    held = []
    repair_path = folder / 'nature_reference_repair/reviewed_replacements.json'
    repairs = json.loads(repair_path.read_text()) if repair_path.exists() else {}
    for candidate in candidates['rules']:
        entity = candidate['entities'][0]
        entity_id = entity['review_entity_id']
        receipt_path = Path(candidate['evidence']['source_receipt'])
        repair = repairs.get(entity_id)
        if repair:
            receipt_path = ROOT.resolve() / repair['source_receipt']
        if entity_id == 'https://openalex.org/I919571938':
            receipt_path = ROOT.resolve() / 'nature_institution_review/primary_sources/5b09bba67f60896a157c9b2e2852c95a1c35453e84ee5025b89521765ceebc0d.json'
        receipt = json.loads(receipt_path.read_text())
        if entity_id not in APPROVED and not repair:
            held.append({'address': candidate['address'], 'entity_id': entity_id,
                         'source_receipt': str(receipt_path), 'response': receipt,
                         'status': 'unresolved_official_body_unavailable',
                         'reason': '403, transport failure or maintenance page cannot support official confirmation; no new mapping applied.',
                         'matched_tasks': candidate['matched_tasks']})
            continue
        raw = receipt_path.with_suffix('.raw.gz')
        digest = hashlib.sha256(gzip.decompress(raw.read_bytes())).hexdigest()
        readable = receipt.get('status') == 200 or (
            receipt.get('status') == 'web_read_success'
            and receipt.get('source_format') == 'serialized_web_tool_result'
            and receipt.get('retrieval_method') == 'web.open_and_find'
        )
        if not readable or digest != receipt.get('sha256'):
            raise ValueError(f'Official body no longer validates: {entity_id}')
        university = candidate['entity_type'] == 'university'
        rules.append({'entity_id': entity_id, 'name': entity['display_name'],
                      'entity_type': 'education' if university else candidate['entity_type'],
                      'exact_names': [' '.join(candidate['address'].casefold().split())],
                      'ownership_status': 'officially_confirmed_university_self' if university else 'officially_confirmed_nonuniversity',
                      'university_targets': {entity_id: 1.0} if university else {}, 'source': str(receipt_path.relative_to(ROOT.resolve())),
                      'raw_receipt': str(raw.relative_to(ROOT.resolve())), 'raw_sha256': digest,
                      'official_url': receipt['url'], 'locator': repair['locator'] if repair else APPROVED[entity_id],
                      'matched_tasks': candidate['matched_tasks'],
                      'scope': 'Exact whole publisher signature only, 2023–2025 current qualified cohort. Named institution or explicitly identified medical school within university; no hospital affiliation or country reassignment inferred.',
                      'prior_review': candidate['evidence']})
    document = {'status': 'reviewed', 'rules': rules, 'network_requests': 0,
                'unresolved': held, 'historical_raw_paths_resolved_to_current_location': True}
    atomic_json(ROOT / RULE_PATH, document)
    return document


def run() -> dict:
    document = prepare()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for rule in document['rules']:
            evidence = {'source': RULE_PATH, 'source_receipt': rule['source'],
                        'raw_receipt': rule['raw_receipt'], 'raw_sha256': rule['raw_sha256']}
            db.execute('INSERT OR IGNORE INTO identities VALUES(?,?,?,?,?,?,?)',
                       (rule['entity_id'], rule['name'], rule['entity_type'], '[]',
                        json.dumps(evidence), rule['ownership_status'], '{}'))
        db.commit()
        for unresolved in document['unresolved']:
            for task in unresolved['matched_tasks']:
                source = 'cached_nature_official_evidence_review'
                prior = db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                                   (task['task_id'], source, RULE_PATH)).fetchone()
                if not prior:
                    attempt(db, task['task_id'], source, 'pending', RULE_PATH,
                            {'reason': unresolved['reason'], 'source_receipt': unresolved['source_receipt'],
                             'network_requests': 0, 'final_task_closure': False})
        for rule in document['rules']:
            for task in rule['matched_tasks']:
                source = 'cached_nature_official_rule_available'
                prior = db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                                   (task['task_id'], source, RULE_PATH)).fetchone()
                if not prior:
                    attempt(db, task['task_id'], source, 'pending', RULE_PATH,
                            {'reason': 'Official identity rule available; exact publisher slot applicability still required.',
                             'network_requests': 0, 'final_task_closure': False})
        db.close()
    result = apply_rules(RULE_PATH, 'nature_reused_exact_status.json')
    result.update(reviewed_reusable_entities=len(document['rules']),
                  unresolved_official_references=len(document['unresolved']),
                  papers_changed=result.get('papers_changed', 0), network_requests=0)
    atomic_json(ROOT / 'supplement/official_entities/nature_reused_exact_status.json', result)
    review = json.loads((ROOT / 'nature_institution_review/affiliation_decisions.json').read_text())
    held = [{**item, 'nature_affected_dois': sorted({doi for row in review
              if row['address'] == item['address'] for doi in row.get('dois', [])}),
             'next_action': 'Review other cached official evidence or applicable public alternative; no repeat of failed URL.'}
            for item in document['unresolved']]
    atomic_json(ROOT / 'supplement/official_entities/nature_official_reference_reopen.json',
                {'status': 'unresolved_source_support_review_required' if held else 'identified_invalid_references_repaired', 'records': held,
                 'nature_dois': len({d for row in held for d in row['nature_affected_dois']}),
                 'scope': 'Only invalid references found in exact-reuse candidates; not a complete Nature audit.'})
    return result


if __name__ == '__main__':
    run()
