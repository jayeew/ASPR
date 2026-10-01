"""Source-bound fusion tasks packed once by proposition, with original report coverage."""
from __future__ import annotations

import asyncio
from typing import Any

from figure_pipeline.fig3_revision.materials import batches, terms

from .config import RISK_TYPES
from .evaluation import Evaluation
from .io import artifact, read, write
from .models import BranchPresences, Issues, Risks, Transitions
from .tasks import ask, exact_items, identity_groups


def compact_unit(unit: dict, key: str) -> dict:
    return {'unit_key': key, 'quote': unit['quote'], 'claim_ids': unit['claim_ids'],
            'original_locator': unit.get('original_locator', '')}


def compact_support(label: dict | None) -> dict | None:
    if label is None:
        return None
    return {k: label[k] for k in ('support', 'scope_correct', 'errors')}


def report_units(evaluation: Evaluation, condition: str) -> list[dict[str, Any]]:
    labels = {r['unit_id']: r for r in evaluation.get('support', condition)['units']}
    return [{'unit_key': condition+'/'+u['unit_id'], 'unit': u, 'support': labels.get(u['unit_id'])}
            for u in evaluation.get('extract', condition)['units']]


async def canonical_issues(evaluation: Evaluation, branch: dict[str, Any]) -> list[dict]:
    path = artifact(evaluation.config.output, 'canonical_issues', evaluation.ident)
    if path.exists():
        return read(path)['issues']
    units = {u['unit_id']: u for u in branch['units']}
    errors = [{'unit_key': r['unit_id'], 'quote': units[r['unit_id']]['quote'],
               'errors': r['errors'], 'scope': units[r['unit_id']].get('original_locator', '')}
              for r in branch['support'] if r['errors'] or r['support'] == 'contradicted']
    async def local(group: list[dict], index: int) -> dict:
        instruction = ('Group errors by the SAME scientific proposition, object and scope. Each supplied '
            'unit_key occurs exactly once. Different errors remain distinct. Unverifiable alone is not an error.')
        value = await evaluation.ask('fusion', {'errors': group}, Issues, f'issues_{index:04d}', instruction)
        keys = [k for c in value['issues'] for k in c['unit_keys']]
        expected = {u['unit_key'] for u in group}
        if set(keys) != expected or len(keys) != len(expected):
            value = await ask(evaluation, 'fusion', {'errors': group}, Issues,
                              f'issues_complete_{index:04d}', instruction)
            keys = [k for c in value['issues'] for k in c['unit_keys']]
            if set(keys) != expected or len(keys) != len(expected):
                raise ValueError('Local issue partition remains incomplete')
        return value
    values = await evaluation.map('canonical_errors', batches(errors, 8, 6000), local)
    originals = {f'C{i:04d}': c for i, c in enumerate((c for v in values for c in v['issues']), 1)}
    candidates = [{'candidate_id': k, 'summary': c['proposition'], 'scope': c['scope'], 'kind': 'error'}
                  for k, c in originals.items()]
    groups = await identity_groups(evaluation, candidates, 'issue_identities', 'clusters')
    issues = []
    for i, group in enumerate(groups, 1):
        keys = [u for k in group['member_ids'] for u in originals[k]['unit_keys']]
        branches = sorted({units[k]['origin_branch'] for k in keys})
        issues.append({'issue_id': f'{evaluation.ident}/ISSUE{i:04d}', 'proposition': group['summary'],
            'scope': group['scope'], 'unit_keys': keys, 'origin_branches': branches,
            'source_group': 'Shared' if len(branches) > 1 else ('GEAR-only' if branches == ['gear'] else 'Graph-only'),
            'fusion_input_keys': sorted({units[k]['finding_key'] for k in keys})})
    write(path, {'issues': issues})
    return issues


def counter_candidates(issue: dict, units: dict, labels: dict) -> list[dict]:
    claim_ids = {c for k in issue['unit_keys'] for c in units[k]['claim_ids']}
    query = terms(issue['proposition']+' '+issue['scope'])
    allowed = [u for u in units.values() if u['origin_branch'] not in issue['origin_branches']
               and labels[u['unit_id']]['support'] == 'supported'
               and labels[u['unit_id']]['scope_correct'] is True and not labels[u['unit_id']]['errors']]
    ranked = sorted(allowed, key=lambda u: (not bool(claim_ids & set(u['claim_ids'])),
                                           -len(query & terms(u['quote'])), u['unit_id']))[:12]
    return [{**compact_unit(u, u['unit_id']), 'support': compact_support(labels[u['unit_id']])} for u in ranked]


async def correction_dependencies(evaluation: Evaluation, rows: list[dict], lookup: dict,
                                  units: dict, labels: dict, reports: dict, index: int) -> dict:
    details, pending = {}, {'E': [], 'G': []}
    for row in rows:
        issue = lookup[row['issue_id']]
        if row['full_status'] != 'corrected' or len(issue['origin_branches']) != 1:
            continue
        old = evaluation.config.output/'annotations/checkpoints/fig4_fusion/F'/evaluation.ident/(
            'correction_dependency_'+issue['issue_id'].split('/')[-1]+'.json')
        if old.exists():
            details[issue['issue_id']] = read(old)['items'][0]
        else:
            pending['E' if issue['origin_branches'] == ['gear'] else 'G'].append(row)
    for comparator, candidates in pending.items():
        for part, group in enumerate(batches(candidates, 2, 8000)):
            objects, evidence = [], {}
            for row in group:
                issue = lookup[row['issue_id']]
                counter = counter_candidates(issue, units, labels)
                objects.append({'issue': issue, 'confirmed_F_correction': row,
                    'original_branch_units': [compact_unit(units[k], k) for k in issue['unit_keys']],
                    'counterevidence_candidates': counter})
                material = evaluation.material({'issue': issue, 'candidate_assertions': counter}, 6000)
                evidence.update({b['block_id']: b for b in material['evidence_blocks']})
            result = await exact_items(evaluation, 'fusion', {'objects': objects,
                'comparator_condition': comparator, 'complete_comparator_report': reports[comparator],
                'evidence_blocks': list(evidence.values())}, Transitions,
                f'correction_dependencies_{index:04d}_{comparator}_{part:02d}',
                'Assess EACH issue separately: does its confirmed F correction depend on the other branch? '
                'Preserve confirmed F fate/quote. Check the COMPLETE comparator report; silence is not '
                'correction. GEAR-only comparator E removes Graph; Graph-only comparator G removes GEAR. '
                'Dependency requires no correction in the comparator and original-source-supported '
                'counterevidence in that issue actual other-branch candidates. Candidates only establish '
                'positive counterevidence, never exhaustive absence. Missing evidence is unresolved. '
                'Return every supplied issue_id exactly once, exact source IDs/quotes, concise reasons.',
                'issue_id', {r['issue_id'] for r in group})
            details.update({r['issue_id']: r for r in result['items']})
    return details


async def transitions(evaluation: Evaluation, branch: dict, issues: list[dict]) -> dict:
    path = artifact(evaluation.config.output, 'fusion_transitions', evaluation.ident)
    if path.exists():
        return read(path)
    units = {u['unit_id']: u for u in branch['units']}
    labels = {s['unit_id']: s for s in branch['support']}
    full = report_units(evaluation, 'F')
    full_by_key = {u['unit_key']: u for u in full}
    # Whole original reports establish global propagation/absence; labels stay in external indexes.
    reports = {c: evaluation.report(c)['body'] for c in ('F', 'E', 'G')}
    full_index = [{**compact_unit(u['unit'], u['unit_key'])} for u in full]
    write(artifact(evaluation.config.output, 'fusion_context', evaluation.ident),
          {'reports': reports, 'full_units': full_index, 'branch_units_artifact': str(
              artifact(evaluation.config.output, 'branch_units', evaluation.ident))})
    async def one(group: list[dict], index: int) -> dict:
        inputs = [{'issue_id': issue['issue_id'], 'units': [{**compact_unit(units[k], k),
            'support': compact_support(labels[k])} for k in issue['unit_keys']]} for issue in group]
        evidence = {}
        for issue in group:
            material = evaluation.material({'issue': issue, 'original_assertions': [
                compact_unit(units[k], k) for k in issue['unit_keys']]}, 6000)
            evidence.update({b['block_id']: b for b in material['evidence_blocks']})
        payload = {'errors': group, 'units': inputs, 'report': reports['F'], 'full_units': full_index,
                   'evidence_blocks': list(evidence.values())}
        instruction = ('Return one item for EVERY supplied issue_id. Compare this specific scientific error '
            'with the COMPLETE ORIGINAL F report and original evidence. Silence is not correction. '
            'Corrected requires explicit source-supported correction/narrowing; not_propagated requires global '
            'absence in these complete reports; propagated requires actual F wording carrying the same error. '
            'Preserve correction_present/error_propagated separately; propagation dominates when both occur. '
            'This task assesses F fate ONLY. Comparator/other-branch dependence is assessed separately: '
            'set without_other_branch_status=unresolved (not_applicable for Shared), '
            'other_branch_counterevidence=[], counterevidence_unit_keys=[], branch_dependent_correction=false. '
            'Use supplied Full keys, verbatim full_quote, original source IDs/locations. Keep reasons concise.')
        result = await exact_items(evaluation, 'fusion', payload, Transitions,
            f'transitions_packed_{index:04d}', instruction, 'issue_id', {i['issue_id'] for i in group})
        lookup = {i['issue_id']: i for i in group}
        for row in result['items']:
            issue = lookup[row['issue_id']]
            row['full_unit_keys'] = [k for k in row['full_unit_keys'] if k in full_by_key]
            if row['error_propagated']:
                row['full_status'] = 'propagated'
            if row['full_status'] in {'corrected', 'propagated'} and (not row['full_quote']
                    or row['full_quote'] not in reports['F'] or not row['full_unit_keys']):
                row.update(full_status='unresolved', correction_present=False, error_propagated=False)
        dependencies = await correction_dependencies(evaluation, result['items'], lookup,
                                                     units, labels, reports, index)
        for row in result['items']:
            issue = lookup[row['issue_id']]
            if row['issue_id'] in dependencies:
                detail = dependencies[row['issue_id']]
                row.update({k: detail[k] for k in ('without_other_branch_status',
                    'other_branch_counterevidence', 'counterevidence_unit_keys', 'branch_dependent_correction')})
            supplied = {u['unit_key'] for u in counter_candidates(issue, units, labels)}
            row['branch_dependent_correction'] = bool(row['branch_dependent_correction']
                and row['full_status'] == 'corrected' and len(issue['origin_branches']) == 1
                and row['without_other_branch_status'] in {'not_propagated', 'propagated'}
                and supplied.intersection(row['counterevidence_unit_keys']) and row['other_branch_counterevidence'])
            row.update(**{k: v for k, v in issue.items() if k != 'issue_id'}, paper_id=evaluation.ident,
                canonical_issue_id=issue['issue_id'], exposed_to_F=True,
                audit_status='assessed' if row['full_status'] != 'unresolved' else 'unresolved',
                source_ids=sorted({e['source_id'] for e in row['evidence']}),
                original_locations=[e['location'] for e in row['evidence']],
                counterevidence_search='selected_positive_candidates; not exhaustive absence evidence')
        return result
    values = await evaluation.engine.map(evaluation.ident, 'fig4_fusion', 'F', 'transitions_packed',
                                         batches(issues, 4, 5000), one)
    result = {'items': [r for v in values for r in v['items']], 'assessment_complete': True}
    write(path, result)
    return result


async def branch_presence(evaluation: Evaluation, candidates: list[dict], branch: dict) -> dict:
    path = artifact(evaluation.config.output, 'full_error_branch_presence', evaluation.ident)
    if path.exists():
        return read(path)['items']
    passages = [{'finding_key': f['finding_key'], 'original_text': f['finding']['text']}
                for f in branch['findings']]
    chunks = batches(passages, 1000, 28000)
    groups = batches(candidates, 6, 5000)
    jobs = [{'group': g, 'passages': c} for g in groups for c in chunks]
    async def one(job: dict, index: int) -> dict:
        rows = job['group']
        result = await exact_items(evaluation, 'fusion', {'units': [
            {**compact_unit(r['unit'], r['unit_key']), 'support': compact_support(r['support'])} for r in rows],
            'original_branch_passages': job['passages']}, BranchPresences,
            f'branch_presence_{index:04d}', 'For EACH full_unit_key determine whether the SAME scientific '
            'error/proposition with the SAME scope already occurs in these ORIGINAL actual branch findings. '
            'This is a partial branch chunk: absence applies only to these passages. Mere topical similarity '
            'or a properly bounded branch statement is not the same error. Return matched_finding_keys '
            'only from supplied passages. Retain incomplete correspondence as complete=false.',
            'full_unit_key', {r['unit_key'] for r in rows})
        allowed = {p['finding_key'] for p in job['passages']}
        for row in result['items']:
            row['matched_finding_keys'] = [k for k in row['matched_finding_keys'] if k in allowed]
            if row['already_in_branch'] and not row['matched_finding_keys']:
                row['complete'] = False
        return result
    values = await evaluation.engine.map(evaluation.ident, 'fig4_fusion', 'F', 'branch_presence', jobs, one)
    aligned = {}
    for candidate in candidates:
        key = candidate['unit_key']
        rows = [r for v in values for r in v['items'] if r['full_unit_key'] == key]
        aligned[key] = {'already_in_branch': any(r['already_in_branch'] for r in rows),
            'complete': len(rows) == len(chunks) and all(r['complete'] for r in rows),
            'matched_finding_keys': sorted({k for r in rows for k in r['matched_finding_keys']}),
            'original_branch_chunks': len(chunks)}
    write(path, {'items': aligned})
    return aligned


async def risks(evaluation: Evaluation, branch: dict) -> dict:
    path = artifact(evaluation.config.output, 'fusion_risks', evaluation.ident)
    if path.exists():
        return read(path)
    full = report_units(evaluation, 'F')
    labels = {r['unit_id']: r for r in branch['support']}
    supported = [u for u in branch['units'] if labels[u['unit_id']]['support'] == 'supported'
                 and labels[u['unit_id']]['scope_correct'] is True and not labels[u['unit_id']]['errors']]
    candidates = [r for r in full if r['support'] is None or r['support']['errors']
                  or r['support']['scope_correct'] is not True
                  or r['support']['support'] in {'contradicted', 'not_verifiable', 'partly_supported'}]
    presence = await branch_presence(evaluation, candidates, branch)
    new = [r for r in candidates if presence[r['unit_key']]['complete']
           and not presence[r['unit_key']]['already_in_branch']]
    jobs = [{'family': 'new', 'objects': g} for g in batches(new, 6, 5000)]
    jobs += [{'family': 'changed', 'objects': g} for g in batches(supported, 6, 5000)]
    full_body = evaluation.report('F')['body']
    branch_body = evaluation.report('branch_input')['body']
    async def one(job: dict, index: int) -> dict:
        active = RISK_TYPES[:4] if job['family'] == 'new' else RISK_TYPES[4:]
        objects = job['objects']
        packed = ([{**compact_unit(r['unit'], r['unit_key']), 'support': compact_support(r['support']),
                    'branch_presence': presence[r['unit_key']]} for r in objects] if job['family'] == 'new'
                  else [{**compact_unit(u, u['unit_id']), 'support': compact_support(labels[u['unit_id']])}
                        for u in objects])
        sources = [r['unit'] for r in objects] if job['family'] == 'new' else objects
        result = await ask(evaluation, 'fusion', {'units': packed, 'risk_types': active,
            'report': full_body, **evaluation.object_material(sources, 3000)}, Risks,
            f'risks_packed_{index:04d}', 'Assess EACH supplied object using ORIGINAL Full wording and original '
            'scientific evidence. Return assessments for every supplied risk_type. For new errors the full '
            'actual branch was scanned separately; source-check the claimed error, not just new wording. '
            'For supported-content changes compare the supplied ORIGINAL supported branch assertion to the '
            'COMPLETE Full report: unsupported_downgrade, wrong_modification, context_dropped, omission_only. '
            'Omission alone is not an erroneous assertion; justified uncertainty is not a downgrade. '
            'Events must be attributable only to supplied objects and their supplied branch/full keys. '
            'Use verbatim quotes and source locations. Uncertain support/scope/alignment is unresolved, '
            'not absent. Do not repair original wording to make it correct.')
        if {r['risk_type'] for r in result['assessments']} != set(active):
            result = await ask(evaluation, 'fusion', {'units': packed, 'risk_types': active,
                'report': full_body, **evaluation.object_material(sources, 3000)}, Risks,
                f'risks_packed_{index:04d}_complete', 'Complete EVERY requested risk_type using supplied '
                'original evidence and Full report; preserve unresolved scientific judgments.')
        if {r['risk_type'] for r in result['assessments']} != set(active):
            raise ValueError('Risk response remains incomplete')
        branch_ids = {u['unit_id'] for u in branch['units']}
        full_ids = {r['unit_key'] for r in full}
        for event in result['events']:
            located = (event['risk_type'] in active and set(event['branch_unit_keys']) <= branch_ids
                and set(event['full_unit_keys']) <= full_ids and all(q and q in branch_body for q in event['branch_quotes'])
                and (not event['full_quote'] or event['full_quote'] in full_body))
            if job['family'] == 'new':
                located = located and bool(event['full_unit_keys']) and all(
                    k in presence and presence[k]['complete'] and not presence[k]['already_in_branch']
                    for k in event['full_unit_keys'])
            else:
                located = located and bool(event['branch_unit_keys'])
            if event['risk_type'] != 'omission_only':
                located = located and bool(event['full_quote'])
            if not located:
                event['confirmed'] = False
                for a in result['assessments']:
                    if a['risk_type'] == event['risk_type']:
                        a['assessment_complete'] = False
                        a['unresolved_unit_count'] += 1
        return {'family': job['family'], **result}
    values = await evaluation.engine.map(evaluation.ident, 'fig4_fusion', 'F', 'risks_packed', jobs, one)
    events = []
    for value in values:
        for event in value['events']:
            event.update(event_id=f'{evaluation.ident}/RISK{len(events)+1:05d}', paper_id=evaluation.ident)
            events.append(event)
    unknown = sum(r['support'] is None or r['support']['scope_correct'] is None
                  or r['support']['support'] in {'not_verifiable', 'partly_supported'} for r in full)
    incomplete_presence = sum(not r['complete'] for r in presence.values())
    assessments = []
    for kind in RISK_TYPES:
        family = 'new' if kind in RISK_TYPES[:4] else 'changed'
        rows = [a for v in values if v['family'] == family for a in v['assessments'] if a['risk_type'] == kind]
        unresolved = sum(a['unresolved_unit_count'] for a in rows)
        if family == 'new':
            unresolved = max(unresolved, unknown, incomplete_presence)
        eligible = len(full) if family == 'new' else len(supported)
        assessments.append({'risk_type': kind, 'eligible_unit_count': eligible,
            'assessment_complete': all(a['assessment_complete'] for a in rows) and unresolved == 0,
            'unresolved_unit_count': unresolved, 'applicability': eligible > 0})
    result = {'events': events, 'assessments': assessments}
    write(path, result)
    return result


async def evaluate_fusion(evaluation: Evaluation) -> None:
    branch = await evaluation.branch_inputs()
    evaluation.stage, evaluation.method, evaluation.pool_condition = 'fusion', 'F', 'F'
    issues = await canonical_issues(evaluation, branch)
    results = await asyncio.gather(transitions(evaluation, branch, issues), risks(evaluation, branch),
                                   return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise RuntimeError('Fusion lanes incomplete: '+'; '.join(str(e) for e in errors))
