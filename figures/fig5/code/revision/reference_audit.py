from __future__ import annotations

import copy
from typing import Any

from figure_pipeline.fig3_revision.models import Record
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_robustness.models import ConditionalPart

from .materials import original_blocks, packet
from .models import Config
from .runtime import Runner


class Correction(Record):
    view: str
    part: ConditionalPart


class Audit(Record):
    corrections: list[Correction]
    rationale: str


def inconsistencies(references: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    indexed = {v: {(p['question_id'], p['part_id']): p for p in r['parts']} for v, r in references.items()}
    issues = []
    for lower, upper in [('E25', 'E50'), ('E50', 'E75'), ('E75', 'F'), ('E50', 'F')]:
        if lower not in indexed or upper not in indexed:
            continue
        for key, part in indexed[lower].items():
            if part['answerability'] == 'answerable' and indexed[upper][key]['answerability'] == 'unanswerable':
                issues.append({'lower': lower, 'upper': upper, 'question_id': key[0], 'part_id': key[1]})
    return issues


async def audit_references(config: Config, runner: Runner, paper: str) -> bool:
    from . import prompts
    from .pipeline import fixed, public, validate_reference, visible, visible_aliases
    root = config.output / 'validated_reference'
    references = {v: read(root / v / f'{paper}.json') for v in ('E25', 'E50', 'E75', 'F') if (root / v / f'{paper}.json').exists()}
    audit_path = config.output / 'reference_audit_adoption' / f'{paper}.json'
    audit_key = 'nested_originals'
    if audit_path.exists():
        previous = read(audit_path)
        if not previous['remaining_inconsistencies'] or previous.get('all_views_checked'):
            return True
        # One closure pass for early pair-only audits; never repeat to force a desired label.
        audit_key = 'nested_originals_closure'
    issues = inconsistencies(references)
    write(config.output / 'reference_audit_checks' / f'{paper}.json', {'views': sorted(references), 'issues': issues})
    if not issues:
        return True
    target_ids = {(issue['question_id'], issue['part_id']) for issue in issues}
    keys = {(v, q, p) for v in references for q, p in target_ids}
    base = packet(config, paper, 'F')
    material = {'full_material_catalog': visible(base), 'independent_work_aliases': visible_aliases(config, paper, base),
                'public_tasks': public(config, paper), 'fixed_reference': fixed(config, paper),
                'visible_blocks_by_view': {v: [b['block_id'] for b in original_blocks(packet(config, paper, v))] for v in references},
                'issues': issues, 'previous_references': {v: [p for p in r['parts'] if (v, p['question_id'], p['part_id']) in keys] for v, r in references.items()}}
    instruction = '''Audit ONLY the listed nested-evidence reference inconsistencies before candidate evaluation.
No candidate reports or scores are available. Return one correction for EVERY (view, question_id, part_id)
in previous_references, including unchanged judgments, and no others. Use the existing public task scope.
The full_material_catalog.original_evidence array contains actual historical originals INSIDE that catalog.
For a given view, ONLY blocks in visible_blocks_by_view[view] are available; manuscript and native graph
are identical in these E views. Never use upper-view-only evidence to justify a lower-view judgment.
Investigate whether an earlier judgment overlooked a visible source, misread JSON nesting, demanded private
exemplar extras, or accepted inadequate support. Do not enforce a monotonic curve or a desired answer.
An actual conflict or irreducible task-scope ambiguity can remain unresolved with its specific reason.
For each correction preserve the same evidence standard, exact task IDs, visible source quotes and limits.
The prior labels are fallible model judgments, not gold labels. Retain a negative judgment if justified.
''' + prompts.REFERENCE.replace('The supplied visible_packet is the ONLY evidence available under this condition.',
                               'Use the view-specific visibility mapping above as the ONLY evidence available under that condition.')
    write(config.output / 'reference_audit_inputs' / audit_key / f'{paper}.json', material)
    result = await runner.call(paper, 'reference_audit', audit_key, instruction, material, Audit, additional=True)
    observed = [(c['view'], c['part']['question_id'], c['part']['part_id']) for c in result['corrections']]
    if len(observed) != len(keys) or set(observed) != keys:
        raise ValueError('Reference audit changed requested correction identities')
    corrected = copy.deepcopy(references)
    for correction in result['corrections']:
        view, part = correction['view'], correction['part']
        corrected[view]['parts'] = [part if (p['question_id'], p['part_id']) == (part['question_id'], part['part_id']) else p for p in corrected[view]['parts']]
    for view, value in corrected.items():
        validate_reference(value, fixed(config, paper), packet(config, paper, view))
    changed = []
    for view, value in corrected.items():
        if value == references[view]:
            continue
        write(config.output / 'reference_pre_audit' / audit_key / view / f'{paper}.json', references[view])
        write(root / view / f'{paper}.json', value)
        changed.append(view)
    if audit_path.exists():
        write(config.output / 'reference_audit_adoption_history' / audit_key / f'{paper}.json', read(audit_path))
    write(audit_path, {'issues': issues, 'changed_views': changed, 'remaining_inconsistencies': inconsistencies(corrected), 'all_views_checked': True,
                       'rationale': result['rationale'], 'candidate_reports_consulted': False})
    from .reference_reuse import sync_order_references
    sync_order_references(config)
    return True
