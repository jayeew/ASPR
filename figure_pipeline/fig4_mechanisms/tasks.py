"""Small source-bound scientific tasks; reuse complete responses, repair actual omissions."""
from __future__ import annotations

from collections import Counter
from typing import Any, Literal

import networkx as nx
from pydantic import create_model

from figure_pipeline.fig3_revision.materials import encoded

from .models import IdentityLink, IdentityLinks


async def ask(evaluation: Any, role: str, payload: dict, schema: Any, step: str,
              instruction: str) -> dict:
    # Callers pack original task objects explicitly. Never recursively summarize judgments.
    if len(encoded(payload)) > 90000:
        raise ValueError(f'{step}: packed scientific task exceeds 90000 characters; split originals')
    return await evaluation.engine.ask(evaluation.ident, 'fig4_'+role, evaluation.method, step,
        role, instruction, payload, schema, True)


def complete_ids(rows: list[dict], field: str, expected: set[str]) -> bool:
    keys = [r[field] for r in rows]
    return set(keys) == expected and len(keys) == len(expected)


async def exact_items(evaluation: Any, role: str, payload: dict, schema: Any, step: str,
                      instruction: str, field: str, expected: set[str]) -> dict:
    value = await ask(evaluation, role, payload, schema, step, instruction)
    if complete_ids(value['items'], field, expected):
        return value
    # A format-valid response is not a completed scientific task. Repair only its bad IDs.
    counts = Counter(r[field] for r in value['items'])
    good = [r for r in value['items'] if r[field] in expected and counts[r[field]] == 1]
    missing = expected-{r[field] for r in good}
    repair = await ask(evaluation, role, {**payload, 'requested_ids': sorted(missing)}, schema,
        step+'_missing', instruction+' Return ONLY requested_ids, exactly once; use the unchanged original evidence.')
    fixed = good+[r for r in repair['items'] if r[field] in missing]
    if not complete_ids(fixed, field, expected):
        raise ValueError(f'{step}: scientific response still has missing/duplicate IDs')
    return {'items': fixed}


async def identity_groups(evaluation: Any, candidates: list[dict], step: str,
                          role: str = 'clusters') -> list[dict]:
    if not candidates:
        return []
    originals = {r['candidate_id']: r for r in candidates}
    ids = set(originals)
    # Reuse nonconflicting identity memberships from earlier incomplete merges.
    groups = []
    for old_role in (role, 'fusion'):
        for old_step in (step+'_membership_repair', step):
            path = evaluation.config.output/'annotations/checkpoints'/('fig4_'+old_role)/(evaluation.method or 'papers')/evaluation.ident/(old_step+'.json')
            if path.exists():
                from .io import read
                groups = read(path).get('groups', [])
                break
        if groups:
            break
    count = Counter(k for g in groups for k in g['member_ids'] if k in ids)
    links = []
    for group in groups:
        keys = [k for k in group['member_ids'] if k in ids and count[k] == 1]
        if keys:
            links.extend({'candidate_id': k, 'representative_id': keys[0]} for k in keys)
    pending = ids-{r['candidate_id'] for r in links}
    if pending:
        choices = Literal[tuple(sorted(ids))]
        row = create_model('RequestedIdentityLink', __base__=IdentityLink,
            candidate_id=(choices, ...), representative_id=(choices, ...))
        schema = create_model('RequestedIdentityLinks', __base__=IdentityLinks, items=(list[row], ...))
        result = await exact_items(evaluation, role, {'candidate_clusters': candidates,
            'requested_ids': sorted(pending), 'existing_identity_links': links}, schema, step+'_links',
            'For EACH requested candidate_id choose one supplied representative_id for the SAME scientific '
            'proposition, object, relation and scope. Merely sharing a topic is not identity. Preserve distinct '
            'increments/errors and qualifications. Read ALL candidates for global synonym alignment. '
            'Return ONLY requested IDs exactly once, two IDs per row; no membership lists or rewritten summaries. '
            'Use itself as representative ONLY when scientifically distinct, never as a missing-answer fallback. '
            'Prior unambiguous memberships are reused; resolve actual missing/ambiguous cases against original '
            'candidate descriptions. Identity is transitive, but do not link different scientific scopes.',
            'candidate_id', pending)
        links.extend(result['items'])
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_edges_from((r['candidate_id'], r['representative_id']) for r in links)
    return [{'member_ids': sorted(nodes), 'summary': originals[min(nodes)]['summary'],
             'scope': originals[min(nodes)].get('scope', ''), 'kind': originals[min(nodes)]['kind']}
            for nodes in sorted(nx.connected_components(graph), key=lambda nodes: min(nodes))]
