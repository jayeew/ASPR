from __future__ import annotations

import random
from typing import Any

import numpy as np

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_robustness.materials import original_blocks, source_catalog
from figure_pipeline.fig5_robustness.models import Config
from figure_pipeline.fig5_robustness.pipeline import blind_reference, evaluation_material

SOURCE = Config().output
OUT = SOURCE.parent / 'fig5_four_panel'
CONDITIONS = ('F', 'E50', 'K5', 'LUNA')


def canonical_sources(data: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    aliases = source_catalog(data)
    parent = {v: v for v in aliases.values()}
    def root(key: str) -> str:
        while parent[key] != key:
            key = parent[key]
        return key
    merges = []
    for block in original_blocks(data):
        keys = sorted({aliases[p['source_id']] for p in block['provenance']})
        if len(keys) > 1:
            merges.append({'block_id': block['block_id'], 'source_keys': keys,
                           'provenance': block['provenance']})
            for key in keys[1:]:
                parent[root(key)] = root(keys[0])
    return {k: root(v) for k, v in aliases.items()}, merges


def material(paper: str) -> tuple[dict[str, Any], dict[str, str]]:
    data, mapping = evaluation_material(Config(), paper, 'historical_verification')
    fixed = read(SOURCE / 'baseline/reference/papers' / f'{paper}.json')
    refs = read(SOURCE / 'validated_reference' / f'{paper}.json')['views']
    packets = {v: f'V{i:02d}' for i, v in enumerate(
        random.Random(f'{Config().seed}:{paper}:views').sample(['F', 'E50', 'K5'], 3), 1)}
    data['reference_questions'] = fixed['questions']
    data['conditional_references'] = {packets[v['view_id']]: blind_reference(v['parts'], packets) for v in refs}
    for candidate in data['candidates']:
        candidate.pop('conditional_reference')
    aliases, _ = canonical_sources(read(SOURCE / 'inputs/F' / f'{paper}.json'))
    data['independent_work_aliases'] = aliases
    return data, mapping


def audit_condition(paper: str, aliases: dict[str, str]) -> dict[str, Any]:
    mask = read(SOURCE / 'source_masks' / f'{paper}.json')
    path = SOURCE / 'report_inputs/E50' / f'{paper}.json'
    if not path.exists():
        return {'paper_id': paper, 'eligible': False, 'missing_reason': 'E50 report/input unavailable'}
    actual = read(path)
    kept_ids = {p['source_id'] for b in actual['original_evidence'] for p in b['provenance']}
    old_to_new = {v: aliases[k] for k, v in mask['source_aliases'].items()}
    removed = {old_to_new[v] for v in mask['removed']}
    visible = {aliases[s] for s in kept_ids}
    leakage = sorted(removed & visible)
    prepared = read(SOURCE / 'inputs/E50' / f'{paper}.json')
    writer_matches = actual['original_evidence'] == original_blocks(prepared)
    branches_match = all(read(SOURCE / 'analysis_inputs' / ('E50_' + branch) / f'{paper}.json')['original_historical_passages']
                         == prepared[branch + '_evidence']['evidence_blocks'] for branch in ('gear', 'graph'))
    return {'paper_id': paper, 'old_count': mask['total'], 'canonical_count': len(set(aliases.values())),
            'retained_count': len(visible), 'retained_fraction': len(visible) / len(set(aliases.values())),
            'retained_sources': sorted(visible), 'removed_sources': sorted(set(aliases.values()) - visible),
            'leaked_removed_sources': leakage, 'writer_matches_prepared': writer_matches,
            'branches_match_prepared': branches_match, 'eligible': not leakage and writer_matches and branches_match}


def audit_reference_labels(groups: list[dict[str, Any]]) -> None:
    rows = []
    for g in groups:
        if not g['reference_label_corrected']:
            continue
        paper = g['paper_id']
        refs = {v['view_id']: {p['part_id']: p for p in v['parts']}
                for v in read(SOURCE / 'validated_reference' / f'{paper}.json')['views']}
        blocks = original_blocks(read(SOURCE / 'inputs/E50' / f'{paper}.json'))
        visible = {p['source_id'] for b in blocks for p in b['provenance']} | {b['block_id'] for b in blocks}
        history = [e for e in refs['F'][g['part_id']]['evidence'] if e['kind'] == 'history']
        absent = [e['source_id'] for e in history if e['source_id'] not in visible]
        assert absent and len(absent) == len(history), 'Reinspect the explicit reference correction'
        rows.append({**g, 'full_history_evidence_ids': [e['source_id'] for e in history],
                     'not_visible_in_E50': absent, 'E50_reference_evidence': refs['E50'][g['part_id']]['evidence']})
    write(OUT / 'reference_label_audit.json', rows)
    write_csv(OUT / 'reference_label_audit.csv', rows)


def prepare() -> None:
    cohort = read(SOURCE / 'cohort.json')
    backgrounds, merges, audits = [], [], []
    for row in read(SOURCE / 'paper_coverage.json'):
        paper = row['paper_id']
        aliases, duplicate = canonical_sources(read(SOURCE / 'inputs/F' / f'{paper}.json'))
        full = read(SOURCE / 'inputs/F' / f'{paper}.json')
        fulltext = {aliases[p['source_id']] for b in original_blocks(full) for p in b['provenance']
                    if p['source_type'] == 'fulltext'}
        backgrounds.append({**row, 'original_readable_sources': row['readable_sources'],
                            'readable_sources': len(set(aliases.values())),
                            'fulltext_fraction': len(fulltext) / len(set(aliases.values()))})
        merges.extend({'paper_id': paper, **d} for d in duplicate)
        if paper in cohort['selected']:
            audits.append(audit_condition(paper, aliases))
    median = float(np.median([r['readable_sources'] for r in backgrounds]))
    for r in backgrounds:
        r['history_group'] = 'lower' if r['readable_sources'] <= median else 'higher'
        r['history_median_cut'] = median
    groups = []
    reviewed_absence = {
        's41467-026-68412-5': ['H2a', 'H2b'], 's41467-026-68493-2': ['H1a', 'H2a', 'H2b'],
        's41467-026-68592-0': ['H2a'], 's41467-026-68809-2': ['H1a'],
        's41467-026-68885-4': ['H1a', 'H2a'], 's41467-026-68887-2': ['H2a'],
        's41467-026-69098-5': ['H2a', 'H2b'], 's41467-026-69411-2': ['H1a', 'H1b'],
        's42003-026-09591-1': ['H1a', 'H1b'], 's42004-026-01949-0': ['H2a'],
        's42004-026-01990-z': ['H1a', 'H1b'],
    }
    for paper in cohort['selected']:
        fixed = read(SOURCE / 'baseline/reference/papers' / f'{paper}.json')
        refs = {v['view_id']: {(p['question_id'], p['part_id']): p for p in v['parts']}
                for v in read(SOURCE / 'validated_reference' / f'{paper}.json')['views']}
        for q in fixed['questions']:
            if q['aspect'] != 'historical_verification' or not q['applicable']:
                continue
            for p in q['answer_parts']:
                key = q['question_id'], p['part_id']
                full, reduced = refs['F'][key], refs['E50'][key]
                dependent = any(e['kind'] == 'history' for e in full['evidence'])
                group = ('retained' if reduced['answerability'] == 'answerable' else
                         'lost' if reduced['answerability'] == 'unanswerable' else 'reference_unresolved')
                normalized = reduced['answerability']
                correction = p['part_id'] in reviewed_absence.get(paper, [])
                if correction:
                    assert normalized == 'unresolved'
                    normalized, group = 'unanswerable', 'lost'
                eligible = full['answerability'] == 'answerable' and dependent
                groups.append({'paper_id': paper, 'question_id': key[0], 'part_id': key[1],
                               'F_answerability': full['answerability'], 'E50_answerability': normalized,
                               'original_E50_answerability': reduced['answerability'],
                               'reference_label_corrected': correction,
                               'correction_basis': 'Candidate-free reference explicitly establishes missing required originals and insufficient scientific alternatives; normalize material insufficiency, not scientific truth' if correction else '',
                               'history_dependent': dependent, 'eligible': eligible, 'group': group,
                               'F_reason': full['reason'], 'E50_reason': reduced['reason']})
    audit_reference_labels(groups)
    for name, rows in [('paper_background', backgrounds), ('source_identity_merges', merges),
                       ('e50_input_audit', audits), ('frozen_history_groups', groups)]:
        write(OUT / (name + '.json'), rows)
        write_csv(OUT / (name + '.csv'), rows)
    write(OUT / 'protocol.json', {'source': str(SOURCE), 'selected': cohort['selected'],
          'repeated': cohort['repeated'], 'new_request_cap': 30, 'new_generation_calls': 0,
          'history_cut_rule': '<= median versus > median; before scores', 'median': median,
          'grouping_rule': 'Existing candidate-free conditional references, F-answerable H parts with history evidence; unresolved excluded explicitly',
          'prior_requests': len((SOURCE / 'call_ledger.jsonl').read_text().splitlines()),
          'evaluation_kind': 'model-assisted, not human gold'})
    for paper in cohort['selected']:
        packet, mapping = material(paper)
        if not (OUT / 'evaluation_inputs' / f'{paper}.json').exists():
            write(OUT / 'evaluation_inputs' / f'{paper}.json', packet)
        write(OUT / 'mapping' / f'{paper}.json', mapping)
    print({'papers': len(backgrounds), 'stress': len(audits), 'merges': len(merges),
           'invalid_E50': sum(not r['eligible'] for r in audits), 'median': median})


if __name__ == '__main__':
    prepare()
