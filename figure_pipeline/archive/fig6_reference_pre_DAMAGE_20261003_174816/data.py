"""Read one saved study case without modifying its scientific artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fig6_reference'
STUDY = ROOT / 'outputs/innovation_200_20260907'
PAPER = 's41467-026-68515-z'
CASE = STUDY / 'papers' / PAPER
SPAN_IDS = ['S-02525db57a20f14d9a59', 'S-2f82d1144da5a6742327',
            'S-da2a8e1da7a0ff515539', 'S-4e3961e44f7dfc52a84d',
            'S-2776c0f419f5017a3c26', 'S-7ffc42dd9722a8ba8321']
WORKS = [('W3003918836', 1), ('W3137126940', 2),
         ('W4413771983', 3), ('W4226023345', 5)]
HISTORY = ['s41467-025-61270-7::C01', 's41467-024-52091-1::C03',
           's41467-023-36372-9::C01', 's41467-025-61270-7::C02',
           's41467-023-40859-w::C01', 's41467-024-49084-5::C01',
           's41467-025-62314-8::C03', 's41467-024-48163-x::C03',
           's41467-023-36184-x::C01', 's41467-025-65502-8::C01']
LABELS = ['Sealed liquid cells', 'Short-pulse strategy', 'Interface explanation',
          'Imaging quality', 'L1-stalk dynamics', 'Slow-motion hypothesis']


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def trace(path: Path) -> dict[str, Any]:
    """Last saved state per evidence key; preserve raw append-only traces separately."""
    return {r['evidence_id']: r for line in path.read_text().splitlines()
            if (r := json.loads(line))}


def screen() -> list[dict[str, Any]]:
    rows = []
    for line in (STUDY / 'papers.jsonl').read_text().splitlines():
        paper = json.loads(line)
        base = STUDY / 'papers' / paper['paper_id']
        paths = [base / 'shared/claims.json', base / 'graph/joint/facts.json',
                 STUDY / 'reports/fusion' / (paper['paper_id'] + '.json')]
        complete = all(p.exists() for p in paths)
        claims = read(paths[0])['claims'] if paths[0].exists() else []
        facts = read(paths[1]) if paths[1].exists() else {}
        cards = len(list((base / 'gear').glob('[0-9][0-9]/gear_card.json')))
        rows.append({'paper_id': paper['paper_id'], 'title': paper['title'],
                     'complete_artifacts': complete and cards == len(claims),
                     'claims': len(claims), 'historical_nodes': facts.get('historical_neighbor_count'),
                     'historical_edges': len(facts.get('historical_edges', [])),
                     'shared_neighbors': len(facts.get('shared_historical_neighbors', {})),
                     'grounding_narrowings': sum(c['internal_support'] == 'partially_supported' for c in claims),
                     'selected': paper['paper_id'] == PAPER})
    return rows


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = screen()
    (OUT / 'data').mkdir(exist_ok=True)
    with (OUT / 'data/case_screening.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ir = read(CASE / 'shared/paper_ir.json')
    spans = {s['span_id']: s for s in ir['spans']}
    claims = read(CASE / 'shared/claims.json')['claims']
    facts = read(CASE / 'graph/joint/facts.json')
    gear, graph, neighbors = {}, {}, {}
    for i in range(1, 7):
        key = f'C{i:02}'
        gear[key] = trace(CASE / f'gear/{i:02}/evidence_trace.jsonl')
        entries = trace(CASE / f'graph/{i:02}/evidence_trace.jsonl')
        graph[key] = entries[f'GRAPH:{PAPER}::CLAIM::{i:02}']['payload']
        for neighbor in graph[key]['neighbors']:
            neighbors[neighbor['claim_id']] = neighbor
    history = {f'H{i:02}': neighbors[ident] for i, ident in enumerate(HISTORY, 1)}
    works = {f'E{i:02}': gear[f'C{claim:02}'][f'WORK:{PAPER}::CLAIM::{claim:02}:https://openalex.org/{work}']['payload']
             for i, (work, claim) in enumerate(WORKS, 1)}
    relations = []
    for ci in (1, 2, 3, 5):
        for ei, (work, _) in enumerate(WORKS, 1):
            key = f'RELATION:{PAPER}::CLAIM::{ci:02}:https://openalex.org/{work}'
            entry = gear[f'C{ci:02}'].get(key)
            relations.append({'claim': f'C{ci:02}', 'work': f'E{ei:02}', 'evidence_key': key if entry else None,
                              'assessment': entry['payload'] if entry else None})
    local_edges = {tuple(sorted(edge)) for g in graph.values() for edge in g['neighbor_edges']}
    extra = [e for e in facts['historical_edges'] if tuple(sorted(e)) not in local_edges]
    report = read(STUDY / f'reports/fusion/{PAPER}.json')
    data = {'paper': read(CASE / 'innovation_input.json'), 'claims': claims,
            'claim_labels': LABELS, 'manuscript': {f'M{i:02}': {**spans[ident], 'supporting_span_ids': claims[i-1]['support_span_ids'], 'text': '\n\n'.join(spans[sid]['text'] for sid in claims[i-1]['support_span_ids'])} for i, ident in enumerate(SPAN_IDS, 1)},
            'display_prior_quote': 'The sample remains liquid for the duration of the laser pulse',
            'works': works, 'relations': relations, 'graph_cards': graph,
            'historical_nodes': history, 'all_historical_nodes': neighbors,
            'joint': facts, 'joint_only_edges': extra, 'report': report,
            'selection': {'screened': len(rows), 'complete': sum(r['complete_artifacts'] for r in rows),
                          'basis': 'Artifact completeness; user-requested physics/biochemistry scope; accessible technique-to-biology narrative; six complementary claims; independent prior works with full-text passages; joint-only historical edges; explicit unresolved mechanism. No performance score used.'}}
    write(OUT / 'data/snapshot.json', data)
    export_sources(data)


def export_sources(data: dict[str, Any]) -> None:
    paths = [CASE / name for name in ['innovation_input.json', 'shared/claims.json',
             'shared/paper_ir.json', 'graph/joint/facts.json', 'graph/joint/analysis.json', 'graph/joint/report.md']]
    paths += list(CASE.glob('gear/[0-9][0-9]/*.json'))
    paths += list(CASE.glob('gear/[0-9][0-9]/evidence_trace.jsonl'))
    paths += list(CASE.glob('graph/[0-9][0-9]/*.json*'))
    paths += [STUDY / f'reports/fusion/{PAPER}.json', STUDY / f'reports/fusion/{PAPER}.md']
    manifest = []
    for path in sorted(set(paths)):
        dest = OUT / 'sources' / path.relative_to(STUDY)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        manifest.append({'original_path': str(path.relative_to(ROOT)), 'bundle_path': str(dest.relative_to(OUT)),
                         'sha256_at_export': hashlib.sha256(path.read_bytes()).hexdigest()})
    write(OUT / 'data/source_manifest.json', manifest)
    write(OUT / 'data/identity_map.json', {
        'claims': {f'C{i:02}': c['claim_id'] for i, c in enumerate(data['claims'], 1)},
        'manuscript': {k: {'primary_span_id': v['span_id'], 'supporting_span_ids': v['supporting_span_ids']} for k, v in data['manuscript'].items()},
        'historical_nodes': {k: v['claim_id'] for k, v in data['historical_nodes'].items()},
        'works': {k: {'doi': v['doi'], 'title': v['title']} for k, v in data['works'].items()}})
