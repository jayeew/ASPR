"""Read accepted outputs and write figure-specific source tables; no model calls."""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/fig4_reference/experiment'
OUT = ROOT / 'outputs/fig4_reference'
ORDER = ['T', 'E', 'G', 'F', 'F_noJ', 'F_noM', 'F_noP']
NAMES = {'T': 'Paper only', 'E': 'GEAR only', 'G': 'Graph only', 'F': 'Full system',
         'F_noJ': 'Without joint graph', 'F_noM': 'Without structural values',
         'F_noP': 'Without citation paths'}
ASPECTS = ['historical_verification', 'knowledge_position', 'joint_contribution',
           'structural_resolution', 'citation_contact']
LABELS = ['Historical verification', 'Knowledge neighborhood', 'Joint knowledge structure',
          'Quantitative structure', 'Citation contact']
SILK = 's41467-026-68499-w'
THZ = 's41467-026-68290-x'
POLICY = 's43247-026-03223-6'


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def table(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT / 'data' / (name + '.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                          for k, v in row.items()} for row in rows])


def prepare() -> None:
    absolute = read(SOURCE / 'configuration_summary.json')
    effects = read(SOURCE / 'paired_effects_summary.json')
    pairs = read(SOURCE / 'paired_effects.json')
    metrics = read(SOURCE / 'paper_aspect_metrics.json')
    changes = read(SOURCE / 'paired_changes.json')
    parts = read(SOURCE / 'answer_parts.json')
    with (SOURCE / 'input_information.csv').open() as handle:
        facts = list(csv.DictReader(handle))
    for row in facts:
        for key in row:
            if key not in ('paper_id', 'citation_opportunity'):
                row[key] = int(row[key])
        row['citation_opportunity'] = row['citation_opportunity'] == 'True'
    valid = {(r['paper_id'], r['other'], r['aspect']) for r in pairs if r['complete_pair']}
    transition_rows = []
    for other in ['G', 'E']:
        for aspect in ASPECTS:
            records = [r for r in changes if r['other'] == other and r['aspect'] == aspect
                       and (r['paper_id'], other, aspect) in valid]
            counts = Counter(r['transition'] for r in records)
            transition_rows.append(dict(other=other, aspect=aspect, total=len(records),
                **{s: counts[s] for s in ['retained', 'gained', 'lost', 'neither_correct']}))
    selected = [r for r in parts if (r['paper_id'] == POLICY and r['aspect'] == ASPECTS[0]
                    and r['part_id'] == 'H1a' and r['condition'] in ['G', 'F'])
                or (r['paper_id'] == THZ and r['aspect'] == ASPECTS[2]
                    and r['condition'] in ['F', 'F_noJ'])
                or (r['paper_id'] == SILK and r['aspect'] in ASPECTS[3:]
                    and r['condition'] in ['F', 'F_noM', 'F_noP'])]
    opportunities = {p: read(SOURCE / 'factual_opportunities' / (p + '.json')) for p in [THZ, SILK]}
    snapshot = dict(absolute=absolute, effects=effects, pairs=pairs, metrics=metrics, facts=facts,
                    transitions=transition_rows, case_parts=selected, opportunities=opportunities,
                    cohort=read(SOURCE / 'cohort.json'), source=str(SOURCE))
    write(OUT / 'data/snapshot.json', snapshot)
    table('a_coverage_matrix', [r for r in absolute if r['metric'] == 'content_coverage'])
    table('b_paper_effects', pairs)
    table('b_effect_summary', effects)
    table('c_part_transitions', transition_rows)
    ordered = sorted(facts, key=lambda r: (r['historical_edges_only_in_joint'], r['paper_id']))
    table('d_graph_facts_and_order', [dict(order=i + 1, **r) for i, r in enumerate(ordered)])
    table('df_paper_coverage', [r for r in metrics if r['aspect'] in ASPECTS[2:]])
    profiles = opportunities[SILK]['structural_profiles']
    table('e_silk_structural_profiles', [dict(paper_id=SILK, claim_id=r['claim']['claim_id'],
                                              **r['metrics']) for r in profiles])
    table('case_report_evidence', selected)
    for name in ['configuration_summary.csv', 'configuration_summary_without_development.csv',
                 'paired_effects_summary_without_development.csv', 'graph_screening.csv',
                 'aspect_applicability.csv', 'conditions.json', 'protocol.json']:
        shutil.copy2(SOURCE / name, OUT / 'data' / name)
    write(OUT / 'data/source_files.json', dict(source_root=str(SOURCE),
          reference_style=str(OUT / 'design/original_reference_style.json'),
          notes='All scientific judgments reused. Transitions are paired part counts, not macro averages.'))
    print(f'Prepared figure sources: {OUT / "data"}', flush=True)
