from __future__ import annotations

from typing import Any

from gear.codex_cli import _strict_response_schema

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.client import BASE
from figure_pipeline.fig3_revision.config import ROOT
from figure_pipeline.fig3_revision.materials import encoded, fits
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_rerun.config import CONDITIONS, NAMES, Report
from figure_pipeline.fig4_rerun.materials import payload as reused_payload

from .config import DIMENSIONS, Config, Evaluation
from .prompts import EVALUATOR, RUBRIC, WRITER


def payload(config: Config, paper: str, condition: str) -> dict[str, Any]:
    value = reused_payload(config, paper, condition)
    if condition == 'F_noP':
        # A removed path label is unknown, not a confirmed semantic-only relationship.
        for card in value['graph']['cards']:
            for neighbor in card['neighbors']:
                neighbor.pop('edge_type', None)
    return value


def opportunities(graph: dict[str, Any]) -> dict[str, Any]:
    cards = graph['cards']
    counts = [int(n.get(field, 0) or 0) for c in cards for n in c['neighbors']
              for field in ('direct_citation', 'two_hop_path_count', 'shared_reference_count')]
    return {
        'scientific_increment': True, 'knowledge_relations': True,
        'joint_explanation': bool(graph.get('joint') and len(cards) >= 2),
        'structure_explanation': any(c['neighbors'] and c['metrics'] for c in cards),
        'citation_explanation': any(v > 0 for v in counts),
        'citation_note': 'Positive contacts in the existing snapshot only; zero is not global absence.',
    }


def prepare(config: Config) -> None:
    ids = read(config.source / 'pilot.json')['paper_ids']
    roster = {p['paper_id']: p for p in read(config.source / 'papers.json')}
    inventory, sizes = [], []
    for paper in ids:
        original = config.source / 'inputs/papers' / f'{paper}.json'
        base = read(original)
        write(config.output / 'inputs/papers' / f'{paper}.json', base)
        core_path = config.source / 'inputs/evaluation' / f'{paper}.json'
        fixed = read(core_path)
        checklist_path = ROOT / 'outputs/fig3_reference/study/annotations/quality_checklist/papers' / f'{paper}.json'
        checklist = read(checklist_path)
        blocks = {b['block_id']: b for b in fixed['evidence_blocks']}
        for field in ('gear_evidence', 'graph_evidence'):
            for block in base[field]['evidence_blocks']:
                if block['block_id'] in blocks:
                    known = blocks[block['block_id']]['provenance']
                    for source in block['provenance']:
                        if source not in known:
                            known.append(source)
                else:
                    blocks[block['block_id']] = block
        material = {**fixed, 'evidence_blocks': list(blocks.values()), 'native_graph': base['graph'],
                    'fixed_opportunities': opportunities(base['graph']),
                    'fig3_criteria': [{'dimension': d['dimension'], 'criteria': d['criteria']}
                                      for d in checklist['dimensions']]}
        write(config.output / 'inputs/evaluation' / f'{paper}.json', material)
        for name, path in [('paper_branch_inputs', original), ('fixed_core_reference', core_path),
                           ('fig3_quality_criteria', checklist_path)]:
            inventory.append({'paper_id': paper, 'material': name, 'source_file': str(path), 'state': 'reused'})
        for condition in CONDITIONS:
            value = payload(config, paper, condition)
            size = len(encoded(value))
            schema = _strict_response_schema(Report.model_json_schema())
            if not fits(config, value, BASE + '\n' + WRITER + '\nINPUT:\n', schema):
                raise ValueError(f'Generation input too large: {paper} {condition}; no calls made')
            sizes.append({'paper_id': paper, 'condition': condition, 'configuration': NAMES[condition],
                          'generation_chars': size, 'generation_fields': list(value),
                          'fixed_evaluation_chars': len(encoded(material)),
                          'citation_opportunity': material['fixed_opportunities']['citation_explanation']})
        # Leave room for a substantial report; actual payload is checked again before every request.
        schema = _strict_response_schema(Evaluation.model_json_schema())
        if not fits(config, {'report': '字' * 30000, **material}, BASE + '\n' + EVALUATOR + '\nINPUT:\n', schema):
            raise ValueError(f'Evaluation input too large: {paper}; no calls made')
    write(config.output / 'papers.json', [roster[p] for p in ids])
    write(config.output / 'pilot.json', {'paper_ids': ids})
    write(config.output / 'protocol.json', {
        'paper_count': 5, 'conditions': NAMES, 'dimensions': DIMENSIONS, 'call_limit': 70,
        'reports': 35, 'evaluations': 35, 'model': config.model, 'generation_effort': 'high',
        'evaluation_effort': 'xhigh', 'report_length_target': None, 'writer': WRITER,
        'evaluator': EVALUATOR, 'rubric': RUBRIC, 'source_root': str(config.source),
        'single_direct_writer': True, 'single_evaluation': True, 'automatic_retries': False,
        'path_mask_note': 'Delete edge_type rather than substitute a semantic_only negative label.',
        'scope': 'report_input_effects_under_common_writer_not_end_to_end_branch_ranking',
        'remaining_95_authorized': False,
    })
    write_csv(config.output / 'reuse_inventory.csv', inventory)
    write_csv(config.output / 'input_inventory.csv', sizes)
    print(f'Prepared original {len(ids)} papers, 35 report tasks and 35 evaluations; no model calls.', flush=True)
