from __future__ import annotations

from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_revision.materials import equivalent, original_blocks

from .interventions import OUT


def source_profile(data: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    blocks = original_blocks(data)
    works = {aliases[p['source_id']] for b in blocks for p in b['provenance']}
    fulltext = {aliases[p['source_id']] for b in blocks for p in b['provenance'] if p['source_type'] == 'fulltext'}
    return {'works': sorted(works), 'work_count': len(works), 'fulltext_work_count': len(fulltext),
            'unique_original_characters': sum(len(b['text']) for b in blocks)}


def graph_profile(data: dict[str, Any]) -> dict[str, int]:
    neighbors = [n for c in data['graph']['cards'] for n in c['neighbors']]
    return {'graph_parent_papers': len({n['parent_paper_id'] for n in neighbors}),
            'graph_unique_neighbor_claims': len({n['claim_id'] for n in neighbors}),
            'graph_target_neighbor_links': len(neighbors)}


def main() -> None:
    rows, controls = [], []
    for path in (OUT / 'inputs/F').glob('*.json'):
        paper, full = path.stem, read(path)
        aliases = read(OUT / 'identities' / path.name)
        base = source_profile(full, aliases)
        graph_base = graph_profile(full)
        profile = {}
        for other in (OUT / 'inputs').glob(f'*/{path.name}'):
            condition, data = other.parent.name, read(other)
            current = source_profile(data, aliases)
            profile[condition] = current
            history_only = condition.startswith('E') or condition in ('CRITICAL', 'NONCRITICAL')
            graph_only = condition in ('G50', 'G25')
            graph = graph_profile(data)
            rows.append({'paper_id': paper, 'condition': condition, **current, **graph,
                         'retained_graph_parent_fraction': graph['graph_parent_papers'] / graph_base['graph_parent_papers'] if graph_base['graph_parent_papers'] else None,
                         'retained_source_fraction': current['work_count'] / base['work_count'],
                         'retained_character_fraction': current['unique_original_characters'] / base['unique_original_characters'],
                         'history_intervention_preserves_graph': equivalent(data['graph'], full['graph']) if history_only else None,
                         'graph_intervention_preserves_originals': original_blocks(data) == original_blocks(full) if graph_only else None})
        if all(c in profile for c in ('CRITICAL', 'NONCRITICAL')):
            critical, control = profile['CRITICAL'], profile['NONCRITICAL']
            lost = base['unique_original_characters'] - critical['unique_original_characters']
            other_lost = base['unique_original_characters'] - control['unique_original_characters']
            controls.append({'paper_id': paper, 'critical_deleted_works': base['work_count'] - critical['work_count'],
                             'control_deleted_works': base['work_count'] - control['work_count'],
                             'critical_deleted_characters': lost, 'control_deleted_characters': other_lost,
                             'control_to_critical_deleted_characters_ratio': other_lost / lost if lost else None,
                             'critical_deleted_fulltext_works': base['fulltext_work_count'] - critical['fulltext_work_count'],
                             'control_deleted_fulltext_works': base['fulltext_work_count'] - control['fulltext_work_count']})
    for name, data in [('material_audit', rows), ('deletion_matching_audit', controls)]:
        write(OUT / f'{name}.json', data)
        write_csv(OUT / f'{name}.csv', data)
    invalid = [r for r in rows if r['history_intervention_preserves_graph'] is False or r['graph_intervention_preserves_originals'] is False]
    if invalid:
        raise ValueError(f'Intervention isolation failed for {len(invalid)} packets')
    print(f'Audited {len(rows)} material packets and {len(controls)} paired deletion controls.')


if __name__ == '__main__':
    main()
