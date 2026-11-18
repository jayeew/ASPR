"""Deterministic original-passage selection with an explicit coverage record."""
from __future__ import annotations

from typing import Any

from figure_pipeline.fig3_revision.materials import blocks, terms

from .io import encoded

SOURCE_FIELDS = ('source_id', 'original_source_id', 'source_type', 'title', 'doi', 'work_id',
                 'publication_date', 'year', 'historical_use', 'source_notice')
STRUCTURAL_KEYS = ('"metrics"', '"joint_component_merge_count"',
                   '"joint_newly_connected_historical_pairs"', '"historical_components_before"',
                   '"historical_components_after"', '"shared_historical_neighbors"')


def source_blocks(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in sources:
        for block in blocks(source['text'], size=2400):
            # Small overlap keeps short quotations intact across a block boundary.
            start = max(0, block['start'] - 160)
            result.append({**{key: source[key] for key in SOURCE_FIELDS if key in source},
                           'start': start, 'end': block['end'],
                           'text': source['text'][start:block['end']]})
    return result


def evidence_quotes(query: Any) -> list[str]:
    if isinstance(query, dict):
        own = [value for key, value in query.items() if key in {'quote', 'manuscript_quote'}
               and isinstance(value, str) and len(value.strip()) >= 12]
        return own + [quote for value in query.values() for quote in evidence_quotes(value)]
    if isinstance(query, list):
        return [quote for value in query for quote in evidence_quotes(value)]
    return []


def material_group(row: dict[str, Any]) -> str:
    if row['source_type'] == 'manuscript':
        return 'manuscript'
    if row['source_type'] in {'graph_observation', 'historical_claim_summary'}:
        return 'graph'
    return 'historical'


def rank_blocks(candidates: list[dict[str, Any]], query: Any) -> list[tuple[int, int, dict[str, Any]]]:
    query_text = encoded(query)
    vocabulary, quotes = terms(query_text), evidence_quotes(query)
    ranked = []
    for index, row in enumerate(candidates):
        score = len(vocabulary & terms(row['text']))
        score += sum(40 for quote in quotes if quote in row['text'])
        if row['source_id'] in query_text:
            score += 12
        if row['source_type'] == 'graph_observation':
            score += 24 * any(key in row['text'] for key in STRUCTURAL_KEYS)
            score += 8 * row['source_id'].startswith('JOINT_GRAPH:')
        ranked.append((score, index, row))
    return sorted(ranked, key=lambda item: (-item[0], item[1]))


def diversify(rows: list[tuple[int, int, dict[str, Any]]]) -> list[tuple[int, int, dict[str, Any]]]:
    first, rest, seen = [], [], set()
    for row in rows:
        source = row[2]
        ident = str(source.get('doi') or source.get('work_id') or source['source_id']).casefold()
        (rest if ident in seen else first).append(row)
        seen.add(ident)
    return first + rest


class Selection:
    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.used = 2  # JSON array brackets; each later row also contributes a comma.
        self.rows: list[dict[str, Any]] = []
        self.selected: set[int] = set()
        self.by_group = {'manuscript': 0, 'historical': 0, 'graph': 0}

    def take(self, rows: list[tuple[int, int, dict[str, Any]]], target: int | None = None) -> None:
        taken = 0
        for _, index, row in rows:
            if target is not None and taken >= target:
                return
            cost = len(encoded(row)) + int(bool(self.rows))
            if index in self.selected or self.used + cost > self.budget:
                continue
            self.selected.add(index)
            self.rows.append(row)
            self.used += cost
            taken += cost
            self.by_group[material_group(row)] += cost


def pack_sources(sources: list[dict[str, Any]], query: Any, budget: int = 72000) -> dict[str, Any]:
    """Diversify original sources; never substitute generated judgments for evidence."""
    if budget < 2:
        raise ValueError('Source budget must accommodate a JSON array')
    candidates = source_blocks(sources)
    ranked = rank_blocks(candidates, query)
    grouped = {name: diversify([row for row in ranked if material_group(row[2]) == name])
               for name in ('manuscript', 'historical', 'graph')}
    chosen = Selection(budget)
    # Reserve manuscript evidence first; unfilled allocations flow to the final pass.
    chosen.take(grouped['manuscript'], round(.30 * budget))
    chosen.take(grouped['historical'], round(.45 * budget))
    chosen.take(grouped['graph'], round(.25 * budget))
    chosen.take(diversify(ranked))
    selected, used = chosen.rows, chosen.used
    return {'source_passages': selected, 'material_coverage': {
        'available_sources': len(sources), 'available_blocks': len(candidates),
        'selected_sources': len({r['source_id'] for r in selected}),
        'selected_blocks': len(selected), 'selected_characters': used,
        'budget_characters': budget,
        'selected_characters_by_group': chosen.by_group,
        'allocation_targets': {'manuscript': .30, 'historical': .45, 'graph': .25},
        'selection': ('Original passages ranked by query overlap, exact supplied source quotations, and recorded '
                      'structural-field anchors; diversified by original work. Manuscript first (30%), historical '
                      '(45%), graph (25%), then unused capacity reassigned. No new retrieval or generated source summary.'),
        'source_blocks_overlap_characters': 160,
        'scope': 'Selected passages only. Absence here does not establish absence in the full source or literature.'}}


def packet_sources(packet: dict[str, Any]) -> list[dict[str, Any]]:
    """Validation sees only passages actually delivered, not unobserved full sources."""
    return [dict(row) for row in packet['source_passages']]


def full_source_packet(bundle: dict[str, Any]) -> dict[str, Any]:
    sources = {s['source_id']: s for s in bundle['sources']}
    selected = []
    for span in bundle['full_rerun_selection']:
        source = sources[span['source_id']]
        start, end = span['start'], span['end']
        if not 0 <= start < end <= len(source['text']):
            raise ValueError('Full evidence interval is outside its original source')
        selected.append({**{key: source[key] for key in SOURCE_FIELDS if key in source},
                         'start': start, 'end': end, 'text': source['text'][start:end]})
    by_group = {key: sum(len(encoded(s)) for s in selected if material_group(s) == key)
                for key in ('manuscript', 'historical', 'graph')}
    return {'source_passages': selected, 'material_coverage': {
        'full_rerun': True, 'selected_sources': len({s['source_id'] for s in selected}),
        'selected_blocks': len(selected), 'selected_characters': len(encoded(selected)),
        'selected_characters_by_group': by_group, 'budget_characters': None,
        'selection': 'Complete manuscript and merged original GEAR/Graph/Full evidence intervals; no truncation.',
        'scope': 'Complete supplied manuscript; selected existing historical and graph evidence only.'}}
