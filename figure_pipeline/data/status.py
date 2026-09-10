"""Separate execution evidence from scientific evidence availability."""
from __future__ import annotations
from typing import Any

def operational(snapshot: dict[str, Any]) -> list[dict]:
    rows = []
    for paper in snapshot['ROSTER']:
        claims = [c for c in snapshot['CLAIMS'] if c['paper_id'] == paper['paper_id']]
        analyzed = sum(c['gear_card_available'] for c in claims)
        usable = sum(c['usable_compared_historical_passage'] is True for c in claims)
        rows.append(dict(paper_id=paper['paper_id'], journal=paper['journal_name'], requested_claims=len(claims), analyzed_claims=analyzed, usable_evidence_claims=usable, execution_status=None, execution_status_reason='Snapshot lacks scheduler/attempt records; absence cannot distinguish not_run, running, failed.', saved_card_status='no_saved_card' if not analyzed else ('partial_saved_cards' if analyzed < len(claims) else 'all_cards_saved'), evidence_coverage_among_analyzed=usable/analyzed if analyzed else None, execution_fraction=analyzed/len(claims), graph_neighbor_fraction=sum(c['graph_neighbor_count'] > 0 for c in claims)/len(claims)))
    return rows
