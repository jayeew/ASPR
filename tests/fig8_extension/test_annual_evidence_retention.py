from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.fig8_extension import publisher_delta_apply as module


@pytest.mark.parametrize('year,label,rule_label,expected', [
    (2024, 'False', 'False', True),
    (2025, 'False', 'False', False),
    (2024, 'True', 'False', False),
    (2024, 'False', 'True', False),
])
def test_annual_citation_requires_same_year_and_current_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    year: int, label: str, rule_label: str, expected: bool,
) -> None:
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path / 'rules.json'
    path.write_text(json.dumps([{'source_id': 'S1', 'year': 2024, 'label': rule_label}]))
    reference = 'rules.json#S1:2024'
    old = {'source_id': 'S1', 'online_year': 2024,
           'nature_portfolio_at_verified_online_year': 'False',
           'portfolio_evidence_ids': json.dumps([reference])}
    assert module.retained_annual_evidence(old, year, label) == ([reference] if expected else [])


def test_dated_primary_citation_survives_matching_annual_label() -> None:
    old = {'source_id': 'S1', 'online_year': 2024, 'duplicate_of': '',
           'nature_portfolio_at_verified_online_year': 'False',
           'portfolio_evidence_ids': json.dumps(['aps-proof', 'stale-proof'])}
    evidence = {'id': 'aps-proof', 'identity': 'matched', 'family': 'aps',
                'kind': 'article', 'class': 'research', 'online_date': '2024-03-01'}
    result = module.updated_decision(old, [evidence], [], {('S1', 2024): 'False'})
    assert json.loads(result['portfolio_evidence_ids']) == ['aps-proof']
    moved = {**evidence, 'online_date': '2025-03-01'}
    result = module.updated_decision(old, [moved], [], {('S1', 2025): 'False'})
    assert json.loads(result['portfolio_evidence_ids']) == []
    mismatch = {**evidence, 'identity': 'title_mismatch'}
    result = module.updated_decision(old, [mismatch], [], {('S1', 0): 'False'})
    assert json.loads(result['portfolio_evidence_ids']) == []
