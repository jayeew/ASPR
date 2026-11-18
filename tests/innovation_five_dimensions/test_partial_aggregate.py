from __future__ import annotations

import importlib
from pathlib import Path

from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS
from figure_pipeline.innovation_five_dimensions.io import write

MODULE = importlib.import_module('figure_pipeline.innovation_five_dimensions.aggregate')


def assessment(paper: str, method: str) -> dict:
    rows = []
    for dim in DIMENSIONS:
        state = 'assessed'
        if paper == 'P4' or (paper in {'P1', 'DAMAGE'} and dim == 'historical_originality'):
            state = 'technical_failure'
        elif paper == 'P2' and dim == 'historical_originality':
            state = 'unresolved'
        score = (0 if method == 'full' else 2) if state == 'assessed' else None
        checks = [{'support': 'supported', 'scope_correct': True}] if state == 'assessed' else []
        rows.append({'dimension': dim, 'status': state, 'score': score, 'checks': checks,
                     'report_quotes': [], 'reason': 'reference quotation failure' if state == 'technical_failure'
                     else 'Original scientific state preserved.'})
    return {'dimensions': rows}


def test_partial_reference_keeps_other_dimensions_and_distinguishes_denominators(
    tmp_path: Path, monkeypatch,
) -> None:
    roster = [{'paper_id': f'P{i}', 'cohort': 'main', 'methods': ['full', 'gear']} for i in range(1, 5)]
    roster.append({'paper_id': 'DAMAGE', 'cohort': 'damage', 'methods': ['full']})
    write(tmp_path / 'inputs/roster.json', roster)
    monkeypatch.setattr(MODULE, 'load_result', lambda output, paper, method: (assessment(paper, method), None))
    result = MODULE.aggregate(tmp_path)
    historical = next(r for r in result['summary'] if r['cohort'] == 'main' and r['method'] == 'full'
                      and r['dimension'] == 'historical_originality')
    assert historical['n'] == 1 and historical['estimate'] == 0
    assert historical['n_unresolved'] == 1 and historical['n_technical_failure'] == 2
    assert historical['expected_papers'] == sum(historical[k] for k in
        ('n', 'n_unresolved', 'n_not_applicable', 'n_technical_failure', 'n_missing'))
    pairs = {r['dimension']: r for r in result['paired_effects'] if r['comparator'] == 'gear'}
    assert pairs['historical_originality']['n'] == 1
    assert pairs['historical_originality']['estimate'] == -2
    assert pairs['historical_originality']['expected_pairs'] == 4
    assert pairs['historical_originality']['non_assessed_pairs'] == 3
    assert pairs['knowledge_organisation']['n'] == 3
    assert pairs['knowledge_organisation']['estimate'] == -2
    counts = result['completion']
    assert counts['main_completed_evaluations'] == 4
    assert counts['main_partial_evaluations'] == 2
    assert counts['main_technical_failure_evaluations'] == 2
    assert counts['main_available_evaluations'] == 6
    assert counts['damage_completed_evaluations'] == 0
    assert counts['damage_partial_evaluations'] == 1
    assert counts['damage_available_evaluations'] == 1
    support = next(r for r in result['support_summary'] if r['cohort'] == 'main' and r['method'] == 'full'
                   and r['dimension'] == 'historical_originality')
    assert support['evaluated_reports'] == 2  # assessed + original scientific unresolved, no technical failure
    assert support['reports_with_checks'] == 1
    assert support['supported_estimate'] == 1
    assert len(result['technical_failures']) == 13  # 2 main partial + 10 all-failed + 1 case
    assert (tmp_path / 'tables/technical_failures.csv').is_file()
