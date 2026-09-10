"""Pure paper-paired statistics; no rendering or live data access."""
from __future__ import annotations
from typing import Any
import numpy as np

METRICS = [('coverage_full_rate', 'Contribution coverage ↑'), ('found_and_agreed_rate', 'Found and agreed ↑'), ('reason_complete_rate', 'Complete reasons ↑'), ('contradiction_rate', 'Same-scope contradiction ↓')]


def interval(values: list[float] | np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        raise ValueError('Cannot bootstrap an empty comparison')
    draws = np.random.default_rng(6082026).choice(arr, (10000, len(arr)), replace=True).mean(axis=1)
    return [float(arr.mean()), *map(float, np.quantile(draws, [.025, .975]))]


def paired(snapshot: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    evaluation = snapshot['EVAL']
    summaries, points = [], []
    for metric, label in METRICS:
        ids = sorted(p for p in evaluation['graph'] if p in evaluation['direct_llm'] and all(evaluation[s][p]['metrics'].get(metric) is not None for s in ['graph', 'direct_llm']))
        direct = np.array([evaluation['direct_llm'][p]['metrics'][metric] * 100 for p in ids])
        graph = np.array([evaluation['graph'][p]['metrics'][metric] * 100 for p in ids])
        summaries.append(dict(metric=metric, label=label, n=len(ids), direct=interval(direct), graph=interval(graph), delta=interval(graph-direct)))
        points.extend(dict(paper_id=p, metric=metric, direct=float(d), graph=float(g), delta=float(g-d)) for p, d, g in zip(ids, direct, graph))
    return summaries, points


def transitions(snapshot: dict[str, Any]) -> list[dict]:
    rows = []
    for paper, graph in snapshot['EVAL']['graph'].items():
        direct = snapshot['EVAL']['direct_llm'].get(paper, {})
        sides = [{m['reference_id']: m for m in (v.get('evaluation') or {}).get('matches', [])} for v in [direct, graph]]
        for ref in sorted(set(sides[0]) & set(sides[1])):
            a, b = (s[ref] for s in sides)
            rows.append(dict(paper_id=paper, reference_id=ref, direct_scope=a['scope'], graph_scope=b['scope'], direct_stance=a['predicted_stance'], graph_stance=b['predicted_stance'], direct_reason=a['reason_coverage'], graph_reason=b['reason_coverage'], direct_rationale=a['rationale'], graph_rationale=b['rationale'], independent_correctness=None))
    return rows
