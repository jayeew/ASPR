"""Simple process-calibration metrics for a separately trained calibrator."""

from __future__ import annotations

from typing import Dict, Sequence


def calibration_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    bins: int = 10,
) -> Dict[str, float | None]:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be paired")
    if not probabilities:
        return {"brier": None, "ece": None, "auroc": None}
    for value in probabilities:
        if not 0.0 <= value <= 1.0:
            raise ValueError("calibrated probabilities must lie in [0, 1]")
    brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(outcomes)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, probability in enumerate(probabilities)
            if lower <= probability < upper
            or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = sum(probabilities[item] for item in members) / len(members)
        accuracy = sum(outcomes[item] for item in members) / len(members)
        ece += len(members) / len(outcomes) * abs(confidence - accuracy)
    return {"brier": brier, "ece": ece, "auroc": _auroc(probabilities, outcomes)}


def _auroc(probabilities: Sequence[float], outcomes: Sequence[int]) -> float | None:
    positives = [score for score, outcome in zip(probabilities, outcomes) if outcome]
    negatives = [
        score for score, outcome in zip(probabilities, outcomes) if not outcome
    ]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


__all__ = ["calibration_metrics"]
