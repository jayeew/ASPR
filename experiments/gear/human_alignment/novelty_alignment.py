"""Novelty conclusion metrics reported conditional on availability."""

from __future__ import annotations

from typing import Dict, Sequence

from gear.review_contracts import NoveltyJudgment

LABELS = tuple(NoveltyJudgment)


def novelty_alignment_metrics(
    human: Sequence[NoveltyJudgment],
    agent: Sequence[NoveltyJudgment],
) -> Dict[str, object]:
    if len(human) != len(agent):
        raise ValueError("human and agent novelty labels must be paired")
    if not human:
        return {
            "n": 0,
            "macro_f1": None,
            "balanced_accuracy": None,
            "weighted_kappa": None,
            "confusion_matrix": {},
        }
    matrix = {left.value: {right.value: 0 for right in LABELS} for left in LABELS}
    for truth, prediction in zip(human, agent):
        matrix[truth.value][prediction.value] += 1
    recalls = []
    f1s = []
    for label in LABELS:
        tp = matrix[label.value][label.value]
        fn = sum(matrix[label.value].values()) - tp
        fp = sum(row[label.value] for row in matrix.values()) - tp
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = recalls[-1]
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "n": len(human),
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "weighted_kappa": _weighted_kappa(human, agent),
        "confusion_matrix": matrix,
    }


def human_human_agreement(
    first: Sequence[NoveltyJudgment],
    second: Sequence[NoveltyJudgment],
) -> Dict[str, object]:
    metrics = novelty_alignment_metrics(first, second)
    metrics["raw_agreement"] = (
        sum(left == right for left, right in zip(first, second)) / len(first)
        if first
        else None
    )
    return metrics


def _weighted_kappa(
    human: Sequence[NoveltyJudgment], agent: Sequence[NoveltyJudgment]
) -> float:
    order = {label: index for index, label in enumerate(LABELS)}
    maximum = max(len(LABELS) - 1, 1)
    observed = sum(
        ((order[left] - order[right]) / maximum) ** 2
        for left, right in zip(human, agent)
    ) / len(human)
    human_counts = [sum(label == item for item in human) for label in LABELS]
    agent_counts = [sum(label == item for item in agent) for label in LABELS]
    expected = 0.0
    total = len(human)
    for left_index, left_count in enumerate(human_counts):
        for right_index, right_count in enumerate(agent_counts):
            expected += (
                left_count
                * right_count
                / (total * total)
                * ((left_index - right_index) / maximum) ** 2
            )
    return 1.0 - observed / expected if expected else 1.0


__all__ = ["human_human_agreement", "novelty_alignment_metrics"]
