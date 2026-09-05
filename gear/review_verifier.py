"""Innovation-only system comparison and reviewer agreement verification."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gear.config import GearConfig

from .review_contracts import (
    DiscussionResolvedReference,
    EvaluationCase,
    EvaluationSummary,
    FusionResult,
    GearBranchResult,
    GearEvidenceStatus,
    GraphBranchResult,
    ReviewerStance,
    ReviewerView,
)
from gear.artifacts import read_model, write_jsonl, write_model
from gear.model_client import LazyRoleClient


def _gear_stance(status: GearEvidenceStatus) -> ReviewerStance:
    return {
        GearEvidenceStatus.ANTECEDENT_FOUND: ReviewerStance.CHALLENGED,
        GearEvidenceStatus.RESIDUAL_EXTENSION: ReviewerStance.INCREMENTAL_OR_LIMITED,
        GearEvidenceStatus.BOUNDED_NO_ANTECEDENT: ReviewerStance.RECOGNIZED,
        GearEvidenceStatus.INCONCLUSIVE: ReviewerStance.UNRESOLVED,
        GearEvidenceStatus.INTERNALLY_UNSUPPORTED: ReviewerStance.CHALLENGED,
    }[status]


def load_predictions(system_name: str, path: Path) -> list[dict[str, object]]:
    if system_name == "graph_only":
        result = read_model(path, GraphBranchResult)
        return [{"id": x.claim_id, "text": x.claim_text, "stance": None} for x in result.claims]
    if system_name == "gear_only":
        result = read_model(path, GearBranchResult)
        return [{"id": x.claim.claim_id, "text": x.claim.normalized_claim_text, "stance": _gear_stance(x.status)} for x in result.claim_cards]
    if system_name in {"passive_fusion", "active_graph_gear"}:
        result = read_model(path, FusionResult)
        return [{"id": x.joint_claim_id, "text": x.statement, "stance": _gear_stance(x.evidence_status) if x.evidence_status else ReviewerStance.UNRESOLVED} for x in result.joint_claim_cards]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [{"id": str(x.get("claim_id", index)), "text": str(x.get("claim_text") or x.get("assessment") or ""), "stance": ReviewerStance(str(x.get("stance", "unresolved")))} for index, x in enumerate(payload.get("claims", []), 1)]


class InnovationEvaluator:
    def __init__(self, config: GearConfig, embedding_model: Path) -> None:
        self.config = config
        self.embedding_model = embedding_model
        self.judge = LazyRoleClient(config, "evaluation_judge")
        self._encoder: object | None = None

    def evaluate(self, system_name: str, predictions: list[dict[str, object]],
                 reference: DiscussionResolvedReference, output_dir: Path) -> EvaluationSummary:
        refs = reference.claims
        pairs = self._candidate_pairs(predictions, refs)
        details: list[EvaluationCase] = []
        matched_pred: set[str] = set()
        matched_ref: set[str] = set()
        for prediction_index, reference_index, similarity in pairs:
            prediction, human = predictions[prediction_index], refs[reference_index]
            prediction_id = str(prediction["id"])
            available = (
                prediction_id not in matched_pred
                and human.reviewer_claim_id not in matched_ref
            )
            is_match = available and self._judge_match(
                str(prediction["text"]), human.target_claim_text
            )
            if is_match:
                matched_pred.add(prediction_id)
                matched_ref.add(human.reviewer_claim_id)
            details.append(EvaluationCase(
                paper_id=reference.paper_id, system_name=system_name,
                predicted_claim_id=str(prediction["id"]),
                reference_claim_id=human.reviewer_claim_id,
                semantic_similarity=similarity, judge_match=is_match,
                predicted_stance=prediction.get("stance"), reference_stance=human.stance,
            ))
        precision = len(matched_pred) / len(predictions) if predictions else None
        recall = len(matched_ref) / len(refs) if refs else None
        if precision is None or recall is None:
            f1 = None
        else:
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        stance_f1 = self._stance_macro_f1(details)
        detail_path = output_dir / f"{system_name}_details.jsonl"
        write_jsonl(detail_path, details)
        summary = EvaluationSummary(
            system_name=system_name, paper_count=1,
            predicted_claim_count=len(predictions), reference_claim_count=len(refs),
            claim_precision=precision, claim_recall=recall, claim_f1=f1,
            stance_macro_f1=stance_f1, details_path=str(detail_path),
        )
        write_model(output_dir / f"{system_name}_summary.json", summary)
        return summary

    def _candidate_pairs(self, predictions: list[dict[str, object]], refs: object) -> list[tuple[int, int, float]]:
        if not predictions or not refs:
            return []
        texts = [str(x["text"]) for x in predictions] + [x.target_claim_text for x in refs]
        matrix = self._encode(texts)
        p = matrix[:len(predictions)]
        r = matrix[len(predictions):]
        scores = p @ r.T
        pairs = [(i, j, float(scores[i, j])) for i in range(len(predictions)) for j in range(len(refs)) if scores[i, j] >= 0.35]
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(str(self.embedding_model), trust_remote_code=True)
        return np.asarray(self._encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True))

    def _judge_match(self, predicted: str, reference: str) -> bool:
        raw = self.judge.generate_json(
            system="Decide whether two statements identify the same atomic scientific innovation. Shared topic alone is not a match. Return JSON.",
            user=json.dumps({"system_claim": predicted, "human_claim": reference}, ensure_ascii=False),
            response_schema={"type": "object", "properties": {"match": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["match", "reason"], "additionalProperties": False},
        )
        return bool(raw["match"])

    @staticmethod
    def _stance_macro_f1(details: list[EvaluationCase]) -> float | None:
        matched = [x for x in details if x.judge_match and x.predicted_stance is not None]
        if not matched:
            return None
        scores = []
        for label in ReviewerStance:
            tp = sum(x.predicted_stance == label and x.reference_stance == label for x in matched)
            fp = sum(x.predicted_stance == label and x.reference_stance != label for x in matched)
            fn = sum(x.predicted_stance != label and x.reference_stance == label for x in matched)
            denominator = 2 * tp + fp + fn
            scores.append(2 * tp / denominator if denominator else 0.0)
        return sum(scores) / len(scores)


def aggregate_summaries(summaries: list[EvaluationSummary], output_path: Path) -> None:
    grouped: dict[str, list[EvaluationSummary]] = defaultdict(list)
    for row in summaries:
        grouped[row.system_name].append(row)
    payload = {}
    for system, rows in grouped.items():
        claim_scores = [x.claim_f1 for x in rows if x.claim_f1 is not None]
        payload[system] = {
            "paper_count": len(rows),
            "scorable_paper_count": len(claim_scores),
            "macro_claim_f1": float(np.mean(claim_scores)) if claim_scores else None,
            "macro_stance_f1": float(np.mean([x.stance_macro_f1 for x in rows if x.stance_macro_f1 is not None])) if any(x.stance_macro_f1 is not None for x in rows) else None,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_human_agreement(
    evaluator: InnovationEvaluator,
    views: list[ReviewerView],
    output_dir: Path,
) -> list[EvaluationSummary]:
    """Auxiliary pairwise agreement; it is not the primary system metric."""
    summaries: list[EvaluationSummary] = []
    for left_index, left in enumerate(views):
        predictions = [
            {"id": claim.reviewer_claim_id, "text": claim.target_claim_text, "stance": claim.stance}
            for claim in left.claims
        ]
        for right in views[left_index + 1:]:
            reference = DiscussionResolvedReference(paper_id=right.paper_id, claims=right.claims)
            name = f"human_{left.reviewer_id}_vs_{right.reviewer_id}"
            summaries.append(evaluator.evaluate(name, predictions, reference, output_dir))
    return summaries
