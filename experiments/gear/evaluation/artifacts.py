"""Validated loading and context construction for evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gear.contracts import PaperIR
from gear.review_contracts import ReviewBundle, StructuredReview
from gear.trace import sha256_file, sha256_value

from .contracts import (
    EvaluationContextPack,
    EvaluationManifestV1,
    RevisionIssueLabel,
)


def load_manifest(path: Path) -> EvaluationManifestV1:
    manifest = EvaluationManifestV1.model_validate_json(path.read_text())
    base = path.parent.resolve()
    payload = manifest.model_dump(mode="python")
    payload["human_release_dir"] = _resolve(base, manifest.human_release_dir)
    for item, case in zip(payload["cases"], manifest.cases, strict=True):
        for field in (
            "manuscript_path",
            "metadata_path",
            "clean_run_dir",
            "prior_art_gold_path",
        ):
            value = getattr(case, field)
            if value is not None:
                item[field] = _resolve(base, value)
    return EvaluationManifestV1.model_validate(payload)


def load_human_release(
    release_dir: Path,
) -> tuple[dict[str, StructuredReview], dict[str, list[RevisionIssueLabel]]]:
    release_path = release_dir / "release_manifest.json"
    release = json.loads(release_path.read_text())
    review_path = release_dir / "human_structured_reviews.jsonl"
    _validate_declared_hash(review_path, release, "human_structured_reviews_sha256")
    reviews = {row.paper_id: row for row in _read_jsonl(review_path, StructuredReview)}
    if release.get("record_count") is not None and int(release["record_count"]) != len(
        reviews
    ):
        raise ValueError("human release review count mismatch")
    source_path = release_dir / "source_manifest.json"
    if source_path.is_file() and release.get("source_manifest_sha256") is not None:
        source_payload = json.loads(source_path.read_text())
        if release["source_manifest_sha256"] not in {
            sha256_value(source_payload),
            sha256_file(source_path),
        }:
            raise ValueError("human release source manifest hash mismatch")
    labels: dict[str, list[RevisionIssueLabel]] = {}
    sidecar = release_dir / "revision_issue_labels.jsonl"
    if sidecar.exists():
        _validate_declared_hash(sidecar, release, "revision_issue_labels_sha256")
        for row in _read_jsonl(sidecar, RevisionIssueLabel):
            labels.setdefault(row.paper_id, []).append(row)
        label_count = sum(len(rows) for rows in labels.values())
        if (
            release.get("revision_issue_label_count") is not None
            and int(release["revision_issue_label_count"]) != label_count
        ):
            raise ValueError("human release revision label count mismatch")
    return reviews, labels


def load_review_bundle(run_dir: Path) -> ReviewBundle:
    path = run_dir / "review_bundle.json"
    return ReviewBundle.model_validate_json(path.read_text())


def build_context_pack(
    paper_ir: PaperIR,
    human: StructuredReview,
    gear: StructuredReview,
) -> EvaluationContextPack:
    referenced = {
        key
        for review in (human, gear)
        for key in [
            *review.summary.evidence_keys,
            *(key for point in review.all_points() for key in point.evidence_keys),
        ]
        if key.startswith("P:")
    }
    selected = []
    for span in paper_ir.spans:
        section = " ".join(span.section_path).casefold()
        include = f"P:{span.span_id}" in referenced or any(
            label in section
            for label in (
                "abstract",
                "method",
                "result",
                "table",
                "figure",
                "limitation",
            )
        )
        if include:
            selected.append(span.model_dump(mode="json"))
    content_hash = sha256_value(selected)
    return EvaluationContextPack(
        paper_id=paper_ir.paper_id,
        spans=selected,
        omitted_span_count=max(0, len(paper_ir.spans) - len(selected)),
        content_sha256=content_hash,
    )


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            try:
                rows.append(model.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid {path.name} line {number}: {exc}") from exc
    return rows


def _validate_declared_hash(path: Path, release: dict[str, Any], field: str) -> None:
    declared = release.get(field)
    if declared is not None and declared != sha256_file(path):
        raise ValueError(f"human release hash mismatch: {path.name}")


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()


__all__ = [
    "build_context_pack",
    "load_human_release",
    "load_manifest",
    "load_review_bundle",
]
