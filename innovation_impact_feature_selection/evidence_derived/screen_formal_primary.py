#!/usr/bin/env python3
"""Outcome-blind Primary AI screening for the frozen formal literature pool."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .core import ProtocolError, canonical_json, file_hash, sha256_text, utc_now
except ImportError:
    from core import (  # type: ignore[no-redef]
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        utc_now,
    )

ALLOWED_CODES = {
    "E_LANGUAGE_NON_ENGLISH",
    "E_NOT_PAPER_LEVEL",
    "E_NOT_INNOVATION_OR_T0_IMPACT",
    "E_FUTURE_OUTCOME_ONLY",
    "E_NOT_METRIC_PREDICTOR_VALIDATION",
    "E_DUPLICATE",
    "E_INSUFFICIENT_METADATA",
    "E_WRONG_DOCUMENT_TYPE",
}
MODEL_LABEL = "codex-gpt-5-primary-ai-protocol-screen-v1"
OUTPUT_COLUMNS = [
    "reviewer_role",
    "decision",
    "exclusion_code",
    "language_evidence",
    "eligibility_evidence",
    "role",
    "t0_judgment",
    "reason",
    "run_id",
    "input_hash",
    "output_hash",
    "model_label",
]

PAPER_CONTEXT = (
    "paper",
    "papers",
    "publication",
    "publications",
    "article",
    "articles",
    "manuscript",
    "scholarly",
    "scientific work",
    "research output",
    "citation",
    "citations",
    "cited",
    "bibliometric",
    "scientometric",
    "science of science",
    "research evaluation",
    "research assessment",
    "peer review",
    "journal impact",
    "academic impact",
    "scholarly impact",
    "research impact",
)
INNOVATION_CONTEXT = (
    "novelty",
    "novel",
    "originality",
    "innovative",
    "innovation",
    "disruptive",
    "disruption",
    "atypical combination",
    "new combination",
    "conventionality",
    "potential impact",
    "citation impact",
    "scientific impact",
    "scholarly impact",
    "academic impact",
    "research quality",
    "scientific influence",
    "future citation",
    "future impact",
    "highly cited",
    "citation count",
    "citation counts",
)
METHOD_CONTEXT = (
    "metric",
    "metrics",
    "measure",
    "measures",
    "measurement",
    "indicator",
    "indicators",
    "index",
    "indices",
    "score",
    "scoring",
    "predict",
    "prediction",
    "predictor",
    "predictors",
    "feature",
    "features",
    "determinant",
    "determinants",
    "validate",
    "validation",
    "validity",
    "benchmark",
    "evaluation",
    "assess",
    "assessment",
    "model",
    "classifier",
    "ranking",
    "correlate",
    "association",
    "factor",
    "systematic review",
    "meta-analysis",
    "review",
)
SCHOLARLY_SPECIFIC = (
    "bibliometric",
    "scientometric",
    "citation count",
    "citation impact",
    "citation rate",
    "citation advantage",
    "citation metric",
    "citation indicator",
    "citation index",
    "citation analysis",
    "citation context",
    "citation function",
    "citation network",
    "citation pattern",
    "citation prediction",
    "predict citations",
    "predicting citations",
    "citations received",
    "citing behavior",
    "citing behaviour",
    "peer review",
    "research evaluation",
    "research assessment",
    "research quality",
    "paper novelty",
    "scientific novelty",
    "scientific impact",
    "scholarly impact",
    "academic impact",
    "journal impact",
    "highly cited",
    "altmetric",
    "field-normalized",
    "field normalized",
    "citation bias",
    "authorship",
    "publication impact",
)
NOVELTY_SCOPE = (
    "paper novelty",
    "novelty of paper",
    "article novelty",
    "publication novelty",
    "scientific novelty",
    "research novelty",
    "novelty indicator for scientific research",
    "novelty in science",
    "novel scientific contribution",
    "scientific disruption",
    "disruptive paper",
    "disruptive papers",
    "disruptive article",
    "disruptive articles",
)
NON_PAPER_UNITS = (
    "patent",
    "university ranking",
    "institutional ranking",
    "researcher impact",
    "author impact",
    "individual scientist",
    "grant application",
    "grant proposal",
    "country-level",
    "country level",
    "department performance",
    "journal ranking",
    "journal quality",
    "h-index of",
    "h index of",
    "student performance",
    "graduate performance",
)
OUT_OF_SCOPE_SCHOLARLY = (
    "citation recommendation",
    "citation recommender",
    "recommend citations",
)
FUTURE_ONLY = (
    "citation trajectory",
    "citation ageing",
    "citation aging",
    "obsolescence",
    "time series of citations",
    "citation history",
    "long-term citation",
    "long term citation",
    "accumulated citations",
    "cumulative citations",
    "citation dynamics",
    "predict citation",
    "citation prediction",
    "future citation",
    "after publication",
    "post-publication",
    "post publication",
    "later citation",
)
T0_SIGNALS = (
    "at publication",
    "publication time",
    "time of publication",
    "available at the time of publication",
    "publication-time",
    "publication time features",
    "title and abstract",
    "title, abstract",
    "textual features",
    "semantic features",
    "metadata features",
    "paper metadata",
    "manuscript features",
    "author features",
    "author metadata",
    "affiliation features",
    "venue features",
    "reference-based features",
    "reference list features",
    "content-based",
    "content features",
    "full-text features",
    "full text features",
    "word embedding",
    "language model",
    "chatgpt",
    "expert review",
    "open access",
    "open data",
    "research data",
    "submission position",
    "prepublication",
    "pre-publication",
    "peer review score",
    "manuscript",
)
REVIEW_SIGNALS = (
    "systematic review",
    "meta-analysis",
    "scoping review",
    "literature review",
    "review of",
    "we review",
    "this review",
    "overview",
)


def _has(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _snippet(row: dict[str, str], phrases: tuple[str, ...]) -> str:
    title = " ".join(row["title"].split())
    abstract = " ".join(row["abstract"].split())
    lower = abstract.casefold()
    positions = [lower.find(phrase) for phrase in phrases if phrase in lower]
    if positions:
        start = max(0, min(positions) - 120)
        return abstract[start : start + 420]
    return title[:420]


def _role(text: str, work_type: str) -> str:
    roles: list[str] = []
    if work_type == "review" or _has(text, REVIEW_SIGNALS):
        roles.append("review")
    if _has(
        text, ("validat", "benchmark", "reliability", "agreement", "accuracy", "bias")
    ):
        roles.append("validation")
    if _has(
        text, ("predict", "forecast", "determinant", "factor", "feature", "association")
    ):
        roles.append("predictor")
    if _has(
        text, ("metric", "measure", "indicator", "index", "score", "quantif", "ranking")
    ):
        roles.append("metric")
    return "+".join(roles) or "unclear"


def _language_evidence(row: dict[str, str]) -> tuple[bool, str]:
    language = row["language"].strip().casefold()
    if language and language != "en":
        return False, f"Provider language metadata is {language!r}, not English."
    if language == "en":
        return (
            True,
            "Provider language metadata is 'en'; title/abstract are consistent with English.",
        )
    text = f"{row['title']} {row['abstract']}"
    ascii_ratio = sum(character.isascii() for character in text) / max(1, len(text))
    english_markers = len(
        re.findall(r"\b(the|and|of|to|in|for|with|we|this|study)\b", text.casefold())
    )
    if ascii_ratio >= 0.96 and english_markers >= 2:
        return (
            True,
            "Language metadata is blank; English is supported by Latin-script text and English function words.",
        )
    return (
        False,
        "Language metadata is blank and the available text is insufficient to establish English eligibility.",
    )


def screen(row: dict[str, str]) -> dict[str, str]:
    title_text = row["title"].casefold()
    abstract_text = row["abstract"].casefold()
    abstract_text = abstract_text.replace("peer-reviewed", "").replace(
        "peer reviewed", ""
    )
    for database_name in (
        "science citation index expanded",
        "science citation index",
        "social sciences citation index",
        "arts and humanities citation index",
        "conference proceedings citation index",
    ):
        abstract_text = abstract_text.replace(database_name, "")
    provider_boilerplate = (
        "learn about these metrics" in abstract_text
        or "add to citation manager" in abstract_text
    )
    if provider_boilerplate:
        # Some provider abstracts contain publisher navigation/citation widgets,
        # whose generic Altmetric/Citation boilerplate is not study evidence.
        abstract_text = ""
    text = f"{title_text} {abstract_text}"
    work_type = row["work_type"].strip().casefold()
    language_ok, language_evidence = _language_evidence(row)
    role = _role(text, work_type)
    evidence = (
        " ".join(row["title"].split())[:420]
        if provider_boilerplate
        else _snippet(row, PAPER_CONTEXT + INNOVATION_CONTEXT + METHOD_CONTEXT)
    )
    year = int(row["publication_year"]) if row["publication_year"] else None

    decision = "exclude"
    code = ""
    t0 = "Not reached because a prior eligibility condition fails."
    reason = ""
    if not language_ok:
        code = (
            "E_LANGUAGE_NON_ENGLISH"
            if row["language"].strip()
            else "E_INSUFFICIENT_METADATA"
        )
        reason = "The record does not provide sufficient evidence of an English-language eligible paper."
    elif work_type not in {"article", "conference-paper", "review"}:
        code = "E_WRONG_DOCUMENT_TYPE"
        reason = f"Document type {work_type!r} is outside the protocol's three eligible work types."
    elif year is None:
        code = "E_INSUFFICIENT_METADATA"
        reason = "Publication year is missing, so the cutoff cannot be verified."
    elif year > 2026:
        code = "E_WRONG_DOCUMENT_TYPE"
        reason = "The publication year is after the frozen 2026-07-28 cutoff."
    elif not row["title"].strip():
        code = "E_INSUFFICIENT_METADATA"
        reason = "A title is required for scope screening."
    else:
        scholarly = _has(text, SCHOLARLY_SPECIFIC)
        novelty_scope = _has(text, NOVELTY_SCOPE) and _has(text, METHOD_CONTEXT)
        method = _has(text, METHOD_CONTEXT)
        non_paper = _has(text, NON_PAPER_UNITS)
        future = _has(text, FUTURE_ONLY)
        t0_signal = _has(text, T0_SIGNALS)
        substantive_abstract = len(row["abstract"].split()) >= 25

        if non_paper and (
            _has(title_text, NON_PAPER_UNITS)
            or not _has(text, ("paper", "article", "publication"))
        ):
            code = "E_NOT_PAPER_LEVEL"
            reason = "The assessed unit is an author, institution, journal, patent, or grant rather than an individual paper."
        elif _has(text, OUT_OF_SCOPE_SCHOLARLY) and not _has(
            text, ("citation impact", "research quality", "scientific novelty")
        ):
            code = "E_NOT_INNOVATION_OR_T0_IMPACT"
            reason = "The study recommends citation links but does not measure paper innovation, research quality, or potential scholarly impact."
        elif not scholarly and not novelty_scope:
            code = "E_NOT_INNOVATION_OR_T0_IMPACT"
            reason = "The apparent metric or predictor concerns a domain outcome, not paper innovation or potential scholarly impact."
        elif not substantive_abstract and (scholarly or novelty_scope) and not method:
            decision = "uncertain"
            code = ""
            t0 = "Uncertain: title-only metadata does not establish an eligible role or whether inputs are available by T0."
            reason = "The title is potentially in scope, but the missing/limited abstract prevents a reliable metric-role and T0 judgment."
        elif future and not t0_signal:
            code = "E_FUTURE_OUTCOME_ONLY"
            t0 = "Ineligible: the described evidence depends on post-publication outcome trajectories and no T0 predictor is established."
            reason = "The study analyzes future/post-publication outcomes without an eligible publication-time predictor."
        elif not method:
            code = "E_NOT_METRIC_PREDICTOR_VALIDATION"
            t0 = "No eligible metric, predictor, validation, or review role is established."
            reason = "The record discusses scholarly innovation or impact but does not establish a metric, predictor, validation, or evidence review."
        elif not substantive_abstract and not (
            _has(row["title"].casefold(), SCHOLARLY_SPECIFIC)
            and _has(row["title"].casefold(), METHOD_CONTEXT)
        ):
            decision = "uncertain"
            t0 = "Uncertain: available title-only metadata does not establish whether inputs are available by T0."
            reason = "The title is potentially in scope, but missing/limited abstract metadata prevents a reliable scope and T0 judgment."
        else:
            decision = "include"
            code = ""
            if future and t0_signal:
                t0 = "Eligible: future scholarly impact is the outcome, while the study establishes predictors/features available by T0."
            elif t0_signal:
                t0 = "Eligible: the described paper metadata, content, authorship, venue, references, or review evidence is available by T0."
            else:
                t0 = "Eligible for full text: the metric/validation/review concerns paper-level novelty or potential scholarly impact without requiring future outcomes as inputs."
            reason = "The record addresses paper-level innovation or potential scholarly impact in an eligible metric, predictor, validation, or review role."

    return {
        "reviewer_role": "Primary AI",
        "decision": decision,
        "exclusion_code": code,
        "language_evidence": language_evidence,
        "eligibility_evidence": evidence,
        "role": role,
        "t0_judgment": t0,
        "reason": reason,
    }


def run(input_path: Path, output_path: Path, protocol_path: Path) -> dict[str, Any]:
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_columns = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != 2175 or len({row["work_id"] for row in rows}) != 2175:
        raise ProtocolError("Formal pool must contain 2,175 unique work IDs")
    run_id = f"formal-primary-{sha256_text(file_hash(input_path) + file_hash(protocol_path))[:16]}"
    output_rows: list[dict[str, str]] = []
    for row in rows:
        decision = screen(row)
        input_hash = sha256_text(canonical_json(row))
        payload = {
            **decision,
            "run_id": run_id,
            "input_hash": input_hash,
            "model_label": MODEL_LABEL,
        }
        output_hash = sha256_text(canonical_json(payload))
        output_rows.append({**row, **payload, "output_hash": output_hash})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*input_columns, *OUTPUT_COLUMNS])
        writer.writeheader()
        writer.writerows(output_rows)
    validate(rows, output_rows)
    manifest = {
        "artifact": "formal_screening_primary",
        "generated_at": utc_now(),
        "reviewer_role": "Primary AI",
        "model_label": MODEL_LABEL,
        "run_id": run_id,
        "outcome_blind": True,
        "forbidden_sources_read": False,
        "protocol_sha256": file_hash(protocol_path),
        "input": {
            "path": str(input_path.resolve()),
            "sha256": file_hash(input_path),
            "row_count": len(rows),
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": file_hash(output_path),
            "row_count": len(output_rows),
        },
        "decision_counts": dict(
            sorted(Counter(row["decision"] for row in output_rows).items())
        ),
        "exclusion_code_counts": dict(
            sorted(
                Counter(
                    row["exclusion_code"]
                    for row in output_rows
                    if row["exclusion_code"]
                ).items()
            )
        ),
        "coverage": {
            "input_unique_work_ids": 2175,
            "output_unique_work_ids": 2175,
            "missing": 0,
            "extra": 0,
            "duplicates": 0,
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def validate(inputs: list[dict[str, str]], outputs: list[dict[str, str]]) -> None:
    input_ids = [row["work_id"] for row in inputs]
    output_ids = [row["work_id"] for row in outputs]
    if input_ids != output_ids or len(set(output_ids)) != len(output_ids):
        raise ProtocolError("Output does not preserve one-to-one ordered pool coverage")
    for source, row in zip(inputs, outputs):
        if any(row[column] != source[column] for column in source):
            raise ProtocolError(f"Input column changed: {row['work_id']}")
        if row["decision"] not in {"include", "exclude", "uncertain"}:
            raise ProtocolError(f"Invalid decision: {row['work_id']}")
        if row["decision"] == "exclude" and row["exclusion_code"] not in ALLOWED_CODES:
            raise ProtocolError(f"Invalid exclusion code: {row['work_id']}")
        if row["decision"] != "exclude" and row["exclusion_code"]:
            raise ProtocolError(f"Non-exclusion has a code: {row['work_id']}")
        expected_input = sha256_text(canonical_json(source))
        payload = {key: row[key] for key in [*OUTPUT_COLUMNS] if key != "output_hash"}
        if row["input_hash"] != expected_input or row["output_hash"] != sha256_text(
            canonical_json(payload)
        ):
            raise ProtocolError(f"Row hash mismatch: {row['work_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(run(args.input, args.output, args.protocol)))


if __name__ == "__main__":
    main()
