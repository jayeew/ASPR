"""Build the H2 final scope triage for the v3 coverage benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "v3_coverage_scope_triage_H2_v4.csv"
BRIEF = ROOT / "V3_COVERAGE_SCOPE_TRIAGE_BRIEF_V4.md"
OUTPUT = ROOT / "outputs" / "v3_coverage_scope_triage_H2_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "v3_coverage_scope_triage_H2_completed_v4.manifest.json"


def sha256(path: Path) -> str:
    """Return an artifact SHA-256 hash."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalized(value: str) -> str:
    """Normalize a label for transparent, repeatable scope rules."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def has_any(label: str, phrases: set[str]) -> bool:
    """Check complete phrases against a normalized label."""
    padded = f" {label} "
    return any(f" {phrase} " in padded for phrase in phrases)


CLEAR_EXCLUSION_PHRASES = {
    "amstar", "blinded outcome", "boolean search", "case report", "clinical", "confounding",
    "consort", "core outcome", "diagnostic", "effect size", "endpoint", "evidence applicability",
    "evidence consistency", "evidence directness", "evidence dose response", "evidence generalizability",
    "evidence relevance", "evidence reliability", "evidence strength", "fragility index", "gold standard",
    "hardy weinberg", "imaging protocol", "included study", "inter reviewer", "ischemia",
    "longitudinal imaging", "mendelian randomization", "meta analysis", "multi reviewer", "non radiomic",
    "outcome measure", "outcome reporting", "participant demographics", "patient selection", "phantom study",
    "pleiotropy", "prespecified outcome", "prespecified review", "primary outcome", "prisma",
    "protocol defined", "randomization", "reported effect", "reporting bias", "review ",
    "reviewer ", "risk of bias", "search method", "selection bias", "spin bias", "statistical ",
    "study design", "study geographic", "study measurement", "study quality", "study sample",
    "study sampling", "study site", "study spatial", "study sponsor", "study validity", "surgeon",
    "surgical", "systematic review", "teacher education", "technique description", "trial enrolment",
    "winner s curse", "animal age", "biomedical study", "economic evaluation", "value for money",
    "evaluation practice", "experimental study", "experimental control", "exposure identification",
    "external validity", "methodological quality", "multicenter study", "phenotypic feature",
    "preclinical", "program evaluation", "publication based trial", "researcher participant",
    "publication and funding bias", "researcher reflexivity", "species interaction", "statistical power", "statistical precision",
}

AMBIGUOUS_EXCLUSION_LABELS = {
    "anecdotal only evaluation practice", "biological justification", "biological plausibility",
    "clinical research status", "de implementation recommendation", "experimental platform",
    "experimental scalability", "international study design", "methodological quality predictor",
    "methodological quality", "methodological quality score", "mmr question method alignment",
    "researcher participant role overlap", "researcher reflexivity", "scoti score",
    "study geographic region", "study site environmental context", "study sponsor identity",
    "study design adjustment covariate", "teacher education manuscript quality",
    "technique description reporting", "user based evaluation practice",
}

PAPER_LEVEL_TERMS = {
    "abstract", "article", "author", "authorship", "backward citation", "citation", "cited",
    "coauth", "collaboration", "conference", "conflict of interest", "country", "data ",
    "disciplin", "document", "doi", "field", "funding", "institution", "interdisciplin",
    "journal", "keyword", "knowledge", "latent", "linguistic", "manuscript", "network",
    "novel", "open access", "open science", "orcid", "paper", "publication", "reference",
    "research code", "research data", "scientific writing", "team", "tenure", "title", "topic",
    "writing", "access", "reader cognitive", "public full text", "shared data", "subject classification",
}

DIRECT_TERMS = {
    "novel", "originality", "creativity", "interdisciplin", "diversity", "rao stirling",
    "conventionality", "new concept", "innovation inertia", "knowledge combination", "knowledge synergy",
    "research pivot", "topic redundancy", "topical diversity",
}

CONTEXT_TERMS = {
    "abstract availability", "abstract language", "abstract length", "access", "affiliation", "country",
    "document type", "journal", "language", "publication year", "publication recency", "title", "topic",
    "field year", "doi", "open access", "public full text", "data availability", "article length",
}


def is_clear_exclusion(label: str, row: dict[str, str]) -> bool:
    """Identify labels unambiguously outside a paper-feature recovery scope."""
    if row["ai_triage_decision"] == row["h1_triage_decision"] == "scope_exclude":
        return True
    if label in AMBIGUOUS_EXCLUSION_LABELS:
        return False
    return has_any(label, CLEAR_EXCLUSION_PHRASES)


def is_plausibly_paper_level(label: str, row: dict[str, str]) -> bool:
    """Identify labels that can plausibly yield a publication-time paper feature."""
    if row["ai_triage_decision"] == row["h1_triage_decision"] == "recover_priority":
        return True
    if has_any(label, PAPER_LEVEL_TERMS):
        return True
    return row["archived_scope_role"] in {
        "direct_innovation", "t0_opportunity", "context_control"
    } and "study" not in label and "review" not in label


def recover_role(label: str, row: dict[str, str]) -> str:
    """Assign only a tentative H2 scope role for a recovery action."""
    if has_any(label, DIRECT_TERMS) or row["archived_scope_role"] == "direct_innovation":
        return "direct_innovation"
    if row["archived_scope_role"] == "t0_opportunity":
        return "t0_opportunity"
    if has_any(label, CONTEXT_TERMS) or row["archived_scope_role"] == "context_control":
        return "context_control"
    return "t0_substantive"


def source_terms(row: dict[str, str]) -> str:
    """Return three concise English source-search phrases for a non-exclusion."""
    name = re.sub(r"\s+", " ", row["canonical_name_en"].strip().lower())
    return f"{name}; scholarly publication; measurement"


def fill_row(row: dict[str, str]) -> None:
    """Fill exactly the five H2 fields for one benchmark label."""
    label = normalized(row["canonical_name_en"])
    if is_clear_exclusion(label, row):
        row["h2_final_triage_decision"] = "scope_exclude"
        row["h2_final_scope_role_assessment"] = "out_of_scope"
        row["h2_final_rationale"] = (
            f"{row['canonical_name_en']} denotes a clinical/study-specific outcome or procedure, systematic-review "
            "method, post-publication result, or other non-paper-level construct outside the v4 target."
        )
        row["h2_final_minimum_source_evidence_needed"] = "none_for_clear_scope_exclusion"
        row["h2_final_search_terms_en"] = ""
        return
    if is_plausibly_paper_level(label, row):
        role = recover_role(label, row)
        row["h2_final_triage_decision"] = "recover_priority"
        row["h2_final_scope_role_assessment"] = role
        row["h2_final_rationale"] = (
            f"{row['canonical_name_en']} plausibly denotes a paper-level {role.replace('_', ' ')} construct "
            "observable at publication; recover original English application evidence before any operational decision."
        )
        row["h2_final_minimum_source_evidence_needed"] = "original application"
        row["h2_final_search_terms_en"] = source_terms(row)
        return
    row["h2_final_triage_decision"] = "needs_source_evidence"
    row["h2_final_scope_role_assessment"] = "uncertain"
    row["h2_final_rationale"] = (
        f"{row['canonical_name_en']} alone does not establish a paper-level publication-time construct versus a "
        "study-specific, procedural, or post-publication construct; original source evidence is needed before scope judgment."
    )
    row["h2_final_minimum_source_evidence_needed"] = "original application"
    row["h2_final_search_terms_en"] = source_terms(row)


def main() -> None:
    """Create completed scope triage CSV and a traceability manifest."""
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or len(rows) != 432:
        raise ValueError("Expected the 432-row v3 coverage scope triage input.")
    h2_fields = [
        "h2_final_triage_decision", "h2_final_scope_role_assessment", "h2_final_rationale",
        "h2_final_minimum_source_evidence_needed", "h2_final_search_terms_en",
    ]
    protected = [name for name in fieldnames if name not in h2_fields]
    protected_before = [{name: row[name] for name in protected} for row in rows]
    for row in rows:
        fill_row(row)
    if protected_before != [{name: row[name] for name in protected} for row in rows]:
        raise AssertionError("A non-H2 field changed.")
    allowed_decisions = {"recover_priority", "scope_exclude", "needs_source_evidence"}
    allowed_roles = {
        "direct_innovation", "t0_substantive", "t0_opportunity", "context_control",
        "out_of_scope", "uncertain",
    }
    if any(row["h2_final_triage_decision"] not in allowed_decisions for row in rows):
        raise ValueError("Invalid final scope decision.")
    if any(row["h2_final_scope_role_assessment"] not in allowed_roles for row in rows):
        raise ValueError("Invalid final scope role.")
    if any(not row["h2_final_rationale"] for row in rows):
        raise ValueError("Every row requires a rationale.")
    if any(
        not row["h2_final_search_terms_en"] and row["h2_final_triage_decision"] != "scope_exclude"
        for row in rows
    ):
        raise ValueError("Only clear exclusions may have blank search terms.")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    review_required = [row for row in rows if row["h2_review_required"] == "1"]
    manifest = {
        "schema_version": "v3_coverage_scope_triage_h2_manifest_v4",
        "status": "complete",
        "reviewer_role": "H2",
        "reviewer_id": "H2_V3_COVERAGE_SCOPE_ADJUDICATOR_V4",
        "model": "codex_configured_default",
        "model_digest": "codex-thread:/root/h2_adjudication",
        "review_completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "method": "Blind construct-level triage using only the label and v4 scope brief. Decisions specify only source-recovery priority, never formula verification, data mapping, dimension formation, or feature approval. Clear clinical/study/review/procedural/non-paper labels are excluded; plausible publication-time paper constructs are recovery priorities; ambiguous labels require original source evidence.",
        "qwen_or_ollama_used": False,
        "input_artifacts": {str(INPUT): sha256(INPUT), str(BRIEF): sha256(BRIEF)},
        "output_artifact": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "feature_count": len(rows),
        "h2_review_required": {"required_count": len(review_required), "completed_count": len(review_required)},
        "decision_counts": dict(sorted(Counter(row["h2_final_triage_decision"] for row in rows).items())),
        "scope_role_counts": dict(sorted(Counter(row["h2_final_scope_role_assessment"] for row in rows).items())),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
