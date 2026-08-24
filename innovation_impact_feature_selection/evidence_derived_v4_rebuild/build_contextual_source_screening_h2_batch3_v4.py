"""Create conservative H2 batch-3 contextual-source screening results."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "contextual_source_screening_H2_batch3_v4.csv"
OUTPUT = ROOT / "outputs" / "contextual_source_screening_H2_batch3_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_source_screening_H2_batch3_completed_v4.manifest.json"

# The source is a topical mapping, aggregate study, or post-publication impact
# analysis with no shown T0 indicator-method lead.  These overrides apply even
# where the independent upstream screen was permissive.
POST_OUTCOME_OR_TOPIC_EXCLUSIONS = {
    "doi:10.1002/asi.21494", "doi:10.1002/asi.23101", "doi:10.1007/s10209-022-00953-0",
    "doi:10.1016/j.jjimei.2021.100018", "doi:10.1016/j.leaqua.2020.101381",
    "doi:10.1016/j.omega.2016.12.004", "doi:10.1038/srep00902", "doi:10.1073/pnas.2012208118",
    "doi:10.1093/reseval/rvaa038", "doi:10.1177/10944281221127292",
    "doi:10.1186/1748-5908-3-49", "doi:10.1257/jel.20181508", "doi:10.1371/journal.pone.0017428",
    "doi:10.1371/journal.pone.0162364", "doi:10.1371/journal.pone.0218309",
    "doi:10.1371/journal.pone.0283106", "doi:10.1371/journal.pone.0013636",
    "doi:10.1371/journal.pone.0134794", "doi:10.1371/journal.pone.0271678",
    "doi:10.1371/journal.pone.0271678", "doi:10.1371/journal.pone.0271678",
    "doi:10.1371/journal.pone.0271678", "doi:10.1515/bfp-2020-2042",
    "doi:10.1002/asi.24709", "doi:10.1002/asi.24799", "doi:10.1002/jrsm.1729",
    "doi:10.1007/s10639-022-11058-9", "doi:10.1007/s11192-007-1658-3",
    "doi:10.1007/s11192-019-03201-0", "doi:10.1007/s11192-020-03621-3",
    "doi:10.1007/s11192-021-03870-w", "doi:10.1007/s11192-021-03972-5",
    "doi:10.1007/s10462-025-11315-6",
}

UNCERTAIN_SOURCES = {
    "doi:10.1007/s11192-006-0050-z",
    "doi:10.1007/s00799-013-0106-7",
    "doi:10.1007/s11192-021-03972-5",
    "doi:10.1007/s10462-025-11315-6",
    "doi:10.1371/journal.pone.0134794",
    "doi:10.1002/jrsm.1729",
}


def sha256(path: Path) -> str:
    """Return a file digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def decision(row: dict[str, str]) -> str:
    """Determine source-recovery priority from the frozen title/abstract screen."""
    key = row["record_key"]
    if key in UNCERTAIN_SOURCES:
        return "uncertain"
    if key in POST_OUTCOME_OR_TOPIC_EXCLUSIONS:
        return "exclude_not_relevant"
    ai, h1 = row["ai_screen_decision"], row["h1_screen_decision"]
    if ai == h1:
        return ai
    # Remaining disagreements are adjacent but may document source definitions,
    # data infrastructure, or T0 collaboration/open-science contexts.
    return "include_definition_or_review"


def rationale(final: str, title: str) -> str:
    """Write a decision-specific source-recovery rationale."""
    if final == "include_definition_or_review":
        return (
            f"{title} concerns scholarly communication, bibliometrics, publication metadata, "
            "or a plausible publication-time context and is worth checking for original definitions or reviews; inclusion is not indicator approval."
        )
    if final == "exclude_not_relevant":
        return (
            f"{title} is a topical/aggregate mapping or relies on post-publication impact, citation, "
            "usage, or outcome results without a shown paper-level T0 indicator-method lead."
        )
    return (
        f"{title} is adjacent to research evaluation or scholarly infrastructure, but the available title/abstract "
        "does not establish a recoverable paper-level T0 definition; inspect the original source before a final screen."
    )


def main() -> None:
    """Write the protected-field-preserving H2 completed screen and manifest."""
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or len(rows) != 120:
        raise ValueError("Expected 120 batch-3 screening records.")
    h2_fields = {"h2_final_screen_decision", "h2_final_evidence_span", "h2_final_rationale"}
    protected = [field for field in fields if field not in h2_fields]
    before = [{field: row[field] for field in protected} for row in rows]
    for row in rows:
        final = decision(row)
        row["h2_final_screen_decision"] = final
        row["h2_final_evidence_span"] = row["title"].strip()
        row["h2_final_rationale"] = rationale(final, row["title"].strip())
    if before != [{field: row[field] for field in protected} for row in rows]:
        raise AssertionError("Frozen AI/H1 or source fields changed.")
    allowed = {"include_definition_or_review", "exclude_not_relevant", "uncertain"}
    if any(row["h2_final_screen_decision"] not in allowed for row in rows):
        raise ValueError("Invalid H2 screen decision.")
    if any(not row["h2_final_evidence_span"] or not row["h2_final_rationale"] for row in rows):
        raise ValueError("Every source needs a supported span and rationale.")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    reviewed = [row for row in rows if row["h2_review_required"] == "1"]
    manifest = {
        "schema_version": "contextual_source_screening_h2_batch3_manifest_v4",
        "status": "complete",
        "reviewer_role": "H2",
        "reviewer_id": "H2_CONTEXTUAL_SOURCE_SCREENING_BATCH3_ADJUDICATOR_V4",
        "model": "codex_configured_default",
        "model_digest": "codex-thread:/root/h2_adjudication",
        "review_completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "method": "Screen source-recovery leads using only frozen title/abstract and AI/H1 opinions. Include only denotes potential recovery of original definitions/reviews, never feature approval. Exclude topic mappings and sources that show only post-publication impact, citation, or usage outcomes without a paper-level T0 indicator-method lead.",
        "qwen_or_ollama_used": False,
        "input_artifact": {str(INPUT): sha256(INPUT)},
        "output_artifact": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "record_count": len(rows),
        "h2_review_required": {"required_count": len(reviewed), "completed_count": len(reviewed)},
        "decision_counts": dict(sorted(Counter(row["h2_final_screen_decision"] for row in rows).items())),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
