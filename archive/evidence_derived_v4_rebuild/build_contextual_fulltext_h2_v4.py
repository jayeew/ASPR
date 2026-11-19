"""Create H2 final adjudications from the verified contextual full texts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_INPUT = ROOT / "outputs" / "contextual_fulltext_source_review_H2_v4.csv"
MENTION_INPUT = ROOT / "outputs" / "contextual_fulltext_indicator_mentions_H2_v4.csv"
SOURCE_OUTPUT = ROOT / "outputs" / "contextual_fulltext_source_review_H2_completed_v4.csv"
MENTION_OUTPUT = ROOT / "outputs" / "contextual_fulltext_indicator_mentions_H2_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_fulltext_h2_manifest_v4.json"


def sha256(path: Path) -> str:
    """Return one artifact's SHA-256 digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


SOURCE_DECISIONS = {
    "doi:10.1007/s10489-017-1105-y": (
        "review_discovery_only",
        "The verified text surveys bibliometric procedures, citation-network representations, and field-level measures. It is useful to locate original sources but does not itself apply an eligible focal-paper T0 formula.",
    ),
    "doi:10.1007/s10961-017-9637-1": (
        "review_discovery_only",
        "The verified text uses co-citation and bibliographic coupling for aggregate field mapping. It supplies methodological terminology and cited sources, not a scalar focal-paper T0 application.",
    ),
    "doi:10.1007/s11192-010-0202-z": (
        "no_relevant_indicator",
        "The verified text measures annual database coverage, publication growth, and country aggregates; it contains no focal-paper innovation, opportunity, or control indicator for extraction.",
    ),
    "doi:10.1007/s11192-016-2150-8": (
        "review_discovery_only",
        "This verified state-of-the-art report discusses bibliometric indicators and their uses. It is discovery context only and is not an original focal-paper T0 formula/application authority.",
    ),
    "doi:10.1007/s11192-017-2296-z": (
        "review_discovery_only",
        "The verified comparison concerns co-citation, bibliographic coupling, and hybrid clustering over document sets. It can direct methodology recovery but does not apply a focal-paper T0 indicator.",
    ),
    "doi:10.1007/s11192-022-04486-4": (
        "formula_or_application",
        "The methodology explicitly classifies each focal paper as local, national, or international from listed organization and country affiliations. The article's later citation-impact analyses are outcomes and are not extracted as a feature.",
    ),
    "doi:10.1007/s11192-024-05116-x": (
        "no_relevant_indicator",
        "The verified text operationalizes corpus-level document-to-person mention extraction in historical-science mapping. It does not define or apply an eligible focal-paper T0 innovation, opportunity, or control indicator.",
    ),
    "doi:10.1007/s40037-021-00695-4": (
        "review_discovery_only",
        "The verified pedagogical overview discusses h-index and journal indicators as bibliometrics terminology. It is useful for discovery but not sole original authority for a focal-paper T0 candidate.",
    ),
    "doi:10.1016/j.ejor.2015.04.002": (
        "review_discovery_only",
        "The verified paper reviews scientometric theory and practice and points to original metric sources. It does not itself authorize a new focal-paper formula/application.",
    ),
    "doi:10.1016/j.emj.2026.03.002": (
        "review_discovery_only",
        "The verified manuscript critically reviews assumptions and clustering practice in bibliometric systematic reviews. It is discovery context, not an explicit focal-paper T0 indicator application.",
    ),
    "doi:10.1371/journal.pone.0253129": (
        "review_discovery_only",
        "The verified systematic review synthesizes open-access citation-advantage studies and their confounders. It can guide original-study recovery but is not itself a single formula/application authority.",
    ),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one UTF-8 CSV with its header."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write one UTF-8 CSV reproducibly."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fill_sources() -> list[dict[str, str]]:
    """Fill H2 source dispositions while preserving all source-review inputs."""
    fields, rows = read_csv(SOURCE_INPUT)
    if len(rows) != 11:
        raise ValueError("Expected 11 verified full-text source rows.")
    protected = [field for field in fields if field not in {
        "h2_final_source_disposition", "h2_final_source_notes"
    }]
    before = [{field: row[field] for field in protected} for row in rows]
    for row in rows:
        try:
            decision, note = SOURCE_DECISIONS[row["record_key"]]
        except KeyError as error:
            raise ValueError(f"Missing H2 source decision: {row['record_key']}") from error
        row["h2_final_source_disposition"] = decision
        row["h2_final_source_notes"] = note
    if before != [{field: row[field] for field in protected} for row in rows]:
        raise AssertionError("Protected source-review fields changed.")
    write_csv(SOURCE_OUTPUT, fields, rows)
    return rows


def fill_mentions() -> list[dict[str, str]]:
    """Retain the one verified paper-collaboration candidate with full H2 fields."""
    fields, rows = read_csv(MENTION_INPUT)
    if len(rows) != 1:
        raise ValueError("Expected one verified full-text indicator candidate.")
    h2_fields = {
        "h2_decision", "raw_name_en", "canonical_name_en", "source_role", "formula_location",
        "evidence_span", "formula", "parameters", "required_data", "maximum_information_time",
        "scope_role", "requires_future", "extraction_notes",
    }
    protected = [field for field in fields if field not in h2_fields]
    before = [{field: row[field] for field in protected} for row in rows]
    row = rows[0]
    if row["record_key"] != "doi:10.1007/s11192-022-04486-4":
        raise ValueError("Unexpected full-text indicator source.")
    row.update({
        "h2_decision": "retain_as_candidate",
        "raw_name_en": "Collaboration type (local, national, international)",
        "canonical_name_en": "Paper collaboration geographic scope",
        "source_role": "original_application",
        "formula_location": "Methodology, collaboration-type definitions, p. 7521.",
        "evidence_span": "Local type Papers in this category must include only one organization and one country. National type Papers with national collaboration must include a number of organizations greater than one and include only a single country. International type Papers are classified under this category when two or more countries are collaborating.",
        "formula": "Categorical definition: local = one organization and one country; national = more than one organization and one country; international = two or more collaborating countries.",
        "parameters": "organization count; affiliated-country count",
        "required_data": "focal-paper identifier; focal-paper author-affiliation organization identifiers; focal-paper author-affiliation country identifiers",
        "maximum_information_time": "T0 (classification uses the focal paper's listed organizations and countries)",
        "scope_role": "t0_opportunity",
        "requires_future": "false",
        "extraction_notes": "Retained only as a candidate paper-level collaboration-context classification. The source's citation counts, h-index, highly cited paper measures, and other citation-impact analyses occur later and are outcomes, not T0 feature inputs. This retention is not final feature selection.",
    })
    if before != [{field: row[field] for field in protected} for row in rows]:
        raise AssertionError("Protected AI/H1 mention fields changed.")
    write_csv(MENTION_OUTPUT, fields, rows)
    return rows


def main() -> None:
    """Produce both H2 adjudications and a shared provenance manifest."""
    source_rows = fill_sources()
    mention_rows = fill_mentions()
    source_counts = Counter(row["h2_final_source_disposition"] for row in source_rows)
    mention_counts = Counter(row["h2_decision"] for row in mention_rows)
    manifest = {
        "schema_version": "contextual_fulltext_h2_manifest_v4",
        "status": "complete",
        "reviewer_role": "H2",
        "reviewer_id": "H2_VERIFIED_FULLTEXT_ADJUDICATOR_V4",
        "model": "codex_configured_default",
        "model_digest": "codex-thread:/root/h2_adjudication",
        "review_completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "method": "Adjudicate only verified-fulltext source utility and explicitly extracted candidate indicators. A formula/application disposition requires an explicit focal-paper T0 application. Citation impact and other post-publication outcomes are never imported as features.",
        "qwen_or_ollama_used": False,
        "input_artifacts": {
            str(SOURCE_INPUT): sha256(SOURCE_INPUT),
            str(MENTION_INPUT): sha256(MENTION_INPUT),
        },
        "output_artifacts": {
            str(SOURCE_OUTPUT): sha256(SOURCE_OUTPUT),
            str(MENTION_OUTPUT): sha256(MENTION_OUTPUT),
        },
        "source_record_count": len(source_rows),
        "source_disposition_counts": dict(sorted(source_counts.items())),
        "indicator_candidate_count": len(mention_rows),
        "indicator_decision_counts": dict(sorted(mention_counts.items())),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
