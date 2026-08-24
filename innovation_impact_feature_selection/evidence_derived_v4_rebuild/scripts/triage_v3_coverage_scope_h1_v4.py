"""Generate the independent H1 blind v3-coverage scope-triage artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "v3_coverage_scope_triage_input_v4.csv"
OUTPUT = ROOT / "outputs" / "v3_coverage_scope_triage_H1_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "v3_coverage_scope_triage_H1_completed_v4.manifest.json"

SCOPE_EXCLUDE = {
    "EF0020", "EF0054", "EF0059", "EF0061", "EF0062", "EF0091", "EF0112",
    "EF0113", "EF0125", "EF0153", "EF0162", "EF0166", "EF0168", "EF0179",
    "EF0191", "EF0217", "EF0222", "EF0224", "EF0225", "EF0233", "EF0241",
    "EF0251", "EF0252", "EF0253", "EF0267", "EF0270", "EF0271", "EF0274",
    "EF0276", "EF0277", "EF0278", "EF0279", "EF0290", "EF0291", "EF0293",
    "EF0301", "EF0320", "EF0349", "EF0350", "EF0351", "EF0352", "EF0353",
    "EF0354", "EF0355", "EF0356", "EF0357", "EF0358", "EF0359", "EF0360",
    "EF0361", "EF0362", "EF0363", "EF0364", "EF0365", "EF0366", "EF0367",
    "EF0368", "EF0369", "EF0389", "EF0404", "EF0405", "EF0406", "EF0407",
    "EF0408", "EF0409", "EF0410", "EF0411", "EF0412",
}

NEEDS_EVIDENCE = {
    "EF0014", "EF0018", "EF0021", "EF0022", "EF0023", "EF0024", "EF0026",
    "EF0053", "EF0055", "EF0056", "EF0057", "EF0058", "EF0060", "EF0064",
    "EF0065", "EF0066", "EF0067", "EF0068", "EF0069", "EF0071", "EF0087",
    "EF0088", "EF0090", "EF0092", "EF0093", "EF0094", "EF0095", "EF0100",
    "EF0110", "EF0121", "EF0122", "EF0128", "EF0131", "EF0132", "EF0133",
    "EF0134", "EF0135", "EF0136", "EF0137", "EF0138", "EF0139", "EF0140",
    "EF0141", "EF0142", "EF0143", "EF0144", "EF0145", "EF0146", "EF0152",
    "EF0158", "EF0163", "EF0165", "EF0167", "EF0172", "EF0180", "EF0184",
    "EF0189", "EF0193", "EF0194", "EF0195", "EF0202", "EF0203", "EF0206",
    "EF0210", "EF0212", "EF0215", "EF0218", "EF0220", "EF0223", "EF0226",
    "EF0227", "EF0228", "EF0229", "EF0230", "EF0231", "EF0246", "EF0248",
    "EF0255", "EF0256", "EF0259", "EF0260", "EF0263", "EF0264", "EF0266",
    "EF0268", "EF0272", "EF0273", "EF0275", "EF0285", "EF0286", "EF0292",
    "EF0297", "EF0299", "EF0300", "EF0302", "EF0303", "EF0305", "EF0308",
    "EF0311", "EF0312", "EF0314", "EF0315", "EF0321", "EF0324", "EF0330",
    "EF0332", "EF0335", "EF0341", "EF0342", "EF0343", "EF0344", "EF0346",
    "EF0347", "EF0348", "EF0370", "EF0371", "EF0374", "EF0375", "EF0377",
    "EF0378", "EF0379", "EF0382", "EF0383", "EF0384", "EF0385", "EF0386",
    "EF0387", "EF0388", "EF0392", "EF0394", "EF0395", "EF0396", "EF0402",
    "EF0413", "EF0414", "EF0415", "EF0421", "EF0422", "EF0425", "EF0428",
    "EF0429", "EF0430", "EF0431",
}


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def role_for(label: str) -> str:
    """Assess broad scope from the label itself, without legacy-role reliance."""
    lowered = label.lower()
    if any(word in lowered for word in ("novelty", "originality", "creativity", "interdisciplinarity", "multidisciplinarity", "diversity", "knowledge-combination", "new-concept")):
        return "direct_innovation"
    if any(word in lowered for word in ("author", "authorship", "collaboration", "coauthor", "network", "journal", "funding", "access", "institution", "tenure", "career", "conference", "publisher", "orcid", "peer-review", "opportunity")):
        return "t0_opportunity"
    if any(word in lowered for word in ("country", "geograph", "language", "publication year", "document type", "field-year", "study design", "research site", "theoretical perspective", "clinical research", "sample size", "sampling location")):
        return "context_control"
    return "t0_substantive"


def search_terms(label: str) -> str:
    """Provide compact English source-recovery terms for non-excluded labels."""
    terms = [part.strip().lower() for part in label.replace("/", " ").replace("-", " ").split() if part.strip()]
    core = " ".join(terms[:4])
    return "; ".join((core, "scholarly article", "measurement"))


def source_need(label: str) -> str:
    """Select the least expansive allowed source-evidence request."""
    lowered = label.lower()
    if any(word in lowered for word in ("entropy", "rao", "div", "connectivity", "centrality", "mutual-information", "index", "score")):
        return "mathematical foundation"
    if any(word in lowered for word in ("validity", "quality", "reliability", "accuracy", "evaluation")):
        return "validation"
    return "original application"


def triage(row: dict[str, str]) -> dict[str, str]:
    """Fill only the next source-recovery action required by the brief."""
    feature_id = row["v3_feature_id"]
    label = row["canonical_name_en"]
    if feature_id in SCOPE_EXCLUDE:
        row.update(
            triage_decision="scope_exclude",
            scope_role_assessment="out_of_scope",
            rationale="Clear clinical/study-result or systematic-review procedural construct, not a paper-level v4 feature candidate.",
            minimum_source_evidence_needed="none_for_clear_scope_exclusion",
            search_terms_en="",
        )
        return row
    if feature_id in NEEDS_EVIDENCE:
        row.update(
            triage_decision="needs_source_evidence",
            scope_role_assessment="uncertain",
            rationale="Label alone does not establish a paper-level construct, timing, or whether it is a study-specific assessment.",
            minimum_source_evidence_needed=source_need(label),
            search_terms_en=search_terms(label),
        )
        return row
    role = role_for(label)
    row.update(
        triage_decision="recover_priority",
        scope_role_assessment=role,
        rationale="Label plausibly denotes a paper-level publication-time construct within the v4 scope; recover original English evidence before any later judgment.",
        minimum_source_evidence_needed=source_need(label),
        search_terms_en=search_terms(label),
    )
    return row


def main() -> None:
    """Write the H1 triage CSV and an integrity manifest."""
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = [triage(dict(row)) for row in reader]
    if len(rows) != 432:
        raise ValueError(f"Expected 432 rows, found {len(rows)}")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    decisions = Counter(row["triage_decision"] for row in rows)
    manifest: dict[str, Any] = {
        "schema_version": "v3_coverage_scope_triage_h1_manifest_v4",
        "run_id": "v3-coverage-scope-triage-h1-independent-v4-20260819",
        "reviewer_role": "H1",
        "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": "Independent label-only scope triage under V3_COVERAGE_SCOPE_TRIAGE_BRIEF_V4.md; no AI/H2 output, formulas, data mappings, or final-feature decisions used.",
        "input_artifacts": {str(INPUT): sha256(INPUT), str(ROOT / "V3_COVERAGE_SCOPE_TRIAGE_BRIEF_V4.md"): sha256(ROOT / "V3_COVERAGE_SCOPE_TRIAGE_BRIEF_V4.md")},
        "output_artifacts": {str(OUTPUT): sha256(OUTPUT)},
        "row_count": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "independence_guards": {"read_ai_or_h2_results": False, "used_qwen_or_ollama": False},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "manifest": str(MANIFEST), **manifest["decision_counts"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
