"""Lightweight output checks for Fig. 4new."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "outputs/fig04/new"


def main() -> int:
    required = [OUTPUT / f"figure_full.{suffix}" for suffix in ("png", "svg", "pdf")]
    required.extend(
        OUTPUT / "panels" / f"fig04_{letter}.{suffix}"
        for letter in "abcdef"
        for suffix in ("png", "svg", "pdf")
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Fig. 4new outputs: " + ", ".join(missing))
    audit = json.loads((OUTPUT / "audit_report.json").read_text(encoding="utf-8"))
    if audit["reviewer_alignment_label_rows"] != 6280:
        raise ValueError("published-review alignment labels are incomplete")
    if audit["claim_b_original_papers"] != 30 or audit["claim_c_tasks"] != 78:
        raise ValueError("Fig. 4new cohort counts changed")
    completion = pd.read_csv(OUTPUT / "data_20260829/claim_b_evidence_completion.csv")
    required = {
        "claim_id",
        "manuscript_evidence_keys",
        "prior_work_identifier",
        "prior_work_excerpt",
        "prior_work_location",
        "cutoff_verified",
        "relation_evidence_key",
        "relation_rationale",
        "relation_status",
        "residual_novelty_eligible",
    }
    missing_columns = sorted(required - set(completion.columns))
    if missing_columns:
        raise ValueError(f"Claim B evidence table misses columns: {missing_columns}")
    eligible = completion[completion["residual_novelty_eligible"]]
    if (
        eligible.empty
        or eligible[required - {"claim_id", "residual_novelty_eligible"}]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Claim B evaluable claims lack required evidence completion")
    assessments_path = OUTPUT / "data_20260829/claim_b_independent_assessments.csv"
    if assessments_path.is_file():
        assessments = pd.read_csv(assessments_path)
        if (
            not assessments.empty
            and not assessments["evidence_completion_eligible"].all()
        ):
            raise ValueError(
                "Residual-support metrics include non-evaluable Claim B claims"
            )
    print("Fig. 4new output checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
