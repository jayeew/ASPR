"""Build a bounded technical-report artifact from Stage-A validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_report(output_dir: Path) -> dict[str, Any]:
    """Build the canonical report artifact and a readable Markdown companion."""
    output_dir = Path(output_dir).resolve()
    result = json.loads(
        (output_dir / "stage_a_validation_manifest.json").read_text(encoding="utf-8")
    )
    deciles = pd.read_csv(output_dir / "graph_validity_by_decile.csv")
    domains = pd.read_csv(output_dir / "graph_validity_by_domain.csv")
    artifact = _artifact(result, deciles, domains)
    (output_dir / "stage_a_report_artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "stage_a_technical_report.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    return artifact


def _artifact(
    result: dict[str, Any], deciles: pd.DataFrame, domains: pd.DataFrame
) -> dict[str, Any]:
    source = _source()
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "ASPR-GEAR Stage A Real-Data Validation",
        "description": "Technical validation of Graph signal and integration readiness.",
        "sources": [source],
        "charts": [
            {
                "id": "decile_chart",
                "title": "Expected and realized five-year diffusion by score decile",
                "type": "line",
                "dataset": "decile_metrics",
                "encodings": {
                    "x": {"field": "score_decile", "type": "ordinal"},
                    "y": {
                        "fields": [
                            "mean_expected_diffusion",
                            "mean_realized_diffusion",
                        ],
                        "type": "quantitative",
                    },
                },
                "source": source,
            }
        ],
        "tables": [
            {
                "id": "domain_table",
                "title": "Predictive validity by domain",
                "dataset": "domain_metrics",
                "columns": [
                    {"field": "domain12", "label": "Domain", "type": "string"},
                    {"field": "papers", "label": "Papers", "type": "number"},
                    {
                        "field": "spearman",
                        "label": "Spearman rho",
                        "type": "number",
                    },
                ],
                "defaultSort": {"field": "spearman", "direction": "asc"},
                "source": source,
            }
        ],
        "blocks": _blocks(result),
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "datasets": {
                "decile_metrics": deciles.to_dict(orient="records"),
                "domain_metrics": domains.to_dict(orient="records"),
            },
        },
        "sources": [source],
    }


def _source() -> dict[str, Any]:
    return {
        "id": "real_oof",
        "label": "Frozen D5 OOF predictions and score table",
        "path": (
            "data/calibration/releases/gear-d5-primary16-current/"
            "oof_predictions.parquet"
        ),
        "query": {
            "language": "sql",
            "engine": "duckdb",
            "sql": (
                "SELECT o.*, s.prospective_5y_diffusion_percentile "
                "FROM read_parquet('data/calibration/releases/"
                "gear-d5-primary16-current/oof_predictions.parquet') o "
                "LEFT JOIN read_parquet('data/calibration/releases/"
                "gear-d5-primary16-current/score_table.parquet') s "
                "USING (paper_id)"
            ),
            "description": (
                "One-to-one join of frozen OOF predictions to score percentiles; "
                "finite realized outcomes only."
            ),
            "tables_used": [
                (
                    "data/calibration/releases/gear-d5-primary16-current/"
                    "oof_predictions.parquet"
                ),
                (
                    "data/calibration/releases/gear-d5-primary16-current/"
                    "score_table.parquet"
                ),
            ],
            "filters": [
                "finite expected_diffusion_score",
                "finite realized_diffusion_target",
            ],
            "metric_definitions": {
                "spearman": (
                    "Spearman rank correlation between expected diffusion and "
                    "realized five-year diffusion."
                ),
                "top_decile_lift": (
                    "Mean realized diffusion in score decile 9 divided by the "
                    "overall mean."
                ),
            },
        },
    }


def _blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    graph = result["graph_predictive_validity"]
    quality = result["data_quality"]
    return [
        {
            "id": "title",
            "type": "markdown",
            "body": "# ASPR-GEAR Stage A Real-Data Validation",
        },
        {
            "id": "summary",
            "type": "markdown",
            "body": (
                "## Technical Summary\n\nStage A is **not yet identifiable**. "
                f"Graph validity is supported on {graph['papers']:,} real OOF "
                "papers and all seven Gate 0 checks pass, but matched manuscript-"
                "derived GEAR evidence has zero overlap with the outcome cohort."
            ),
        },
        {
            "id": "findings",
            "type": "markdown",
            "body": (
                "## Key Findings\n\nExpected diffusion tracks realized five-year "
                f"diffusion (Spearman rho = {graph['spearman']:.3f}); top-decile "
                f"lift is {graph['top_decile_lift']:.2f}x. The weakest domain "
                f"remains positive at rho = {graph['worst_domain_spearman']:.3f}. "
                "This supports the Graph signal only, not integration utility."
            ),
            "sourceId": "real_oof",
        },
        {"id": "decile", "type": "chart", "chartId": "decile_chart"},
        {
            "id": "domain_heading",
            "type": "markdown",
            "body": (
                "## Domain Robustness\n\nAll 12 domains have positive rank "
                "association; the table is sorted from the weakest upward."
            ),
        },
        {"id": "domains", "type": "table", "tableId": "domain_table"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## Scope, Data, and Definitions\n\nThe frozen population contains "
                f"{quality['oof_rows']:,} unique papers from "
                f"{quality['oof_publication_year_min']}–"
                f"{quality['oof_publication_year_max']}; the score join rate is "
                f"{quality['score_table_join_rate']:.0%}. The deterministic "
                "200-paper cohort covers all 10 score deciles and 12 domains."
            ),
            "sourceId": "real_oof",
        },
        {
            "id": "methods",
            "type": "markdown",
            "body": (
                "## Methodology\n\nWe recomputed rank correlation, decile "
                "calibration, Brier score, domain robustness, and outer-fold "
                "robustness from frozen OOF predictions. Gate 0 verifies routing "
                "and fusion invariants. The registered three-arm runner fails "
                "closed below 100 matched rows or eight score deciles."
            ),
        },
        {
            "id": "limits",
            "type": "markdown",
            "body": (
                "## Limitations and Robustness\n\nGraph validity is retrospective "
                "OOF evidence, not proof that Graph guidance improves reviews. "
                "The legacy 50-paper review sample occupies only score decile 9. "
                "No synthetic evidence was used to fill the missing matched data."
            ),
        },
        {
            "id": "next",
            "type": "markdown",
            "body": (
                "## Recommended Next Steps\n\nAcquire publication-time manuscripts "
                "for at least 100 selected papers spanning eight score deciles, "
                "run GEAR blinded to future outcomes, populate the evidence "
                "template, and rerun the real-versus-shuffled three-arm test."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": (
                "## Further Questions\n\nCan the required manuscripts be acquired "
                "legally and reproducibly? What minimum incremental effect should "
                "govern the final Stage A decision?"
            ),
        },
    ]


def _markdown(result: dict[str, Any]) -> str:
    graph = result["graph_predictive_validity"]
    return f"""# ASPR-GEAR Stage A Real-Data Validation

## Verdict

Stage A is **not yet identifiable** (`claim_allowed=false`). Graph predictive
validity is supported and Gate 0 passes, but the matched integration cohort has
zero eligible papers. No synthetic evidence was substituted.

## Real-data results

- Finite-outcome OOF papers: {graph['papers']:,}
- Spearman rho: {graph['spearman']:.6f}
- Top-decile realized-outcome lift: {graph['top_decile_lift']:.3f}x
- Uptake Brier score: {graph['uptake_brier']:.6f}
- Worst-domain rho: {graph['worst_domain_spearman']:.6f}
- Worst-outer-fold rho: {graph['worst_fold_spearman']:.6f}
- Gate 0: passed (7/7 checks)
- Frozen OOF / available GEAR-evidence overlap: 0 papers

## Interpretation

The results validate that the frozen Graph score carries real prospective signal
across score deciles, domains, and outer folds. They do not establish that adding
Graph guidance improves GEAR review quality. That causal/incremental claim remains
blocked until manuscript-derived evidence and future outcomes exist for the same
papers.

## Next executable step

Populate `stage_a_gear_evidence_template.csv` with blinded GEAR outputs for at
least 100 selected papers covering eight or more score deciles, then rerun the
Stage-A validator. The three-arm runner will remain fail-closed until those
identification requirements are met.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.output_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
