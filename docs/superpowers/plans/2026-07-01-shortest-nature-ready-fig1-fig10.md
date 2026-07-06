# Shortest Nature-Ready Fig1-Fig10 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current Fig1-Fig10 package into a defensible Nature submission evidence package centered on graph-perturbation innovation prediction, while demoting ASPR application figures to Extended Data unless their replacement gates pass.

**Architecture:** Treat Fig1/Fig2/Fig3/Fig4/Fig6 as main-claim blockers and Fig5/Fig7/Fig8/Fig9/Fig10 as optional or Extended Data unless their strong gates pass. Every figure must write machine-readable manifests, quality reports, source tables, and claim-ledger-compatible captions. The shortest path is to fix empirical evidence first, then visual and final-assembly language.

**Tech Stack:** Python 3, pandas, numpy, scipy/sklearn, networkx, matplotlib, OpenAlex/Semantic Scholar APIs, local embedding backend such as BGE-M3 or SPECTER2, unittest, Makefile.

---

## Current Evidence Snapshot From 2026-06-30 Rerun

- Fig1 passes its current main-figure gates: deterministic hybrid sampling is recorded, edge cap is no longer counted as uncontrolled truncation, and common 15-year horizon is consistent.
- Fig2 fails: `total_eligible_papers=5087 < 8000`, `relaxed_control_tier_ratio=0.338 > 0.25`, `reference_closure_coverage=0`.
- Fig3 fails: learned OOF Spearman is `0.293 < 0.45`; top-decile future top20 enrichment is `4.37x < 5x`; CRISPR, Graphene, and Perovskite are below the sample target; CRISPR and Graphene have fewer than 30 high/landmark cases.
- Fig4 fails: fixed sample is `47/50`; all cases use `lexical_fallback` and `local_fig4_manifest`; `soft_claim_recall=0`, `claim_evidence_coverage=0`, `covered_peer_aspects=0`, `missing_peer_point_rate=1.0`.
- Fig5 fails its forecast/backtest gate because `precision_at_10`, `ndcg_at_10`, and explicit baseline comparison columns are absent.
- Fig6 is pipeline-ready only: cached/proxy robustness passes, but `nature_strong_claim_ready=0` because full graph reruns are missing.
- Fig7 can only claim that Nature Portfolio has the top aggregate VCI point estimate; strict interval separation fails.
- Fig8 renders deterministically but needs a quality report/run manifest in the same contract as other figures.
- Fig9 is an assumed ASPR-Qwen placeholder and must remain Extended Data unless a real checkpoint output is saved.
- Fig10 is a pipeline audit: same-rubric generic baseline is `47/47` with `282` matched peer-review points, but true disabled-module reruns, blinded human preference, and checkpoint ASPR-Qwen are missing.

---

## File Structure And Ownership

### Cross-Figure Infrastructure

- Modify: `Makefile`
  - Add robust targets that do not force expensive Fig4 audit reruns when cached audit is valid.
  - Keep `figures-nature-check` as the final honest gate.
- Modify: `experiments/nature_ready_checks.py`
  - Read new Fig2/Fig3/Fig4/Fig6 strong-gate artifacts.
  - Keep Fig7-Fig10 Extended Data failures nonblocking for the main claim.
- Modify: `experiments/kg_perturbation_final_assembly/build_final_assembly.py`
  - Use claim ledger text only; never hard-code performance phrasing.
- Test: `tests/test_nature_ready_claims.py`
- Test: `tests/test_reproducibility_manifest.py`
- Test: `tests/test_no_leakage_features.py`

### Main Evidence Figures

- Modify: `experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py`
- Modify: `experiments/kg_perturbation_fig2/fig2_empirical_panels.py`
- Create: `experiments/kg_perturbation_fig2/build_fig2_reference_closure.py`
- Create: `experiments/kg_perturbation_fig2/build_fig2_strong_inputs.py`
- Modify: `experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py`
- Create: `experiments/kg_perturbation_fig3/build_fig3_topup_inputs.py`
- Create: `experiments/kg_perturbation_fig3/build_fig3_text_llm_baselines.py`
- Modify: `experiments/kg_perturbation_fig4/main_fig4.py`
- Create: `experiments/kg_perturbation_fig4/build_fig4_retrieval_audit.py`
- Create: `experiments/kg_perturbation_fig6/build_fig6_full_rerun.py`
- Modify: `experiments/kg_perturbation_fig6/build_fig6_robustness.py`

### Optional / Extended Data Figures

- Modify: `experiments/kg_perturbation_fig5/fig5_forecast_outcomes.py`
- Modify: `experiments/kg_perturbation_fig7/build_fig7_venue_contribution.py`
- Modify: `experiments/kg_perturbation_fig8/render_fig8.py`
- Modify: `experiments/kg_perturbation_fig9/build_fig9_case.py`
- Modify: `experiments/kg_perturbation_fig10/build_fig10_generic_baseline.py`
- Modify: `experiments/kg_perturbation_fig10/build_fig10_ablation.py`

---

## Main Text Figure Strategy

Use this as the target main manuscript package:

1. Fig1: graph-perturbation construction and no-leakage standardized trajectories.
2. Fig2: empirical indicator validation after sample/control/reference-closure repair.
3. Fig3: prediction model with no-leakage baselines and holdouts.
4. Fig4: external peer-review novelty/significance validation.
5. Fig6: full graph-rerun robustness.
6. Optional Fig5 if forecast backtest beats baselines; otherwise Fig8 architecture overview.

Default Extended Data:

- Fig5 if backtest remains weak.
- Fig7 unless strict interval separation passes.
- Fig8 if not used as main overview.
- Fig9 unless real ASPR-Qwen checkpoint exists.
- Fig10 unless true reruns plus blinded human preference pass.

---

## Task 1: Make The Rerun Entrypoints Reliable

**Files:**
- Modify: `Makefile`
- Modify: `experiments/kg_perturbation_fig4/main_fig4.py`
- Modify: `experiments/kg_perturbation_fig10/build_fig10_generic_baseline.py`
- Test: `tests/test_reproducibility_manifest.py`

- [ ] **Step 1: Add a failing test for cached Fig4 audit reuse and bounded Fig10 baseline**

Add this test to `tests/test_reproducibility_manifest.py`:

```python
def test_makefile_has_rerun_safe_targets() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")
    assert "figures-main-nature:" in text
    assert "--reuse-audit" in text
    assert "$(FIG10_MAX_CASES)" in text
    assert "figures-extended:" in text
```

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
python3 -m unittest tests.test_reproducibility_manifest -v
```

Expected before implementation: failure mentioning `--reuse-audit` or `FIG10_MAX_CASES`.

- [ ] **Step 3: Implement Fig4 audit reuse**

In `experiments/kg_perturbation_fig4/main_fig4.py`, add parser flags:

```python
parser.add_argument("--reuse-audit", action="store_true")
parser.add_argument("--force-audit", action="store_true")
```

Then change the audit block:

```python
audit_path = args.output_dir / "fig4_input_audit.csv"
if "audit" in stages:
    if args.reuse_audit and audit_path.exists() and not args.force_audit:
        progress_log(f"Reusing Fig.4 audit table at {audit_path}.", args.quiet)
    else:
        audit_markdown_inputs(
            args.markdown_root,
            args.output_dir,
            journal_scope=args.journal_scope,
            quiet=args.quiet,
            audit_max_records=args.audit_max_records if args.audit_max_records > 0 else None,
        )
```

- [ ] **Step 4: Add bounded Fig10 model settings to Makefile**

In `Makefile`, add:

```make
FIG10_MAX_CASES ?= 0
FIG10_TIMEOUT ?= 120
```

Change the Fig10 generic baseline command to:

```make
$(PYTHON) experiments/kg_perturbation_fig10/build_fig10_generic_baseline.py \
	--fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
	--out-dir outputs/kg_perturbation_fig10 \
	--model-name $(FIG10_MODEL) --timeout $(FIG10_TIMEOUT) --max-cases $(FIG10_MAX_CASES)
```

- [ ] **Step 5: Run verification**

Run:

```bash
python3 -m unittest tests.test_reproducibility_manifest -v
```

Expected: pass.

---

## Task 2: Fig1 Finalize As Main Method Figure

**Files:**
- Modify: `experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py`
- Modify: `experiments/nature_ready_checks.py`
- Test: `tests/test_fig1_sampling_horizon.py`

**Experiment:** Confirm that the main figure uses standardized perturbation trajectories, not cross-domain raw density comparisons.

- [ ] **Step 1: Add a failing caption/claim test**

Extend `tests/test_fig1_sampling_horizon.py`:

```python
def test_fig1_caption_forbids_raw_density_superiority() -> None:
    from experiments.nature_ready_checks import build_claim_ledger

    ledger = build_claim_ledger()
    fig1 = [row for row in ledger if row["figure"] == "Fig.1"][0]
    assert "standardized structural trajectories" in fig1["allowed_claim"]
    assert "raw density superiority" in fig1["forbidden_claim"]
```

- [ ] **Step 2: Update claim ledger**

In `experiments/nature_ready_checks.py`, set the Fig1 row:

```python
"allowed_claim": "Fig.1 defines graph perturbation with deterministic edge sampling and standardized structural trajectories.",
"forbidden_claim": "Do not compare domains by raw density superiority or treat sampling_target_edges as an uncontrolled edge cap.",
```

- [ ] **Step 3: Rebuild Fig1**

Run:

```bash
python3 experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py \
  --config experiments/kg_perturbation_fig1/configs/v6a_display_crispr.yaml \
           experiments/kg_perturbation_fig1/configs/v6a_display_graphene.yaml \
           experiments/kg_perturbation_fig1/configs/v6a_display_ipsc.yaml \
           experiments/kg_perturbation_fig1/configs/v6a_display_exoplanets.yaml \
  --out-dir outputs/redraw_v6a_best_fig1 \
  --corpus-dir data/knowledge_corpus/v2_publication_v6a_locked_candidate
```

- [ ] **Step 4: Acceptance criteria**

Verify:

```bash
python3 -m unittest tests.test_fig1_sampling_horizon -v
python3 -m experiments.nature_ready_checks --out-dir outputs/kg_perturbation_final_assembly || true
```

Expected Fig1 criteria:

- `edge_cap_not_hit_all_domains=1`
- `final_cumulative_horizon_consistent=1`
- `edge_sampling_manifest_present=1`
- Captions only describe standardized trajectories.

---

## Task 3: Fig2 Strong Empirical Indicator Validation

**Files:**
- Create: `experiments/kg_perturbation_fig2/build_fig2_reference_closure.py`
- Create: `experiments/kg_perturbation_fig2/build_fig2_strong_inputs.py`
- Modify: `experiments/kg_perturbation_fig2/fig2_empirical_panels.py`
- Test: `tests/test_fig2_reference_closure.py`

**Experiments:**
- Fig2-A eligible corpus top-up.
- Fig2-B reference closure measurement.
- Fig2-C matched-control tier repair.

- [ ] **Step 1: Add failing reference-closure test**

Create `tests/test_fig2_reference_closure.py`:

```python
from __future__ import annotations

import unittest
import pandas as pd

from experiments.kg_perturbation_fig2.fig2_empirical_panels import build_fig2_quality_gates


class Fig2ReferenceClosureTests(unittest.TestCase):
    def test_fig2_strong_gates_require_eligible_controls_and_closure(self) -> None:
        report = build_fig2_quality_gates(
            n_domains=10,
            total_eligible_papers=8000,
            active_future_outcomes=["a", "b", "c", "d", "e"],
            relaxed_control_tier_ratio=0.24,
            reference_closure_measured_all_domains=True,
            min_reference_closure_coverage=0.81,
            significant_expected_links=4,
            mechanism_composite_partial_spearman=0.21,
        )
        self.assertTrue(report["overall_pass"])
        self.assertEqual(1, report["checks"]["reference_closure_coverage_min80pct"])
```

- [ ] **Step 2: Implement `build_fig2_reference_closure.py`**

Create a CLI that reads eligible works and materializes OpenAlex references:

```python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def build_reference_closure(works: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    reports = []
    for domain, group in works.groupby("domain"):
        materialized = 0
        total = 0
        for _, row in group.iterrows():
            refs = str(row.get("referenced_works", "") or "").split(";")
            refs = [ref for ref in refs if ref]
            total += len(refs)
            for ref in refs:
                rows.append({"domain": domain, "paper_id": row["paper_id"], "referenced_work_id": ref})
                materialized += 1
        reports.append(
            {
                "domain": domain,
                "eligible_papers": int(len(group)),
                "referenced_works_count": int(total),
                "materialized_reference_count": int(materialized),
                "coverage_materialized": float(materialized / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(reports)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--works", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    works = pd.read_csv(args.works)
    closure, report = build_reference_closure(works)
    closure.to_csv(args.out_dir / "reference_closure_table.csv", index=False)
    report.to_csv(args.out_dir / "fig2_reference_closure_report.csv", index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Build strong Fig2 inputs**

Create `experiments/kg_perturbation_fig2/build_fig2_strong_inputs.py` with these controls:

```python
MATCH_COLUMNS_EXACT = ["domain", "publication_year", "article_type", "reference_count_decile", "venue_family"]
MATCH_COLUMNS_RELAXED = ["domain", "publication_year", "article_type", "reference_count_decile"]
```

The script must write:

- `outputs/redraw_v6a_best_fig2/fig2_eligible_papers.csv`
- `outputs/redraw_v6a_best_fig2/reference_closure_table.csv`
- `outputs/redraw_v6a_best_fig2/fig2_reference_closure_report.csv`
- `outputs/redraw_v6a_best_fig2/fig2_control_tier_audit.csv`

- [ ] **Step 4: Add Fig2 gate calculation**

In `fig2_empirical_panels.py`, make `build_fig2_quality_gates` accept:

```python
total_eligible_papers: int
relaxed_control_tier_ratio: float
reference_closure_measured_all_domains: bool
min_reference_closure_coverage: float
```

Quality checks must be:

```python
"total_eligible_papers_min8000": int(total_eligible_papers >= 8000)
"relaxed_control_tier_ratio_max25pct": int(relaxed_control_tier_ratio <= 0.25)
"reference_closure_coverage_min80pct": int(reference_closure_measured_all_domains and min_reference_closure_coverage >= 0.8)
```

- [ ] **Step 5: Run Fig2 experiments**

Run:

```bash
python3 experiments/kg_perturbation_fig2/build_fig2_strong_inputs.py \
  --source data/knowledge_corpus/v2_publication_v6a_locked_candidate/views/fig2 \
  --out-dir outputs/redraw_v6a_best_fig2/fig2_strong_input \
  --pre-cutoff-max-year 2018 \
  --future-window-start 2019 \
  --future-window-end 2025 \
  --min-total-eligible 8000

python3 experiments/kg_perturbation_fig2/fig2_empirical_panels.py \
  --data-dir outputs/redraw_v6a_best_fig2/fig2_strong_input \
  --out-dir outputs/redraw_v6a_best_fig2 \
  --evidence-mode strong \
  --panel all --export-tables --quiet
```

- [ ] **Step 6: Acceptance criteria**

Expected in `outputs/redraw_v6a_best_fig2/fig2_quality_gates.json`:

- `overall_pass=true`
- `total_eligible_papers >= 8000`
- `relaxed_control_tier_ratio <= 0.25`
- `min_reference_closure_coverage >= 0.8`
- Caption can say "multi-domain empirical validation".

---

## Task 4: Fig3 Prediction Model, Holdouts, And Baselines

**Files:**
- Create: `experiments/kg_perturbation_fig3/build_fig3_topup_inputs.py`
- Create: `experiments/kg_perturbation_fig3/build_fig3_text_llm_baselines.py`
- Modify: `experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py`
- Test: `tests/test_fig3_holdout_baselines.py`
- Test: `tests/test_no_leakage_features.py`

**Experiments:**
- Fig3-A low-domain top-up.
- Fig3-B temporal holdout.
- Fig3-C leave-domain-out.
- Fig3-D baseline comparison.
- Fig3-E no-leakage audit.

- [ ] **Step 1: Add failing holdout/baseline test**

Create `tests/test_fig3_holdout_baselines.py`:

```python
from __future__ import annotations

import unittest
import pandas as pd

from experiments.kg_perturbation_fig3.fig3_empirical_weight_learning import build_fig3_quality_gates


class Fig3HoldoutBaselineTests(unittest.TestCase):
    def test_fig3_quality_requires_prediction_strength_and_baselines(self) -> None:
        baselines = pd.DataFrame(
            {
                "model": ["equal_weights", "best_single_indicator", "citation_only", "venue_only", "text_embedding_only", "generic_llm_title_abstract", "learned_weight_oof"],
                "oof_spearman": [0.20, 0.24, 0.05, 0.06, 0.10, 0.08, 0.46],
            }
        )
        report = build_fig3_quality_gates(
            learned_oof_spearman=0.46,
            top_decile_future_top20_enrichment=5.1,
            min_papers_per_domain=350,
            min_landmark_or_high_cases_per_domain=30,
            temporal_holdout_spearman_ci_low=0.05,
            leave_domain_out_spearman_ci_low=0.04,
            baseline_comparison=baselines,
        )
        self.assertTrue(report["overall_pass"])
```

- [ ] **Step 2: Top up low-sample domains**

Create `build_fig3_topup_inputs.py` to enforce:

```python
DOMAIN_MIN_PAPERS = {
    "crispr": 350,
    "graphene_2d_materials": 350,
    "perovskite_solar_cells": 350,
}
MIN_LANDMARK_OR_HIGH_CASES = 30
```

The script must write:

- `outputs/redraw_v6a_best_fig3/fig3_input/multi_domain/works.csv`
- `outputs/redraw_v6a_best_fig3/fig3_input/multi_domain/citations.csv`
- `outputs/redraw_v6a_best_fig3/fig3_input/multi_domain/fig3_domain_topup_audit.csv`

- [ ] **Step 3: Implement temporal holdout**

In `fig3_empirical_weight_learning.py`, add:

```python
TEMPORAL_TRAIN_MAX_YEAR = 2012
TEMPORAL_VALIDATION_START_YEAR = 2013
TEMPORAL_VALIDATION_END_YEAR = 2018
FUTURE_OUTCOME_START_YEAR = 2019
FUTURE_OUTCOME_END_YEAR = 2025
```

Write:

- `fig3_temporal_holdout.csv`
- `fig3_temporal_holdout_bootstrap.csv`

The gate passes only when bootstrap CI low is above zero.

- [ ] **Step 4: Implement leave-domain-out**

Add a loop:

```python
for held_out_domain in sorted(frame["domain"].unique()):
    train = frame[frame["domain"] != held_out_domain]
    test = frame[frame["domain"] == held_out_domain]
```

Write:

- `fig3_leave_domain_out.csv`
- `fig3_leave_domain_out_bootstrap.csv`

The gate passes only when aggregate CI low is above zero and every held-out domain has nonnegative point estimate.

- [ ] **Step 5: Add missing baselines**

Create `build_fig3_text_llm_baselines.py` with outputs:

- `fig3_text_embedding_baseline.csv`
- `fig3_generic_llm_title_abstract_baseline.csv`
- `fig3_venue_only_baseline.csv`
- `fig3_reference_count_only_baseline.csv`
- `fig3_citation_only_baseline.csv`

Merge these into `fig3_baseline_comparison.csv`.

- [ ] **Step 6: Update Fig3 strong gates**

Fig3 passes only when:

```python
learned_oof_spearman >= 0.45
top_decile_future_top20_enrichment >= 5.0
min_papers_per_domain >= 350
min_landmark_or_high_cases_per_domain >= 30
temporal_holdout_spearman_ci_low > 0
leave_domain_out_spearman_ci_low > 0
learned_score beats equal_weights, best_single_indicator, citation_only, venue_only, text_embedding_only, generic_llm_title_abstract
```

- [ ] **Step 7: Run Fig3**

Run:

```bash
python3 experiments/kg_perturbation_fig3/build_fig3_topup_inputs.py \
  --source data/knowledge_corpus/v2_publication_v6a_locked_candidate/views/fig3 \
  --out-dir outputs/redraw_v6a_best_fig3/fig3_input/multi_domain

python3 experiments/kg_perturbation_fig3/build_fig3_text_llm_baselines.py \
  --fig3-input-dir outputs/redraw_v6a_best_fig3/fig3_input/multi_domain \
  --out-dir outputs/redraw_v6a_best_fig3/multi_domain

python3 experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py \
  --data-dir outputs/redraw_v6a_best_fig3/fig3_input \
  --out-dir outputs/redraw_v6a_best_fig3 \
  --run-mode multi_domain \
  --domains crispr exoplanets gamma_ray_bursts_and_supernovae genetics_aging_and_longevity_in_model_organisms graphene_2d_materials ipsc_reprogramming microbiome_metagenomics perovskite_solar_cells topological_insulators ubiquitin_and_proteasome_pathways \
  --panel all --export-tables --diagnostics --n-weight-samples 5000 --formats png svg --quiet
```

- [ ] **Step 8: Acceptance criteria**

Expected:

- `learned_oof_spearman >= 0.45`
- `top_vs_bottom_score_decile_rgpm_top20_enrichment >= 5.0`
- CRISPR, Graphene, Perovskite each `>=350` papers.
- Every domain has `>=30` landmark/high cases.
- `tests.test_no_leakage_features` passes.

---

## Task 5: Fig4 External Peer-Review Validation

**Files:**
- Modify: `experiments/kg_perturbation_fig4/main_fig4.py`
- Create: `experiments/kg_perturbation_fig4/build_fig4_retrieval_audit.py`
- Test: `tests/test_fig4_external_validation.py`

**Experiments:**
- Fig4-A freeze exactly 50 evaluable cases.
- Fig4-B real prior-art retrieval.
- Fig4-C real embedding backend.
- Fig4-D semantic matcher calibration.
- Fig4-E Fig3 score versus peer novelty/significance.

- [ ] **Step 1: Fix the 47/50 sample blocker**

Add a screen freeze stage that writes:

- `fig4_candidate_screen.csv`
- `fig4_fixed_sample_manifest.csv`

The selection rule must be:

```python
screen_pass == True
has_peer_review_text == True
has_title == True
has_abstract_or_body_text == True
```

If fewer than 50 cases pass, fail before writing `fig4_manifest.csv`.

- [ ] **Step 2: Add top-up candidates**

In `main_fig4.py`, expand candidate scope in this order:

1. Existing Nature Communications markdown.
2. Other Nature Portfolio markdown under `/mnt/d/aspr_nature_markdown`.
3. Additional local markdown directories passed through `--markdown-root`.

Write `fig4_candidate_screen.csv` with columns:

```python
["paper_id", "journal", "year", "has_peer_review_text", "has_abstract_or_body_text", "screen_pass", "exclusion_reason"]
```

- [ ] **Step 3: Enforce real retrieval**

Create `build_fig4_retrieval_audit.py` that writes one JSON file per paper:

```python
{
  "paper_id": "s41467-023-39516-z",
  "retrieval_provider": "openalex",
  "query": "...",
  "top_k": 10,
  "retrieved": [
    {"rank": 1, "title": "...", "abstract": "...", "year": 2020, "score": 0.82, "source": "openalex"}
  ],
  "excluded": [
    {"title": "...", "reason": "post_publication_year"}
  ]
}
```

Fail the strong gate if any case uses:

- `local_fig4_manifest`
- `local_fallback`

- [ ] **Step 4: Enforce real embeddings**

In `main_fig4.py`, add:

```python
parser.add_argument("--embedding-backend", choices=["bge-m3", "specter2"], default="bge-m3")
```

Fail the strong gate if:

```python
embedding_backend == "lexical_fallback"
```

- [ ] **Step 5: Calibrate semantic matcher on 20 cases**

Write:

- `fig4_matcher_calibration_set.csv`
- `fig4_matcher_thresholds.json`
- `fig4_case_level_examples.json`
- `fig4_failure_examples.json`

Threshold acceptance:

```python
soft_claim_recall > 0
claim_evidence_coverage > 0
covered_peer_aspects > 0
missing_peer_point_rate < 1.0
```

- [ ] **Step 6: External validation model**

Write:

- `fig4_peer_external_validation.csv`
- `fig4_peer_external_validation_bootstrap.csv`

Metrics:

```python
spearman(fig3_score, peer_novelty) > 0 with bootstrap CI low > 0
spearman(fig3_score, peer_significance) > 0 with bootstrap CI low > 0
```

- [ ] **Step 7: Run Fig4**

Run:

```bash
python3 experiments/kg_perturbation_fig4/main_fig4.py \
  --markdown-root /mnt/d/aspr_nature_markdown \
  --output-dir outputs/kg_perturbation_fig4_full50 \
  --sample-size 50 --journal-scope all \
  --retrieval-provider openalex \
  --embedding-backend bge-m3 \
  --judge-backend heuristic \
  --reuse-audit \
  --require-fixed-sample \
  --forbid-lightweight \
  --forbid-local-retrieval \
  --forbid-lexical-fallback \
  --quiet
```

- [ ] **Step 8: Acceptance criteria**

Expected:

- `fig4_manifest.csv` has exactly 50 rows.
- `embedding_backend != lexical_fallback`.
- `retrieval_source != local_fig4_manifest`.
- `soft_claim_recall > 0`.
- `claim_evidence_coverage > 0`.
- `covered_peer_aspects > 0`.
- `missing_peer_point_rate < 1.0`.
- Fig3 score versus peer novelty/significance bootstrap CI low is above 0.
- Caption does not contain "human-like peer-review performance".

---

## Task 6: Fig5 Forecast Backtest Or Demotion

**Files:**
- Modify: `experiments/kg_perturbation_fig5/fig5_forecast_outcomes.py`
- Test: `tests/test_fig5_forecast_backtest.py`

**Experiments:**
- Fig5-A retrospective cutoffs.
- Fig5-B ranking metrics.
- Fig5-C baseline comparison.
- Fig5-D failure cases.

- [ ] **Step 1: Add precision and NDCG columns**

In `build_backtest`, write these columns:

```python
"precision_at_10"
"ndcg_at_10"
"baseline_method"
"baseline_precision_at_10"
"baseline_ndcg_at_10"
"delta_precision_at_10"
"delta_ndcg_at_10"
```

- [ ] **Step 2: Compute NDCG**

Add:

```python
def ndcg_at_k(relevance: list[float], k: int) -> float:
    gains = relevance[:k]
    dcg = sum((2.0 ** rel - 1.0) / math.log2(idx + 2.0) for idx, rel in enumerate(gains))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(idx + 2.0) for idx, rel in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0
```

- [ ] **Step 3: Run cutoffs**

Run:

```bash
python3 -m experiments.kg_perturbation_fig5.fig5_forecast_outcomes \
  --fig3-run-dir outputs/redraw_v6a_best_fig3/multi_domain \
  --fig3-input-dir outputs/redraw_v6a_best_fig3/fig3_input/multi_domain \
  --out-dir outputs/kg_perturbation_fig5 \
  --backtest-windows 2010:2015 2015:2020 2020:2025 \
  --formats png svg --quiet
```

- [ ] **Step 4: Main/Extended decision**

Promote Fig5 to main only if:

```python
mean(delta_precision_at_10) > 0
mean(delta_ndcg_at_10) > 0
bootstrap_ci_low(delta_ndcg_at_10) > 0
```

Otherwise set claim ledger:

```python
main_or_extended_data = "extended"
allowed_claim = "Retrospective forecast audit with visible failure cases."
forbidden_claim = "Do not claim robust forecasting performance."
```

---

## Task 7: Fig6 Full Graph-Rerun Robustness

**Files:**
- Create: `experiments/kg_perturbation_fig6/build_fig6_full_rerun.py`
- Modify: `experiments/kg_perturbation_fig6/build_fig6_robustness.py`
- Test: `tests/test_fig6_full_rerun.py`
- Test: `tests/test_fig6_robustness.py`

**Experiments:**
- Fig6-A OpenAlex versus Semantic Scholar.
- Fig6-B reference closure on/off.
- Fig6-C five sampling seeds.
- Fig6-D graph construction variants.
- Fig6-E cutoff perturbation.

- [ ] **Step 1: Create rerun manifest schema**

`fig6_full_rerun_manifest.csv` must contain:

```python
[
  "rerun_id",
  "source",
  "reference_closure",
  "edge_sampling_seed",
  "graph_construction",
  "cutoff_year_delta",
  "metadata_fetch_status",
  "graph_build_status",
  "indicator_status",
  "input_hash",
]
```

- [ ] **Step 2: Create stability tables**

`fig6_indicator_stability.csv`:

```python
["rerun_id", "metric", "baseline_mean", "rerun_mean", "delta", "direction_preserved"]
```

`fig6_rank_stability.csv`:

```python
["rerun_id", "rank_spearman", "top_decile_jaccard", "learned_score_direction_preserved"]
```

- [ ] **Step 3: Run robustness grid**

Minimum grid:

```python
sources = ["openalex", "semantic_scholar"]
reference_closure = ["on", "off"]
seeds = [20260630, 20260631, 20260632, 20260633, 20260634]
graph_constructions = ["direct_only", "direct_plus_bc", "direct_plus_bc_cocitation"]
cutoff_year_deltas = [-1, 0, 1]
```

For the shortest acceptable run, use all five seeds and at least:

```python
("openalex", "on", "direct_plus_bc_cocitation", 0)
("openalex", "off", "direct_plus_bc_cocitation", 0)
("semantic_scholar", "on", "direct_plus_bc_cocitation", 0)
("openalex", "on", "direct_only", 0)
("openalex", "on", "direct_plus_bc", 0)
("openalex", "on", "direct_plus_bc_cocitation", -1)
("openalex", "on", "direct_plus_bc_cocitation", 1)
```

- [ ] **Step 4: Update Fig6 quality report**

Fig6 strong gate passes only when:

```python
full_graph_rerun_artifacts_present == 1
min(rank_spearman) >= 0.8
all(learned_score_direction_preserved == 1)
n_reruns >= 7
```

- [ ] **Step 5: Run**

Run:

```bash
python3 experiments/kg_perturbation_fig6/build_fig6_full_rerun.py \
  --fig1-config-dir experiments/kg_perturbation_fig1/configs \
  --fig3-run-dir outputs/redraw_v6a_best_fig3/multi_domain \
  --out-dir outputs/kg_perturbation_fig6/full_rerun \
  --seeds 20260630 20260631 20260632 20260633 20260634

python3 experiments/kg_perturbation_fig6/build_fig6_robustness.py
```

- [ ] **Step 6: Acceptance criteria**

Expected in `outputs/kg_perturbation_fig6/figure_quality_report.json`:

- `nature_strong_claim_ready=1`
- `full_graph_rerun_artifacts_present=1`
- `rank_stability_ge_0_8=1`
- `learned_score_direction_preserved=1`

---

## Task 8: Fig7 Venue Claim Sensitivity

**Files:**
- Modify: `experiments/kg_perturbation_fig7/build_fig7_venue_contribution.py`
- Test: `tests/test_fig7_venue_contribution.py`

**Experiments:**
- Fig7-A article-only.
- Fig7-B review-excluded.
- Fig7-C field-year dense cells only.
- Fig7-D reference-count matched.
- Fig7-E team-size/open-access adjusted.
- Fig7-F per-paper VCI.

- [ ] **Step 1: Keep Fig7 Extended Data by default**

In claim ledger:

```python
main_or_extended_data = "extended"
allowed_claim = "Nature Portfolio has the top aggregate VCI point estimate in this corpus."
forbidden_claim = "Do not claim dominance, superiority, or significant outperformance unless strict interval separation passes."
```

- [ ] **Step 2: Add sensitivity output**

Write `fig7_sensitivity_summary.csv` with:

```python
["sensitivity", "nature_rank", "nature_vci", "runner_up", "strict_interval_separation", "pairwise_ci_low", "pairwise_ci_high"]
```

- [ ] **Step 3: Acceptance criteria**

Fig7 can move to main only if:

```python
strict_interval_separation == 1
pairwise_ci_low > 0
all sensitivity rows keep Nature rank == 1
```

Otherwise it remains Extended Data.

---

## Task 9: Fig8 Architecture Overview Contract

**Files:**
- Modify: `experiments/kg_perturbation_fig8/render_fig8.py`
- Test: `tests/test_fig8_renderer.py`

**Experiment:** Deterministic architecture figure with no performance implication.

- [ ] **Step 1: Write quality report**

At the end of `render_fig8.py`, write `figure_quality_report.json`:

```python
{
  "figure": "fig8",
  "overall_pass": true,
  "status_label": "non_performance_architecture_overview",
  "quality_gates": {
    "checks": {
      "renderer_source_controlled": 1,
      "no_generative_image_dependency": 1,
      "two_lanes_present": 1,
      "performance_claim_absent": 1
    }
  }
}
```

- [ ] **Step 2: Ensure two lanes**

The diagram must visibly name:

- `graph-perturbation innovation prediction`
- `ASPR application/review assistant`

- [ ] **Step 3: Acceptance criteria**

Run:

```bash
python3 -m experiments.kg_perturbation_fig8.render_fig8 --out-dir outputs/kg_perturbation_fig8
python3 -m unittest tests.test_fig8_renderer -v
```

Expected: renderer produces PNG/SVG/PDF plus quality report.

---

## Task 10: Fig9 ASPR-Qwen Checkpoint Boundary

**Files:**
- Modify: `experiments/kg_perturbation_fig9/build_fig9_case.py`
- Test: `tests/test_fig9_checkpoint_boundary.py`

**Experiments:**
- Fig9-A real checkpoint case if checkpoint exists.
- Fig9-B prototype storyboard if checkpoint does not exist.

- [ ] **Step 1: Require checkpoint metadata for strong use**

`fig9_aspr_qwen_output.json` must contain:

```python
{
  "checkpoint_invoked": true,
  "model_hash": "...",
  "training_config": {...},
  "data_version": "...",
  "prompt": "...",
  "decoding_config": {...},
  "seed": 20260630,
  "runtime": {...}
}
```

- [ ] **Step 2: Add checkpoint command**

Support:

```bash
python3 experiments/kg_perturbation_fig9/build_fig9_case.py \
  --markdown-root /mnt/d/aspr_nature_markdown \
  --output-dir outputs/kg_perturbation_fig9 \
  --checkpoint-path "$ASPR_QWEN_CHECKPOINT" \
  --checkpoint-mode real
```

- [ ] **Step 3: Enforce demotion when checkpoint is missing**

If `--checkpoint-mode real` is not used or checkpoint metadata is incomplete:

```python
main_or_extended_data = "extended"
allowed_claim = "Prototype case storyboard."
forbidden_claim = "Do not claim ASPR-Qwen checkpoint performance."
```

- [ ] **Step 4: Acceptance criteria**

Fig9 can support a performance-adjacent claim only when:

- `aspr_qwen_boundary != assumed placeholder`
- checkpoint metadata fields are complete.
- model hash and prompt are saved.
- output can be regenerated with the same seed.

---

## Task 11: Fig10 True Ablation And Human Preference

**Files:**
- Modify: `experiments/kg_perturbation_fig10/build_fig10_generic_baseline.py`
- Modify: `experiments/kg_perturbation_fig10/build_fig10_ablation.py`
- Test: `tests/test_fig10_true_ablation_human_preference.py`
- Test: `tests/test_fig10_ablation.py`

**Experiments:**
- Fig10-A current generic LLM same-rubric baseline.
- Fig10-B true disabled-module reruns.
- Fig10-C checkpoint ASPR-Qwen variant.
- Fig10-D blinded human preference.

- [ ] **Step 1: Make generic baseline resumable**

In `build_fig10_generic_baseline.py`, add:

```python
parser.add_argument("--resume", action="store_true")
parser.add_argument("--skip-existing", action="store_true")
```

Before calling Ollama, skip case if it already exists in `fig10_generic_llm_baseline_outputs.jsonl` with `run_status=="ok"`.

- [ ] **Step 2: Run generic baseline**

Run bounded smoke first:

```bash
python3 experiments/kg_perturbation_fig10/build_fig10_generic_baseline.py \
  --fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
  --out-dir outputs/kg_perturbation_fig10 \
  --model-name qwen3:8b \
  --timeout 120 \
  --max-cases 3 \
  --resume --skip-existing
```

Then full:

```bash
python3 experiments/kg_perturbation_fig10/build_fig10_generic_baseline.py \
  --fig4-metrics outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv \
  --out-dir outputs/kg_perturbation_fig10 \
  --model-name qwen3:8b \
  --timeout 120 \
  --max-cases 0 \
  --resume --skip-existing
```

- [ ] **Step 3: True module reruns**

Write `fig10_true_module_rerun_results.csv` with all variants:

```python
[
  "full ASPR",
  "no graph agent",
  "no ASPR-Qwen",
  "no prior-art retrieval",
  "no evidence trace",
  "no fusion",
  "no verifier",
  "generic LLM-only baseline",
]
```

Required columns:

```python
["variant", "case_id", "source", "run_status", "review_text_path", "evidence_trace_path", "runtime_seconds", "failure_reason"]
```

- [ ] **Step 4: Human preference protocol**

Write `fig10_human_preference.csv`:

```python
["case_id", "comparison", "evaluator_id", "blind_setting", "preference", "novelty", "prior_art", "evidence_grounding", "usefulness", "factuality"]
```

Minimum:

- 3 evaluators.
- system names hidden.
- full ASPR versus generic LLM-only comparisons.
- report paired bootstrap or Wilcoxon signed-rank for each dimension.

- [ ] **Step 5: Strong gate**

Fig10 can support strong performance claims only when:

```python
true_disabled_module_reruns == 1
blinded_human_preference == 1
checkpoint_generated_aspr_qwen == 1
full ASPR significantly beats generic LLM-only in evidence_grounding or prior_art or usefulness
```

Otherwise:

```python
main_or_extended_data = "extended"
allowed_claim = "Pipeline audit and failure analysis."
forbidden_claim = "Do not claim ASPR superiority or causal module contribution."
```

---

## Task 12: Final Assembly And Nature Check

**Files:**
- Modify: `experiments/kg_perturbation_final_assembly/build_final_assembly.py`
- Modify: `experiments/nature_ready_checks.py`
- Test: `tests/test_final_assembly.py`
- Test: `tests/test_nature_ready_claims.py`

- [ ] **Step 1: Ensure captions derive from claim ledger**

No final assembly caption may contain a phrase that appears in a figure's `forbidden_claim`.

- [ ] **Step 2: Run full current package**

Run:

```bash
make figures-current
python3 experiments/kg_perturbation_final_assembly/build_final_assembly.py
make figures-nature-check
```

- [ ] **Step 3: Expected final status before submission**

`make figures-nature-check` must return exit code 0 with:

```python
overall_pass == true
main blockers all passed:
  claim_ledger_written
  fig1_fig3_strong_gates
  fig4_external_validation
  fig6_full_rerun_robustness
```

Extended Data failures may remain only if:

```python
blocking_scope == "extended_data_nonblocking"
main_or_extended_data == "extended"
caption uses downgraded wording
```

- [ ] **Step 4: Run tests**

Run:

```bash
make test-nature-ready
python3 -m unittest \
  tests.test_fig2_reference_closure \
  tests.test_fig3_holdout_baselines \
  tests.test_fig4_external_validation \
  tests.test_fig6_full_rerun \
  tests.test_fig9_checkpoint_boundary \
  tests.test_fig10_true_ablation_human_preference \
  -v
```

Expected: all tests pass.

---

## Shortest Execution Order

1. Task 1: make reruns reliable.
2. Task 2: lock Fig1 as ready and prevent overclaiming.
3. Task 3: repair Fig2 sample/control/reference closure.
4. Task 4: repair Fig3 sample, baselines, holdouts, no-leakage prediction strength.
5. Task 5: repair Fig4 external validation.
6. Task 7: run Fig6 full rerun robustness.
7. Task 12: final assembly and Nature check.
8. Task 6 only if Fig5 should be a main optional forecast figure.
9. Task 8 for packaging polish.
10. Tasks 9-11 only if Extended Data performance/application claims are desired.

---

## Self-Review

### Spec Coverage

- Cross-figure claim ledger and Makefile rerun reliability are covered by Task 1 and Task 12.
- Fig1 sampling/horizon and caption constraint are covered by Task 2.
- Fig2 eligible corpus, reference closure, and controls are covered by Task 3.
- Fig3 top-up, holdouts, baselines, enrichment, and no-leakage are covered by Task 4.
- Fig4 fixed sample, retrieval, embedding, matcher, and external validation are covered by Task 5.
- Fig5 backtest metrics and demotion rule are covered by Task 6.
- Fig6 full graph rerun is covered by Task 7.
- Fig7 sensitivity and downgraded wording are covered by Task 8.
- Fig8 deterministic non-performance renderer contract is covered by Task 9.
- Fig9 checkpoint boundary is covered by Task 10.
- Fig10 true ablation and human preference are covered by Task 11.
- Final assembly and Nature check are covered by Task 12.

### Placeholder Scan

This plan avoids open-ended implementation labels. Every task names files, outputs, commands, and acceptance criteria.

### Type Consistency

The gate names match the current code direction:

- `edge_cap_not_hit_all_domains`
- `final_cumulative_horizon_consistent`
- `total_eligible_papers_min8000`
- `reference_closure_coverage_min80pct`
- `learned_oof_spearman_ge_0_45`
- `top_decile_enrichment_ge_5x`
- `fig4_external_validation`
- `nature_strong_claim_ready`
- `true_disabled_module_reruns`
- `blinded_human_preference`
- `checkpoint_generated_aspr_qwen`

