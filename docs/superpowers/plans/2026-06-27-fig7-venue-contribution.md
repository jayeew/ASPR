# Fig7 Venue Contribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Fig.7 assets that rank venue-family contribution to graph-perturbing research with Nature Portfolio highlighted and explicit data-quality gaps.

**Architecture:** Add one bounded, reproducible builder under `experiments/kg_perturbation_fig7/`. It reads existing Fig.3 paper-level metrics, tops up only missing venue metadata for those papers from OpenAlex, computes field-year and control-adjusted contribution indices, and writes figure panels plus audit tables to `outputs/kg_perturbation_fig7/`.

**Tech Stack:** Python, pandas, numpy, requests, Pillow, existing `experiments.figure_quality` manifest helpers.

---

### Task 1: Fig.7 Builder

**Files:**
- Create: `experiments/kg_perturbation_fig7/build_fig7_venue_contribution.py`
- Modify: none

- [ ] **Step 1: Add data loading and OpenAlex metadata cache**

Implement functions that load `fig3_score_table.csv`, `fig3_publication_day_indicators.csv`, `fig3_future_graph_deltas.csv`, extract OpenAlex work IDs from `paper_id`, fetch at most those IDs from OpenAlex in batches, and append JSONL cache rows to `outputs/kg_perturbation_fig7/openalex_work_metadata.jsonl`.

- [ ] **Step 2: Add venue-family mapping**

Implement `map_venue_family()` with auditable rules for Nature Portfolio, Science family, Cell Press, PNAS, Lancet family, IEEE/ACM, Elsevier, Springer Nature, Wiley, ACS, APS, RSC, OUP, CUP, MDPI, Frontiers, PLOS, eLife, IOP, AIP, missing venues, and other publishers.

- [ ] **Step 3: Add normalized metrics**

Compute field-year normalized publication-day score from `S_w`, residualize controls for article type and log reference count, and compute parallel normalized future impact from `RGPM`. Save paper-level metrics and venue-family summaries.

- [ ] **Step 4: Add panels**

Render six PNG panels with Pillow: venue portfolio map, controlled VCI ranking, top-K enrichment, mechanism signature heatmap, publication-day signal versus future impact, and control/audit summary. Compose `fig7_full.png`.

- [ ] **Step 5: Add quality gates and notes**

Write `figure_quality_report.json`, `run_manifest.json`, `fig7_gap_list.md`, and `fig7_methods.md`. Mark the headline as strict only when Nature Portfolio ranks first and its interval clears the runner-up interval.

### Task 2: Focused Tests

**Files:**
- Create: `tests/test_fig7_venue_contribution.py`

- [ ] **Step 1: Test venue-family mapping**

Check that representative source/publisher combinations map to Nature Portfolio, Science family, Cell Press, PNAS, Lancet family, and Elsevier non-Cell/Lancet.

- [ ] **Step 2: Test normalization and enrichment**

Use a small fixture to confirm field-year normalization returns finite values and top-K enrichment identifies the family with more top papers.

### Task 3: Execution And QA

**Files:**
- Generated: `outputs/kg_perturbation_fig7/*`

- [ ] **Step 1: Run tests**

Run: `python3 tests/test_fig7_venue_contribution.py`

- [ ] **Step 2: Generate Fig.7 outputs**

Run: `python3 experiments/kg_perturbation_fig7/build_fig7_venue_contribution.py`

- [ ] **Step 3: Inspect artifacts**

Open the generated PNG dimensions and quality report. Confirm audit CSVs exist, metadata coverage is reported, and the gap list explains whether the headline is strictly supported.
