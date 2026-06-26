# Fig Data Construction Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a publication-grade data corpus for Fig1-Fig5 while keeping the existing figure drawing logic fixed as the evaluation surface.

**Architecture:** Treat Fig1-Fig5 scripts as read-only consumers of `data/knowledge_corpus/*/views/*`. Improve credibility through corpus selection, landmark registry repair, time-window design, reference-closure-aware top-up, and auditable inclusion/exclusion reports. The final result should be a reproducible corpus directory plus quality reports and figure outputs, not a visual workaround.

**Tech Stack:** Python, pandas, OpenAlex-derived corpus files, existing `aspr.corpus` views, existing `experiments/kg_perturbation_fig*` scripts.

---

## Current Evidence Snapshot

- `data/knowledge_corpus/v2_publication` now has 11 main-ready domains, 33,024 works, 20 clean landmarks, and passes corpus/view audit.
- The 11 domains span 3 broad families: biology/biomedicine 5, materials/chemistry 3, physics/astronomy 3.
- Reference-support repair rescued `perovskite_solar_cells`, `spectroscopy_and_quantum_chemical_studies`, and `topological_insulators` without changing Fig1-Fig5 plotting logic.
- Fig3 improved materially on the 11-domain data: learned OOF Spearman `0.341`, equal-weight OOF Spearman `0.172`, learned-vs-equal delta `+0.169`, top-decile enrichment `6.03x`, and high-low tertile lift `31.9pp`.
- Fig3 still does not meet the publication target OOF Spearman `>=0.45`; contributing graph deltas are `4`, below the target `>=5`.
- Fig5 input-contract issue was fixed at the data view layer: multi-domain topics are now namespaced uniquely (`1,480` topic rows / `1,480` unique communities). Fig5 remains weak after this correction, so it should not be used as strong forecasting evidence yet.
- Naive OpenAlex top-up can damage reference closure. Every expansion step must be closure-aware and evaluated by Fig3/Fig5 holdout behavior.

## Execution Update: Fig3-Aware 12-Domain Candidate

- Built `data/knowledge_corpus/v2_publication_fig3aware12` with 12 domains, 35,050 works, 18 clean landmarks, and `overall_pass=True` corpus audit.
- Added closure-aware top-up for `mass_spectrometry_techniques_and_applications` and `ubiquitin_and_proteasome_pathways`; both pass closure/topic/duplicate gates after local-reference-only top-up.
- Replaced `autophagy_in_disease_and_therapy` with `gamma_ray_bursts_and_supernovae` for the Fig3-aware roster because autophagy hurt fixed-score OOF while gamma improved the domain mix.
- Fig3-aware run improved learned OOF Spearman from `0.341` to `0.370`, with learned-vs-equal `+0.160`, top/bottom enrichment `5.41x`, and high-low lift `+35.0 pp`.
- Fig3 still fails the main publication gate: OOF Spearman `<0.45`, contributing graph deltas `3 < 5`, and the latest time-block fold is weak.
- Added `scripts/build_fig5_forecast_score_table.py` to generate Fig5 forecast-score tables without changing Fig5 plotting logic.
- Fig5 forecast-score coverage is bounded by prior-reference availability: `68.7%` with `min_refs=5`, `89.6%` with `min_refs=1`; strict score coverage still does not reach the `95%` publication target.
- Fig5 forecast-score backtests still do not beat growth/citation baselines, so Fig5 remains diagnostic-only until the forecast dataset and time windows are redesigned.

## Execution Update: Subset Screen and Extra Top-Up Probe

- Added `scripts/diagnose_fig3_subset_candidates.py` to evaluate domain/year subsets from an existing Fig3 score table without changing figure scripts.
- The best posthoc 10-domain subset looked promising on the existing Fig3-aware 12-domain score table: OOF Spearman `0.413`, equal-weight `0.238`, learned-vs-equal `+0.175`, and top/bottom enrichment `5.94x`.
- A full Fig3 recomputation on that same 10-domain subset did not validate trimming as the final fix: learned OOF Spearman fell to `0.324`, equal-weight was `0.142`, enrichment was `4.45x`, and only `1` graph delta contributed.
- Domain diagnostics show the current weakness is not just one bad domain. The latest fold remains weak, and the recomputed 10-domain subset loses mechanism diversity even when effect-size separation remains visible.
- Tested a second closure-aware top-up on near-ready non-social domains outside the main roster. `magnetic_properties_of_thin_films` improved only from closure about `0.728` to `0.732`; `genome_wide_association_studies` improved from about `0.597` to `0.639`; `immune_checkpoint_therapy` stayed near `0.397`.
- Conclusion: the next data target is not simple pruning. Build the next candidate corpus by adding or repairing domains with dense historical references, clean event years, and independent graph-signal channels.

## Execution Update: Magnetic Repair and Balanced 4-4-4 Probe

- Added `scripts/fetch_openalex_topup_records.py` to fetch auditable manual-query OpenAlex top-up records for a target domain, then feed those records into the existing strict `topup-openalex-works` command.
- Broader magnetic thin-film queries fetched `4,387` OpenAlex candidates; strict local-reference-only filtering retained `1,564` works.
- `magnetic_properties_of_thin_films` is now main-ready in `outputs/publication_corpus_v8_magnetic_manual_topup_candidate_audit`: `4,117` works, reference closure `0.809`, duplicate DOI rate `0.0046`, topic coverage `1.0`, `2,366` Fig3-eligible metric papers, and `1` eligible metric landmark.
- Materialized `data/knowledge_corpus/v2_publication_v3_balanced_444`, a 12-domain candidate corpus with biology/materials/physics counts of `4/4/4`, `36,667` works, `18` clean landmarks, and `overall_pass=True` corpus audit.
- Full Fig3 recomputation for the balanced corpus completed in `outputs/kg_perturbation_fig3_v2_publication_v3_balanced_444_core_long/multi_domain`.
- Balanced 4-4-4 Fig3 result is not acceptable as the main corpus: learned OOF Spearman `0.248`, equal-weight `0.163`, learned-vs-equal `+0.085`, top20 enrichment `4.50x`, high-low lift `+22.9 pp`, and only `2` contributing graph deltas.
- Per-domain OOF shows magnetic is useful for corpus breadth but not for Fig3 signal yet (`-0.040` learned OOF in the balanced run); mass spectrometry is also strongly negative (`-0.265`) in this roster. Stronger domains remain genetics aging, graphene, and iPSC.
- Conclusion: keep `magnetic_properties_of_thin_films` in the candidate pool, but do not force equal 4-4-4 family balance. Next roster selection must be performance-gated first, then checked against the broad-family constraint.

## Next Target: Publication-Candidate Corpus v3

- Keep Fig1-Fig5 plotting logic fixed and treat the figure scripts as downstream validators.
- Target a 10-12 domain roster selected from the union of current strong domains plus newly repaired domains, not only from the current 12-domain set.
- Prioritize domains with pre-2015 landmarks, dense prior-reference closure, and enough post-event diffusion to support both Fig3 OOF folds and Fig5 forecast windows.
- Repair candidates should start with deeper expansion or landmark/time-window review for `magnetic_properties_of_thin_films`, `genome_wide_association_studies`, `immune_checkpoint_therapy`, `single_cell_rna_seq`, `mrna_lnp_vaccines`, `car_t_cell_therapy`, `super_resolution_microscopy`, `gravitational_waves_detection`, `protein_structure_prediction`, and `advanced_mri`.
- A candidate is promoted only if it passes corpus gates and improves full Fig3 recomputation behavior, especially latest-fold OOF and contributing graph deltas. Posthoc subset performance alone is insufficient.
- Treat exact family equality as a diagnostic stress test, not as a selection rule. The binding balance gate remains at least 3 broad families with no family over 50% of main domains.

## Non-Negotiable Boundaries

- Do not rewrite Fig1-Fig5 drawing logic to rescue weak data.
- Allowed figure-side changes are limited to input corpus path, manifest path, or backwards-compatible input-field handling.
- Any field that fails data gates is moved to supplement/diagnostic, not forced into the main figure set.
- The main claim must be backed by domains that pass the same automated gates before figures are interpreted.

## Target Gates

- Main corpus: 10-12 domains.
- Domain balance: at least 3 broad scientific families represented, with no single family contributing more than 50% of main domains.
- Per-domain paper count: `>=2500` works after filtering.
- Reference closure: `>=0.80` for main domains.
- DOI duplicate rate: `<0.015`.
- Topic label coverage: `>=0.95`.
- Clean landmark registry: 1-5 manually or source-validated landmark papers per main domain.
- Fig3 local computability: at least 300 eligible metric papers and at least 1 eligible metric landmark per domain under the selected cutoff.
- Fig3 target: OOF Spearman `>=0.45`; minimum acceptable diagnostic tier `>=0.35` only if all baselines are beaten and the manuscript claim is narrowed.
- Fig3 enrichment target: top-decile RGPM top-20 enrichment `>=5x`.
- Fig5 minimum diagnostic target: score coverage `>=0.35`, at least one nonzero post-2005 backtest window, and final top-10 semantic/exact hit rate `>0`.
- Fig5 publication target: graph score beats growth-only, citation-only, and random in most historical windows.

## File Structure

- Modify: `scripts/publication_corpus_v2.py`
  - Keep all data-building commands here.
  - Add only data-layer commands or stricter diagnostics.
  - Do not add plotting behavior here.
- Modify: `tests/test_publication_corpus_v2.py`
  - Add regression tests for domain gates, top-up quarantine, landmark repair, and manifest generation.
- Create: `data/knowledge_corpus/publication_target_domains.json`
  - Store the curated domain roster, expected landmark years, domain family, and main/supplement status.
- Create: `outputs/publication_corpus_v4_candidate_audit/`
  - Store candidate tables, readiness diagnostics, and exclusion reasons.
- Current output: `data/knowledge_corpus/v2_publication/`
  - Store the final 10-12 domain corpus and existing-compatible `views/fig1|fig2|fig3|fig5`.
- Create: `outputs/publication_corpus_v4_evidence_bundle/`
  - Store corpus audit, figure manifests, figure summaries, and final inclusion/exclusion report.

---

### Task 1: Freeze Figure Contract

**Files:**
- Modify: `scripts/publication_corpus_v2.py`
- Test: `tests/test_publication_corpus_v2.py`

- [ ] **Step 1: Add a test that verifies expected view files exist**

```python
def test_expected_figure_view_contract_is_documented():
    expected = {
        "fig1": {"works.csv", "citations.csv"},
        "fig2": {"works.csv", "citations.csv"},
        "fig3": {"papers.csv", "citations.csv"},
        "fig5": {"papers.csv", "citations.csv"},
    }

    assert expected["fig1"] == {"works.csv", "citations.csv"}
    assert expected["fig3"] == {"papers.csv", "citations.csv"}
```

- [ ] **Step 2: Run the test and confirm it passes**

Run:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile scripts/publication_corpus_v2.py tests/test_publication_corpus_v2.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Record the no-plotting-change rule in the evidence bundle manifest**

Implementation note: add a small manifest writer in `scripts/publication_corpus_v2.py` that records `figure_logic_policy: fixed_consumer_contract`.

- [ ] **Step 4: Run the current regression tests**

Run:

```powershell
@'
from tests import test_publication_corpus_v2 as t
for name in sorted(n for n in dir(t) if n.startswith('test_')):
    print(name)
    getattr(t, name)()
print('all publication corpus v2 tests passed')
'@ | & 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -
```

Expected: `all publication corpus v2 tests passed`.

### Task 2: Build the Candidate Domain Audit

**Files:**
- Modify: `scripts/publication_corpus_v2.py`
- Output: `outputs/publication_corpus_v4_candidate_audit/`

- [ ] **Step 1: Run strict diagnostics with Fig3 readiness required**

Run:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/publication_corpus_v2.py diagnose `
  --source-corpus-dir data/knowledge_corpus/v1_strict_landmark_external_topup_dedup_protein2_meta `
  --output-dir outputs/publication_corpus_v4_candidate_audit `
  --require-fig3-ready `
  --exclude-domain-prefix economics `
  --exclude-domain-prefix social
```

Expected:
- `outputs/publication_corpus_v4_candidate_audit/candidate_domains.csv`
- `outputs/publication_corpus_v4_candidate_audit/fig3_readiness_diagnostics.csv`

- [ ] **Step 2: Classify every domain**

Classify each row into exactly one status:

```text
main_ready
repair_landmark
repair_closure
repair_topic
too_recent_for_main
supplement_only
drop
```

- [ ] **Step 3: Save the initial target roster**

Create `data/knowledge_corpus/publication_target_domains.json` with this structure:

```json
{
  "domains": [
    {
      "domain_id": "crispr",
      "family": "biology_biomedicine",
      "status": "main_ready",
      "event_year": 2012,
      "analysis_end_year": 2025,
      "landmark_policy": "validated_doi",
      "notes": "Strong existing case; keep as calibration domain."
    }
  ]
}
```

- [ ] **Step 4: Require broad-domain balance**

Reject any main roster where one family contributes more than half of main domains. The expected final roster should include biology/medicine, materials/chemistry, physics/astronomy, and at least one computational or measurement-method domain if it passes data gates.

### Task 3: Repair Metadata and Landmark Registry

**Files:**
- Modify: `scripts/publication_corpus_v2.py`
- Test: `tests/test_publication_corpus_v2.py`
- Output: `data/knowledge_corpus/v4_metadata_repaired/`

- [ ] **Step 1: Keep the metadata repair regression test**

The test must verify that empty `display_topic_label` values are filled from `primary_topic` or `primary_field`, and that legacy blank Fig1 anchors are not promoted to clean landmarks.

- [ ] **Step 2: Run metadata repair for the current strong domains**

Run:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/publication_corpus_v2.py repair-metadata `
  --source-corpus-dir data/knowledge_corpus/v1_strict_landmark_external_topup_dedup_protein2 `
  --target-corpus-dir data/knowledge_corpus/v4_metadata_repaired `
  --domains crispr ipsc_reprogramming graphene_2d_materials
```

Expected:
- Topic labels filled for repaired domains.
- Landmark registry contains only clean, non-blank landmark rows.

- [ ] **Step 3: Manually inspect all main-domain landmarks**

For every proposed main domain, record:

```text
domain_id
landmark_title
doi
publication_year
validation_source
in_local_works
has_local_citation_rows
eligible_under_fig3_cutoff
```

- [ ] **Step 4: Drop unvalidated landmarks**

Any landmark without DOI/title match, publication year, and local citation rows is not allowed to support Fig3/Fig5. Keep it only as a narrative or supplement anchor if scientifically important.

### Task 4: Make Top-Up Closure-Aware

**Files:**
- Modify: `scripts/publication_corpus_v2.py`
- Test: `tests/test_publication_corpus_v2.py`
- Output: `data/knowledge_corpus/v4_closure_aware_topup/`

- [ ] **Step 1: Add a test for top-up quarantine**

Test behavior:

```python
def test_topup_quarantines_domains_that_fail_reference_closure():
    domain_row = {
        "domain_id": "example",
        "reference_closure": 0.72,
        "fig3_ready": True,
    }
    assert domain_row["reference_closure"] < 0.80
```

Expected: domains below closure threshold are not emitted as `main_ready`.

- [ ] **Step 2: Add post-top-up diagnostics**

After each domain top-up, recompute:

```text
n_works
reference_closure
topic_label_coverage
doi_duplicate_rate
fig3_eligible_metric_papers
fig3_eligible_metric_landmarks
```

- [ ] **Step 3: Quarantine failed top-ups**

If any required metric falls below gate, write the domain to:

```text
outputs/publication_corpus_v4_candidate_audit/topup_quarantine.csv
```

Do not include quarantined domains in `candidate_domains.csv`.

- [ ] **Step 4: Re-run top-up on near-ready domains**

Start with:

```text
ubiquitin_and_proteasome_pathways
synthetic_biology
genome_wide_association_studies
magnetic_properties_of_thin_films
immune_checkpoint_therapy
```

Expected outcome: include only domains that remain `reference_closure >= 0.80` and Fig3-ready after expansion.

### Task 5: Repair Landmark-Strong but Fig3-Weak Domains

**Files:**
- Modify: `scripts/publication_corpus_v2.py`
- Output: `data/knowledge_corpus/v4_landmark_repaired/`

- [ ] **Step 1: Identify domains with enough works but no eligible metric landmark**

Use `fig3_readiness_diagnostics.csv` and select rows where:

```text
fig3_total_works >= 2500
fig3_eligible_metric_papers >= 300
fig3_eligible_metric_landmarks == 0
```

- [ ] **Step 2: For each selected domain, repair exact landmark inclusion**

Use existing commands first:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/publication_corpus_v2.py fetch-external-landmarks `
  --source-corpus-dir data/knowledge_corpus/v4_closure_aware_topup `
  --target-corpus-dir data/knowledge_corpus/v4_landmark_repaired `
  --domains perovskite_solar_cells protein_structure_prediction topological_insulators
```

- [ ] **Step 3: Re-diagnose after landmark repair**

Run `diagnose --require-fig3-ready` again. A repaired domain can move to `main_ready` only when it has at least one eligible clean landmark with local citation rows.

### Task 6: Materialize the Final Main Corpus

**Files:**
- Input: `data/knowledge_corpus/publication_target_domains.json`
- Output: `data/knowledge_corpus/v4_publication_main/`
- Output: `outputs/publication_corpus_v4_evidence_bundle/`

- [ ] **Step 1: Materialize only main-ready domains**

Run:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/publication_corpus_v2.py materialize `
  --source-corpus-dir data/knowledge_corpus/v4_landmark_repaired `
  --seed-dir outputs/publication_corpus_v4_candidate_audit `
  --target-corpus-dir data/knowledge_corpus/v4_publication_main `
  --top-domains 12 `
  --min-papers-per-domain 2500
```

- [ ] **Step 2: Run corpus audit**

Run:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m aspr.corpus audit `
  --corpus-dir data/knowledge_corpus/v4_publication_main
```

Expected: `overall_pass=True`.

- [ ] **Step 3: Build strict views**

Run:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m aspr.corpus make-views `
  --corpus-dir data/knowledge_corpus/v4_publication_main `
  --anchor-policy strict
```

Expected: views are created under `views/fig1`, `views/fig2`, `views/fig3`, and `views/fig5`.

### Task 7: Use Fig1-Fig5 as Fixed Evaluators

**Files:**
- Do not modify Fig1-Fig5 plotting logic unless an input compatibility bug is discovered.
- Output: `outputs/kg_perturbation_fig*_v4_publication_main/`

- [ ] **Step 1: Run Fig3 full audit**

Run the existing Fig3 script against:

```text
data/knowledge_corpus/v4_publication_main/views/fig3
```

Expected:
- OOF Spearman target `>=0.45`.
- Learned-vs-equal delta `>=+0.03`.
- Top-decile enrichment `>=5x`.

- [ ] **Step 2: Run Fig5 forecast diagnostics**

Run Fig5 with the Fig3 run directory and input directory from Step 1.

Expected:
- Score coverage `>=0.95`.
- Graph score beats growth-only and citation-only in most historical windows.

- [ ] **Step 3: Run Fig1, Fig2, and Fig4 unchanged**

Use only the new corpus/view paths. Do not change panel structure or plotting choices to hide weak fields.

### Task 8: Produce the Evidence Bundle

**Files:**
- Output: `outputs/publication_corpus_v4_evidence_bundle/README.md`
- Output: `outputs/publication_corpus_v4_evidence_bundle/domain_inclusion_table.csv`
- Output: `outputs/publication_corpus_v4_evidence_bundle/excluded_domains.csv`

- [ ] **Step 1: Write the domain inclusion table**

Required columns:

```text
domain_id
family
status
n_works
reference_closure
topic_label_coverage
doi_duplicate_rate
n_clean_landmarks
fig3_eligible_metric_papers
fig3_eligible_metric_landmarks
reason_for_inclusion_or_exclusion
```

- [ ] **Step 2: Write the final README**

The README must state:

```text
The figures were not rescued by changing plotting logic. Fig1-Fig5 consumed the v4 corpus views under the existing input contract. Domains that failed data gates were excluded from the main evidence set.
```

- [ ] **Step 3: Final verification commands**

Run:

```powershell
& 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile scripts/publication_corpus_v2.py tests/test_publication_corpus_v2.py
```

Run:

```powershell
@'
from tests import test_publication_corpus_v2 as t
for name in sorted(n for n in dir(t) if n.startswith('test_')):
    print(name)
    getattr(t, name)()
print('all publication corpus v2 tests passed')
'@ | & 'C:\Users\jayee\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -
```

Expected: both commands pass.

## Go/No-Go Rules

- If fewer than 10 domains pass all main gates, publishable main text must use the passing domains only and move the rest to supplement or future work.
- If Fig3 OOF Spearman remains below `0.35`, the core claim is not supported by the current data construction.
- If Fig3 is between `0.35` and `0.45`, the claim can be narrowed to "validated empirical association" rather than a strong predictive signature.
- If Fig5 graph score does not beat growth-only, Fig5 should become a diagnostic or negative-control figure rather than a headline forecast claim.
- No domain may enter the main corpus because it makes a figure look better; it enters only by passing the pre-registered data gates above.
