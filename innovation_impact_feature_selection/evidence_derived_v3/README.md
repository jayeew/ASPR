# Evidence-derived innovation and potential-impact selection v3

This directory is a standalone, auditable workflow for deriving search
concept domains, model dimensions, and publication-time indicators from
English evidence. It does not import code into ASPR and does not modify the
v1/v2 evidence packages.

The workflow reports six separate, emergent counts:

- `K`: non-redundant evidence-derived search concept domains;
- `Q`: non-redundant logical query families;
- `P`: OpenAlex-executable physical queries;
- `M`: candidate model dimensions derived after indicator extraction;
- `D`: retained predictive dimensions;
- `F`: final indicator families across retained roles.

None is configured as a quota. Search domains are not model dimensions, and
physical API splits are not counted as new logical queries.

## State machine

```text
bootstrap inventory
→ deterministic saturation strata
→ sequential AI/H1 screening + H2 adjudication
→ source-preserving term/indicator extraction
→ three consecutive H2-approved dual-zero rounds
→ code-terms
→ derive-search-frame
→ validate-search-frame
→ retrieve
→ screen-literature
→ extract-indicators
→ derive-dimensions
→ select-indicators
→ audit
```

Every state has finer-grained export/import commands so AI, Human 1 (`H1`),
and Human 2 (`H2`) decisions are stored separately.
`H1` and `H2` are real-human roles: no model or automated helper may fill,
copy, or backfill either worksheet. Local-model output is accepted only under
the explicitly labelled `AI` role. Forced task regeneration refuses to
overwrite a nonblank human worksheet, and import rejects both the invalidated
artifact quarantine and the registered hashes of known automated H1/H2
files.

## Initial setup and domain-agnostic bootstrap

Run commands from this directory:

```bash
python3 pipeline.py init
python3 pipeline.py bootstrap --inventory-only
python3 pipeline.py derive-discovery-frame
python3 pipeline.py hydrate-development-seeds
python3 pipeline.py bootstrap-saturation
python3 pipeline.py expand-development-citations
```

The bootstrap query uses only the three frozen object/target/evidence blocks
in `bootstrap_query_v3.json`; it contains no old 12-domain labels. OpenAlex
language is intentionally not filtered at retrieval time. English-only
eligibility is applied during screening so non-English results remain in the
PRISMA denominator.

The broad expression is inventoried but is not downloaded in full. Discovery
uses the frozen design in `saturation_protocol_v3.json`: deterministic
OpenAlex random samples stratified by publication-year band and work type,
target/evidence-term oversamples, evidence-sourced formula-term probes, and
the complete forward/backward citation network of the 53 development papers.
No relevance-ranked Top-K cutoff is used, and no field/domain label is a
sampling stratum. The defensible claim is **systematic evidence mapping to
preregistered term-and-indicator saturation**, not an exhaustive census of
all broad-query hits.

The local OpenAlex snapshot is registered and hashed for historical coverage,
provenance, and seed/indexability checks. API retrieval covers deterministic
sampling and records newer than the snapshot. With multiple free keys, set a
comma-separated `OPENALEX_API_KEYS`; the scheduler records only non-secret
key-slot budgets, alternates available slots, and resumes after daily reset.
Keys are never persisted.

## Sequential evidence-saturation review

```bash
python3 pipeline.py assign-discovery-round --iteration 1
python3 pipeline.py ai-screen-discovery --iteration 1
python3 pipeline.py prepare-human-tasks
```

Each screening, term-coding, and dimension-coding import contains exactly one
reviewer role. H2 rows are rejected until both AI and H1 rows exist, and a
primary row becomes immutable once H2 has adjudicated it. Language and
eligibility evidence must both be exact title/abstract spans. Runs created
before this exact-language-evidence gate can be migrated once, without
changing any decision, using:

```bash
python3 pipeline.py normalize-ai-language-evidence
```

The command archives the prior AI CSVs, replaces only the legacy provenance
prose with the exact title (or abstract fallback), rewrites the derived AI
exports, and records a deterministic change hash.

`prepare-human-tasks` creates only currently actionable worksheets under
`outputs/human_tasks/` and never overwrites a nonblank existing worksheet.
An automated draft may be imported as H1/H2 only after the project owner
attests that a human reviewed and adopted the exact hashed file. The import
then receives a distinct `human_attested_automated_draft` provenance label;
it is not reported as an originally manual or independently generated blind
draft. Unreviewed automated trial files remain quarantined and hash-blocked.
Current worksheet types are:

- `round_01_screening_H1_BLIND.csv` contains source records but no AI
  decisions;
- after H1 import, the H2 adjudication file contains both independent
  decisions and only the mandatory H2 queue;
- `term_coding_H1_BLIND.csv` contains source evidence but no AI coding;
- after H1 term coding, the H2 file contains all retained assignments and
  disagreements;
- `hidden_validation_seed_search_log_H2.csv` documents H2's exact independent
  review search plus backward and forward citation tracing;
- `hidden_validation_seeds_H2.csv` is completed independently by H2.

For direct manual H1 work, the standard-library-only local helper reads only
the selected blind worksheet, rejects any AI/H2 comparison column, validates
source evidence, atomically saves every decision, and resumes at the first
blank row:

```bash
python3 human_review_cli.py screen \
  --input outputs/human_tasks/round_01_screening_H1_BLIND.csv
python3 human_review_cli.py status \
  --input outputs/human_tasks/round_01_screening_H1_BLIND.csv
python3 human_review_cli.py validate \
  --input outputs/human_tasks/round_01_screening_H1_BLIND.csv
python3 human_review_cli.py code-terms \
  --input outputs/human_tasks/term_coding_H1_BLIND.csv
python3 human_review_cli.py term-status \
  --input outputs/human_tasks/term_coding_H1_BLIND.csv
python3 human_review_cli.py validate-terms \
  --input outputs/human_tasks/term_coding_H1_BLIND.csv
```

For large H2 queues, `draft_h2_assistance.py` can create review aids only
under `outputs/unreviewed_automated_h2_drafts_20260729/`. Screening drafts
start from the human-reviewed H1 decision and use a fresh model suggestion
only for H1-uncertain rows. Crossref drafts contain deterministic triage
suggestions. The provisional term draft exposes domain support and blank H2
merge/split fields; it cannot authorize `K`, especially while its
`direct_authorizing_term_count` is zero.

These files are deliberately non-importable: both their directory and exact
decision-file hashes are blocked. A human must review/adopt an exact file,
record an attestation, and amend the provenance allowlist before import.
Rejected model codebook attempts remain in the quarantine as evidence that
source/role validation prevented a mechanically copied taxonomy.

Import a completed H1 screening file, then regenerate the action register:

```bash
python3 pipeline.py import-screening \
  --input outputs/human_tasks/round_01_screening_H1_BLIND.csv
python3 pipeline.py prepare-human-tasks
```

After H2 screening is imported:

```bash
python3 pipeline.py finalize-discovery-screening --iteration 1
python3 pipeline.py export-discovery-extraction \
  --iteration 1 --extractor H1
python3 pipeline.py import-discovery-extraction \
  --input completed_round_01_extraction.csv
```

Every included record must have an explicit completed H1 extraction
disposition, including `no_relevant_items=true` when no item is found.
Title/abstract terms and indicator names must be verbatim substrings with an
exact evidence span. H2 then adjudicates every H1-retained indicator family.
After term coding and indicator adjudication, the database computes the two
novelty counts:

```bash
python3 pipeline.py discovery-novelty-status --iteration 1
```

H2 records those audited counts; a submitted count that differs from the
adjudicated database evidence is rejected. A freeze request is also rejected
unless the current and two immediately preceding fully reviewed rounds are
all dual-zero:

```bash
python3 pipeline.py record-discovery-saturation \
  --iteration 1 \
  --new-terms 4 \
  --new-indicator-families 2 \
  --decision continue \
  --notes "H2-verified novelty counts and reconciliation reference."
```

If either novelty endpoint is non-zero, assign the next deterministic rank
slice, screen/extract it, update term coding, and continue. This stopping rule,
not a target number of papers, domains, queries, dimensions, or indicators,
determines the discovery volume.

## Term extraction, independent coding, and dynamic K/Q/P

```bash
python3 pipeline.py export-term-extraction
python3 pipeline.py import-terms --input reviewed_terms.csv

# Separate files preserve reviewer roles and exact source provenance.
python3 pipeline.py export-term-coding --reviewer AI
python3 pipeline.py export-term-coding --reviewer H1
python3 pipeline.py export-term-coding --reviewer H2
python3 pipeline.py import-term-coding --input completed_ai.csv
python3 pipeline.py import-term-coding --input completed_h1.csv
python3 pipeline.py import-term-coding --input completed_h2.csv

python3 pipeline.py derive-search-frame
```

Every included term retains its exact English source span. H2 adjudication is
required for disagreements and for included domain assignments, so domain
merge/split decisions cannot be made by AI alone. A domain supported only by
v2 pilot terms or v2-derived development hints is rejected, and the same rule
applies to every logical query family. The 53 development papers remain
development/recall seeds, but names copied from the old
`formula_authorization` field are marked `development_seed_hint`; they may
add synonyms only after a directly verified English title/abstract term
establishes the family. Logical duplicates are archived; URL-length splits
share their parent logical-query ID.

## PRESS and recall validation

```bash
python3 pipeline.py export-press
python3 pipeline.py import-press --input completed_press_review.csv

python3 pipeline.py export-seed-template
python3 pipeline.py export-hidden-seed-search-log
python3 pipeline.py import-hidden-seed-search-log \
  --input completed_hidden_seed_search_log_H2.csv
python3 pipeline.py import-seeds --input hidden_validation_seeds.csv
python3 pipeline.py validate-search-frame
```

The hidden validation seed file must be supplied by H2 after initial query
derivation. Before validation, H2 must document completed independent review
search, backward citation tracing, and forward citation tracing, including
the exact query or seed, source, execution time, and retrieved/screened/
eligible counts. Each completed route also lists every eligible DOI it found;
its distinct DOI count must equal `eligible_seed_count`, and the union of
those route-level DOI lists must exactly equal the eligible English H2
hidden-validation seed set. Validation inventories every physical query,
archives zero-hit families, verifies PRESS, checks DOI indexability, and
requires recall of all eligible indexable development and hidden seeds.
Recall requests OR-batch up to 40 seed DOIs per physical query, while the
database still records the matching physical-query IDs separately for every
seed; this preserves the recall test while keeping free-API use proportional
to `P × ceil(seeds/40)` instead of `P × seeds`.
When H2 proposes a redundant logical query, validation cursor-pages both the
candidate and its asserted covering query and archives it only if the complete
candidate result set is a subset and H2 has confirmed no independent
construct role.

If an eligible English seed is not indexed by OpenAlex:

```bash
python3 pipeline.py export-seed-supplements
python3 pipeline.py import-seed-supplements \
  --input completed_seed_supplements.csv
```

After all issues are resolved:

```bash
python3 pipeline.py freeze-search-frame
```

Freezing writes a deterministic manifest and prevents in-place mutation of
terms/domains/queries. Evidence-driven term changes require a new version.
After freeze, rerunning `init` also refuses to replace an already registered
protocol, code, or source hash; restoring the frozen file or using the
H2-authorized reopen path is required.
Every pre-freeze re-derivation is preserved in `search_frame_versions` with
its term-input hash, K/Q/P, full query definitions, and superseded/frozen
status; earlier query frames are not silently overwritten.

## Formal retrieval, screening, and Crossref validation

```bash
python3 pipeline.py retrieve

python3 pipeline.py export-screening --reviewer AI
python3 pipeline.py export-screening --reviewer H1
python3 pipeline.py import-screening --input completed_ai.csv
python3 pipeline.py import-screening --input completed_h1.csv
python3 pipeline.py export-screening --reviewer H2
python3 pipeline.py import-screening --input completed_h2.csv
python3 pipeline.py finalize-screening

python3 pipeline.py crossref-validate --scope all --max-records 250
python3 pipeline.py export-crossref-conflicts
python3 pipeline.py import-crossref-resolutions \
  --input completed_crossref_conflicts.csv
```

`retrieve` does not cursor through every hit of every frozen formal query.
It registers one maximum-size (up to 10,000 records) seeded deterministic
review pool for each physical query, records the provider's full inventory
count, and adds those pools as `formal_search_family` strata. Pool pages are
fetched lazily only when the next frozen review-rank slice needs them.
Sequential
`assign-discovery-round` review then continues across the original broad
strata, formal query strata, and citation strata until the same dual-novelty
stopping rule is met. Unreviewed pool members remain an auditable sampling
frame and are not silently treated as screened exclusions.
The consecutive-zero counter is phase-specific: the terminology-phase freeze
that permits `K/Q/P` derivation cannot satisfy the later formal
indicator-discovery freeze. The formal phase must independently accumulate
three dual-zero rounds.

If formal search or citation evidence introduces a genuinely new
non-redundant English search-term family, H2 must authorize
`reopen-search-frame --notes ...`. The prior frozen manifest is copied to a
hash-named archive and retained in SQLite; terms are independently recoded,
a new `K/Q/P` version is derived and PRESS/recall validated, and formal pools
are rerun. A frozen frame is never silently edited in place.

H2 reviews all disagreements, all inclusions/uncertain records, and a stable
DOI-hash 10% sample of concordant exclusions. AI decisions require an English
title/abstract evidence span and cannot delete records. Non-English records
receive `E_LANGUAGE_NON_ENGLISH`.

OpenAlex performs discovery and citation tracking. Crossref validates DOI,
title, year, type, and publication metadata only. Crossref validation commits
each record independently and bounded batches are resumable. Already
validated, resolved, or queued-conflict records are not fetched again.
`CROSSREF_MAILTO` may be set in the environment to use Crossref's identified
polite pool; the contact value is never stored in SQLite or output files.
When DOI, title similarity (at least 0.85), and type agree, a publication-year
difference is preserved as `validated_date_variant` rather than sent to H2;
this captures online-first versus issue-year differences. DOI, title, type,
and provider errors remain in the H2 conflict queue. A stable Crossref DOI
endpoint 404 is recorded as `crossref_doi_not_found` and is not repeatedly
requested; it still requires H2 bibliographic resolution and never
automatically excludes a paper.

## Full-text indicator census and dynamic M/D/F

```bash
python3 pipeline.py acquire-open-fulltexts
python3 pipeline.py export-indicator-extraction
python3 pipeline.py import-indicators --input completed_indicators_H1.csv
python3 pipeline.py export-indicator-adjudication
python3 pipeline.py import-indicators --input completed_indicators_H2.csv
python3 pipeline.py export-data-audit
python3 pipeline.py import-data-audit --input completed_data_audit.csv

python3 pipeline.py ai-code-dimensions
python3 pipeline.py export-dimension-coding --reviewer H1
python3 pipeline.py import-dimension-coding --input completed_h1.csv
python3 pipeline.py export-dimension-coding --reviewer H2
python3 pipeline.py import-dimension-coding --input completed_h2.csv
python3 pipeline.py derive-dimensions
python3 pipeline.py select-indicators
```

Reviews may discover indicators but cannot authorize formulas. A final
indicator requires verified English full-text formula evidence, exact
location, units/parameters/direction/missing rule, T0 compliance, audited
data readiness, non-constant data, bias safeguards, and H2 approval.
Every `english_fulltext_verified=true` mention must also provide a lawful
HTTP(S) source, access/licence note, and a local evidence file. Import
computes the file's SHA-256, rejects a conflicting claimed hash, and registers
that exact version in `source_snapshots`; deleting or changing it becomes an
audit blocker.
For UTF-8 text and PDFs, the importer extracts local content and requires the
normalized English evidence span to occur in the frozen file. Formula
locations must identify a page, table, equation, appendix, or section.
Before extraction, `acquire-open-fulltexts` considers only HTTP(S) PDF
locations that OpenAlex explicitly marks `is_oa=true`. It downloads bounded
files atomically, validates the PDF signature, records redirects/content
type/access statement, computes SHA-256, registers the file as a candidate
source snapshot, and resumes completed or failed records. The extraction
queue is prefilled with those artifacts, but reviewers must still verify the
English text, lawful access statement, formula, and exact location.
Because broad retrieval stores compact metadata, this command first refreshes
`primary_location`, `best_oa_location`, and `locations` only for the small
finally included source set, alternating configured OpenAlex key slots and
recording a provider-payload hash without persisting a key.
Unavailable full text remains explicitly queued and cannot satisfy the
formula gate.
H1 must disposition every included source and extract every mention. The H2
worksheet then carries forward H1's evidence while blanking the H2 approval
and adjudication fields; the indicator stage remains incomplete until H2 has
reviewed every source disposition and approved or excluded every retained
mention.
The two imports are deliberately sequential: an H2 row is rejected unless
the same source and mention identity already have an H1 record. SQLite keeps
the normalized H1 and H2 rows separately in `indicator_source_reviews` and
`indicator_mention_reviews`, in addition to immutable CSV snapshots and the
resolved operational tables. Once an H2 adjudication exists, the associated
H1 row cannot be overwritten in place.
The data-audit import requires reproducible derivation/input SHA-256 hashes
and verifies row, valid, unique, and missingness counts. Passing rows must
provide the actual local derivation artifact and frozen input manifest; the
importer computes both hashes, checks any claimed hashes, and registers the
files in `source_snapshots`. Unchecked spreadsheet claims cannot satisfy the
data-quality or non-constant gates.

Canonical indicator families are built before model dimensions. H2 must
adjudicate every final family-to-dimension mapping. A candidate dimension is
eliminated if no indicator passes every gate or fewer than two independent
research teams support it. Every indicator mention records a normalized team
ID plus author/affiliation evidence reviewed by H2; repeated papers or
spelling variants from one team therefore count once. Opportunity,
context-control, and sensitivity sets are retained separately and do not
inflate `D`.
Only teams attached to an H2-approved, English-full-text-verified mention are
counted. The recorded author/affiliation evidence must itself occur as a
normalized exact span in the frozen local full text; an unchecked team label
cannot satisfy the two-team dimension gate.
The local AI and H1 code dimensions independently from formula, T0,
required-data, source-team, and mention evidence. The H1 worksheet contains no
AI/H2 comparison columns. Only after both are complete does the H2 worksheet
expose their codes for merge/split, exclusion, and multi-label adjudication;
the local AI model digest and prompt/input hashes remain in the audit trail.

Redundancy selection is frozen before results are known. The numeric
`selection_priority` is not freely chosen: it must equal the source-role map
in `screening_rules_v3.json` (`original_definition` first, followed by
mathematical foundation, original application, validation, and review-only
discovery). Evidence strength uses a closed vocabulary. `stability_score`
must be in `[0,1]`, and every row supplies `stability_basis` stating the exact
source-reported or frozen-audit statistic and normalization; absent
quantitative stability evidence is recorded as zero rather than guessed.

## Citation saturation

For included reviews and indicator-defining/validation papers:

```bash
python3 pipeline.py citation-track \
  --iteration 1 --scope reviews_and_indicator_sources
```

This frozen scope is the union of included English reviews and papers with
an H2-reviewed extracted or full-text-pending indicator source disposition.
Backward OpenAlex work IDs are deduplicated across sources, records already
present in the local evidence database are reused, and the remaining IDs are
queried in frozen batches of at most 100. Forward searches batch at most 50
source IDs per cursor-paged query. Physical definitions, result counts,
cursor checkpoints, and hits are stored in the normal query audit tables;
the free-credit-aware scheduler rotates configured API-key slots per request
and fails over without storing either key. This changes request count, not
citation eligibility or the source-to-target edge set.
New records return to screening and indicator extraction. After reconciling
new English terms and canonical indicator families, H2 records the formal
citation round:

```bash
python3 pipeline.py record-saturation-round \
  --iteration 1 \
  --new-records 0 \
  --new-terms 0 \
  --new-indicator-families 0 \
  --decision freeze \
  --notes "No new non-redundant English term or indicator family."
```

A discovery-frame freeze requires three consecutive H2-approved dual-zero
rounds. Later formal citation tracking must also finish with an H2-approved
zero-novelty round; otherwise the run remains incomplete and a revised,
newly frozen search-frame version is required.

## Audit and outputs

```bash
python3 pipeline.py audit
python3 pipeline.py status
python3 tests_v3.py
```

The `outputs/` directory contains:

- the independent SQLite audit database;
- frozen protocol/search hashes;
- raw/canonical term evidence tables;
- logical/physical queries, PRESS, and seed-recall tables;
- PRISMA literature dispositions and agreement results;
- the complete indicator mention/family library and separate H1/H2 source
  and mention review tables;
- candidate-dimension mappings and merge/split reasons;
- every hard-gate and redundancy decision;
- role-separated, training-ready final features;
- a requirement-by-requirement `completion_matrix_v3.csv`;
- an audit report and hash manifest.

`audit` is intentionally allowed to finish with status `INCOMPLETE`: it lists
the exact missing human or retrieval gate and never treats an unfinished
pipeline as a completed systematic review.

The audit additionally exports discovery strata, source evidence, retrieval
runs, review rounds, indicator candidates, and the sequential saturation
curve. It blocks completion if any active discovery stratum is incomplete,
the three-round H2 stopping rule is absent, no hidden H2 validation seed is
registered, the H2 search-log DOI set does not reconcile to the hidden seeds,
or PRESS/recall/full-text/feature gates remain unresolved.

## Secrets and read-only boundary

`OPENALEX_API_KEY` or comma-separated `OPENALEX_API_KEYS` is read from the
process environment or the workspace `.env` through a literal,
non-executable parser. Keys are never written to URLs in logs, SQLite,
manifests, or reports.

The hashes of the v2 database, evidence table, and indicator catalog are
registered at initialization and rechecked during every audit. Any mutation
is reported as a blocker.
