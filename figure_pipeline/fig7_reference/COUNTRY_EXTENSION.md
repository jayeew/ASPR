# Fig7 country extension

This independent extension appends panels c–d to the accepted two-panel SVG.
The current canvas is 1400 × 1374; panels e and f were removed from the figure.
Their statistical tables remain available as supporting data.
It reads the existing claims, phrase mappings and saved prospective directions;
it does not invoke prediction, extraction, graph insertion or word-cloud layout.

The publication environment used for this run is
`/home/jayee/miniconda3/envs/openrlhf/bin/python`.
Additional pinned plotting dependencies are in `country-requirements.in`.

```bash
python -m figure_pipeline.fig7_reference.country_extension audit
python -m figure_pipeline.fig7_reference.country_extension fetch --workers 4
python -m figure_pipeline.fig7_reference.country_extension aggregate
python -m figure_pipeline.fig7_reference.country_extension render
python -m pytest -q tests/innovation_v2/test_fig7_country_extension.py
```

Default output: `outputs/fig7_reference/country_extension/`. `--output-dir`
can isolate another run. Metadata requests use the configured `OPENALEX_API_KEY`
or first `OPENALEX_API_KEYS` entry. Keys are not included in saved requests.
Raw batch and single-work responses are reusable checkpoints; unsuccessful
requests remain in `errors/` and are retried on resume. Do not remove raw records.

`audit` fixes the 25,023-paper inventory and records hashes of the original
two-panel assets. `fetch` uses paper IDs, then DOI for misses, and retrieves
single-work responses for lists with at least 100 authors. The observed API
does not accept `authors_count` in `select` and does not reliably return a
truncation flag; the conservative 100-author check is therefore required.

Countries are publication-time institutional countries/regions, not citizenship.
Missing country mappings can be repaired only when every raw address of the
affected author has an explicit country/region suffix and matches the local paper
after Unicode, whitespace and punctuation normalization. Original character spans,
raw addresses and codes are retained. A metadata country counter explained fully
by these verified added addresses is recorded as corrected, not silently ignored.
Other missingness and conflicts remain unresolved. Optional complete local
affiliation-block reviews live in `data/local_verifications.json` and require
actual verbatim source quotes and an explicit completeness assertion.

`aggregate` saves all-country tables and the exact chart data. Primary c–e
statistics exclude unresolved papers; f retains their mass in Unresolved.
Country weighting is 1/k per paper, not author weighting. The strict historical
coverage gate is at least 95% overall and in each year. The renderer labels
below-threshold output DRAFT and still exports actual data.

The top 12 countries are selected by historical fractional paper mass. Historical
topics are the first 16 displayed phrases by original frequency. Future labels are
selected by saved prominence and label order, deduplicating identical direction-ID
sets before taking 12. Topic choices do not depend on country distributions.

The generated `methods_and_caption.md` contains bilingual definitions and caveats.
`qa/` records original asset hashes, coverage, conservation tests, and the
same-renderer pixel comparison of the original upper 1400 × 890 region.
