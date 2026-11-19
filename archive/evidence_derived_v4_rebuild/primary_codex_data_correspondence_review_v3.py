from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from common import sha256_file, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "human_tasks"
    / "formal_terminal_data_correspondence_AI_v3.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "independent_codex_review_v3"
    / "formal_terminal_data_correspondence_AI_REVIEWED_v3.csv"
)
PROTOCOL = ROOT / "PRIMARY_CODEX_DATA_CORRESPONDENCE_PROTOCOL_V3.json"
REVIEWER_ID = "primary_codex_data_correspondence_v3"
MODEL = "codex_configured_default"
MODEL_DIGEST = "codex-thread:019fa728-bf6c-7453-9af8-9ade78756aae"
COMPLETED_AT = "2026-07-30T08:40:00+00:00"
RUN_ID = "primary_codex_data_correspondence_v3_20260730T084000Z"
PROVENANCE_FIELDS = (
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
)


def _candidate(
    sources: List[str],
    columns: List[str],
    derivation: str,
    equivalence: str,
) -> Dict[str, Any]:
    """Create a formula-completion candidate mapping."""
    return {
        "match_decision": "candidate_formula_completion",
        "local_source_ids_json": json.dumps(sources),
        "local_columns_json": json.dumps(columns),
        "derivation_description": derivation,
        "construct_equivalence_notes": equivalence,
        "reason": (
            "Frozen T0 inputs plausibly support this existing family, but "
            "the current literature fields do not yet establish exact "
            "formula/statistic equivalence. Send to targeted English "
            "primary/foundational formula completion before approval."
        ),
    }


def _candidate_mappings() -> Dict[str, Dict[str, Any]]:
    """Enumerate every plausible correspondence found in the T0 inventory."""
    reference_fields = [
        "paper_references:paper_id",
        "paper_references:reference_id",
        "reference_metadata:reference_id",
        "reference_metadata:field_id",
    ]
    return {
        "EF0031": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:primary_topic",
                "papers_common:display_topic_label",
            ],
            "Use the frozen OpenAlex-derived primary topic label/code.",
            "Requires confirmation that the source's article-topic object "
            "is the same single primary-topic classification.",
        ),
        "EF0033": _candidate(
            ["target_openalex_metadata"],
            [
                "target_openalex_metadata:paper_id",
                "target_openalex_metadata:openalex_country_count",
            ],
            "Use the number of distinct author-affiliation countries.",
            "A count is only one country-diversity operationalization; the "
            "source may instead require shares, entropy, or identities.",
        ),
        "EF0038": _candidate(
            ["target_openalex_metadata"],
            [
                "target_openalex_metadata:paper_id",
                "target_openalex_metadata:openalex_author_count",
            ],
            "Use the publication-time number of distinct listed authors.",
            "Exactness requires the source to define author/team size as a "
            "paper-level count with compatible group-authorship handling.",
        ),
        "EF0052": {
            "match_decision": "exact_derivable",
            "local_source_ids_json": json.dumps(
                ["papers_common", "paper_references", "reference_metadata"]
            ),
            "local_columns_json": json.dumps(
                [
                    "papers_common:paper_id",
                    "papers_common:publication_year",
                    "paper_references:paper_id",
                    "paper_references:reference_id",
                    "reference_metadata:reference_id",
                    "reference_metadata:reference_year",
                ]
            ),
            "derivation_description": (
                "Mean over valid backward citations of focal publication "
                "year minus referenced publication year; empty or "
                "zero-valid-reference denominators return missing."
            ),
            "construct_equivalence_notes": (
                "The verified source explicitly defines the same "
                "paper-level mean backward-citation age; deterministic "
                "implementation and edge-case tests are frozen separately."
            ),
            "reason": (
                "Exact source/object/statistic/T0 correspondence is already "
                "verified; final retention still requires independent "
                "operationalization H2 approval and data-quality gates."
            ),
        },
        "EF0070": _candidate(
            ["opportunity_features"],
            [
                "opportunity_features:paper_id",
                "opportunity_features:bc_degree_per_reference_t0",
                "opportunity_features:bc_shared_reference_strength_t0",
                "opportunity_features:bc_component_share_t0",
                "opportunity_features:bc_local_clustering_t0",
                "opportunity_features:bc_harmonic_closeness_t0",
            ],
            "Use focal position in the strictly prior bibliographic-coupling "
            "paper network.",
            "The source must use the same citation-network node/edge type; "
            "coauthorship or future-citation neighborhoods are mismatches.",
        ),
        "EF0073": _candidate(
            ["control_features", "papers_common", "paper_references",
             "reference_metadata"],
            [
                "control_features:paper_id",
                "control_features:reference_age_median",
                "control_features:reference_age_iqr",
                "papers_common:publication_year",
                "paper_references:paper_id",
                "paper_references:reference_id",
                "reference_metadata:reference_year",
            ],
            "Measure the paper's backward-reference age distribution.",
            "Source completion must identify mean, median, tail, or another "
            "specific recency statistic; these variants are not identical.",
        ),
        "EF0074": _candidate(
            ["paper_references", "reference_metadata"],
            reference_fields,
            "Build the distribution of cited references over frozen fields.",
            "Requires the same field taxonomy, counting unit, fractional "
            "assignment, and aggregation rule as the source.",
        ),
        "EF0075": _candidate(
            ["control_features"],
            [
                "control_features:paper_id",
                "control_features:log_prior_reference_popularity_median",
            ],
            "Use the frozen median prior popularity of cited references.",
            "The source must define prior impact using the same T0-bounded "
            "citation statistic and aggregation; lifetime impact is invalid.",
        ),
        "EF0086": _candidate(
            ["target_openalex_metadata"],
            [
                "target_openalex_metadata:paper_id",
                "target_openalex_metadata:openalex_author_count",
                "target_openalex_metadata:openalex_country_count",
                "target_openalex_metadata:openalex_institution_count",
            ],
            "Derive collaboration categories from paper-level team, "
            "institution, and country counts.",
            "The source's collaboration-type categories and thresholds must "
            "be recovered before the encoding can be called identical.",
        ),
        "EF0115": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_variety",
                "innovation_candidate_features:field_shannon_entropy",
                "innovation_candidate_features:field_pielou_evenness",
                "innovation_candidate_features:field_disparity_cosine_mean",
                "innovation_candidate_features:rao_stirling_integration",
            ],
            "Characterize cited-field variety, balance, and disparity.",
            "The family is broad; exact formula completion must identify the "
            "specific component or composite rather than authorize all.",
        ),
        "EF0117": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_div_index",
            ],
            "Use the frozen multiplicative DIV integration indicator.",
            "Must confirm the same relative-variety, balance, disparity "
            "components and normalization; additive dive is not equivalent.",
        ),
        "EF0118": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:document_type",
                "papers_common:work_type",
            ],
            "Use the frozen publication-time document/work-type category.",
            "Exactness requires the source's category vocabulary to map "
            "deterministically to the frozen OpenAlex/document taxonomy.",
        ),
        "EF0129": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_shannon_entropy",
                "innovation_candidate_features:field_pielou_evenness",
            ],
            "Use cited-field Shannon entropy or its normalized evenness.",
            "Formula completion must distinguish raw entropy from normalized "
            "Pielou evenness and identify zero/single-category behavior.",
        ),
        "EF0130": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_variety",
                "innovation_candidate_features:field_relative_variety",
            ],
            "Count occupied cited-reference fields or normalize by the "
            "frozen field universe.",
            "Raw variety, relative variety, and entropy-of-variety are "
            "different statistics and require source-specific resolution.",
        ),
        "EF0148": _candidate(
            ["papers_common", "control_features"],
            [
                "papers_common:paper_id",
                "papers_common:publication_year",
                "papers_common:domain12",
                "papers_common:openalex_primary_subfield",
                "control_features:venue_family",
            ],
            "Construct frozen field-by-year (optionally venue) strata.",
            "The source's normalization population and field taxonomy must "
            "match; the inventory only proves the required context exists.",
        ),
        "EF0186": _candidate(
            ["target_openalex_metadata"],
            [
                "target_openalex_metadata:paper_id",
                "target_openalex_metadata:openalex_country_count",
            ],
            "Define international collaboration as more than one distinct "
            "author-affiliation country, subject to source definition.",
            "Country count supports a binary cross-country indicator but not "
            "leadership, shares, or named-country composition.",
        ),
        "EF0188": _candidate(
            ["target_openalex_metadata"],
            [
                "target_openalex_metadata:paper_id",
                "target_openalex_metadata:openalex_country_count",
            ],
            "Use the number of distinct author-affiliation countries.",
            "The source must define international scale by the same count "
            "rather than author share, distance, or institution count.",
        ),
        "EF0196": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:source_id",
                "papers_common:primary_field",
                "papers_common:openalex_primary_field",
                "papers_common:openalex_primary_subfield",
            ],
            "Represent the venue's frozen field/subfield scope.",
            "A source-specific interdisciplinary-journal or breadth measure "
            "may need multiple fields unavailable in this one-label schema.",
        ),
        "EF0197": _candidate(
            ["papers_common", "control_features"],
            [
                "papers_common:paper_id",
                "papers_common:source_id",
                "papers_common:source_display_name",
                "control_features:venue_family",
            ],
            "Use the frozen source identifier/name or venue-family category.",
            "The source's journal identity encoding must be specified; "
            "identity is a context control, not intrinsic paper quality.",
        ),
        "EF0204": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:journal_family",
                "papers_common:venue_family",
                "papers_common:source_id",
            ],
            "Use the frozen journal/venue-family category.",
            "The source's journal-type taxonomy and reference group must be "
            "recovered before exact encoding.",
        ),
        "EF0205": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:venue_family",
                "papers_common:source_id",
            ],
            "Use the frozen venue-family/source category.",
            "The broad family needs a source-defined venue taxonomy; it "
            "cannot absorb journal prestige or impact.",
        ),
        "EF0209": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_variety",
                "innovation_candidate_features:field_gini_balance",
                "innovation_candidate_features:field_shannon_entropy",
                "innovation_candidate_features:field_disparity_cosine_mean",
                "innovation_candidate_features:rao_stirling_integration",
                "innovation_candidate_features:field_div_index",
            ],
            "Measure interdisciplinarity of the focal cited knowledge base.",
            "Formula completion must resolve the particular component or "
            "composite and its field-distance construction.",
        ),
        "EF0211": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:uzzi_conventionality_median_t0",
                "innovation_candidate_features:hypergeom_conventionality_median_t0",
                "innovation_candidate_features:source_pair_mean_surprisal",
            ],
            "Summarize strictly prior cited-source-pair conventionality.",
            "Uzzi Monte Carlo, hypergeometric, and empirical-surprisal "
            "statistics are different parameterizations requiring an exact "
            "source match.",
        ),
        "EF0232": _candidate(
            ["target_openalex_metadata"],
            [
                "target_openalex_metadata:paper_id",
                "target_openalex_metadata:openalex_institution_count",
            ],
            "Define multi-institutional collaboration from the distinct "
            "institution count, typically count greater than one.",
            "The source must define institution deduplication and binary or "
            "count encoding; multicenter clinical design is not assumed.",
        ),
        "EF0238": _candidate(
            ["opportunity_features"],
            [
                "opportunity_features:paper_id",
                "opportunity_features:bc_degree_per_reference_t0",
                "opportunity_features:bc_harmonic_closeness_t0",
            ],
            "Use degree or harmonic closeness in the strictly prior "
            "bibliographic-coupling paper network.",
            "Generic network centrality is insufficient: node, edge, "
            "normalization, and pre-T0 graph must match the source.",
        ),
        "EF0239": _candidate(
            ["opportunity_features"],
            [
                "opportunity_features:paper_id",
                "opportunity_features:bc_component_share_t0",
                "opportunity_features:eligible_prior_paper_count",
            ],
            "Use component share in the strictly prior bibliographic-"
            "coupling paper graph.",
            "The source must concern the same paper graph; a scientist "
            "collaboration giant component is a construct mismatch.",
        ),
        "EF0257": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_variety",
                "innovation_candidate_features:field_gini_balance",
                "innovation_candidate_features:field_disparity_cosine_mean",
                "innovation_candidate_features:rao_stirling_integration",
                "innovation_candidate_features:field_div_index",
            ],
            "Measure paper-level cited-field interdisciplinarity.",
            "The umbrella family requires source resolution to a specific "
            "component/composite; topical and author interdisciplinarity "
            "are not interchangeable.",
        ),
        "EF0258": _candidate(
            ["opportunity_features"],
            [
                "opportunity_features:paper_id",
                "opportunity_features:bc_degree_per_reference_t0",
                "opportunity_features:bc_shared_reference_strength_t0",
                "opportunity_features:bc_component_share_t0",
                "opportunity_features:bc_local_clustering_t0",
                "opportunity_features:bc_harmonic_closeness_t0",
            ],
            "Use focal structural position in a strictly prior paper "
            "knowledge-overlap network.",
            "The source must define the same bibliographic-coupling graph "
            "and statistic; future citation networks are prohibited.",
        ),
        "EF0260": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_variety",
                "innovation_candidate_features:field_shannon_entropy",
            ],
            "Measure the breadth/distribution of fields in references.",
            "Multidisciplinarity must be source-defined at the paper's "
            "knowledge-input level, not journal or author level.",
        ),
        "EF0261": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:novelty_u_t0_source",
                "innovation_candidate_features:first_time_source_pair_share",
                "innovation_candidate_features:reference_overlap_novelty_t0",
            ],
            "Use a strictly prior recombinational or reference-overlap "
            "paper novelty statistic.",
            "The broad family cannot approve all novelty constructs; source "
            "completion must identify one exact knowledge unit and formula.",
        ),
        "EF0289": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:uzzi_conventionality_median_t0",
                "innovation_candidate_features:hypergeom_conventionality_median_t0",
                "innovation_candidate_features:source_pair_mean_surprisal",
            ],
            "Measure conventionality of strictly prior cited-source pairs.",
            "The source's null model, pair universe, center statistic, and "
            "time cutoff must match one materialized variant.",
        ),
        "EF0307": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:publication_year",
            ],
            "Use the focal publication year as a context-control value.",
            "Confirm that year is a background control rather than a proxy "
            "for future exposure and define coding/centering.",
        ),
        "EF0309": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:rao_stirling_integration",
            ],
            "Use sum over cited-field pairs of p_k p_l d_kl with frozen "
            "strictly prior cosine distances.",
            "The source formula, ordered/unordered sum, diagonal handling, "
            "taxonomy, and distance normalization must match.",
        ),
        "EF0312": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_gini_balance",
                "innovation_candidate_features:field_pielou_evenness",
                "innovation_candidate_features:field_gini_simpson",
                "innovation_candidate_features:field_hhi",
            ],
            "Measure evenness/concentration of cited-field shares.",
            "Gini balance, Pielou evenness, Simpson, and HHI are different "
            "families unless the source explicitly defines the variant.",
        ),
        "EF0313": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:first_time_source_pair_share",
                "innovation_candidate_features:first_time_source_pair_count",
                "innovation_candidate_features:reference_overlap_novelty_t0",
                "innovation_candidate_features:novelty_u_t0_source",
            ],
            "Measure novelty of the focal reference/source combination "
            "against strictly prior combinations.",
            "First incidence, rarity tail, and reference-set dissimilarity "
            "are distinct and require a source-specific match.",
        ),
        "EF0314": _candidate(
            ["paper_references", "control_features"],
            [
                "paper_references:paper_id",
                "paper_references:reference_id",
                "control_features:paper_id",
                "control_features:log_reference_count",
            ],
            "Count distinct declared backward-reference edges; transform "
            "with log1p only as a separately declared project transform.",
            "The source must define raw number of references and duplicate/"
            "missing-reference handling; raw count and log count are not "
            "separate indicator families.",
        ),
        "EF0315": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_disparity_cosine_mean",
                "innovation_candidate_features:field_disparity_cosine_max",
                "innovation_candidate_features:field_disparity_cosine_p90",
            ],
            "Measure cognitive distance among occupied cited fields using "
            "strictly prior field citation profiles.",
            "Mean, maximum, percentile, weighted/unweighted, and mutual "
            "information disparity are not interchangeable.",
        ),
        "EF0316": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_shannon_entropy",
                "innovation_candidate_features:field_gini_simpson",
                "innovation_candidate_features:field_hill_q0",
                "innovation_candidate_features:field_hill_q1",
                "innovation_candidate_features:field_hill_q2",
                "innovation_candidate_features:rao_true_diversity_q2",
            ],
            "Compute a source-specified cited-field diversity statistic.",
            "Entropy, Simpson/Hill numbers, and Rao diversity are separate "
            "mathematical variants requiring explicit source resolution.",
        ),
        "EF0317": _candidate(
            ["paper_references", "reference_metadata",
             "innovation_candidate_features", "control_features"],
            [
                "paper_references:paper_id",
                "paper_references:reference_id",
                "reference_metadata:reference_year",
                "reference_metadata:field_id",
                "control_features:reference_age_median",
                "control_features:reference_age_iqr",
                "innovation_candidate_features:valid_reference_count",
            ],
            "Derive source-specified size, age, or field-composition "
            "characteristics of each paper's reference list.",
            "This umbrella label is not a formula; only a named statistic "
            "recovered from the source can proceed.",
        ),
        "EF0318": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_variety",
                "innovation_candidate_features:field_relative_variety",
            ],
            "Count occupied cited-reference fields, optionally normalized "
            "by the frozen eligible field universe.",
            "Raw and relative variety require separate parameter evidence; "
            "a count is not entropy-of-variety.",
        ),
        "EF0327": _candidate(
            ["target_openalex_metadata"],
            [
                "target_openalex_metadata:paper_id",
                "target_openalex_metadata:openalex_author_count",
                "target_openalex_metadata:openalex_institution_count",
                "target_openalex_metadata:openalex_country_count",
            ],
            "Derive a source-specified paper-level collaboration indicator "
            "from team, institution, and country counts.",
            "The broad label requires exact source categories; counts do "
            "not reconstruct prior collaboration networks.",
        ),
        "EF0328": _candidate(
            ["opportunity_features"],
            [
                "opportunity_features:paper_id",
                "opportunity_features:eligible_prior_paper_count",
            ],
            "Use the number of eligible strictly prior papers in the frozen "
            "comparison/network universe.",
            "The source's research-community population and field boundary "
            "must match; this is not author-community size.",
        ),
        "EF0331": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:domain12",
                "papers_common:openalex_primary_field",
                "papers_common:openalex_primary_subfield",
            ],
            "Use the frozen paper field/domain classification.",
            "The source's field taxonomy and granularity must map exactly; "
            "field is a context/opportunity variable.",
        ),
        "EF0373": _candidate(
            ["innovation_candidate_features"],
            [
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:novelty_u_t0_source",
                "innovation_candidate_features:first_time_source_pair_share",
                "innovation_candidate_features:reference_overlap_novelty_t0",
            ],
            "Use one source-defined strictly prior paper novelty measure.",
            "Scientific novelty is an umbrella construct; formula completion "
            "must identify one exact observable rather than infer a score.",
        ),
        "EF0403": _candidate(
            ["papers_common"],
            [
                "papers_common:paper_id",
                "papers_common:primary_field",
                "papers_common:openalex_primary_field",
                "papers_common:openalex_primary_subfield",
                "papers_common:domain12",
            ],
            "Use a frozen subject/field classification code.",
            "The source classification scheme and crosswalk must be "
            "specified before categorical equivalence.",
        ),
        "EF0419": _candidate(
            ["control_features"],
            [
                "control_features:paper_id",
                "control_features:title_word_count",
            ],
            "Use the number of whitespace/tokenized title words.",
            "Formula completion must confirm that source title length is in "
            "words rather than characters, bytes, or another tokenizer.",
        ),
    }


def _construct_mismatches() -> Dict[str, str]:
    """Identify salient near-name mappings that must not be substituted."""
    return {
        "EF0017": (
            "The local field_div_index is multiplicative and cited-field "
            "based, while additive dive requires entropy variety plus "
            "entropy balance minus mutual information from a source-"
            "compatible joint categorical distribution."
        ),
        "EF0082": (
            "Available network columns describe a prior paper "
            "bibliographic-coupling graph, not the scientist collaboration "
            "network required for algebraic connectivity."
        ),
        "EF0083": (
            "bc_local_clustering_t0 is a focal-paper coefficient in a "
            "bibliographic-coupling graph, not a scientist-node coefficient "
            "in the source's pre-T0 collaboration network."
        ),
        "EF0084": (
            "No eigenvector centrality is inventoried, and the available "
            "graph is a paper bibliographic-coupling graph rather than the "
            "source's scientist collaboration graph."
        ),
        "EF0085": (
            "No source-equivalent scientist collaboration knowledge-density "
            "measure is present; bibliographic-coupling strength is a "
            "different edge and construct."
        ),
        "EF0190": (
            "Only distinct-country count is inventoried; it cannot recover "
            "named-country composition, shares, or trial-management-team "
            "nationalities."
        ),
        "EF0234": (
            "Distinct OpenAlex institution count does not identify clinical "
            "study centres or multicentre design breadth."
        ),
        "EF0235": (
            "The inventory lacks the joint categorical random-variable "
            "cross-tabulation needed for mutual-information disparity; "
            "cosine field distance is a different statistic."
        ),
        "EF0240": (
            "The source identifies top n-gram births using subsequent "
            "corpus-wide use and field-period aggregation; no paper-level "
            "T0-equivalent text history is inventoried."
        ),
        "EF0288": (
            "log_team_prior_nature_output_max is the maximum prior Nature "
            "output among team members, not a generic focal-author prior "
            "publication count."
        ),
        "EF0319": (
            "The source requires eigenvalues of a pre-T0 scientist "
            "collaboration Laplacian; no such author-history graph is "
            "inventoried."
        ),
        "EF0423": (
            "Inventoried diversity features use cited-reference fields, "
            "whereas topic interdisciplinarity is measured from paper "
            "content/topics."
        ),
        "EF0424": (
            "No source-equivalent content-topic interdisciplinarity measure "
            "or complete title/abstract topic mixture is inventoried."
        ),
        "EF0427": (
            "A single primary-topic label and cited-field diversity do not "
            "equal within-paper topical diversity."
        ),
    }


def review_rows(input_path: Path) -> List[Dict[str, Any]]:
    """Apply the outcome-blind complete-inventory correspondence review."""
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    mappings = _candidate_mappings()
    mismatches = _construct_mismatches()
    observed = {str(row["feature_id"]) for row in rows}
    unknown = sorted((set(mappings) | set(mismatches)) - observed)
    if unknown:
        raise ValueError(f"Unknown family IDs in review rules: {unknown}")
    prompt_hash = sha256_file(PROTOCOL)
    reviewed: List[Dict[str, Any]] = []
    for row in rows:
        feature_id = str(row["feature_id"])
        if feature_id in mappings:
            row.update(mappings[feature_id])
        elif feature_id in mismatches:
            row.update(
                {
                    "match_decision": "construct_mismatch",
                    "reason": mismatches[feature_id],
                }
            )
        else:
            row.update(
                {
                    "match_decision": "no_match",
                    "reason": (
                        "The complete frozen T0 inventory contains no "
                        "column or deterministic input set for the measured "
                        "object and statistic represented by this family; "
                        "no similarly named field was substituted."
                    ),
                }
            )
        row.update(
            {
                "draft_method": (
                    "primary_codex_complete_local_inventory_review"
                ),
                "independent_ai_review_status": "complete",
                "independent_ai_reviewer_id": REVIEWER_ID,
                "independent_ai_reviewed_at": COMPLETED_AT,
                "independent_ai_review_action": row["match_decision"],
                "independent_ai_run_id": RUN_ID,
                "independent_ai_model": MODEL,
                "independent_ai_prompt_sha256": prompt_hash,
            }
        )
        reviewed.append(row)
    return reviewed


def main() -> None:
    """Write all 432 reviewed rows and an exact-hash manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    rows = review_rows(input_path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        fields = list(csv.DictReader(handle).fieldnames or [])
    fields.extend(
        field for field in PROVENANCE_FIELDS if field not in fields
    )
    write_csv(output_path, rows, fields)
    counts: Dict[str, int] = {}
    for row in rows:
        decision = str(row["match_decision"])
        counts[decision] = counts.get(decision, 0) + 1
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "run_id": RUN_ID,
        "action": "complete_local_t0_data_correspondence_review",
        "artifact_path": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "reviewer_role": "AI",
        "reviewer_id": REVIEWER_ID,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "protocol_path": str(PROTOCOL),
        "prompt_sha256": sha256_file(PROTOCOL),
        "parameters": {
            "rows": len(rows),
            "decision_counts": counts,
            "all_families_reviewed": True,
            "target_count_influence": False,
            "model_outcomes_used": False,
            "qwen_or_ollama_used": False,
            "prohibited_sources_used": False,
        },
        "item_count": len(rows),
        "completed_at": COMPLETED_AT,
        "status": "complete",
        "target_count_influence": False,
        "qwen_or_ollama_used": False,
        "prohibited_sources_used": False,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "artifact_path": str(output_path),
                "artifact_sha256": sha256_file(output_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "decision_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
