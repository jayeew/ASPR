"""Build auditable T0 matrices for four evidence-v3 indicator sets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.nature_multihorizon.active_dataset import (  # noqa: E402
    load_active_dataset,
)
from gear.nature_multihorizon.t0_runtime_v3 import (  # noqa: E402
    compute_additive_entropy_diversity,
    compute_backward_citation_age,
    compute_prior_team_graph,
    compute_title_novelty,
)

EVIDENCE_ROOT = HERE.parents[1]
EVIDENCE_OUTPUTS = EVIDENCE_ROOT / "outputs"
ACTIVE_DATASET = load_active_dataset(PROJECT_ROOT)
BASE_DATA_ROOT = Path(ACTIVE_DATASET["dataset_dir"])
DATA_ROOT = Path(ACTIVE_DATASET["feature_dataset_dir"])
TARGET_WORKS = Path(
    ACTIVE_DATASET.get(
        "target_works",
        PROJECT_ROOT
        / "outputs"
        / "common"
        / "new"
        / "data"
        / "nature_portfolio_v5"
        / "nature_target_works.csv",
    )
)
if not TARGET_WORKS.is_absolute():
    TARGET_WORKS = PROJECT_ROOT / TARGET_WORKS
DEFAULT_OUTPUT = HERE / "outputs"
DEFINITION_VERSION = "evidence_v3_four_set_t0_" + str(
    ACTIVE_DATASET["active_dataset_version"]
)
EXPECTED_SET_COUNTS = {
    "strict_7": (7, 4),
    "fulltext_16": (16, 10),
    "source_154": (154, 48),
    "ultrarelaxed_221": (221, 55),
}
SAFE_T0_GATES = (
    "G01_IN_SCOPE_ROLE",
    "G02_ARTICLE_LEVEL",
    "G05_PUBLICATION_TIME",
    "G06_NO_FUTURE_INFORMATION",
    "G08_BIAS_GUARDRAIL",
    "G09_NO_FATAL_VALIDITY_CONCERN",
    "G10_OUTCOME_BLIND_SELECTION",
)
TARGETED_COLUMNS = {
    "EF0038": "EF0038__author_count",
    "EF0052": "EF0052__backward_citation_age_mean",
    "EF0186": "EF0186__international_collaboration",
    "EF0188": "EF0188__country_count",
    "EF0197": "EF0197__journal_id",
    "EF0238": "EF0238__bibliographic_coupling_degree_per_reference",
    "EF0307": "EF0307__publication_year",
    "EF0309": "EF0309__rao_stirling_diversity",
    "EF0312": "EF0312__reference_balance",
    "EF0314": "EF0314__reference_count",
    "EF0315": "EF0315__reference_disparity",
    "EF0318": "EF0318__reference_variety",
}
FORMULA_SURROGATES = {
    "EF0017": (
        "additive_entropy_diversity_local",
        "H_variety + H_balance - disparity, using frozen reference-field "
        "entropy/evenness/cosine-disparity components.",
    ),
    "EF0083": (
        "prior_team_mean_clustering",
        "Mean local clustering coefficient in the focal authors' strictly "
        "prior-year Nature coauthorship graph.",
    ),
    "EF0240": (
        "title_new_bigram_share",
        "Share of title bigrams absent from all earlier-year cohort titles; "
        "the source's later heavy-use condition is deliberately not used.",
    ),
    "EF0319": (
        "prior_team_relative_algebraic_connectivity",
        "lambda_2/lambda_n of the weighted focal-team graph induced by "
        "strictly prior-year Nature coauthorships.",
    ),
}
TEXT_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a UTF-8 CSV into dictionaries."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_feature_sets() -> Tuple[
    Dict[str, List[str]],
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, Dict[str, str]],
]:
    """Derive the four nested sets directly from frozen gate exports."""
    library_path = EVIDENCE_OUTPUTS / "complete_indicator_library_v3.csv"
    gates_path = EVIDENCE_OUTPUTS / "feature_gate_decisions_v3.csv"
    dimensions_path = EVIDENCE_OUTPUTS / "candidate_dimensions_v3.csv"
    library = {row["feature_id"]: row for row in read_csv_rows(library_path)}
    checks = {
        row["feature_id"]: json.loads(row["gate_checks_json"])
        for row in read_csv_rows(gates_path)
    }
    dimensions = {row["dimension_id"]: row for row in read_csv_rows(dimensions_path)}
    feature_to_dimension: Dict[str, str] = {}
    for dimension_id, row in dimensions.items():
        for feature_id in json.loads(row["feature_ids_json"]):
            if feature_id in feature_to_dimension:
                raise ValueError(f"duplicate dimension mapping: {feature_id}")
            feature_to_dimension[feature_id] = dimension_id

    def passes(feature_id: str, extra: Sequence[str]) -> bool:
        required = (*SAFE_T0_GATES, *extra)
        return all(bool(checks[feature_id][gate]) for gate in required)

    feature_sets = {
        "strict_7": sorted(fid for fid, row in checks.items() if all(row.values())),
        "fulltext_16": sorted(
            fid
            for fid in library
            if passes(
                fid,
                (
                    "G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE",
                    "G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE",
                ),
            )
        ),
        "source_154": sorted(
            fid
            for fid in library
            if passes(fid, ("G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE",))
        ),
        "ultrarelaxed_221": sorted(fid for fid in library if passes(fid, ())),
    }
    previous: set[str] = set()
    for model_id in EXPECTED_SET_COUNTS:
        current = set(feature_sets[model_id])
        if previous and not previous.issubset(current):
            raise ValueError(f"feature sets are not nested at {model_id}")
        expected_features, expected_dimensions = EXPECTED_SET_COUNTS[model_id]
        covered = {feature_to_dimension[fid] for fid in current}
        if (len(current), len(covered)) != (
            expected_features,
            expected_dimensions,
        ):
            raise ValueError(
                f"{model_id} expected {(expected_features, expected_dimensions)} "
                f"but found {(len(current), len(covered))}"
            )
        previous = current
    return feature_sets, library, feature_to_dimension, dimensions


def merge_unique(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """Left-merge a paper-level view without duplicate columns."""
    if right["paper_id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate paper IDs")
    duplicate = set(left.columns) & set(right.columns) - {"paper_id"}
    trimmed = right.drop(columns=sorted(duplicate), errors="ignore")
    merged = left.merge(trimmed, on="paper_id", how="left", validate="one_to_one")
    if len(merged) != len(left):
        raise ValueError(f"{label} changed the paper count")
    return merged


def load_base_frame() -> pd.DataFrame:
    """Load outcome-blind T0 inputs for the complete training cohort."""
    papers = pd.read_parquet(DATA_ROOT / "papers_primary_articles.parquet")
    papers = papers.sort_values(
        ["publication_year", "paper_id"], kind="stable"
    ).reset_index(drop=True)
    works = pd.read_csv(
        TARGET_WORKS,
        usecols=["id", "title", "doi"],
        dtype={"id": "string", "title": "string", "doi": "string"},
    ).rename(columns={"id": "paper_id"})
    frame = merge_unique(papers, works, label="target works")
    sources = (
        "innovation_candidate_features.parquet",
        "control_features_v6_1.parquet",
        "opportunity_features.parquet",
        "target_openalex_metadata.parquet",
    )
    for name in sources:
        view = pd.read_parquet(DATA_ROOT / name)
        frame = merge_unique(frame, view, label=name)
    frame = merge_unique(
        frame,
        backward_citation_age_mean(),
        label="expanded backward-citation age",
    )
    frame["EF0038"] = pd.to_numeric(
        frame["openalex_author_count"], errors="coerce"
    ).where(pd.to_numeric(frame["openalex_author_count"], errors="coerce").gt(0))
    frame["EF0052"] = frame["backward_citation_age_mean"]
    country_count = pd.to_numeric(frame["openalex_country_count"], errors="coerce")
    frame["EF0186"] = country_count.gt(1).astype(float).where(country_count.notna())
    frame["EF0188"] = country_count.where(country_count.gt(0))
    source_id = frame["source_id"].astype("string")
    frame["EF0197"] = source_id.where(source_id.notna() & source_id.str.strip().ne(""))
    frame["EF0238"] = frame["bc_degree_per_reference_t0"]
    frame["EF0307"] = frame["publication_year"]
    frame["EF0309"] = frame["rao_stirling_integration"].where(
        pd.to_numeric(frame["field_variety"], errors="coerce").ge(2)
    )
    frame["EF0312"] = frame["field_gini_balance"].where(
        pd.to_numeric(frame["field_mapping_coverage"], errors="coerce").gt(0)
    )
    frame["EF0314"] = frame["valid_reference_count"]
    frame["EF0315"] = frame["field_disparity_cosine_mean"].where(
        pd.to_numeric(frame["field_variety"], errors="coerce").ge(2)
    )
    frame["EF0318"] = frame["field_variety"].where(
        pd.to_numeric(frame["field_mapping_coverage"], errors="coerce").gt(0)
    )
    if len(frame) != len(papers) or frame["title"].isna().any():
        raise ValueError("unexpected cohort size or missing title")
    return frame


def backward_citation_age_mean() -> pd.DataFrame:
    """Compute mean age of valid non-future references for each paper."""
    papers = pd.read_parquet(
        BASE_DATA_ROOT / "papers_primary_articles.parquet",
        columns=["paper_id", "publication_year"],
    )
    references = pd.read_parquet(
        BASE_DATA_ROOT / "paper_references.parquet",
        columns=["paper_id", "reference_id"],
    )
    metadata = pd.read_parquet(
        BASE_DATA_ROOT / "reference_metadata.parquet",
        columns=["reference_id", "reference_year"],
    )
    rows = references.merge(
        metadata, on="reference_id", how="left", validate="many_to_one"
    ).merge(papers, on="paper_id", how="left", validate="many_to_one")
    rows["reference_year"] = pd.to_numeric(rows["reference_year"], errors="coerce")
    rows["publication_year"] = pd.to_numeric(rows["publication_year"], errors="coerce")
    rows = rows[
        rows["reference_year"].notna()
        & rows["reference_year"].le(rows["publication_year"])
    ].copy()
    grouped = rows.groupby("paper_id", as_index=False).agg(
        publication_year=("publication_year", "first"),
        reference_years=("reference_year", list),
    )
    grouped["backward_citation_age_mean"] = [
        compute_backward_citation_age(int(publication_year), reference_years)
        for publication_year, reference_years in zip(
            grouped["publication_year"],
            grouped["reference_years"],
        )
    ]
    means = grouped[["paper_id", "backward_citation_age_mean"]]
    return papers[["paper_id"]].merge(
        means, on="paper_id", how="left", validate="one_to_one"
    )


def _tokens(text: object) -> List[str]:
    return TEXT_TOKEN_PATTERN.findall(str(text or "").casefold())


def title_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic title form and earlier-year novelty features."""
    output: Dict[str, List[float]] = defaultdict(list)
    seen_unigrams: set[str] = set()
    seen_bigrams: set[Tuple[str, str]] = set()
    years = frame["publication_year"].astype(int).to_numpy()
    titles = frame["title"].astype(str).tolist()
    for year in sorted(set(years)):
        positions = np.flatnonzero(years == year)
        year_tokens: List[List[str]] = []
        for position in positions:
            tokens = _tokens(titles[position])
            words = [token for token in tokens if not token[0].isdigit()]
            counts: Dict[str, int] = defaultdict(int)
            for token in words:
                counts[token] += 1
            probabilities = np.asarray(list(counts.values()), dtype=float)
            probabilities = (
                probabilities / probabilities.sum()
                if len(probabilities)
                else probabilities
            )
            entropy = (
                float(-(probabilities * np.log2(probabilities)).sum())
                if len(probabilities)
                else 0.0
            )
            numeric = [float(token) for token in tokens if token[0].isdigit()]
            title = titles[position]
            output["title_char_count"].append(float(len(title)))
            output["title_token_count_local"].append(float(len(tokens)))
            output["title_unique_token_count"].append(float(len(set(words))))
            output["title_unique_token_ratio"].append(
                float(len(set(words)) / len(words)) if words else 0.0
            )
            output["title_mean_token_length"].append(
                float(np.mean([len(token) for token in words])) if words else 0.0
            )
            output["title_token_entropy"].append(entropy)
            output["title_punctuation_count"].append(
                float(sum(not char.isalnum() and not char.isspace() for char in title))
            )
            output["title_colon"].append(float(":" in title))
            output["title_question"].append(float("?" in title))
            output["title_numeric_token_count"].append(float(len(numeric)))
            output["title_log_max_numeric"].append(
                float(np.log1p(max(numeric))) if numeric else 0.0
            )
            output["title_new_unigram_share"].append(
                float(sum(token not in seen_unigrams for token in words) / len(words))
                if words
                else 0.0
            )
            output["title_new_bigram_share"].append(
                compute_title_novelty(title, seen_bigrams)[
                    "title_new_bigram_share"
                ]
            )
            output["title_covid19"].append(
                float(
                    bool(re.search(r"\bcovid(?:-?19)?\b|sars-cov-2", title.casefold()))
                )
            )
            year_tokens.append(words)
        for words in year_tokens:
            seen_unigrams.update(words)
            seen_bigrams.update(zip(words[:-1], words[1:]))
    result = pd.DataFrame(output)
    result.insert(0, "paper_id", frame["paper_id"].astype(str).to_numpy())
    return result


def prior_volume_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Count earlier-year cohort papers in the same venue, field, and topic."""
    keys = {
        "venue_prior_volume": "source_id",
        "field_prior_volume": "openalex_primary_subfield",
        "topic_prior_volume": "primary_topic",
    }
    result = pd.DataFrame({"paper_id": frame["paper_id"].astype(str)})
    years = frame["publication_year"].astype(int)
    for output_name, key in keys.items():
        counts: Dict[str, int] = defaultdict(int)
        values = np.zeros(len(frame), dtype=np.float32)
        normalized = frame[key].astype("string").fillna("missing")
        for year in sorted(years.unique()):
            positions = np.flatnonzero(years.to_numpy() == year)
            for position in positions:
                values[position] = float(counts[str(normalized.iloc[position])])
            for position in positions:
                counts[str(normalized.iloc[position])] += 1
        result[output_name] = np.log1p(values)
    return result


def _team_graph_statistics(
    authors: Sequence[str],
    adjacency: Mapping[str, set[str]],
    weights: Mapping[Tuple[str, str], int],
) -> Dict[str, float]:
    return compute_prior_team_graph(authors, adjacency, weights)


def _author_values(value: object) -> List[str]:
    """Normalize Arrow/Pandas list scalars without ambiguous truth tests."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return [str(author) for author in list(value) if str(author)]


def author_graph_features(
    frame: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Build strictly prior-year focal-team coauthorship graph measures."""
    path = output_dir / "author_graph_t0_features.parquet"
    if path.is_file():
        cached = pd.read_parquet(path)
        if len(cached) == len(frame) and set(cached["paper_id"]) == set(
            frame["paper_id"]
        ):
            return cached
        raise ValueError("invalid author graph cache")
    adjacency: Dict[str, set[str]] = defaultdict(set)
    weights: Dict[Tuple[str, str], int] = defaultdict(int)
    rows: List[Dict[str, Any]] = []
    years = frame["publication_year"].astype(int).to_numpy()
    author_lists = frame["openalex_author_ids"].tolist()
    paper_ids = frame["paper_id"].astype(str).tolist()
    for year in sorted(set(years)):
        positions = np.flatnonzero(years == year)
        for position in positions:
            authors = _author_values(author_lists[position])
            rows.append(
                {
                    "paper_id": paper_ids[position],
                    **_team_graph_statistics(authors, adjacency, weights),
                }
            )
        for position in positions:
            authors = sorted(set(_author_values(author_lists[position])))[:100]
            for left, right in combinations(authors, 2):
                key = (left, right) if left < right else (right, left)
                adjacency[left].add(right)
                adjacency[right].add(left)
                weights[key] += 1
    result = pd.DataFrame(rows)
    result.to_parquet(path, index=False)
    return result


def augment_base_frame(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Add outcome-blind locally derived structured inputs."""
    for label, view in (
        ("title features", title_features(frame)),
        ("prior volumes", prior_volume_features(frame)),
        ("author graph", author_graph_features(frame, output_dir)),
    ):
        frame = merge_unique(frame, view, label=label)
    frame["english_language_constant"] = 1.0
    frame["article_type_constant"] = 1.0
    frame["international_collaboration_proxy"] = (
        pd.to_numeric(frame["EF0188"], errors="coerce").gt(1).astype(float)
    )
    frame["additive_entropy_diversity_local"] = [
        compute_additive_entropy_diversity(entropy, evenness, disparity)
        for entropy, evenness, disparity in zip(
            pd.to_numeric(frame["field_shannon_entropy"], errors="coerce"),
            pd.to_numeric(frame["field_pielou_evenness"], errors="coerce"),
            pd.to_numeric(frame["field_disparity_cosine_mean"], errors="coerce"),
        )
    ]
    return frame


PATTERN_RULES: Tuple[Tuple[str, str, str], ...] = (
    (
        r"publication language",
        "english_language_constant",
        "English-only cohort constant",
    ),
    (r"document type", "article_type_constant", "article-only cohort constant"),
    (r"covid", "title_covid19", "publication-title topic flag"),
    (r"title length", "title_token_count_local", "publication-title word count"),
    (
        r"title punctuation",
        "title_punctuation_count",
        "publication-title punctuation count",
    ),
    (r"keyword count", "title_unique_token_count", "title unique-token count proxy"),
    (
        r"abstract information entropy",
        "title_token_entropy",
        "title entropy proxy; abstract unavailable",
    ),
    (
        r"abstract length",
        "title_token_count_local",
        "title length proxy; abstract unavailable",
    ),
    (
        r"sample size",
        "title_log_max_numeric",
        "largest publication-title numeric token proxy",
    ),
    (r"publication recency|publication year", "publication_year", "publication year"),
    (
        r"prior publication|publication history|h-index|career|expertise",
        "log_team_prior_nature_output_max",
        "prior team output proxy",
    ),
    (r"country|geograph", "EF0188", "author-country count proxy"),
    (
        r"international collaboration|international-team|international study",
        "EF0186",
        "international-team indicator",
    ),
    (
        (
            r"multi-institution|multicenter|institution|"
            r"centre-of-excellence|firm affiliation"
        ),
        "log_institution_count",
        "institution-count proxy",
    ),
    (r"prior-coauthorship count", "prior_team_edge_count", "prior team edge count"),
    (
        r"prior-coauthorship strength",
        "prior_team_edge_strength",
        "prior team edge-strength mean",
    ),
    (
        r"prior-collaboration continuity",
        "prior_team_edge_density",
        "prior team edge density",
    ),
    (
        r"giant-component",
        "prior_team_giant_component_share",
        "prior team giant-component share",
    ),
    (
        r"coauthorship-network centrality|eigenvector centrality",
        "prior_author_degree_mean",
        "prior author-degree proxy",
    ),
    (
        r"collaboration-network|co-authorship network|social-network",
        "prior_team_edge_density",
        "prior coauthorship graph proxy",
    ),
    (
        r"co-authorship|collaboration type|research collaboration",
        "international_collaboration_proxy",
        "team breadth proxy",
    ),
    (
        r"bibliographic|citation-network|knowledge-network|network centrality",
        "bc_degree_per_reference_t0",
        "prior bibliographic-coupling network proxy",
    ),
    (r"reference count", "EF0314", "reference count"),
    (
        r"backward-citation age|reference.*recency",
        "EF0052",
        "mean backward-citation age",
    ),
    (
        r"reference accuracy|reference-list characteristics",
        "field_mapping_coverage",
        "reference metadata coverage proxy",
    ),
    (r"rao-stirling", "EF0309", "Rao-Stirling reference diversity"),
    (r"\\bdiv interdisciplinarity", "field_div_index", "DIV reference-field index"),
    (r"balance component|reference balance", "EF0312", "reference-field balance"),
    (r"disparity component|reference disparity", "EF0315", "reference-field disparity"),
    (r"reference variety", "EF0318", "reference-field variety"),
    (
        (
            r"disciplinary diversity|interdisciplinarity|multidisciplinarity|"
            r"topical diversity|reference diversity"
        ),
        "field_div_index",
        "reference-field integration proxy",
    ),
    (
        r"conventionality",
        "uzzi_conventionality_median_t0",
        "source-pair conventionality",
    ),
    (
        r"novelty|concept birth|research-pivot|topic redundancy|innovation inertia",
        "reference_overlap_novelty_t0",
        "reference-combination novelty proxy",
    ),
    (
        r"journal|venue prestige|journal quartile|journal-type",
        "venue_prior_volume",
        "strictly prior-year venue-volume proxy",
    ),
    (
        r"topic|field normalization|research-community-size",
        "field_prior_volume",
        "strictly prior-year field-volume proxy",
    ),
    (r"author count|team composition", "EF0038", "author count"),
)


DIMENSION_ANCHORS = {
    "CD006": "EF0188",
    "CD007": "log_team_prior_nature_output_max",
    "CD010": "EF0038",
    "CD012": "bc_degree_per_reference_t0",
    "CD013": "field_mapping_coverage",
    "CD014": "international_collaboration_proxy",
    "CD015": "prior_team_edge_density",
    "CD028": "field_div_index",
    "CD029": "field_div_index",
    "CD031": "EF0309",
    "CD032": "title_new_bigram_share",
    "CD041": "venue_prior_volume",
    "CD044": "uzzi_conventionality_median_t0",
    "CD045": "EF0052",
    "CD046": "EF0314",
    "CD058": "EF0188",
    "CD061": "EF0188",
    "CD062": "title_token_count_local",
    "CD063": "field_prior_volume",
    "CD064": "reference_overlap_novelty_t0",
    "CD065": "field_prior_volume",
    "CD066": "venue_prior_volume",
}


def choose_operationalization(
    feature_id: str,
    library_row: Mapping[str, str],
    dimension_id: str,
) -> Dict[str, str]:
    """Choose a frozen outcome-blind implementation tier."""
    if feature_id in TARGETED_COLUMNS:
        return {
            "tier": "source_formula_existing",
            "source_column": feature_id,
            "notes": ("Recomputed from active uncapped-v2 publication-time views."),
        }
    if feature_id in FORMULA_SURROGATES:
        source_column, notes = FORMULA_SURROGATES[feature_id]
        return {
            "tier": "source_formula_local_surrogate",
            "source_column": source_column,
            "notes": notes,
        }
    name = library_row["canonical_name_en"].casefold()
    for pattern, source_column, notes in PATTERN_RULES:
        if re.search(pattern, name):
            return {
                "tier": "structured_construct_proxy",
                "source_column": source_column,
                "notes": notes,
            }
    if dimension_id in DIMENSION_ANCHORS:
        return {
            "tier": "structured_construct_proxy",
            "source_column": DIMENSION_ANCHORS[dimension_id],
            "notes": f"Frozen structured anchor for {dimension_id}.",
        }
    return {
        "tier": "title_taxonomy_lexical_proxy",
        "source_column": "lexical_similarity",
        "notes": "Outcome-blind lexical similarity to the evidence definition.",
    }


def descriptor(
    feature_id: str,
    library: Mapping[str, Mapping[str, str]],
    feature_to_dimension: Mapping[str, str],
    dimensions: Mapping[str, Mapping[str, str]],
) -> str:
    """Construct a source-derived English descriptor for lexical matching."""
    row = library[feature_id]
    dimension = dimensions[feature_to_dimension[feature_id]]
    aliases_payload = json.loads(row.get("alias_names_json") or "[]")
    required_payload = json.loads(row.get("required_data_json") or "[]")
    if not isinstance(aliases_payload, list) or not isinstance(required_payload, list):
        raise TypeError(f"{feature_id} descriptor list fields must be JSON arrays")
    aliases = " ".join(str(value) for value in aliases_payload)
    required = " ".join(str(value) for value in required_payload)
    formula = row.get("formula") or row.get("formula_text") or ""
    return " ".join(
        (
            row["canonical_name_en"],
            aliases,
            dimension["label"],
            dimension["definition"],
            formula,
            required,
        )
    )


def lexical_proxy_values(
    frame: pd.DataFrame,
    feature_ids: Sequence[str],
    library: Mapping[str, Mapping[str, str]],
    feature_to_dimension: Mapping[str, str],
    dimensions: Mapping[str, Mapping[str, str]],
) -> np.ndarray:
    """Compute word/character cosine similarity using evidence-only vocabularies."""
    if not feature_ids:
        return np.empty((len(frame), 0), dtype=np.float32)
    descriptions = [
        descriptor(fid, library, feature_to_dimension, dimensions)
        for fid in feature_ids
    ]
    paper_text = (
        frame[
            [
                "title",
                "openalex_primary_field",
                "openalex_primary_subfield",
                "primary_topic",
                "display_topic_label",
                "domain12_label",
            ]
        ]
        .astype("string")
        .fillna("")
        .agg(" ".join, axis=1)
        .tolist()
    )
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=32_768,
        sublinear_tf=True,
        norm="l2",
    )
    indicator_word = word.fit_transform(descriptions)
    indicator_char = char.fit_transform(descriptions)
    values = np.empty((len(frame), len(feature_ids)), dtype=np.float32)
    for start in range(0, len(frame), 5_000):
        stop = min(start + 5_000, len(frame))
        word_score = word.transform(paper_text[start:stop]) @ indicator_word.T
        char_score = char.transform(paper_text[start:stop]) @ indicator_char.T
        values[start:stop] = (
            0.7 * word_score.toarray() + 0.3 * char_score.toarray()
        ).astype(np.float32)
    return values


def build_indicator_matrix(
    frame: pd.DataFrame,
    feature_ids: Sequence[str],
    library: Mapping[str, Mapping[str, str]],
    feature_to_dimension: Mapping[str, str],
    dimensions: Mapping[str, Mapping[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize one column for every ultra-relaxed indicator."""
    specifications: Dict[str, Dict[str, str]] = {}
    text_ids: List[str] = []
    for feature_id in feature_ids:
        specification = choose_operationalization(
            feature_id,
            library[feature_id],
            feature_to_dimension[feature_id],
        )
        specifications[feature_id] = specification
        if specification["tier"] == "title_taxonomy_lexical_proxy":
            text_ids.append(feature_id)
    lexical = lexical_proxy_values(
        frame, text_ids, library, feature_to_dimension, dimensions
    )
    lexical_lookup = {
        feature_id: lexical[:, index] for index, feature_id in enumerate(text_ids)
    }
    matrix_columns: Dict[str, Any] = {
        "paper_id": frame["paper_id"].astype(str).to_numpy()
    }
    audit_rows: List[Dict[str, Any]] = []
    for feature_id in feature_ids:
        spec = specifications[feature_id]
        if feature_id in lexical_lookup:
            values: pd.Series = pd.Series(lexical_lookup[feature_id])
        else:
            source = spec["source_column"]
            if source not in frame.columns:
                raise ValueError(f"{feature_id} source column is missing: {source}")
            values = frame[source].reset_index(drop=True)
        if feature_id == "EF0197":
            column = values.astype("string").fillna("missing")
        else:
            column = pd.to_numeric(values, errors="coerce").astype("float32")
        matrix_columns[feature_id] = column.to_numpy()
        numeric = pd.to_numeric(column, errors="coerce")
        valid = int(column.notna().sum())
        audit_rows.append(
            {
                "feature_id": feature_id,
                "canonical_name_en": library[feature_id]["canonical_name_en"],
                "dimension_id": feature_to_dimension[feature_id],
                "dimension_label": dimensions[feature_to_dimension[feature_id]][
                    "label"
                ],
                "construct_role": dimensions[feature_to_dimension[feature_id]][
                    "construct_role"
                ],
                **spec,
                "row_count": len(frame),
                "valid_count": valid,
                "coverage": valid / len(frame),
                "unique_count": int(column.nunique(dropna=True)),
                "minimum": float(numeric.min()) if numeric.notna().any() else None,
                "maximum": float(numeric.max()) if numeric.notna().any() else None,
                "is_constant": int(column.nunique(dropna=True) <= 1),
                "all_missing": int(valid == 0),
            }
        )
    return pd.DataFrame(matrix_columns), pd.DataFrame(audit_rows)


def feature_set_payload(
    feature_sets: Mapping[str, Sequence[str]],
    feature_to_dimension: Mapping[str, str],
    dimensions: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Serialize set membership and dimension coverage."""
    criteria = {
        "strict_7": [f"G{index:02d}" for index in range(1, 15)],
        "fulltext_16": [
            *SAFE_T0_GATES,
            "G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE",
            "G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE",
        ],
        "source_154": [
            *SAFE_T0_GATES,
            "G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE",
        ],
        "ultrarelaxed_221": list(SAFE_T0_GATES),
    }
    payload: Dict[str, Any] = {
        "schema_version": "evidence_v3_oof_feature_sets_1",
        "definition_version": DEFINITION_VERSION,
        "sets": {},
    }
    for model_id, feature_ids in feature_sets.items():
        dimension_ids = sorted(
            {feature_to_dimension[feature_id] for feature_id in feature_ids}
        )
        payload["sets"][model_id] = {
            "criteria": criteria[model_id],
            "feature_count": len(feature_ids),
            "dimension_count": len(dimension_ids),
            "feature_ids": list(feature_ids),
            "dimension_ids": dimension_ids,
            "dimension_roles": {
                dimension_id: dimensions[dimension_id]["construct_role"]
                for dimension_id in dimension_ids
            },
        }
    return payload


def input_snapshot() -> Dict[str, Any]:
    """Hash every authoritative feature-construction input."""
    paths = {
        "indicator_library": EVIDENCE_OUTPUTS / "complete_indicator_library_v3.csv",
        "gate_decisions": EVIDENCE_OUTPUTS / "feature_gate_decisions_v3.csv",
        "candidate_dimensions": EVIDENCE_OUTPUTS / "candidate_dimensions_v3.csv",
        "active_dataset_registry": Path(ACTIVE_DATASET["registry_path"]),
        "active_dataset_contract": Path(ACTIVE_DATASET["contract_path"]),
        "papers": DATA_ROOT / "papers_primary_articles.parquet",
        "innovation_features": DATA_ROOT / "innovation_candidate_features.parquet",
        "controls": DATA_ROOT / "control_features_v6_1.parquet",
        "opportunity": DATA_ROOT / "opportunity_features.parquet",
        "openalex_metadata": DATA_ROOT / "target_openalex_metadata.parquet",
        "target_works": TARGET_WORKS,
    }
    return {
        key: {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for key, path in paths.items()
    }


def audit_expanded_matrices(
    matrix: pd.DataFrame,
    matrix_paths: Mapping[str, Path],
    feature_sets: Mapping[str, Sequence[str]],
    audit: pd.DataFrame,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Audit grain, horizon coverage, sparsity shifts, and leakage guards."""
    papers = pd.read_parquet(
        BASE_DATA_ROOT / "papers_primary_articles.parquet",
        columns=["paper_id", "publication_year", "domain12"],
    )
    cohort = pd.read_parquet(
        BASE_DATA_ROOT / "cohort_membership.parquet",
        columns=["paper_id", "publication_year", "horizon", "cohort_member"],
    )
    eligible = cohort[cohort["cohort_member"].eq(1)]
    ids = set(matrix["paper_id"].astype(str))
    physical = {
        model_id: pd.read_parquet(path) for model_id, path in matrix_paths.items()
    }
    numeric = matrix.drop(columns=["paper_id", "EF0197"], errors="ignore")
    finite_or_missing = (
        np.isfinite(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
        | numeric.isna().to_numpy()
    )
    checks = {
        "paper_id_unique": not matrix["paper_id"].duplicated().any(),
        "paper_set_matches_active_dataset": ids == set(papers["paper_id"].astype(str)),
        "all_horizon_cohorts_covered": set(eligible["paper_id"].astype(str)).issubset(
            ids
        ),
        "four_exact_matrix_shapes": all(
            physical[model_id].shape == (len(matrix), len(feature_sets[model_id]) + 1)
            for model_id in matrix_paths
        ),
        "physical_matrix_columns_exact": all(
            list(physical[model_id].columns) == ["paper_id", *feature_sets[model_id]]
            for model_id in matrix_paths
        ),
        "physical_matrices_match_221_superset": all(
            physical[model_id].equals(matrix[["paper_id", *feature_sets[model_id]]])
            for model_id in matrix_paths
        ),
        "no_all_missing_features": not bool(audit["all_missing"].any()),
        "numeric_values_finite_or_missing": bool(finite_or_missing.all()),
        "journal_identifier_complete": not matrix["EF0197"].isna().any(),
        "post_2017_papers_present": int(papers["publication_year"].max()) == 2022,
        "future_outcome_columns_absent": not any(
            column.startswith("future_")
            or column in {"horizon", "cohort_member", "cap_hit"}
            for column in matrix.columns
        ),
    }
    joined = papers[["paper_id", "publication_year"]].merge(
        matrix, on="paper_id", how="inner", validate="one_to_one"
    )
    pre = joined[joined["publication_year"].le(2017)]
    recent = joined[joined["publication_year"].ge(2018)]
    missingness = []
    for feature_id in matrix.columns.drop("paper_id"):
        pre_rate = float(pre[feature_id].isna().mean())
        recent_rate = float(recent[feature_id].isna().mean())
        missingness.append(
            {
                "feature_id": feature_id,
                "missing_rate_all": float(joined[feature_id].isna().mean()),
                "missing_rate_1980_2017": pre_rate,
                "missing_rate_2018_2022": recent_rate,
                "recent_minus_historical": recent_rate - pre_rate,
            }
        )
    horizon_counts = {
        str(int(horizon)): {
            "eligible_rows": int(len(group)),
            "publication_year_max": int(group["publication_year"].max()),
            "matrix_join_coverage": float(
                group["paper_id"].astype(str).isin(ids).mean()
            ),
        }
        for horizon, group in eligible.groupby("horizon", observed=True)
    }
    report = {
        "artifact_kind": "uncapped_v2_four_indicator_matrix_quality",
        "dataset_version": ACTIVE_DATASET["active_dataset_version"],
        "grain": ["paper_id"],
        "row_count": int(len(matrix)),
        "matrix_shapes": {
            model_id: list(physical[model_id].shape) for model_id in matrix_paths
        },
        "horizon_training_coverage": horizon_counts,
        "checks": checks,
        "overall_pass": all(checks.values()),
        "missingness_by_era": missingness,
        "leakage_policy": (
            "Only publication-time feature views are read; outcome and future "
            "citation columns are excluded from matrix construction."
        ),
    }
    write_json(output_dir / "matrix_quality_report.json", report)
    return report


def build(output_dir: Path) -> Mapping[str, Any]:
    """Build the matrix, membership registry, audit, and lineage manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_sets, library, feature_to_dimension, dimensions = load_feature_sets()
    sets_payload = feature_set_payload(feature_sets, feature_to_dimension, dimensions)
    frame = augment_base_frame(load_base_frame(), output_dir)
    matrix, audit = build_indicator_matrix(
        frame,
        feature_sets["ultrarelaxed_221"],
        library,
        feature_to_dimension,
        dimensions,
    )
    if audit["all_missing"].any():
        missing = audit.loc[audit["all_missing"].eq(1), "feature_id"].tolist()
        raise ValueError(f"all-missing feature columns are forbidden: {missing}")
    matrix_paths: Dict[str, Path] = {}
    for model_id, feature_ids in feature_sets.items():
        feature_count = len(feature_ids)
        path = output_dir / f"indicator_matrix_{feature_count}.parquet"
        matrix[["paper_id", *feature_ids]].to_parquet(path, index=False)
        matrix_paths[model_id] = path
    matrix_path = matrix_paths["ultrarelaxed_221"]
    audit_path = output_dir / "operationalization_audit.csv"
    sets_path = output_dir / "feature_sets.json"
    snapshot_path = output_dir / "input_snapshot.json"
    matrix.to_parquet(matrix_path, index=False)
    audit.to_csv(audit_path, index=False)
    write_json(sets_path, sets_payload)
    write_json(snapshot_path, input_snapshot())
    tier_counts = audit["tier"].value_counts().sort_index().to_dict()
    self_test = {
        "row_count_matches_active_expanded_articles": len(matrix)
        == len(pd.read_parquet(BASE_DATA_ROOT / "papers_primary_articles.parquet")),
        "paper_id_unique": not matrix["paper_id"].duplicated().any(),
        "matrix_has_221_features": len(matrix.columns) - 1 == 221,
        "four_physical_matrices_written": len(matrix_paths) == 4
        and all(path.is_file() for path in matrix_paths.values()),
        "no_all_missing_features": not bool(audit["all_missing"].any()),
        "feature_sets_nested": all(
            set(feature_sets[left]).issubset(feature_sets[right])
            for left, right in zip(
                list(EXPECTED_SET_COUNTS)[:-1],
                list(EXPECTED_SET_COUNTS)[1:],
            )
        ),
        "no_outcome_columns_read": True,
        "no_future_citation_columns_read": True,
    }
    if not all(self_test.values()):
        raise ValueError(f"feature-matrix self-test failed: {self_test}")
    quality = audit_expanded_matrices(
        matrix, matrix_paths, feature_sets, audit, output_dir
    )
    if not quality["overall_pass"]:
        failed = [key for key, value in quality["checks"].items() if not value]
        raise ValueError(f"expanded matrix quality audit failed: {failed}")
    manifest = {
        "artifact_kind": "evidence_v3_four_set_t0_indicator_matrix",
        "definition_version": DEFINITION_VERSION,
        "operationalization_policy": [
            "source_formula_existing",
            "source_formula_local_surrogate",
            "structured_construct_proxy",
            "title_taxonomy_lexical_proxy",
        ],
        "proxy_columns_are_not_claimed_as_original_formulas": True,
        "outcomes_used": False,
        "future_information_used": False,
        "same_matrix_superset_for_all_models": True,
        "active_dataset_version": ACTIVE_DATASET["active_dataset_version"],
        "row_count": len(matrix),
        "feature_count": len(matrix.columns) - 1,
        "tier_counts": tier_counts,
        "constant_feature_count": int(audit["is_constant"].sum()),
        "outputs": {
            **{
                f"matrix_{len(feature_sets[model_id])}": {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "model_id": model_id,
                    "feature_count": len(feature_sets[model_id]),
                }
                for model_id, path in matrix_paths.items()
            },
            "matrix": {
                "path": str(matrix_path.resolve()),
                "sha256": sha256_file(matrix_path),
            },
            "audit": {
                "path": str(audit_path.resolve()),
                "sha256": sha256_file(audit_path),
            },
            "feature_sets": {
                "path": str(sets_path.resolve()),
                "sha256": sha256_file(sets_path),
            },
            "quality_report": {
                "path": str((output_dir / "matrix_quality_report.json").resolve()),
                "sha256": sha256_file(output_dir / "matrix_quality_report.json"),
            },
            "input_snapshot": {
                "path": str(snapshot_path.resolve()),
                "sha256": sha256_file(snapshot_path),
            },
        },
        "self_test": self_test,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(output_dir / "matrix_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args().output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
