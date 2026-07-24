"""Outcome-blind measurement and limited silver-label validity audits for v6."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from aspr.corpus import normalize_openalex_id

from .evidence_registry import load_evidence_registry, registry_sha256
from .feature_materializer_v6 import (
    annual_field_distances,
    build_v6_reference_feature_table,
)
from .features_v6 import (
    field_disparity_mean,
    field_pielou_evenness,
    field_variety,
    rao_stirling_integration,
)
from .modeling_v6 import bootstrap_spearman_interval, safe_spearman
from .source_audit_v6 import sha256_file


CONSTRUCT_AUDIT_VERSION = "aspr-v6-construct-audit-1"
C1_METRICS: Tuple[str, ...] = (
    "field_variety",
    "field_pielou_evenness",
    "field_disparity_cosine_mean",
    "rao_stirling_integration",
)
N1_METRICS: Tuple[str, ...] = (
    "novelty_u_t0_source",
    "uzzi_atypicality_p10_t0",
)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _stable_hash(value: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def _parse_references(value: Any) -> List[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = list(value)
    else:
        parsed = []
    return sorted(
        {
            normalized
            for item in parsed
            if (normalized := normalize_openalex_id(item))
        }
    )


def select_stability_sample(
    papers: pd.DataFrame,
    features: pd.DataFrame,
    *,
    max_per_stratum: int,
    salt: str,
    min_valid_references: int,
    min_field_mapping_coverage: float,
) -> pd.DataFrame:
    """Select a deterministic outcome-blind domain-by-era audit sample."""
    columns = [
        "paper_id",
        "publication_year",
        "domain12",
        *C1_METRICS,
        "valid_reference_count",
        "field_mapping_coverage",
    ]
    frame = papers[["paper_id", "publication_year", "domain12"]].merge(
        features[[name for name in columns if name != "publication_year" and name != "domain12"]],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )
    eligible = (
        frame["publication_year"].le(2013)
        & frame["valid_reference_count"].ge(int(min_valid_references))
        & frame["field_mapping_coverage"].ge(
            float(min_field_mapping_coverage)
        )
        & frame[list(C1_METRICS)].notna().all(axis=1)
    )
    frame = frame[eligible].copy()
    frame["publication_era_5y"] = (
        frame["publication_year"].astype(int) // 5 * 5
    )
    frame["selection_hash"] = frame["paper_id"].astype(str).map(
        lambda value: _stable_hash(value, salt=salt)
    )
    return (
        frame.sort_values("selection_hash", kind="stable")
        .groupby(
            ["domain12", "publication_era_5y"],
            group_keys=False,
            sort=True,
        )
        .head(int(max_per_stratum))
        .sort_values(["domain12", "publication_era_5y", "selection_hash"])
        .reset_index(drop=True)
    )


def reference_subsampling_stability(
    sample: pd.DataFrame,
    bibliography: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    field_events: pd.DataFrame,
    *,
    fraction: float,
    repetitions: int,
    salt: str,
    field_profile_window_years: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute C1 metrics after deterministic reference subsampling."""
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("subsampling fraction must be between zero and one")
    selected_ids = set(sample["paper_id"].astype(str))
    bibliography = bibliography[
        bibliography["paper_id"].astype(str).isin(selected_ids)
    ].copy()
    metadata = reference_metadata[
        ["reference_id", "reference_year", "field_id"]
    ].copy()
    joined = (
        bibliography.merge(
            metadata,
            on="reference_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            sample[["paper_id", "publication_year"]],
            on="paper_id",
            how="inner",
            validate="many_to_one",
        )
    )
    joined = joined[
        joined["reference_year"].notna()
        & joined["reference_year"].lt(joined["publication_year"])
        & joined["field_id"].fillna("").astype(str).ne("")
    ]
    fields_by_paper = {
        str(paper_id): group["field_id"].astype(str).tolist()
        for paper_id, group in joined.groupby("paper_id", sort=False)
    }
    distances = annual_field_distances(
        field_events,
        sample["publication_year"].unique(),
        window_years=int(field_profile_window_years),
    )
    indexed = sample.set_index("paper_id", drop=False)
    full = indexed[list(C1_METRICS)]
    rows: List[Dict[str, Any]] = []
    for repetition in range(int(repetitions)):
        values = []
        for paper_id, row in indexed.iterrows():
            fields = fields_by_paper.get(str(paper_id), [])
            sample_size = max(2, int(np.floor(float(fraction) * len(fields))))
            seed = int(
                _stable_hash(
                    f"{repetition}:{paper_id}", salt=salt
                )[:16],
                16,
            ) % (2**32)
            rng = np.random.default_rng(seed)
            positions = rng.choice(
                len(fields),
                size=min(sample_size, len(fields)),
                replace=False,
            )
            sampled_fields = [fields[position] for position in positions]
            year_distances = distances[int(row["publication_year"])]
            try:
                disparity = field_disparity_mean(
                    sampled_fields, year_distances
                )
                rao = rao_stirling_integration(
                    sampled_fields, year_distances
                )
            except KeyError:
                disparity = np.nan
                rao = np.nan
            values.append(
                {
                    "paper_id": str(paper_id),
                    "field_variety": field_variety(sampled_fields),
                    "field_pielou_evenness": field_pielou_evenness(
                        sampled_fields
                    ),
                    "field_disparity_cosine_mean": disparity,
                    "rao_stirling_integration": rao,
                }
            )
        recomputed = pd.DataFrame(values).set_index("paper_id")
        for metric in C1_METRICS:
            paired = pd.concat(
                [full[metric], recomputed[metric]],
                axis=1,
                keys=["full", "subsample"],
            ).dropna()
            denominator = np.maximum(
                np.abs(paired["full"].to_numpy(dtype=float)), 1e-6
            )
            relative_error = np.abs(
                paired["subsample"].to_numpy(dtype=float)
                - paired["full"].to_numpy(dtype=float)
            ) / denominator
            rows.append(
                {
                    "repetition": repetition + 1,
                    "metric": metric,
                    "n_paired": len(paired),
                    "spearman": safe_spearman(
                        paired["full"], paired["subsample"]
                    ),
                    "median_relative_error": float(
                        np.median(relative_error)
                    ),
                }
            )
    repetitions_frame = pd.DataFrame(rows)
    summary = (
        repetitions_frame.groupby("metric", sort=True)
        .agg(
            n_paired_min=("n_paired", "min"),
            spearman_median=("spearman", "median"),
            spearman_min=("spearman", "min"),
            median_relative_error_median=(
                "median_relative_error",
                "median",
            ),
            median_relative_error_max=(
                "median_relative_error",
                "max",
            ),
        )
        .reset_index()
    )
    return repetitions_frame, summary


def metric_correlation_audit(features: pd.DataFrame) -> pd.DataFrame:
    """Report pairwise rank association and explicit redundancy flags."""
    rows = []
    metrics = (*N1_METRICS, *C1_METRICS)
    for left, right in itertools.combinations(metrics, 2):
        paired = features[[left, right]].dropna()
        rows.append(
            {
                "left_metric": left,
                "right_metric": right,
                "left_dimension": (
                    "N1_RECOMBINATION"
                    if left in N1_METRICS
                    else "C1_KNOWLEDGE_DIVERSITY"
                ),
                "right_dimension": (
                    "N1_RECOMBINATION"
                    if right in N1_METRICS
                    else "C1_KNOWLEDGE_DIVERSITY"
                ),
                "n_paired": len(paired),
                "spearman": safe_spearman(paired[left], paired[right]),
            }
        )
    return pd.DataFrame(rows)


def review_silver_validation(
    *,
    target_works_path: Path,
    review_papers_path: Path,
    review_scores_path: Path,
    reference_metadata: pd.DataFrame,
    field_events: pd.DataFrame,
    field_profile_window_years: int,
    bootstrap_iterations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    """Use only matched frozen heuristic review labels as non-gold evidence."""
    target_columns = [
        "id",
        "doi",
        "year",
        "referenced_works",
    ]
    targets = pd.read_csv(
        target_works_path,
        usecols=target_columns,
        low_memory=False,
    )
    reviews = pd.read_csv(review_papers_path)
    targets["doi_norm"] = (
        targets["doi"].fillna("").astype(str).str.lower().str.strip()
    )
    reviews["doi_norm"] = (
        reviews["doi"].fillna("").astype(str).str.lower().str.strip()
    )
    reviews = reviews.rename(columns={"paper_id": "review_paper_id"})
    targets = targets[targets["doi_norm"].ne("")].drop_duplicates(
        "doi_norm", keep="last"
    )
    matched = reviews[["review_paper_id", "doi_norm"]].merge(
        targets,
        on="doi_norm",
        how="inner",
        validate="one_to_one",
    )
    matched["openalex_paper_id"] = matched["id"].map(normalize_openalex_id)
    matched["publication_year"] = pd.to_numeric(
        matched["year"], errors="coerce"
    )
    matched = matched[matched["publication_year"].notna()].copy()
    matched["publication_year"] = matched["publication_year"].astype(int)
    matched["domain12"] = "review_silver_external"
    matched["references"] = matched["referenced_works"].map(
        _parse_references
    )
    bibliography = (
        matched[["openalex_paper_id", "references"]]
        .explode("references")
        .rename(
            columns={
                "openalex_paper_id": "paper_id",
                "references": "reference_id",
            }
        )
        .dropna()
        .drop_duplicates()
    )
    work_view = reference_metadata.rename(
        columns={
            "reference_id": "work_id",
            "reference_year": "publication_year",
        }
    )
    focal = matched[
        ["openalex_paper_id", "publication_year", "domain12"]
    ].rename(columns={"openalex_paper_id": "paper_id"})
    features = build_v6_reference_feature_table(
        focal,
        bibliography,
        work_view,
        field_citation_events=field_events,
        field_profile_window_years=int(field_profile_window_years),
    )
    labels = pd.read_csv(review_scores_path)
    labels = labels[
        labels["source_kind"].eq("peer_review")
        & labels["aspect"].isin(["novelty", "prior_art_comparison"])
        & labels["judgement_success"].eq(True)
    ]
    label_values = labels.pivot(
        index="paper_id", columns="aspect", values="score_1_5"
    ).reset_index().rename(columns={"paper_id": "review_paper_id"})
    label_confidence = (
        labels.groupby("paper_id", as_index=False)["confidence"]
        .min()
        .rename(
            columns={
                "paper_id": "review_paper_id",
                "confidence": "minimum_label_confidence",
            }
        )
    )
    output = (
        matched[
            [
                "review_paper_id",
                "openalex_paper_id",
                "doi_norm",
                "publication_year",
            ]
        ]
        .merge(
            features,
            left_on="openalex_paper_id",
            right_on="paper_id",
            how="inner",
            suffixes=("", "_feature"),
            validate="one_to_one",
        )
        .merge(label_values, on="review_paper_id", validate="one_to_one")
        .merge(
            label_confidence,
            on="review_paper_id",
            validate="one_to_one",
        )
    )
    correlation_rows = []
    for metric in C1_METRICS:
        for label in ("novelty", "prior_art_comparison"):
            paired = output[[metric, label]].dropna()
            rho = safe_spearman(paired[metric], paired[label])
            ci_low, ci_high = bootstrap_spearman_interval(
                paired[metric],
                paired[label],
                iterations=int(bootstrap_iterations),
                seed=int(seed) + len(correlation_rows),
            )
            correlation_rows.append(
                {
                    "metric": metric,
                    "silver_label": label,
                    "n_paired": len(paired),
                    "spearman": rho,
                    "spearman_ci_low": ci_low,
                    "spearman_ci_high": ci_high,
                    "label_source": "peer_review_heuristic_extractor",
                    "evidence_role": "supportive_only_never_gold",
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    summary = {
        "n_review_papers": int(len(reviews)),
        "n_matched_frozen_targets": int(len(output)),
        "minimum_label_confidence": float(
            output["minimum_label_confidence"].min()
        )
        if len(output)
        else 0.0,
        "all_c1_novelty_directions_positive": bool(
            correlations.loc[
                correlations["silver_label"].eq("novelty"), "spearman"
            ].gt(0).all()
        ),
        "confirmatory_eligible": False,
        "confirmatory_exclusion_reason": (
            "heuristic labels, low confidence, fewer than the preregistered "
            "minimum matched papers; supportive direction only"
        ),
    }
    return output, correlations, summary


def run_construct_validation(
    *,
    project_root: Path,
    config_path: Path,
    dataset_dir: Path,
    output_root: Path,
) -> Tuple[Mapping[str, Any], Path]:
    """Run and persist the complete pre-holdout measurement audit."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_root = Path(output_root).resolve()
    config = _load_json(config_path)
    registry_path = project_root / str(config["evidence_registry_path"])
    registry = load_evidence_registry(registry_path)
    registry_hash = registry_sha256(registry)
    validation = config["construct_validation_protocol"]
    subsampling = validation["reference_subsampling"]
    input_manifest_path = dataset_dir / "input_views_manifest.json"
    feature_manifest_path = dataset_dir / "publication_features_manifest.json"
    event_manifest_path = dataset_dir / "field_events_manifest.json"
    lineage = {
        "construct_audit_version": CONSTRUCT_AUDIT_VERSION,
        "config_sha256": sha256_file(config_path),
        "innovation_registry_sha256": registry_hash,
        "input_views_artifact_id": _load_json(input_manifest_path)[
            "artifact_id"
        ],
        "publication_features_artifact_id": _load_json(
            feature_manifest_path
        )["artifact_id"],
        "field_events_artifact_id": _load_json(event_manifest_path)[
            "artifact_id"
        ],
        "review_silver_manifest_sha256": sha256_file(
            project_root / "data/review_innovation_opinions_v1/manifest.json"
        ),
        "code_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parent
                / "feature_materializer_v6.py",
                Path(__file__).resolve().parent / "features_v6.py",
                Path(__file__).resolve().parent / "modeling_v6.py",
            )
        },
        "sealed_holdout_accessed": False,
        "future_influence_outcomes_used": False,
        "network_policy": "forbidden",
    }
    run_hash = _canonical_hash(lineage)
    output_dir = output_root / (
        f"construct_validation_{run_hash.removeprefix('sha256:')[:12]}"
    )
    manifest_path = output_dir / "construct_validation_manifest.json"
    if manifest_path.is_file():
        return _load_json(manifest_path), output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    papers = pd.read_parquet(dataset_dir / "papers_primary_articles.parquet")
    features = pd.read_parquet(dataset_dir / "innovation_features.parquet")
    bibliography = pd.read_parquet(dataset_dir / "paper_references.parquet")
    metadata = pd.read_parquet(dataset_dir / "reference_metadata.parquet")
    events = pd.read_parquet(
        dataset_dir / "field_citation_events_aggregated.parquet"
    )
    sample = select_stability_sample(
        papers,
        features,
        max_per_stratum=int(
            subsampling["max_papers_per_domain_5y_stratum"]
        ),
        salt=str(subsampling["selection_hash_salt"]),
        min_valid_references=int(subsampling["min_valid_references"]),
        min_field_mapping_coverage=float(
            subsampling["min_field_mapping_coverage"]
        ),
    )
    repetitions, stability = reference_subsampling_stability(
        sample,
        bibliography,
        metadata,
        events,
        fraction=float(subsampling["fraction"]),
        repetitions=int(subsampling["repetitions"]),
        salt=str(subsampling["selection_hash_salt"]),
        field_profile_window_years=int(
            config["feature_protocol"]["field_profile_window_years"]
        ),
    )
    stability["spearman_gate"] = stability["spearman_min"].ge(
        float(subsampling["spearman_min"])
    )
    stability["relative_error_gate"] = stability[
        "median_relative_error_max"
    ].le(float(subsampling["median_relative_error_max"]))
    stability["overall_pass"] = (
        stability["spearman_gate"] & stability["relative_error_gate"]
    )
    development_features = papers.loc[
        papers["publication_year"].le(2013), ["paper_id"]
    ].merge(
        features[["paper_id", *N1_METRICS, *C1_METRICS]],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )
    correlations = metric_correlation_audit(development_features)
    collinearity_threshold = float(
        validation["discriminant_redundancy"][
            "absolute_spearman_collinearity_flag"
        ]
    )
    correlations["collinearity_flag"] = correlations["spearman"].abs().ge(
        collinearity_threshold
    )
    target_root = project_root / "outputs/nature_portfolio_v5"
    review_features, review_correlations, review_summary = (
        review_silver_validation(
            target_works_path=target_root / "nature_target_works.csv",
            review_papers_path=project_root
            / "data/review_innovation_opinions_v1/papers.csv",
            review_scores_path=project_root
            / "data/review_innovation_opinions_v1/innovation_aspect_scores.csv",
            reference_metadata=metadata,
            field_events=events,
            field_profile_window_years=int(
                config["feature_protocol"]["field_profile_window_years"]
            ),
            bootstrap_iterations=2000,
            seed=int(config["validation_protocol"]["seed"]),
        )
    )
    summary = {
        "artifact_kind": "aspr_v6_construct_validation",
        "stability_sample_rows": len(sample),
        "stability_strata": int(
            sample.groupby(["domain12", "publication_era_5y"]).ngroups
        ),
        "all_c1_reference_stability_pass": bool(
            stability["overall_pass"].all()
        ),
        "c1_discriminant_noncollinearity_pass": bool(
            ~correlations["collinearity_flag"].any()
        ),
        "maximum_absolute_pairwise_spearman": float(
            correlations["spearman"].abs().max()
        ),
        "review_silver": dict(review_summary),
        "c1_measurement_gate_pass": bool(
            stability["overall_pass"].all()
            and ~correlations["collinearity_flag"].any()
        ),
        "sealed_holdout_accessed": False,
        "future_influence_outcomes_used": False,
        "limitations": [
            "The review validation is a 23-paper, low-confidence heuristic silver set and cannot promote a construct by itself.",
            "No frozen human-gold or overlapping landmark focal-paper set exists locally.",
            "Discriminant noncollinearity supports separability but does not make knowledge-base diversity direct novelty.",
        ],
    }
    paths = {
        "stability_sample": output_dir / "stability_sample.parquet",
        "stability_repetitions": output_dir
        / "reference_subsampling_repetitions.csv",
        "stability_summary": output_dir
        / "reference_subsampling_summary.csv",
        "metric_correlations": output_dir / "metric_correlations.csv",
        "review_features": output_dir / "review_silver_features.parquet",
        "review_correlations": output_dir
        / "review_silver_correlations.csv",
        "summary": output_dir / "construct_validation_summary.json",
    }
    sample.to_parquet(paths["stability_sample"], index=False)
    repetitions.to_csv(paths["stability_repetitions"], index=False)
    stability.to_csv(paths["stability_summary"], index=False)
    correlations.to_csv(paths["metric_correlations"], index=False)
    review_features.to_parquet(paths["review_features"], index=False)
    review_correlations.to_csv(paths["review_correlations"], index=False)
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact_kind": "aspr_v6_construct_validation",
        "lineage": lineage,
        "summary": summary,
        "outputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest, output_dir


__all__ = [
    "C1_METRICS",
    "CONSTRUCT_AUDIT_VERSION",
    "N1_METRICS",
    "metric_correlation_audit",
    "reference_subsampling_stability",
    "review_silver_validation",
    "run_construct_validation",
    "select_stability_sample",
]
