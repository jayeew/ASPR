"""Offline materialization stages for the v6 common Nature cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from aspr.corpus import normalize_openalex_id

from .cohorts import build_cohort_membership, cohort_quality_summary
from .contracts import CohortSpec
from .evidence_registry import load_evidence_registry, registry_sha256
from .feature_materializer_v6 import (
    aggregate_field_citation_events_from_edges,
    build_v6_reference_feature_table,
)
from .prediction_features_v6 import (
    build_bibliographic_opportunity_features,
    build_registered_control_features,
)
from .prediction_registry_v6 import (
    load_prediction_registry,
    prediction_registry_sha256,
)
from .source_audit_v6 import sha256_file
from .targets import build_diffusion_targets_from_deltas
from .taxonomy import DOMAIN_IDS, build_taxonomy_table


MATERIALIZATION_VERSION = "aspr-v6-local-materialization-1"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _asset_path(config: Mapping[str, Any], asset_id: str, project_root: Path) -> Path:
    for source in config["sources"]:
        if source["asset_id"] == asset_id:
            return _resolve(project_root, str(source["path"]))
    raise KeyError(f"v6 config has no source asset {asset_id}")


def _parse_references(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
    elif isinstance(value, (list, tuple, np.ndarray)):
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


def _write_stage_manifest(
    output_dir: Path,
    stage: str,
    *,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Path],
    counts: Mapping[str, Any],
) -> Dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    code_paths = (
        Path(__file__).resolve(),
        module_root / "cohorts.py",
        module_root / "feature_materializer_v6.py",
        module_root / "features_v6.py",
        module_root / "prediction_features_v6.py",
        module_root / "targets.py",
    )
    output_rows = {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in outputs.items()
    }
    manifest = {
        "artifact_kind": "aspr_v6_local_derived_stage",
        "materialization_version": MATERIALIZATION_VERSION,
        "stage": stage,
        "network_policy": "forbidden",
        "code_sha256": {
            path.name: sha256_file(path) for path in code_paths
        },
        "inputs": dict(inputs),
        "outputs": output_rows,
        "counts": dict(counts),
    }
    manifest["artifact_id"] = _canonical_sha256(manifest)
    path = output_dir / f"{stage}_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_common_input_views(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    source_audit: Mapping[str, Any],
    resume: bool = True,
) -> Dict[str, Any]:
    """Create compact common-cohort, bibliography, and reference views."""
    output_dir.mkdir(parents=True, exist_ok=True)
    papers_path = output_dir / "papers_common_all.parquet"
    primary_path = output_dir / "papers_primary_articles.parquet"
    references_path = output_dir / "paper_references.parquet"
    metadata_path = output_dir / "reference_metadata.parquet"
    coverage_path = output_dir / "domain12_coverage.parquet"
    manifest_path = output_dir / "input_views_manifest.json"
    expected = (
        papers_path,
        primary_path,
        references_path,
        metadata_path,
        coverage_path,
        manifest_path,
    )
    if resume and all(path.is_file() for path in expected):
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    v5_root = _asset_path(config, "nature_v5_openalex_outputs", project_root)
    future_root = _asset_path(
        config, "nature_v5_future_multihorizon", project_root
    )
    requested = pd.read_parquet(
        future_root / "future_request_manifest.parquet",
        columns=["paper_id"],
    )
    requested_ids = set(
        requested["paper_id"].map(normalize_openalex_id).astype(str)
    )
    paper_columns = [
        "id",
        "year",
        "domain",
        "broad_category",
        "journal_family",
        "source_id",
        "source_display_name",
        "primary_field",
        "openalex_primary_field",
        "openalex_primary_subfield",
        "display_topic_label",
        "primary_topic",
        "document_type",
        "referenced_works",
    ]
    papers = pd.read_csv(
        v5_root / "nature_target_works.csv",
        usecols=paper_columns,
        low_memory=False,
    )
    papers["paper_id"] = papers["id"].map(normalize_openalex_id)
    papers = papers[papers["paper_id"].isin(requested_ids)].copy()
    papers["publication_year"] = pd.to_numeric(
        papers["year"], errors="coerce"
    )
    papers = papers[papers["publication_year"].notna()].copy()
    papers["publication_year"] = papers["publication_year"].astype(int)
    papers["work_type"] = papers["document_type"].fillna("").astype(str)
    papers["venue_family"] = papers["journal_family"].fillna("unknown").astype(
        str
    )
    papers["referenced_works"] = papers["referenced_works"].map(
        _parse_references
    )
    papers, domain_coverage, domain_audit = build_taxonomy_table(papers)
    papers = papers.drop(columns=["id", "year"], errors="ignore")
    papers.to_parquet(papers_path, index=False)
    domain_coverage.to_parquet(coverage_path, index=False)

    primary = papers[
        papers["domain12"].isin(DOMAIN_IDS)
        & papers["work_type"].eq("article")
    ].copy()
    primary.to_parquet(primary_path, index=False)
    bibliography = primary[["paper_id", "referenced_works"]].explode(
        "referenced_works"
    )
    bibliography = bibliography.rename(
        columns={"referenced_works": "reference_id"}
    )
    bibliography = bibliography[
        bibliography["reference_id"].notna()
        & bibliography["reference_id"].astype(str).ne("")
    ].drop_duplicates()
    bibliography.to_parquet(references_path, index=False)

    reference_columns = [
        "id",
        "year",
        "source_id",
        "primary_field",
        "openalex_primary_field",
        "document_type",
        "is_target_work",
    ]
    metadata = pd.read_csv(
        v5_root / "nature_reference_works.csv",
        usecols=reference_columns,
        low_memory=False,
    )
    metadata["reference_id"] = metadata["id"].map(normalize_openalex_id)
    metadata["reference_year"] = pd.to_numeric(
        metadata["year"], errors="coerce"
    )
    metadata["field_id"] = (
        metadata["openalex_primary_field"]
        .fillna(metadata["primary_field"])
        .fillna("")
        .astype(str)
    )
    metadata["source_id"] = metadata["source_id"].fillna("").astype(str)
    metadata = metadata[
        [
            "reference_id",
            "reference_year",
            "source_id",
            "field_id",
            "document_type",
            "is_target_work",
        ]
    ].drop_duplicates("reference_id", keep="last")
    metadata.to_parquet(metadata_path, index=False)
    return _write_stage_manifest(
        output_dir,
        "input_views",
        inputs={
            "source_lineage_id": source_audit.get("source_lineage_id"),
            "v5_target_path": str(v5_root / "nature_target_works.csv"),
            "v5_reference_path": str(v5_root / "nature_reference_works.csv"),
            "future_request_path": str(
                future_root / "future_request_manifest.parquet"
            ),
        },
        outputs={
            "papers_common_all": papers_path,
            "papers_primary_articles": primary_path,
            "paper_references": references_path,
            "reference_metadata": metadata_path,
            "domain12_coverage": coverage_path,
        },
        counts={
            "n_common_papers": len(papers),
            "n_primary_natural_articles": len(primary),
            "n_paper_reference_edges": len(bibliography),
            "n_reference_metadata": len(metadata),
            "domain_audit": domain_audit,
        },
    )


def materialize_field_events(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    upstream_manifest: Mapping[str, Any],
    resume: bool = True,
) -> Dict[str, Any]:
    """Aggregate the existing closure graph into compact field events."""
    events_path = output_dir / "field_citation_events_aggregated.parquet"
    manifest_path = output_dir / "field_events_manifest.json"
    if resume and events_path.is_file() and manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = pd.read_parquet(output_dir / "reference_metadata.parquet")
    v5_root = _asset_path(config, "nature_v5_openalex_outputs", project_root)
    events = aggregate_field_citation_events_from_edges(
        v5_root / "nature_reference_edges.csv", metadata
    )
    events.to_parquet(events_path, index=False)
    return _write_stage_manifest(
        output_dir,
        "field_events",
        inputs={"input_views_artifact_id": upstream_manifest["artifact_id"]},
        outputs={"field_citation_events": events_path},
        counts={
            "n_aggregated_rows": len(events),
            "min_source_year": int(events["source_year"].min()),
            "max_source_year": int(events["source_year"].max()),
        },
    )


def materialize_publication_features(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    input_manifest: Mapping[str, Any],
    event_manifest: Mapping[str, Any],
    resume: bool = True,
) -> Dict[str, Any]:
    """Materialize v6 evidence features and strong controls."""
    features_path = output_dir / "innovation_features.parquet"
    controls_path = output_dir / "control_features.parquet"
    manifest_path = output_dir / "publication_features_manifest.json"
    if (
        resume
        and features_path.is_file()
        and controls_path.is_file()
        and manifest_path.is_file()
    ):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = pd.read_parquet(output_dir / "papers_primary_articles.parquet")
    bibliography = pd.read_parquet(output_dir / "paper_references.parquet")
    metadata = pd.read_parquet(output_dir / "reference_metadata.parquet")
    events = pd.read_parquet(
        output_dir / "field_citation_events_aggregated.parquet"
    )
    work_view = metadata.rename(
        columns={
            "reference_id": "work_id",
            "reference_year": "publication_year",
        }
    )
    features = build_v6_reference_feature_table(
        papers,
        bibliography,
        work_view,
        field_citation_events=events,
        field_profile_window_years=int(
            config["feature_protocol"]["field_profile_window_years"]
        ),
    )
    features.to_parquet(features_path, index=False)
    controls = build_registered_control_features(
        papers, bibliography, work_view
    )
    controls.to_parquet(controls_path, index=False)
    evidence_registry = load_evidence_registry(
        _resolve(project_root, config["evidence_registry_path"])
    )
    return _write_stage_manifest(
        output_dir,
        "publication_features",
        inputs={
            "input_views_artifact_id": input_manifest["artifact_id"],
            "field_events_artifact_id": event_manifest["artifact_id"],
            "innovation_registry_sha256": registry_sha256(evidence_registry),
        },
        outputs={
            "innovation_features": features_path,
            "control_features": controls_path,
        },
        counts={
            "n_feature_rows": len(features),
            "n_control_rows": len(controls),
            "strict_prior_violations": int(
                features["source_max_year"]
                .ge(features["publication_year"])
                .sum()
            ),
            "primary_feature_finite_rates": {
                name: float(
                    pd.to_numeric(features[name], errors="coerce").notna().mean()
                )
                for name in evidence_registry.primary_feature_names
            },
        },
    )


def materialize_opportunity_features(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    input_manifest: Mapping[str, Any],
    resume: bool = True,
) -> Dict[str, Any]:
    """Materialize the prediction-only bibliographic opportunity block."""
    opportunity_path = output_dir / "opportunity_features.parquet"
    manifest_path = output_dir / "opportunity_features_manifest.json"
    if resume and opportunity_path.is_file() and manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = pd.read_parquet(output_dir / "papers_primary_articles.parquet")
    bibliography = pd.read_parquet(output_dir / "paper_references.parquet")
    metadata = pd.read_parquet(output_dir / "reference_metadata.parquet")
    history = papers[
        ["paper_id", "publication_year", "referenced_works"]
    ].rename(columns={"paper_id": "work_id"})
    opportunity = build_bibliographic_opportunity_features(
        papers,
        bibliography,
        history,
        reference_metadata=metadata,
        compute_exact_clustering=False,
        compute_exact_closeness=False,
    )
    opportunity.to_parquet(opportunity_path, index=False)
    prediction_registry = load_prediction_registry(
        _resolve(project_root, config["prediction_registry_path"])
    )
    return _write_stage_manifest(
        output_dir,
        "opportunity_features",
        inputs={
            "input_views_artifact_id": input_manifest["artifact_id"],
            "prediction_registry_sha256": prediction_registry_sha256(
                prediction_registry
            ),
            "graph_scope": "strictly_prior_primary_nature_articles",
        },
        outputs={"opportunity_features": opportunity_path},
        counts={
            "n_rows": len(opportunity),
            "strict_prior_violations": int(
                opportunity["source_max_year"]
                .ge(opportunity["publication_year"])
                .sum()
            ),
            "finite_rates": {
                name: float(
                    pd.to_numeric(opportunity[name], errors="coerce")
                    .notna()
                    .mean()
                )
                for name in prediction_registry.opportunity_feature_names
            },
        },
    )


def materialize_targets_and_cohort(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    input_manifest: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
    resume: bool = True,
) -> Dict[str, Any]:
    """Build zero-inclusive D3/D5/D8 components and common cohort flags."""
    targets_path = output_dir / "targets_zero_inclusive.parquet"
    cohort_path = output_dir / "cohort_membership.parquet"
    quality_path = output_dir / "cohort_quality.json"
    manifest_path = output_dir / "targets_cohort_manifest.json"
    expected = (targets_path, cohort_path, quality_path, manifest_path)
    if resume and all(path.is_file() for path in expected):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = pd.read_parquet(output_dir / "papers_primary_articles.parquet")
    features = pd.read_parquet(output_dir / "innovation_features.parquet")
    future_root = _asset_path(
        config, "nature_v5_future_multihorizon", project_root
    )
    deltas = pd.read_parquet(
        future_root / "future_graph_deltas_multihorizon.parquet"
    )
    selected_ids = set(papers["paper_id"].astype(str))
    deltas = deltas[deltas["paper_id"].astype(str).isin(selected_ids)].copy()
    horizons = tuple(int(item["tau"]) for item in config["horizons"])
    targets = build_diffusion_targets_from_deltas(
        papers,
        deltas,
        horizons=horizons,
        min_future_citers=0,
        min_taxonomy_coverage=0.8,
    )
    targets.to_parquet(targets_path, index=False)
    evidence_registry = load_evidence_registry(
        _resolve(project_root, config["evidence_registry_path"])
    )
    cohort_config = config["cohort_protocol"]
    cohort = build_cohort_membership(
        papers,
        features,
        targets,
        spec=CohortSpec(
            horizons=horizons,
            primary_horizon=5,
            min_future_citers=0,
            min_valid_references=int(
                cohort_config["min_valid_references"]
            ),
            min_reference_metadata_coverage=float(
                cohort_config["min_reference_metadata_coverage"]
            ),
            high_quality_reference_coverage=0.8,
            allowed_work_types=tuple(cohort_config["primary_work_types"]),
            require_all_features_finite=bool(
                cohort_config["require_all_primary_features_finite"]
            ),
            require_reference_quality_for_cohort=bool(
                cohort_config["require_reference_quality_for_cohort"]
            ),
            require_target_quality_for_cohort=bool(
                cohort_config[
                    "require_future_taxonomy_quality_for_uptake_cohort"
                ]
            ),
        ),
        required_feature_names=evidence_registry.primary_feature_names,
        complete_end_year=2025,
    )
    cohort.to_parquet(cohort_path, index=False)
    quality = cohort_quality_summary(cohort)
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _write_stage_manifest(
        output_dir,
        "targets_cohort",
        inputs={
            "input_views_artifact_id": input_manifest["artifact_id"],
            "publication_features_artifact_id": feature_manifest["artifact_id"],
            "future_source": str(
                future_root / "future_graph_deltas_multihorizon.parquet"
            ),
        },
        outputs={
            "targets": targets_path,
            "cohort_membership": cohort_path,
            "cohort_quality": quality_path,
        },
        counts={
            "n_target_rows": len(targets),
            "n_cohort_rows": len(cohort),
            "n_common_members": int(
                cohort.loc[
                    cohort["common_cohort_member"].eq(1), "paper_id"
                ].nunique()
            ),
            "n_observed_zero_rows": int(
                cohort["observed_zero_future_citers"].sum()
            ),
            "quality": quality,
        },
    )


__all__ = [
    "MATERIALIZATION_VERSION",
    "materialize_common_input_views",
    "materialize_field_events",
    "materialize_opportunity_features",
    "materialize_publication_features",
    "materialize_targets_and_cohort",
]
