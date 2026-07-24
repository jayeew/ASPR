"""Reproducible data-quality and lineage audit for ASPR v6.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .candidate_registry_v6_1 import load_candidate_registry_v6_1
from .modeling_v6_1 import load_simple_config
from .source_audit_v6 import sha256_file


AUDIT_VERSION_V6_1 = "aspr-v6.1-data-quality-audit-4"
EXPECTED_DOMAINS: Tuple[str, ...] = (
    "astronomy_space",
    "chemistry",
    "clinical_health",
    "computer_science_ai",
    "earth_climate_environment",
    "ecology_evolution_microbiology",
    "engineering_energy",
    "life_molecular",
    "materials_nanoscience",
    "mathematics_statistics",
    "neuroscience",
    "physics",
)


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def _key_profile(
    name: str,
    frame: pd.DataFrame,
    paper_ids: set[str],
    path: Path,
) -> Dict[str, Any]:
    ids = frame["paper_id"].astype(str)
    observed = set(ids)
    return {
        "view": name,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "n_rows": int(len(frame)),
        "n_unique_papers": int(ids.nunique()),
        "duplicate_paper_rows": int(ids.duplicated().sum()),
        "missing_primary_papers": int(len(paper_ids - observed)),
        "unexpected_extra_papers": int(len(observed - paper_ids)),
    }


def _time_leakage_rows(frame: pd.DataFrame) -> int:
    if not {"source_max_year", "publication_year"}.issubset(frame):
        return 0
    source = pd.to_numeric(frame["source_max_year"], errors="coerce")
    publication = pd.to_numeric(
        frame["publication_year"], errors="coerce"
    )
    return int(
        (source.notna() & publication.notna() & source.ge(publication)).sum()
    )


def _missingness_rows(
    controls: pd.DataFrame,
    features: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for feature in features:
        missing = controls[feature].isna()
        rows.append(
            {
                "scope": "all",
                "domain12": "ALL",
                "feature": feature,
                "n_rows": len(controls),
                "n_missing": int(missing.sum()),
                "missing_rate": float(missing.mean()),
            }
        )
        for domain, group in controls.groupby("domain12", sort=True):
            domain_missing = group[feature].isna()
            rows.append(
                {
                    "scope": "domain12",
                    "domain12": str(domain),
                    "feature": feature,
                    "n_rows": len(group),
                    "n_missing": int(domain_missing.sum()),
                    "missing_rate": float(domain_missing.mean()),
                }
            )
    return rows


def _copied_view_checks(
    v6_root: Path,
    v6_1_root: Path,
    names: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = []
    for name in names:
        source = v6_root / name
        copied = v6_1_root / name
        source_hash = sha256_file(source)
        copied_hash = sha256_file(copied)
        rows.append(
            {
                "name": name,
                "v6_path": str(source),
                "v6_1_path": str(copied),
                "v6_sha256": source_hash,
                "v6_1_sha256": copied_hash,
                "identical": bool(source_hash == copied_hash),
            }
        )
    return rows


def _target_quality(
    dataset_root: Path,
) -> List[Dict[str, Any]]:
    membership = pd.read_parquet(
        dataset_root / "cohort_membership.parquet"
    )
    targets = pd.read_parquet(
        dataset_root / "targets_zero_inclusive.parquet"
    )
    target_uptake = targets[
        ["paper_id", "horizon", "future_uptake"]
    ].rename(columns={"future_uptake": "target_future_uptake"})
    joined = membership.merge(
        target_uptake,
        on=["paper_id", "horizon"],
        how="left",
        validate="one_to_one",
    )
    rows = []
    for horizon, group in joined.groupby("horizon", sort=True):
        members = group[group["cohort_member"].eq(1)]
        membership_uptake = pd.to_numeric(
            members["future_uptake"], errors="coerce"
        )
        target_values = pd.to_numeric(
            members["target_future_uptake"], errors="coerce"
        )
        mismatch = membership_uptake.isna().ne(target_values.isna()) | (
            membership_uptake.notna()
            & target_values.notna()
            & membership_uptake.ne(target_values)
        )
        rows.append(
            {
                "horizon": int(horizon),
                "n_members": int(len(members)),
                "n_unique_papers": int(members["paper_id"].nunique()),
                "missing_uptake_labels": int(
                    target_values.isna().sum()
                ),
                "observed_zero_uptake": int(
                    target_values.eq(0).sum()
                ),
                "membership_target_uptake_mismatches": int(mismatch.sum()),
            }
        )
    return rows


def _reference_overlap_context_quality(
    dataset_root: Path,
) -> Dict[str, Any]:
    """Audit the publication-time history behind overlap novelty."""
    path = dataset_root / "historical_paper_references.parquet"
    frame = pd.read_parquet(path)
    duplicate_work_rows = int(frame["work_id"].astype(str).duplicated().sum())
    length_mismatch_rows = 0
    nonprior_reference_rows = 0
    n_reference_rows = 0
    for row in frame.itertuples(index=False):
        reference_ids = list(row.reference_ids)
        reference_years = list(row.reference_years)
        if len(reference_ids) != len(reference_years):
            length_mismatch_rows += 1
        publication_year = int(row.publication_year)
        for reference_year in reference_years:
            n_reference_rows += 1
            if int(reference_year) >= publication_year:
                nonprior_reference_rows += 1
    fields = (
        frame["openalex_primary_field"].fillna("").astype(str).str.strip()
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_historical_papers": int(len(frame)),
        "n_reference_rows": int(n_reference_rows),
        "duplicate_work_rows": duplicate_work_rows,
        "length_mismatch_rows": int(length_mismatch_rows),
        "nonprior_reference_rows": int(nonprior_reference_rows),
        "missing_primary_field_rows": int(fields.eq("").sum()),
        "publication_year_min": int(frame["publication_year"].min()),
        "publication_year_max": int(frame["publication_year"].max()),
    }


def audit_v6_1_dataset(
    project_root: Path,
    config_path: Path,
) -> Tuple[Mapping[str, Any], Path]:
    """Audit grain, joins, coverage, time boundaries, and v6 immutability."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_simple_config(config_path)
    dataset_root = _resolve(
        project_root, config["paths"]["v6_1_dataset"]
    ).resolve()
    v6_root = _resolve(
        project_root, config["paths"]["v6_dataset"]
    ).resolve()
    output_root = _resolve(
        project_root, config["paths"]["v6_1_analysis"]
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    papers_path = dataset_root / "papers_primary_articles.parquet"
    candidates_path = dataset_root / "innovation_candidate_features.parquet"
    controls_path = dataset_root / "control_features_v6_1.parquet"
    opportunity_path = dataset_root / "opportunity_features.parquet"
    papers = pd.read_parquet(papers_path)
    candidates = pd.read_parquet(candidates_path)
    controls = pd.read_parquet(controls_path)
    opportunity = pd.read_parquet(opportunity_path)
    paper_ids = set(papers["paper_id"].astype(str))
    view_profiles = [
        _key_profile("papers", papers, paper_ids, papers_path),
        _key_profile(
            "innovation_candidates",
            candidates,
            paper_ids,
            candidates_path,
        ),
        _key_profile("controls_v6_1", controls, paper_ids, controls_path),
        _key_profile(
            "opportunity_features",
            opportunity,
            paper_ids,
            opportunity_path,
        ),
    ]
    copied = _copied_view_checks(
        v6_root,
        dataset_root,
        (
            "papers_common_all.parquet",
            "papers_primary_articles.parquet",
            "paper_references.parquet",
            "reference_metadata.parquet",
            "field_citation_events_aggregated.parquet",
            "targets_zero_inclusive.parquet",
            "cohort_membership.parquet",
            "opportunity_features.parquet",
        ),
    )
    k1_features = tuple(config["k1_controls"])
    missing_k1_columns = sorted(set(k1_features) - set(controls))
    if missing_k1_columns:
        raise ValueError(
            f"control view is missing K1 columns: {missing_k1_columns}"
        )
    missingness = pd.DataFrame(
        _missingness_rows(controls, k1_features)
    )
    missingness_path = output_root / "data_quality_k1_missingness.csv"
    missingness.to_csv(missingness_path, index=False)
    domain_counts = (
        papers.groupby("domain12", sort=True)["paper_id"]
        .nunique()
        .rename("n_papers")
        .reset_index()
    )
    domain_counts_path = output_root / "data_quality_domain_counts.csv"
    domain_counts.to_csv(domain_counts_path, index=False)

    openalex_manifest_path = (
        dataset_root / "target_openalex_metadata_manifest.json"
    )
    openalex_manifest = json.loads(
        openalex_manifest_path.read_text(encoding="utf-8")
    )
    overlap_context = _reference_overlap_context_quality(dataset_root)
    target_quality = _target_quality(dataset_root)
    registry_path = _resolve(
        project_root, config["paths"]["candidate_registry"]
    ).resolve()
    registry = load_candidate_registry_v6_1(registry_path)
    control_registry_path = _resolve(
        project_root, config["paths"]["control_registry"]
    ).resolve()
    missing_primary_features = sorted(
        set(registry.primary_feature_names) - set(candidates)
    )
    leakage = {
        "innovation_candidate_features": _time_leakage_rows(candidates),
        "control_features_v6_1": _time_leakage_rows(controls),
        "opportunity_features": _time_leakage_rows(opportunity),
    }
    blockers = []
    if papers["paper_id"].duplicated().any():
        blockers.append("primary paper key is not unique")
    if any(
        item["duplicate_paper_rows"]
        or item["missing_primary_papers"]
        or item["unexpected_extra_papers"]
        for item in view_profiles
    ):
        blockers.append("paper-grain views do not join one-to-one")
    if set(papers["domain12"].astype(str)) != set(EXPECTED_DOMAINS):
        blockers.append("the primary cohort does not contain exactly 12 domains")
    if int(papers["publication_year"].min()) != 1980 or int(
        papers["publication_year"].max()
    ) != 2017:
        blockers.append("the primary paper window is not 1980-2017")
    if any(leakage.values()):
        blockers.append("a publication-time view contains future information")
    if not all(item["identical"] for item in copied):
        blockers.append("a copied v6 source view changed in v6.1")
    if int(openalex_manifest["n_files_completed"]) != int(
        openalex_manifest["n_files_registered"]
    ):
        blockers.append("the local OpenAlex snapshot scan is incomplete")
    if any(
        int(item["missing_uptake_labels"]) > 0 for item in target_quality
    ):
        blockers.append("a cohort member lacks a future-uptake label")
    if any(
        int(item["membership_target_uptake_mismatches"]) > 0
        for item in target_quality
    ):
        blockers.append(
            "cohort membership and target views disagree on future uptake"
        )
    if missing_primary_features:
        blockers.append(
            "the frozen primary registry has unmaterialized features: "
            + ",".join(missing_primary_features)
        )
    if (
        overlap_context["n_historical_papers"] == 0
        or overlap_context["duplicate_work_rows"]
        or overlap_context["length_mismatch_rows"]
        or overlap_context["nonprior_reference_rows"]
        or overlap_context["missing_primary_field_rows"]
    ):
        blockers.append(
            "the reference-overlap history failed grain, field, or time checks"
        )

    report = {
        "artifact_kind": "aspr_v6_1_data_quality_audit",
        "audit_version": AUDIT_VERSION_V6_1,
        "assessment": (
            "ready_to_model" if not blockers else "needs_revision"
        ),
        "blockers": blockers,
        "dataset_grain": "one row per Nature primary article",
        "n_primary_papers": int(len(papers)),
        "n_unique_primary_papers": int(papers["paper_id"].nunique()),
        "publication_year_min": int(papers["publication_year"].min()),
        "publication_year_max": int(papers["publication_year"].max()),
        "n_domains": int(papers["domain12"].nunique()),
        "view_profiles": view_profiles,
        "publication_time_leakage_rows": leakage,
        "v6_immutable_view_checks": copied,
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "control_registry_path": str(control_registry_path),
        "control_registry_sha256": sha256_file(control_registry_path),
        "primary_innovation_features": list(
            registry.primary_feature_names
        ),
        "missing_primary_innovation_features": missing_primary_features,
        "openalex_control_metadata": openalex_manifest,
        "reference_overlap_context_quality": overlap_context,
        "target_quality_by_horizon": target_quality,
        "outputs": {
            "k1_missingness": {
                "path": str(missingness_path),
                "sha256": sha256_file(missingness_path),
            },
            "domain_counts": {
                "path": str(domain_counts_path),
                "sha256": sha256_file(domain_counts_path),
            },
        },
        "limitations": [
            "OpenAlex subfield and team metadata come from the frozen current "
            "snapshot; their historical assignment time is not observable.",
            "Whole-cohort raw innovation-feature missingness is retained and "
            "is handled only inside training folds with missing indicators.",
        ],
    }
    report_path = output_root / "data_quality_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report, report_path


__all__ = ["AUDIT_VERSION_V6_1", "audit_v6_1_dataset"]
