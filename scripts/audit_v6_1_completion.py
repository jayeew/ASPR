"""Audit the implemented ASPR v6.1 plan requirement by requirement."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.candidate_registry_v6_1 import (
    EXPECTED_ANGLES,
    load_candidate_registry_v6_1,
    verify_search_log,
)
from aspr.nature_multihorizon.modeling_v6_1 import (
    build_v6_1_feature_sets,
    load_simple_config,
)
from aspr.nature_multihorizon.source_audit_v6 import sha256_file


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "nature_multihorizon" / "v6_1_simple.json"
)
REQUIRED_SEARCH_FIELDS = {
    "search_id",
    "database",
    "query",
    "date",
    "page_or_sort",
    "result_titles_screened",
    "result_dois_screened",
    "deduplication",
    "decision",
}
REQUIRED_FAMILIES = {
    "commonness_lower_tail",
    "commonness_mean",
    "thresholded_pair_rarity",
    "prior_reference_overlap_mean",
    "z_distribution_left_tail",
    "z_distribution_centre",
    "exact_hypergeometric_z_left_tail",
    "exact_hypergeometric_z_centre",
    "first_pair_incidence",
    "distance_weighted_first_pairs",
    "future_confirmed_first_pairs",
    "category_variety",
    "focal_field_breadth",
    "category_balance",
    "hill_effective_categories",
    "unweighted_field_distance",
    "share_weighted_integration_composite",
    "multiplicative_div_composite",
    "effective_similarity_diversity",
    "historical_community_bridging",
    "network_coherence",
    "semantic_reference_distance",
}
EXPECTED_FOLDS = (
    (1985, 1986, 1999),
    (1999, 2000, 2004),
    (2004, 2005, 2009),
    (2009, 2010, 2012),
    (2012, 2013, 2013),
    (2013, 2014, 2017),
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _single(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern}, found {len(paths)}")
    return paths[0]


def _check(
    rows: List[Dict[str, Any]],
    check_id: str,
    requirement: str,
    passed: bool,
    evidence: Any,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "requirement": requirement,
            "passed": bool(passed),
            "evidence": evidence,
        }
    )


def audit(config_path: Path) -> Path:
    """Create a machine-readable completion audit and fail on any gap."""
    config_path = config_path.resolve()
    config = load_simple_config(config_path)
    catalog_path = _resolve(config["paths"]["candidate_catalog"])
    registry_path = _resolve(config["paths"]["candidate_registry"])
    control_path = _resolve(config["paths"]["control_registry"])
    dataset_root = _resolve(config["paths"]["v6_1_dataset"])
    analysis_root = _resolve(config["paths"]["v6_1_analysis"])
    catalog = load_candidate_registry_v6_1(catalog_path)
    registry = load_candidate_registry_v6_1(registry_path)
    control = _load(control_path)
    search = _load(verify_search_log(catalog, PROJECT_ROOT))
    screening = _load(_single(analysis_root, "screening_*/screening_manifest.json"))
    oof = _load(_single(analysis_root, "oof_*/oof_run_manifest.json"))
    audit_report = _load(analysis_root / "data_quality_report.json")
    reproducibility = _load(analysis_root / "reproducibility_report.json")
    validation = _load(analysis_root / "validation_summary.json")
    decisions = pd.read_csv(screening["outputs"]["decisions"]["path"])
    metrics = pd.read_csv(oof["outputs"]["metrics"]["path"])
    comparisons = pd.read_csv(oof["outputs"]["comparisons"]["path"])
    domain_metrics = pd.read_csv(oof["outputs"]["domain_metrics"]["path"])
    feature_sets = build_v6_1_feature_sets(registry, config)
    rows: List[Dict[str, Any]] = []

    primary = [
        candidate
        for candidate in registry.candidates.values()
        if candidate.final_role == "primary"
    ]
    primary_angles = {candidate.angle_id for candidate in primary}
    _check(
        rows,
        "C01",
        "Exactly five source-backed observation angles are retained.",
        set(registry.observation_angles) == EXPECTED_ANGLES
        and primary_angles == EXPECTED_ANGLES,
        {"angles": sorted(primary_angles), "n_primary": len(primary)},
    )
    _check(
        rows,
        "C02",
        "The candidate census covers every predeclared mathematical family.",
        REQUIRED_FAMILIES.issubset(
            {item.mathematical_family for item in catalog.candidates.values()}
        ),
        {
            "n_candidates": len(catalog.candidates),
            "n_sources": len(catalog.sources),
        },
    )
    search_records_valid = all(
        REQUIRED_SEARCH_FIELDS.issubset(item)
        and len(item["result_titles_screened"])
        == len(item["result_dois_screened"])
        for item in search["search_records"]
    )
    databases = " ".join(item["database"] for item in search["search_records"])
    required_database_tokens = (
        "Crossref",
        "OpenAlex",
        "PubMed",
        "Publisher",
        "Google Scholar",
        "Chinese",
    )
    _check(
        rows,
        "C03",
        "The multi-source bilingual search log is complete and reproducible.",
        search_records_valid
        and search["cutoff_date"] == "2026-07-24"
        and all(token in databases for token in required_database_tokens),
        {
            "records": len(search["search_records"]),
            "cutoff": search["cutoff_date"],
        },
    )
    chase = search["citation_chasing"]
    _check(
        rows,
        "C04",
        "Citation chasing stops after two consecutive empty family rounds.",
        len(chase) >= 3
        and not chase[-1]["new_mathematical_families"]
        and not chase[-2]["new_mathematical_families"],
        {"last_rounds": [chase[-2]["round"], chase[-1]["round"]]},
    )
    primary_evidence_valid = all(
        item.original_source_ids
        and item.paper_application_source_ids
        and all(
            registry.sources[source_id].peer_reviewed
            for source_id in (
                *item.original_source_ids,
                *item.paper_application_source_ids,
            )
        )
        for item in primary
    )
    _check(
        rows,
        "C05",
        "Every primary metric has peer-reviewed formula and paper-level evidence.",
        primary_evidence_valid,
        {"primary_ids": [item.candidate_id for item in primary]},
    )
    _check(
        rows,
        "C06",
        "Every primary metric passes I1-I10 and all frozen runtime thresholds.",
        all(all(item.gate_checks.values()) for item in primary),
        {
            "screening_artifact_id": screening["artifact_id"],
            "future_outcomes_used": screening["lineage"][
                "future_influence_outcomes_used"
            ],
        },
    )
    _check(
        rows,
        "C07",
        "Screening and registry freeze occur without outcomes or network data.",
        screening["lineage"]["future_influence_outcomes_used"] is False
        and screening["lineage"]["network_used"] is False
        and all(item.oof_used_for_selection is False for item in primary),
        screening["lineage"],
    )
    distance_rows = decisions.set_index("candidate_id").loc[
        ["A3.FIRST_DISTANCE_MEAN", "A3.FIRST_DISTANCE_SUM"]
    ]
    _check(
        rows,
        "C08",
        "First-pair distance debt stays excluded when coverage fails.",
        distance_rows["proposed_final_role"].eq("excluded").all()
        and distance_rows["coverage_pass"].eq(0).all(),
        distance_rows[
            [
                "overall_coverage",
                "minimum_domain_coverage",
                "proposed_final_role",
            ]
        ].to_dict("index"),
    )
    _check(
        rows,
        "C09",
        "Novelty-U/Uzzi approximations are downgraded when fidelity or stability fails.",
        all(
            registry.candidates[item].final_role != "primary"
            for item in ("A1.NOVELTY_U", "A2.UZZI_P10", "A2.UZZI_MEDIAN")
        ),
        {
            item: registry.candidates[item].final_role
            for item in ("A1.NOVELTY_U", "A2.UZZI_P10", "A2.UZZI_MEDIAN")
        },
    )
    final_names = tuple(item.code_name for item in primary)
    _check(
        rows,
        "C10",
        "The main model reads all and only frozen primary innovation features.",
        tuple(feature_sets["final_innovation_plus_k1"])
        == tuple((*config["k1_controls"], *final_names)),
        {"primary_features": list(final_names)},
    )
    _check(
        rows,
        "C11",
        "K0/K1/K2 controls have the fixed sizes and source records.",
        len(config["k0_controls"]) == 5
        and len(config["k1_controls"]) == 11
        and len(config["k2_additional_controls"]) == 5
        and all(item["source_ids"] for item in control["features"].values()),
        {
            "k0": len(config["k0_controls"]),
            "k1": len(config["k1_controls"]),
            "k2_additional": len(config["k2_additional_controls"]),
        },
    )
    _check(
        rows,
        "C12",
        "Only local frozen experimental data are used and OpenAlex scanning is offline.",
        config["network_policy_for_experiment"] == "forbidden"
        and config["raw_data_policy"] == "local_frozen_only"
        and audit_report["openalex_control_metadata"]["network_used"] is False
        and audit_report["openalex_control_metadata"]["coverage"] == 1.0,
        audit_report["openalex_control_metadata"],
    )
    immutable = audit_report["v6_immutable_view_checks"]
    _check(
        rows,
        "C13",
        "v6 frozen input views are byte-identical and v6.1 uses independent outputs.",
        all(item["identical"] for item in immutable)
        and config["paths"]["v6_dataset"] != config["paths"]["v6_1_dataset"]
        and "v6_1_r5" in config["paths"]["v6_1_analysis"],
        {"n_identical_views": len(immutable)},
    )
    _check(
        rows,
        "C14",
        "Data grain, joins, leakage, coverage, and target semantics pass audit.",
        audit_report["assessment"] == "ready_to_model"
        and not audit_report["blockers"]
        and audit_report["n_primary_papers"]
        == audit_report["n_unique_primary_papers"]
        and all(
            count == 0
            for count in audit_report["publication_time_leakage_rows"].values()
        ),
        {
            "papers": audit_report["n_primary_papers"],
            "domains": audit_report["n_domains"],
        },
    )
    folds = tuple(
        (
            int(item["train_year_max"]),
            int(item["test_year_min"]),
            int(item["test_year_max"]),
        )
        for item in config["temporal_folds"]
    )
    _check(
        rows,
        "C15",
        "The fixed six-fold 1980-2017 temporal OOF protocol is used.",
        folds == EXPECTED_FOLDS and config["model"]["parameter_id"] == "medium",
        {"folds": folds, "parameter_id": config["model"]["parameter_id"]},
    )
    _check(
        rows,
        "C16",
        "D5 is headline; D3/D8 are directional; conditional Spearman is absent.",
        config["main_horizon"] == 5
        and tuple(config["supplementary_horizons"]) == (3, 8)
        and oof["conditional_spearman_reported"] is False
        and oof["feature_selection_from_oof"] is False
        and oof["parameter_selection_from_oof"] is False,
        {
            "headline": oof["headline_metric"],
            "conditional_reported": oof["conditional_spearman_reported"],
        },
    )
    d5 = metrics.set_index(["horizon", "model_id"]).loc[
        (5, "final_innovation_plus_k1"), "spearman_expected"
    ]
    comparison = comparisons.set_index("baseline_model_id")
    _check(
        rows,
        "C17",
        "All D5 performance, noninferiority, and incremental-value gates pass.",
        bool(oof["acceptance"]["all_required_gates_pass"])
        and float(d5) >= 0.75
        and float(comparison.loc["k1_controls", "gain_ci_low"]) > 0.0
        and float(
            comparison.loc["b0_v6_primary_plus_k0", "gain_ci_low"]
        )
        >= -0.005,
        {
            "d5": float(d5),
            "gain_vs_k1": float(
                comparison.loc["k1_controls", "spearman_gain"]
            ),
            "gain_vs_k1_ci_low": float(
                comparison.loc["k1_controls", "gain_ci_low"]
            ),
        },
    )
    _check(
        rows,
        "C18",
        "D3/D8 directions are positive and all twelve domains remain.",
        oof["acceptance"]["d3_gain_over_k1"] > 0
        and oof["acceptance"]["d8_gain_over_k1"] > 0
        and domain_metrics["domain12"].nunique() == 12,
        {
            "d3_gain": oof["acceptance"]["d3_gain_over_k1"],
            "d8_gain": oof["acceptance"]["d8_gain_over_k1"],
            "domains": int(domain_metrics["domain12"].nunique()),
        },
    )
    _check(
        rows,
        "C19",
        "Output hashes, same-label fairness, and a fresh 66-cell replay pass.",
        reproducibility["assessment"] == "pass"
        and reproducibility["full_replay_checkpoint_root_preexisting"] is False
        and reproducibility["full_replay_exact_prediction_match"] is True
        and reproducibility["n_full_replay_checkpoints"] == 66
        and reproducibility["same_test_papers_and_labels_across_models"] is True,
        reproducibility,
    )
    _check(
        rows,
        "C20",
        "Every OOF output hash in the final manifest resolves unchanged.",
        all(
            sha256_file(Path(item["path"])) == item["sha256"]
            for item in oof["outputs"].values()
        ),
        {"n_outputs": len(oof["outputs"]), "artifact_id": oof["artifact_id"]},
    )
    _check(
        rows,
        "C21",
        "The final full Nature multihorizon regression suite passes.",
        validation["assessment"] == "pass"
        and validation["failed_tests"] == 0
        and validation["passed_tests"] >= 138,
        validation,
    )

    failed = [item for item in rows if not item["passed"]]
    report: Dict[str, Any] = {
        "artifact_kind": "aspr_v6_1_plan_completion_audit",
        "assessment": "pass" if not failed else "fail",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_catalog_sha256": sha256_file(catalog_path),
        "candidate_registry_sha256": sha256_file(registry_path),
        "screening_artifact_id": screening["artifact_id"],
        "oof_artifact_id": oof["artifact_id"],
        "n_checks": len(rows),
        "n_passed": len(rows) - len(failed),
        "n_failed": len(failed),
        "checks": rows,
    }
    report["artifact_id"] = _canonical_hash(report)
    output = analysis_root / "completion_audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if failed:
        raise ValueError(
            "completion audit failed: "
            + ", ".join(item["check_id"] for item in failed)
        )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(audit(args.config))


if __name__ == "__main__":
    main()
