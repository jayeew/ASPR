"""Orchestration, archiving, and integrity audit for the Nature-dense Fig. 1."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from experiments.common.new.adapters.contracts import PRIMARY_FEATURES

from .descriptive_analysis import (
    _indicator_display_filter,
    _trajectory_display_scales,
    run_descriptive_analysis,
)
from .descriptive_render import render_descriptive_figure
from .event_data import canonical_hash, sha256_file, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DESIGN_VERSION = "fig1-old-spacious-transition-v6.4"


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(config_path: Path) -> Dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("design_version") != DESIGN_VERSION:
        raise ValueError(
            f"The canonical Fig.1 runner requires {DESIGN_VERSION}"
        )
    return config


def _load_frozen_selection(
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    path = _resolve(str(config["frozen_selection_file"]))
    if sha256_file(path) != str(config["frozen_selection_sha256"]):
        raise ValueError("Frozen selection file does not match its hash")
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================================
# Versioned archive and prepare stage
# ============================================================================


def _archive_inventory(archive_dir: Path) -> Mapping[str, Any]:
    artifacts = {
        str(path.relative_to(archive_dir)): {
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
        for path in sorted(archive_dir.rglob("*"))
        if path.is_file() and path.name != "archive_manifest.json"
    }
    return {
        "artifact_kind": "fig1_pre_nature_dense_archive",
        "archive_path": str(archive_dir.resolve()),
        "artifacts": artifacts,
    }


def _archive_predecessor_if_needed(
    config: Mapping[str, Any],
    output_dir: Path,
) -> Mapping[str, Any] | None:
    """Copy the prior canonical result once before replacing it."""
    current_manifest = output_dir / "run_manifest.json"
    if not current_manifest.is_file():
        return None
    try:
        current = json.loads(current_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        current = {}
    if current.get("design_version") == config["design_version"]:
        return None
    archive_dir = _resolve(str(config["archive"]["predecessor_archive"]))
    excluded = set(str(value) for value in config["archive"]["exclude_directories"])
    if not archive_dir.exists():
        archive_dir.mkdir(parents=True, exist_ok=False)
        for child in sorted(output_dir.iterdir()):
            if child.name.startswith("archive_") or child.name in excluded:
                continue
            target = archive_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
    inventory = _archive_inventory(archive_dir)
    inventory = {
        **inventory,
        "source_design_version": current.get("design_version"),
        "preserved_before_design_version": config["design_version"],
    }
    inventory["artifact_id"] = canonical_hash(inventory)
    write_json(archive_dir / "archive_manifest.json", inventory)
    return inventory


def _prepare(
    config: Mapping[str, Any],
    data_dir: Path,
) -> Mapping[str, Any]:
    """Verify frozen local inputs; this stage performs no network fetch."""
    paths = {
        "focal_works": data_dir / str(config["data"]["focal_works_file"]),
        "indicator_features": (
            data_dir / str(config["data"]["indicator_features_file"])
        ),
        "frozen_selection": _resolve(str(config["frozen_selection_file"])),
        "topic_short_labels": _resolve(
            str(config["topic_short_labels_file"])
        ),
    }
    replacement_manifest = config["data"].get(
        "replacement_manifest_file"
    )
    if replacement_manifest:
        paths["replacement_manifest"] = (
            data_dir / str(replacement_manifest)
        )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Fig.1 reuses frozen local data; missing artifact(s): "
            + ", ".join(missing)
        )
    expected_hashes = {
        "frozen_selection": str(config["frozen_selection_sha256"]),
        "topic_short_labels": str(config["topic_short_labels_sha256"]),
    }
    for name, expected in expected_hashes.items():
        observed = sha256_file(paths[name])
        if observed != expected:
            raise ValueError(
                f"{name} hash mismatch: expected {expected}, observed {observed}"
            )
    payload = {
        "artifact_kind": "fig1_nature_dense_prepare",
        "design_version": config["design_version"],
        "network_fetch_performed": False,
        "new_external_dataset_created": False,
        "sources": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for name, path in paths.items()
        },
    }
    payload["artifact_id"] = canonical_hash(payload)
    write_json(data_dir / "fig1_nature_dense_prepare.json", payload)
    return payload


# ============================================================================
# Deterministic artifact fingerprint
# ============================================================================


def _deterministic_paths(output_dir: Path) -> Sequence[Path]:
    paths: list[Path] = []
    for path in sorted((output_dir / "panel_data").rglob("*")):
        if path.is_file():
            paths.append(path)
    for name in (
        "analysis_manifest.json",
        "panel_text.json",
        "chart_contract.json",
        "render_manifest.json",
        "figure_full.png",
        "figure_full.svg",
        "figure_full.pdf",
    ):
        path = output_dir / name
        if path.is_file():
            paths.append(path)
    for path in sorted((output_dir / "domains").rglob("*")):
        if path.is_file():
            paths.append(path)
    for path in sorted((output_dir / "qa").rglob("*")):
        if path.is_file():
            paths.append(path)
    return paths


def _reproducibility_manifest(
    output_dir: Path,
    design_version: str,
    previous: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    hashes = {
        str(path.relative_to(output_dir)): sha256_file(path)
        for path in _deterministic_paths(output_dir)
    }
    previous_hashes = (
        previous.get("artifact_hashes", {})
        if previous
        and previous.get("design_version") == design_version
        and previous.get("complete_render_bundle") is True
        else {}
    )
    if previous_hashes:
        comparison = hashes == previous_hashes
        status = "MATCHED_PREVIOUS_SAME_VERSION" if comparison else "MISMATCH"
    else:
        comparison = None
        status = "BASELINE_CREATED"
    payload = {
        "artifact_kind": "fig1_reproducibility_fingerprint",
        "design_version": design_version,
        "complete_render_bundle": True,
        "comparison_status": status,
        "matches_previous_same_version": comparison,
        "artifact_hashes": hashes,
    }
    payload["artifact_id"] = canonical_hash(payload)
    write_json(output_dir / "reproducibility_manifest.json", payload)
    return payload


# ============================================================================
# Audit helpers
# ============================================================================


def _edge_key_rows(frame: pd.DataFrame) -> set[Tuple[str, str]]:
    return {
        tuple(sorted((str(row.source), str(row.target))))
        for row in frame.itertuples(index=False)
    }


def _transition_equations_hold(
    active_edges: pd.DataFrame,
    transitions: pd.DataFrame,
) -> bool:
    for (domain, view), group in transitions.groupby(
        ["domain", "display_scope"],
        sort=True,
    ):
        active_group = active_edges[
            active_edges["domain"].eq(domain)
            & active_edges["display_scope"].eq(view)
        ]
        skeleton_sets = []
        for stage_index in range(4):
            stage = group[group["stage_index"].eq(stage_index)]
            skeleton_sets.append(_edge_key_rows(stage))
            if stage_index == 0:
                continue
            current_actual = _edge_key_rows(
                active_group[active_group["stage_index"].eq(stage_index)]
            )
            previous_actual = _edge_key_rows(
                active_group[
                    active_group["stage_index"].eq(stage_index - 1)
                ]
            )
            current_by_status = _edge_key_rows(
                stage[stage["status"].isin(["retained", "gained"])]
            )
            previous_by_status = _edge_key_rows(
                stage[stage["status"].isin(["retained", "lost"])]
            )
            if (
                current_actual != current_by_status
                or previous_actual != previous_by_status
            ):
                return False
        if any(value != skeleton_sets[0] for value in skeleton_sets[1:]):
            return False
    return True


def _representatives_are_real(
    representatives: pd.DataFrame,
    focal: pd.DataFrame,
) -> bool:
    source = focal[
        [
            "work_id",
            "domain",
            "primary_topic_id",
            "publication_year",
        ]
    ].copy()
    source["work_id"] = source["work_id"].astype(str)
    merged = representatives.merge(
        source,
        left_on=["paper_id", "domain"],
        right_on=["work_id", "domain"],
        how="left",
        suffixes=("_display", "_source"),
        validate="many_to_one",
    )
    if merged["work_id"].isna().any():
        return False
    return bool(
        merged["topic_id"].eq(
            merged["primary_topic_id"].astype(str)
        ).all()
        and merged["publication_year_source"]
        .between(merged["start_year"], merged["end_year"])
        .all()
    )


def _annual_baselines_hold(annual: pd.DataFrame) -> bool:
    for (_, _, feature), group in annual.groupby(
        ["episode_id", "domain", "feature"],
        sort=True,
    ):
        pre = group[
            group["event_time"].isin([-3, -2, -1])
            & group["eligible"].astype(bool)
        ]["oriented_percentile_median_raw"]
        if len(pre) != 3:
            return False
        expected = float(np.median(pre.to_numpy(dtype=float)))
        observed = group["pre_yearly_median_baseline"].to_numpy(dtype=float)
        if not np.allclose(observed, expected, rtol=0, atol=1e-12):
            return False
        valid = group[group["eligible"].astype(bool)]
        if not np.allclose(
            valid["delta_median"].to_numpy(dtype=float),
            valid["oriented_percentile_median_raw"].to_numpy(dtype=float)
            - expected,
            rtol=0,
            atol=1e-12,
        ):
            return False
    return True


def _layout_separation_holds(
    nodes: pd.DataFrame,
    view: str,
    minimum: float,
) -> bool:
    """Check deterministic topic centres remain visually separated."""
    display_column = f"display_{view}"
    x_column = f"{view}_x"
    y_column = f"{view}_y"
    unique = nodes[nodes[display_column].astype(bool)].drop_duplicates(
        ["domain", "node_id"]
    )
    for _, group in unique.groupby("domain", sort=True):
        points = group[[x_column, y_column]].to_numpy(dtype=float)
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                if float(np.linalg.norm(points[left] - points[right])) < minimum:
                    return False
    return True


def _bootstrap_intervals_hold(
    effects: pd.DataFrame,
    draws: pd.DataFrame,
) -> bool:
    for row in effects.itertuples(index=False):
        sample = draws[
            draws["episode_id"].eq(str(row.episode_id))
            & draws["feature"].eq(str(row.feature))
        ]["effect"].to_numpy(dtype=float)
        if len(sample) != int(row.bootstrap_draws):
            return False
        low, high = np.quantile(sample, [0.025, 0.975])
        if not np.allclose(
            [low, high],
            [float(row.ci_low), float(row.ci_high)],
            rtol=0,
            atol=1e-12,
        ):
            return False
    return True


def _expected_png_size(
    config: Mapping[str, Any],
) -> Tuple[int, int]:
    width = float(config["plot"]["main_width_mm"]) / 25.4
    height = float(config["plot"]["main_height_mm"]) / 25.4
    dpi = int(config["plot"]["dpi"])
    return int(round(width * dpi)), int(round(height * dpi))


# ============================================================================
# Full artifact audit
# ============================================================================


def audit_descriptive_figure(
    config: Mapping[str, Any],
    data_dir: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    """Validate data lineage, transitions, statistics, and final exports."""
    panel_data = output_dir / "panel_data"
    required = (
        panel_data / "domain_selection.csv",
        panel_data / "case_refresh_audit.csv",
        panel_data / "snapshot_summary.csv",
        panel_data / "snapshot_nodes.parquet",
        panel_data / "snapshot_edges.parquet",
        panel_data / "transition_edges.parquet",
        panel_data / "representative_papers.parquet",
        panel_data / "landmark_papers.csv",
        panel_data / "indicator_window_summary.csv",
        panel_data / "indicator_selection.csv",
        panel_data / "indicator_trajectories.csv",
        panel_data / "annual_indicator_trajectories.csv",
        panel_data / "trajectory_display_scales.csv",
        panel_data / "indicator_display_filter.csv",
        panel_data / "indicator_effects.csv",
        panel_data / "indicator_effect_bootstrap.parquet",
        panel_data / "topic_label_audit.csv",
        panel_data / "community_label_selection.csv",
        panel_data / "frozen_selection_snapshot.json",
        output_dir / "analysis_manifest.json",
        output_dir / "panel_text.json",
        output_dir / "chart_contract.json",
        output_dir / "render_manifest.json",
        output_dir / "reproducibility_manifest.json",
    )
    checks: Dict[str, bool] = {
        "required_analysis_and_contracts_exist": all(
            path.is_file() for path in required
        )
    }
    focal_path = data_dir / str(config["data"]["focal_works_file"])
    indicators_path = data_dir / str(
        config["data"]["indicator_features_file"]
    )
    checks["frozen_local_sources_exist"] = bool(
        focal_path.is_file() and indicators_path.is_file()
    )
    checks["frozen_selection_hash_matches"] = (
        sha256_file(_resolve(str(config["frozen_selection_file"])))
        == str(config["frozen_selection_sha256"])
    )
    checks["topic_short_label_hash_matches"] = (
        sha256_file(_resolve(str(config["topic_short_labels_file"])))
        == str(config["topic_short_labels_sha256"])
    )
    if not checks["required_analysis_and_contracts_exist"]:
        report = {
            "artifact_kind": "fig1_cumulative_transition_audit",
            "design_version": config["design_version"],
            "status": "BLOCKED_MISSING_ARTIFACTS",
            "checks": checks,
            "passed": False,
        }
        report["artifact_id"] = canonical_hash(report)
        write_json(output_dir / "audit_report.json", report)
        return report
    frozen = _load_frozen_selection(config)
    expected_cases = sorted(
        frozen["cases"],
        key=lambda value: int(value["selection_rank"]),
    )
    expected_domains = [str(case["domain"]) for case in expected_cases]
    expected_episodes = [str(case["episode_id"]) for case in expected_cases]
    expected_counts = [len(case["features"]) for case in expected_cases]
    domain_selection = pd.read_csv(panel_data / "domain_selection.csv")
    case_refresh = pd.read_csv(panel_data / "case_refresh_audit.csv")
    summaries = pd.read_csv(panel_data / "snapshot_summary.csv")
    nodes = pd.read_parquet(panel_data / "snapshot_nodes.parquet")
    active_edges = pd.read_parquet(panel_data / "snapshot_edges.parquet")
    transitions = pd.read_parquet(panel_data / "transition_edges.parquet")
    representatives = pd.read_parquet(
        panel_data / "representative_papers.parquet"
    )
    indicator_selection = pd.read_csv(
        panel_data / "indicator_selection.csv"
    )
    annual = pd.read_csv(panel_data / "annual_indicator_trajectories.csv")
    trajectory_scales = pd.read_csv(
        panel_data / "trajectory_display_scales.csv"
    )
    indicator_display = pd.read_csv(
        panel_data / "indicator_display_filter.csv"
    )
    effects = pd.read_csv(panel_data / "indicator_effects.csv")
    bootstrap = pd.read_parquet(
        panel_data / "indicator_effect_bootstrap.parquet"
    )
    community_labels = pd.read_csv(
        panel_data / "community_label_selection.csv"
    )
    selected = domain_selection[
        domain_selection["selected"].astype(bool)
    ].sort_values("selection_rank", kind="stable")
    checks["ten_candidate_domains"] = bool(
        domain_selection["domain"].nunique()
        == len(set(config["main_domains"]))
        == 10
    )
    checks["eleven_registered_episodes"] = (
        len(domain_selection) == len(config["episodes"]) == 11
    )
    top_four_refresh = case_refresh[
        case_refresh["within_top_four_change_rank"].astype(bool)
    ]
    refresh_counts = (
        top_four_refresh.groupby("episode_id")[
            "passes_display_effect_threshold"
        ]
        .sum()
        .to_dict()
    )
    refresh_episode_id = str(
        config["selection"]["refresh_episode_id"]
    )
    minimum_refresh_count = int(
        config["selection"]["minimum_refresh_top_four_effect_count"]
    )
    checks["eleven_episode_display_refresh_is_auditable"] = bool(
        case_refresh["episode_id"].nunique() == len(config["episodes"])
        and refresh_counts.get(refresh_episode_id, 0)
        >= minimum_refresh_count
        and case_refresh.loc[
            case_refresh["used_for_descriptive_refresh"].astype(bool),
            "episode_id",
        ].drop_duplicates().tolist()
        == [refresh_episode_id]
    )
    checks["exact_frozen_four_cases"] = bool(
        selected["domain"].astype(str).tolist() == expected_domains
        and selected["episode_id"].astype(str).tolist() == expected_episodes
        and selected["eligible"].astype(bool).all()
    )
    chosen = indicator_selection[
        indicator_selection["selected"].astype(bool)
    ].sort_values(["domain", "display_rank"], kind="stable")
    actual_counts = [
        int(
            chosen[
                chosen["episode_id"].eq(case["episode_id"])
            ]["feature"].nunique()
        )
        for case in expected_cases
    ]
    checks["exact_indicator_counts_4_5_5_4"] = actual_counts == expected_counts
    checks["exact_frozen_indicator_lists"] = all(
        chosen[
            chosen["episode_id"].eq(case["episode_id"])
        ]
        .sort_values("display_rank", kind="stable")["feature"]
        .astype(str)
        .tolist()
        == [str(value) for value in case["features"]]
        for case in expected_cases
    )
    checks["only_frozen_eight_indicator_family"] = set(
        indicator_selection["feature"].astype(str)
    ) == set(PRIMARY_FEATURES)
    checks["selected_indicators_pass_original_gates"] = bool(
        chosen["eligible"].astype(bool).all()
    )
    allowed_relations = {
        "landmark_bearing_topic",
        "direct_landmark_neighbor",
        "pre_landmark_context",
        "field_backbone_context",
    }
    direct_neighbors = community_labels[
        community_labels["community_relation"].eq(
            "direct_landmark_neighbor"
        )
        & community_labels["active"].astype(bool)
    ]
    checks["community_label_relationships_are_auditable"] = bool(
        set(community_labels["community_relation"].astype(str))
        .issubset(allowed_relations)
        and community_labels.loc[
            community_labels["community_relation"].eq(
                "direct_landmark_neighbor"
            ),
            "landmark_coupling_weight",
        ].gt(0.0).all()
        and direct_neighbors.groupby("domain")["stage_index"]
        .nunique()
        .ge(1)
        .reindex(expected_domains, fill_value=False)
        .all()
    )
    snapshot_counts = summaries.sort_values(
        ["domain", "stage_index"],
        kind="stable",
    )
    snapshot_timing = snapshot_counts.merge(
        selected[["domain", "landmark_start_year"]],
        on="domain",
        how="left",
        validate="many_to_one",
    )
    counts_by_domain = snapshot_counts.groupby("domain")["paper_count"]
    ends_by_domain = snapshot_counts.groupby("domain")["end_year"]
    checks["four_cumulative_nonempty_graph_snapshots"] = bool(
        summaries.groupby("domain")["stage_index"].nunique().eq(4).all()
        and summaries["cumulative_snapshot"].astype(bool).all()
        and summaries.groupby("domain")["start_year"].nunique().eq(1).all()
        and snapshot_timing["start_year"]
        .eq(
            snapshot_timing["landmark_start_year"]
            + int(config["graph"]["history_start_offset"])
        )
        .all()
        and ends_by_domain.apply(
            lambda values: np.diff(values.to_numpy(dtype=int)).tolist()
            == [3, 3, 3]
        ).all()
        and counts_by_domain.apply(
            lambda values: bool(np.all(np.diff(values.to_numpy(dtype=int)) > 0))
        ).all()
        and summaries["paper_count"].gt(0).all()
    )
    first_snapshots = snapshot_counts[
        snapshot_counts["stage_index"].eq(0)
    ]
    final_snapshots = snapshot_counts[
        snapshot_counts["stage_index"].eq(3)
    ]
    checks["graph_snapshot_paper_counts_are_expanded"] = bool(
        first_snapshots["paper_count"]
        .ge(int(config["graph"]["minimum_baseline_snapshot_papers"]))
        .all()
        and final_snapshots["paper_count"]
        .ge(int(config["graph"]["minimum_final_snapshot_papers"]))
        .all()
    )
    checks["main_graph_limits_and_minimum_edges"] = bool(
        summaries["main_active_topic_count"]
        .le(int(config["graph"]["main_maximum_display_nodes"]))
        .all()
        and summaries["main_active_edge_count"]
        .between(
            int(config["graph"]["minimum_active_display_edges"]),
            int(
                config["graph"][
                    "main_maximum_active_edges_per_snapshot"
                ]
            ),
        )
        .all()
    )
    checks["detail_graph_limits_and_minimum_edges"] = bool(
        summaries["detail_active_topic_count"]
        .le(int(config["graph"]["detail_maximum_display_nodes"]))
        .all()
        and summaries["detail_active_edge_count"]
        .between(
            int(config["graph"]["minimum_active_display_edges"]),
            int(
                config["graph"][
                    "detail_maximum_active_edges_per_snapshot"
                ]
            ),
        )
        .all()
    )
    checks["topic_layout_centres_pass_collision_gate"] = bool(
        _layout_separation_holds(
            nodes,
            "main",
            0.78
            * float(config["graph"]["main_layout_minimum_separation"]),
        )
        and _layout_separation_holds(
            nodes,
            "detail",
            0.78
            * float(config["graph"]["detail_layout_minimum_separation"]),
        )
    )
    checks["transition_set_equations_hold"] = _transition_equations_hold(
        active_edges,
        transitions,
    )
    checks["representative_cap_and_uniqueness"] = bool(
        representatives.groupby(
            ["domain", "stage_index", "topic_id"]
        ).size().le(
            int(
                config["graph"][
                    "maximum_representative_papers_per_topic"
                ]
            )
        ).all()
        and not representatives.duplicated(
            ["domain", "stage_index", "topic_id", "paper_id"]
        ).any()
    )
    focal = pd.read_parquet(focal_path)
    checks["representative_paper_ids_are_real_and_in_window"] = (
        _representatives_are_real(representatives, focal)
    )
    checks["all_displayed_topics_have_frozen_raw_and_short_labels"] = bool(
        nodes["topic_name_raw"].fillna("").astype(str).str.len().gt(0).all()
        and nodes["topic_label"].fillna("").astype(str).str.len().gt(0).all()
    )
    landmark_states = nodes[nodes["landmark_topic"].astype(bool)]
    checks["landmark_topics_start_at_t0_and_persist"] = bool(
        len(landmark_states) > 0
        and landmark_states.loc[
            landmark_states["stage_index"].eq(0),
            "active",
        ].eq(False).all()
        and landmark_states.loc[
            landmark_states["stage_index"].eq(0),
            "suppressed_pre_landmark",
        ].astype(bool).all()
        and landmark_states.loc[
            landmark_states["stage_index"].ge(1),
            "active",
        ].astype(bool).all()
        and ~landmark_states.loc[
            landmark_states["stage_index"].ge(1),
            "suppressed_pre_landmark",
        ].astype(bool).any()
    )
    expected_times = set(range(-6, 9))
    group_sizes = annual.groupby(
        ["episode_id", "feature"],
        sort=True,
    ).size()
    checks["fifteen_annual_slots_per_selected_indicator"] = bool(
        group_sizes.eq(15).all()
        and len(group_sizes) == sum(expected_counts)
        and all(
            set(group["event_time"].astype(int)) == expected_times
            for _, group in annual.groupby(
                ["episode_id", "feature"],
                sort=True,
            )
        )
    )
    minimum_n = int(config["indicators"]["minimum_annual_valid"])
    invalid = annual["n_valid"].lt(minimum_n)
    invalid_reasons = (
        annual.loc[invalid, "missing_reason"].fillna("").astype(str)
    )
    valid_reasons = (
        annual.loc[~invalid, "missing_reason"].fillna("").astype(str)
    )
    checks["annual_missingness_is_explicit_and_not_interpolated"] = bool(
        invalid_reasons.str.len().gt(0).all()
        and annual.loc[invalid, ["delta_q25", "delta_median", "delta_q75"]]
        .isna()
        .all(axis=None)
        and valid_reasons.eq("").all()
    )
    checks["annual_baseline_formula_reproduced"] = _annual_baselines_hold(
        annual
    )
    expected_scales = _trajectory_display_scales(annual, config)
    scale_keys = ["episode_id", "domain", "feature", "display_rank"]
    observed_scales = trajectory_scales.sort_values(
        scale_keys,
        kind="stable",
    ).reset_index(drop=True)
    expected_scales = expected_scales.sort_values(
        scale_keys,
        kind="stable",
    ).reset_index(drop=True)
    scale_settings = config["indicators"]["trajectory_display_scale"]
    scale_step = float(scale_settings["round_to"])
    scale_minimum = float(scale_settings["minimum_limit"])
    scale_maximum = float(scale_settings["maximum_limit"])
    checks["feature_specific_trajectory_scales_are_frozen_and_reproduced"] = bool(
        len(observed_scales) == sum(expected_counts)
        and observed_scales[scale_keys].equals(expected_scales[scale_keys])
        and np.allclose(
            observed_scales["display_limit"],
            expected_scales["display_limit"],
            atol=1e-12,
            rtol=0.0,
        )
        and observed_scales["display_label"]
        .astype(str)
        .equals(expected_scales["display_label"].astype(str))
        and observed_scales["display_limit"]
        .between(scale_minimum, scale_maximum)
        .all()
        and np.allclose(
            observed_scales["display_limit"] / scale_step,
            np.round(observed_scales["display_limit"] / scale_step),
            atol=1e-10,
            rtol=0.0,
        )
        and observed_scales["median_clipped_count"].eq(0).all()
    )
    expected_display = _indicator_display_filter(annual, effects, config)
    display_keys = ["episode_id", "domain", "feature", "display_rank"]
    observed_display = indicator_display.sort_values(
        display_keys,
        kind="stable",
    ).reset_index(drop=True)
    expected_display = expected_display.sort_values(
        display_keys,
        kind="stable",
    ).reset_index(drop=True)
    displayed_counts = (
        observed_display[observed_display["display"].astype(bool)]
        .groupby("domain")
        .size()
    )
    checks["weak_indicator_display_filter_is_frozen_and_reproduced"] = bool(
        len(observed_display) == sum(expected_counts)
        and observed_display[display_keys].equals(
            expected_display[display_keys]
        )
        and observed_display["display"]
        .astype(bool)
        .equals(expected_display["display"].astype(bool))
        and observed_display["display_reason"]
        .astype(str)
        .equals(expected_display["display_reason"].astype(str))
        and displayed_counts.between(
            int(config["indicator_display"]["minimum_per_domain"]),
            int(config["indicator_display"]["maximum_per_domain"]),
        ).all()
        and int(observed_display["display"].astype(bool).sum())
        < sum(expected_counts)
        and observed_display[
            "display_only_not_feature_selection"
        ].astype(bool).all()
    )
    checks["one_effect_per_selected_indicator"] = bool(
        len(effects) == sum(expected_counts)
        and effects["eligible"].astype(bool).all()
    )
    checks["two_thousand_stratified_bootstraps_reproduced"] = bool(
        effects["bootstrap_draws"]
        .eq(int(config["bootstrap"]["draws"]))
        .all()
        and _bootstrap_intervals_hold(effects, bootstrap)
    )
    forbidden_tokens = ("d5", "oof", "future_impact")
    selection_columns = " ".join(
        list(domain_selection.columns)
        + list(indicator_selection.columns)
        + list(annual.columns)
        + list(effects.columns)
    ).lower()
    checks["no_future_outcome_selection_columns"] = not any(
        token in selection_columns for token in forbidden_tokens
    )
    indicator_source = pd.read_parquet(
        indicators_path,
        columns=[
            "publication_year",
            "source_max_year",
            *PRIMARY_FEATURES,
        ],
    )
    has_feature = indicator_source[list(PRIMARY_FEATURES)].notna().any(axis=1)
    checks["publication_time_only"] = bool(
        indicator_source.loc[has_feature, "source_max_year"]
        .lt(indicator_source.loc[has_feature, "publication_year"])
        .fillna(False)
        .all()
    )
    analysis = json.loads(
        (output_dir / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    disclosure = analysis["selection_disclosure"]
    checks["selection_bias_and_time_boundary_disclosed"] = bool(
        disclosure["domain_selection_uses_graph_change"]
        and disclosure["indicator_selection_uses_indicator_change"]
        and disclosure["display_refresh_used_annual_and_bootstrap_contrasts"]
        and disclosure["selection_frozen_before_current_rerender"]
        and not disclosure["future_impact_outcome_used"]
        and not disclosure["d5_used"]
        and not disclosure["oof_prediction_used"]
    )
    contract = json.loads(
        (output_dir / "chart_contract.json").read_text(encoding="utf-8")
    )
    checks["claim_boundary_is_noncausal_and_nonrepresentative"] = bool(
        "does not establish" in contract["claim_boundary"].lower()
        and "representative" in contract["claim_boundary"].lower()
        and contract["selection_disclosure"]["selection_is_exploratory"]
    )
    visual_encoding = contract["visual_encoding"]
    checks["zoom_and_common_effect_scales_are_explicit"] = bool(
        "explicit" in visual_encoding["trajectory_scale"].lower()
        and "shared" in visual_encoding["effect_scale"].lower()
        and np.isclose(
            float(
                config["indicators"]["trajectory_display_scale"][
                    "shared_effect_limit"
                ]
            ),
            0.70,
        )
    )
    checks["display_filter_is_disclosed_as_display_only"] = bool(
        contract["indicator_display_rule"]["role"]
        == "display_only_not_feature_selection"
        and not contract["selection_disclosure"][
            "display_filter_changes_model_features"
        ]
    )
    rendered: list[Path] = [
        output_dir / f"figure_full.{extension}"
        for extension in ("png", "svg", "pdf")
    ]
    for domain in expected_domains:
        rendered.extend(
            output_dir
            / "domains"
            / domain
            / f"figure_{domain}.{extension}"
            for extension in ("png", "svg", "pdf")
        )
    checks["combined_and_four_domain_figures_exist"] = bool(
        len(rendered) == 15
        and all(path.is_file() and path.stat().st_size > 0 for path in rendered)
    )
    expected_width, expected_height = _expected_png_size(config)
    observed_width, observed_height = Image.open(
        output_dir / "figure_full.png"
    ).size
    checks["main_png_is_183_by_168_mm_at_600_dpi"] = bool(
        abs(observed_width - expected_width) <= 1
        and abs(observed_height - expected_height) <= 1
        and int(config["plot"]["dpi"]) == 600
    )
    svg_text = (output_dir / "figure_full.svg").read_text(
        encoding="utf-8"
    )
    pdf_bytes = (output_dir / "figure_full.pdf").read_bytes()
    checks["inactive_topics_are_not_rendered_or_legend_labeled"] = bool(
        "inactive_topics" not in visual_encoding
        and "not rendered"
        in visual_encoding["inactive_topic_policy"].lower()
        and "inactive topic" not in svg_text.lower()
    )
    checks["dejavu_sans_is_editable_or_embedded"] = bool(
        "DejaVu Sans" in svg_text
        and (b"DejaVuSans" in pdf_bytes or b"DejaVu Sans" in pdf_bytes)
    )
    render = json.loads(
        (output_dir / "render_manifest.json").read_text(encoding="utf-8")
    )
    checks["no_text_is_clipped_outside_canvas"] = all(
        int(value["out_of_canvas_text_count"]) == 0
        for value in render["layout_qa"].values()
    )
    checks["grayscale_and_deuteranopia_previews_exist"] = all(
        Path(value["path"]).is_file()
        for value in render["accessibility_previews"].values()
    )
    reproducibility = json.loads(
        (output_dir / "reproducibility_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    checks["deterministic_hash_baseline_or_match"] = (
        reproducibility["comparison_status"]
        in {"BASELINE_CREATED", "MATCHED_PREVIOUS_SAME_VERSION"}
    )
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "artifact_kind": "fig1_cumulative_transition_audit",
        "design_version": config["design_version"],
        "status": (
            "DESCRIPTIVE_SELECTED_CASES"
            if all(checks.values())
            else "FAILED_AUDIT"
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": (
            "Passing this audit establishes local reproducibility, "
            "publication-time data safety, transition accounting, explicit "
            "missingness, and deterministic descriptive intervals. It does "
            "not make this selected-case figure causal or representative."
        ),
        "observed_png_pixels": {
            "width": int(observed_width),
            "height": int(observed_height),
        },
    }
    report["artifact_id"] = canonical_hash(report)
    write_json(output_dir / "audit_report.json", report)
    return report


# ============================================================================
# Run manifest and public entry
# ============================================================================


def _run_manifest(
    config_path: Path,
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    stage: str,
    audit: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    artifacts: Dict[str, Any] = {}
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "run_manifest.json"
            and not any(part.startswith("archive_") for part in path.parts)
            and "design_schematic" not in path.parts
        ):
            artifacts[str(path.relative_to(output_dir))] = {
                "path": str(path.resolve()),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    identity = {
        "artifact_kind": "fig1_cumulative_transition_run",
        "design_version": config["design_version"],
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "status": (
            "DESCRIPTIVE_SELECTED_CASES"
            if audit and audit.get("passed")
            else "INCOMPLETE_OR_FAILED"
        ),
        "passed": bool(audit and audit.get("passed")),
        "artifacts": artifacts,
    }
    return {**identity, "artifact_id": canonical_hash(identity)}


def run_descriptive_figure(
    config_path: Path,
    stage: str = "all",
) -> Mapping[str, Any]:
    """Execute prepare, analysis, render, reproducibility, and audit stages."""
    if stage not in {"prepare", "run", "plot", "audit", "all"}:
        raise ValueError(f"Unsupported stage: {stage}")
    config_path = config_path.resolve()
    config = _load_config(config_path)
    output_dir = _resolve(str(config["output_dir"]))
    data_dir = _resolve(str(config["data_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if stage in {"run", "plot", "all"}:
        _archive_predecessor_if_needed(config, output_dir)
    previous_repro: Mapping[str, Any] | None = None
    repro_path = output_dir / "reproducibility_manifest.json"
    if repro_path.is_file():
        try:
            previous_repro = json.loads(
                repro_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            previous_repro = None
    audit: Mapping[str, Any] | None = None
    if stage in {"prepare", "all"}:
        _prepare(config, data_dir)
    if stage in {"run", "all"}:
        run_descriptive_analysis(config, data_dir, output_dir)
    if stage in {"plot", "all"}:
        render_descriptive_figure(config, output_dir)
    if stage in {"plot", "all"}:
        _reproducibility_manifest(
            output_dir,
            str(config["design_version"]),
            previous_repro,
        )
    if stage in {"audit", "all"}:
        audit = audit_descriptive_figure(config, data_dir, output_dir)
    manifest = _run_manifest(
        config_path,
        config,
        output_dir,
        stage=stage,
        audit=audit,
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


__all__ = ["audit_descriptive_figure", "run_descriptive_figure"]
