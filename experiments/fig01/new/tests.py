"""Deterministic unit and artifact checks for the current Fig. 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.common.new.adapters.contracts import PRIMARY_FEATURES
from experiments.fig01.new.descriptive import audit_descriptive_figure
from experiments.fig01.new.descriptive_analysis import (
    _bootstrap_year_medians,
    _graph_snapshot_windows,
    _indicator_display_filter,
    _rank_and_select_episodes,
    _select_indicators,
    _stage_windows,
    _weighted_jaccard_distance,
)
from experiments.fig01.new.event_data import sha256_file
from experiments.fig01.new.run import run_figure1


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).with_name("config.json")


def _candidate_screen_rows(config: dict) -> pd.DataFrame:
    rows = []
    for index, episode in enumerate(config["episodes"]):
        rows.append(
            {
                "episode_id": str(episode["episode_id"]),
                "domain": str(episode["domain"]),
                "pre_late_edge_turnover": index / 10,
                "mean_successive_edge_turnover": index / 11,
                "pre_late_topic_turnover": index / 12,
                "absolute_log_edge_count_change": index / 13,
                "absolute_log_effective_topic_change": index / 14,
                "eligible": True,
            }
        )
    return pd.DataFrame(rows)


def _indicator_rows(config: dict, frozen: dict) -> pd.DataFrame:
    rows = []
    for case in frozen["cases"]:
        for feature_index, feature in enumerate(PRIMARY_FEATURES):
            for stage_index in range(4):
                rows.append(
                    {
                        "episode_id": str(case["episode_id"]),
                        "domain": str(case["domain"]),
                        "stage_index": stage_index,
                        "feature": feature,
                        "n_valid": 100,
                        "coverage": 1.0,
                        "oriented_percentile_median": (
                            0.1 + feature_index * 0.01 * stage_index
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _unit_checks() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["design_version"] == "fig1-old-spacious-transition-v6.4"
    assert len(set(config["main_domains"])) == 10
    assert len(config["episodes"]) == 11
    assert config["selection"]["display_domain_count"] == 4
    assert np.isclose(
        sum(config["selection"]["domain_score_weights"].values()),
        1.0,
    )
    selection_path = PROJECT_ROOT / config["frozen_selection_file"]
    assert (
        sha256_file(selection_path)
        == config["frozen_selection_sha256"]
    )
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    assert [len(case["features"]) for case in frozen["cases"]] == [4, 5, 5, 4]
    assert [case["domain"] for case in frozen["cases"]] == [
        "crispr",
        "graphene_2d_materials",
        "click_chemistry_cuaac",
        "genome_wide_association_studies",
    ]
    assert frozen["cases"][-1]["episode_id"] == "gwas_2007"
    windows = _stage_windows(config["episodes"][0], config)
    assert len(windows) == 4
    assert all(
        row["end_year"] - row["start_year"] + 1 == 3
        for row in windows
    )
    graph_windows = _graph_snapshot_windows(
        config["episodes"][0],
        config,
    )
    graph_widths = [
        row["end_year"] - row["start_year"] + 1
        for row in graph_windows
    ]
    assert graph_widths == [
        6,
        9,
        12,
        15,
    ]
    increment_widths = [
        row["increment_end_year"] - row["increment_start_year"] + 1
        for row in graph_windows
    ]
    assert increment_widths == [
        6,
        3,
        3,
        3,
    ]
    assert config["graph"]["cumulative_snapshots"] is True
    assert config["graph"]["history_start_offset"] == -6
    assert tuple(PRIMARY_FEATURES) == (
        "reference_overlap_novelty_t0",
        "hypergeom_conventionality_median_t0",
        "first_time_source_pair_share",
        "field_gini_balance",
        "reference_other_field_share",
        "field_variety",
        "field_disparity_cosine_mean",
        "rao_stirling_integration",
    )
    left = {("a", "b"): 0.5, ("b", "c"): 0.5}
    right = {("a", "b"): 0.5, ("b", "d"): 0.5}
    assert np.isclose(_weighted_jaccard_distance(left, left), 0.0)
    assert np.isclose(
        _weighted_jaccard_distance(left, right),
        2.0 / 3.0,
    )
    screened = _rank_and_select_episodes(
        _candidate_screen_rows(config),
        config,
    )
    assert screened["selected"].sum() == 4
    selection, trajectories = _select_indicators(
        _indicator_rows(config, frozen),
        config,
        frozen,
    )
    selected_counts = (
        selection[selection["selected"].astype(bool)]
        .groupby("episode_id")["feature"]
        .nunique()
        .to_dict()
    )
    assert selected_counts == {
        case["episode_id"]: len(case["features"])
        for case in frozen["cases"]
    }
    assert trajectories["episode_id"].nunique() == 4
    values = np.arange(1.0, 6.0)
    first = _bootstrap_year_medians(
        values,
        25,
        np.random.default_rng(42),
    )
    second = _bootstrap_year_medians(
        values,
        25,
        np.random.default_rng(42),
    )
    assert np.array_equal(first, second)
    annual_rows = []
    for display_rank, feature in enumerate(("f0", "f1", "f2", "f3"), 1):
        for event_time in range(-6, 9):
            value = (
                0.03 * event_time
                if feature == "f1"
                else 0.01 * event_time
            )
            annual_rows.append(
                {
                    "episode_id": "e0",
                    "domain": "d0",
                    "feature": feature,
                    "display_rank": display_rank,
                    "event_time": event_time,
                    "eligible": True,
                    "delta_median": value,
                }
            )
    effect_frame = pd.DataFrame(
        {
            "episode_id": ["e0"] * 4,
            "domain": ["d0"] * 4,
            "feature": ["f0", "f1", "f2", "f3"],
            "effect": [0.20, 0.01, 0.02, 0.00],
        }
    )
    display_filter = _indicator_display_filter(
        pd.DataFrame(annual_rows),
        effect_frame,
        config,
    )
    assert display_filter.loc[
        display_filter["display"].astype(bool),
        "feature",
    ].tolist() == ["f0", "f1", "f2"]
    assert config["indicators"]["annual_start_offset"] == -6
    assert config["indicators"]["annual_end_offset"] == 8
    assert config["indicators"]["minimum_annual_valid"] == 15
    scale = config["indicators"]["trajectory_display_scale"]
    assert scale["mode"] == "feature_specific_symmetric_zoom"
    assert scale["minimum_limit"] == 0.2
    assert scale["maximum_limit"] == 0.7
    assert scale["round_to"] == 0.05
    assert scale["shared_effect_limit"] == 0.7
    display = config["indicator_display"]
    assert display["role"] == "display_only_not_feature_selection"
    assert display["paired_panels_require_late_pre_effect"] is True
    assert display["minimum_absolute_late_pre_effect"] == 0.09
    assert display["minimum_annual_peak_to_peak"] == 0.35
    assert display["minimum_per_domain"] == 3
    assert display["maximum_per_domain"] == 4
    assert config["bootstrap"]["draws"] == 2000
    assert config["plot"]["main_width_mm"] == 183
    assert config["plot"]["main_height_mm"] == 168
    assert config["plot"]["dpi"] == 600


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete local analysis/render pipeline first.",
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run deterministic unit tests without generated artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _unit_checks()
    if args.full:
        run_figure1(CONFIG_PATH, stage="all")
    if not args.unit_only:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        output_dir = PROJECT_ROOT / config["output_dir"]
        data_dir = PROJECT_ROOT / config["data_dir"]
        report = audit_descriptive_figure(config, data_dir, output_dir)
        failed = [
            name
            for name, passed in report["checks"].items()
            if not passed
        ]
        if failed:
            raise AssertionError(
                f"Fig.1 artifact checks failed: {failed}"
            )
    print("Fig.1 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
