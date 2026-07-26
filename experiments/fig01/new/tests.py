"""Post-run checks for the canonical current-formula Fig.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

from experiments.common.new.adapters.contracts import PRIMARY_FEATURES
from experiments.fig01.new.run import run_figure1


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fig01" / "new"


def load_manifest() -> Dict[str, Any]:
    """Load the canonical Fig.1 manifest."""
    path = OUTPUT_DIR / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            "Run `python3 -m experiments.fig01.new.run --stage all` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse test options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Rebuild Fig.1 before validating its artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate formula scope, time boundary, and exported figure files."""
    args = parse_args(argv)
    if args.full:
        run_figure1(Path(__file__).with_name("config.json"), stage="all")
    manifest = load_manifest()
    failed = [name for name, passed in manifest["checks"].items() if not passed]
    if failed:
        raise AssertionError(f"Failed restoration checks: {failed}")
    panel_data = OUTPUT_DIR / "panel_data"
    for obsolete_name in (
        "perturbation_metrics.csv",
        "dominant_parameter_trajectories.csv",
        "angle_window_trajectories.csv",
    ):
        if (panel_data / obsolete_name).exists():
            raise AssertionError(
                f"Historical metric output must not exist: {obsolete_name}"
            )
    indicator_map = pd.read_csv(panel_data / "primary_indicator_map.csv")
    if tuple(indicator_map["feature"]) != tuple(PRIMARY_FEATURES):
        raise AssertionError("Output does not contain the frozen eight features")
    if not indicator_map["final_role"].eq("primary").all():
        raise AssertionError("A non-primary indicator entered the figure")
    selection = pd.read_csv(panel_data / "indicator_selection.csv")
    selected = selection[selection["selected"]]
    selected_counts = selected.groupby("domain").size()
    if len(selected_counts) != 4 or not selected_counts.between(4, 5).all():
        raise AssertionError(
            f"Expected 4–5 selected indicators per domain: {selected_counts}"
        )
    if (
        selected.groupby("domain")["feature"]
        .apply(lambda values: tuple(sorted(values)))
        .nunique()
        == 1
    ):
        raise AssertionError("All domains unexpectedly selected the same set")
    trajectories = pd.read_csv(
        panel_data / "indicator_trajectories.csv"
    )
    plotted = trajectories[trajectories["selected"]]
    if plotted["oriented_percentile_median"].isna().any():
        raise AssertionError("A plotted indicator trajectory contains missing data")
    snapshots = pd.read_csv(panel_data / "snapshot_summary.csv")
    if snapshots["displayed_topic_count"].le(0).any():
        raise AssertionError("A graph snapshot is empty")
    matched = pd.read_parquet(
        panel_data / "matched_paper_indicator_features.parquet",
        columns=["publication_year", "source_max_year"],
    )
    comparable = matched["source_max_year"].notna()
    if (
        matched.loc[comparable, "source_max_year"]
        >= matched.loc[comparable, "publication_year"]
    ).any():
        raise AssertionError("Publication-time boundary failed")
    source = (
        PROJECT_ROOT / "experiments" / "fig01" / "new" / "run.py"
    ).read_text(encoding="utf-8")
    forbidden_calls = (
        "compute_" + "perturbation_metrics(",
        "draw_" + "metric_panel(",
        "dominant_" + "parameter_table(",
    )
    found = [call for call in forbidden_calls if call in source]
    if found:
        raise AssertionError(f"Historical formula calls found: {found}")
    for extension in ("png", "svg", "pdf"):
        artifact = manifest["rendered"][extension]
        path = Path(artifact["path"])
        expected = OUTPUT_DIR / f"figure_full.{extension}"
        if path.resolve() != expected.resolve():
            raise AssertionError(f"Noncanonical artifact name: {path}")
        if not path.exists() or path.stat().st_size != int(artifact["size_bytes"]):
            raise AssertionError(f"Missing or size-mismatched artifact: {path}")
    if len(manifest["domain_rendered"]) != 4:
        raise AssertionError("Expected four domain-specific figure bundles")
    for slug, artifacts in manifest["domain_rendered"].items():
        for extension in ("png", "svg", "pdf"):
            artifact = artifacts[extension]
            path = Path(artifact["path"])
            expected = (
                OUTPUT_DIR
                / "domains"
                / slug
                / f"figure_{slug}.{extension}"
            )
            if path.resolve() != expected.resolve():
                raise AssertionError(f"Noncanonical domain artifact: {path}")
            if (
                not path.exists()
                or path.stat().st_size != int(artifact["size_bytes"])
            ):
                raise AssertionError(
                    f"Missing or size-mismatched artifact: {path}"
                )
    print("Canonical current-formula Fig.1 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
