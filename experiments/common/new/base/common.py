"""Shared contracts, styling and frozen-input helpers for the Nature figure suite."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr-v6-1-nature-figures-mpl")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

INK = "#172033"
GRAY = "#667085"
MID_GRAY = "#98A2B3"
LIGHT_GRAY = "#D0D5DD"
PALE_GRAY = "#F2F4F7"
BLUE = "#176B87"
LIGHT_BLUE = "#9BBEC8"
ORANGE = "#D97706"
LIGHT_ORANGE = "#F4D6A0"
PURPLE = "#7353A5"
LIGHT_PURPLE = "#D4C5E8"
OLIVE = "#718355"
PINK = "#B95C83"
VERMILLION = "#B54708"
WHITE = "#FFFFFF"

ANGLE_ORDER: Tuple[str, ...] = (
    "A1_COMBINATION_RARITY",
    "A2_ATYPICALITY_CONVENTIONALITY",
    "A3_FIRST_TIME_COMBINATION",
    "A4_KNOWLEDGE_BREADTH_BALANCE",
    "A5_COGNITIVE_DISTANCE_INTEGRATION",
)

ANGLE_LABELS = {
    "A1_COMBINATION_RARITY": "Combination rarity",
    "A2_ATYPICALITY_CONVENTIONALITY": "Atypicality & conventionality",
    "A3_FIRST_TIME_COMBINATION": "First-time combinations",
    "A4_KNOWLEDGE_BREADTH_BALANCE": "Breadth & balance",
    "A5_COGNITIVE_DISTANCE_INTEGRATION": "Distance & integration",
}

ANGLE_SHORT = {
    "A1_COMBINATION_RARITY": "A1 Rarity",
    "A2_ATYPICALITY_CONVENTIONALITY": "A2 Atypicality",
    "A3_FIRST_TIME_COMBINATION": "A3 First-time",
    "A4_KNOWLEDGE_BREADTH_BALANCE": "A4 Breadth",
    "A5_COGNITIVE_DISTANCE_INTEGRATION": "A5 Integration",
}

ANGLE_COLORS = {
    "A1_COMBINATION_RARITY": BLUE,
    "A2_ATYPICALITY_CONVENTIONALITY": ORANGE,
    "A3_FIRST_TIME_COMBINATION": OLIVE,
    "A4_KNOWLEDGE_BREADTH_BALANCE": PURPLE,
    "A5_COGNITIVE_DISTANCE_INTEGRATION": PINK,
}

FEATURE_LABELS = {
    "reference_overlap_novelty_t0": "Reference-overlap novelty",
    "hypergeom_conventionality_median_t0": "Median conventionality",
    "first_time_source_pair_share": "First-pair share",
    "field_gini_balance": "Field balance",
    "reference_other_field_share": "Outside-field references",
    "field_variety": "Field variety",
    "field_disparity_cosine_mean": "Mean cognitive distance",
    "rao_stirling_integration": "Rao–Stirling integration",
}

MODEL_LABELS = {
    "k0_controls": "K0 controls",
    "k1_controls": "K1 controls",
    "innovation_only": "Innovation only",
    "b0_v6_primary_plus_k0": "B0 v6 + K0",
    "provisional_core8_plus_k1": "Provisional 8 + K1",
    "final_innovation_plus_k1": "Final 8 + K1",
    "k2_controls": "K2 controls",
    "final_innovation_plus_k2": "Final 8 + K2",
}


def configure_style() -> None:
    """Apply one restrained publication-style visual system."""
    mpl.rcParams.update(
        {
            "font.family": ["DejaVu Sans", "Microsoft YaHei"],
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "axes.edgecolor": GRAY,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 15,
            "figure.titleweight": "bold",
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "grid.color": LIGHT_GRAY,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a configuration path against the project root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    """Read one UTF-8 JSON object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write deterministic human-readable JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Path(path)


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_payload(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible payload."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(text: str, base_seed: int = 0) -> int:
    """Create a deterministic 32-bit seed from text."""
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def numeric(series: pd.Series) -> pd.Series:
    """Coerce values to finite numeric cells."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    """Compute Spearman correlation with conservative degeneracy handling."""
    paired = pd.concat([numeric(left), numeric(right)], axis=1).dropna()
    if len(paired) < 3 or paired.iloc[:, 0].nunique() < 2 or paired.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(paired.corr(method="spearman").iloc[0, 1])


def percentile_rank(values: pd.Series, ids: pd.Series | None = None) -> pd.Series:
    """Return deterministic percentiles with ID-based tie breaking."""
    frame = pd.DataFrame({"value": numeric(values)})
    if ids is None:
        frame["stable_id"] = frame.index.astype(str)
    else:
        frame["stable_id"] = ids.astype(str).to_numpy()
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame["value"].notna()
    ordered = frame.loc[valid].sort_values(["value", "stable_id"], kind="stable")
    if len(ordered):
        output.loc[ordered.index] = (np.arange(len(ordered), dtype=float) + 0.5) / len(ordered)
    return output


def grouped_percentile(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: Sequence[str],
    *,
    id_column: str,
) -> pd.Series:
    """Return percentiles within one or more frozen normalization groups."""
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    grouper: str | List[str]
    grouper = list(group_columns)
    if len(grouper) == 1:
        grouper = grouper[0]
    for _, group in frame.groupby(grouper, dropna=False, sort=True):
        output.loc[group.index] = percentile_rank(
            group[value_column],
            group[id_column],
        ).to_numpy()
    return output


def bootstrap_mean_interval(
    values: Sequence[float] | np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Return mean and percentile bootstrap interval."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(int(iterations), len(array)), replace=True)
    estimates = draws.mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def paired_bootstrap_difference(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Tuple[float, float, float, int]:
    """Return paired mean difference and percentile interval."""
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if frame.empty:
        return float("nan"), float("nan"), float("nan"), 0
    difference = frame["left"].to_numpy(float) - frame["right"].to_numpy(float)
    mean, low, high = bootstrap_mean_interval(
        difference,
        iterations=iterations,
        seed=seed,
    )
    return mean, low, high, len(difference)


def clean_axes(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    """Remove excess scaffolding while retaining axis anchors."""
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis)
        ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, panel: str, title: str) -> None:
    """Draw a consistent panel label and neutral title."""
    ax.set_title(f"{panel}  {title}", loc="left", pad=8, color=INK)


def figure_title(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    draft: bool = False,
) -> None:
    """Draw suite-level title, subtitle and optional draft status."""
    fig.suptitle(title, x=0.015, y=0.995, ha="left", color=INK)
    fig.text(0.015, 0.968, subtitle, ha="left", va="top", color=GRAY, fontsize=8.5)
    if draft:
        fig.text(
            0.985,
            0.985,
            "DRAFT — missing required human evidence",
            ha="right",
            va="top",
            color=VERMILLION,
            fontsize=9,
            fontweight="bold",
        )


def draft_panel(ax: plt.Axes, title: str, message: str) -> None:
    """Render a clear blocked-evidence panel without invented values."""
    ax.set_axis_off()
    ax.set_facecolor("#FFF8F1")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(LIGHT_ORANGE)
        spine.set_linestyle("--")
    ax.text(
        0.04,
        0.90,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=INK,
        fontweight="bold",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.55,
        "DRAFT",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=VERMILLION,
        fontsize=22,
        fontweight="bold",
        alpha=0.82,
    )
    ax.text(
        0.5,
        0.34,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=GRAY,
        fontsize=8,
        wrap=True,
    )


def export_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Export one figure to the requested deterministic formats."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        kwargs: Dict[str, Any] = {"bbox_inches": "tight"}
        if extension.lower() == "png":
            kwargs["dpi"] = int(dpi)
        fig.savefig(path, **kwargs)
        outputs[extension] = path
    return outputs


def source_record(path: Path, role: str) -> Dict[str, Any]:
    """Build one lineage record for a local frozen input."""
    path = Path(path).resolve()
    return {
        "path": str(path),
        "role": role,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def software_record() -> Dict[str, Any]:
    """Return compact runtime provenance."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
    }


@dataclass(frozen=True)
class SuitePaths:
    """Resolved paths declared by the suite configuration."""

    project_root: Path
    config_path: Path
    output_root: Path
    paths: Mapping[str, Path]

    def __getitem__(self, key: str) -> Path:
        return self.paths[key]


@dataclass
class FigureBundle:
    """All reproducible inputs needed to render one multi-panel figure."""

    figure_id: int
    title: str
    status: str
    tables: Dict[str, pd.DataFrame]
    panel_text: Dict[str, Any]
    chart_contract: Dict[str, Any]
    source_paths: List[Path]
    notes: List[str]


def resolve_suite_paths(
    config_path: Path,
    output_root: Path,
) -> Tuple[Dict[str, Any], SuitePaths]:
    """Load configuration and resolve all declared local paths."""
    config_path = Path(config_path).resolve()
    config = load_json(config_path)
    resolved = {
        key: resolve_path(PROJECT_ROOT, value)
        for key, value in config["paths"].items()
    }
    return config, SuitePaths(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        output_root=Path(output_root).resolve(),
        paths=resolved,
    )


def ensure_required_files(paths: Iterable[Path]) -> None:
    """Fail early when a frozen input is absent."""
    missing = [str(Path(path)) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required frozen inputs:\n" + "\n".join(missing))


def flatten_output_records(
    records: MutableMapping[str, Dict[str, Any]],
    *,
    prefix: str,
    paths: Mapping[str, Path],
) -> None:
    """Append hashes for a group of generated outputs."""
    for extension, path in paths.items():
        path = Path(path).resolve()
        records[f"{prefix}_{extension}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
