"""Deterministic draw-only renderers for release-bound Fig.1--Fig.10 views."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr-matplotlib")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


MECHANISMS = (
    "boundary_perturbation",
    "community_diffusion",
    "interdisciplinarity",
    "knowledge_recombination",
    "knowledge_brokerage",
)


def _tables(view_dir: Path) -> Dict[str, pd.DataFrame]:
    root = Path(view_dir).resolve()
    manifest_path = root / "view_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("view_manifest.json has no plot-data outputs")
    tables: Dict[str, pd.DataFrame] = {}
    declared = set()
    for record in outputs:
        relative = Path(str(record.get("path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 2
            or relative.parts[0] != "data"
            or relative.suffix.lower() != ".csv"
        ):
            raise ValueError(f"Invalid renderer plot-data path: {relative}")
        path = (root / relative).resolve()
        if path.parent != (root / "data").resolve() or not path.is_file():
            raise FileNotFoundError(path)
        if relative.stem in tables:
            raise ValueError(f"Duplicate renderer table name: {relative.stem}")
        tables[relative.stem] = pd.read_csv(path, low_memory=False)
        declared.add(relative.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in (root / "data").iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual != declared:
        raise ValueError("Renderer data directory does not match view manifest")
    return tables


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")


def _message(ax: plt.Axes, title: str, message: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(0.02, 0.88, message, transform=ax.transAxes, va="top", wrap=True)
    ax.axis("off")


def _evidence_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
) -> None:
    if frame.empty or "value" not in frame:
        status = (
            str(frame.iloc[0].get("availability_status", "not materialized"))
            if not frame.empty
            else "not materialized"
        )
        _message(ax, title, f"Release-bound evidence unavailable ({status}).")
        return
    rows = frame.copy()
    rows["__value"] = _numeric(rows, "value")
    rows = rows[np.isfinite(rows["__value"])].copy()
    if rows.empty:
        _message(ax, title, "No finite release-bound evidence rows.")
        return
    evidence = rows.get(
        "evidence_id", pd.Series("unknown_evidence", index=rows.index)
    ).astype(str)
    metric = rows.get("metric", pd.Series("unknown_metric", index=rows.index)).astype(str)
    labels = evidence + " | " + metric
    y = np.arange(len(rows))
    ax.barh(y, rows["__value"], color="#5C8374")
    if {"ci_low", "ci_high"}.issubset(rows):
        low = _numeric(rows, "ci_low")
        high = _numeric(rows, "ci_high")
        valid = np.isfinite(low) & np.isfinite(high)
        if valid.any():
            center = rows.loc[valid, "__value"].to_numpy(float)
            ax.errorbar(
                center,
                y[valid.to_numpy()],
                xerr=np.vstack(
                    [
                        np.maximum(0, center - low[valid].to_numpy(float)),
                        np.maximum(0, high[valid].to_numpy(float) - center),
                    ]
                ),
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
            )
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set(xlabel="Registered estimate", title=title)


def _fig01(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    frame = tables["mechanism_trajectories"].copy()
    if "horizon" in frame:
        frame = frame[pd.to_numeric(frame["horizon"], errors="coerce").eq(5)]
    nodes = tables["graph_nodes"].copy()
    edges = tables["graph_edges"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    graph_ax, ax = axes
    if "node_id" in nodes and nodes["node_id"].astype(str).str.len().gt(0).any():
        graph = nx.Graph()
        graph.add_nodes_from(nodes["node_id"].astype(str))
        if {"left_id", "right_id"}.issubset(edges.columns):
            graph.add_edges_from(
                (str(row.left_id), str(row.right_id))
                for row in edges.itertuples(index=False)
                if str(row.left_id) and str(row.right_id)
            )
        positions = nx.spring_layout(graph, seed=20260710, iterations=60)
        community = nodes.set_index("node_id").get(
            "community_id", pd.Series(0, index=nodes["node_id"])
        )
        colors = [float(community.get(node, 0)) for node in graph.nodes]
        nx.draw_networkx_edges(graph, positions, ax=graph_ax, width=0.45, alpha=0.25)
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=graph_ax,
            node_size=18,
            node_color=colors,
            cmap="tab20",
            alpha=0.9,
        )
        graph_ax.set_title("A  Strictly prior graph snapshot")
        graph_ax.axis("off")
    else:
        _message(graph_ax, "A  Prior graph snapshot", "Graph sample unavailable in this candidate view.")
    year = _numeric(frame, "publication_year")
    plotted = 0
    for mechanism in MECHANISMS:
        column = f"mechanism__{mechanism}_mean"
        if column not in frame:
            continue
        values = _numeric(frame, column)
        count_column = f"mechanism__{mechanism}_count"
        counts = pd.to_numeric(
            frame.get(count_column, pd.Series(np.nan, index=frame.index)),
            errors="coerce",
        )
        yearly_rows = pd.DataFrame(
            {"year": year, "value": values, "count": counts}
        ).dropna()
        yearly_rows = yearly_rows[yearly_rows["count"].gt(0)].copy()
        yearly_rows["weighted_value"] = (
            yearly_rows["value"] * yearly_rows["count"]
        )
        yearly = yearly_rows.groupby("year", dropna=False).agg(
            weighted_value=("weighted_value", "sum"),
            count=("count", "sum"),
        )
        yearly["value"] = yearly["weighted_value"] / yearly["count"]
        if yearly.empty:
            continue
        ax.plot(
            yearly.index,
            yearly["value"].values,
            marker="o",
            linewidth=1.8,
            label=mechanism.replace("_", " "),
        )
        plotted += 1
    if not plotted:
        _message(ax, "B  Five-mechanism trajectories", "No evaluable mechanism trajectory rows.")
    else:
        ax.set(
            xlabel="Publication year",
            ylabel="Paper-weighted OOF mechanism channel (mean)",
        )
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.2)
        ax.set_title("B  τ=5 OOF five-mechanism trajectories")
    fig.tight_layout()
    return fig


def _fig02(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    quality = tables["feature_quality"].copy()
    definitions = tables["feature_definitions"].copy()
    redundancy = tables["feature_redundancy"].copy()
    mapping = tables["mechanism_mapping"].copy()
    relationships = tables["target_relationships"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    if quality.empty:
        _message(axes[0, 0], "A  Feature quality and definitions", "No feature-quality rows.")
    else:
        q = quality.sort_values(["feature_group", "feature"], kind="stable")
        axes[0, 0].barh(
            q["feature"],
            _numeric(q, "finite_coverage"),
            color=np.where(q["feature_group"].eq("core8"), "#176B87", "#9BBEC8"),
        )
        axes[0, 0].axvline(0.95, color="#B31312", linestyle="--", linewidth=1)
        versions = (
            ", ".join(sorted(definitions.get("definition_version", pd.Series(dtype=str)).dropna().astype(str).unique()))
            or "version unavailable"
        )
        axes[0, 0].set(
            xlim=(0, 1.02),
            xlabel="Finite coverage",
            title=f"A  8 core + 10 auxiliary definitions ({versions})",
        )
    if not mapping.empty:
        matrix = pd.crosstab(mapping["mechanism"], mapping["feature"])
        axes[0, 1].imshow(matrix.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axes[0, 1].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=8)
        axes[0, 1].set_yticks(range(len(matrix.index)), [value.replace("_", " ") for value in matrix.index], fontsize=8)
        axes[0, 1].set_title("B  8 indicators → 5 mechanisms")
    else:
        _message(axes[0, 1], "B  Mechanism map", "No mapping rows.")
    if not redundancy.empty and {"feature_left", "feature_right", "spearman"}.issubset(redundancy):
        feature_names = sorted(
            set(redundancy["feature_left"].astype(str))
            | set(redundancy["feature_right"].astype(str))
        )
        matrix = pd.DataFrame(np.nan, index=feature_names, columns=feature_names)
        matrix_values = matrix.to_numpy(copy=True)
        np.fill_diagonal(matrix_values, 1.0)
        matrix.iloc[:, :] = matrix_values
        for row in redundancy.itertuples(index=False):
            value = pd.to_numeric(pd.Series([row.spearman]), errors="coerce").iloc[0]
            if np.isfinite(value):
                matrix.loc[str(row.feature_left), str(row.feature_right)] = float(value)
                matrix.loc[str(row.feature_right), str(row.feature_left)] = float(value)
        masked = np.ma.masked_invalid(matrix.to_numpy(float))
        cmap = plt.get_cmap("coolwarm").with_extremes(bad="#D9D9D9")
        image = axes[1, 0].imshow(masked, vmin=-1, vmax=1, cmap=cmap)
        axes[1, 0].set_xticks(range(len(feature_names)), feature_names, rotation=90, fontsize=7)
        axes[1, 0].set_yticks(range(len(feature_names)), feature_names, fontsize=7)
        axes[1, 0].set_title("C  Core-indicator redundancy (Spearman)")
        fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    else:
        _message(axes[1, 0], "C  Core-indicator redundancy", "No evaluable pairwise correlations.")
    if not relationships.empty:
        pivot = relationships.pivot(index="feature", columns="target", values="spearman")
        positions = np.arange(len(pivot))
        width = 0.8 / max(1, len(pivot.columns))
        for index, target in enumerate(pivot.columns):
            values = pd.to_numeric(pivot[target], errors="coerce")
            finite = values.notna()
            axes[1, 1].barh(
                positions[finite] + index * width,
                values[finite],
                height=width,
                label=target,
            )
        axes[1, 1].set_yticks(positions + width * (len(pivot.columns) - 1) / 2, pivot.index)
        axes[1, 1].axvline(0, color="black", linewidth=0.7)
        axes[1, 1].set(xlabel="Spearman", title="D  D5/S5 target relationships (missing stays missing)")
        axes[1, 1].legend(frameon=False)
    else:
        _message(axes[1, 1], "D  Target relationships", "No evaluable target relationship rows.")
    fig.tight_layout()
    return fig


def _fig03(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    metrics = tables["oof_metrics"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sensitivity = (
        metrics.get("sensitivity", pd.Series("main", index=metrics.index))
        .fillna("main")
        .astype(str)
    )

    def draw_rows(
        ax: plt.Axes,
        frame: pd.DataFrame,
        *,
        title: str,
        xlabel: str,
        label_columns: tuple[str, ...],
        sort_columns: tuple[str, ...] | None = None,
        reference: float = 0.0,
        color: str = "#176B87",
    ) -> None:
        rows = frame.copy()
        rows["__value"] = _numeric(rows, "value")
        rows = rows[np.isfinite(rows["__value"])].copy()
        if rows.empty:
            _message(ax, title, "No registered rows for this pre-locked panel.")
            return
        rows = rows.sort_values(
            list(sort_columns or label_columns), kind="stable"
        )
        rows["__label"] = rows[list(label_columns)].astype(str).agg(" | ".join, axis=1)
        y = np.arange(len(rows))
        ax.barh(y, rows["__value"], color=color, alpha=0.9)
        if {"ci_low", "ci_high"}.issubset(rows):
            low = _numeric(rows, "ci_low")
            high = _numeric(rows, "ci_high")
            finite_ci = np.isfinite(low) & np.isfinite(high)
            if finite_ci.any():
                center = rows.loc[finite_ci, "__value"].to_numpy(float)
                ax.errorbar(
                    center,
                    y[finite_ci.to_numpy()],
                    xerr=np.vstack(
                        [
                            np.maximum(0.0, center - low[finite_ci].to_numpy(float)),
                            np.maximum(0.0, high[finite_ci].to_numpy(float) - center),
                        ]
                    ),
                    fmt="none",
                    ecolor="black",
                    capsize=2,
                    linewidth=0.8,
                )
        ax.set_yticks(y, rows["__label"], fontsize=8)
        ax.axvline(reference, color="black", linestyle="--" if reference else "-", linewidth=0.7)
        ax.set(xlabel=xlabel, title=title)

    model_order = {
        name: index
        for index, name in enumerate(
            (
                "domain_year_only",
                "bibliographic_aux10_ridge",
                "mechanism5_equal_weight",
                "mechanism5_simplex",
                "gam18",
                "hgb18",
                "rank_blend",
            )
        )
    }
    panel_a = metrics[
        pd.to_numeric(metrics.get("horizon"), errors="coerce").eq(5)
        & metrics.get("scope", pd.Series("", index=metrics.index)).eq(
            "development_oof_all_models"
        )
        & metrics.get("metric", pd.Series("", index=metrics.index)).eq(
            "rho_global_calibrated"
        )
        & sensitivity.eq("main")
    ].copy()
    panel_a["model_order"] = panel_a.get("model_id", pd.Series(dtype=str)).map(
        model_order
    ).fillna(999)
    draw_rows(
        axes[0, 0],
        panel_a,
        title="A  τ=5 model comparison (development OOF)",
        xlabel="Spearman ρ (point estimate; CI not computed per model)",
        label_columns=("model_id",),
        sort_columns=("model_order", "model_id"),
    )

    four_metrics = (
        "rho_global_calibrated",
        "rho_global_uncalibrated",
        "rho_domain_macro",
        "rho_conditional",
    )
    panel_b = metrics[
        metrics.get("model_id", pd.Series("", index=metrics.index)).eq(
            "nested_selector"
        )
        & metrics.get("scope", pd.Series("", index=metrics.index)).eq(
            "development_oof"
        )
        & metrics.get("metric", pd.Series("", index=metrics.index)).isin(
            four_metrics
        )
        & sensitivity.eq("main")
    ].copy()
    draw_rows(
        axes[0, 1],
        panel_b,
        title="B  Locked four OOF summaries",
        xlabel="Spearman ρ (95% clustered CI)",
        label_columns=("horizon", "metric"),
    )

    scope = metrics.get("scope", pd.Series("", index=metrics.index)).astype(str)
    panel_c = metrics[
        scope.str.contains("sealed_temporal_holdout", regex=False)
        & metrics.get("metric", pd.Series("", index=metrics.index)).eq(
            "rho_global_calibrated"
        )
        & sensitivity.eq("main")
    ].copy()
    draw_rows(
        axes[1, 0],
        panel_c,
        title="C  Sealed and strict temporal tests",
        xlabel="Spearman ρ (95% clustered CI)",
        label_columns=("horizon", "model_id", "scope"),
        color="#B31312",
    )

    panel_d = metrics[
        metrics.get("model_id", pd.Series("", index=metrics.index)).eq(
            "nested_selector"
        )
        & metrics.get("scope", pd.Series("", index=metrics.index)).eq(
            "development_oof"
        )
        & metrics.get("metric", pd.Series("", index=metrics.index)).eq(
            "top_decile_enrichment"
        )
        & sensitivity.eq("main")
    ].copy()
    draw_rows(
        axes[1, 1],
        panel_d,
        title="D  Predicted top-decile enrichment",
        xlabel="Enrichment × (95% clustered CI)",
        label_columns=("horizon", "model_id", "scope"),
        reference=1.0,
        color="#5C8374",
    )
    fig.tight_layout()
    return fig


def _fig04(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    frame = tables["score_strata"].copy()
    if "horizon" in frame:
        frame = frame[pd.to_numeric(frame["horizon"], errors="coerce").eq(5)]
    validation = tables["peer_review_validation"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    if frame.empty:
        _message(ax, "A  OOF score strata", "No OOF strata rows.")
        _evidence_bars(
            axes[1], validation, "B  Peer-review external validity"
        )
        return fig
    frame["n"] = _numeric(frame, "n").fillna(0)
    frame["weighted_mean"] = _numeric(frame, "mean") * frame["n"]
    summary = frame.groupby("score_stratum", observed=True).agg(
        n=("n", "sum"), weighted_sum=("weighted_mean", "sum")
    )
    summary["mean"] = summary["weighted_sum"] / summary["n"].replace(0, np.nan)
    summary = summary.reindex(["low", "middle", "high"])
    shown = summary[np.isfinite(summary["mean"]) & summary["n"].gt(0)]
    bars = ax.bar(
        shown.index,
        shown["mean"],
        color=[
            {"low": "#9BBEC8", "middle": "#5C8374", "high": "#176B87"}[stratum]
            for stratum in shown.index
        ],
    )
    for bar, count in zip(bars, shown["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={int(count)}", ha="center", va="bottom")
    ax.set(
        ylabel="OOF calibrated performance score",
        title="A  τ=5 pre-registered OOF strata",
    )
    _evidence_bars(
        axes[1], validation, "B  Peer-review external validity"
    )
    fig.suptitle("Fig.4 | New-score peer-review validation", fontweight="bold")
    fig.tight_layout()
    return fig


def _fig05(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    frame = tables["forecast_scores"].copy()
    backtest = tables["frontier_backtest"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    evidence = frame[
        frame.get("score_is_out_of_sample", pd.Series(0, index=frame.index)).eq(1)
    ]
    descriptive = frame[
        frame.get("score_scope", pd.Series("", index=frame.index)).eq(
            "full_fit_descriptive"
        )
    ]
    evidence_values = _numeric(
        evidence, "score_performance_percentile"
    ).dropna()
    descriptive_values = _numeric(
        descriptive, "score_performance_percentile"
    ).dropna()
    if evidence_values.empty and descriptive_values.empty:
        _message(ax, "A  τ=5 forecast scores", "No eligible τ=5 scores.")
    else:
        bins = np.linspace(0, 1, 21)
        if not evidence_values.empty:
            ax.hist(
                evidence_values,
                bins=bins,
                color="#176B87",
                alpha=0.8,
                label="OOF + sealed score distribution",
            )
        if not descriptive_values.empty:
            ax.hist(
                descriptive_values,
                bins=bins,
                histtype="step",
                linewidth=2,
                color="#D95F02",
                label="Full-fit descriptive forecast",
            )
        ax.set(
            xlabel="τ=5 performance percentile",
            ylabel="Papers",
            title="A  Out-of-sample and descriptive score distributions",
        )
        ax.legend(frameon=False)
    _evidence_bars(axes[1], backtest, "B  AI-frontier forecast backtest")
    fig.suptitle("Fig.5 | τ=5 frontier forecasting", fontweight="bold")
    fig.tight_layout()
    return fig


def _metric_bars(frame: pd.DataFrame, title: str) -> plt.Figure:
    values = frame.copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    if values.empty or "value" not in values:
        status = (
            str(values.iloc[0].get("availability_status", "not_run"))
            if not values.empty
            else "not_run"
        )
        _message(axes[0, 0], title, f"No registered result rows ({status}).")
        for ax in axes.flat[1:]:
            ax.axis("off")
        return fig
    values["__value"] = _numeric(values, "value")
    values = values[np.isfinite(values["__value"])].copy()
    values["__metric"] = values.get(
        "metric", pd.Series("unknown", index=values.index)
    ).astype(str)
    for column, default in (
        ("horizon", "?"),
        ("model_id", "unknown_model"),
        ("scope", "unknown_scope"),
        ("sensitivity", "main"),
    ):
        if column not in values:
            values[column] = default
    values["__variant"] = values.get(
        "evidence_id", pd.Series("", index=values.index)
    ).fillna("").astype(str)
    values["__variant"] = values["__variant"].where(
        values["__variant"].str.len().gt(0),
        values["sensitivity"].fillna("main").astype(str),
    )
    values["__label"] = (
        "τ"
        + values["horizon"].astype(str)
        + " | "
        + values["model_id"].astype(str)
        + " | "
        + values["scope"].astype(str)
        + " | "
        + values["__variant"]
        + " | "
        + values["__metric"]
    )
    families = (
        (
            axes[0, 0],
            values["__metric"].str.startswith("rho_"),
            "A  Spearman robustness",
            "Spearman ρ",
            0.0,
        ),
        (
            axes[0, 1],
            values["__metric"].eq("top_decile_enrichment"),
            "B  Top-decile enrichment",
            "Enrichment ×",
            1.0,
        ),
        (
            axes[1, 0],
            values["__metric"].str.contains("ratio|share", regex=True),
            "C  Ratios and shares",
            "Proportion",
            0.0,
        ),
        (
            axes[1, 1],
            values["__metric"].str.startswith("n_")
            | values["__metric"].str.contains("finite.*rows", regex=True),
            "D  Evaluated rows",
            "Count",
            0.0,
        ),
    )
    for ax, mask, panel_title, xlabel, reference in families:
        shown = values[mask].sort_values(
            ["horizon", "model_id", "scope", "sensitivity", "__metric"],
            kind="stable",
        )
        if shown.empty:
            _message(ax, panel_title, "No registered rows in this metric family.")
            continue
        y = np.arange(len(shown))
        ax.barh(y, shown["__value"], color="#176B87")
        ax.set_yticks(y, shown["__label"], fontsize=6)
        ax.axvline(
            reference,
            color="black",
            linestyle="--" if reference else "-",
            linewidth=0.7,
        )
        ax.set(xlabel=xlabel, title=panel_title)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def _fig06(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    return _metric_bars(tables["robustness_metrics"], "Fig.6 | Robustness and sensitivity registry")


def _fig07(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    frame = tables["venue_family_summary"].copy()
    inference = tables["venue_family_inference"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    column = "conditional_score_mean"
    if frame.empty or column not in frame:
        for index, ax in enumerate(axes.flat):
            if index == 5:
                _evidence_bars(ax, inference, "F  Controlled inference")
            else:
                _message(
                    ax,
                    f"{chr(65 + index)}  Venue-family panel",
                    "No evaluable venue-family rows.",
                )
        return fig
    if "horizon" in frame:
        frame = frame[pd.to_numeric(frame["horizon"], errors="coerce").eq(5)]
    if frame.empty:
        for index, ax in enumerate(axes.flat):
            _message(
                ax,
                f"{chr(65 + index)}  τ=5 venue-family panel",
                "No evaluable τ=5 venue-family rows.",
            )
        return fig
    frame["n"] = _numeric(frame, "n").fillna(0)
    for ax, value_column, title, xlabel in (
        (
            axes[0, 0],
            "conditional_score_mean",
            "A  Conditioned prediction rank",
            "Domain/period-conditioned score rank",
        ),
        (
            axes[0, 1],
            "future_diffusion_mean",
            "B  Future diffusion contribution",
            "Domain/period-conditioned target rank",
        ),
        (
            axes[0, 2],
            "predicted_top_share",
            "C  Predicted high-score enrichment",
            "Predicted top-decile share",
        ),
    ):
        if value_column not in frame:
            _message(ax, title, f"Missing {value_column}.")
            continue
        weighted = _numeric(frame, value_column) * frame["n"]
        summary_frame = frame.assign(__weighted=weighted).groupby(
            "venue_family", dropna=False
        ).agg(weighted=("__weighted", "sum"), n=("n", "sum"))
        summary = (
            summary_frame["weighted"]
            / summary_frame["n"].replace(0, np.nan)
        ).sort_values()
        ax.barh(summary.index.astype(str), summary.values, color="#5C8374")
        ax.set(xlabel=xlabel, title=title)

    mechanism_columns = [
        f"mechanism__{mechanism}_mean" for mechanism in MECHANISMS
    ]
    if set(mechanism_columns).issubset(frame):
        weighted_rows = []
        for venue, group in frame.groupby("venue_family", dropna=False):
            row = {"venue_family": str(venue)}
            for mechanism, column_name in zip(MECHANISMS, mechanism_columns):
                values = _numeric(group, column_name)
                valid = np.isfinite(values) & group["n"].gt(0)
                row[mechanism] = (
                    float(np.average(values[valid], weights=group.loc[valid, "n"]))
                    if valid.any()
                    else np.nan
                )
            weighted_rows.append(row)
        matrix = pd.DataFrame(weighted_rows).set_index("venue_family")
        image = axes[1, 0].imshow(
            np.ma.masked_invalid(matrix.to_numpy(float)),
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        axes[1, 0].set_xticks(
            range(len(matrix.columns)),
            [name.replace("_", "\n") for name in matrix.columns],
            fontsize=7,
        )
        axes[1, 0].set_yticks(
            range(len(matrix.index)), matrix.index.astype(str), fontsize=8
        )
        axes[1, 0].set_title("D  Five-mechanism signatures")
        fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    else:
        _message(axes[1, 0], "D  Five-mechanism signatures", "Mechanism columns unavailable.")

    if "publication_period" in frame:
        for venue, group in frame.groupby("venue_family", dropna=False):
            group = group.sort_values("publication_period")
            axes[1, 1].plot(
                _numeric(group, "publication_period"),
                _numeric(group, "conditional_score_mean"),
                marker="o",
                label=str(venue),
            )
        axes[1, 1].set(
            xlabel="Five-year publication period",
            ylabel="Conditioned score rank",
            title="E  Time migration",
        )
        axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)
    else:
        _message(axes[1, 1], "E  Time migration", "Publication period unavailable.")
    _evidence_bars(axes[1, 2], inference, "F  Controlled inference")
    fig.suptitle(
        "Fig.7 | τ=5 within-Nature-Portfolio venue-family comparison",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def _fig08(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    frame = tables["architecture_contract"].sort_values("order")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")
    x = np.linspace(0.04, 0.96, len(frame))
    for index, (_, row) in enumerate(frame.iterrows()):
        ax.text(x[index], 0.55, str(row["component"]).replace("_", "\n"), ha="center", va="center", fontsize=9, bbox={"boxstyle": "round,pad=0.45", "fc": "#DDE6ED", "ec": "#176B87"})
        if index:
            ax.annotate("", xy=(x[index] - 0.045, 0.55), xytext=(x[index - 1] + 0.045, 0.55), arrowprops={"arrowstyle": "->", "color": "#526D82"})
    ax.set_title("Fig.8 | Dual-score ASPR evidence architecture", loc="left", fontweight="bold")
    return fig


def _fig09(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    frame = tables["case_profiles"].copy()
    evidence = tables["case_evidence"].copy()
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(121, polar=True)
    evidence_ax = fig.add_subplot(122)
    if frame.empty or str(frame.iloc[0].get("case_status", "scored")) != "scored":
        status = (
            "case registry is empty"
            if frame.empty
            else str(frame.iloc[0].get("case_status", "unavailable"))
        )
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"Fixed case unavailable in release\n({status})",
            transform=ax.transAxes,
            ha="center",
        )
        _evidence_bars(
            evidence_ax, evidence, "B  Fixed-case evidence-chain rerun"
        )
        return fig
    row = frame.iloc[0]
    raw_values = pd.to_numeric(
        pd.Series([row.get(f"mechanism__{name}") for name in MECHANISMS]),
        errors="coerce",
    )
    if not np.isfinite(raw_values).all():
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "Fixed case has incomplete mechanism channels\n(no zero imputation)",
            transform=ax.transAxes,
            ha="center",
        )
        _evidence_bars(
            evidence_ax, evidence, "B  Fixed-case evidence-chain rerun"
        )
        return fig
    values = raw_values.astype(float).tolist()
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    ax.plot(angles, values, color="#176B87", linewidth=2)
    ax.fill(angles, values, color="#9BBEC8", alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(angles[:-1], [name.replace("_", "\n") for name in MECHANISMS], fontsize=8)
    scope = str(row.get("score_scope", "unknown_scope"))
    observable = str(row.get("outcome_observable", "unknown"))
    claim = str(row.get("claim_scope", ""))
    ax.set_title("A  Fixed-case descriptive five-mechanism profile")
    ax.text(
        0.5,
        -0.16,
        f"scope={scope}; outcome_observable={observable}\n{claim}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        wrap=True,
    )
    _evidence_bars(
        evidence_ax, evidence, "B  Fixed-case evidence-chain rerun"
    )
    fig.suptitle("Fig.9 | Release-bound fixed-case audit", fontweight="bold")
    fig.tight_layout()
    return fig


def _fig10(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    return _metric_bars(tables["ablation_metrics"], "Fig.10 | Registered ablations")


RENDERERS: Dict[int, Callable[[Mapping[str, pd.DataFrame]], plt.Figure]] = {
    1: _fig01,
    2: _fig02,
    3: _fig03,
    4: _fig04,
    5: _fig05,
    6: _fig06,
    7: _fig07,
    8: _fig08,
    9: _fig09,
    10: _fig10,
}


def render_figure(
    view_dir: Path,
    figure: int,
    output_path: Path,
    *,
    draft_watermark: str | None = None,
) -> Path:
    """Render one figure strictly from its validated CSV view."""
    if figure not in RENDERERS:
        raise ValueError("figure must be in 1..10")
    tables = _tables(Path(view_dir))
    figure_object = RENDERERS[figure](tables)
    if draft_watermark:
        figure_object.text(
            0.5,
            0.5,
            draft_watermark,
            ha="center",
            va="center",
            fontsize=28,
            color="#B31312",
            alpha=0.16,
            rotation=28,
            fontweight="bold",
        )
    output = Path(output_path)
    if output.suffix.lower() not in {".png", ".pdf", ".svg", ".tif", ".tiff"}:
        raise ValueError("output must be PNG, PDF, SVG, TIF, or TIFF")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_object.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure_object)
    return output
