from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fig03.old.fig3_empirical_weight_learning import (
    METRIC_KEYS,
    attach_metadata,
    field_distance_matrix,
    field_year_standardize,
    load_raw_data,
    modularity_fixed_partition,
    pair_zscore_lookup,
    rao_stirling,
    shannon_entropy,
    simpson_diversity,
    uzzi_atypicality,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v2_publication" / "views" / "fig5" / "multi_domain"
DEFAULT_FIG3_RUN_DIR = PROJECT_ROOT / "outputs" / "fig03/old" / "multi_domain"


def load_weights(path: Path) -> pd.Series:
    """Read Fig. 3 learned weights and return a METRIC_KEYS-aligned series."""
    if not path.exists():
        raise FileNotFoundError(f"Missing weight file: {path}")
    raw = pd.read_csv(path)
    if "weight" not in raw.columns:
        raise ValueError(f"{path} must contain a weight column")
    metric_col = next((col for col in raw.columns if col != "weight"), raw.columns[0])
    weights = pd.Series(pd.to_numeric(raw["weight"], errors="coerce").to_numpy(), index=raw[metric_col].astype(str))
    out = weights.reindex(METRIC_KEYS).fillna(0.0).astype(float)
    if not np.isfinite(out.to_numpy(dtype=float)).all() or float(out.abs().sum()) <= 0.0:
        raise ValueError(f"No finite non-zero Fig. 3 weights found in {path}")
    return out


def build_source_reference_index(citations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Index citation metadata by citing paper id."""
    return {str(source): group.copy() for source, group in citations.groupby("source", sort=False)}


def build_outgoing_edge_index(citations: pd.DataFrame) -> Dict[str, List[str]]:
    """Index outgoing citation targets by source id for local graph construction."""
    out: Dict[str, List[str]] = {}
    for source, group in citations.groupby("source", sort=False):
        out[str(source)] = group["target"].astype(str).tolist()
    return out


def references_for_paper(source_refs: Dict[str, pd.DataFrame], paper_id: str, paper_year: int) -> pd.DataFrame:
    """Return local prior references for one source paper."""
    refs = source_refs.get(str(paper_id))
    if refs is None or refs.empty:
        return pd.DataFrame()
    return refs[pd.to_numeric(refs["target_year"], errors="coerce") < int(paper_year)].copy()


def build_local_reference_graph_indexed(
    refs: pd.DataFrame,
    outgoing_edges: Dict[str, List[str]],
    paper_id: Optional[str] = None,
    paper_comm: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """Build the local reference graph without scanning the full citation table."""
    ref_ids = set(refs["target"].astype(str))
    graph = nx.Graph()
    graph.add_nodes_from(ref_ids)
    for source in ref_ids:
        for target in outgoing_edges.get(source, []):
            if target in ref_ids:
                graph.add_edge(source, target)
    comm_map = dict(zip(refs["target"].astype(str), refs["target_community"].astype(int)))
    if paper_id is not None:
        pid = str(paper_id)
        graph.add_node(pid)
        for ref_id in ref_ids:
            graph.add_edge(pid, ref_id)
        if paper_comm is not None:
            comm_map[pid] = int(paper_comm)
    return graph, comm_map


def publication_metric_row(
    paper: Any,
    refs: pd.DataFrame,
    required_metrics: List[str],
    dist_lookup: Dict[Tuple[str, str], float],
    pairz_lookup: Dict[Tuple[str, str], float],
    outgoing_edges: Dict[str, List[str]],
) -> Dict[str, object]:
    """Compute publication-day indicators for one paper."""
    pid = str(getattr(paper, "id"))
    year = int(getattr(paper, "year"))
    pcomm = int(getattr(paper, "display_community"))

    ref_fields = refs["target_field"].astype(str).tolist()
    ref_comms = refs["target_community"].astype(int).tolist()
    ref_ids = refs["target"].astype(str).tolist()
    required = set(required_metrics)
    metric_values = {key: np.nan for key in METRIC_KEYS}
    degree_p = np.nan
    eff_size = np.nan
    inv_constraint = np.nan

    needs_graph = bool(required & {"B", "DeltaQ0", "BurtIP"})
    if needs_graph:
        gm, comm_m = build_local_reference_graph_indexed(refs, outgoing_edges)
        g0, comm_0 = build_local_reference_graph_indexed(refs, outgoing_edges, paper_id=pid, paper_comm=pcomm)
        degree_p = float(g0.degree(pid))
        if "DeltaQ0" in required:
            q_minus = modularity_fixed_partition(gm, comm_m)
            q_zero = modularity_fixed_partition(g0, comm_0)
            metric_values["DeltaQ0"] = -(q_zero - q_minus)
        if "B" in required:
            try:
                metric_values["B"] = float(nx.betweenness_centrality(g0, normalized=True).get(pid, 0.0))
            except Exception:
                metric_values["B"] = 0.0
        if "BurtIP" in required:
            try:
                eff_size = nx.effective_size(g0, nodes=[pid]).get(pid, 0.0)  # type: ignore[attr-defined]
            except Exception:
                eff_size = float(len(ref_ids))
            metric_values["BurtIP"] = float(eff_size / max(1.0, len(ref_ids)))
            try:
                constraint = nx.constraint(g0, nodes=[pid]).get(pid, 1.0)  # type: ignore[attr-defined]
                inv_constraint = float(1.0 / max(constraint, 1e-9))
            except Exception:
                inv_constraint = np.nan
    if "RS" in required:
        metric_values["RS"] = rao_stirling(ref_fields, dist_lookup)
    if "Uzzi" in required:
        metric_values["Uzzi"] = uzzi_atypicality(ref_fields, pairz_lookup)
    if "RTD" in required:
        metric_values["RTD"] = simpson_diversity(ref_comms)
    if "PDE" in required:
        metric_values["PDE"] = shannon_entropy(ref_fields)

    row: Dict[str, object] = {
        "paper_id": pid,
        "title": getattr(paper, "title"),
        "domain": getattr(paper, "domain"),
        "year": year,
        "primary_field": getattr(paper, "primary_field"),
        "display_community": pcomm,
        "is_landmark": int(getattr(paper, "is_landmark")),
        "reference_count": len(ref_ids),
        "cited_by_count": float(getattr(paper, "cited_by_count", np.nan)),
        **metric_values,
        "degree_p": degree_p,
        "effective_size": float(eff_size) if np.isfinite(eff_size) else np.nan,
        "constraint_inv": inv_constraint,
        "field_variety": float(len(set(ref_fields))),
        "field_simpson": simpson_diversity(ref_fields),
        "community_variety": float(len(set(ref_comms))),
    }
    return row

def compute_publication_day_metrics(
    fig3_input_dir: Path,
    required_metrics: List[str],
    min_refs: int,
    max_papers: Optional[int],
    progress_interval: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute Fig. 3 publication-day indicators without future-RGPM filtering."""
    raw = load_raw_data(fig3_input_dir)
    works, citations = attach_metadata(raw)
    works = works.sort_values(["year", "id"]).reset_index(drop=True)
    progress_interval = max(1, int(progress_interval))
    dist_cache: Dict[int, Dict[Tuple[str, str], float]] = {}
    pairz_cache: Dict[int, Dict[Tuple[str, str], float]] = {}
    source_refs = build_source_reference_index(citations)
    outgoing_edges = build_outgoing_edge_index(citations)
    rows: List[Dict[str, object]] = []
    skipped_min_refs = 0

    for idx, paper in enumerate(works.itertuples(index=False), start=1):
        if max_papers is not None and max_papers > 0 and len(rows) >= max_papers:
            break
        pid = str(getattr(paper, "id"))
        year = int(getattr(paper, "year"))
        refs = references_for_paper(source_refs, pid, year)
        if len(refs) < int(min_refs):
            skipped_min_refs += 1
            row = None
        else:
            if "RS" in required_metrics and year not in dist_cache:
                dist_cache[year] = field_distance_matrix(citations, year)
            if "Uzzi" in required_metrics and year not in pairz_cache:
                pairz_cache[year] = pair_zscore_lookup(citations, year)
            row = publication_metric_row(
                paper,
                refs,
                required_metrics=required_metrics,
                dist_lookup=dist_cache.get(year, {}),
                pairz_lookup=pairz_cache.get(year, {}),
                outgoing_edges=outgoing_edges,
            )
        if row is not None:
            rows.append(row)
        if idx == 1 or idx % progress_interval == 0 or idx == len(works):
            print(
                f"[fig5-score] scanned {idx:,}/{len(works):,}; scored={len(rows):,}; "
                f"skipped_min_refs={skipped_min_refs:,}",
                flush=True,
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        raise ValueError("No papers could be scored. Lower --min-refs or check citations.csv.")
    manifest = {
        "n_input_works": int(len(works)),
        "n_scored_papers": int(len(metrics)),
        "n_skipped_min_refs": int(skipped_min_refs),
        "min_refs": int(min_refs),
        "score_coverage": float(len(metrics) / max(1, len(works))),
    }
    return metrics, manifest


def build_score_table(metrics: pd.DataFrame, weights: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Standardize publication-day indicators and apply Fig. 3 weights."""
    metrics_z, feature_diag, active_metric_keys = field_year_standardize(metrics)
    z_cols = [f"{key}_z" for key in METRIC_KEYS]
    X = metrics_z[z_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    w = weights.reindex(METRIC_KEYS).fillna(0.0).to_numpy(dtype=float)
    active = [key for key in active_metric_keys if key in METRIC_KEYS]
    equal_w = np.ones(len(active), dtype=float) / max(1, len(active))
    active_cols = [f"{key}_z" for key in active]
    X_active = metrics_z[active_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    meta_cols = [
        "paper_id",
        "title",
        "domain",
        "year",
        "primary_field",
        "display_community",
        "is_landmark",
        "reference_count",
        "cited_by_count",
    ]
    score_table = metrics_z[meta_cols].copy()
    score_table["S_w"] = X @ w
    score_table["S_equal"] = X_active @ equal_w if len(active) else 0.0
    score_table["score_source"] = "forecast_publication_day_score"
    score_table["score_is_oof"] = 0
    for key in METRIC_KEYS:
        score_table[f"{key}_z"] = metrics_z[f"{key}_z"]
    return score_table, feature_diag, active


def write_outputs(
    out_dir: Path,
    score_table: pd.DataFrame,
    feature_diag: pd.DataFrame,
    weights: pd.Series,
    active_metrics: List[str],
    manifest: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write Fig. 5-compatible score table and audit manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    score_table.to_csv(out_dir / "fig3_score_table.csv", index=False)
    feature_diag.to_csv(out_dir / "forecast_score_feature_diagnostics.csv", index=False)
    weights.rename("weight").to_csv(out_dir / "forecast_score_weights.csv")
    payload = {
        **manifest,
        "fig3_input_dir": str(args.fig3_input_dir),
        "weight_file": str(args.weight_file),
        "active_metric_keys": active_metrics,
        "weights": {key: float(weights.get(key, 0.0)) for key in METRIC_KEYS},
        "provenance_note": (
            "Forecast scores apply Fig. 3 learned weights to publication-day indicators for all "
            "papers with enough local references. These scores are not strict out-of-fold Fig. 3 validation scores."
        ),
    }
    (out_dir / "forecast_score_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Fig. 5 forecast score table from Fig. 3 weights.")
    parser.add_argument("--fig3-input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory with works/citations/topics CSVs.")
    parser.add_argument("--fig3-run-dir", type=Path, default=DEFAULT_FIG3_RUN_DIR, help="Fig. 3 run directory with fig3_best_weights.csv.")
    parser.add_argument("--weight-file", type=Path, default=None, help="Optional explicit fig3_best_weights.csv path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory containing Fig. 5-compatible fig3_score_table.csv.")
    parser.add_argument("--min-refs", type=int, default=5, help="Minimum number of local prior references required for scoring.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional debug cap on scored papers.")
    parser.add_argument("--progress-interval", type=int, default=5000, help="Progress logging interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.weight_file is None:
        args.weight_file = args.fig3_run_dir / "fig3_best_weights.csv"
    weights = load_weights(args.weight_file)
    required_metrics = [key for key in METRIC_KEYS if abs(float(weights.get(key, 0.0))) > 1e-12]
    if not required_metrics:
        raise ValueError("No non-zero Fig. 3 weights are available for forecast scoring.")
    print(f"[fig5-score] computing weighted metrics: {', '.join(required_metrics)}", flush=True)
    metrics, manifest = compute_publication_day_metrics(
        fig3_input_dir=args.fig3_input_dir,
        required_metrics=required_metrics,
        min_refs=args.min_refs,
        max_papers=args.max_papers,
        progress_interval=args.progress_interval,
    )
    score_table, feature_diag, active_metrics = build_score_table(metrics, weights)
    write_outputs(args.out_dir, score_table, feature_diag, weights, active_metrics, manifest, args)
    print(
        f"[fig5-score] wrote {args.out_dir} with {len(score_table):,} scored papers "
        f"({manifest['score_coverage']:.1%} coverage)",
        flush=True,
    )


if __name__ == "__main__":
    main()
