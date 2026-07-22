from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.nature_ready_checks import detect_no_leakage_feature_violations  # noqa: E402
from scripts.fig3_v5_rank_learning import run_nested_rank_learning, write_json  # noqa: E402
from scripts.nature_portfolio_v5 import (  # noqa: E402
    DEFAULT_V5_CORPUS_DIR,
    DEFAULT_V5_OUTPUT_DIR,
    build_time_block_folds,
    entropy_from_counts,
    parse_referenced_works,
    read_csv,
    robust_percentile,
    simpson_from_counts,
    utc_now,
)


BASE_FEATURE_COLS = [
    "reference_count_log",
    "degree_p",
    "field_variety",
    "field_simpson",
    "field_entropy",
    "community_variety",
    "community_simpson",
    "community_entropy",
    "reference_age_mean",
    "reference_age_median",
    "reference_age_std",
    "recent_reference_share",
    "classic_reference_share",
]
TARGET_CANDIDATES = [
    "RGPM_latent_future_percentile",
    "RGPM_weighted_latent_future_percentile",
    "RGPM_publication_day_residual_balanced",
    "RGPM_legacy_structural_residual",
]
FUTURE_DELTA_COLS = [
    "n_future_citers",
    "future_community_reach",
    "future_field_reach",
    "future_subfield_reach",
    "future_field_entropy",
    "future_topic_entropy",
    "future_field_simpson",
    "future_topic_simpson",
]


def reference_lookup(reference_works: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if reference_works.empty or "id" not in reference_works.columns:
        return {}
    refs = reference_works.copy()
    for col in ["year", "openalex_primary_field", "openalex_primary_subfield", "display_community", "display_topic_label"]:
        if col not in refs.columns:
            refs[col] = ""
    return {str(row["id"]): row for row in refs.to_dict("records")}


def reference_profile_features(works: pd.DataFrame, refs: pd.DataFrame) -> pd.DataFrame:
    lookup = reference_lookup(refs)
    rows: List[Dict[str, Any]] = []
    for paper in works.to_dict("records"):
        paper_id = str(paper.get("id"))
        year = pd.to_numeric(paper.get("year"), errors="coerce")
        ref_ids = parse_referenced_works(paper.get("referenced_works"))
        ref_rows = [lookup[rid] for rid in ref_ids if rid in lookup]
        fields = [str(row.get("openalex_primary_field") or row.get("primary_field") or "") for row in ref_rows]
        fields = [field for field in fields if field]
        communities = [str(row.get("display_community") or row.get("display_topic_label") or "") for row in ref_rows]
        communities = [comm for comm in communities if comm]
        ref_years = [pd.to_numeric(row.get("year"), errors="coerce") for row in ref_rows]
        ref_years = [int(v) for v in ref_years if pd.notna(v) and int(v) > 0]
        ages = [max(0, int(year) - ref_year) for ref_year in ref_years] if pd.notna(year) else []
        field_counts = pd.Series(fields).value_counts() if fields else pd.Series(dtype=int)
        community_counts = pd.Series(communities).value_counts() if communities else pd.Series(dtype=int)
        rows.append(
            {
                "paper_id": paper_id,
                "degree_p": float(len(ref_ids)),
                "reference_count_log": math.log1p(float(len(ref_ids))),
                "field_variety": float(field_counts.size),
                "field_simpson": simpson_from_counts(field_counts.to_numpy(dtype=float)) if len(field_counts) else 0.0,
                "field_entropy": entropy_from_counts(field_counts.to_numpy(dtype=float)) if len(field_counts) else 0.0,
                "community_variety": float(community_counts.size),
                "community_simpson": simpson_from_counts(community_counts.to_numpy(dtype=float)) if len(community_counts) else 0.0,
                "community_entropy": entropy_from_counts(community_counts.to_numpy(dtype=float)) if len(community_counts) else 0.0,
                "reference_age_mean": float(np.mean(ages)) if ages else 0.0,
                "reference_age_median": float(np.median(ages)) if ages else 0.0,
                "reference_age_std": float(np.std(ages)) if ages else 0.0,
                "recent_reference_share": float(np.mean([age <= 5 for age in ages])) if ages else 0.0,
                "classic_reference_share": float(np.mean([age >= 20 for age in ages])) if ages else 0.0,
                "reference_metadata_coverage": float(len(ref_rows) / max(1, len(ref_ids))),
            }
        )
    return pd.DataFrame(rows)


def add_latent_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    delta_cols = [col for col in FUTURE_DELTA_COLS if col in out.columns]
    if not delta_cols:
        raise ValueError("No future delta columns available to build v5 RGPM targets.")
    ranked = pd.DataFrame({col: robust_percentile(out[col]) for col in delta_cols})
    latent = ranked.mean(axis=1)
    out["RGPM_latent_future_percentile"] = robust_percentile(latent)
    weights = np.asarray([0.40 if col == "n_future_citers" else 1.0 for col in delta_cols], dtype=float)
    weights = weights / weights.sum()
    weighted = ranked.to_numpy(dtype=float) @ weights
    out["RGPM_weighted_latent_future_percentile"] = robust_percentile(pd.Series(weighted, index=out.index))
    balanced_cols = [col for col in delta_cols if col != "n_future_citers"]
    balanced = ranked[balanced_cols].mean(axis=1) if balanced_cols else ranked.mean(axis=1)
    covariates = [col for col in ["year", "reference_count_log"] if col in out.columns]
    residual = balanced.copy()
    if covariates and len(out) >= len(covariates) + 4:
        X_parts = [np.ones((len(out), 1), dtype=float)]
        for col in covariates:
            vals = pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            sd = float(np.nanstd(vals))
            vals = (vals - float(np.nanmean(vals))) / sd if sd > 1e-12 else np.zeros_like(vals)
            X_parts.append(vals.reshape(-1, 1))
        X = np.hstack(X_parts)
        y = balanced.to_numpy(dtype=float)
        try:
            beta = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X.T @ X + 1e-3 * np.eye(X.shape[1])) @ y
        residual = pd.Series(y - X @ beta, index=out.index)
    out["RGPM_publication_day_residual_balanced"] = robust_percentile(residual)
    out["RGPM_legacy_structural_residual"] = out["RGPM_latent_future_percentile"]
    return out


def build_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    works = read_csv(args.works)
    refs = read_csv(args.reference_works)
    future = read_csv(args.future_graph_deltas)
    if works.empty:
        raise FileNotFoundError(f"No works table found: {args.works}")
    if future.empty:
        raise FileNotFoundError(f"No future graph deltas found: {args.future_graph_deltas}")
    features = reference_profile_features(works, refs)
    frame = works.merge(features, left_on="id", right_on="paper_id", how="left")
    frame = frame.merge(future, left_on="id", right_on="paper_id", how="inner", suffixes=("", "_future"))
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame[frame["year"].notna()].copy()
    frame["fold_id"] = build_time_block_folds(frame["year"], n_folds=args.n_folds)
    frame = add_latent_targets(frame)
    for col in BASE_FEATURE_COLS:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    if args.min_reference_metadata_coverage > 0 and "reference_metadata_coverage" in frame.columns:
        frame = frame[pd.to_numeric(frame["reference_metadata_coverage"], errors="coerce").fillna(0.0) >= args.min_reference_metadata_coverage].copy()
    return frame.reset_index(drop=True)


def target_sensitivity_rows(frame: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target in TARGET_CANDIDATES:
        if target not in frame.columns:
            continue
        result = run_nested_rank_learning(
            frame,
            feature_cols=BASE_FEATURE_COLS,
            target_col=target,
            seed=args.seed,
            max_pairs=args.max_pairs,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_grid=args.model_grid,
            l2_grid=args.l2_grid,
            target_transform="raw",
            source_summary={"n_contributing_graph_deltas": len([c for c in FUTURE_DELTA_COLS if c in frame.columns])},
        )
        run_dir = out_dir / f"target_{target}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result.oof_table.to_csv(run_dir / "fig3_v5_nature_full_oof_score_table.csv", index=False)
        result.model_selection.to_csv(run_dir / "fig3_v5_nature_full_model_selection.csv", index=False)
        result.cv_summary.to_csv(run_dir / "fig3_v5_nature_full_cv_summary.csv", index=False)
        write_json(run_dir / "fig3_v5_nature_full_diagnostics_summary.json", result.summary)
        rows.append(
            {
                "target_col": target,
                "run_dir": str(run_dir),
                "learned_oof_spearman": result.summary.get("learned_oof_spearman"),
                "equal_weight_oof_spearman": result.summary.get("equal_weight_oof_spearman"),
                "best_single_oof_spearman": result.summary.get("best_single_oof_spearman"),
                "latest_fold_test_spearman": result.summary.get("latest_fold_test_spearman"),
                "n_rows": int(len(result.oof_table)),
            }
        )
    return pd.DataFrame(rows).sort_values(["learned_oof_spearman", "latest_fold_test_spearman"], ascending=False)


def run_fig3_v5(args: argparse.Namespace) -> Dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_training_frame(args)
    frame.to_csv(args.out_dir / "fig3_v5_nature_full_feature_table.csv", index=False)
    violations = detect_no_leakage_feature_violations(frame, BASE_FEATURE_COLS)
    if violations:
        raise ValueError(f"Leakage-like feature columns detected: {violations}")
    sensitivity = target_sensitivity_rows(frame, args.out_dir, args)
    sensitivity_path = args.out_dir / "fig3_v5_nature_full_target_sensitivity.csv"
    sensitivity.to_csv(sensitivity_path, index=False)
    if sensitivity.empty:
        raise ValueError("No v5 target sensitivity runs completed.")
    best = sensitivity.iloc[0].to_dict()
    best_dir = Path(str(best["run_dir"]))
    oof = pd.read_csv(best_dir / "fig3_v5_nature_full_oof_score_table.csv", low_memory=False)
    model = pd.read_csv(best_dir / "fig3_v5_nature_full_model_selection.csv", low_memory=False)
    oof.to_csv(args.out_dir / "fig3_v5_nature_full_oof_score_table.csv", index=False)
    model.to_csv(args.out_dir / "fig3_v5_nature_full_model_selection.csv", index=False)
    manifest = {
        "artifact_kind": "fig3_v5_nature_full_run",
        "created_at": utc_now(),
        "corpus_dir": str(args.corpus_dir),
        "out_dir": str(args.out_dir),
        "feature_table": str(args.out_dir / "fig3_v5_nature_full_feature_table.csv"),
        "oof_score_table": str(args.out_dir / "fig3_v5_nature_full_oof_score_table.csv"),
        "model_selection": str(args.out_dir / "fig3_v5_nature_full_model_selection.csv"),
        "target_sensitivity": str(sensitivity_path),
        "selected_target": best.get("target_col"),
        "learned_oof_spearman": best.get("learned_oof_spearman"),
        "latest_fold_test_spearman": best.get("latest_fold_test_spearman"),
        "n_rows": int(len(frame)),
        "feature_cols": BASE_FEATURE_COLS,
        "no_leakage_feature_contract": "publication_day_reference_graph_features_only",
        "leakage_violations": violations,
    }
    write_json(args.out_dir / "fig3_v5_nature_full_manifest.json", manifest)
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fig3 v5 rank learning on Nature Portfolio full corpus.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_V5_CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_V5_OUTPUT_DIR / "fig3_nature_full_v5")
    parser.add_argument("--works", type=Path, default=None)
    parser.add_argument("--reference-works", type=Path, default=None)
    parser.add_argument("--future-graph-deltas", type=Path, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--min-reference-metadata-coverage", type=float, default=0.0)
    parser.add_argument("--model-grid", nargs="+", default=["simplex_pairwise", "signed_pairwise", "ridge_rank"])
    parser.add_argument("--l2-grid", nargs="+", type=float, default=[0.0, 0.001, 0.01, 0.1])
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    args.works = args.works or (args.corpus_dir / "works.csv")
    args.reference_works = args.reference_works or (args.corpus_dir / "reference_works.csv")
    args.future_graph_deltas = args.future_graph_deltas or (args.corpus_dir / "future_graph_deltas.csv")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = run_fig3_v5(args)
    if not args.quiet:
        print(
            f"[fig3-nature-v5] selected {manifest['selected_target']} "
            f"OOF={float(manifest['learned_oof_spearman']):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
