from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fig3_v5_rank_learning import (  # noqa: E402
    DEFAULT_FEATURE_COLS,
    add_feature_expansion,
    run_nested_rank_learning,
    safe_numeric,
    safe_spearman,
    write_json,
)


FUTURE_DELTA_COLS = [
    "community_reach",
    "field_entropy",
    "cross_community_adoption",
    "path_shortening",
    "modularity_shock",
    "partition_change",
    "boundary_mixing",
    "hub_formation",
]
PUBLICATION_DAY_FEATURES = [
    "degree_p",
    "effective_size",
    "constraint_inv",
    "field_variety",
    "field_simpson",
    "community_variety",
    "reference_count_log",
]
INFINITE_ENRICHMENT_SENTINEL = 999.0


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_fig3_tables(fig3_run_dir: Path) -> pd.DataFrame:
    score = pd.read_csv(fig3_run_dir / "fig3_oof_score_table.csv", low_memory=False)
    indicators = pd.read_csv(fig3_run_dir / "fig3_publication_day_indicators.csv", low_memory=False)
    future = pd.read_csv(fig3_run_dir / "fig3_future_graph_deltas.csv", low_memory=False)
    controls = pd.read_csv(fig3_run_dir / "fig3_diagnostics_controls.csv", low_memory=False)

    keep_indicator = [
        "paper_id",
        "degree_p",
        "effective_size",
        "constraint_inv",
        "field_variety",
        "field_simpson",
        "community_variety",
    ]
    keep_future = ["paper_id", "n_future_citers"] + [col for col in FUTURE_DELTA_COLS if col in future.columns]
    keep_controls = [
        "paper_id",
        "n_controls",
        "control_tier",
        "n_delta_scale_floor_used",
        "n_delta_mad_zero",
        "n_delta_z_clipped",
    ]
    frame = score.merge(indicators[[col for col in keep_indicator if col in indicators.columns]], on="paper_id", how="left")
    frame = frame.merge(future[[col for col in keep_future if col in future.columns]], on="paper_id", how="left")
    frame = frame.merge(controls[[col for col in keep_controls if col in controls.columns]], on="paper_id", how="left")
    frame["reference_count_log"] = np.log1p(safe_numeric(frame.get("reference_count", pd.Series(0, index=frame.index)), 0.0))
    return frame


def cohort_mask(frame: pd.DataFrame, cohort: str) -> pd.Series:
    n_future = safe_numeric(frame.get("n_future_citers", pd.Series(0, index=frame.index)), 0.0)
    n_controls = safe_numeric(frame.get("n_controls", pd.Series(0, index=frame.index)), 0.0)
    z_clipped = safe_numeric(frame.get("n_delta_z_clipped", pd.Series(99, index=frame.index)), 99.0)
    scale_floor = safe_numeric(frame.get("n_delta_scale_floor_used", pd.Series(99, index=frame.index)), 99.0)
    tier = frame.get("control_tier", pd.Series("", index=frame.index)).fillna("").astype(str)
    if cohort == "broad":
        return pd.Series(True, index=frame.index)
    if cohort == "moderate":
        return (n_future >= 5) & (n_controls >= 75) & (tier != "all_non_landmark") & (z_clipped <= 2)
    if cohort == "strict":
        strict_tiers = {"field_year", "field_year_refbin", "field_year3"}
        return (n_future >= 10) & (n_controls >= 100) & tier.isin(strict_tiers) & (z_clipped <= 1) & (scale_floor <= 5)
    raise ValueError(f"Unknown reliability cohort: {cohort}")


def add_reliability_cohorts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for cohort in ["broad", "moderate", "strict"]:
        out[f"cohort_{cohort}"] = cohort_mask(out, cohort).astype(int)
    return out


def robust_rank_frame(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in cols:
        values = safe_numeric(frame.get(col, pd.Series(np.nan, index=frame.index)), np.nan)
        if values.notna().sum() <= 2:
            out[col] = 0.0
            continue
        out[col] = values.rank(method="average", pct=True).fillna(0.5)
    return out


def first_pc(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 3 or x.shape[1] < 2:
        return np.zeros(x.shape[0], dtype=float)
    try:
        u, s, _ = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(x.shape[0], dtype=float)
    return u[:, 0] * s[0]


def add_latent_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    delta_cols = [col for col in FUTURE_DELTA_COLS if col in out.columns]
    ranked = robust_rank_frame(out, delta_cols)
    latent = first_pc(ranked.to_numpy(dtype=float))
    positive_anchor = ranked.mean(axis=1).to_numpy(dtype=float)
    if safe_spearman(latent, positive_anchor) < 0:
        latent = -latent
    out["RGPM_latent_future_factor"] = latent
    out["RGPM_latent_future_percentile"] = pd.Series(latent, index=out.index).rank(method="average", pct=True)
    return out


def strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): strict_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strict_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [strict_json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def summarize_enrichment(effect_summary: Mapping[str, Any]) -> Dict[str, Any]:
    values: List[float] = []
    has_infinite = False
    for key in [
        "top_vs_bottom_score_decile_rgpm_top10_enrichment",
        "top_vs_bottom_score_decile_rgpm_top20_enrichment",
    ]:
        raw = effect_summary.get(key, 0.0)
        try:
            value = float(raw or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value):
            values.append(value)
        elif value > 0:
            has_infinite = True
    finite_max = max(values) if values else 0.0
    gate_value = INFINITE_ENRICHMENT_SENTINEL if has_infinite else finite_max
    return {
        "top_bottom_enrichment": float(gate_value),
        "top_bottom_enrichment_finite_max": float(finite_max),
        "top_bottom_enrichment_had_infinite": bool(has_infinite),
    }


def feature_columns_for_set(feature_set: str) -> List[str]:
    if feature_set == "seven":
        return list(DEFAULT_FEATURE_COLS)
    if feature_set == "publication_day_plus":
        return list(DEFAULT_FEATURE_COLS) + list(PUBLICATION_DAY_FEATURES)
    raise ValueError(f"Unknown feature set: {feature_set}")


def summarize_domain_balance(frame: pd.DataFrame) -> Dict[str, Any]:
    counts = frame.get("domain", pd.Series("", index=frame.index)).astype(str).value_counts()
    return {
        "n_rows": int(len(frame)),
        "n_domains": int(counts.size),
        "min_rows_per_domain": int(counts.min()) if not counts.empty else 0,
        "max_domain_share": float(counts.max() / max(1, counts.sum())) if not counts.empty else 0.0,
        "domain_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
    }


def run_one_probe(
    frame: pd.DataFrame,
    *,
    cohort: str,
    target_col: str,
    feature_set: str,
    expansion: str,
    source_summary: Mapping[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    sub = frame[frame[f"cohort_{cohort}"].astype(int) == 1].copy().reset_index(drop=True)
    if int(args.cohort_domain_min_rows) > 0 and "domain" in sub.columns:
        counts = sub["domain"].astype(str).value_counts()
        keep_domains = set(counts[counts >= int(args.cohort_domain_min_rows)].index.astype(str))
        sub = sub[sub["domain"].astype(str).isin(keep_domains)].copy().reset_index(drop=True)
    feature_cols = feature_columns_for_set(feature_set)
    sub, expanded_features = add_feature_expansion(sub, feature_cols, expansion)
    result = run_nested_rank_learning(
        sub,
        feature_cols=expanded_features,
        target_col=target_col,
        target_transform="raw",
        source_summary=source_summary,
        seed=args.seed,
        max_pairs=args.max_pairs,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_grid=args.model_grid,
        l2_grid=args.l2_grid,
    )
    run_name = f"{cohort}__{target_col}__{feature_set}__{expansion}".replace("/", "_")
    run_dir = out_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result.oof_table.to_csv(run_dir / "fig3_v6a_oof_score_table.csv", index=False)
    result.cv_summary.to_csv(run_dir / "fig3_v6a_cv_summary.csv", index=False)
    result.model_selection.to_csv(run_dir / "fig3_v6a_model_selection.csv", index=False)
    write_json(run_dir / "fig3_v6a_effect_summary.json", strict_json_value(result.effect_summary))
    write_json(run_dir / "fig3_v6a_diagnostics_summary.json", strict_json_value(result.summary))
    balance = summarize_domain_balance(sub)
    latest = result.summary.get("latest_fold_test_spearman")
    enrichment_summary = summarize_enrichment(result.effect_summary)
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "cohort": cohort,
        "target_col": target_col,
        "feature_set": feature_set,
        "feature_expansion": expansion,
        "cohort_domain_min_rows": int(args.cohort_domain_min_rows),
        "n_rows": balance["n_rows"],
        "n_domains": balance["n_domains"],
        "min_rows_per_domain": balance["min_rows_per_domain"],
        "max_domain_share": balance["max_domain_share"],
        "learned_oof_spearman": result.summary.get("learned_oof_spearman"),
        "equal_weight_oof_spearman": result.summary.get("equal_weight_oof_spearman"),
        "learned_vs_equal_delta": result.summary.get("learned_vs_equal_delta"),
        "best_single_oof_spearman": result.summary.get("best_single_oof_spearman"),
        "latest_fold_test_spearman": latest,
        "n_contributing_graph_deltas": result.summary.get("n_contributing_graph_deltas"),
        **enrichment_summary,
    }


def pass_gate(row: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(row.get("learned_oof_spearman", 0.0) or 0.0) >= float(args.min_oof)
        and float(row.get("latest_fold_test_spearman", 0.0) or 0.0) >= float(args.min_latest_fold)
        and float(row.get("learned_vs_equal_delta", 0.0) or 0.0) >= float(args.min_learned_vs_equal)
        and int(row.get("n_contributing_graph_deltas", 0) or 0) >= int(args.min_contributing_deltas)
        and float(row.get("top_bottom_enrichment", 0.0) or 0.0) >= float(args.min_enrichment)
        and int(row.get("n_rows", 0) or 0) >= int(args.min_rows)
        and int(row.get("n_domains", 0) or 0) >= int(args.min_domains)
        and int(row.get("min_rows_per_domain", 0) or 0) >= int(args.min_rows_per_domain)
        and float(row.get("max_domain_share", 1.0) or 1.0) <= float(args.max_domain_share)
    )


def build_decision(matrix: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Any]:
    ranked = matrix.sort_values(
        ["v6a_gate_pass", "learned_oof_spearman", "latest_fold_test_spearman", "top_bottom_enrichment"],
        ascending=[False, False, False, False],
    )
    best = ranked.head(1).to_dict("records")
    final_pass = bool(matrix["v6a_gate_pass"].astype(bool).any()) if not matrix.empty else False
    return strict_json_value({
        "created_at": utc_now(),
        "artifact_kind": "fig3_v6a_reliability_latent_probe_decision",
        "policy": "pre_registered_reliability_cohort_latent_target_probe",
        "thresholds": {
            "min_oof": float(args.min_oof),
            "min_latest_fold": float(args.min_latest_fold),
            "min_learned_vs_equal": float(args.min_learned_vs_equal),
            "min_contributing_deltas": int(args.min_contributing_deltas),
            "min_enrichment": float(args.min_enrichment),
            "min_rows": int(args.min_rows),
            "min_domains": int(args.min_domains),
            "min_rows_per_domain": int(args.min_rows_per_domain),
            "max_domain_share": float(args.max_domain_share),
        },
        "final_pass": final_pass,
        "best_run": best[0] if best else {},
        "n_runs": int(len(matrix)),
        "n_passing_runs": int(matrix["v6a_gate_pass"].astype(bool).sum()) if "v6a_gate_pass" in matrix.columns else 0,
        "next_step": "independent_recompute_and_materialization_gate"
        if final_pass
        else "round2_event_centric_graph_or_new_publication_day_features",
    })


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    frame = read_fig3_tables(args.fig3_run_dir)
    frame = add_reliability_cohorts(add_latent_targets(frame))
    source_summary = read_json(args.fig3_run_dir / "fig3_diagnostics_summary.json")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "fig3_v6a_augmented_input.csv", index=False)

    rows: List[Dict[str, Any]] = []
    for cohort in args.cohorts:
        for target_col in args.target_cols:
            if target_col not in frame.columns:
                continue
            for feature_set in args.feature_sets:
                for expansion in args.feature_expansions:
                    rows.append(
                        run_one_probe(
                            frame,
                            cohort=cohort,
                            target_col=target_col,
                            feature_set=feature_set,
                            expansion=expansion,
                            source_summary=source_summary,
                            args=args,
                            out_dir=args.out_dir,
                        )
                    )
    matrix = pd.DataFrame(rows)
    if not matrix.empty:
        matrix["v6a_gate_pass"] = [int(pass_gate(row, args)) for row in matrix.to_dict("records")]
        matrix = matrix.sort_values(["v6a_gate_pass", "learned_oof_spearman"], ascending=[False, False])
    matrix.to_csv(args.out_dir / "fig3_v6a_probe_matrix.csv", index=False)
    decision = build_decision(matrix, args)
    write_json(args.out_dir / "fig3_v6a_probe_decision.json", decision)
    return decision


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig3 v6A reliability-gated latent-target probe.")
    parser.add_argument("--fig3-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", default=["broad", "moderate", "strict"])
    parser.add_argument(
        "--target-cols",
        nargs="+",
        default=["RGPM", "RGPM_latent_future_percentile", "RGPM_v3_balanced"],
    )
    parser.add_argument("--feature-sets", nargs="+", default=["seven", "publication_day_plus"])
    parser.add_argument("--feature-expansions", nargs="+", default=["linear", "interactions"])
    parser.add_argument(
        "--cohort-domain-min-rows",
        type=int,
        default=0,
        help="Drop domains with fewer rows inside a reliability cohort before training. Default keeps all domains.",
    )
    parser.add_argument("--model-grid", nargs="+", default=["simplex_pairwise", "signed_pairwise", "ridge_rank"])
    parser.add_argument("--l2-grid", nargs="+", type=float, default=[0.0, 0.001, 0.01, 0.1])
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--min-oof", type=float, default=0.45)
    parser.add_argument("--min-latest-fold", type=float, default=0.35)
    parser.add_argument("--min-learned-vs-equal", type=float, default=0.03)
    parser.add_argument("--min-contributing-deltas", type=int, default=5)
    parser.add_argument("--min-enrichment", type=float, default=5.0)
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-domains", type=int, default=8)
    parser.add_argument("--min-rows-per-domain", type=int, default=20)
    parser.add_argument("--max-domain-share", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    decision = run_probe(args)
    if not args.quiet:
        status = "PASS" if decision.get("final_pass") else "FAIL"
        print(f"[fig3-v6a] {status}; best={decision.get('best_run', {})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
