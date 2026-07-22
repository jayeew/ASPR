from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_FEATURE_COLS = ["B_z", "RS_z", "DeltaQ0_z", "Uzzi_z", "RTD_z", "BurtIP_z", "PDE_z"]
DEFAULT_MODEL_GRID = ["simplex_pairwise", "signed_pairwise", "ridge_rank"]
DEFAULT_L2_GRID = [0.0, 0.001, 0.01, 0.1]
TRANSFORMED_TARGET_COL = "_v5_training_target"


@dataclass
class TimeFold:
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class FittedLinearModel:
    model_kind: str
    weights: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    l2: float
    inner_spearman: float

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x = (x_raw - self.mean) / self.scale
        return x @ self.weights


@dataclass
class RankLearningResult:
    oof_table: pd.DataFrame
    cv_summary: pd.DataFrame
    model_selection: pd.DataFrame
    effect_summary: Dict[str, Any]
    summary: Dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_numeric(values: Any, default: float = 0.0) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(dtype=float)


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return float("nan")
    if frame["x"].nunique(dropna=True) <= 1 or frame["y"].nunique(dropna=True) <= 1:
        return float("nan")
    return float(frame["x"].corr(frame["y"], method="spearman"))


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Project a vector onto the non-negative unit simplex."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v.copy()
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, len(u) + 1) > (cssv - 1.0)
    if not rho_candidates.any():
        return np.ones_like(v) / float(len(v))
    rho = int(np.flatnonzero(rho_candidates)[-1])
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)
    total = float(w.sum())
    return w / total if total > 0 else np.ones_like(v) / float(len(v))


def outer_time_folds(frame: pd.DataFrame) -> List[TimeFold]:
    fold_ids = pd.to_numeric(frame["fold_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
    out: List[TimeFold] = []
    for fold in sorted(int(v) for v in np.unique(fold_ids) if int(v) > 0):
        train_idx = np.flatnonzero(fold_ids < fold)
        test_idx = np.flatnonzero(fold_ids == fold)
        if len(train_idx) and len(test_idx):
            out.append(TimeFold(fold_id=fold, train_idx=train_idx, test_idx=test_idx))
    return out


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return mean, scale


def standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def feature_matrix(frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    cols = []
    for col in feature_cols:
        if col in frame.columns:
            cols.append(safe_numeric(frame[col], 0.0).to_numpy(dtype=float))
        else:
            cols.append(np.zeros(len(frame), dtype=float))
    return np.vstack(cols).T if cols else np.zeros((len(frame), 0), dtype=float)


def add_feature_expansion(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    expansion: str,
) -> tuple[pd.DataFrame, List[str]]:
    if expansion == "linear":
        return frame.copy(), list(feature_cols)
    if expansion != "interactions":
        raise ValueError(f"Unknown feature expansion: {expansion}")
    out = frame.copy()
    expanded = list(feature_cols)
    for col in feature_cols:
        new_col = f"{col}_sq"
        out[new_col] = safe_numeric(out.get(col, pd.Series(0.0, index=out.index)), 0.0) ** 2
        expanded.append(new_col)
    for i, left in enumerate(feature_cols):
        left_values = safe_numeric(out.get(left, pd.Series(0.0, index=out.index)), 0.0)
        for right in feature_cols[i + 1 :]:
            new_col = f"{left}_x_{right}"
            out[new_col] = left_values * safe_numeric(out.get(right, pd.Series(0.0, index=out.index)), 0.0)
            expanded.append(new_col)
    return out, expanded


def sample_pair_diffs(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    max_pairs: int,
) -> np.ndarray:
    valid = np.flatnonzero(np.isfinite(y))
    if len(valid) < 3:
        return np.zeros((0, x.shape[1]), dtype=float)
    ranks = pd.Series(y[valid]).rank(method="average", pct=True).to_numpy(dtype=float)
    top = valid[ranks >= 0.60]
    bottom = valid[ranks <= 0.40]
    n_pairs = int(max(100, max_pairs))
    if len(top) and len(bottom):
        top_idx = rng.choice(top, size=n_pairs // 2, replace=True)
        bottom_idx = rng.choice(bottom, size=n_pairs // 2, replace=True)
        first = np.concatenate([top_idx, rng.choice(valid, size=n_pairs - len(top_idx), replace=True)])
        second = np.concatenate([bottom_idx, rng.choice(valid, size=n_pairs - len(bottom_idx), replace=True)])
    else:
        first = rng.choice(valid, size=n_pairs, replace=True)
        second = rng.choice(valid, size=n_pairs, replace=True)
    swap = y[first] < y[second]
    hi = first.copy()
    lo = second.copy()
    hi[swap], lo[swap] = second[swap], first[swap]
    keep = np.isfinite(y[hi]) & np.isfinite(y[lo]) & (np.abs(y[hi] - y[lo]) > 1e-12)
    if not keep.any():
        return np.zeros((0, x.shape[1]), dtype=float)
    return x[hi[keep]] - x[lo[keep]]


def fit_pairwise_ranker(
    x: np.ndarray,
    y: np.ndarray,
    *,
    model_kind: str,
    l2: float,
    seed: int,
    max_pairs: int,
    epochs: int,
    learning_rate: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = sample_pair_diffs(x, y, rng=rng, max_pairs=max_pairs)
    n_features = int(x.shape[1])
    if diffs.shape[0] == 0:
        return np.ones(n_features, dtype=float) / max(1, n_features)
    if model_kind == "simplex_pairwise":
        weights = np.ones(n_features, dtype=float) / max(1, n_features)
    else:
        weights = np.zeros(n_features, dtype=float)
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for step in range(1, int(epochs) + 1):
        margins = np.clip(diffs @ weights, -50.0, 50.0)
        coef = -1.0 / (1.0 + np.exp(margins))
        grad = (diffs * coef[:, None]).mean(axis=0) + float(l2) * weights
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        update = learning_rate * (m / (1.0 - beta1**step)) / (np.sqrt(v / (1.0 - beta2**step)) + eps)
        weights = weights - update
        if model_kind == "simplex_pairwise":
            weights = project_simplex(weights)
        elif np.linalg.norm(weights) > 10.0:
            weights = weights / np.linalg.norm(weights) * 10.0
    return weights


def fit_ridge_rank(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return np.zeros(x.shape[1], dtype=float)
    target_rank = pd.Series(y[valid]).rank(method="average", pct=True).to_numpy(dtype=float)
    target_rank = target_rank - float(np.mean(target_rank))
    xtx = x[valid].T @ x[valid]
    penalty = float(l2) * np.eye(x.shape[1])
    try:
        return np.linalg.solve(xtx + penalty, x[valid].T @ target_rank)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + penalty) @ (x[valid].T @ target_rank)


def fit_linear_rank_model(
    train_frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    model_kind: str,
    l2: float,
    seed: int,
    max_pairs: int,
    epochs: int,
    learning_rate: float,
    inner_spearman: float = float("nan"),
) -> FittedLinearModel:
    x_raw = feature_matrix(train_frame, feature_cols)
    y = safe_numeric(train_frame[target_col], np.nan).to_numpy(dtype=float)
    mean, scale = standardize_fit(x_raw)
    x = standardize_apply(x_raw, mean, scale)
    if model_kind in {"simplex_pairwise", "signed_pairwise"}:
        weights = fit_pairwise_ranker(
            x,
            y,
            model_kind=model_kind,
            l2=l2,
            seed=seed,
            max_pairs=max_pairs,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    elif model_kind == "ridge_rank":
        weights = fit_ridge_rank(x, y, l2=max(l2, 1e-6))
    else:
        raise ValueError(f"Unknown v5 model kind: {model_kind}")
    return FittedLinearModel(
        model_kind=model_kind,
        weights=weights,
        mean=mean,
        scale=scale,
        l2=float(l2),
        inner_spearman=float(inner_spearman),
    )


def choose_model_for_outer_fold(
    train_frame: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    *,
    model_grid: Sequence[str],
    l2_grid: Sequence[float],
    seed: int,
    max_pairs: int,
    epochs: int,
    learning_rate: float,
) -> FittedLinearModel:
    folds = outer_time_folds(train_frame.reset_index(drop=True))
    if folds:
        inner = folds[-1]
        inner_train = train_frame.iloc[inner.train_idx].copy()
        inner_valid = train_frame.iloc[inner.test_idx].copy()
    else:
        inner_train = train_frame.copy()
        inner_valid = train_frame.copy()
    best_model: Optional[FittedLinearModel] = None
    best_score = -float("inf")
    for model_kind in model_grid:
        for l2 in l2_grid:
            model = fit_linear_rank_model(
                inner_train,
                feature_cols,
                target_col,
                model_kind=model_kind,
                l2=float(l2),
                seed=seed,
                max_pairs=max_pairs,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            score = safe_spearman(model.predict(feature_matrix(inner_valid, feature_cols)), inner_valid[target_col])
            if not math.isfinite(score):
                score = -float("inf")
            if best_model is None or score > best_score:
                best_score = score
                best_model = FittedLinearModel(
                    model_kind=model.model_kind,
                    weights=model.weights,
                    mean=model.mean,
                    scale=model.scale,
                    l2=model.l2,
                    inner_spearman=float(score),
                )
    if best_model is None:
        raise ValueError("No v5 rank-learning model could be selected")
    return fit_linear_rank_model(
        train_frame,
        feature_cols,
        target_col,
        model_kind=best_model.model_kind,
        l2=best_model.l2,
        seed=seed + 17,
        max_pairs=max_pairs,
        epochs=epochs,
        learning_rate=learning_rate,
        inner_spearman=best_model.inner_spearman,
    )


def equal_score(frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if "S_equal" in frame.columns:
        return safe_numeric(frame["S_equal"], np.nan).to_numpy(dtype=float)
    x = feature_matrix(frame, feature_cols)
    return np.nanmean(x, axis=1)


def apply_target_transform(frame: pd.DataFrame, target_col: str, transform: str) -> pd.DataFrame:
    out = frame.copy()
    target = safe_numeric(out[target_col], np.nan)
    if transform == "raw":
        out[TRANSFORMED_TARGET_COL] = target
    elif transform == "within_domain_percentile":
        if "domain" not in out.columns:
            raise ValueError("within_domain_percentile target transform requires a domain column")
        out[TRANSFORMED_TARGET_COL] = (
            target.groupby(out["domain"].astype(str)).rank(method="average", pct=True)
        )
    elif transform == "within_domain_year_residual":
        if "domain" not in out.columns or "year" not in out.columns:
            raise ValueError("within_domain_year_residual target transform requires domain and year columns")
        year = safe_numeric(out["year"], np.nan)
        domain_center = target.groupby(out["domain"].astype(str)).transform("median")
        year_bin = (np.floor(year / 5.0) * 5.0).fillna(-1).astype(int)
        year_center = target.groupby(year_bin).transform("median")
        global_center = float(np.nanmedian(target))
        out[TRANSFORMED_TARGET_COL] = target - domain_center - year_center + global_center
    else:
        raise ValueError(f"Unknown target transform: {transform}")
    return out


def compute_effect_summary(score: Sequence[float], target: Sequence[float]) -> Dict[str, Any]:
    frame = pd.DataFrame({"score": score, "target": target}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return {}
    frame["target_percentile"] = frame["target"].rank(method="average", pct=True) * 100.0
    try:
        frame["score_tertile"] = pd.qcut(frame["score"].rank(method="first"), q=3, labels=["low", "mid", "high"])
    except ValueError:
        frame["score_tertile"] = "mid"
    try:
        frame["score_decile"] = pd.qcut(frame["score"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    except ValueError:
        frame["score_decile"] = np.ceil(frame["score"].rank(method="average", pct=True) * 10).clip(1, 10).astype(int)
    low = frame[frame["score_tertile"].astype(str) == "low"]["target_percentile"]
    high = frame[frame["score_tertile"].astype(str) == "high"]["target_percentile"]
    top_decile = frame[frame["score_decile"] == frame["score_decile"].max()]
    bottom_decile = frame[frame["score_decile"] == frame["score_decile"].min()]
    top20_top = float((top_decile["target_percentile"] >= 80.0).mean()) if not top_decile.empty else float("nan")
    top20_bottom = float((bottom_decile["target_percentile"] >= 80.0).mean()) if not bottom_decile.empty else float("nan")
    top10_top = float((top_decile["target_percentile"] >= 90.0).mean()) if not top_decile.empty else float("nan")
    top10_bottom = float((bottom_decile["target_percentile"] >= 90.0).mean()) if not bottom_decile.empty else float("nan")
    return {
        "n": int(len(frame)),
        "high_score_tertile_rgpm_percentile_median": float(np.nanmedian(high)) if len(high) else float("nan"),
        "low_score_tertile_rgpm_percentile_median": float(np.nanmedian(low)) if len(low) else float("nan"),
        "high_vs_low_tertile_median_rgpm_lift_pp": float(np.nanmedian(high) - np.nanmedian(low))
        if len(high) and len(low)
        else float("nan"),
        "top_score_decile_rgpm_top20_rate": top20_top,
        "bottom_score_decile_rgpm_top20_rate": top20_bottom,
        "top_vs_bottom_score_decile_rgpm_top20_enrichment": float(top20_top / top20_bottom)
        if top20_bottom and math.isfinite(top20_bottom)
        else float("inf") if top20_top > 0 else float("nan"),
        "top_score_decile_rgpm_top10_rate": top10_top,
        "bottom_score_decile_rgpm_top10_rate": top10_bottom,
        "top_vs_bottom_score_decile_rgpm_top10_enrichment": float(top10_top / top10_bottom)
        if top10_bottom and math.isfinite(top10_bottom)
        else float("inf") if top10_top > 0 else float("nan"),
    }


def run_nested_rank_learning(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    seed: int = 2028,
    max_pairs: int = 50_000,
    epochs: int = 400,
    learning_rate: float = 0.05,
    model_grid: Sequence[str] = DEFAULT_MODEL_GRID,
    l2_grid: Sequence[float] = DEFAULT_L2_GRID,
    target_transform: str = "raw",
    source_summary: Optional[Mapping[str, Any]] = None,
) -> RankLearningResult:
    data = apply_target_transform(frame.copy().reset_index(drop=True), target_col, target_transform)
    data["fold_id"] = safe_numeric(data.get("fold_id", pd.Series(-1, index=data.index)), -1).astype(int)
    data[target_col] = safe_numeric(data[target_col], np.nan)
    data[TRANSFORMED_TARGET_COL] = safe_numeric(data[TRANSFORMED_TARGET_COL], np.nan)
    oof = np.full(len(data), np.nan, dtype=float)
    model_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    for fold in outer_time_folds(data):
        train = data.iloc[fold.train_idx].copy()
        test = data.iloc[fold.test_idx].copy()
        model = choose_model_for_outer_fold(
            train,
            feature_cols,
                TRANSFORMED_TARGET_COL,
            model_grid=model_grid,
            l2_grid=l2_grid,
            seed=seed + fold.fold_id * 1009,
            max_pairs=max_pairs,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        train_score = model.predict(feature_matrix(train, feature_cols))
        test_score = model.predict(feature_matrix(test, feature_cols))
        oof[fold.test_idx] = test_score
        cv_rows.append(
            {
                "fold": int(fold.fold_id),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "model_kind": model.model_kind,
                "l2": model.l2,
                "inner_spearman": model.inner_spearman,
                "train_spearman": safe_spearman(train_score, train[TRANSFORMED_TARGET_COL]),
                "test_spearman": safe_spearman(test_score, test[TRANSFORMED_TARGET_COL]),
            }
        )
        row = {"fold": int(fold.fold_id), "model_kind": model.model_kind, "l2": model.l2}
        for col, weight in zip(feature_cols, model.weights):
            row[f"w_{col.replace('_z', '')}"] = float(weight)
        model_rows.append(row)

    mask = np.isfinite(oof) & np.isfinite(data[TRANSFORMED_TARGET_COL].to_numpy(dtype=float))
    equal = equal_score(data, feature_cols)
    learned = safe_spearman(oof[mask], data.loc[mask, TRANSFORMED_TARGET_COL])
    equal_rho = safe_spearman(equal[mask], data.loc[mask, TRANSFORMED_TARGET_COL])
    best_single = max(
        [
            safe_spearman(data.loc[mask, col], data.loc[mask, TRANSFORMED_TARGET_COL])
            for col in feature_cols
            if col in data.columns
        ],
        default=float("nan"),
    )
    effect = compute_effect_summary(oof[mask], data.loc[mask, TRANSFORMED_TARGET_COL])
    cv = pd.DataFrame(cv_rows)
    latest = cv.sort_values("fold").tail(1)
    latest_spearman = float(latest["test_spearman"].iloc[0]) if not latest.empty else float("nan")
    source = dict(source_summary or {})
    summary = {
        "created_at": utc_now(),
        "artifact_kind": "fig3_v5_rank_learning",
        "method": "nested_time_block_linear_rank_learning",
        "target_col": target_col,
        "target_transform": target_transform,
        "feature_cols": list(feature_cols),
        "model_grid": list(model_grid),
        "l2_grid": [float(v) for v in l2_grid],
        "learned_oof_spearman": learned,
        "equal_weight_oof_spearman": equal_rho,
        "learned_vs_equal_delta": learned - equal_rho if math.isfinite(learned) and math.isfinite(equal_rho) else float("nan"),
        "best_single_oof_spearman": best_single,
        "learned_vs_best_single_delta": learned - best_single if math.isfinite(learned) and math.isfinite(best_single) else float("nan"),
        "latest_fold_test_spearman": latest_spearman,
        "n_contributing_graph_deltas": int(source.get("n_contributing_graph_deltas", 0) or 0),
        "contributing_graph_deltas": source.get("contributing_graph_deltas", []),
        "source_fig3_status_label": source.get("status_label", ""),
        "data_profile": source.get("data_profile", {}),
        "effect_summary": effect,
    }
    out = data.copy()
    out["S_v5_oof"] = oof
    out["S_equal_v5_reference"] = equal
    out["v5_target_raw"] = out[target_col]
    out["v5_target"] = out[TRANSFORMED_TARGET_COL]
    return RankLearningResult(
        oof_table=out,
        cv_summary=cv,
        model_selection=pd.DataFrame(model_rows),
        effect_summary=effect,
        summary=summary,
    )


def read_source_summary(fig3_run_dir: Path) -> Dict[str, Any]:
    path = fig3_run_dir / "fig3_diagnostics_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_from_dir(args: argparse.Namespace) -> RankLearningResult:
    score_path = args.score_table or (args.fig3_run_dir / "fig3_oof_score_table.csv")
    if not score_path.exists():
        raise FileNotFoundError(f"Missing Fig3 score table: {score_path}")
    frame = pd.read_csv(score_path, low_memory=False)
    if args.target_col not in frame.columns:
        raise ValueError(f"Target column not found in score table: {args.target_col}")
    frame, feature_cols = add_feature_expansion(frame, args.feature_cols, args.feature_expansion)
    source_summary = read_source_summary(args.fig3_run_dir)
    return run_nested_rank_learning(
        frame,
        feature_cols=feature_cols,
        target_col=args.target_col,
        seed=args.seed,
        max_pairs=args.max_pairs,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_grid=args.model_grid,
        l2_grid=args.l2_grid,
        target_transform=args.target_transform,
        source_summary=source_summary,
    )


def write_outputs(result: RankLearningResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.oof_table.to_csv(out_dir / "fig3_v5_oof_score_table.csv", index=False)
    result.cv_summary.to_csv(out_dir / "fig3_v5_cv_summary.csv", index=False)
    result.model_selection.to_csv(out_dir / "fig3_v5_model_selection.csv", index=False)
    write_json(out_dir / "fig3_v5_effect_summary.json", result.effect_summary)
    write_json(out_dir / "fig3_v5_diagnostics_summary.json", result.summary)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig3 v5 nested rank-learning method experiment.")
    parser.add_argument("--fig3-run-dir", type=Path, required=True)
    parser.add_argument("--score-table", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-col", default="RGPM")
    parser.add_argument(
        "--target-transform",
        choices=["raw", "within_domain_percentile", "within_domain_year_residual"],
        default="raw",
    )
    parser.add_argument("--feature-cols", nargs="+", default=DEFAULT_FEATURE_COLS)
    parser.add_argument("--feature-expansion", choices=["linear", "interactions"], default="linear")
    parser.add_argument("--model-grid", nargs="+", default=DEFAULT_MODEL_GRID)
    parser.add_argument("--l2-grid", nargs="+", type=float, default=DEFAULT_L2_GRID)
    parser.add_argument("--max-pairs", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_from_dir(args)
    write_outputs(result, args.out_dir)
    if not args.quiet:
        print(
            "[fig3-v5] "
            f"OOF={result.summary['learned_oof_spearman']:.3f} "
            f"latest={result.summary['latest_fold_test_spearman']:.3f} "
            f"delta_vs_equal={result.summary['learned_vs_equal_delta']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
