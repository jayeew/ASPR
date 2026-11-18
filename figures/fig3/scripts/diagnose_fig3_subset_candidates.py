from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_TARGET_OOF = 0.45


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path, low_memory=False)


def safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_spearman(a: Sequence[object], b: Sequence[object]) -> float:
    frame = pd.DataFrame({"a": pd.to_numeric(pd.Series(a), errors="coerce"), "b": pd.to_numeric(pd.Series(b), errors="coerce")})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 4 or frame["a"].nunique() <= 1 or frame["b"].nunique() <= 1:
        return float("nan")
    return float(frame["a"].rank(method="average").corr(frame["b"].rank(method="average")))


def top_tail_mean(values: pd.Series, q: float = 0.20) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    n = max(1, int(np.ceil(len(numeric) * float(q))))
    return float(numeric.nlargest(n).mean())


def effect_summary(frame: pd.DataFrame, score_col: str, target_col: str) -> Dict[str, float]:
    st = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, target_col]).copy()
    if len(st) < 30:
        return {
            "high_vs_low_tertile_median_rgpm_lift_pp": float("nan"),
            "top_vs_bottom_score_decile_rgpm_top20_enrichment": float("nan"),
        }
    st["rgpm_percentile"] = st[target_col].rank(method="average", pct=True) * 100.0
    try:
        st["score_tertile"] = pd.qcut(st[score_col].rank(method="first"), q=3, labels=["low", "mid", "high"])
    except ValueError:
        pct = st[score_col].rank(method="average", pct=True)
        st["score_tertile"] = pd.cut(pct, bins=[0.0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"], include_lowest=True)
    low_med = safe_float(st.loc[st["score_tertile"].eq("low"), "rgpm_percentile"].median())
    high_med = safe_float(st.loc[st["score_tertile"].eq("high"), "rgpm_percentile"].median())
    try:
        st["score_decile"] = pd.qcut(st[score_col].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    except ValueError:
        st["score_decile"] = np.ceil(st[score_col].rank(method="average", pct=True) * 10).clip(1, 10).astype(int)
    rgpm_top20 = st[target_col] >= st[target_col].quantile(0.80)
    top = st["score_decile"].eq(st["score_decile"].max())
    bottom = st["score_decile"].eq(st["score_decile"].min())
    top_rate = float(rgpm_top20[top].mean()) if top.any() else float("nan")
    bottom_rate = float(rgpm_top20[bottom].mean()) if bottom.any() else float("nan")
    enrichment = top_rate / bottom_rate if np.isfinite(bottom_rate) and bottom_rate > 0 else float("nan")
    return {
        "high_vs_low_tertile_median_rgpm_lift_pp": high_med - low_med,
        "top_vs_bottom_score_decile_rgpm_top20_enrichment": enrichment,
    }


def load_family_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("domains", []) if isinstance(data, Mapping) else []
    return {str(row.get("domain_id") or row.get("domain") or ""): str(row.get("family") or "unknown") for row in rows}


def family_balance(domains: Sequence[str], family_map: Mapping[str, str]) -> Tuple[str, int, float]:
    families = [family_map.get(domain, "unknown") for domain in domains]
    if not families:
        return "", 0, float("nan")
    counts = pd.Series(families).value_counts()
    top_family = str(counts.index[0])
    top_count = int(counts.iloc[0])
    return top_family, top_count, float(top_count / len(families))


def score_subset(
    score_table: pd.DataFrame,
    domains: Sequence[str],
    score_col: str,
    target_col: str,
    family_map: Mapping[str, str],
    max_year: Optional[int] = None,
) -> Dict[str, Any]:
    sub = score_table[score_table["domain"].astype(str).isin(set(domains))].copy()
    if max_year is not None:
        sub = sub[pd.to_numeric(sub["year"], errors="coerce") <= int(max_year)].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, target_col])
    present_domains = sorted(sub["domain"].astype(str).unique().tolist())
    per_domain = sub.groupby("domain").size() if not sub.empty else pd.Series(dtype=int)
    effects = effect_summary(sub, score_col, target_col)
    top_family, top_family_count, top_family_share = family_balance(present_domains, family_map)
    equal_col = "S_equal" if "S_equal" in sub.columns else ""
    return {
        "domains": " ".join(present_domains),
        "n_domains": int(len(present_domains)),
        "max_year": int(max_year) if max_year is not None else "",
        "n_papers": int(len(sub)),
        "min_papers_per_domain": int(per_domain.min()) if len(per_domain) else 0,
        "learned_oof_spearman": safe_spearman(sub[score_col], sub[target_col]) if not sub.empty else float("nan"),
        "equal_weight_spearman": safe_spearman(sub[equal_col], sub[target_col]) if equal_col else float("nan"),
        "learned_vs_equal_delta": (
            safe_spearman(sub[score_col], sub[target_col]) - safe_spearman(sub[equal_col], sub[target_col])
            if equal_col
            else float("nan")
        ),
        "top_family": top_family,
        "top_family_count": top_family_count,
        "top_family_share": top_family_share,
        **effects,
    }


def build_domain_summary(score_table: pd.DataFrame, score_col: str, target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for domain, sub in score_table.groupby("domain", sort=True):
        clean = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, target_col])
        row = {
            "domain": str(domain),
            "n_papers": int(len(clean)),
            "min_year": int(pd.to_numeric(clean["year"], errors="coerce").min()) if not clean.empty else "",
            "max_year": int(pd.to_numeric(clean["year"], errors="coerce").max()) if not clean.empty else "",
            "learned_oof_spearman": safe_spearman(clean[score_col], clean[target_col]),
            "equal_weight_spearman": safe_spearman(clean["S_equal"], clean[target_col]) if "S_equal" in clean.columns else float("nan"),
            "s_w_spearman": safe_spearman(clean["S_w"], clean[target_col]) if "S_w" in clean.columns else float("nan"),
            "median_rgpm": safe_float(pd.to_numeric(clean[target_col], errors="coerce").median()),
        }
        row.update(effect_summary(clean, score_col, target_col))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("learned_oof_spearman", ascending=False, na_position="last")


def build_fold_summary(score_table: pd.DataFrame, score_col: str, target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold_id, sub in score_table.groupby("fold_id", sort=True):
        fold = int(safe_float(fold_id, -1))
        if fold <= 0:
            continue
        clean = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, target_col])
        rows.append(
            {
                "fold_id": fold,
                "n_papers": int(len(clean)),
                "min_year": int(pd.to_numeric(clean["year"], errors="coerce").min()) if not clean.empty else "",
                "max_year": int(pd.to_numeric(clean["year"], errors="coerce").max()) if not clean.empty else "",
                "learned_oof_spearman": safe_spearman(clean[score_col], clean[target_col]),
                "equal_weight_spearman": safe_spearman(clean["S_equal"], clean[target_col]) if "S_equal" in clean.columns else float("nan"),
                "n_domains": int(clean["domain"].nunique()) if "domain" in clean.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def build_domain_fold_summary(score_table: pd.DataFrame, score_col: str, target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (domain, fold_id), sub in score_table.groupby(["domain", "fold_id"], sort=True):
        fold = int(safe_float(fold_id, -1))
        if fold <= 0:
            continue
        clean = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, target_col])
        rows.append(
            {
                "domain": str(domain),
                "fold_id": fold,
                "n_papers": int(len(clean)),
                "min_year": int(pd.to_numeric(clean["year"], errors="coerce").min()) if not clean.empty else "",
                "max_year": int(pd.to_numeric(clean["year"], errors="coerce").max()) if not clean.empty else "",
                "learned_oof_spearman": safe_spearman(clean[score_col], clean[target_col]),
                "equal_weight_spearman": safe_spearman(clean["S_equal"], clean[target_col]) if "S_equal" in clean.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["fold_id", "learned_oof_spearman"], ascending=[True, False], na_position="last")


def generate_subset_candidates(
    score_table: pd.DataFrame,
    score_col: str,
    target_col: str,
    family_map: Mapping[str, str],
    min_domains: int,
    max_domains: int,
    year_cutoffs: Sequence[Optional[int]],
) -> pd.DataFrame:
    all_domains = sorted(score_table["domain"].astype(str).unique().tolist())
    rows: List[Dict[str, Any]] = []
    for n_domains in range(int(min_domains), min(int(max_domains), len(all_domains)) + 1):
        for domains in itertools.combinations(all_domains, n_domains):
            for max_year in year_cutoffs:
                row = score_subset(score_table, domains, score_col, target_col, family_map, max_year=max_year)
                row["candidate_id"] = f"d{n_domains}_y{max_year or 'all'}_{len(rows) + 1:05d}"
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["passes_family_balance"] = (out["top_family_share"] <= 0.50).astype(int)
    out["passes_min_domains"] = out["n_domains"].between(int(min_domains), int(max_domains)).astype(int)
    out["passes_min_domain_papers"] = (out["min_papers_per_domain"] >= 300).astype(int)
    out["passes_top_enrichment"] = (out["top_vs_bottom_score_decile_rgpm_top20_enrichment"] >= 5.0).astype(int)
    out["oof_gap_to_0_45"] = (DEFAULT_TARGET_OOF - out["learned_oof_spearman"]).clip(lower=0.0)
    out["screening_score"] = (
        out["learned_oof_spearman"].fillna(-1.0)
        + 0.05 * out["learned_vs_equal_delta"].fillna(0.0)
        + 0.02 * np.minimum(out["top_vs_bottom_score_decile_rgpm_top20_enrichment"].fillna(0.0), 8.0)
        + 0.03 * out["passes_family_balance"]
        + 0.02 * out["passes_min_domain_papers"]
    )
    return out.sort_values(
        ["passes_family_balance", "passes_min_domain_papers", "learned_oof_spearman", "top_vs_bottom_score_decile_rgpm_top20_enrichment"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def parse_year_cutoffs(text: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for item in str(text or "all").split(","):
        item = item.strip()
        if not item or item.lower() in {"all", "none"}:
            out.append(None)
        else:
            out.append(int(item))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Fig. 3 domain/year subset candidates from an existing score table.")
    parser.add_argument("--fig3-run-dir", type=Path, required=True, help="Directory containing fig3_score_table.csv.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for diagnostics.")
    parser.add_argument("--score-col", default="S_w_oof", help="Score column to evaluate.")
    parser.add_argument("--target-col", default="RGPM", help="Target outcome column.")
    parser.add_argument("--family-roster", type=Path, default=None, help="Optional JSON roster with domain family metadata.")
    parser.add_argument("--min-domains", type=int, default=10)
    parser.add_argument("--max-domains", type=int, default=12)
    parser.add_argument("--year-cutoffs", default="all,2012,2013,2014,2015", help="Comma-separated max years; use all for no cutoff.")
    args = parser.parse_args()

    score_table = read_csv_required(args.fig3_run_dir / "fig3_score_table.csv")
    for col in [args.score_col, args.target_col, "domain", "year"]:
        if col not in score_table.columns:
            raise ValueError(f"fig3_score_table.csv is missing required column {col}")
    score_table = score_table.copy()
    score_table["year"] = pd.to_numeric(score_table["year"], errors="coerce")
    family_map = load_family_map(args.family_roster)
    year_cutoffs = parse_year_cutoffs(args.year_cutoffs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    domain_summary = build_domain_summary(score_table, args.score_col, args.target_col)
    fold_summary = build_fold_summary(score_table, args.score_col, args.target_col)
    domain_fold_summary = build_domain_fold_summary(score_table, args.score_col, args.target_col)
    candidates = generate_subset_candidates(
        score_table,
        args.score_col,
        args.target_col,
        family_map,
        min_domains=args.min_domains,
        max_domains=args.max_domains,
        year_cutoffs=year_cutoffs,
    )
    domain_summary.to_csv(args.out_dir / "fig3_domain_oof_diagnostics.csv", index=False)
    fold_summary.to_csv(args.out_dir / "fig3_fold_oof_diagnostics.csv", index=False)
    domain_fold_summary.to_csv(args.out_dir / "fig3_domain_fold_oof_diagnostics.csv", index=False)
    candidates.to_csv(args.out_dir / "fig3_subset_candidate_matrix.csv", index=False)
    candidates.head(25).to_csv(args.out_dir / "fig3_subset_candidate_top25.csv", index=False)
    manifest = {
        "fig3_run_dir": str(args.fig3_run_dir),
        "score_col": args.score_col,
        "target_col": args.target_col,
        "n_score_rows": int(len(score_table)),
        "n_domains": int(score_table["domain"].nunique()),
        "year_cutoffs": ["all" if item is None else int(item) for item in year_cutoffs],
        "best_candidate": candidates.head(1).to_dict(orient="records")[0] if not candidates.empty else {},
        "interpretation": "Posthoc subset screen from existing Fig. 3 score table; use only to prioritize full recomputation candidates.",
    }
    (args.out_dir / "fig3_subset_diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[fig3-subset] wrote {args.out_dir}")
    if not candidates.empty:
        best = candidates.iloc[0]
        print(
            f"[fig3-subset] best posthoc candidate: rho={best['learned_oof_spearman']:.3f}, "
            f"domains={best['n_domains']}, max_year={best['max_year'] or 'all'}"
        )


if __name__ == "__main__":
    main()
