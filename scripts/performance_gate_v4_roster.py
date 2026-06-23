from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.performance_gate_v3_roster import extract_run_metrics  # noqa: E402
from scripts.publication_corpus_v2 import (  # noqa: E402
    FIGURE_LOGIC_POLICY,
    clean_landmark_registry,
    nonempty,
    read_csv,
    slugify,
    write_json,
)


DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v4_screen_graph"
DEFAULT_REGISTRY_CSV = PROJECT_ROOT / "data" / "knowledge_corpus" / "landmark_registry_v4.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "performance_gated_roster_v4"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _family_from_field(field_name: object) -> str:
    text = str(field_name or "").lower()
    if any(token in text for token in ["biology", "medicine", "health", "genomics", "life"]):
        return "biology_biomedicine"
    if any(token in text for token in ["material", "chemistry", "chemical", "energy"]):
        return "materials_chemistry"
    if any(token in text for token in ["physics", "astronomy", "planet", "earth"]):
        return "physics_astronomy"
    if any(token in text for token in ["computer", "method", "statistics", "mathematics"]):
        return "methods_computing"
    return "other_science"


def _domain_column(frame: pd.DataFrame) -> str:
    if "slug" in frame.columns:
        return "slug"
    if "domain" in frame.columns:
        return "domain"
    return ""


def text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("").astype(str)
    return pd.Series("", index=frame.index, dtype=str)


def corpus_domain_quality(
    corpus_dir: Path,
    registry_csv: Path,
    *,
    min_screen_works: int,
    min_local_closure: float,
    max_main_year: int,
) -> pd.DataFrame:
    works = read_csv(corpus_dir / "works.csv")
    citations = read_csv(corpus_dir / "citations.csv")
    domains = read_csv(corpus_dir / "domains.csv")
    landmarks = read_csv(corpus_dir / "landmarks.csv")
    registry = read_csv(registry_csv)
    rows: List[Dict[str, Any]] = []
    if works.empty:
        return pd.DataFrame(columns=["domain", "quality_pass", "quality_score", "failure_reasons"])

    works = works.copy()
    works["domain"] = text_column(works, "domain").map(slugify)
    works["id"] = text_column(works, "id")
    works["year"] = pd.to_numeric(works.get("year", 0), errors="coerce")
    work_domain = works.set_index("id")["domain"].to_dict()
    selected_ids = set(works["id"].astype(str))

    citation_stats: Dict[str, Dict[str, float]] = {}
    if not citations.empty and {"source", "target"}.issubset(citations.columns):
        cit = citations.copy()
        cit["source_domain"] = cit["source"].astype(str).map(work_domain)
        cit = cit[cit["source_domain"].astype(str) != ""].copy()
        cit["local_target"] = cit["target"].astype(str).isin(selected_ids).astype(int)
        grouped = cit.groupby("source_domain", sort=False).agg(
            n_citation_rows=("source", "count"),
            local_reference_closure=("local_target", "mean"),
        )
        citation_stats = grouped.to_dict("index")

    domain_meta: Dict[str, Dict[str, Any]] = {}
    if not domains.empty:
        domain_col = _domain_column(domains)
        for row in domains.to_dict("records"):
            domain = slugify(row.get(domain_col))
            if domain:
                domain_meta[domain] = dict(row)

    lm = landmarks if not landmarks.empty else registry
    if lm.empty:
        lm = pd.DataFrame(columns=["domain", "year", "doi", "landmark_source", "evidence_type"])
    lm = lm.copy()
    lm["domain"] = text_column(lm, "domain").map(slugify)
    if "include_main" in lm.columns:
        lm = lm[pd.to_numeric(lm["include_main"], errors="coerce").fillna(1).astype(int) != 0].copy()
    lm["year"] = pd.to_numeric(lm.get("year", 0), errors="coerce")
    lm["doi_present"] = text_column(lm, "doi").str.strip().ne("")
    source_text = (
        text_column(lm, "landmark_source")
        + " "
        + text_column(lm, "evidence_type")
        + " "
        + text_column(lm, "source_registry")
    ).str.lower()
    lm["legacy_main_evidence"] = source_text.str.contains("fig1_anchor|legacy", regex=True, na=False)
    lm_stats = lm.groupby("domain", sort=False).agg(
        n_landmarks=("doi_present", "sum"),
        first_landmark_year=("year", "min"),
        has_legacy_main_evidence=("legacy_main_evidence", "max"),
    )

    for domain, group in works.groupby("domain", sort=True):
        meta = domain_meta.get(domain, {})
        lm_row = lm_stats.loc[domain].to_dict() if domain in lm_stats.index else {}
        cit_row = citation_stats.get(domain, {})
        n_works = int(len(group))
        n_landmarks = safe_int(lm_row.get("n_landmarks"))
        first_year = safe_int(lm_row.get("first_landmark_year"), default=9999)
        closure = safe_float(cit_row.get("local_reference_closure"), default=0.0)
        reasons: List[str] = []
        if n_works < int(min_screen_works):
            reasons.append("too_few_screen_works")
        if n_landmarks < 1 or n_landmarks > 3:
            reasons.append("landmark_count_out_of_range")
        if first_year > int(max_main_year):
            reasons.append("landmark_after_cutoff")
        if bool(lm_row.get("has_legacy_main_evidence", False)):
            reasons.append("legacy_main_evidence")
        if closure < float(min_local_closure):
            reasons.append("low_local_reference_closure")
        score = (
            min(1.0, n_works / max(1.0, float(min_screen_works)))
            + min(1.0, closure / max(0.001, float(min_local_closure)))
            + min(1.0, n_landmarks / 3.0)
        ) / 3.0
        rows.append(
            {
                "domain": domain,
                "display_name": nonempty(meta.get("display_name")) or domain.replace("_", " "),
                "field_name": nonempty(meta.get("field_name")),
                "family": _family_from_field(meta.get("field_name")),
                "n_works": n_works,
                "n_landmarks": n_landmarks,
                "first_landmark_year": first_year if first_year != 9999 else "",
                "n_citation_rows": safe_int(cit_row.get("n_citation_rows")),
                "local_reference_closure": closure,
                "quality_score": score,
                "quality_pass": int(not reasons),
                "failure_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows).sort_values(["quality_pass", "quality_score", "domain"], ascending=[False, False, True])


def extract_effect_enrichment(run_dir: Path) -> float:
    payload = load_json_optional(run_dir / "fig3_effect_summary.json")
    if not payload:
        return 0.0
    top10 = safe_float(payload.get("top_vs_bottom_score_decile_rgpm_top10_enrichment"))
    top20 = safe_float(payload.get("top_vs_bottom_score_decile_rgpm_top20_enrichment"))
    return max(top10, top20)


def final_family_share(domains: Sequence[str], domain_screen: pd.DataFrame) -> Dict[str, Any]:
    final_set = {slugify(domain) for domain in domains}
    if domain_screen.empty or not final_set:
        return {"family_counts": {}, "max_family_share": 1.0 if final_set else 0.0}
    frame = domain_screen[domain_screen["domain"].astype(str).isin(final_set)].copy()
    counts = frame["family"].fillna("other_science").astype(str).value_counts().to_dict()
    missing = len(final_set) - int(sum(counts.values()))
    if missing > 0:
        counts["other_science"] = int(counts.get("other_science", 0)) + missing
    max_share = max(counts.values()) / max(1, len(final_set)) if counts else 0.0
    return {"family_counts": {str(k): int(v) for k, v in counts.items()}, "max_family_share": float(max_share)}


def evaluate_final_gate(
    final_run_dir: Path,
    corpus_dir: Path,
    domain_screen: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metrics = extract_run_metrics(final_run_dir, corpus_dir)
    enrichment = extract_effect_enrichment(final_run_dir)
    balance = final_family_share(metrics.get("domains", []), domain_screen)
    checks = {
        "learned_oof_spearman": safe_float(metrics.get("learned_oof_spearman")) >= float(args.min_oof),
        "learned_vs_equal_delta": safe_float(metrics.get("learned_vs_equal_delta")) >= float(args.min_learned_vs_equal),
        "learned_beats_equal": safe_float(metrics.get("learned_oof_spearman"))
        > safe_float(metrics.get("equal_weight_oof_spearman")),
        "latest_fold": safe_float(metrics.get("latest_fold_test_spearman")) >= float(args.min_latest_fold),
        "contributing_graph_deltas": safe_int(metrics.get("n_contributing_graph_deltas")) >= int(args.min_contributing_deltas),
        "top_bottom_enrichment": enrichment >= float(args.min_enrichment),
        "min_domains": safe_int(metrics.get("n_domains")) >= int(args.min_domains),
        "max_domains": safe_int(metrics.get("n_domains")) <= int(args.max_domains),
        "family_balance": safe_float(balance.get("max_family_share"), default=1.0) <= float(args.max_family_share),
    }
    if args.baseline_run_dir:
        baseline = extract_run_metrics(args.baseline_run_dir, corpus_dir)
        checks["learned_beats_baseline"] = safe_float(metrics.get("learned_oof_spearman")) > safe_float(
            baseline.get("learned_oof_spearman")
        )
    else:
        baseline = {}
    return {
        "run_dir": str(final_run_dir),
        "metrics": metrics,
        "baseline": baseline,
        "top_bottom_enrichment": enrichment,
        "family_counts": balance["family_counts"],
        "max_family_share": balance["max_family_share"],
        "latest_fold_ideal_pass": safe_float(metrics.get("latest_fold_test_spearman")) >= float(args.ideal_latest_fold),
        "checks": checks,
        "final_pass": bool(all(checks.values())),
    }


def build_publication_roster(final_domains: Sequence[str], domain_screen: pd.DataFrame, registry_csv: Path) -> Dict[str, Any]:
    final = sorted(dict.fromkeys(slugify(domain) for domain in final_domains))
    registry = clean_landmark_registry(read_csv(registry_csv), max_landmarks_per_domain=3)
    event_years = (
        pd.to_numeric(registry.get("year", pd.Series(dtype=float)), errors="coerce")
        .groupby(registry.get("domain", pd.Series(dtype=str)).astype(str).map(slugify))
        .min()
        .dropna()
        .astype(int)
        .to_dict()
        if not registry.empty
        else {}
    )
    screen = domain_screen.set_index("domain").to_dict("index") if not domain_screen.empty else {}
    family_counts: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for domain in final:
        family = nonempty((screen.get(domain) or {}).get("family")) or "other_science"
        family_counts[family] = family_counts.get(family, 0) + 1
        rows.append(
            {
                "domain_id": domain,
                "family": family,
                "status": "main_ready",
                "event_year": int(event_years[domain]) if domain in event_years else "",
                "analysis_end_year": 2025,
                "landmark_policy": "strict_manual_v4",
                "notes": "performance_gated_roster_v4_final_pass",
            }
        )
    return {
        "created_at": utc_now(),
        "artifact_kind": "publication_target_domain_roster_v4",
        "figure_logic_policy": FIGURE_LOGIC_POLICY,
        "selection_policy": "performance_gated_roster_v4",
        "n_domains": int(len(rows)),
        "family_counts": family_counts,
        "domains": rows,
    }


def write_outputs(
    domain_screen: pd.DataFrame,
    final_eval: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    quality_pass = domain_screen[domain_screen["quality_pass"].astype(int) == 1].copy() if not domain_screen.empty else pd.DataFrame()
    candidate = quality_pass.head(int(args.top_screen_domains)).copy()
    candidate["recommended_role"] = "final_recompute_candidate"
    selected_domains = candidate["domain"].astype(str).tolist()

    final_domains = final_eval.get("metrics", {}).get("domains", []) if final_eval.get("final_pass") else []
    if final_domains:
        candidate = domain_screen[domain_screen["domain"].astype(str).isin(set(final_domains))].copy()
        candidate["recommended_role"] = "main_candidate"

    supplemental = domain_screen[~domain_screen["domain"].astype(str).isin(set(candidate.get("domain", [])))].copy()
    if not supplemental.empty:
        supplemental["recommended_role"] = "supplemental_or_failed_screen"
        supplemental["exclusion_reason"] = supplemental["failure_reasons"].where(
            supplemental["failure_reasons"].astype(str) != "",
            "below_locked_screen_rank_or_pending_final_gate",
        )

    candidate.to_csv(args.out_dir / "candidate_domains.csv", index=False)
    supplemental.to_csv(args.out_dir / "supplemental_domains.csv", index=False)
    domain_screen.to_csv(args.out_dir / "domain_screen.csv", index=False)

    decision = {
        "created_at": utc_now(),
        "artifact_kind": "performance_gate_decision_v4",
        "figure_logic_policy": FIGURE_LOGIC_POLICY,
        "selection_policy": "large_candidate_pool_strict_performance_gate_v4",
        "thresholds": {
            "min_oof": float(args.min_oof),
            "min_learned_vs_equal": float(args.min_learned_vs_equal),
            "min_latest_fold": float(args.min_latest_fold),
            "ideal_latest_fold": float(args.ideal_latest_fold),
            "min_contributing_deltas": int(args.min_contributing_deltas),
            "min_enrichment": float(args.min_enrichment),
            "max_family_share": float(args.max_family_share),
            "min_domains": int(args.min_domains),
            "max_domains": int(args.max_domains),
            "top_screen_domains": int(args.top_screen_domains),
        },
        "screen": {
            "n_screen_domains": int(len(domain_screen)),
            "n_quality_pass": int(len(quality_pass)),
            "candidate_domains": selected_domains,
        },
        "final_evaluation": final_eval,
        "final_pass": bool(final_eval.get("final_pass", False)),
        "final_domains": final_domains,
        "materialization_status": "eligible_for_v2_publication" if final_eval.get("final_pass") else "blocked_pending_or_failed_final_gate",
    }
    write_json(args.out_dir / "performance_gate_decision_v4.json", decision)
    if final_eval.get("final_pass"):
        roster = build_publication_roster(final_domains, domain_screen, args.registry_csv)
        write_json(args.out_dir / "publication_target_domains.json", roster)
        if args.target_roster_path:
            write_json(args.target_roster_path, roster)
    return decision


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a performance-gated v4 publication roster decision.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--registry-csv", type=Path, default=DEFAULT_REGISTRY_CSV)
    parser.add_argument("--final-run-dir", type=Path, default=None)
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-roster-path", type=Path, default=None)
    parser.add_argument("--min-screen-works", type=int, default=800)
    parser.add_argument("--min-local-closure", type=float, default=0.02)
    parser.add_argument("--max-main-year", type=int, default=2015)
    parser.add_argument("--top-screen-domains", type=int, default=25)
    parser.add_argument("--min-oof", type=float, default=0.45)
    parser.add_argument("--min-learned-vs-equal", type=float, default=0.03)
    parser.add_argument("--min-latest-fold", type=float, default=0.35)
    parser.add_argument("--ideal-latest-fold", type=float, default=0.40)
    parser.add_argument("--min-contributing-deltas", type=int, default=5)
    parser.add_argument("--min-enrichment", type=float, default=5.0)
    parser.add_argument("--max-family-share", type=float, default=0.50)
    parser.add_argument("--min-domains", type=int, default=8)
    parser.add_argument("--max-domains", type=int, default=12)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    domain_screen = corpus_domain_quality(
        args.corpus_dir,
        args.registry_csv,
        min_screen_works=args.min_screen_works,
        min_local_closure=args.min_local_closure,
        max_main_year=args.max_main_year,
    )
    final_eval: Dict[str, Any] = {
        "final_pass": False,
        "status": "pending_final_fig3_recompute" if args.final_run_dir is None else "missing_final_run_outputs",
    }
    if args.final_run_dir is not None and (args.final_run_dir / "fig3_diagnostics_summary.json").exists():
        final_eval = evaluate_final_gate(args.final_run_dir, args.corpus_dir, domain_screen, args)
    decision = write_outputs(domain_screen, final_eval, args)
    if not args.quiet:
        status = "PASS" if decision["final_pass"] else "PENDING/FAIL"
        print(f"[performance-gate-v4] {status}; screen_domains={decision['screen']['n_screen_domains']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
