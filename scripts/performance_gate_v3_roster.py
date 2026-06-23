from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.publication_corpus_v2 import (  # noqa: E402
    FIGURE_LOGIC_POLICY,
    clean_landmark_registry,
    nonempty,
    read_csv,
    slugify,
    write_json,
)


DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v3_openalex_graph"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "performance_gated_roster_v3"
DEFAULT_ROSTER_PATH = PROJECT_ROOT / "data" / "knowledge_corpus" / "publication_target_domains_v3_performance_gated.json"
DEFAULT_CANDIDATE_DOMAIN = "magnetic_properties_of_thin_films"


def utc_now() -> str:
    """Return a UTC timestamp for reproducible manifests."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_domains(run_dir: Path, corpus_dir: Path) -> List[str]:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        payload = _load_json(manifest)
        domains = payload.get("domains")
        if isinstance(domains, list) and domains:
            return [slugify(domain) for domain in domains]

    selection = run_dir.parent / "fig3_run_selection.json"
    if selection.exists():
        payload = _load_json(selection)
        summary = payload.get("summary", {})
        domains = summary.get("data_profile", {}).get("domains")
        if isinstance(domains, list) and domains:
            return [slugify(domain) for domain in domains]

    score_path = run_dir / "fig3_oof_score_table.csv"
    if not score_path.exists():
        score_path = run_dir / "fig3_score_table.csv"
    if score_path.exists():
        score = pd.read_csv(score_path, usecols=lambda col: col == "domain", low_memory=False)
        if "domain" in score.columns:
            return sorted(score["domain"].dropna().astype(str).map(slugify).unique().tolist())

    domains = read_csv(corpus_dir / "domains.csv")
    if domains.empty:
        return []
    col = "slug" if "slug" in domains.columns else "domain"
    return sorted(domains[col].dropna().astype(str).map(slugify).unique().tolist())


def extract_run_metrics(run_dir: Path, corpus_dir: Path) -> Dict[str, Any]:
    """Extract the three performance-gate metrics from one Fig. 3 run."""
    summary = _load_json(run_dir / "fig3_diagnostics_summary.json")
    cv = pd.read_csv(run_dir / "fig3_cv_summary.csv")
    if cv.empty or "fold" not in cv.columns or "test_spearman" not in cv.columns:
        raise ValueError(f"{run_dir / 'fig3_cv_summary.csv'} is missing fold/test_spearman")
    cv = cv.copy()
    cv["fold"] = pd.to_numeric(cv["fold"], errors="coerce")
    latest = cv.sort_values("fold").tail(1).iloc[0]
    contributing = summary.get("contributing_graph_deltas", [])
    if not isinstance(contributing, list):
        contributing = []
    domains = _run_domains(run_dir, corpus_dir)
    if not domains:
        data_profile = summary.get("data_profile", {})
        profile_domains = data_profile.get("domains")
        if isinstance(profile_domains, list):
            domains = [slugify(domain) for domain in profile_domains]
    learned = float(summary.get("learned_oof_spearman", float("nan")))
    equal = float(summary.get("equal_weight_oof_spearman", float("nan")))
    latest_test = float(latest["test_spearman"])
    return {
        "run_dir": str(run_dir),
        "domains": sorted(dict.fromkeys(domains)),
        "n_domains": int(len(set(domains))),
        "learned_oof_spearman": learned,
        "equal_weight_oof_spearman": equal,
        "learned_vs_equal_delta": learned - equal,
        "latest_fold": int(latest["fold"]),
        "latest_fold_test_spearman": latest_test,
        "n_contributing_graph_deltas": int(summary.get("n_contributing_graph_deltas", len(contributing))),
        "contributing_graph_deltas": contributing,
        "overall_pass": bool(summary.get("overall_pass", False)),
        "status_label": nonempty(summary.get("status_label")),
    }


def strictly_improves(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, bool]:
    """Return per-metric strict improvement checks."""
    return {
        "improves_oof": float(candidate["learned_oof_spearman"]) > float(baseline["learned_oof_spearman"]),
        "improves_latest_fold": float(candidate["latest_fold_test_spearman"])
        > float(baseline["latest_fold_test_spearman"]),
        "improves_contributing_deltas": int(candidate["n_contributing_graph_deltas"])
        > int(baseline["n_contributing_graph_deltas"]),
    }


def _family_from_field(field_name: str) -> str:
    text = field_name.lower()
    if any(token in text for token in ["biology", "medicine", "health"]):
        return "biology_biomedicine"
    if any(token in text for token in ["material", "chemistry", "chemical"]):
        return "materials_chemistry"
    if any(token in text for token in ["physics", "astronomy", "earth", "planet"]):
        return "physics_astronomy"
    if "computer" in text:
        return "computer_science"
    return "other_science"


def build_domain_rows(corpus_dir: Path, final_domains: Sequence[str]) -> pd.DataFrame:
    domains = read_csv(corpus_dir / "domains.csv")
    final_set = {slugify(domain) for domain in final_domains}
    if domains.empty:
        return pd.DataFrame({"domain": sorted(final_set)})
    out = domains.copy()
    domain_col = "slug" if "slug" in out.columns else "domain"
    out["domain"] = out[domain_col].astype(str).map(slugify)
    out = out[out["domain"].isin(final_set)].copy()
    out["family"] = out.get("field_name", pd.Series("", index=out.index)).astype(str).map(_family_from_field)
    out["publication_score"] = 1.0
    out["recommended_role"] = "main_candidate"
    out["status"] = "main_ready"
    out["reason_for_inclusion_or_exclusion"] = "performance_gated_v3"
    missing = sorted(final_set - set(out["domain"].astype(str)))
    if missing:
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    {
                        "domain": missing,
                        "display_name": missing,
                        "field_name": "",
                        "family": "other_science",
                        "publication_score": 1.0,
                        "recommended_role": "main_candidate",
                        "status": "main_ready",
                        "reason_for_inclusion_or_exclusion": "performance_gated_v3",
                    }
                ),
            ],
            ignore_index=True,
        )
    return out.sort_values("domain").reset_index(drop=True)


def write_seed_outputs(
    corpus_dir: Path,
    out_dir: Path,
    final_domains: Sequence[str],
    decision: Mapping[str, Any],
    target_roster_path: Optional[Path],
) -> None:
    """Write seed files consumed by publication_corpus_v2.materialize."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final_domains = sorted(dict.fromkeys(slugify(domain) for domain in final_domains))
    domain_rows = build_domain_rows(corpus_dir, final_domains)
    domain_rows.to_csv(out_dir / "candidate_domains.csv", index=False)
    domain_rows.to_csv(out_dir / "domain_inclusion_table.csv", index=False)
    domain_rows.to_csv(out_dir / "domain_status.csv", index=False)

    landmarks = clean_landmark_registry(read_csv(corpus_dir / "landmarks.csv"), max_landmarks_per_domain=3)
    landmarks = landmarks[landmarks["domain"].astype(str).isin(final_domains)].copy()
    landmarks.to_csv(out_dir / "landmark_registry_v2_seed.csv", index=False)

    event_years = (
        pd.to_numeric(landmarks.get("year", pd.Series(dtype=float)), errors="coerce")
        .groupby(landmarks.get("domain", pd.Series(dtype=str)).astype(str))
        .min()
        .dropna()
        .astype(int)
        .to_dict()
        if not landmarks.empty
        else {}
    )
    family_counts: Dict[str, int] = {}
    roster_domains: List[Dict[str, Any]] = []
    for row in domain_rows.to_dict("records"):
        domain = slugify(row.get("domain"))
        family = nonempty(row.get("family")) or "other_science"
        family_counts[family] = family_counts.get(family, 0) + 1
        roster_domains.append(
            {
                "domain_id": domain,
                "family": family,
                "status": "main_ready",
                "event_year": int(event_years[domain]) if domain in event_years else "",
                "analysis_end_year": 2025,
                "landmark_policy": "strict_manual_v3",
                "notes": "performance_gated_v3",
            }
        )
    roster = {
        "created_at": utc_now(),
        "artifact_kind": "publication_target_domain_roster",
        "figure_logic_policy": FIGURE_LOGIC_POLICY,
        "selection_policy": "strict_performance_gate_v3",
        "n_domains": int(len(roster_domains)),
        "family_counts": family_counts,
        "domains": roster_domains,
    }
    write_json(out_dir / "publication_target_domains.json", roster)
    if target_roster_path is not None:
        write_json(target_roster_path, roster)
    write_json(
        out_dir / "v2_publication_seed_manifest.json",
        {
            "created_at": utc_now(),
            "artifact_kind": "performance_gated_v3_seed",
            "source_corpus_dir": str(corpus_dir),
            "figure_logic_policy": FIGURE_LOGIC_POLICY,
            "selection_policy": "strict_performance_gate_v3",
            "decision": dict(decision),
            "final_domains": final_domains,
            "n_final_domains": int(len(final_domains)),
            "n_seed_landmarks": int(len(landmarks)),
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Select a v3 publication roster using strict Fig. 3 performance gates.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-domain", default=DEFAULT_CANDIDATE_DOMAIN)
    parser.add_argument("--target-roster-path", type=Path, default=DEFAULT_ROSTER_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    baseline = extract_run_metrics(args.baseline_run_dir, args.corpus_dir)
    candidate = extract_run_metrics(args.candidate_run_dir, args.corpus_dir)
    candidate_domain = slugify(args.candidate_domain)
    checks = strictly_improves(candidate, baseline)
    include_candidate = bool(all(checks.values()) and candidate_domain in set(candidate["domains"]))
    final_domains = candidate["domains"] if include_candidate else baseline["domains"]

    decision = {
        "created_at": utc_now(),
        "candidate_domain": candidate_domain,
        "include_candidate": include_candidate,
        "gate_policy": "include only if OOF, latest time-block fold, and contributing deltas strictly improve",
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "final_domains": final_domains,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "performance_gate_decision.json", decision)
    pd.DataFrame(
        [
            {"run": "baseline", **{k: v for k, v in baseline.items() if k not in {"domains", "contributing_graph_deltas"}}},
            {"run": "candidate", **{k: v for k, v in candidate.items() if k not in {"domains", "contributing_graph_deltas"}}},
        ]
    ).to_csv(args.out_dir / "performance_gate_summary.csv", index=False)
    pd.DataFrame(
        [
            {"metric": key, "passed": int(value)}
            for key, value in checks.items()
        ]
    ).to_csv(args.out_dir / "performance_gate_checks.csv", index=False)
    write_seed_outputs(args.corpus_dir, args.out_dir, final_domains, decision, args.target_roster_path)

    if not args.quiet:
        outcome = "included" if include_candidate else "excluded"
        print(f"[performance-gate-v3] {candidate_domain} {outcome}; final domains={len(final_domains)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
