from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.corpus import audit_corpus, audit_strict_views, make_views  # noqa: E402


DEFAULT_SOURCE_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v2_publication"
DEFAULT_TARGET_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v2_publication_v6a_locked_candidate"
DEFAULT_LOCKED_PROBE_DIR = PROJECT_ROOT / "outputs" / "fig3_v6a_independent_v3_strong11_no_magnetic_locked"
LOCKED_RUN_NAME = "moderate__RGPM_latent_future_percentile__publication_day_plus__linear"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def domain_series(frame: pd.DataFrame, column: str = "domain") -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("").astype(str)
    return pd.Series("", index=frame.index, dtype=str)


def slug_series(frame: pd.DataFrame) -> pd.Series:
    if "slug" in frame.columns:
        return frame["slug"].fillna("").astype(str)
    return domain_series(frame, "domain")


def load_locked_domains(locked_run_dir: Path) -> List[str]:
    oof_path = locked_run_dir / "fig3_v6a_oof_score_table.csv"
    oof = read_csv(oof_path)
    if oof.empty or "domain" not in oof.columns:
        raise ValueError(f"{oof_path} does not contain v6A OOF domain rows")
    return sorted(d for d in oof["domain"].fillna("").astype(str).unique() if d)


def moderate_cohort_counts(locked_probe_dir: Path) -> Dict[str, int]:
    augmented = read_csv(locked_probe_dir / "fig3_v6a_augmented_input.csv")
    if augmented.empty or "domain" not in augmented.columns or "cohort_moderate" not in augmented.columns:
        return {}
    mask = pd.to_numeric(augmented["cohort_moderate"], errors="coerce").fillna(0).astype(int).eq(1)
    counts = augmented.loc[mask, "domain"].fillna("").astype(str).value_counts()
    return {str(domain): int(count) for domain, count in counts.to_dict().items()}


def filter_topics(topics: pd.DataFrame, domains: set[str]) -> pd.DataFrame:
    if topics.empty:
        return topics.copy()
    if "domain" not in topics.columns:
        return topics.copy()
    return topics[topics["domain"].fillna("").astype(str).isin(domains)].copy()


def filter_topic_edges(topic_edges: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    if topic_edges.empty or topics.empty or "community" not in topics.columns:
        return topic_edges.copy()
    communities = set(pd.to_numeric(topics["community"], errors="coerce").dropna().astype(int))
    out = topic_edges.copy()
    source_col = "source_community" if "source_community" in out.columns else "source"
    target_col = "target_community" if "target_community" in out.columns else "target"
    if source_col not in out.columns or target_col not in out.columns:
        return out
    source = pd.to_numeric(out[source_col], errors="coerce")
    target = pd.to_numeric(out[target_col], errors="coerce")
    return out[source.isin(communities) & target.isin(communities)].copy()


def filter_publication_roster(source_corpus_dir: Path, target_corpus_dir: Path, domains: Sequence[str]) -> None:
    domain_set = set(domains)
    works = read_csv(source_corpus_dir / "works.csv")
    citations = read_csv(source_corpus_dir / "citations.csv")
    topics = read_csv(source_corpus_dir / "topics.csv")
    topic_edges = read_csv(source_corpus_dir / "topic_edges.csv")
    domain_table = read_csv(source_corpus_dir / "domains.csv")
    landmarks = read_csv(source_corpus_dir / "landmarks.csv")
    if works.empty or "domain" not in works.columns or "id" not in works.columns:
        raise ValueError(f"{source_corpus_dir / 'works.csv'} has no usable domain/id rows")

    works = works[domain_series(works).isin(domain_set)].copy()
    selected_ids = set(works["id"].fillna("").astype(str))
    citations = citations[citations.get("source", pd.Series("", index=citations.index)).fillna("").astype(str).isin(selected_ids)].copy()
    topics = filter_topics(topics, domain_set)
    topic_edges = filter_topic_edges(topic_edges, topics)
    if not domain_table.empty:
        domain_table = domain_table[slug_series(domain_table).isin(domain_set)].copy()
    if not landmarks.empty:
        landmarks = landmarks[domain_series(landmarks).isin(domain_set)].copy()

    target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    works.to_csv(target_corpus_dir / "works.csv", index=False)
    citations.to_csv(target_corpus_dir / "citations.csv", index=False)
    topics.to_csv(target_corpus_dir / "topics.csv", index=False)
    topic_edges.to_csv(target_corpus_dir / "topic_edges.csv", index=False)
    domain_table.to_csv(target_corpus_dir / "domains.csv", index=False)
    landmarks.to_csv(target_corpus_dir / "landmarks.csv", index=False)


def filtered_publication_targets(source_corpus_dir: Path, target_corpus_dir: Path, domains: Sequence[str]) -> Dict[str, Any]:
    source_path = source_corpus_dir / "publication_target_domains.json"
    source = read_json(source_path)
    domain_set = set(domains)
    rows = [
        dict(row)
        for row in source.get("domains", [])
        if str(row.get("domain_id", row.get("domain", ""))) in domain_set
    ]
    payload = {
        "artifact_kind": "publication_target_domain_roster_v6a_locked_candidate",
        "created_at": utc_now(),
        "source_roster": str(source_path),
        "domains": rows,
        "n_domains": len(rows),
        "policy": "v6A moderate reliability cohort, min 20 OOF rows per domain",
    }
    write_json(target_corpus_dir / "publication_target_domains.json", payload)
    return payload


def build_gate_decision(
    *,
    source_corpus_dir: Path,
    target_corpus_dir: Path,
    locked_probe_dir: Path,
    locked_run_dir: Path,
    validation_probe_dirs: Sequence[Path],
    domains: Sequence[str],
    quality_report: Mapping[str, Any],
    strict_view_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    source_domains = set(domain_series(read_csv(source_corpus_dir / "domains.csv"), "slug").replace("", pd.NA).dropna())
    kept = set(domains)
    moderate_counts = moderate_cohort_counts(locked_probe_dir)
    excluded = []
    for domain in sorted(source_domains - kept):
        excluded.append(
            {
                "domain": domain,
                "moderate_cohort_rows": int(moderate_counts.get(domain, 0)),
                "reason": "v6a_moderate_reliability_rows_below_domain_floor",
            }
        )
    validations = []
    for probe_dir in validation_probe_dirs:
        decision_path = probe_dir / "fig3_v6a_probe_decision.json"
        decision = read_json(decision_path)
        best = decision.get("best_run", {})
        validations.append(
            {
                "probe_dir": str(probe_dir),
                "final_pass": bool(decision.get("final_pass", False)),
                "learned_oof_spearman": best.get("learned_oof_spearman"),
                "latest_fold_test_spearman": best.get("latest_fold_test_spearman"),
                "learned_vs_equal_delta": best.get("learned_vs_equal_delta"),
                "n_contributing_graph_deltas": best.get("n_contributing_graph_deltas"),
                "n_rows": best.get("n_rows"),
                "n_domains": best.get("n_domains"),
                "min_rows_per_domain": best.get("min_rows_per_domain"),
                "max_domain_share": best.get("max_domain_share"),
                "top_bottom_enrichment": best.get("top_bottom_enrichment"),
                "top_bottom_enrichment_finite_max": best.get("top_bottom_enrichment_finite_max"),
            }
        )

    main_decision = read_json(locked_probe_dir / "fig3_v6a_probe_decision.json")
    checks = {
        "main_locked_probe_pass": bool(main_decision.get("final_pass", False)),
        "all_validation_probes_pass": bool(validations) and all(bool(row.get("final_pass")) for row in validations),
        "quality_audit_pass": bool(quality_report.get("overall_pass", False)),
        "strict_view_audit_pass": bool(strict_view_audit.get("overall_pass", False)),
        "n_domains_8_to_12": 8 <= len(domains) <= 12,
        "excluded_domains_have_documented_reason": all(bool(row.get("reason")) for row in excluded),
    }
    payload = {
        "created_at": utc_now(),
        "artifact_kind": "performance_gate_decision_v6a_locked",
        "policy": "locked v6A main result: moderate reliability cohort, latent future percentile target, publication-day-plus linear features",
        "source_corpus_dir": str(source_corpus_dir),
        "target_corpus_dir": str(target_corpus_dir),
        "locked_probe_dir": str(locked_probe_dir),
        "locked_run_dir": str(locked_run_dir),
        "main_domains": list(domains),
        "n_main_domains": len(domains),
        "excluded_domains": excluded,
        "main_probe": main_decision,
        "independent_validations": validations,
        "quality_report_path": str(target_corpus_dir / "quality_report.json"),
        "strict_view_audit_path": str(target_corpus_dir / "strict_view_audit.json"),
        "checks": checks,
        "final_pass": bool(all(checks.values())),
    }
    write_json(target_corpus_dir / "performance_gate_decision_v6a.json", payload)
    return payload


def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    locked_probe_dir = args.locked_probe_dir.resolve()
    locked_run_dir = args.locked_run_dir or locked_probe_dir / "runs" / LOCKED_RUN_NAME
    locked_run_dir = locked_run_dir.resolve()
    domains = load_locked_domains(locked_run_dir)
    filter_publication_roster(args.source_corpus_dir, args.target_corpus_dir, domains)
    quality_report = audit_corpus(args.target_corpus_dir, min_papers_per_domain=args.min_papers_per_domain)
    make_views(args.target_corpus_dir, anchor_policy="strict")
    strict_view_audit = audit_strict_views(args.target_corpus_dir)
    filtered_publication_targets(args.source_corpus_dir, args.target_corpus_dir, domains)
    source_manifest = read_json(args.source_corpus_dir / "manifest.json")
    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_publication_v6a_locked_candidate_corpus",
        "source_corpus_dir": str(args.source_corpus_dir),
        "target_corpus_dir": str(args.target_corpus_dir),
        "figure_contract": "Root tables and views remain compatible with existing Fig1-Fig5 readers.",
        "figure_logic_policy": "no Fig1-Fig5 plotting logic changes",
        "anchor_policy": "strict",
        "selection_policy": "locked v6A moderate reliability cohort, min 20 rows per domain",
        "source_manifest": source_manifest,
        "domains": domains,
        "n_domains": int(len(domains)),
        "n_works": int(len(read_csv(args.target_corpus_dir / "works.csv"))),
        "n_citations_source_refs": int(len(read_csv(args.target_corpus_dir / "citations.csv"))),
        "n_landmarks": int(len(read_csv(args.target_corpus_dir / "landmarks.csv"))),
        "quality_overall_pass": bool(quality_report.get("overall_pass", False)),
        "strict_view_overall_pass": bool(strict_view_audit.get("overall_pass", False)),
        "min_papers_per_domain": int(args.min_papers_per_domain),
    }
    write_json(args.target_corpus_dir / "manifest.json", manifest)
    return build_gate_decision(
        source_corpus_dir=args.source_corpus_dir,
        target_corpus_dir=args.target_corpus_dir,
        locked_probe_dir=locked_probe_dir,
        locked_run_dir=locked_run_dir,
        validation_probe_dirs=args.validation_probe_dirs,
        domains=domains,
        quality_report=quality_report,
        strict_view_audit=strict_view_audit,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize the locked Fig3 v6A publication candidate corpus.")
    parser.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_SOURCE_CORPUS_DIR)
    parser.add_argument("--target-corpus-dir", type=Path, default=DEFAULT_TARGET_CORPUS_DIR)
    parser.add_argument("--locked-probe-dir", type=Path, default=DEFAULT_LOCKED_PROBE_DIR)
    parser.add_argument("--locked-run-dir", type=Path)
    parser.add_argument("--validation-probe-dirs", type=Path, nargs="+", default=[])
    parser.add_argument("--min-papers-per-domain", type=int, default=2500)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    decision = materialize(parse_args(argv))
    status = "PASS" if decision.get("final_pass") else "FAIL"
    print(f"[v6a-materialize] {status}; target={decision.get('target_corpus_dir')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
