from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def effect_enrichment(effect: Mapping[str, Any]) -> float:
    return max(
        safe_float(effect.get("top_vs_bottom_score_decile_rgpm_top10_enrichment")),
        safe_float(effect.get("top_vs_bottom_score_decile_rgpm_top20_enrichment")),
    )


def build_decision(
    v5_run_dir: Path,
    baseline_run_dir: Path,
    *,
    min_oof: float,
    min_latest_fold: float,
    min_learned_vs_equal: float,
    min_contributing_deltas: int,
    min_enrichment: float,
) -> Dict[str, Any]:
    v5 = read_json(v5_run_dir / "fig3_v5_diagnostics_summary.json")
    effect = read_json(v5_run_dir / "fig3_v5_effect_summary.json")
    baseline = read_json(baseline_run_dir / "fig3_diagnostics_summary.json")
    enrichment = effect_enrichment(effect)
    checks = {
        "learned_oof_spearman": safe_float(v5.get("learned_oof_spearman")) >= float(min_oof),
        "learned_vs_equal_delta": safe_float(v5.get("learned_vs_equal_delta")) >= float(min_learned_vs_equal),
        "learned_beats_equal": safe_float(v5.get("learned_oof_spearman"))
        > safe_float(v5.get("equal_weight_oof_spearman")),
        "latest_fold": safe_float(v5.get("latest_fold_test_spearman")) >= float(min_latest_fold),
        "contributing_graph_deltas": safe_int(v5.get("n_contributing_graph_deltas")) >= int(min_contributing_deltas),
        "top_bottom_enrichment": enrichment >= float(min_enrichment),
        "beats_baseline": safe_float(v5.get("learned_oof_spearman"))
        > safe_float(baseline.get("learned_oof_spearman")),
    }
    return {
        "created_at": utc_now(),
        "artifact_kind": "performance_gate_decision_v5_method",
        "selection_policy": "nested_time_block_rank_learning_must_beat_fig3_baseline",
        "thresholds": {
            "min_oof": float(min_oof),
            "min_latest_fold": float(min_latest_fold),
            "min_learned_vs_equal": float(min_learned_vs_equal),
            "min_contributing_deltas": int(min_contributing_deltas),
            "min_enrichment": float(min_enrichment),
        },
        "checks": checks,
        "final_pass": bool(all(checks.values())),
        "v5_run_dir": str(v5_run_dir),
        "baseline_run_dir": str(baseline_run_dir),
        "v5_metrics": {
            "learned_oof_spearman": safe_float(v5.get("learned_oof_spearman")),
            "equal_weight_oof_spearman": safe_float(v5.get("equal_weight_oof_spearman")),
            "learned_vs_equal_delta": safe_float(v5.get("learned_vs_equal_delta")),
            "latest_fold_test_spearman": safe_float(v5.get("latest_fold_test_spearman")),
            "n_contributing_graph_deltas": safe_int(v5.get("n_contributing_graph_deltas")),
            "top_bottom_enrichment": enrichment,
        },
        "baseline_metrics": {
            "learned_oof_spearman": safe_float(baseline.get("learned_oof_spearman")),
            "equal_weight_oof_spearman": safe_float(baseline.get("equal_weight_oof_spearman")),
            "learned_vs_equal_delta": safe_float(baseline.get("learned_vs_equal_delta")),
        },
        "materialization_status": "eligible_for_v2_publication" if all(checks.values()) else "blocked_failed_v5_method_gate",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate Fig3 v5 method experiments against fixed publication thresholds.")
    parser.add_argument("--v5-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-oof", type=float, default=0.45)
    parser.add_argument("--min-latest-fold", type=float, default=0.35)
    parser.add_argument("--min-learned-vs-equal", type=float, default=0.03)
    parser.add_argument("--min-contributing-deltas", type=int, default=5)
    parser.add_argument("--min-enrichment", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    decision = build_decision(
        args.v5_run_dir,
        args.baseline_run_dir,
        min_oof=args.min_oof,
        min_latest_fold=args.min_latest_fold,
        min_learned_vs_equal=args.min_learned_vs_equal,
        min_contributing_deltas=args.min_contributing_deltas,
        min_enrichment=args.min_enrichment,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "performance_gate_decision_v5.json", decision)
    if not args.quiet:
        status = "PASS" if decision["final_pass"] else "FAIL"
        print(f"[performance-gate-v5] {status}: {decision['v5_metrics']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
