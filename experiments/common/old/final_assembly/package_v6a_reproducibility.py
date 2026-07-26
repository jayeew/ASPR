from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v2_publication_v6a_locked_candidate"
DEFAULT_REPRO_DIR = DEFAULT_CORPUS_DIR / "reproducibility"
DEFAULT_FINAL_FIG3_DIR = PROJECT_ROOT / "outputs" / "fig03/old/work/v6a_locked_candidate10_full_recompute"
DEFAULT_FINAL_V6A_DIR = PROJECT_ROOT / "outputs" / "fig3_v6a_final_materialized_candidate10_locked"
DEFAULT_VALIDATION_DIRS = [
    PROJECT_ROOT / "outputs" / "fig3_v6a_locked_v4_final_bio_methods_phys10",
    PROJECT_ROOT / "outputs" / "fig3_v6a_independent_v3_all12_locked",
    PROJECT_ROOT / "outputs" / "fig3_v6a_independent_v3_strong11_no_magnetic_locked",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def checksums(root: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        out[rel] = {"sha256": sha256(path), "bytes": file_size(path)}
    return out


def write_readme(repro_dir: Path, manifest: Mapping[str, Any]) -> None:
    best = manifest["final_metrics"]
    lines = [
        "# Fig3 v6A Reproducibility Package",
        "",
        "This directory contains the standardized data package needed to reproduce the locked Fig3 v6A result without relying on `outputs/`.",
        "",
        "## Final Metrics",
        "",
        f"- OOF Spearman: `{best['learned_oof_spearman']}`",
        f"- Latest fold Spearman: `{best['latest_fold_test_spearman']}`",
        f"- Learned vs equal delta: `{best['learned_vs_equal_delta']}`",
        f"- Contributing deltas: `{best['n_contributing_graph_deltas']}`",
        f"- Domains: `{best['n_domains']}`",
        f"- Rows: `{best['n_rows']}`",
        "",
        "## Contents",
        "",
        "- `fig3_recompute/`: final materialized-candidate Fig3 recompute tables.",
        "- `v6a_final/`: locked v6A output generated from the final materialized candidate.",
        "- `validation_probes/`: independent locked validation outputs.",
        "- `metadata/`: corpus manifest, audits, gate decision, and evidence report.",
        "- `reproducibility_manifest.json`: machine-readable manifest and commands.",
        "- `checksums.sha256.json`: checksums for files in this package.",
        "",
        "## Reproduce The Final v6A Metric",
        "",
        "Run from the ASPR repository root:",
        "",
        "```bash",
        "python scripts/fig3_v6a_reliability_latent_probe.py \\",
        "  --fig3-run-dir data/knowledge_corpus/v2_publication_v6a_locked_candidate/reproducibility/fig3_recompute/multi_domain \\",
        "  --out-dir data/knowledge_corpus/v2_publication_v6a_locked_candidate/reproducibility/recomputed_v6a_check \\",
        "  --cohorts moderate \\",
        "  --target-cols RGPM_latent_future_percentile \\",
        "  --feature-sets publication_day_plus \\",
        "  --feature-expansions linear \\",
        "  --cohort-domain-min-rows 20 \\",
        "  --max-pairs 60000 \\",
        "  --epochs 450 \\",
        "  --learning-rate 0.05 \\",
        "  --quiet",
        "```",
        "",
        "Expected decision:",
        "",
        "```text",
        "final_pass=true",
        "learned_oof_spearman≈0.5465543427",
        "latest_fold_test_spearman≈0.6421779791",
        "learned_vs_equal_delta≈0.4153234770",
        "n_contributing_graph_deltas=5",
        "```",
        "",
        "## Reproduce From Corpus Views",
        "",
        "If the full Fig3 recompute needs to be regenerated, use the materialized corpus views:",
        "",
        "```bash",
        "python experiments/fig03/old/fig3_empirical_weight_learning.py \\",
        "  --data-dir data/knowledge_corpus/v2_publication_v6a_locked_candidate/views/fig3 \\",
        "  --out-dir <your_fig3_recompute_out_dir> \\",
        "  --run-mode multi_domain \\",
        "  --domains crispr exoplanets gamma_ray_bursts_and_supernovae genetics_aging_and_longevity_in_model_organisms graphene_2d_materials ipsc_reprogramming microbiome_metagenomics perovskite_solar_cells topological_insulators ubiquitin_and_proteasome_pathways \\",
        "  --panel all --export-tables --diagnostics --audit-only --skip-sensitivity --n-weight-samples 5000 --quiet",
        "```",
        "",
        "Then run the v6A command above with `--fig3-run-dir <your_fig3_recompute_out_dir>/multi_domain`.",
        "",
    ]
    (repro_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def package(args: argparse.Namespace) -> Dict[str, Any]:
    corpus_dir = args.corpus_dir.resolve()
    repro_dir = args.repro_dir.resolve()
    if not corpus_dir.exists():
        raise FileNotFoundError(corpus_dir)
    if corpus_dir not in repro_dir.parents and repro_dir != corpus_dir:
        raise ValueError(f"Repro dir must live under corpus dir: {repro_dir}")

    if repro_dir.exists() and args.clean:
        shutil.rmtree(repro_dir)
    repro_dir.mkdir(parents=True, exist_ok=True)

    copy_path(args.final_fig3_dir.resolve(), repro_dir / "fig3_recompute")
    copy_path(args.final_v6a_dir.resolve(), repro_dir / "v6a_final")
    validation_root = repro_dir / "validation_probes"
    if validation_root.exists():
        shutil.rmtree(validation_root)
    validation_root.mkdir(parents=True, exist_ok=True)
    for src in args.validation_dirs:
        copy_path(src.resolve(), validation_root / src.name)

    metadata_dir = repro_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "manifest.json",
        "performance_gate_decision_v6a.json",
        "quality_report.json",
        "strict_view_audit.json",
        "strict_view_audit.csv",
        "publication_target_domains.json",
    ]:
        src = corpus_dir / name
        if src.exists():
            copy_path(src, metadata_dir / name)
    evidence = args.final_v6a_dir.resolve() / "v6a_final_evidence_report.md"
    if evidence.exists():
        copy_path(evidence, metadata_dir / "v6a_final_evidence_report.md")

    gate = read_json(corpus_dir / "performance_gate_decision_v6a.json")
    final_decision = read_json(repro_dir / "v6a_final" / "fig3_v6a_probe_decision.json")
    final_metrics = dict(final_decision.get("best_run", {}))
    manifest = {
        "artifact_kind": "fig3_v6a_reproducibility_package",
        "created_at": utc_now(),
        "corpus_dir": str(corpus_dir.relative_to(PROJECT_ROOT)),
        "repro_dir": str(repro_dir.relative_to(PROJECT_ROOT)),
        "source_outputs": {
            "final_fig3_dir": str(args.final_fig3_dir),
            "final_v6a_dir": str(args.final_v6a_dir),
            "validation_dirs": [str(path) for path in args.validation_dirs],
        },
        "final_metrics": final_metrics,
        "gate_checks": gate.get("checks", {}),
        "gate_final_pass": bool(gate.get("final_pass", False)),
        "main_domains": gate.get("main_domains", []),
        "reproduce_v6a_command": [
            "python",
            "scripts/fig3_v6a_reliability_latent_probe.py",
            "--fig3-run-dir",
            "data/knowledge_corpus/v2_publication_v6a_locked_candidate/reproducibility/fig3_recompute/multi_domain",
            "--out-dir",
            "data/knowledge_corpus/v2_publication_v6a_locked_candidate/reproducibility/recomputed_v6a_check",
            "--cohorts",
            "moderate",
            "--target-cols",
            "RGPM_latent_future_percentile",
            "--feature-sets",
            "publication_day_plus",
            "--feature-expansions",
            "linear",
            "--cohort-domain-min-rows",
            "20",
            "--max-pairs",
            "60000",
            "--epochs",
            "450",
            "--learning-rate",
            "0.05",
            "--quiet",
        ],
    }
    write_json(repro_dir / "reproducibility_manifest.json", manifest)
    write_readme(repro_dir, manifest)
    write_json(repro_dir / "checksums.sha256.json", checksums(repro_dir))
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package final Fig3 v6A reproducibility data under data/.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--repro-dir", type=Path, default=DEFAULT_REPRO_DIR)
    parser.add_argument("--final-fig3-dir", type=Path, default=DEFAULT_FINAL_FIG3_DIR)
    parser.add_argument("--final-v6a-dir", type=Path, default=DEFAULT_FINAL_V6A_DIR)
    parser.add_argument("--validation-dirs", type=Path, nargs="+", default=DEFAULT_VALIDATION_DIRS)
    parser.add_argument("--clean", action="store_true", help="Replace an existing reproducibility package.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    manifest = package(parse_args(argv))
    print(
        f"[v6a-repro] packaged {manifest['repro_dir']} "
        f"OOF={manifest['final_metrics'].get('learned_oof_spearman')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
