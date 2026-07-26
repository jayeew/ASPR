"""Safely migrate ASPR experiment code and outputs to a canonical layout.

The migration never overwrites a target. Historical scientific artifacts are
moved, not deleted. Only interpreter/test caches are removed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIGURE_CODE_MOVES: Tuple[Tuple[str, str], ...] = tuple(
    (
        f"experiments/kg_perturbation_fig{figure}",
        f"experiments/fig{figure:02d}/old",
    )
    for figure in range(1, 11)
) + tuple(
    (
        f"experiments/new/fig{figure}",
        f"experiments/fig{figure:02d}/new",
    )
    for figure in (1, 2, 3, 4, 5, 6, 7, 9, 10)
)

SHARED_CODE_MOVES: Tuple[Tuple[str, str], ...] = (
    (
        "experiments/aspr_v6_1_nature_figures",
        "experiments/common/new/base",
    ),
    (
        "experiments/aspr_v6_1_figures",
        "experiments/common/old/v6_1_figures_r1",
    ),
    ("experiments/new/common", "experiments/common/new/adapters"),
    (
        "experiments/kg_perturbation_v2",
        "experiments/common/old/kg_perturbation_v2",
    ),
    (
        "experiments/kg_perturbation_final_assembly",
        "experiments/common/old/final_assembly",
    ),
    (
        "experiments/nature_iteration",
        "experiments/common/old/nature_iteration",
    ),
    (
        "experiments/nature_submission_audit",
        "experiments/common/old/submission_audit",
    ),
    (
        "experiments/nature_submission_optimization",
        "experiments/common/old/submission_optimization",
    ),
    ("experiments/new/README.md", "experiments/common/new/README.md"),
    ("experiments/new/run_all.py", "experiments/common/new/run_all.py"),
    ("experiments/new/__init__.py", "experiments/common/new/__init__.py"),
)

AUXILIARY_CODE_MOVES: Tuple[Tuple[str, str], ...] = (
    (
        "scripts/build_fig5_forecast_score_table.py",
        "experiments/fig05/old/build_fig5_forecast_score_table.py",
    ),
    (
        "scripts/draw_fig3_v6a_publication_summary.py",
        "experiments/fig03/old/draw_fig3_v6a_publication_summary.py",
    ),
    (
        "scripts/organize_nature_final_outputs.py",
        "experiments/common/old/final_assembly/organize_nature_final_outputs.py",
    ),
    (
        "scripts/package_v6a_reproducibility.py",
        "experiments/common/old/final_assembly/package_v6a_reproducibility.py",
    ),
    (
        "scripts/screen_corpus_candidates.py",
        "experiments/fig03/old/screen_corpus_candidates.py",
    ),
    (
        "scripts/run_fig4_demo50_remaining.sh",
        "experiments/fig04/old/run_demo50_remaining.sh",
    ),
)

FIGURE_OUTPUT_MOVES: Tuple[Tuple[str, str], ...] = tuple(
    (
        f"outputs/nature_final/fig{figure:02d}_{name}",
        f"outputs/fig{figure:02d}/old",
    )
    for figure, name in (
        (1, "knowledge_perturbation"),
        (2, "empirical_validation"),
        (3, "score_learning"),
        (4, "peer_review_alignment"),
        (5, "ai_frontier"),
        (6, "robustness"),
        (7, "venue_contribution"),
        (8, "aspr_framework"),
        (9, "case_run_instance"),
        (10, "ablation_evidence"),
    )
) + (
    (
        "outputs/experiments_new/fig1",
        "outputs/fig01/old/superseded_event_study",
    ),
) + tuple(
    (
        f"outputs/experiments_new/fig{figure}",
        f"outputs/fig{figure:02d}/new",
    )
    for figure in (2, 3, 4, 5, 6, 7, 9, 10)
)

SHARED_OUTPUT_MOVES: Tuple[Tuple[str, str], ...] = (
    (
        "outputs/experiments_new/fig1_legacy_restored",
        "outputs/fig01/new",
    ),
    (
        "outputs/experiments_new/_cache",
        "outputs/common/new/cache",
    ),
    (
        "outputs/aspr_v6_1_nature_figures_r2",
        "outputs/common/new/base_suite",
    ),
    (
        "outputs/aspr_v6_1_figures_r1",
        "outputs/common/new/baseline_suite_r1",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r5_local",
        "outputs/common/new/model/v6_1_r5",
    ),
    (
        "outputs/nature_portfolio_v5",
        "outputs/common/new/data/nature_portfolio_v5",
    ),
    (
        "outputs/nature_multihorizon_v6_local",
        "outputs/common/old/model/v6",
    ),
    (
        "outputs/nature_multihorizon_v6_1_local",
        "outputs/common/old/model/v6_1_initial",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r1_local",
        "outputs/common/old/model/v6_1_r1",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r2_local",
        "outputs/common/old/model/v6_1_r2",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r3_local",
        "outputs/common/old/model/v6_1_r3",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r4_local",
        "outputs/common/old/model/v6_1_r4",
    ),
    (
        "outputs/nature_final/assembly",
        "outputs/common/old/final_assembly",
    ),
    (
        "outputs/nature_portfolio_v5_gold.nohup",
        "outputs/common/old/logs/nature_portfolio_v5_gold_failed.nohup",
    ),
    (
        "outputs/experiments_new",
        "outputs/common/new/extension_suite",
    ),
    ("outputs/nature_final", "outputs/common/old/final_suite"),
)

CACHE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

PATH_REPLACEMENTS: Tuple[Tuple[str, str], ...] = tuple(
    (
        f"experiments/kg_perturbation_fig{figure}",
        f"experiments/fig{figure:02d}/old",
    )
    for figure in range(1, 11)
) + tuple(
    (
        f"experiments/new/fig{figure}",
        f"experiments/fig{figure:02d}/new",
    )
    for figure in (1, 2, 3, 4, 5, 6, 7, 9, 10)
) + tuple(
    (
        f"experiments.kg_perturbation_fig{figure}",
        f"experiments.fig{figure:02d}.old",
    )
    for figure in range(1, 11)
) + tuple(
    (
        f"experiments.new.fig{figure}",
        f"experiments.fig{figure:02d}.new",
    )
    for figure in (1, 2, 3, 4, 5, 6, 7, 9, 10)
) + (
    (
        "outputs/experiments_new/fig1",
        "outputs/fig01/old/superseded_event_study",
    ),
) + tuple(
    (
        f"outputs/experiments_new/fig{figure}",
        f"outputs/fig{figure:02d}/new",
    )
    for figure in (2, 3, 4, 5, 6, 7, 9, 10)
) + (
    (
        "outputs/experiments_new/fig1_legacy_restored",
        "outputs/fig01/new",
    ),
    (
        "outputs/nature_final/fig01_knowledge_perturbation",
        "outputs/fig01/old",
    ),
    (
        "outputs/nature_final/fig02_empirical_validation",
        "outputs/fig02/old",
    ),
    (
        "outputs/nature_final/fig03_score_learning",
        "outputs/fig03/old",
    ),
    (
        "outputs/nature_final/fig04_peer_review_alignment",
        "outputs/fig04/old",
    ),
    (
        "outputs/nature_final/fig05_ai_frontier",
        "outputs/fig05/old",
    ),
    ("outputs/nature_final/fig06_robustness", "outputs/fig06/old"),
    (
        "outputs/nature_final/fig07_venue_contribution",
        "outputs/fig07/old",
    ),
    ("outputs/nature_final/fig08_aspr_framework", "outputs/fig08/old"),
    (
        "outputs/nature_final/fig09_case_run_instance",
        "outputs/fig09/old",
    ),
    (
        "outputs/nature_final/fig10_ablation_evidence",
        "outputs/fig10/old",
    ),
    (
        "experiments/aspr_v6_1_nature_figures",
        "experiments/common/new/base",
    ),
    (
        "experiments.aspr_v6_1_nature_figures",
        "experiments.common.new.base",
    ),
    (
        "experiments/aspr_v6_1_figures",
        "experiments/common/old/v6_1_figures_r1",
    ),
    (
        "experiments.aspr_v6_1_figures",
        "experiments.common.old.v6_1_figures_r1",
    ),
    ("experiments/new/common", "experiments/common/new/adapters"),
    ("experiments.new.common", "experiments.common.new.adapters"),
    (
        "experiments/kg_perturbation_final_assembly",
        "experiments/common/old/final_assembly",
    ),
    (
        "experiments.kg_perturbation_final_assembly",
        "experiments.common.old.final_assembly",
    ),
    (
        "experiments/kg_perturbation_v2",
        "experiments/common/old/kg_perturbation_v2",
    ),
    (
        "experiments.kg_perturbation_v2",
        "experiments.common.old.kg_perturbation_v2",
    ),
    (
        "experiments/nature_iteration",
        "experiments/common/old/nature_iteration",
    ),
    (
        "experiments.nature_iteration",
        "experiments.common.old.nature_iteration",
    ),
    (
        "experiments/nature_submission_audit",
        "experiments/common/old/submission_audit",
    ),
    (
        "experiments.nature_submission_audit",
        "experiments.common.old.submission_audit",
    ),
    (
        "experiments/nature_submission_optimization",
        "experiments/common/old/submission_optimization",
    ),
    (
        "experiments.nature_submission_optimization",
        "experiments.common.old.submission_optimization",
    ),
    ("experiments/new/run_all.py", "experiments/common/new/run_all.py"),
    ("experiments.new.run_all", "experiments.common.new.run_all"),
    (
        "outputs/aspr_v6_1_nature_figures_r2",
        "outputs/common/new/base_suite",
    ),
    (
        "outputs/aspr_v6_1_figures_r1",
        "outputs/common/new/baseline_suite_r1",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r5_local",
        "outputs/common/new/model/v6_1_r5",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r4_local",
        "outputs/common/old/model/v6_1_r4",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r3_local",
        "outputs/common/old/model/v6_1_r3",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r2_local",
        "outputs/common/old/model/v6_1_r2",
    ),
    (
        "outputs/nature_multihorizon_v6_1_r1_local",
        "outputs/common/old/model/v6_1_r1",
    ),
    (
        "outputs/nature_multihorizon_v6_1_local",
        "outputs/common/old/model/v6_1_initial",
    ),
    (
        "outputs/nature_multihorizon_v6_local",
        "outputs/common/old/model/v6",
    ),
    (
        "outputs/nature_portfolio_v5",
        "outputs/common/new/data/nature_portfolio_v5",
    ),
    (
        "outputs/experiments_new/_cache",
        "outputs/common/new/cache",
    ),
    (
        "outputs/nature_final/assembly",
        "outputs/common/old/final_assembly",
    ),
    ("outputs/kg_perturbation_fig4_full50", "outputs/fig04/old/work/full50"),
    ("outputs/kg_perturbation_fig4_demo50", "outputs/fig04/old/work/demo50"),
    ("outputs/redraw_v6a_best_fig1", "outputs/fig01/old/work/redraw_v6a_best"),
    ("outputs/redraw_v6a_best_fig2", "outputs/fig02/old/work/redraw_v6a_best"),
    ("outputs/redraw_v6a_best_fig3", "outputs/fig03/old/work/redraw_v6a_best"),
    (
        "outputs/kg_perturbation_fig3_auto",
        "outputs/fig03/old/work/auto",
    ),
) + tuple(
    (
        f"outputs/kg_perturbation_fig{figure}",
        f"outputs/fig{figure:02d}/old/work/kg_perturbation",
    )
    for figure in range(1, 11)
) + (
    (
        "outputs/kg_perturbation_final_assembly",
        "outputs/common/old/final_assembly_work",
    ),
    (
        "outputs/kg_perturbation_v2_evidence",
        "outputs/common/old/kg_perturbation_v2/evidence",
    ),
    (
        "outputs/kg_perturbation_v2_rendered",
        "outputs/common/old/kg_perturbation_v2/rendered",
    ),
    (
        "outputs/kg_perturbation_v2_final",
        "outputs/common/old/kg_perturbation_v2/final",
    ),
    ("outputs/nature_iter", "outputs/common/old/nature_iteration"),
    (
        "outputs/nature_submission_audit",
        "outputs/common/old/submission_audit",
    ),
    (
        "outputs/nature_submission_optimization",
        "outputs/common/old/submission_optimization",
    ),
    ("outputs/experiments_new", "outputs/common/new/extension_suite"),
    ("outputs/nature_final", "outputs/common/old/final_suite"),
    ("experiments/new", "experiments/common/new"),
    ("experiments.new", "experiments.common.new"),
    (
        "scripts/build_fig5_forecast_score_table.py",
        "experiments/fig05/old/build_fig5_forecast_score_table.py",
    ),
    (
        "scripts.build_fig5_forecast_score_table",
        "experiments.fig05.old.build_fig5_forecast_score_table",
    ),
    (
        "scripts/draw_fig3_v6a_publication_summary.py",
        "experiments/fig03/old/draw_fig3_v6a_publication_summary.py",
    ),
    (
        "scripts/organize_nature_final_outputs.py",
        "experiments/common/old/final_assembly/organize_nature_final_outputs.py",
    ),
    (
        "scripts/package_v6a_reproducibility.py",
        "experiments/common/old/final_assembly/package_v6a_reproducibility.py",
    ),
    (
        "scripts/screen_corpus_candidates.py",
        "experiments/fig03/old/screen_corpus_candidates.py",
    ),
    (
        "scripts.screen_corpus_candidates",
        "experiments.fig03.old.screen_corpus_candidates",
    ),
    (
        "scripts/run_fig4_demo50_remaining.sh",
        "experiments/fig04/old/run_demo50_remaining.sh",
    ),
) + tuple(
    replacement
    for source, target in (
        (
            "kg_perturbation_fig3_v6a_locked_candidate10_full_recompute",
            "fig03/old/work/v6a_locked_candidate10_full_recompute",
        ),
        (
            "kg_perturbation_fig1_strict_best4",
            "fig01/old/work/strict_best4",
        ),
        (
            "kg_perturbation_fig3_strict_broad10",
            "fig03/old/work/strict_broad10",
        ),
        (
            "kg_perturbation_fig4_full50",
            "fig04/old",
        ),
        (
            "kg_perturbation_fig4_demo50",
            "fig04/old/work/demo50",
        ),
        (
            "redraw_v6a_best_fig1",
            "fig01/old",
        ),
        (
            "redraw_v6a_best_fig2",
            "fig02/old",
        ),
        (
            "redraw_v6a_best_fig3",
            "fig03/old",
        ),
        (
            "kg_perturbation_fig3_audit",
            "fig03/old/work/audit",
        ),
        (
            "kg_perturbation_fig3_auto",
            "fig03/old/work/auto",
        ),
        (
            "kg_perturbation_final_assembly",
            "common/old/final_assembly",
        ),
        (
            "nature_submission_optimization",
            "common/old/submission_optimization",
        ),
        (
            "nature_submission_audit",
            "common/old/submission_audit",
        ),
        (
            "nature_iter",
            "common/old/nature_iteration",
        ),
        *(
            (
                f"kg_perturbation_fig{figure}",
                f"fig{figure:02d}/old",
            )
            for figure in range(1, 11)
        ),
    )
    for replacement in (
        (f'"{source}"', f'"{target}"'),
        (f"'{source}'", f"'{target}'"),
    )
)

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".sh",
    ".txt",
}


def _path_size(path: Path) -> int:
    """Return the byte size of a file, symlink, or directory tree."""
    if path.is_symlink():
        return int(path.lstat().st_size)
    if path.is_file():
        return int(path.stat().st_size)
    return sum(
        int(item.stat().st_size)
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def _path_file_count(path: Path) -> int:
    """Return a stable file count without following symlink targets."""
    if path.is_symlink() or path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _move_one(
    source_rel: str,
    target_rel: str,
    *,
    apply: bool,
) -> Dict[str, object]:
    """Move one exact path, refusing collisions."""
    source = PROJECT_ROOT / source_rel
    target = PROJECT_ROOT / target_rel
    source_present = source.exists() or source.is_symlink()
    target_present = target.exists() or target.is_symlink()
    if source_present and target_present:
        raise FileExistsError(
            f"Both migration source and target exist: {source} -> {target}"
        )
    if not source_present:
        if target_present:
            return {
                "source": source_rel,
                "target": target_rel,
                "status": "already_moved",
                "file_count": _path_file_count(target),
                "size_bytes": _path_size(target),
            }
        return {
            "source": source_rel,
            "target": target_rel,
            "status": "source_absent",
            "file_count": 0,
            "size_bytes": 0,
        }
    record = {
        "source": source_rel,
        "target": target_rel,
        "status": "planned" if not apply else "moved",
        "file_count": _path_file_count(source),
        "size_bytes": _path_size(source),
    }
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    return record


def _cache_paths() -> List[Path]:
    """Find only disposable interpreter and test caches."""
    roots = (
        PROJECT_ROOT / "aspr",
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
    )
    paths: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_dir() and path.name in CACHE_DIR_NAMES
        )
    root_pytest = PROJECT_ROOT / ".pytest_cache"
    if root_pytest.is_dir():
        paths.append(root_pytest)
    return sorted(set(paths), key=lambda path: len(path.parts), reverse=True)


def _clean_caches(*, apply: bool) -> Dict[str, object]:
    """Remove only cache directories and loose bytecode files."""
    directories = _cache_paths()
    bytecode = sorted(
        {
            path
            for root_name in ("aspr", "experiments", "scripts", "tests")
            for path in (PROJECT_ROOT / root_name).rglob("*.py[co]")
            if path.is_file()
        }
    )
    size_bytes = sum(_path_size(path) for path in directories)
    size_bytes += sum(path.stat().st_size for path in bytecode)
    if apply:
        for path in directories:
            if path.exists():
                shutil.rmtree(path)
        for path in bytecode:
            path.unlink(missing_ok=True)
    return {
        "cache_directories": [
            str(path.relative_to(PROJECT_ROOT)) for path in directories
        ],
        "bytecode_files": [
            str(path.relative_to(PROJECT_ROOT)) for path in bytecode
        ],
        "size_bytes": int(size_bytes),
        "status": "removed" if apply else "planned",
    }


def _remove_known_empty_directories(*, apply: bool) -> List[str]:
    """Remove only migration source directories that are now empty."""
    candidates = (
        PROJECT_ROOT / "experiments" / "new",
        PROJECT_ROOT / "outputs" / "experiments_new",
        PROJECT_ROOT / "outputs" / "nature_final",
    )
    removed: List[str] = []
    for path in candidates:
        if not path.is_dir() or any(path.iterdir()):
            continue
        removed.append(str(path.relative_to(PROJECT_ROOT)))
        if apply:
            path.rmdir()
    return removed


def _reference_files() -> Iterable[Path]:
    """Yield current executable/configuration/docs files, not frozen outputs."""
    roots = (
        PROJECT_ROOT / "aspr",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
    )
    excluded = {
        (PROJECT_ROOT / "scripts" / "reorganize_experiment_layout.py").resolve()
    }
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.resolve() in excluded:
                continue
            if "docs/superpowers/plans" in path.as_posix():
                continue
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path
    for path in (PROJECT_ROOT / "README.md", PROJECT_ROOT / "Makefile"):
        if path.is_file():
            yield path


def _rewrite_depth_sensitive_roots(path: Path, text: str) -> str:
    """Adjust project-root parent depth after adding old/new nesting."""
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    if relative.startswith("experiments/fig") and "/old/" in relative:
        text = text.replace(
            "Path(__file__).resolve().parents[2]",
            "Path(__file__).resolve().parents[3]",
        )
    if relative.startswith("experiments/common/new/base/") or relative.startswith(
        "experiments/common/old/"
    ):
        text = text.replace(
            "Path(__file__).resolve().parents[2]",
            "Path(__file__).resolve().parents[4]",
        )
    if relative.startswith("experiments/common/new/adapters/"):
        text = text.replace(
            "Path(__file__).resolve().parents[3]",
            "Path(__file__).resolve().parents[4]",
        )
    if relative == "experiments/fig01/old/run_crispr_example.sh":
        text = text.replace(
            'SCRIPT_DIR/../.." && pwd',
            'SCRIPT_DIR/../../.." && pwd',
        )
    audit_readers = (
        "experiments/common/old/final_assembly/",
        "experiments/common/old/nature_iteration/",
        "experiments/common/old/submission_audit/",
        "experiments/common/old/submission_optimization/",
        "experiments/nature_ready_checks.py",
    )
    if relative.startswith(audit_readers):
        for figure in range(1, 11):
            old_prefix = f"outputs/fig{figure:02d}/old"
            for work_name in (
                "work/kg_perturbation",
                "work/redraw_v6a_best",
                "work/full50",
            ):
                text = text.replace(
                    f"{old_prefix}/{work_name}",
                    old_prefix,
                )
    if relative == (
        "experiments/common/old/final_assembly/"
        "organize_nature_final_outputs.py"
    ):
        organizer_sources = {
            "redraw_v6a_best_fig1": "fig01/old",
            "redraw_v6a_best_fig2": "fig02/old",
            "redraw_v6a_best_fig3": "fig03/old",
            "kg_perturbation_fig4_full50": "fig04/old",
            "kg_perturbation_fig5": "fig05/old",
            "kg_perturbation_fig6": "fig06/old",
            "kg_perturbation_fig7": "fig07/old",
            "kg_perturbation_fig8": "fig08/old",
            "kg_perturbation_fig9": "fig09/old",
            "kg_perturbation_fig10": "fig10/old",
            "kg_perturbation_final_assembly": "common/old/final_assembly",
        }
        for source, target in organizer_sources.items():
            text = text.replace(source, target)
    return text


def _rewrite_references(*, apply: bool) -> Dict[str, object]:
    """Rewrite active path/module references after the physical migration."""
    replacements = sorted(
        set(PATH_REPLACEMENTS),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    changed: List[str] = []
    replacement_count = 0
    for path in sorted(set(_reference_files())):
        original = path.read_text(encoding="utf-8")
        updated = original
        for source, target in replacements:
            occurrences = updated.count(source)
            if occurrences:
                updated = updated.replace(source, target)
                replacement_count += occurrences
        updated = _rewrite_depth_sensitive_roots(path, updated)
        if updated == original:
            continue
        changed.append(str(path.relative_to(PROJECT_ROOT)))
        if apply:
            path.write_text(updated, encoding="utf-8")
    return {
        "status": "rewritten" if apply else "planned",
        "changed_files": changed,
        "replacement_count": int(replacement_count),
    }


def _write_manifest(payload: Mapping[str, object]) -> Path:
    """Write the migration manifest after the output tree has moved."""
    path = PROJECT_ROOT / "outputs" / "common" / "layout_migration_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def reorganize(*, apply: bool) -> Dict[str, object]:
    """Execute or preview the non-overwriting migration."""
    moves: List[Dict[str, object]] = []
    for source, target in (
        *FIGURE_CODE_MOVES,
        *SHARED_CODE_MOVES,
        *AUXILIARY_CODE_MOVES,
        *FIGURE_OUTPUT_MOVES,
        *SHARED_OUTPUT_MOVES,
    ):
        moves.append(_move_one(source, target, apply=apply))
    references = _rewrite_references(apply=apply)
    cleanup = _clean_caches(apply=apply)
    empty_dirs = _remove_known_empty_directories(apply=apply)
    payload: Dict[str, object] = {
        "schema_version": "1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply" if apply else "dry_run",
        "policy": {
            "overwrite_targets": False,
            "historical_scientific_artifacts_deleted": False,
            "deleted_content_classes": [
                "__pycache__",
                "*.pyc",
                "*.pyo",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
            ],
        },
        "moves": moves,
        "reference_rewrite": references,
        "cache_cleanup": cleanup,
        "empty_directories_removed": empty_dirs,
    }
    if apply:
        manifest_path = _write_manifest(payload)
        payload["manifest_path"] = str(manifest_path.relative_to(PROJECT_ROOT))
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the migration. Without this flag, print a dry run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    payload = reorganize(apply=bool(args.apply))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
