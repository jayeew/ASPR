#!/usr/bin/env python3
"""Single, resumable CLI for Nature Multi-Horizon V1.

The CLI keeps Nature v5 read-only, publishes each computational stage through
an atomic artifact store, and never resolves an implicit ``latest`` analysis.
Long-running recovery and OpenAlex acquisition only start through explicit
subcommands; ``--dry-run`` is safe for source inspection and scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.nature_multihorizon.artifact_store import (  # noqa: E402
    ArtifactStore,
    audit_stage,
    hash_file,
    hash_json,
)
from gear.nature_multihorizon.contracts import (  # noqa: E402
    FeatureSpec,
    HorizonSpec,
    ReleaseChannel,
    SplitSpec,
)
from gear.nature_multihorizon.figure_views import (  # noqa: E402
    OPTIONAL_FIGURE_EVIDENCE,
    export_figure_views,
)
from gear.nature_multihorizon.quality import audit_pipeline_tables, write_quality_report  # noqa: E402
from gear.nature_multihorizon.release import (  # noqa: E402
    audit_release,
    build_release_manifest,
    freeze_candidate_path,
    load_release,
    publish_release,
    release_directory,
    validate_candidate_for_freeze,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "nature_multihorizon" / "v1.json"
DATASET_STAGES: Tuple[str, ...] = (
    "ingest-v5",
    "taxonomy",
    "future-citers",
    "graphs",
    "features",
    "targets",
    "cohorts",
    "structural",
    "splits",
)
ANALYSIS_STAGES: Tuple[str, ...] = ("train", "evaluate")
PIPELINE_STAGES: Tuple[str, ...] = DATASET_STAGES + ANALYSIS_STAGES
STAGE_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "ingest-v5": (),
    "taxonomy": ("ingest-v5",),
    "future-citers": ("taxonomy",),
    "graphs": ("ingest-v5", "taxonomy"),
    "features": ("ingest-v5", "taxonomy", "graphs"),
    "targets": ("taxonomy", "future-citers"),
    "cohorts": ("taxonomy", "future-citers", "features", "targets"),
    "structural": (
        "ingest-v5",
        "taxonomy",
        "future-citers",
        "graphs",
        "cohorts",
    ),
    "splits": ("taxonomy", "cohorts"),
    "train": ("features", "targets", "cohorts", "splits"),
    "evaluate": (
        "taxonomy",
        "features",
        "targets",
        "cohorts",
        "structural",
        "splits",
        "train",
    ),
}
REUSABLE_PUBLICATION_STAGES = frozenset(
    {"ingest-v5", "taxonomy", "graphs", "features"}
)
RAW_ONLY_RELEASE_ARTIFACTS = frozenset(
    {
        "paper_references",
        "reference_works",
        "reference_edges",
        "future_citers",
        "future_fetch_status",
        "structural_deltas",
    }
)
PRIMARY_KEYS: Dict[str, Tuple[str, ...]] = {
    "papers.parquet": ("paper_id",),
    "taxonomy_manual_audit_sample.parquet": ("paper_id",),
    "paper_references.parquet": ("paper_id", "reference_id"),
    "reference_works.parquet": ("reference_id",),
    "reference_edges.parquet": ("source_reference_id", "target_reference_id"),
    "future_citers.parquet": ("paper_id", "horizon", "citer_id"),
    "future_fetch_status.parquet": ("paper_id", "requested_horizon"),
    "future_request_manifest.parquet": ("paper_id", "requested_horizon"),
    "future_graph_deltas_multihorizon.parquet": ("paper_id", "horizon"),
    "graph_snapshots.parquet": ("cutoff_year", "graph_id"),
    "features_raw.parquet": ("paper_id",),
    "targets.parquet": ("paper_id", "horizon"),
    "cohort_membership.parquet": ("paper_id", "horizon"),
    "structural_subset.parquet": ("paper_id", "horizon"),
    "structural_deltas.parquet": ("paper_id", "horizon"),
    "structural_targets.parquet": ("paper_id", "horizon"),
    "split_membership.parquet": ("paper_id", "horizon", "split_id"),
    "oof_predictions.parquet": ("paper_id", "horizon", "model_id"),
    "sealed_holdout_predictions.parquet": ("paper_id", "horizon", "model_id"),
    "strict_label_holdout_predictions.parquet": ("paper_id", "horizon", "model_id"),
    "paper_scores.parquet": ("paper_id", "horizon"),
    "oof_paper_scores.parquet": ("paper_id", "horizon"),
    "model_ledger.parquet": ("horizon", "outer_fold", "candidate_id"),
    "evaluation_metrics.parquet": (
        "horizon",
        "model_id",
        "scope",
        "metric",
        "sensitivity",
    ),
}


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _config_path(config: Mapping[str, Any], section: str, key: str) -> Path:
    value = config.get(section, {}).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration is missing {section}.{key}")
    return _resolve_path(value)


def _hash_code() -> str:
    paths = list(
        sorted((PROJECT_ROOT / "gear" / "nature_multihorizon").glob("*.py"))
    )
    paths.extend(
        sorted(
            (PROJECT_ROOT / "experiments" / "kg_perturbation_v2").glob(
                "*.py"
            )
        )
    )
    paths.append(Path(__file__).resolve())
    paths = sorted(set(path for path in paths if path.is_file()))
    return hash_json({str(path.relative_to(PROJECT_ROOT)): hash_file(path) for path in paths})


def _hash_publication_code() -> str:
    names = (
        "artifact_store.py",
        "contracts.py",
        "v5_adapter.py",
        "taxonomy.py",
        "graph_snapshots.py",
        "features.py",
    )
    paths = [
        PROJECT_ROOT / "gear" / "nature_multihorizon" / name for name in names
    ]
    file_hashes = {
        str(path.relative_to(PROJECT_ROOT)): hash_file(path) for path in paths
    }
    # Hash only the orchestration that can change publication-time tables.
    # Downstream target/model/figure CLI edits must not invalidate expensive
    # ingest, taxonomy, graph, or feature stages.
    runner_names = (
        "_run_atomic_stage",
        "_read_stage_table",
        "_require_stage",
        "command_ingest",
        "command_taxonomy",
        "command_graphs",
        "command_features",
    )
    runner_sources = {
        name: inspect.getsource(globals()[name])
        for name in runner_names
        if name in globals()
    }
    return hash_json(
        {
            "files": file_hashes,
            "runner_sources": runner_sources,
        }
    )


def _hash_dirty_diff() -> str:
    scope = [
        "gear/nature_multihorizon",
        "scripts/run_nature_multihorizon.py",
        "configs/nature_multihorizon",
        "experiments/common/old/kg_perturbation_v2",
    ]
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *scope],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        unstaged = subprocess.run(
            ["git", "diff", "--binary", "--", *scope],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        staged = subprocess.run(
            ["git", "diff", "--cached", "--binary", "--", *scope],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "--", *scope],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        untracked = {}
        for relative in sorted(line for line in untracked_output.splitlines() if line):
            path = PROJECT_ROOT / relative
            if path.is_file():
                untracked[relative] = hash_file(path)
        return hash_json(
            {
                "status_sha256": hashlib.sha256(status).hexdigest(),
                "unstaged_diff_sha256": hashlib.sha256(unstaged).hexdigest(),
                "staged_diff_sha256": hashlib.sha256(staged).hexdigest(),
                "untracked_files": untracked,
            }
        )
    except (OSError, subprocess.CalledProcessError):
        return hash_json({"status": "git-diff-unavailable"})


def _source_file_signature(path: Path) -> Dict[str, Any]:
    """Return a stable signature without rereading a multi-GB raw file."""

    stat = path.stat()
    signature: Dict[str, Any] = {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= 64 * 1024 * 1024:
        signature["sha256"] = hash_file(path)
        return signature
    sample_size = 4 * 1024 * 1024
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_size))
        handle.seek(max(0, stat.st_size - sample_size))
        digest.update(handle.read(sample_size))
    signature["head_tail_sha256"] = f"sha256:{digest.hexdigest()}"
    signature["signature_kind"] = "size_mtime_head_tail"
    return signature


def _source_snapshot_id(
    source_dir: Path,
    future_source_dir: Optional[Path] = None,
) -> str:
    inventory: Dict[str, Any] = {}
    data_files = (
        "nature_target_works.csv",
        "nature_reference_edges.csv",
        "nature_reference_works.csv",
        "nature_source_roster.csv",
    )
    for name in data_files:
        path = source_dir / name
        if path.is_file():
            inventory[name] = _source_file_signature(path)
    success = source_dir / "reference_closure_recovery" / "_SUCCESS"
    if success.is_file():
        inventory["reference_closure_success"] = success.read_text(
            encoding="utf-8"
        ).strip()
    recovery_manifest = source_dir / "reference_closure_recovery" / "manifest.json"
    if recovery_manifest.is_file():
        payload = _read_json(recovery_manifest)
        inventory["reference_recovery_contract"] = {
            key: payload.get(key)
            for key in (
                "stage_status",
                "n_reference_ids",
                "n_reference_ids_found",
                "n_reference_ids_missing",
                "coverage",
                "bad_json_records_ledger_total",
                "reference_rows",
            )
        }
    snapshot_manifest = source_dir / "reference_closure_snapshot_manifest.json"
    if snapshot_manifest.is_file():
        payload = _read_json(snapshot_manifest)
        inventory["reference_snapshot_contract"] = {
            key: payload.get(key)
            for key in (
                "artifact_kind",
                "n_unique_reference_ids",
                "n_reference_works_found",
                "n_reference_works_missing_locally",
                "local_snapshot_coverage",
            )
        }
    if future_source_dir is not None:
        future_root = Path(future_source_dir)
        for name in (
            "future_multihorizon_manifest.json",
            "data_quality_report.json",
            "future_fetch_status.parquet",
            "future_request_manifest.parquet",
            "future_graph_deltas_multihorizon.parquet",
            "future_citers.parquet",
        ):
            path = future_root / name
            if path.is_file():
                inventory[f"future_multihorizon/{name}"] = _source_file_signature(
                    path
                )
    if not inventory:
        inventory["source_dir"] = str(source_dir)
    return hash_json(inventory)


def _validate_config(config: Mapping[str, Any]) -> None:
    """Fail at CLI startup when JSON and typed scientific contracts diverge."""
    horizon_rows = config.get("horizons")
    if not isinstance(horizon_rows, list) or not horizon_rows:
        raise ValueError("Configuration must define at least one horizon")
    horizons = tuple(HorizonSpec.model_validate(row) for row in horizon_rows)
    if int(config.get("primary_horizon", -1)) not in {item.tau for item in horizons}:
        raise ValueError("primary_horizon is not present in horizons")
    features = config.get("features", {})
    mechanisms = config.get("mechanisms", {})
    FeatureSpec(
        core_features=tuple(features.get("core8", ())),
        auxiliary_features=tuple(features.get("aux10", ())),
        mechanisms={str(name): tuple(values) for name, values in mechanisms.items()},
    )
    cv = config.get("cv", {})
    holdouts = {
        item.tau: (item.sealed_test_start_year, item.sealed_test_end_year)
        for item in horizons
    }
    SplitSpec(
        outer_folds=int(cv.get("outer_folds", 5)),
        inner_folds=int(cv.get("inner_folds", 4)),
        seed=int(cv.get("seed", 20260710)),
        year_bin_width=int(cv.get("year_bin_size", 5)),
        bootstrap_iterations=int(cv.get("bootstrap_repetitions", 2_000)),
        sealed_holdout_years=holdouts,
    )
    from gear.nature_multihorizon.taxonomy import DOMAIN_IDS

    configured_domains = tuple(str(value) for value in config.get("domains", ()))
    if set(configured_domains) != set(DOMAIN_IDS) or len(configured_domains) != 12:
        raise ValueError("Configured domains do not match the locked taxonomy DOMAIN_IDS")
    venue_families = tuple(str(value) for value in config.get("venue_families", ()))
    if len(venue_families) != 6 or len(set(venue_families)) != 6:
        raise ValueError("Exactly six Nature Portfolio venue families are required")


@dataclass(frozen=True)
class Runtime:
    """Resolved identifiers and immutable store paths for one CLI invocation."""

    config_path: Path
    config: Mapping[str, Any]
    source_dir: Path
    snapshot_dir: Path
    store: ArtifactStore
    dataset_id: str
    analysis_id: str
    source_snapshot_id: str
    config_hash: str
    code_hash: str
    publication_code_hash: str
    dirty_diff_hash: str
    release_root: Path
    dataset_entry_root: Path
    reuse_publication_dataset_id: Optional[str] = None
    future_source_dir: Optional[Path] = None

    def stage_identifier(self, stage: str) -> str:
        if stage in ANALYSIS_STAGES:
            return self.analysis_id
        if (
            self.reuse_publication_dataset_id
            and stage in REUSABLE_PUBLICATION_STAGES
        ):
            return self.reuse_publication_dataset_id
        return self.dataset_id

    def stage_payload(self, stage: str) -> Path:
        return self.store.stage_path(self.stage_identifier(stage), stage) / "payload"


def build_runtime(args: argparse.Namespace) -> Runtime:
    config_path = args.config.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    source_dir = _config_path(config, "source", "v5_output_dir")
    configured_future = getattr(args, "future_source_dir", None)
    if configured_future is not None:
        future_source_dir = Path(configured_future).expanduser().resolve()
    else:
        future_value = config.get("source", {}).get("future_multihorizon_dir")
        if isinstance(future_value, str) and future_value:
            candidate = Path(future_value).expanduser()
            future_source_dir = (
                candidate.resolve()
                if candidate.is_absolute()
                else (source_dir / candidate).resolve()
            )
        else:
            future_source_dir = (source_dir / "future_multihorizon").resolve()
    snapshot_dir = _config_path(config, "source", "openalex_snapshot_dir")
    external_root = _config_path(config, "storage", "external_root")
    release_root = _config_path(config, "storage", "analysis_root")
    dataset_entry_root = _config_path(config, "storage", "dataset_root")
    config_hash = hash_file(config_path)
    code_hash = _hash_code()
    publication_code_hash = _hash_publication_code()
    dirty_diff_hash = _hash_dirty_diff()
    source_snapshot_id = _source_snapshot_id(source_dir, future_source_dir)
    common_release_fingerprint: Optional[str] = None
    common_manifest = None
    if args.common_candidate_release is not None:
        common_release_path = args.common_candidate_release.expanduser().resolve()
        common_release = load_release(common_release_path, require_frozen=False)
        common_manifest = common_release.manifest
        if common_manifest.channel is not ReleaseChannel.CANDIDATE:
            raise ValueError("--common-candidate-release must name a candidate release")
        common_release_fingerprint = hash_file(
            common_release.path / "release.json"
        )
        inferred_reuse_id = common_manifest.dataset_id
        if (
            args.reuse_publication_dataset_id is not None
            and args.reuse_publication_dataset_id != inferred_reuse_id
        ):
            raise ValueError(
                "--reuse-publication-dataset-id must equal the common "
                "candidate dataset_id"
            )
        args.reuse_publication_dataset_id = inferred_reuse_id
    elif args.reuse_publication_dataset_id is not None:
        raise ValueError(
            "Publication-stage reuse requires --common-candidate-release"
        )
    dataset_seed = hash_json(
        {
            "source_snapshot_id": source_snapshot_id,
            "config_hash": config_hash,
            # Dataset stages include taxonomy, graph, feature, target, cohort,
            # and split code.  A code change must therefore produce a new
            # immutable dataset identifier instead of colliding with an
            # already-published stage tree.
            "code_hash": code_hash,
            "horizons": list(args.horizons),
            "requested_horizon": int(args.requested_horizon),
            "observation_end_year": int(
                args.complete_observation_year
                or config.get("observation_end_year", 2025)
            ),
            "max_citers_per_work": int(args.max_citers_per_work),
            "future_scope": str(args.future_scope),
            "common_candidate_release": common_release_fingerprint,
            "reuse_publication_dataset_id": args.reuse_publication_dataset_id,
            "max_papers": args.max_papers,
            "smoke_test": bool(args.smoke_test),
        }
    )
    dataset_id = args.dataset_id or f"nmhv1-{dataset_seed.split(':')[-1][:12]}"
    analysis_seed = hash_json(
        {
            "dataset_id": dataset_id,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "horizons": list(args.horizons),
            "sealed_holdout_unlocked": bool(args.unlock_sealed_holdout),
        }
    )
    analysis_id = args.analysis_id or f"nmhv1a-{analysis_seed.split(':')[-1][:12]}"
    runtime = Runtime(
        config_path=config_path,
        config=config,
        source_dir=source_dir,
        future_source_dir=future_source_dir,
        snapshot_dir=snapshot_dir,
        store=ArtifactStore(external_root),
        dataset_id=dataset_id,
        analysis_id=analysis_id,
        source_snapshot_id=source_snapshot_id,
        config_hash=config_hash,
        code_hash=code_hash,
        publication_code_hash=publication_code_hash,
        dirty_diff_hash=dirty_diff_hash,
        release_root=release_root,
        dataset_entry_root=dataset_entry_root,
        reuse_publication_dataset_id=args.reuse_publication_dataset_id,
    )
    if runtime.reuse_publication_dataset_id:
        for stage in REUSABLE_PUBLICATION_STAGES:
            path = runtime.store.stage_path(
                runtime.reuse_publication_dataset_id, stage
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Reusable publication stage is missing: {path}"
                )
            manifest = audit_stage(path).require_ok()
            if manifest.source_snapshot_id != runtime.source_snapshot_id:
                raise ValueError(
                    f"Reusable stage {stage} has a different source_snapshot_id"
                )
            if manifest.config_hash != runtime.config_hash:
                raise ValueError(
                    f"Reusable stage {stage} has a different config_hash"
                )
            if manifest.code_hash != runtime.publication_code_hash:
                raise ValueError(
                    f"Reusable stage {stage} was built by different publication-feature code"
                )
            if common_manifest is None:
                raise ValueError("Reusable stages require an audited common candidate")
            if manifest.output_sha256 not in set(common_manifest.input_artifact_ids):
                raise ValueError(
                    f"Reusable stage {stage} is not in common candidate lineage"
                )
    return runtime


def _row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _stage_protocol(
    runtime: Runtime,
    stage: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    common_release: Optional[Dict[str, Any]] = None
    if args.common_candidate_release is not None:
        source = args.common_candidate_release.expanduser().resolve()
        release_json = source if source.name == "release.json" else source / "release.json"
        common_release = {
            "path": str(release_json),
            "sha256": hash_file(release_json),
        }
    return {
        "schema_version": "1.0.0",
        "stage": stage,
        "stage_identifier": runtime.stage_identifier(stage),
        "dataset_id": runtime.dataset_id,
        "analysis_id": runtime.analysis_id,
        "dependencies": list(STAGE_DEPENDENCIES.get(stage, ())),
        "dependency_output_sha256": {
            dependency: runtime.store.load_manifest(
                runtime.stage_identifier(dependency), dependency
            ).output_sha256
            for dependency in STAGE_DEPENDENCIES.get(stage, ())
        },
        "parameters": {
            "horizons": [int(value) for value in args.horizons],
            "requested_horizon": int(args.requested_horizon),
            "future_scope": str(args.future_scope),
            "future_input_mode": str(
                getattr(args, "_future_input_mode", "not_applicable")
            ),
            "future_source_dir": str(runtime.future_source_dir),
            "common_candidate_release": common_release,
            "reuse_publication_dataset_id": runtime.reuse_publication_dataset_id,
            "complete_observation_year": int(
                args.complete_observation_year
                or runtime.config.get("observation_end_year", 2025)
            ),
            "max_citers_per_work": int(args.max_citers_per_work),
            "max_papers": args.max_papers,
            "smoke_test": bool(args.smoke_test),
            "sealed_holdout_unlocked": bool(args.unlock_sealed_holdout),
        },
    }


def _completed_stage(runtime: Runtime, stage: str) -> Optional[Dict[str, Any]]:
    path = runtime.store.stage_path(runtime.stage_identifier(stage), stage)
    if not path.exists():
        return None
    audit = audit_stage(path)
    manifest = audit.require_ok()
    return {"stage": stage, "path": str(path), "resumed": True, "manifest": manifest.model_dump(mode="json")}


def _run_atomic_stage(
    runtime: Runtime,
    stage: str,
    args: argparse.Namespace,
    builder: Callable[[Path], Mapping[str, Any]],
) -> Dict[str, Any]:
    completed = _completed_stage(runtime, stage)
    if completed is not None:
        if args.resume:
            return completed
        raise FileExistsError(f"Completed immutable stage exists: {completed['path']}")
    if args.dry_run:
        return {
            "stage": stage,
            "stage_id": runtime.stage_identifier(stage),
            "path": str(runtime.store.stage_path(runtime.stage_identifier(stage), stage)),
            "dry_run": True,
        }
    upstream = STAGE_DEPENDENCIES.get(stage, ())
    input_ids = tuple(
        runtime.store.load_manifest(runtime.stage_identifier(name), name).output_sha256
        for name in upstream
    )
    with runtime.store.stage(
        runtime.stage_identifier(stage),
        stage,
        resume=args.resume,
        input_artifact_ids=input_ids,
        source_snapshot_id=runtime.source_snapshot_id,
        config_hash=runtime.config_hash,
        code_hash=(
            runtime.publication_code_hash
            if stage in REUSABLE_PUBLICATION_STAGES
            else runtime.code_hash
        ),
        dirty_diff_hash=runtime.dirty_diff_hash,
    ) as handle:
        payload_dir = handle.path / "payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        result = dict(builder(payload_dir))
        _write_json(
            payload_dir / "stage_protocol.json",
            _stage_protocol(runtime, stage, args),
        )
        for path in payload_dir.rglob("*.parquet"):
            relative = path.relative_to(handle.path).as_posix()
            handle.record_table(relative, _row_count(path), PRIMARY_KEYS.get(path.name, ()))
    if stage in DATASET_STAGES:
        locator = runtime.dataset_entry_root / f"{runtime.dataset_id}.json"
        _write_json(
            locator,
            {
                "schema_version": "1.0.0",
                "dataset_id": runtime.dataset_id,
                "external_store": str(runtime.store.root),
                "stage": stage,
                "stage_path": str(handle.result.path),
                "stage_output_sha256": handle.result.manifest.output_sha256,
                "implicit_latest_forbidden": True,
            },
        )
    return {
        "stage": stage,
        "path": str(handle.result.path),
        "manifest": handle.result.manifest.model_dump(mode="json"),
        "result": result,
    }


def _require_stage(runtime: Runtime, stage: str) -> Path:
    path = runtime.stage_payload(stage)
    audit_stage(path.parent).require_ok()
    return path


def _read_stage_table(runtime: Runtime, stage: str, filename: str) -> pd.DataFrame:
    path = _require_stage(runtime, stage) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Stage {stage} is missing {filename}: {path}")
    return pd.read_parquet(path)


def _future_citers_table(runtime: Runtime) -> Path:
    """Resolve the audited large future-citer table without duplicating it."""

    from gear.nature_multihorizon.future_citers import resolve_future_citers_table

    return resolve_future_citers_table(_require_stage(runtime, "future-citers"))


def command_audit_source(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.future_citers import (
        audit_prebuilt_future_multihorizon,
    )
    from gear.nature_multihorizon.v5_adapter import (
        audit_snapshot_reference_closure,
        audit_v5_source,
    )

    source = audit_v5_source(runtime.source_dir, deep_jsonl=bool(args.deep_jsonl))
    snapshot = audit_snapshot_reference_closure(runtime.source_dir)
    quality = runtime.config.get("quality", {})
    future: Dict[str, Any]
    try:
        audited = audit_prebuilt_future_multihorizon(
            runtime.future_source_dir or runtime.source_dir / "future_multihorizon",
            expected_horizons=args.horizons,
            minimum_success_rate=float(
                quality.get("min_future_fetch_success", 0.99)
            ),
            maximum_missing_checkpoints=int(
                quality.get("max_prebuilt_missing_checkpoints", 5)
            ),
        )
        future = {
            key: value
            for key, value in audited.items()
            if key not in {"status", "requests", "deltas", "manifest", "quality", "paths"}
        }
    except (OSError, ValueError) as exc:
        future = {
            "source_dir": str(runtime.future_source_dir),
            "accepted_for_training": False,
            "error": str(exc),
        }
    source["snapshot_reference_closure"] = snapshot
    source["future_multihorizon"] = future
    source["formal_source_ready"] = bool(
        source.get("formal_source_ready")
        and snapshot.get("ok")
        and future.get("accepted_for_training")
    )
    blockers = list(source.get("blockers", []))
    if not snapshot.get("ok"):
        blockers.append("snapshot_reference_closure_audit_failed")
    if not future.get("accepted_for_training"):
        blockers.append("prebuilt_future_multihorizon_audit_failed")
    source["blockers"] = sorted(set(blockers))
    return source


def command_recover(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.v5_adapter import recover_v5_reference_closure

    return recover_v5_reference_closure(
        runtime.source_dir,
        runtime.snapshot_dir,
        resume=args.resume,
        workers=args.workers,
        max_snapshot_files=args.max_snapshot_files,
        dry_run=args.dry_run,
    )


def command_ingest(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.v5_adapter import (
        audit_reference_recovery,
        audit_snapshot_reference_closure,
        audit_v5_source,
        ingest_v5,
    )

    source_audit = audit_v5_source(runtime.source_dir)
    recovery_manifest_path = runtime.source_dir / "reference_closure_recovery" / "manifest.json"
    recovery_success = runtime.source_dir / "reference_closure_recovery" / "_SUCCESS"
    recovery_manifest = _read_json(recovery_manifest_path) if recovery_manifest_path.is_file() else {}
    snapshot_audit = audit_snapshot_reference_closure(runtime.source_dir)
    use_recovery_contract = bool(recovery_manifest_path.is_file())
    coverage = float(
        recovery_manifest.get("coverage", 0.0)
        if use_recovery_contract
        else snapshot_audit.get("coverage", 0.0)
    )
    minimum_coverage = float(
        runtime.config.get("quality", {}).get("min_global_reference_coverage", 0.70)
    )
    blockers = list(source_audit.get("blockers", []))
    if use_recovery_contract:
        if not recovery_success.is_file() or recovery_manifest.get("stage_status") != "complete":
            blockers.append("reference_recovery_success_marker_missing")
        recovery_audit = audit_reference_recovery(
            runtime.source_dir,
            verify_reference_hash=not args.dry_run,
        )
        if not recovery_audit["ok"]:
            blockers.extend(
                f"reference_recovery_audit:{error}"
                for error in recovery_audit["errors"]
            )
        required_inventory = {"target_works", "paper_references", "reference_works"}
        declared_inventory = set(
            recovery_manifest.get("source_inventory", {})
            if isinstance(recovery_manifest.get("source_inventory"), Mapping)
            else {}
        )
        missing_inventory = sorted(required_inventory - declared_inventory)
        if missing_inventory:
            blockers.append(
                "reference_recovery_inventory_missing:" + ",".join(missing_inventory)
            )
    elif not snapshot_audit.get("ok"):
        blockers.extend(
            f"snapshot_reference_closure_audit:{error}"
            for error in snapshot_audit.get("errors", ())
        )
    if coverage < minimum_coverage:
        blockers.append(f"reference_coverage_below_{minimum_coverage}")
    if blockers and args.dry_run:
        return {
            "stage": "ingest-v5",
            "dry_run": True,
            "raw_go_gate": False,
            "blockers": sorted(set(blockers)),
            "reference_coverage": coverage,
        }
    if blockers and not args.smoke_test:
        raise RuntimeError(f"Raw source GO gate failed: {sorted(set(blockers))}")

    return _run_atomic_stage(
        runtime,
        "ingest-v5",
        args,
        lambda output: ingest_v5(
            runtime.source_dir,
            output,
            include_legacy_future=False,
        ),
    )


def command_import_future(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    """Adopt the completed offline τ3/τ5/τ8 tables without network access."""

    from gear.nature_multihorizon.future_citers import (
        import_prebuilt_future_multihorizon,
    )

    setattr(args, "_future_input_mode", "prebuilt_offline_multihorizon")

    def build(output: Path) -> Mapping[str, Any]:
        quality = runtime.config.get("quality", {})
        return import_prebuilt_future_multihorizon(
            runtime.future_source_dir or runtime.source_dir / "future_multihorizon",
            output,
            expected_horizons=args.horizons,
            minimum_success_rate=float(
                quality.get("min_future_fetch_success", 0.99)
            ),
            maximum_missing_checkpoints=int(
                quality.get("max_prebuilt_missing_checkpoints", 5)
            ),
        )

    return _run_atomic_stage(runtime, "future-citers", args, build)


def command_taxonomy(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.taxonomy import DOMAIN_IDS, build_taxonomy_table

    def build(output: Path) -> Mapping[str, Any]:
        papers = _read_stage_table(runtime, "ingest-v5", "papers.parquet")
        mapped, coverage, audit = build_taxonomy_table(papers)
        mapped.to_parquet(output / "papers.parquet", index=False)
        coverage.to_parquet(output / "taxonomy_coverage.parquet", index=False)
        audit_pool = mapped[mapped["domain12"].isin(DOMAIN_IDS)].copy()
        audit_pool["__audit_order"] = pd.util.hash_pandas_object(
            audit_pool["paper_id"].astype(str), index=False
        ).to_numpy()
        audit_sample = (
            audit_pool.sort_values(
                ["domain12", "__audit_order", "paper_id"], kind="stable"
            )
            .groupby("domain12", group_keys=False)
            .head(25)
            .drop(columns="__audit_order")
        )
        audit_columns = [
            name
            for name in (
                "paper_id",
                "title",
                "primary_topic",
                "openalex_primary_subfield",
                "openalex_primary_field",
                "display_topic_id",
                "domain12",
                "domain12_reason",
            )
            if name in audit_sample
        ]
        audit_sample[audit_columns].to_parquet(
            output / "taxonomy_manual_audit_sample.parquet", index=False
        )
        audit["manual_audit_sample_rows"] = int(len(audit_sample))
        audit["manual_audit_sample_per_domain_max"] = 25
        _write_json(output / "taxonomy_audit.json", audit)
        threshold = float(runtime.config.get("quality", {}).get("min_domain_mapping_coverage", 0.95))
        if float(audit.get("mapping_coverage", 0.0)) < threshold:
            raise RuntimeError(f"Domain mapping coverage {audit.get('mapping_coverage')} is below {threshold}")
        return audit

    return _run_atomic_stage(runtime, "taxonomy", args, build)


def command_future_citers(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.future_citers import (
        fetch_future_citers,
        materialize_future_tables,
        merge_materialized_future_batches,
    )
    from gear.nature_multihorizon.release import load_release
    from gear.corpus import short_openalex_id
    from scripts.build_openalex_v3_citation_graph import OpenAlexClient, split_api_keys

    setattr(args, "_future_input_mode", "online_openalex")

    def build(output: Path) -> Mapping[str, Any]:
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        if "natural_science_eligible" in papers.columns:
            papers = papers[papers["natural_science_eligible"].fillna(False).astype(bool)].copy()
        observation_end_year = int(args.complete_observation_year or runtime.config.get("observation_end_year", 2025))
        if int(args.requested_horizon) != 8:
            raise ValueError(
                "V1 uses the locked tau8 common request; recent tau5/tau3 "
                "requests are selected with --future-scope expanded"
            )
        unlock_audit: Dict[str, Any] = {
            "future_scope": str(args.future_scope),
            "common_candidate_required": args.future_scope == "expanded",
        }
        if args.future_scope == "expanded":
            if args.common_candidate_release is None:
                raise ValueError(
                    "--future-scope expanded requires --common-candidate-release"
                )
            common_path = args.common_candidate_release.expanduser().resolve()
            common_root = common_path.parent if common_path.name == "release.json" else common_path
            common_release = load_release(common_root, require_frozen=False)
            if common_release.manifest.source_snapshot_id != runtime.source_snapshot_id:
                raise ValueError(
                    "Common candidate and expanded run use different v5 source snapshots"
                )
            quality = _read_json(common_release.artifact("quality_report"))
            if quality.get("go_for_training") is not True:
                raise RuntimeError(
                    "Common candidate did not pass data/training quality gates"
                )
            common_requests = pd.read_parquet(
                common_release.artifact("future_request_manifest")
            )
            if (
                common_requests.empty
                or not pd.to_numeric(
                    common_requests["requested_horizon"], errors="coerce"
                ).eq(8).all()
                or pd.to_numeric(
                    common_requests["publication_year"], errors="coerce"
                ).max()
                > 2017
            ):
                raise ValueError(
                    "Unlock release is not the predeclared <=2017 tau8 common queue"
                )
            common_metrics = pd.read_parquet(
                common_release.artifact("evaluation_metrics")
            )
            tau5_oof = common_metrics[
                pd.to_numeric(common_metrics["horizon"], errors="coerce").eq(5)
                & common_metrics["model_id"].eq("nested_selector")
                & common_metrics["scope"].eq("development_oof")
                & common_metrics["metric"].eq("rho_global_calibrated")
            ]
            minimum_common_oof = float(
                runtime.config.get("quality", {}).get(
                    "min_common_candidate_tau5_oof", 0.0
                )
            )
            observed_common_oof = (
                float(pd.to_numeric(tau5_oof["value"], errors="coerce").max())
                if len(tau5_oof)
                else float("nan")
            )
            if (
                not np.isfinite(observed_common_oof)
                or observed_common_oof <= minimum_common_oof
            ):
                raise RuntimeError(
                    "Common candidate OOF did not unlock recent requests: "
                    f"rho={observed_common_oof}, required>{minimum_common_oof}"
                )
            unlock_audit.update(
                {
                    "common_analysis_id": common_release.manifest.analysis_id,
                    "common_release_hash": common_release.manifest.output_sha256,
                    "common_tau5_oof": observed_common_oof,
                    "unlock_passed": True,
                }
            )
        client = OpenAlexClient(
            api_key=os.environ.get("OPENALEX_API_KEY"),
            api_keys=split_api_keys(os.environ.get("OPENALEX_API_KEYS")),
            email=os.environ.get("OPENALEX_EMAIL"),
            sleep_seconds=float(args.sleep_seconds),
            timeout_seconds=int(args.timeout_seconds),
            max_retries=int(args.max_retries),
        )

        def fetcher(paper_id: str, start_year: int, end_year: int, cap: int) -> Iterable[Mapping[str, Any]]:
            filters = [
                f"cites:{short_openalex_id(paper_id)}",
                f"from_publication_date:{start_year}-01-01",
                f"to_publication_date:{end_year}-12-31",
                "is_retracted:false",
                "is_paratext:false",
            ]
            return client.list_works(
                max_records=int(cap),
                filters=filters,
                sort="publication_date:asc",
                per_page=200,
                progress=False,
            )

        batch_specs: List[Tuple[str, int, Optional[int], int, Tuple[int, ...]]] = [
            ("common_tau8_le2017", 8, None, 2017, (3, 5, 8))
        ]
        if args.future_scope == "expanded":
            batch_specs.extend(
                [
                    ("recent_tau5_2018_2020", 5, 2018, 2020, (3, 5)),
                    ("recent_tau3_2021_2022", 3, 2021, 2022, (3,)),
                ]
            )
        checkpoint_base = (
            runtime.store.root
            / "checkpoints"
            / "future_citers"
            / runtime.source_snapshot_id.split(":")[-1][:24]
            / f"cap{int(args.max_citers_per_work)}"
        )
        batch_outputs: List[Path] = []
        counter_rows: Dict[str, Any] = {}
        for batch_name, requested_horizon, minimum_year, maximum_year, derived in batch_specs:
            checkpoint_root = checkpoint_base / batch_name
            batch_output = output / "batches" / batch_name
            counters = fetch_future_citers(
                papers,
                checkpoint_root,
                fetcher,
                requested_horizon=requested_horizon,
                complete_end_year=observation_end_year,
                max_citers_per_work=int(args.max_citers_per_work),
                resume=args.resume,
                retry_failed=args.retry_failed,
                max_papers=args.max_papers,
                min_publication_year=minimum_year,
                max_publication_year=maximum_year,
                request_batch=batch_name,
            )
            materialize_future_tables(
                checkpoint_root,
                batch_output,
                requested_horizon=requested_horizon,
                derived_horizons=derived,
            )
            counter_rows[batch_name] = counters
            batch_outputs.append(batch_output)
        manifest = merge_materialized_future_batches(batch_outputs, output)
        manifest["future_scope"] = str(args.future_scope)
        manifest["unlock_audit"] = unlock_audit
        status = pd.read_parquet(output / "future_fetch_status.parquet")
        expected = pd.read_parquet(output / "future_request_manifest.parquet")
        expected_columns = ["paper_id", "requested_horizon"]
        if "request_batch" in expected:
            expected_columns.append("request_batch")
        joined = expected[expected_columns].merge(
            status[["paper_id", "requested_horizon", "fetch_status"]],
            on=["paper_id", "requested_horizon"],
            how="left",
            validate="one_to_one",
        )
        status_coverage = float(joined["fetch_status"].isin(["success", "failed"]).mean())
        success_rate = float(joined["fetch_status"].eq("success").mean())
        minimum_success = float(
            runtime.config.get("quality", {}).get("min_future_fetch_success", 0.99)
        )
        if status_coverage < 1.0 or success_rate < minimum_success:
            raise RuntimeError(
                "Future-citer GO gate failed; resume with --retry-failed: "
                f"status_coverage={status_coverage:.6f}, success_rate={success_rate:.6f}"
            )
        batch_quality: Dict[str, Dict[str, Any]] = {}
        if "request_batch" not in joined:
            joined["request_batch"] = "unspecified"
        for batch_name, rows in joined.groupby(
            "request_batch", dropna=False, sort=True
        ):
            batch_coverage = float(
                rows["fetch_status"].isin(["success", "failed"]).mean()
            )
            batch_success = float(rows["fetch_status"].eq("success").mean())
            batch_quality[str(batch_name)] = {
                "n_requested": int(len(rows)),
                "status_coverage": batch_coverage,
                "success_rate": batch_success,
            }
            if batch_coverage < 1.0 or batch_success < minimum_success:
                raise RuntimeError(
                    "Future-citer batch GO gate failed; resume with "
                    f"--retry-failed: batch={batch_name}, "
                    f"status_coverage={batch_coverage:.6f}, "
                    f"success_rate={batch_success:.6f}"
                )
        manifest["status_coverage"] = status_coverage
        manifest["success_rate"] = success_rate
        manifest["batch_quality"] = batch_quality
        return {"fetch_batches": counter_rows, "materialized": manifest}

    return _run_atomic_stage(runtime, "future-citers", args, build)


def command_graphs(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.graph_snapshots import build_graph_snapshots

    def build(output: Path) -> Mapping[str, Any]:
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        paper_references = _read_stage_table(runtime, "ingest-v5", "paper_references.parquet")
        reference_works = _read_stage_table(runtime, "ingest-v5", "reference_works.parquet")
        reference_edges = _read_stage_table(runtime, "ingest-v5", "reference_edges.parquet")
        graph_config = runtime.config.get("graph", {})
        catalog = build_graph_snapshots(
            papers,
            paper_references,
            reference_works,
            reference_edges,
            output,
            interval=int(graph_config.get("snapshot_interval_years", 5)),
            max_pairs_per_paper=int(graph_config.get("max_reference_pairs_per_paper", 10_000)),
            seed=int(graph_config.get("seed", 2033)),
        )
        return {"n_snapshots": int(len(catalog))}

    return _run_atomic_stage(runtime, "graphs", args, build)


def command_features(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.features import build_feature_table, feature_quality_summary

    def build(output: Path) -> Mapping[str, Any]:
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        paper_references = _read_stage_table(runtime, "ingest-v5", "paper_references.parquet")
        reference_works = _read_stage_table(runtime, "ingest-v5", "reference_works.parquet")
        catalog_path = _require_stage(runtime, "graphs") / "graph_snapshots.parquet"
        graph_config = runtime.config.get("graph", {})
        features = build_feature_table(
            papers,
            paper_references,
            reference_works,
            catalog_path,
            max_pairs=int(graph_config.get("max_reference_pairs_per_paper", 10_000)),
            seed=int(graph_config.get("seed", 20260710)),
        )
        features.to_parquet(output / "features_raw.parquet", index=False)
        summary = feature_quality_summary(features)
        _write_json(output / "feature_quality.json", summary)
        minimum = float(
            runtime.config.get("quality", {}).get("min_feature_finite_coverage", 0.95)
        )
        minimum_observed = float(summary.get("core8_all_finite_rate", 0.0))
        if (
            minimum_observed < minimum
            or not summary.get("strict_prior_year")
        ) and not args.smoke_test:
            raise RuntimeError(
                f"Feature GO gate failed: minimum finite coverage={minimum_observed:.6f}, "
                f"strict_prior_year={summary.get('strict_prior_year')}"
            )
        return {
            "n_features": int(len(features)),
            "columns": list(features.columns),
            "quality": summary,
        }

    return _run_atomic_stage(runtime, "features", args, build)


def command_targets(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.targets import (
        build_diffusion_targets,
        build_diffusion_targets_from_deltas,
    )

    def build(output: Path) -> Mapping[str, Any]:
        future_stage = _require_stage(runtime, "future-citers")
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        horizons = tuple(int(value) for value in args.horizons)
        minimum_taxonomy_coverage = float(
            runtime.config.get("quality", {}).get(
                "min_future_taxonomy_coverage", 0.80
            )
        )
        delta_path = future_stage / "future_graph_deltas_multihorizon.parquet"
        if delta_path.is_file():
            targets = build_diffusion_targets_from_deltas(
                papers,
                pd.read_parquet(delta_path),
                horizons=horizons,
                min_future_citers=10,
                min_taxonomy_coverage=minimum_taxonomy_coverage,
            )
            target_source = "audited_precomputed_future_deltas"
        else:
            future_path = _future_citers_table(runtime)
            future_citers = pd.read_parquet(future_path)
            future_status = pd.read_parquet(
                future_stage / "future_fetch_status.parquet"
            )
            targets = build_diffusion_targets(
                papers,
                future_citers,
                future_status,
                horizons=horizons,
                min_future_citers=10,
                min_taxonomy_coverage=minimum_taxonomy_coverage,
            )
            target_source = "future_citer_rows"
        targets.to_parquet(output / "targets.parquet", index=False)
        return {
            "n_targets": int(len(targets)),
            "horizons": list(horizons),
            "target_component_source": target_source,
            "structural_target_status": "not_built_until_future_citer_reference_closure_passes",
        }

    return _run_atomic_stage(runtime, "targets", args, build)


def command_cohorts(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.cohorts import build_cohort_membership, cohort_quality_summary
    from gear.nature_multihorizon.contracts import CohortSpec

    def build(output: Path) -> Mapping[str, Any]:
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        features = _read_stage_table(runtime, "features", "features_raw.parquet")
        targets = _read_stage_table(runtime, "targets", "targets.parquet")
        future_status_path = _require_stage(runtime, "future-citers") / "future_fetch_status.parquet"
        if not future_status_path.is_file():
            raise FileNotFoundError("future_fetch_status.parquet is required; API failures cannot be inferred as zero")
        _ = pd.read_parquet(future_status_path)
        membership = build_cohort_membership(
            papers,
            features,
            targets,
            spec=CohortSpec(),
            complete_end_year=int(runtime.config.get("observation_end_year", 2025)),
        )
        membership.to_parquet(output / "cohort_membership.parquet", index=False)
        summary = cohort_quality_summary(membership)
        _write_json(output / "cohort_quality.json", summary)
        failed_horizons = [
            horizon
            for horizon, values in summary.get("by_horizon", {}).items()
            if not values.get("cohort_5000_gate") or not values.get("eight_domains_200_gate")
        ]
        if failed_horizons and not args.smoke_test:
            raise RuntimeError(f"Cohort GO gate failed for horizons: {failed_horizons}")
        return {
            "n_rows": int(len(membership)),
            "n_eligible": int(membership["cohort_member"].sum()),
            "quality": summary,
        }

    return _run_atomic_stage(runtime, "cohorts", args, build)


def command_structural(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.structural import (
        build_structural_validation,
        lock_structural_subset,
        read_future_citers_for_subset,
    )

    def build(output: Path) -> Mapping[str, Any]:
        membership = _read_stage_table(runtime, "cohorts", "cohort_membership.parquet")
        future_citers_path = _future_citers_table(runtime)
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        paper_references = _read_stage_table(runtime, "ingest-v5", "paper_references.parquet")
        graph_catalog = _require_stage(runtime, "graphs") / "graph_snapshots.parquet"
        seed = int(runtime.config.get("cv", {}).get("seed", 20260710))
        subset, subset_audit = lock_structural_subset(
            membership,
            future_citers_path,
            max_papers=5_000,
            min_future_reference_coverage=0.80,
            seed=seed,
        )
        future_citers = read_future_citers_for_subset(
            future_citers_path, subset
        )
        deltas, structural_targets, target_audit = build_structural_validation(
            subset,
            papers,
            paper_references,
            future_citers,
            graph_catalog,
            seed=seed,
        )
        subset.to_parquet(output / "structural_subset.parquet", index=False)
        deltas.to_parquet(output / "structural_deltas.parquet", index=False)
        structural_targets.to_parquet(output / "structural_targets.parquet", index=False)
        audit = {"subset": subset_audit, "targets": target_audit}
        _write_json(output / "structural_audit.json", audit)
        return audit

    return _run_atomic_stage(runtime, "structural", args, build)


def command_splits(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.splits import (
        make_nested_folds,
        split_sealed_holdout,
        split_strict_label_availability,
    )

    def build(output: Path) -> Mapping[str, Any]:
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        membership = _read_stage_table(runtime, "cohorts", "cohort_membership.parquet")
        eligible = membership[
            membership["cohort_member"].fillna(False).astype(bool)
        ].copy()
        # Cohort rows normally already carry these publication-time fields.
        # Merge only genuinely absent columns; an unconditional merge would
        # create ``publication_year_x``/``_y`` and make the sealed split
        # contract impossible to resolve.
        split_metadata = ("publication_year", "domain12", "venue_family")
        missing_metadata = [name for name in split_metadata if name not in eligible]
        if missing_metadata:
            eligible = eligible.merge(
                papers[["paper_id", *missing_metadata]].drop_duplicates("paper_id"),
                on="paper_id",
                how="left",
                validate="many_to_one",
            )
        unresolved = [name for name in split_metadata if name not in eligible]
        if unresolved:
            raise ValueError(f"Cohort is missing split metadata: {unresolved}")
        eligible = eligible.sort_values(
            ["horizon", "paper_id"], kind="stable"
        ).reset_index(drop=True)
        cv = runtime.config.get("cv", {})
        frames: List[pd.DataFrame] = []
        audits: Dict[str, Any] = {}
        for horizon, group in eligible.groupby("horizon", sort=True):
            group = group.reset_index(drop=True)
            holdout = split_sealed_holdout(group, int(horizon))
            strict = split_strict_label_availability(group, int(horizon))
            development = group.iloc[holdout.development_idx].reset_index(drop=True)
            plan = make_nested_folds(
                development,
                n_outer=int(cv.get("outer_folds", 5)),
                n_inner=int(cv.get("inner_folds", 4)),
                seed=int(cv.get("seed", 2033)) + int(horizon),
                year_bin_width=int(cv.get("year_bin_size", 5)),
            )
            assignment = plan.assignments.copy()
            assignment["paper_id"] = development.loc[assignment["row_position"], "paper_id"].to_numpy()
            assignment["horizon"] = int(horizon)
            assignment["split_id"] = assignment["outer_fold"].map(lambda value: f"outer_{int(value)}")
            assignment["is_sealed_holdout"] = False
            strict_papers = set(group.iloc[strict.development_idx]["paper_id"].astype(str))
            assignment["strict_label_available"] = assignment["paper_id"].astype(str).isin(strict_papers)
            frames.append(assignment.drop(columns=["row_position"]))
            if len(holdout.holdout_idx):
                sealed = group.iloc[holdout.holdout_idx][["paper_id"]].copy()
                sealed["horizon"] = int(horizon)
                sealed["outer_fold"] = 0
                sealed["stratification_level"] = "sealed_temporal"
                sealed["stratification_label"] = "sealed_temporal"
                sealed["split_id"] = "sealed_temporal_holdout"
                sealed["is_sealed_holdout"] = True
                sealed["strict_label_available"] = False
                frames.append(sealed)
            audits[str(horizon)] = plan.audit
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result.to_parquet(output / "split_membership.parquet", index=False)
        _write_json(output / "split_audit.json", audits)
        return {"n_rows": int(len(result)), "horizons": sorted(audits)}

    return _run_atomic_stage(runtime, "splits", args, build)


def _model_frame(runtime: Runtime) -> pd.DataFrame:
    papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
    features = _read_stage_table(runtime, "features", "features_raw.parquet")
    targets = _read_stage_table(runtime, "targets", "targets.parquet")
    cohorts = _read_stage_table(runtime, "cohorts", "cohort_membership.parquet")
    splits = _read_stage_table(runtime, "splits", "split_membership.parquet")
    frame = targets.merge(features, on="paper_id", how="inner").merge(
        cohorts, on=["paper_id", "horizon"], how="inner", suffixes=("", "_cohort")
    )
    frame = frame[frame["cohort_member"].fillna(False).astype(bool)]
    missing_metadata = [
        name
        for name in ("publication_year", "domain12", "venue_family")
        if name in papers and name not in frame
    ]
    if missing_metadata:
        frame = frame.merge(
            papers[["paper_id", *missing_metadata]].drop_duplicates("paper_id"),
            on="paper_id",
            how="left",
        )
    modeled = frame.merge(
        splits[["paper_id", "horizon", "outer_fold", "is_sealed_holdout"]],
        on=["paper_id", "horizon"],
        how="inner",
    )
    if "quality_flags" not in modeled:
        modeled["quality_flags"] = ""
    if "cap_hit" in modeled:
        cap_mask = pd.to_numeric(
            modeled["cap_hit"], errors="coerce"
        ).fillna(0).astype(bool)
        modeled.loc[cap_mask, "quality_flags"] = modeled.loc[
            cap_mask, "quality_flags"
        ].fillna("").map(
            lambda value: ";".join(
                item
                for item in (
                    str(value).strip(";"),
                    "future_citer_cap_hit_1000",
                )
                if item
            )
        )
    return modeled.sort_values(["horizon", "paper_id"], kind="stable").reset_index(
        drop=True
    )


def command_train(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.evaluation import run_nested_oof
    from gear.nature_multihorizon.splits import split_strict_label_availability

    def build(output: Path) -> Mapping[str, Any]:
        frame = _model_frame(runtime)
        predictions: List[pd.DataFrame] = []
        ledgers: List[pd.DataFrame] = []
        metrics: List[pd.DataFrame] = []
        holdouts: List[pd.DataFrame] = []
        strict_holdouts: List[pd.DataFrame] = []
        summaries: Dict[str, Any] = {}
        for horizon, group in frame.groupby("horizon", sort=True):
            target_column = "rgpm_d_raw"
            if target_column not in group.columns:
                target_column = f"RGPM-D{int(horizon)}"
            result = run_nested_oof(
                group.reset_index(drop=True),
                horizon=int(horizon),
                target_col=target_column,
                outer_folds=int(runtime.config.get("cv", {}).get("outer_folds", 5)),
                inner_folds=int(runtime.config.get("cv", {}).get("inner_folds", 4)),
                seed=int(runtime.config.get("cv", {}).get("seed", 2033)) + int(horizon),
                run_holdout=bool(args.unlock_sealed_holdout),
            )
            expected_outer = (
                group.loc[~group["is_sealed_holdout"].fillna(False).astype(bool), ["paper_id", "outer_fold"]]
                .drop_duplicates("paper_id")
                .rename(columns={"outer_fold": "expected_outer_fold"})
            )
            observed_outer = (
                result.oof_predictions[["paper_id", "outer_fold"]]
                .drop_duplicates()
                .rename(columns={"outer_fold": "observed_outer_fold"})
            )
            fold_audit = expected_outer.merge(
                observed_outer,
                on="paper_id",
                how="outer",
                validate="one_to_one",
            )
            if (
                fold_audit[["expected_outer_fold", "observed_outer_fold"]].isna().any(axis=None)
                or not fold_audit["expected_outer_fold"].eq(fold_audit["observed_outer_fold"]).all()
            ):
                raise RuntimeError(
                    f"Training folds diverged from frozen split_membership for horizon={horizon}"
                )
            predictions.append(result.oof_predictions)
            ledgers.append(result.model_ledger)
            metrics.append(result.evaluation_metrics)
            if not result.holdout_predictions.empty:
                holdouts.append(result.holdout_predictions)
            if (
                args.unlock_sealed_holdout
                and bool(runtime.config.get("cv", {}).get("run_strict_label_availability_test", True))
            ):
                strict_split = split_strict_label_availability(group.reset_index(drop=True), int(horizon))
                strict_positions = np.concatenate(
                    [strict_split.development_idx, strict_split.holdout_idx]
                )
                strict_frame = group.reset_index(drop=True).iloc[strict_positions].reset_index(drop=True)
                strict_result = run_nested_oof(
                    strict_frame,
                    horizon=int(horizon),
                    target_col=target_column,
                    outer_folds=int(runtime.config.get("cv", {}).get("outer_folds", 5)),
                    inner_folds=int(runtime.config.get("cv", {}).get("inner_folds", 4)),
                    seed=int(runtime.config.get("cv", {}).get("seed", 20260710)) + int(horizon) + 500_000,
                    bootstrap_iterations=int(runtime.config.get("cv", {}).get("bootstrap_repetitions", 2_000)),
                    run_holdout=True,
                )
                if not strict_result.holdout_predictions.empty:
                    strict_part = strict_result.holdout_predictions.copy()
                    strict_part["evaluation_protocol"] = "strict_label_availability"
                    strict_holdouts.append(strict_part)
                strict_metrics = strict_result.evaluation_metrics.copy()
                strict_metrics["scope"] = strict_metrics["scope"].map(
                    lambda value: f"strict_label_availability__{value}"
                )
                metrics.append(strict_metrics)
            summaries[str(horizon)] = result.summary
        oof = pd.concat(predictions, ignore_index=True)
        ledger = pd.concat(ledgers, ignore_index=True)
        metric_frame = pd.concat(metrics, ignore_index=True)
        holdout = pd.concat(holdouts, ignore_index=True) if holdouts else pd.DataFrame()
        strict_holdout = (
            pd.concat(strict_holdouts, ignore_index=True)
            if strict_holdouts
            else pd.DataFrame()
        )
        oof.to_parquet(output / "oof_predictions.parquet", index=False)
        ledger.to_parquet(output / "model_ledger.parquet", index=False)
        metric_frame.to_parquet(output / "evaluation_metrics_nested.parquet", index=False)
        holdout.to_parquet(output / "sealed_holdout_predictions.parquet", index=False)
        strict_holdout.to_parquet(
            output / "strict_label_holdout_predictions.parquet",
            index=False,
        )
        _write_json(output / "training_summary.json", summaries)
        return {
            "n_oof": int(len(oof)),
            "n_ledger": int(len(ledger)),
            "n_holdout": int(len(holdout)),
            "n_strict_label_holdout": int(len(strict_holdout)),
            "sealed_holdout_unlocked": bool(args.unlock_sealed_holdout),
        }

    return _run_atomic_stage(runtime, "train", args, build)


def command_evaluate(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    from gear.nature_multihorizon.evaluation import evaluate_oof_predictions, safe_spearman
    from gear.nature_multihorizon.models import (
        DomainYearCalibrator,
        TargetResidualizer,
        fit_candidate_model,
    )
    from gear.nature_multihorizon.scoring import build_paper_scores, score_frame
    from gear.nature_multihorizon.targets import FoldLocalDiffusionTarget

    import joblib

    def build(output: Path) -> Mapping[str, Any]:
        frame = _model_frame(runtime)
        oof = _read_stage_table(runtime, "train", "oof_predictions.parquet")
        ledger = _read_stage_table(runtime, "train", "model_ledger.parquet")
        metrics = evaluate_oof_predictions(
            oof,
            bootstrap_iterations=int(runtime.config.get("cv", {}).get("bootstrap_repetitions", 2_000)),
            seed=int(runtime.config.get("cv", {}).get("seed", 2033)),
        )
        nested_metrics_path = _require_stage(runtime, "train") / "evaluation_metrics_nested.parquet"
        if nested_metrics_path.is_file():
            nested_metrics = pd.read_parquet(nested_metrics_path)
            metrics = pd.concat([metrics, nested_metrics], ignore_index=True)
        if "sensitivity" not in metrics:
            metrics["sensitivity"] = "main"
        else:
            metrics["sensitivity"] = metrics["sensitivity"].fillna("main").astype(str)
        metrics = metrics.drop_duplicates(
            ["horizon", "model_id", "scope", "metric", "sensitivity"],
            keep="last",
        )
        structural_path = _require_stage(runtime, "structural") / "structural_targets.parquet"
        if structural_path.is_file():
            structural = pd.read_parquet(structural_path)
        else:
            structural = pd.DataFrame()
        structural_components = (
            "modularity_shock",
            "boundary_mixing_change",
            "partition_change",
            "path_shortening",
        )
        if {"paper_id", "horizon", *structural_components}.issubset(
            structural.columns
        ):
            selected = oof[oof["is_selected"].fillna(False).astype(bool)].copy()
            validation = selected.merge(
                structural[["paper_id", "horizon", *structural_components]],
                on=["paper_id", "horizon"],
                how="inner",
                validate="one_to_one",
            )
            tau5 = validation[validation["horizon"].eq(5)].reset_index(drop=True)
            component_values = tau5[list(structural_components)].apply(
                pd.to_numeric, errors="coerce"
            )
            valid_components = component_values.notna().all(axis=1)
            tau5 = tau5.loc[valid_components].reset_index(drop=True)
            component_values = component_values.loc[valid_components].reset_index(
                drop=True
            )
            # RGPM-S used for the confirmatory gate is ranked only within the
            # development OOF validation population. Sealed outcomes therefore
            # cannot change either a development target or its correlation.
            tau5["rgpm_s5_development"] = component_values.rank(
                method="average", pct=True
            ).mean(axis=1)
            rho = safe_spearman(
                tau5["prediction_calibrated"], tau5["rgpm_s5_development"]
            )
            validation_years = pd.to_numeric(
                tau5.get("publication_year", pd.Series(np.nan, index=tau5.index)),
                errors="coerce",
            )
            clusters = (
                tau5.get("domain12", pd.Series("unknown", index=tau5.index)).astype(str)
                + "|"
                + validation_years.floordiv(5).mul(5).astype("Int64").astype(str)
            )
            unique_clusters = clusters.unique()
            bootstrap: List[float] = []
            iterations = int(runtime.config.get("cv", {}).get("bootstrap_repetitions", 2_000))
            if len(tau5) >= 3 and len(unique_clusters) >= 2:
                rng = np.random.default_rng(
                    int(runtime.config.get("cv", {}).get("seed", 20260710)) + 55_555
                )
                indices = {
                    cluster: clusters[clusters.eq(cluster)].index.to_numpy()
                    for cluster in unique_clusters
                }
                for _ in range(iterations):
                    sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
                    rows = np.concatenate([indices[value] for value in sampled])
                    value = safe_spearman(
                        tau5.iloc[rows]["prediction_calibrated"],
                        tau5.iloc[rows]["rgpm_s5_development"],
                    )
                    if np.isfinite(value):
                        bootstrap.append(value)
            ci_low = float(np.percentile(bootstrap, 2.5)) if bootstrap else float("nan")
            ci_high = float(np.percentile(bootstrap, 97.5)) if bootstrap else float("nan")
            metrics = pd.concat(
                [
                    metrics,
                    pd.DataFrame(
                        [
                            {
                                "horizon": 5,
                                "model_id": "nested_selector",
                                "scope": "structural_validation_subset",
                                "metric": "rho_rgpm_s5",
                                "value": rho,
                                "ci_low": ci_low,
                                "ci_high": ci_high,
                                "n": int(len(tau5)),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        if "sensitivity" not in metrics:
            metrics["sensitivity"] = "main"
        else:
            metrics["sensitivity"] = metrics["sensitivity"].fillna("main").astype(str)
        validated_claim_scope = (
            "42 Nature Portfolio sources; pre-publication-year graph; "
            "validated conditionally among papers with at least 10 future citers; "
            "future-citer cap-hit rows flagged and uncapped sensitivity gated"
        )
        oof_scores = build_paper_scores(
            frame,
            oof,
            ledger,
            claim_scope=validated_claim_scope,
        )
        features = _read_stage_table(runtime, "features", "features_raw.parquet")
        papers = _read_stage_table(runtime, "taxonomy", "papers.parquet")
        extra_metadata = ["paper_id"] + [
            column
            for column in ("doi", "title", "source_display_name")
            if column in papers.columns and column not in features.columns
        ]
        scoring_frame = features.copy()
        if len(extra_metadata) > 1:
            scoring_frame = scoring_frame.merge(
                papers[extra_metadata].drop_duplicates("paper_id"),
                on="paper_id",
                how="left",
                validate="one_to_one",
            )
        if "natural_science_eligible" in papers.columns:
            natural_ids = set(
                papers.loc[
                    papers["natural_science_eligible"].fillna(False).astype(bool),
                    "paper_id",
                ].astype(str)
            )
            scoring_frame = scoring_frame[
                scoring_frame["paper_id"].astype(str).isin(natural_ids)
            ].copy()

        membership = _read_stage_table(
            runtime, "cohorts", "cohort_membership.parquet"
        )
        score_ids_by_horizon = {
            int(horizon): set(group.loc[
                group["cohort_member"].fillna(False).astype(bool), "paper_id"
            ].astype(str))
            for horizon, group in membership.groupby("horizon", sort=True)
        }
        cap_ids_by_horizon = {
            int(horizon): set(
                group.loc[
                    group["cohort_member"].fillna(False).astype(bool)
                    & pd.to_numeric(group["cap_hit"], errors="coerce")
                    .fillna(0)
                    .astype(bool),
                    "paper_id",
                ].astype(str)
            )
            for horizon, group in membership.groupby("horizon", sort=True)
        }
        configured_case_dois = {
            str(item.get("doi") or "")
            .lower()
            .replace("https://doi.org/", "")
            .strip()
            for item in runtime.config.get("case_studies", [])
            if item.get("doi")
        }
        case_ids: set[str] = set()
        if configured_case_dois and "doi" in papers:
            normalized_doi = (
                papers["doi"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.replace("https://doi.org/", "", regex=False)
                .str.strip()
            )
            case_ids = set(
                papers.loc[normalized_doi.isin(configured_case_dois), "paper_id"].astype(str)
            )

        full_score_parts: List[pd.DataFrame] = []
        model_dir = output / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        horizon_config = {
            int(item["tau"]): item for item in runtime.config.get("horizons", [])
        }
        for horizon, training in frame.groupby("horizon", sort=True):
            training = training.reset_index(drop=True)
            if not args.unlock_sealed_holdout:
                holdout_start = int(horizon_config[int(horizon)]["sealed_test_start_year"])
                training = training[
                    pd.to_numeric(training["publication_year"], errors="coerce")
                    < holdout_start
                ].reset_index(drop=True)
            selected_rows = ledger[
                ledger["horizon"].eq(horizon)
                & ledger["selected"].fillna(False).astype(bool)
            ]
            if selected_rows.empty:
                raise RuntimeError(f"No nested-selected performance model for horizon={horizon}")
            locked_selection = selected_rows[
                pd.to_numeric(
                    selected_rows["outer_fold"], errors="coerce"
                ).eq(0)
            ]
            if len(locked_selection) != 1:
                raise RuntimeError(
                    f"Expected one full-development locked selection for horizon={horizon}"
                )
            selected_model = str(locked_selection.iloc[0]["candidate_id"])
            # Refit the five RGPM-D component ECDFs on this horizon's actual
            # training rows.  The global descriptive target is never reused as
            # a training label or as a sealed-holdout normalization reference.
            raw_target = FoldLocalDiffusionTarget().fit_transform(training)
            residualizer = TargetResidualizer().fit(training, raw_target)
            adjusted_target = residualizer.transform(training, raw_target)
            mechanism_model = fit_candidate_model(
                "mechanism5_simplex",
                training,
                adjusted_target,
                seed=int(runtime.config.get("cv", {}).get("seed", 20260710)) + int(horizon) + 700_000,
            )
            performance_model = fit_candidate_model(
                selected_model,
                training,
                adjusted_target,
                seed=int(runtime.config.get("cv", {}).get("seed", 20260710)) + int(horizon) + 800_000,
            )
            raw_training_score = performance_model.predict(training)
            calibrator = DomainYearCalibrator().fit(
                training,
                raw_training_score,
                adjusted_target,
            )
            version = f"{runtime.analysis_id}-tau{int(horizon)}-{selected_model}"
            cohort_score_ids = score_ids_by_horizon.get(int(horizon), set())
            allowed_score_ids = cohort_score_ids | case_ids
            horizon_scoring = scoring_frame[
                scoring_frame["paper_id"].astype(str).isin(allowed_score_ids)
            ].copy()
            if horizon_scoring.empty:
                raise RuntimeError(
                    f"No cohort-qualified scoring rows for horizon={horizon}"
                )
            in_validated_population = horizon_scoring["paper_id"].astype(str).isin(
                cohort_score_ids
            )
            is_cap_hit = horizon_scoring["paper_id"].astype(str).isin(
                cap_ids_by_horizon.get(int(horizon), set())
            )
            if "quality_flags" not in horizon_scoring:
                horizon_scoring["quality_flags"] = ""
            horizon_scoring.loc[in_validated_population, "quality_flags"] = (
                horizon_scoring.loc[in_validated_population, "quality_flags"]
                .fillna("")
                .map(
                    lambda value: ";".join(
                        item
                        for item in (
                            str(value).strip(";"),
                            "outcome_conditioned_future_citers_ge_10",
                        )
                        if item
                    )
                )
            )
            horizon_scoring.loc[~in_validated_population, "quality_flags"] = (
                horizon_scoring.loc[~in_validated_population, "quality_flags"]
                .fillna("")
                .map(
                    lambda value: ";".join(
                        item
                        for item in (
                            str(value).strip(";"),
                            "fixed_case_out_of_cohort_extrapolation",
                            "future_citer_eligibility_unknown",
                        )
                        if item
                    )
                )
            )
            horizon_scoring.loc[is_cap_hit, "quality_flags"] = (
                horizon_scoring.loc[is_cap_hit, "quality_flags"]
                .fillna("")
                .map(
                    lambda value: ";".join(
                        item
                        for item in (
                            str(value).strip(";"),
                            "future_citer_cap_hit_1000",
                        )
                        if item
                    )
                )
            )
            scored, _ = score_frame(
                horizon_scoring,
                horizon=int(horizon),
                mechanism_model=mechanism_model,
                performance_model=performance_model,
                calibrator=calibrator,
                performance_percentile_reference=calibrator.calibrated_reference_,
                model_version=version,
                claim_scope=validated_claim_scope,
            )
            publication_year = pd.to_numeric(scored["publication_year"], errors="coerce")
            observable = publication_year + int(horizon) <= int(
                runtime.config.get("observation_end_year", 2025)
            )
            scored["outcome_observable"] = observable.astype(int)
            scored.loc[~observable, "quality_flags"] = scored.loc[
                ~observable, "quality_flags"
            ].fillna("").map(
                lambda value: ";".join(
                    item
                    for item in (str(value).strip(";"), "recent_paper_outcome_not_observed")
                    if item
                )
            )
            scored["score_scope"] = "full_fit_descriptive"
            metadata = ["paper_id"] + [
                column
                for column in ("doi", "title", "source_display_name")
                if column in scoring_frame.columns and column not in scored.columns
            ]
            if len(metadata) > 1:
                scored = scored.merge(
                    scoring_frame[metadata].drop_duplicates("paper_id"),
                    on="paper_id",
                    how="left",
                    validate="one_to_one",
                )
            full_score_parts.append(scored)
            joblib.dump(
                {
                    "analysis_id": runtime.analysis_id,
                    "horizon": int(horizon),
                    "target_adjustment": "fold-compatible log1p(n_future_citers)",
                    "mechanism_model": mechanism_model,
                    "performance_model": performance_model,
                    "calibrator": calibrator,
                    "performance_percentile_reference": calibrator.calibrated_reference_,
                    "model_version": version,
                    "feature_version": "nature-multihorizon-feature-v1",
                },
                model_dir / f"model_bundle_tau{int(horizon)}.joblib",
            )
        scores = pd.concat(full_score_parts, ignore_index=True)
        metrics.to_parquet(output / "evaluation_metrics.parquet", index=False)
        scores.to_parquet(output / "paper_scores.parquet", index=False)
        oof_scores.to_parquet(output / "oof_paper_scores.parquet", index=False)
        return {
            "n_metrics": int(len(metrics)),
            "n_scores": int(len(scores)),
            "n_oof_scores": int(len(oof_scores)),
            "model_bundles": sorted(path.name for path in model_dir.glob("*.joblib")),
        }

    return _run_atomic_stage(runtime, "evaluate", args, build)


def _audit_pipeline_identity(runtime: Runtime) -> None:
    """Fail closed before publishing if stage lineage and runtime diverge."""

    for stage in PIPELINE_STAGES:
        identifier = runtime.stage_identifier(stage)
        manifest = audit_stage(
            runtime.store.stage_path(identifier, stage)
        ).require_ok()
        if manifest.dataset_id != identifier or manifest.stage_name != stage:
            raise ValueError(f"Stage identity mismatch for {stage}")
        if manifest.source_snapshot_id != runtime.source_snapshot_id:
            raise ValueError(f"Stage source_snapshot_id mismatch for {stage}")
        if manifest.config_hash != runtime.config_hash:
            raise ValueError(f"Stage config_hash mismatch for {stage}")
        expected_code_hash = (
            runtime.publication_code_hash
            if stage in REUSABLE_PUBLICATION_STAGES
            else runtime.code_hash
        )
        if manifest.code_hash != expected_code_hash:
            raise ValueError(f"Stage code_hash mismatch for {stage}")
        is_reused = bool(
            runtime.reuse_publication_dataset_id
            and stage in REUSABLE_PUBLICATION_STAGES
        )
        if not is_reused and manifest.dirty_diff_hash != runtime.dirty_diff_hash:
            raise ValueError(f"Stage dirty provenance mismatch for {stage}")
        expected_inputs = tuple(
            runtime.store.load_manifest(
                runtime.stage_identifier(dependency), dependency
            ).output_sha256
            for dependency in STAGE_DEPENDENCIES[stage]
        )
        if tuple(manifest.input_artifact_ids) != expected_inputs:
            raise ValueError(
                f"Stage upstream lineage mismatch for {stage}: "
                f"expected {expected_inputs}, observed "
                f"{tuple(manifest.input_artifact_ids)}"
            )


def _canonical_artifacts(runtime: Runtime) -> Dict[str, Path]:
    mapping = {
        "papers": ("taxonomy", "papers.parquet"),
        "paper_references": ("ingest-v5", "paper_references.parquet"),
        "reference_works": ("ingest-v5", "reference_works.parquet"),
        "reference_edges": ("ingest-v5", "reference_edges.parquet"),
        "future_fetch_status": ("future-citers", "future_fetch_status.parquet"),
        "future_request_manifest": ("future-citers", "future_request_manifest.parquet"),
        "graph_snapshots": ("graphs", "graph_snapshots.parquet"),
        "features_raw": ("features", "features_raw.parquet"),
        "targets": ("targets", "targets.parquet"),
        "cohort_membership": ("cohorts", "cohort_membership.parquet"),
        "structural_subset": ("structural", "structural_subset.parquet"),
        "structural_deltas": ("structural", "structural_deltas.parquet"),
        "structural_targets": ("structural", "structural_targets.parquet"),
        "structural_audit": ("structural", "structural_audit.json"),
        "split_audit": ("splits", "split_audit.json"),
        "split_membership": ("splits", "split_membership.parquet"),
        "oof_predictions": ("train", "oof_predictions.parquet"),
        "sealed_holdout_predictions": ("train", "sealed_holdout_predictions.parquet"),
        "strict_label_holdout_predictions": ("train", "strict_label_holdout_predictions.parquet"),
        "model_ledger": ("train", "model_ledger.parquet"),
        "training_summary": ("train", "training_summary.json"),
        "evaluation_metrics": ("evaluate", "evaluation_metrics.parquet"),
        "paper_scores": ("evaluate", "paper_scores.parquet"),
        "oof_paper_scores": ("evaluate", "oof_paper_scores.parquet"),
    }
    result: Dict[str, Path] = {"future_citers": _future_citers_table(runtime)}
    for name, (stage, filename) in mapping.items():
        path = _require_stage(runtime, stage) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Canonical release artifact is missing: {path}")
        result[name] = path
    graph_root = _require_stage(runtime, "graphs")
    for path in sorted(
        item
        for item in graph_root.iterdir()
        if item.is_file()
        and item.name != "graph_snapshots.parquet"
        and (
            item.name.endswith(".nodes.parquet")
            or item.name.endswith(".edges.parquet")
            or item.name.endswith(".pairs.parquet")
            or item.name == "graph_snapshots_manifest.json"
        )
    ):
        result[f"graph_asset__{path.name}"] = path
    model_root = _require_stage(runtime, "evaluate") / "models"
    for horizon in (3, 5, 8):
        path = model_root / f"model_bundle_tau{horizon}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen scoring bundle is missing: {path}")
        result[f"model_bundle_tau{horizon}"] = path
    return result


def _collect_run_protocol(runtime: Runtime) -> Dict[str, Any]:
    stage_protocols: Dict[str, Any] = {}
    for stage in PIPELINE_STAGES:
        path = _require_stage(runtime, stage) / "stage_protocol.json"
        if not path.is_file():
            raise FileNotFoundError(f"Stage protocol is missing: {path}")
        payload = _read_json(path)
        expected_dependencies = {
            dependency: runtime.store.load_manifest(
                runtime.stage_identifier(dependency), dependency
            ).output_sha256
            for dependency in STAGE_DEPENDENCIES[stage]
        }
        if (
            payload.get("stage") != stage
            or payload.get("stage_identifier") != runtime.stage_identifier(stage)
            or payload.get("dependency_output_sha256")
            != expected_dependencies
        ):
            raise ValueError(f"Stage protocol identity mismatch for {stage}")
        stage_protocols[stage] = payload
    future_parameters = stage_protocols["future-citers"]["parameters"]
    target_parameters = stage_protocols["targets"]["parameters"]
    train_parameters = stage_protocols["train"]["parameters"]
    smoke_stages = sorted(
        stage
        for stage, payload in stage_protocols.items()
        if bool(payload.get("parameters", {}).get("smoke_test"))
    )
    limited_stages = {
        stage: payload.get("parameters", {}).get("max_papers")
        for stage, payload in stage_protocols.items()
        if payload.get("parameters", {}).get("max_papers") is not None
    }
    return {
        "schema_version": "1.0.0",
        "dataset_id": runtime.dataset_id,
        "analysis_id": runtime.analysis_id,
        "source_snapshot_id": runtime.source_snapshot_id,
        "config_hash": runtime.config_hash,
        "code_hash": runtime.code_hash,
        "dirty_diff_hash": runtime.dirty_diff_hash,
        "stage_identifiers": {
            stage: runtime.stage_identifier(stage) for stage in PIPELINE_STAGES
        },
        "resolved_contract": {
            "horizons": target_parameters.get("horizons"),
            "future_scope": future_parameters.get("future_scope"),
            "future_input_mode": future_parameters.get("future_input_mode"),
            "future_source_dir": future_parameters.get("future_source_dir"),
            "requested_horizon": future_parameters.get("requested_horizon"),
            "complete_observation_year": future_parameters.get(
                "complete_observation_year"
            ),
            "max_citers_per_work": future_parameters.get(
                "max_citers_per_work"
            ),
            "common_candidate_release": future_parameters.get(
                "common_candidate_release"
            ),
            "reuse_publication_dataset_id": runtime.reuse_publication_dataset_id,
            "sealed_holdout_unlocked": train_parameters.get(
                "sealed_holdout_unlocked"
            ),
            "smoke_stages": smoke_stages,
            "limited_stages": limited_stages,
        },
        "stages": stage_protocols,
    }


def _registries(
    runtime: Runtime,
    directory: Path,
    run_protocol: Mapping[str, Any],
) -> Dict[str, Path]:
    features = runtime.config.get("features", {})
    mechanisms = runtime.config.get("mechanisms", {})
    feature_entries = [
        {
            "name": name,
            "role": role,
            "definition_version": "nature-multihorizon-feature-v1",
            "source_max_year_column": "source_max_year",
            "coverage_column": "reference_metadata_coverage",
            "valid_pair_count_column": "valid_pair_count",
            "quality_flags_column": "quality_flags",
            "leakage_rule": "source_max_year < publication_year",
        }
        for role, names in (("core8", features.get("core8", [])), ("aux10", features.get("aux10", [])))
        for name in names
    ]
    registry_payloads = {
        "feature_registry": {
            "schema_version": "1.0.0",
            "core8": features.get("core8", []),
            "aux10": features.get("aux10", []),
            "features": feature_entries,
            "strict_source_rule": "source_max_year < publication_year",
        },
        "mechanism_registry": {"schema_version": "1.0.0", "mechanisms": mechanisms},
        "model_registry": {
            "schema_version": "1.0.0",
            "mechanism_model": "mechanism5_simplex_pairwise",
            "performance_candidates": ["gam18", "hgb18", "rank_blend"],
            "fixed_baselines": [
                "domain_year_only",
                "bibliographic_aux10_ridge",
                "mechanism5_equal_weight",
                "mechanism5_simplex",
            ],
            "frozen_bundles": ["model_bundle_tau3", "model_bundle_tau5", "model_bundle_tau8"],
        },
        "case_registry": {
            "schema_version": "1.0.0",
            "cases": runtime.config.get("case_studies", []),
        },
        "run_protocol": dict(run_protocol),
    }
    outputs: Dict[str, Path] = {}
    for name, payload in registry_payloads.items():
        path = directory / f"{name}.json"
        _write_json(path, payload)
        outputs[name] = path
    return outputs


def _build_quality(
    runtime: Runtime,
    artifacts: Mapping[str, Path],
    directory: Path,
    run_protocol: Mapping[str, Any],
) -> Path:
    table_names = (
        "papers",
        "future_fetch_status",
        "future_request_manifest",
        "features_raw",
        "targets",
        "cohort_membership",
        "split_membership",
        "oof_predictions",
        "sealed_holdout_predictions",
        "strict_label_holdout_predictions",
        "structural_subset",
        "structural_targets",
        "paper_scores",
        "model_ledger",
        "evaluation_metrics",
    )
    tables = {name: pd.read_parquet(artifacts[name]) for name in table_names}
    large_tables = (
        "paper_references",
        "reference_works",
        "reference_edges",
        "future_citers",
    )
    legacy_names = {
        "b_z",
        "rs_z",
        "deltaq0_z",
        "uzzi_z",
        "rtd_z",
        "burtip_z",
        "pde_z",
        "s_w",
    }
    legacy_columns: List[str] = []
    for name in large_tables:
        schema_names = pq.ParquetFile(artifacts[name]).schema_arrow.names
        legacy_columns.extend(
            f"{name}.{column}"
            for column in schema_names
            if str(column).casefold() in legacy_names
        )
        tables[name] = pd.DataFrame()
    recovery_manifest_path = (
        runtime.source_dir / "reference_closure_recovery" / "manifest.json"
    )
    recovery_manifest = (
        _read_json(recovery_manifest_path)
        if recovery_manifest_path.is_file()
        else {}
    )
    snapshot_manifest_path = (
        runtime.source_dir / "reference_closure_snapshot_manifest.json"
    )
    snapshot_manifest = (
        _read_json(snapshot_manifest_path)
        if snapshot_manifest_path.is_file()
        else {}
    )
    future_import_path = (
        _require_stage(runtime, "future-citers") / "future_import_audit.json"
    )
    future_import = (
        _read_json(future_import_path) if future_import_path.is_file() else {}
    )
    report = audit_pipeline_tables(
        tables,
        quality_config=runtime.config.get("quality", {}),
        prevalidated={
            "primary_key_tables": large_tables,
            "legacy_columns": legacy_columns,
            "global_reference_metadata_coverage": recovery_manifest.get(
                "coverage",
                snapshot_manifest.get("local_snapshot_coverage"),
            ),
            "future_citer_year_window": True,
            "prevalidation_basis": (
                "immutable ingest stage with SQLite exact-key materialization"
            ),
        },
    )
    report["source_future_multihorizon"] = {
        "overall_pass": future_import.get("source_overall_pass"),
        "accepted_for_training": future_import.get("accepted_for_training"),
        "source_failure_count": future_import.get("source_failure_count"),
        "acceptance_policy": future_import.get("acceptance_policy"),
    }
    if "structural_audit" in artifacts:
        report["structural_validation"] = _read_json(artifacts["structural_audit"])
        report["claim_scope"] = (
            "future knowledge adoption and diffusion only"
            if report["structural_validation"].get("targets", {}).get("go_for_confirmatory") is not True
            else "future diffusion plus confirmatory structural perturbation"
        )
    resolved = dict(run_protocol.get("resolved_contract", {}))
    expected_batches = (
        {
            "common_tau8_le2017",
            "recent_tau5_2018_2020",
            "recent_tau3_2021_2022",
        }
        if resolved.get("future_scope") == "expanded"
        else {"common_tau8_le2017"}
    )
    request_manifest = tables.get("future_request_manifest", pd.DataFrame())
    observed_batches = (
        set(
            request_manifest["request_batch"]
            .dropna()
            .astype(str)
            .unique()
        )
        if "request_batch" in request_manifest
        else set()
    )
    protocol_rows = (
        (
            "protocol_horizons_3_5_8",
            sorted(int(value) for value in (resolved.get("horizons") or []))
            == [3, 5, 8],
            resolved.get("horizons"),
            [3, 5, 8],
        ),
        (
            "protocol_requested_horizon_tau8",
            int(resolved.get("requested_horizon") or -1) == 8,
            resolved.get("requested_horizon"),
            8,
        ),
        (
            "protocol_future_citer_cap_1000",
            int(resolved.get("max_citers_per_work") or -1) == 1_000,
            resolved.get("max_citers_per_work"),
            1_000,
        ),
        (
            "protocol_no_smoke_stages",
            not resolved.get("smoke_stages"),
            resolved.get("smoke_stages"),
            [],
        ),
        (
            "protocol_no_limited_max_papers_stages",
            not resolved.get("limited_stages"),
            resolved.get("limited_stages"),
            {},
        ),
        (
            "protocol_expanded_future_scope",
            resolved.get("future_scope") == "expanded",
            resolved.get("future_scope"),
            "expanded",
        ),
        (
            "protocol_sealed_holdout_unlocked",
            resolved.get("sealed_holdout_unlocked") is True,
            resolved.get("sealed_holdout_unlocked"),
            True,
        ),
        (
            "protocol_common_candidate_bound",
            isinstance(resolved.get("common_candidate_release"), Mapping),
            resolved.get("common_candidate_release"),
            "audited candidate path + sha256",
        ),
        (
            "protocol_future_request_batches",
            observed_batches == expected_batches,
            sorted(observed_batches),
            sorted(expected_batches),
        ),
    )
    for name, passed, value, threshold in protocol_rows:
        report["checks"].append(
            {
                "name": name,
                "status": "pass" if passed else "fail",
                "blocking": True,
                "value": value,
                "threshold": threshold,
            }
        )
    protocol_failures = [
        row for row in report["checks"] if row["status"] == "fail"
    ]
    training_protocol_names = {
        "protocol_horizons_3_5_8",
        "protocol_requested_horizon_tau8",
        "protocol_future_citer_cap_1000",
        "protocol_no_smoke_stages",
        "protocol_no_limited_max_papers_stages",
    }
    if any(
        row["status"] == "fail" and row["name"] in training_protocol_names
        for row in report["checks"]
    ):
        report["go_for_training"] = False
    report["go_for_frozen_release"] = not protocol_failures
    report["n_checks"] = len(report["checks"])
    report["n_failed"] = len(protocol_failures)
    path = directory / "quality_report.json"
    write_quality_report(path, report)
    return path


def _build_view_bundle(
    runtime: Runtime,
    artifacts: Mapping[str, Path],
    directory: Path,
    channel: str,
    analysis_id: Optional[str] = None,
) -> Path:
    draft = directory / "draft_release.json"
    _write_json(
        draft,
        {
            "analysis_id": analysis_id or runtime.analysis_id,
            "channel": channel,
            "artifacts": {
                name: {
                    "path": str(path),
                    "release_path": _release_artifact_path(name, path),
                }
                for name, path in artifacts.items()
            },
        },
    )
    view_root = directory / "figure_views"
    export_figure_views(draft, output_dir=view_root)
    return view_root


def _release_artifact_path(name: str, path: Path) -> str:
    """Return the stable in-release path used by manifests and view lineage."""

    if name.startswith("figure_views/"):
        return name
    if name == "graph_snapshots":
        return "graph_snapshots/graph_snapshots.parquet"
    if name.startswith("graph_asset__"):
        return f"graph_snapshots/{path.name}"
    if name.startswith("figure_evidence_asset__"):
        return f"figure_evidence/assets/{path.name}"
    return f"{name}{path.suffix}"


def _figure_evidence_sources(args: argparse.Namespace) -> Dict[str, Path]:
    """Load optional Wave-B/C tables from one explicit immutable directory."""

    directory_value = getattr(args, "figure_evidence_dir", None)
    if directory_value is None:
        return {}
    directory = Path(directory_value).expanduser().resolve()
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(
            f"--figure-evidence-dir must be a regular directory: {directory}"
        )
    sources: Dict[str, Path] = {}
    evidence_frames: Dict[str, pd.DataFrame] = {}
    for artifact_name in OPTIONAL_FIGURE_EVIDENCE:
        candidates = [
            directory / f"{artifact_name}.parquet",
            directory / f"{artifact_name}.csv",
        ]
        present = [path for path in candidates if path.is_file()]
        if len(present) > 1:
            raise ValueError(
                f"Figure evidence has both CSV and Parquet for {artifact_name}"
            )
        if not present:
            continue
        path = present[0]
        if path.is_symlink() or path.name.endswith(".tmp") or ".tmp-" in path.name:
            raise ValueError(f"Unsafe figure-evidence artifact: {path}")
        frame = (
            pd.read_parquet(path)
            if path.suffix == ".parquet"
            else pd.read_csv(path, low_memory=False)
        )
        if frame.empty:
            raise ValueError(f"Figure-evidence artifact is empty: {path}")
        sources[artifact_name] = path
        evidence_frames[artifact_name] = frame
    unexpected = [
        path.name
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".csv", ".parquet"}
        and path.stem not in OPTIONAL_FIGURE_EVIDENCE
    ]
    if unexpected:
        raise ValueError(
            f"Unknown figure-evidence table(s): {sorted(unexpected)}"
        )
    assets_dir = directory / "assets"
    asset_hashes = set()
    seen_names = set()
    if assets_dir.exists():
        if not assets_dir.is_dir() or assets_dir.is_symlink():
            raise ValueError("figure-evidence assets must be a regular directory")
        for path in sorted(assets_dir.rglob("*")):
            if path.is_dir():
                continue
            if (
                not path.is_file()
                or path.is_symlink()
                or path.name.endswith(".tmp")
                or ".tmp-" in path.name
            ):
                raise ValueError(f"Unsafe figure-evidence asset: {path}")
            if path.name in seen_names:
                raise ValueError(
                    f"Figure-evidence asset basenames must be unique: {path.name}"
                )
            seen_names.add(path.name)
            sources[f"figure_evidence_asset__{path.name}"] = path
            asset_hashes.add(hash_file(path))
    required_provenance = {
        "evidence_id",
        "source_artifact_sha256",
        "protocol_hash",
    }
    for artifact_name, frame in evidence_frames.items():
        missing = sorted(required_provenance - set(frame.columns))
        if missing:
            raise ValueError(
                f"{artifact_name} is missing provenance columns: {missing}"
            )
        declared_hashes = set(
            frame["source_artifact_sha256"].dropna().astype(str)
        ) | set(frame["protocol_hash"].dropna().astype(str))
        unbound = sorted(declared_hashes - asset_hashes)
        if unbound:
            raise ValueError(
                f"{artifact_name} declares hashes absent from assets/: {unbound}"
            )
    return sources


def _release_analysis_id(
    base_analysis_id: str,
    evidence_sources: Mapping[str, Path],
) -> Tuple[str, Dict[str, str]]:
    hashes = {
        name: hash_file(path) for name, path in sorted(evidence_sources.items())
    }
    if not hashes:
        return base_analysis_id, hashes
    fingerprint = hash_json(
        {"base_analysis_id": base_analysis_id, "figure_evidence": hashes}
    ).split(":")[-1][:12]
    return f"{base_analysis_id}-ev{fingerprint}", hashes


def _prepare_release_storage(runtime: Runtime) -> None:
    """Place large immutable releases externally while keeping a stable repo link."""
    stable = runtime.release_root
    if stable.exists() or stable.is_symlink():
        return
    external = runtime.store.root / "releases"
    external.mkdir(parents=True, exist_ok=True)
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.symlink_to(external, target_is_directory=True)


def _promote_explicit_candidate(args: argparse.Namespace) -> Dict[str, Any]:
    """Promote a candidate without recomputing its dataset/analysis identity."""

    if args.release is None:
        raise ValueError(
            "Frozen promotion requires --release <candidate/release.json>"
        )
    source = args.release.expanduser().resolve()
    candidate_path = source.parent if source.name == "release.json" else source
    audit = audit_release(candidate_path, verify_hashes=not args.fast)
    manifest = audit.require_ok()
    if manifest.channel is not ReleaseChannel.CANDIDATE:
        raise ValueError("Frozen promotion source must be a candidate release")
    candidate = load_release(
        candidate_path,
        require_frozen=False,
        verify_hashes=False,
    )
    validate_candidate_for_freeze(candidate)
    target = release_directory(
        candidate_path.parent.parent,
        manifest.analysis_id,
        ReleaseChannel.FROZEN,
    )
    if args.dry_run:
        return {
            "channel": "frozen",
            "analysis_id": manifest.analysis_id,
            "dataset_id": manifest.dataset_id,
            "candidate": str(candidate_path / "release.json"),
            "release": str(target / "release.json"),
            "dry_run": True,
        }
    loaded = freeze_candidate_path(candidate_path)
    return {
        "channel": "frozen",
        "analysis_id": loaded.manifest.analysis_id,
        "dataset_id": loaded.manifest.dataset_id,
        "release": str(loaded.path / "release.json"),
    }


def command_publish(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    channel = str(args.channel)
    if channel == "frozen":
        return _promote_explicit_candidate(args)
    evidence_sources = _figure_evidence_sources(args)
    release_analysis_id, evidence_hashes = _release_analysis_id(
        runtime.analysis_id, evidence_sources
    )
    if args.dry_run:
        return {
            "channel": channel,
            "analysis_id": release_analysis_id,
            "base_training_analysis_id": runtime.analysis_id,
            "dataset_id": runtime.dataset_id,
            "figure_evidence_hashes": evidence_hashes,
            "release": str(
                release_directory(
                    runtime.release_root, release_analysis_id, channel
                )
                / "release.json"
            ),
            "dry_run": True,
        }
    _prepare_release_storage(runtime)
    _audit_pipeline_identity(runtime)
    core = _canonical_artifacts(runtime)
    work_dir = (
        runtime.store.root
        / "release_staging"
        / f".{release_analysis_id}.building-{os.getpid()}"
    )
    if work_dir.exists():
        raise FileExistsError(f"Release staging already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    try:
        run_protocol = _collect_run_protocol(runtime)
        run_protocol["base_training_analysis_id"] = runtime.analysis_id
        run_protocol["analysis_id"] = release_analysis_id
        run_protocol["figure_evidence"] = {
            name: {
                "sha256": evidence_hashes[name],
                "release_path": _release_artifact_path(name, path),
            }
            for name, path in evidence_sources.items()
        }
        registries = _registries(runtime, work_dir, run_protocol)
        quality_path = _build_quality(
            runtime,
            core,
            work_dir,
            run_protocol,
        )
        published_core = {
            name: path
            for name, path in core.items()
            if name not in RAW_ONLY_RELEASE_ARTIFACTS
        }
        release_sources: Dict[str, Path] = {
            **published_core,
            **evidence_sources,
            **registries,
            "quality_report": quality_path,
        }
        view_root = _build_view_bundle(
            runtime,
            release_sources,
            work_dir,
            channel,
            analysis_id=release_analysis_id,
        )
        for path in sorted(item for item in view_root.rglob("*") if item.is_file()):
            relative = path.relative_to(work_dir).as_posix()
            release_sources[relative] = path

        artifact_paths: Dict[str, str] = {}
        for name, path in release_sources.items():
            artifact_paths[name] = _release_artifact_path(name, path)
        row_counts = {
            name: _row_count(path)
            for name, path in published_core.items()
            if path.suffix == ".parquet"
        }
        for name, path in evidence_sources.items():
            if name not in OPTIONAL_FIGURE_EVIDENCE:
                continue
            row_counts[name] = (
                _row_count(path)
                if path.suffix == ".parquet"
                else int(len(pd.read_csv(path, low_memory=False)))
            )
        primary_keys = {
            name: PRIMARY_KEYS.get(path.name, ())
            for name, path in published_core.items()
        }
        manifest = build_release_manifest(
            source_snapshot_id=runtime.source_snapshot_id,
            dataset_id=runtime.dataset_id,
            analysis_id=release_analysis_id,
            channel=channel,
            config_hash=runtime.config_hash,
            code_hash=runtime.code_hash,
            dirty_diff_hash=runtime.dirty_diff_hash,
            source_artifacts=release_sources,
            artifact_paths=artifact_paths,
            input_artifact_ids=tuple(
                runtime.store.load_manifest(
                    runtime.stage_identifier(stage), stage
                ).output_sha256
                for stage in PIPELINE_STAGES
            )
            + tuple(evidence_hashes[name] for name in sorted(evidence_hashes)),
            row_counts=row_counts,
            primary_keys=primary_keys,
        )
        loaded = publish_release(runtime.release_root, manifest, release_sources)
        return {
            "channel": channel,
            "release": str(loaded.path / "release.json"),
            "analysis_id": release_analysis_id,
            "base_training_analysis_id": runtime.analysis_id,
            "dataset_id": runtime.dataset_id,
            "figure_evidence_hashes": evidence_hashes,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def command_export_views(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    release_path = release_directory(runtime.release_root, runtime.analysis_id, args.channel)
    audit = audit_release(release_path, verify_hashes=not args.fast)
    manifest = audit.require_ok()
    view_files = [record.path for record in manifest.artifacts.values() if record.path.startswith("figure_views/")]
    if not view_files:
        raise RuntimeError("Published release contains no figure views; republish under a new analysis_id")
    return {"analysis_id": runtime.analysis_id, "channel": args.channel, "n_view_files": len(view_files), "release": str(release_path / "release.json")}


def command_audit_release(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    release_path = args.release.resolve() if args.release else release_directory(runtime.release_root, runtime.analysis_id, args.channel)
    audit = audit_release(release_path, verify_hashes=not args.fast)
    return {
        "path": str(audit.path),
        "ok": audit.ok,
        "errors": list(audit.errors),
        "unexpected_files": list(audit.unexpected_files),
        "analysis_id": audit.manifest.analysis_id if audit.manifest else None,
        "channel": audit.manifest.channel.value if audit.manifest else None,
    }


def _audit_explicit_release(args: argparse.Namespace) -> Dict[str, Any]:
    """Audit an explicit release without touching current source/runtime state."""

    if args.release is None:
        raise ValueError("--release is required for release-only audit")
    audit = audit_release(
        args.release.expanduser().resolve(), verify_hashes=not args.fast
    )
    return {
        "path": str(audit.path),
        "ok": audit.ok,
        "errors": list(audit.errors),
        "unexpected_files": list(audit.unexpected_files),
        "analysis_id": audit.manifest.analysis_id if audit.manifest else None,
        "dataset_id": audit.manifest.dataset_id if audit.manifest else None,
        "channel": audit.manifest.channel.value if audit.manifest else None,
    }


def _inspect_explicit_figure_views(args: argparse.Namespace) -> Dict[str, Any]:
    """Inspect published views from the supplied immutable release."""

    audited = _audit_explicit_release(args)
    if not audited["ok"]:
        raise RuntimeError("Explicit release failed integrity audit")
    root = Path(str(audited["path"]))
    manifest = audit_release(root, verify_hashes=False).require_ok()
    view_files = [
        record.path
        for record in manifest.artifacts.values()
        if record.path.startswith("figure_views/")
    ]
    if not view_files:
        raise RuntimeError("Published release contains no figure views")
    return {
        "analysis_id": manifest.analysis_id,
        "dataset_id": manifest.dataset_id,
        "channel": manifest.channel.value,
        "n_view_files": len(view_files),
        "release": str(root / "release.json"),
    }


COMMANDS: Dict[str, Callable[[Runtime, argparse.Namespace], Dict[str, Any]]] = {
    "audit-source": command_audit_source,
    "recover-v5-reference-closure": command_recover,
    "ingest-v5": command_ingest,
    "build-taxonomy": command_taxonomy,
    "import-future-multihorizon": command_import_future,
    "fetch-future-citers": command_future_citers,
    "build-graphs": command_graphs,
    "build-features": command_features,
    "build-targets": command_targets,
    "build-cohorts": command_cohorts,
    "build-structural-validation": command_structural,
    "make-splits": command_splits,
    "train": command_train,
    "evaluate": command_evaluate,
    "publish-release": command_publish,
    "export-figure-views": command_export_views,
    "audit-release": command_audit_release,
}
STAGE_COMMANDS: Dict[str, str] = {
    "ingest-v5": "ingest-v5",
    "taxonomy": "build-taxonomy",
    "future-citers": "import-future-multihorizon",
    "graphs": "build-graphs",
    "features": "build-features",
    "targets": "build-targets",
    "cohorts": "build-cohorts",
    "structural": "build-structural-validation",
    "splits": "make-splits",
    "train": "train",
    "evaluate": "evaluate",
}


def command_run(runtime: Runtime, args: argparse.Namespace) -> Dict[str, Any]:
    first = args.from_stage or PIPELINE_STAGES[0]
    last = args.to_stage or PIPELINE_STAGES[-1]
    if first not in PIPELINE_STAGES or last not in PIPELINE_STAGES:
        raise ValueError(f"Stages must be in {list(PIPELINE_STAGES)}")
    start = PIPELINE_STAGES.index(first)
    end = PIPELINE_STAGES.index(last)
    if start > end:
        raise ValueError("--from-stage must not occur after --to-stage")
    results: Dict[str, Any] = {}
    for stage in PIPELINE_STAGES[start : end + 1]:
        command_name = STAGE_COMMANDS[stage]
        if stage == "future-citers" and args.future_scope == "expanded":
            if not args.allow_network:
                raise RuntimeError(
                    "Expanded recent windows require --allow-network; the "
                    "common 3/5/8 input is offline by default"
                )
            command_name = "fetch-future-citers"
        results[stage] = COMMANDS[command_name](runtime, args)
    return {"dataset_id": runtime.dataset_id, "analysis_id": runtime.analysis_id, "stages": results}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nature Portfolio multi-horizon evidence pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--future-source-dir", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--analysis-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--from-stage", choices=PIPELINE_STAGES)
    parser.add_argument("--to-stage", choices=PIPELINE_STAGES)
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-snapshot-files", type=int)
    parser.add_argument("--requested-horizon", type=int, default=8)
    parser.add_argument(
        "--future-scope",
        choices=("common", "expanded"),
        default="common",
        help="common=<=2017 tau8; expanded also adds 2018-20 tau5 and 2021-22 tau3",
    )
    parser.add_argument("--common-candidate-release", type=Path)
    parser.add_argument("--reuse-publication-dataset-id")
    parser.add_argument("--complete-observation-year", type=int)
    parser.add_argument("--max-citers-per-work", type=int, default=1000)
    parser.add_argument("--max-papers", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--unlock-sealed-holdout", action="store_true")
    parser.add_argument("--deep-jsonl", action="store_true")
    parser.add_argument("--channel", choices=("candidate", "frozen"), default="candidate")
    parser.add_argument("--release", type=Path)
    parser.add_argument(
        "--figure-evidence-dir",
        type=Path,
        help=(
            "Optional directory of provenance-bound Wave-B/C CSV/Parquet "
            "tables; content hashes derive a new evidence-release analysis ID."
        ),
    )
    parser.add_argument("--fast", action="store_true", help="Skip expensive release content hashes during audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (*COMMANDS, "run"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--config", type=Path, default=argparse.SUPPRESS)
        command_parser.add_argument(
            "--future-source-dir", type=Path, default=argparse.SUPPRESS
        )
        command_parser.add_argument("--dataset-id", default=argparse.SUPPRESS)
        command_parser.add_argument("--analysis-id", default=argparse.SUPPRESS)
        command_parser.add_argument("--resume", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--retry-failed", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--from-stage", choices=PIPELINE_STAGES, default=argparse.SUPPRESS)
        command_parser.add_argument("--to-stage", choices=PIPELINE_STAGES, default=argparse.SUPPRESS)
        command_parser.add_argument("--horizons", nargs="+", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--workers", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--max-snapshot-files", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--requested-horizon", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument(
            "--future-scope",
            choices=("common", "expanded"),
            default=argparse.SUPPRESS,
        )
        command_parser.add_argument(
            "--common-candidate-release",
            type=Path,
            default=argparse.SUPPRESS,
        )
        command_parser.add_argument(
            "--reuse-publication-dataset-id",
            default=argparse.SUPPRESS,
        )
        command_parser.add_argument("--complete-observation-year", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--max-citers-per-work", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--max-papers", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--sleep-seconds", type=float, default=argparse.SUPPRESS)
        command_parser.add_argument("--timeout-seconds", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--max-retries", type=int, default=argparse.SUPPRESS)
        command_parser.add_argument("--allow-network", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument("--smoke-test", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument(
            "--unlock-sealed-holdout",
            action="store_true",
            default=argparse.SUPPRESS,
        )
        command_parser.add_argument("--deep-jsonl", action="store_true", default=argparse.SUPPRESS)
        command_parser.add_argument(
            "--figure-evidence-dir",
            type=Path,
            default=argparse.SUPPRESS,
        )
        command_parser.add_argument(
            "--channel",
            choices=("candidate", "frozen"),
            default=argparse.SUPPRESS,
        )
        command_parser.add_argument("--release", type=Path, default=argparse.SUPPRESS)
        command_parser.add_argument("--fast", action="store_true", default=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    runtime: Optional[Runtime] = None
    if args.command == "publish-release" and args.channel == "frozen":
        result = _promote_explicit_candidate(args)
    elif args.command == "audit-release" and args.release is not None:
        result = _audit_explicit_release(args)
    elif args.command == "export-figure-views" and args.release is not None:
        result = _inspect_explicit_figure_views(args)
    else:
        if args.command == "publish-release" and (
            not args.dataset_id or not args.analysis_id
        ):
            raise ValueError(
                "Candidate publication requires explicit --dataset-id and "
                "--analysis-id from the completed run"
            )
        runtime = build_runtime(args)
        if args.command == "run":
            result = command_run(runtime, args)
        else:
            result = COMMANDS[args.command](runtime, args)
    if runtime is not None:
        result.setdefault("dataset_id", runtime.dataset_id)
        result.setdefault("analysis_id", runtime.analysis_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
