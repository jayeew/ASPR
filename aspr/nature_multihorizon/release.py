"""Publishing, promotion, loading, and auditing of evidence releases."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from .artifact_store import (
    SUCCESS_FILENAME,
    ArtifactAuditError,
    ArtifactExistsError,
    aggregate_artifact_hash,
    atomic_write_json,
    hash_file,
    manifest_identity_hash,
)
from .contracts import (
    ArtifactRecord,
    ReleaseChannel,
    ReleaseManifest,
    StageStatus,
)


RELEASE_FILENAME = "release.json"
REQUIRED_FROZEN_ARTIFACTS = frozenset(
    {
        "papers",
        "features_raw",
        "graph_snapshots",
        "targets",
        "cohort_membership",
        "split_membership",
        "oof_predictions",
        "sealed_holdout_predictions",
        "strict_label_holdout_predictions",
        "evaluation_metrics",
        "paper_scores",
        "oof_paper_scores",
        "model_ledger",
        "feature_registry",
        "mechanism_registry",
        "model_registry",
        "run_protocol",
        "case_registry",
        "model_bundle_tau3",
        "model_bundle_tau5",
        "model_bundle_tau8",
        "quality_report",
        "structural_subset",
        "structural_targets",
        "future_request_manifest",
    }
)


class ReleaseError(RuntimeError):
    """Base exception for release publication and loading failures."""


class ReleaseAuditError(ReleaseError):
    """Raised when an evidence release is incomplete or has changed."""


@dataclass(frozen=True)
class ReleaseAudit:
    """Non-throwing release integrity report."""

    path: Path
    ok: bool
    errors: Tuple[str, ...]
    unexpected_files: Tuple[str, ...]
    manifest: Optional[ReleaseManifest]

    def require_ok(self) -> ReleaseManifest:
        """Return the manifest or raise when any integrity check failed."""
        if not self.ok or self.manifest is None:
            detail = "; ".join(self.errors) or "release audit failed"
            raise ReleaseAuditError(detail)
        return self.manifest


@dataclass(frozen=True)
class LoadedRelease:
    """Audited release with safe artifact lookup."""

    path: Path
    manifest: ReleaseManifest

    def artifact(self, name: str) -> Path:
        """Resolve a named artifact declared by the release manifest."""
        try:
            record = self.manifest.artifacts[name]
        except KeyError as exc:
            raise KeyError(f"release does not contain artifact {name!r}") from exc
        return self.path / record.path


def release_directory(
    root: Path,
    analysis_id: str,
    channel: Union[ReleaseChannel, str],
) -> Path:
    """Resolve candidate and frozen releases into disjoint immutable roots."""
    identifier = str(analysis_id).strip()
    if not identifier or identifier in {".", ".."} or any(
        separator in identifier for separator in ("/", "\\")
    ):
        raise ReleaseError("analysis_id must be a non-empty path-safe identifier")
    normalized = ReleaseChannel(channel)
    collection = "candidates" if normalized is ReleaseChannel.CANDIDATE else "analyses"
    return Path(root) / collection / identifier


def build_release_manifest(
    *,
    source_snapshot_id: str,
    dataset_id: str,
    analysis_id: str,
    channel: Union[ReleaseChannel, str],
    config_hash: str,
    code_hash: str,
    dirty_diff_hash: str,
    source_artifacts: Mapping[str, Path],
    artifact_paths: Optional[Mapping[str, str]] = None,
    input_artifact_ids: Sequence[str] = (),
    row_counts: Optional[Mapping[str, int]] = None,
    primary_keys: Optional[Mapping[str, Sequence[str]]] = None,
) -> ReleaseManifest:
    """Create a release manifest from source files without copying them."""
    paths = dict(artifact_paths or {})
    counts = dict(row_counts or {})
    keys = {name: tuple(value) for name, value in (primary_keys or {}).items()}
    records: Dict[str, ArtifactRecord] = {}
    for name, source in sorted(source_artifacts.items()):
        source_path = Path(source)
        relative_path = paths.get(name, name)
        if relative_path in {RELEASE_FILENAME, SUCCESS_FILENAME}:
            raise ReleaseError(f"reserved release path: {relative_path}")
        if not source_path.is_file() or source_path.is_symlink():
            raise ReleaseError(f"release source is not a regular file: {source_path}")
        records[name] = ArtifactRecord(
            path=relative_path,
            sha256=hash_file(source_path),
            size_bytes=source_path.stat().st_size,
            row_count=counts.get(name),
            primary_key=keys.get(name, ()),
        )
    return ReleaseManifest(
        source_snapshot_id=source_snapshot_id,
        dataset_id=dataset_id,
        analysis_id=analysis_id,
        channel=ReleaseChannel(channel),
        config_hash=config_hash,
        code_hash=code_hash,
        dirty_diff_hash=dirty_diff_hash,
        input_artifact_ids=tuple(input_artifact_ids),
        artifacts=records,
        output_sha256=aggregate_artifact_hash(records),
        row_counts={name: int(value) for name, value in counts.items()},
        primary_keys=keys,
        stage_status=StageStatus.COMPLETE,
    )


def publish_release(
    root: Path,
    manifest: ReleaseManifest,
    source_artifacts: Mapping[str, Path],
) -> LoadedRelease:
    """Copy and atomically publish an immutable candidate or frozen release."""
    destination = release_directory(root, manifest.analysis_id, manifest.channel)
    if destination.exists():
        raise ArtifactExistsError(f"release already exists and is immutable: {destination}")
    if set(source_artifacts) != set(manifest.artifacts):
        missing = sorted(set(manifest.artifacts) - set(source_artifacts))
        extra = sorted(set(source_artifacts) - set(manifest.artifacts))
        raise ReleaseError(f"source/manifest artifact mismatch; missing={missing}, extra={extra}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.publishing-{os.getpid()}")
    if staging.exists():
        raise ReleaseError(f"release staging path already exists: {staging}")
    staging.mkdir()
    try:
        for name, record in manifest.artifacts.items():
            source = Path(source_artifacts[name])
            _verify_source(source, record)
            output = staging / record.path
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
            _verify_source(output, record)

        atomic_write_json(staging / RELEASE_FILENAME, manifest)
        success = staging / SUCCESS_FILENAME
        with success.open("xb") as handle:
            handle.write((manifest_identity_hash(manifest) + "\n").encode("ascii"))
            handle.flush()
            os.fsync(handle.fileno())

        audit_release(
            staging,
            verify_hashes=True,
            allow_publish_staging=True,
        ).require_ok()
        if destination.exists():
            raise ArtifactExistsError(
                f"release appeared concurrently and cannot be overwritten: {destination}"
            )
        os.rename(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_release(destination, require_frozen=manifest.channel is ReleaseChannel.FROZEN)


def validate_candidate_for_freeze(candidate: LoadedRelease) -> None:
    """Run every read-only frozen-promotion gate against one candidate."""

    if candidate.manifest.channel is not ReleaseChannel.CANDIDATE:
        raise ReleaseError("source release is not a candidate")
    missing = sorted(REQUIRED_FROZEN_ARTIFACTS - set(candidate.manifest.artifacts))
    if missing:
        raise ReleaseError(f"candidate is missing required frozen artifacts: {missing}")
    quality_path = candidate.path / candidate.manifest.artifacts["quality_report"].path
    try:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"candidate quality report is invalid: {exc}") from exc
    if not isinstance(quality, dict) or quality.get("go_for_frozen_release") is not True:
        raise ReleaseError("candidate quality report does not permit frozen promotion")
    protocol_path = candidate.path / candidate.manifest.artifacts["run_protocol"].path
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"candidate run protocol is invalid: {exc}") from exc
    if (
        not isinstance(protocol, dict)
        or protocol.get("dataset_id") != candidate.manifest.dataset_id
        or protocol.get("analysis_id") != candidate.manifest.analysis_id
        or protocol.get("source_snapshot_id")
        != candidate.manifest.source_snapshot_id
    ):
        raise ReleaseError("candidate run protocol does not match release identity")
    graph_catalog_path = candidate.artifact("graph_snapshots")
    try:
        import pandas as pd

        graph_catalog = pd.read_parquet(graph_catalog_path)
    except Exception as exc:
        raise ReleaseError(f"graph snapshot catalog is invalid: {exc}") from exc
    declared_paths = {
        record.path for record in candidate.manifest.artifacts.values()
    }
    catalog_parent = Path(
        candidate.manifest.artifacts["graph_snapshots"].path
    ).parent
    for column in ("node_path", "edge_path", "pair_path"):
        if column not in graph_catalog:
            raise ReleaseError(f"graph snapshot catalog is missing {column}")
        for value in graph_catalog[column].dropna().astype(str):
            relative = (catalog_parent / value).as_posix()
            if (
                relative not in declared_paths
                or not (candidate.path / relative).is_file()
            ):
                raise ReleaseError(
                    f"graph snapshot asset is not release-bound: {relative}"
                )
    artifact_paths = {
        record.path for record in candidate.manifest.artifacts.values()
    }
    missing_views = []
    for index in range(1, 11):
        figure_id = f"fig{index:02d}"
        prefix = f"figure_views/{figure_id}/"
        required = {
            prefix + "_SUCCESS",
            prefix + "view_manifest.json",
            prefix + "panel_spec.json",
            prefix + "caption_stats.json",
        }
        has_data = any(
            path.startswith(prefix + "data/") and path.endswith(".csv")
            for path in artifact_paths
        )
        if not required.issubset(artifact_paths) or not has_data:
            missing_views.append(figure_id)
            continue
        for filename in ("view_manifest.json", "panel_spec.json"):
            payload = json.loads(
                (candidate.path / (prefix + filename)).read_text(
                    encoding="utf-8"
                )
            )
            if (
                str(payload.get("analysis_id") or "")
                != candidate.manifest.analysis_id
                or str(payload.get("figure_id") or "") != figure_id
            ):
                raise ReleaseError(
                    f"{figure_id} {filename} does not match the release identity"
                )
    if missing_views:
        raise ReleaseError(f"candidate is missing figure views: {missing_views}")


def _freeze_loaded_candidate(root: Path, candidate: LoadedRelease) -> LoadedRelease:
    """Promote one fully validated candidate under its exact release root."""

    validate_candidate_for_freeze(candidate)
    frozen_manifest = candidate.manifest.model_copy(
        update={
            "channel": ReleaseChannel.FROZEN,
            "created_at": datetime.now(timezone.utc),
        }
    )
    sources = {
        name: candidate.path / record.path
        for name, record in candidate.manifest.artifacts.items()
    }
    return publish_release(root, frozen_manifest, sources)


def freeze_candidate(root: Path, analysis_id: str) -> LoadedRelease:
    """Promote an audited candidate selected by root and analysis ID."""

    candidate_path = release_directory(root, analysis_id, ReleaseChannel.CANDIDATE)
    candidate = load_release(candidate_path, require_frozen=False)
    return _freeze_loaded_candidate(Path(root), candidate)


def freeze_candidate_path(path: Path) -> LoadedRelease:
    """Promote only the explicitly supplied candidate release path."""

    source = Path(path).expanduser().resolve()
    candidate_path = source.parent if source.name == RELEASE_FILENAME else source
    candidate = load_release(candidate_path, require_frozen=False)
    release_root = candidate_path.parent.parent
    expected = release_directory(
        release_root,
        candidate.manifest.analysis_id,
        ReleaseChannel.CANDIDATE,
    ).resolve()
    if candidate_path.resolve() != expected:
        raise ReleaseError(
            "explicit candidate path is outside its manifest namespace"
        )
    return _freeze_loaded_candidate(release_root, candidate)


def load_release_manifest(path: Path) -> ReleaseManifest:
    """Parse a release manifest from a directory or explicit JSON path."""
    source = Path(path)
    manifest_path = source if source.name == RELEASE_FILENAME else source / RELEASE_FILENAME
    try:
        return ReleaseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReleaseAuditError(f"invalid release manifest {manifest_path}: {exc}") from exc


def audit_release(
    path: Path,
    *,
    verify_hashes: bool = True,
    allow_publish_staging: bool = False,
) -> ReleaseAudit:
    """Validate a release marker, schema, inventory, sizes, and hashes."""
    source = Path(path)
    root = source.parent if source.name == RELEASE_FILENAME else source
    errors = []
    manifest: Optional[ReleaseManifest] = None
    if not root.is_dir():
        return ReleaseAudit(root, False, ("release directory does not exist",), (), None)
    if not (root / SUCCESS_FILENAME).is_file():
        errors.append("missing _SUCCESS marker")
    try:
        manifest = load_release_manifest(root)
    except ReleaseAuditError as exc:
        errors.append(str(exc))

    expected = set()
    if manifest is not None:
        expected_collection = (
            "candidates"
            if manifest.channel is ReleaseChannel.CANDIDATE
            else "analyses"
        )
        is_final_name = root.name == manifest.analysis_id
        is_publish_staging = root.name.startswith(
            f".{manifest.analysis_id}.publishing-"
        )
        valid_name = is_final_name or (
            allow_publish_staging and is_publish_staging
        )
        if root.parent.name != expected_collection or not valid_name:
            errors.append(
                "release directory namespace does not match channel/analysis_id"
            )
        seen_paths = set()
        for name, record in manifest.artifacts.items():
            if record.path in seen_paths:
                errors.append(f"duplicate artifact path: {record.path}")
            seen_paths.add(record.path)
            expected.add(record.path)
            artifact_path = root / record.path
            if not artifact_path.is_file() or artifact_path.is_symlink():
                errors.append(f"missing regular artifact {name}: {record.path}")
                continue
            if artifact_path.stat().st_size != record.size_bytes:
                errors.append(f"size mismatch {name}: {record.path}")
            if verify_hashes and hash_file(artifact_path) != record.sha256:
                errors.append(f"hash mismatch {name}: {record.path}")
        if aggregate_artifact_hash(manifest.artifacts) != manifest.output_sha256:
            errors.append("aggregate output hash mismatch")
        success = root / SUCCESS_FILENAME
        if success.is_file():
            marker = success.read_text(encoding="ascii").strip()
            if marker != manifest_identity_hash(manifest):
                errors.append("_SUCCESS marker does not match manifest identity")

    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix()
        not in {RELEASE_FILENAME, SUCCESS_FILENAME}
    }
    unexpected = tuple(sorted(actual - expected))
    if unexpected:
        errors.append(f"unexpected files: {', '.join(unexpected)}")
    return ReleaseAudit(root, not errors, tuple(errors), unexpected, manifest)


def load_release(
    path: Path,
    *,
    require_frozen: bool = False,
    verify_hashes: bool = True,
) -> LoadedRelease:
    """Load only after a complete integrity audit, optionally requiring frozen."""
    audit = audit_release(path, verify_hashes=verify_hashes)
    manifest = audit.require_ok()
    if require_frozen and manifest.channel is not ReleaseChannel.FROZEN:
        raise ReleaseAuditError("a frozen release is required for this consumer")
    return LoadedRelease(path=audit.path, manifest=manifest)


def _verify_source(path: Path, record: ArtifactRecord) -> None:
    if not path.is_file() or path.is_symlink():
        raise ReleaseError(f"release artifact is not a regular file: {path}")
    if path.stat().st_size != record.size_bytes:
        raise ReleaseError(f"release artifact size changed: {path}")
    if hash_file(path) != record.sha256:
        raise ReleaseError(f"release artifact hash changed: {path}")


__all__ = [
    "LoadedRelease",
    "RELEASE_FILENAME",
    "ReleaseAudit",
    "ReleaseAuditError",
    "ReleaseError",
    "audit_release",
    "build_release_manifest",
    "freeze_candidate",
    "freeze_candidate_path",
    "load_release",
    "load_release_manifest",
    "publish_release",
    "release_directory",
    "validate_candidate_for_freeze",
]
