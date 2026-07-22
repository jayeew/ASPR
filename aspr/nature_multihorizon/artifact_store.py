"""Immutable, resumable artifact storage for the multi-horizon pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel

from .contracts import ArtifactRecord, StageManifest, StageStatus


MANIFEST_FILENAME = "manifest.json"
SUCCESS_FILENAME = "_SUCCESS"


class ArtifactStoreError(RuntimeError):
    """Base exception for artifact storage failures."""


class ArtifactExistsError(ArtifactStoreError):
    """Raised when an immutable completed artifact would be overwritten."""


class IncompleteStageError(ArtifactStoreError):
    """Raised when an unfinished stage exists but resume was not requested."""


class ArtifactAuditError(ArtifactStoreError):
    """Raised when a completed stage fails integrity validation."""


def _json_default(value: Any) -> Any:
    """Convert supported contract values to canonical JSON values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value deterministically for hashing and manifests."""
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_bytes(value: bytes) -> str:
    """Return a namespaced SHA-256 digest."""
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def hash_json(value: Any) -> str:
    """Hash a JSON-serializable value using canonical encoding."""
    return hash_bytes(canonical_json_bytes(value))


def manifest_identity_hash(value: Any) -> str:
    """Bind a success marker to the complete immutable manifest identity."""

    return hash_json(value)


def hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a file and return its namespaced SHA-256 digest."""
    file_path = Path(path)
    if not file_path.is_file() or file_path.is_symlink():
        raise ArtifactStoreError(f"artifact is not a regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def hash_tree(path: Path) -> str:
    """Hash a directory from relative paths, sizes, and file digests."""
    root = Path(path)
    if not root.is_dir():
        raise ArtifactStoreError(f"artifact tree does not exist: {root}")
    records = []
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        if file_path.is_symlink():
            raise ArtifactStoreError(f"symbolic links are not valid artifacts: {file_path}")
        relative = file_path.relative_to(root).as_posix()
        records.append((relative, file_path.stat().st_size, hash_file(file_path)))
    return hash_json(records)


def aggregate_artifact_hash(artifacts: Mapping[str, ArtifactRecord]) -> str:
    """Hash an artifact inventory independently of dictionary insertion order."""
    payload = {
        name: record.model_dump(mode="json")
        for name, record in sorted(artifacts.items())
    }
    return hash_json(payload)


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace one JSON file in an already private staging area."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    payload = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_identifier(value: str, label: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ArtifactStoreError(f"{label} must be a non-empty path-safe identifier")
    return text


@dataclass(frozen=True)
class StageResult:
    """Location and validated manifest returned by a completed stage."""

    path: Path
    manifest: StageManifest


@dataclass
class StageHandle:
    """Writable handle yielded only inside an atomic stage context."""

    path: Path
    final_path: Path
    dataset_id: str
    stage_name: str
    resumed: bool
    input_artifact_ids: Tuple[str, ...] = ()
    source_snapshot_id: Optional[str] = None
    config_hash: Optional[str] = None
    code_hash: Optional[str] = None
    dirty_diff_hash: Optional[str] = None
    _row_counts: Dict[str, int] = field(default_factory=dict)
    _primary_keys: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    _result: Optional[StageResult] = None

    def artifact_path(self, relative_path: str) -> Path:
        """Resolve a safe artifact path inside the private staging directory."""
        record = ArtifactRecord(
            path=relative_path,
            sha256="sha256:" + "0" * 64,
            size_bytes=0,
        )
        resolved = self.path / record.path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def record_table(
        self,
        relative_path: str,
        row_count: int,
        primary_key: Sequence[str] = (),
    ) -> None:
        """Attach table-level row count and primary-key metadata."""
        normalized = ArtifactRecord(
            path=relative_path,
            sha256="sha256:" + "0" * 64,
            size_bytes=0,
        ).path
        if row_count < 0:
            raise ValueError("row_count cannot be negative")
        self._row_counts[normalized] = int(row_count)
        self._primary_keys[normalized] = tuple(str(item) for item in primary_key)

    @property
    def result(self) -> StageResult:
        """Return the finalized stage result after the context exits."""
        if self._result is None:
            raise ArtifactStoreError("stage has not completed")
        return self._result


@dataclass(frozen=True)
class StageAudit:
    """Non-throwing integrity audit result."""

    path: Path
    ok: bool
    errors: Tuple[str, ...]
    unexpected_files: Tuple[str, ...]
    manifest: Optional[StageManifest]

    def require_ok(self) -> StageManifest:
        """Return the manifest or raise a concise integrity exception."""
        if not self.ok or self.manifest is None:
            detail = "; ".join(self.errors) or "stage audit failed"
            raise ArtifactAuditError(detail)
        return self.manifest


class ArtifactStore:
    """Filesystem-backed immutable store with resumable atomic stages."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def stage_path(self, dataset_id: str, stage_name: str) -> Path:
        """Return the immutable completed path for a named dataset stage."""
        dataset = _validate_identifier(dataset_id, "dataset_id")
        stage = _validate_identifier(stage_name, "stage_name")
        return self.root / "datasets" / dataset / "stages" / stage

    def staging_path(self, dataset_id: str, stage_name: str) -> Path:
        """Return the stable path used to resume an interrupted stage."""
        final_path = self.stage_path(dataset_id, stage_name)
        return final_path.with_name(f".{final_path.name}.staging")

    @contextmanager
    def stage(
        self,
        dataset_id: str,
        stage_name: str,
        *,
        resume: bool = False,
        input_artifact_ids: Sequence[str] = (),
        source_snapshot_id: Optional[str] = None,
        config_hash: Optional[str] = None,
        code_hash: Optional[str] = None,
        dirty_diff_hash: Optional[str] = None,
    ) -> Iterator[StageHandle]:
        """Yield a private directory and publish it atomically on success.

        Exceptions leave the staging directory intact.  A later invocation must
        explicitly pass ``resume=True``; completed stage paths are immutable.
        """
        final_path = self.stage_path(dataset_id, stage_name)
        staging_path = self.staging_path(dataset_id, stage_name)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            raise ArtifactExistsError(f"completed stage already exists: {final_path}")

        existed = staging_path.exists()
        if existed and not resume:
            raise IncompleteStageError(
                f"incomplete stage exists; pass resume=True: {staging_path}"
            )
        if not existed:
            staging_path.mkdir()
        elif not staging_path.is_dir():
            raise IncompleteStageError(f"staging path is not a directory: {staging_path}")

        # A crash after metadata creation but before rename is safely resumed by
        # rebuilding these two internal files from the user artifacts.
        for internal_name in (MANIFEST_FILENAME, SUCCESS_FILENAME):
            internal_path = staging_path / internal_name
            if internal_path.exists():
                internal_path.unlink()

        lock_path = final_path.parent / f".{stage_name}.lock"
        with lock_path.open("a+b") as lock_handle:
            self._acquire_lock(lock_handle, lock_path)
            handle = StageHandle(
                path=staging_path,
                final_path=final_path,
                dataset_id=dataset_id,
                stage_name=stage_name,
                resumed=existed,
                input_artifact_ids=tuple(input_artifact_ids),
                source_snapshot_id=source_snapshot_id,
                config_hash=config_hash,
                code_hash=code_hash,
                dirty_diff_hash=dirty_diff_hash,
            )
            try:
                yield handle
                if final_path.exists():
                    raise ArtifactExistsError(
                        f"completed stage appeared concurrently: {final_path}"
                    )
                manifest = self._finalize(handle)
                os.rename(staging_path, final_path)
                handle._result = StageResult(path=final_path, manifest=manifest)
            finally:
                self._release_lock(lock_handle)

    def load_manifest(self, dataset_id: str, stage_name: str) -> StageManifest:
        """Load and audit a completed stage manifest."""
        return audit_stage(self.stage_path(dataset_id, stage_name)).require_ok()

    def _finalize(self, handle: StageHandle) -> StageManifest:
        artifacts = inventory_directory(
            handle.path,
            row_counts=handle._row_counts,
            primary_keys=handle._primary_keys,
        )
        manifest = StageManifest(
            dataset_id=handle.dataset_id,
            stage_name=handle.stage_name,
            source_snapshot_id=handle.source_snapshot_id,
            config_hash=handle.config_hash,
            code_hash=handle.code_hash,
            dirty_diff_hash=handle.dirty_diff_hash,
            stage_status=StageStatus.COMPLETE,
            input_artifact_ids=handle.input_artifact_ids,
            artifacts=artifacts,
            output_sha256=aggregate_artifact_hash(artifacts),
            row_counts=dict(handle._row_counts),
            primary_keys=dict(handle._primary_keys),
        )
        atomic_write_json(handle.path / MANIFEST_FILENAME, manifest)
        success_path = handle.path / SUCCESS_FILENAME
        with success_path.open("xb") as success_handle:
            success_handle.write(
                (manifest_identity_hash(manifest) + "\n").encode("ascii")
            )
            success_handle.flush()
            os.fsync(success_handle.fileno())
        return manifest

    @staticmethod
    def _acquire_lock(lock_handle: Any, lock_path: Path) -> None:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (ImportError, BlockingIOError) as exc:
            raise ArtifactStoreError(f"stage is locked by another process: {lock_path}") from exc

    @staticmethod
    def _release_lock(lock_handle: Any) -> None:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            return


def inventory_directory(
    root: Path,
    *,
    row_counts: Optional[Mapping[str, int]] = None,
    primary_keys: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, ArtifactRecord]:
    """Build a deterministic file inventory for a stage or release directory."""
    base = Path(root)
    counts = dict(row_counts or {})
    keys = {name: tuple(value) for name, value in (primary_keys or {}).items()}
    records: Dict[str, ArtifactRecord] = {}
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if relative in {MANIFEST_FILENAME, SUCCESS_FILENAME, "release.json"}:
            continue
        if path.is_symlink():
            raise ArtifactStoreError(f"symbolic links are not valid artifacts: {path}")
        records[relative] = ArtifactRecord(
            path=relative,
            sha256=hash_file(path),
            size_bytes=path.stat().st_size,
            row_count=counts.get(relative),
            primary_key=keys.get(relative, ()),
        )
    return records


def load_stage_manifest(path: Path) -> StageManifest:
    """Parse a stage manifest from a directory or explicit JSON path."""
    source = Path(path)
    manifest_path = source if source.name == MANIFEST_FILENAME else source / MANIFEST_FILENAME
    try:
        return StageManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactAuditError(f"invalid stage manifest {manifest_path}: {exc}") from exc


def audit_stage(path: Path, *, verify_hashes: bool = True) -> StageAudit:
    """Audit success marker, schema, paths, sizes, hashes, and extra files."""
    root = Path(path)
    errors = []
    manifest: Optional[StageManifest] = None
    if not root.is_dir():
        return StageAudit(root, False, ("stage directory does not exist",), (), None)
    if not (root / SUCCESS_FILENAME).is_file():
        errors.append("missing _SUCCESS marker")
    try:
        manifest = load_stage_manifest(root)
    except ArtifactAuditError as exc:
        errors.append(str(exc))

    expected = set()
    if manifest is not None:
        expected_dataset_root = root.parent.parent
        if (
            root.parent.name != "stages"
            or expected_dataset_root.name != manifest.dataset_id
            or root.name != manifest.stage_name
        ):
            errors.append(
                "stage directory namespace does not match dataset_id/stage_name"
            )
        for name, record in manifest.artifacts.items():
            expected.add(record.path)
            artifact_path = root / record.path
            if name != record.path:
                errors.append(f"artifact key/path mismatch: {name} != {record.path}")
            if not artifact_path.is_file() or artifact_path.is_symlink():
                errors.append(f"missing regular artifact: {record.path}")
                continue
            if artifact_path.stat().st_size != record.size_bytes:
                errors.append(f"size mismatch: {record.path}")
            if verify_hashes and hash_file(artifact_path) != record.sha256:
                errors.append(f"hash mismatch: {record.path}")
        if aggregate_artifact_hash(manifest.artifacts) != manifest.output_sha256:
            errors.append("aggregate output hash mismatch")
        success_path = root / SUCCESS_FILENAME
        if success_path.is_file():
            marker = success_path.read_text(encoding="ascii").strip()
            if marker != manifest_identity_hash(manifest):
                errors.append("_SUCCESS marker does not match manifest identity")

    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
        and item.relative_to(root).as_posix()
        not in {MANIFEST_FILENAME, SUCCESS_FILENAME}
    }
    unexpected = tuple(sorted(actual - expected))
    if unexpected:
        errors.append(f"unexpected files: {', '.join(unexpected)}")
    return StageAudit(root, not errors, tuple(errors), unexpected, manifest)


__all__ = [
    "ArtifactAuditError",
    "ArtifactExistsError",
    "ArtifactStore",
    "ArtifactStoreError",
    "IncompleteStageError",
    "MANIFEST_FILENAME",
    "SUCCESS_FILENAME",
    "StageAudit",
    "StageHandle",
    "StageResult",
    "aggregate_artifact_hash",
    "atomic_write_json",
    "audit_stage",
    "canonical_json_bytes",
    "hash_bytes",
    "hash_file",
    "hash_json",
    "hash_tree",
    "inventory_directory",
    "load_stage_manifest",
    "manifest_identity_hash",
]
