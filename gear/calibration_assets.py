"""Versioned local release registry for pre-publication graph calibration."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional

from pydantic import Field

from .contracts import StrictModel

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback.
    fcntl = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "gear" / "calibration_registry.json"
RELEASE_DATA_ROOT = PROJECT_ROOT / "data" / "calibration" / "releases"
RELEASE_MANIFEST_ROOT = PROJECT_ROOT / "configs" / "gear" / "calibration_releases"
RELEASE_ALIAS = "prepublication_graph_v3:d5_fulltext16"

CORE_ASSET_NAMES = (
    "official_run_manifest",
    "official_model_json",
    "official_model_joblib",
    "official_score_table",
    "feature_matrix_16",
    "matrix_manifest",
    "matrix_input_snapshot",
    "paper_metadata",
    "oof_metrics",
    "oof_fold_metrics",
    "oof_domain_metrics",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class ReleaseAsset(StrictModel):
    file: str
    sha256: str
    size_bytes: int = Field(ge=0)
    role: Literal["runtime", "evaluation"] = "runtime"


class CalibrationReleaseManifest(StrictModel):
    contract: Literal["aspr_calibration_release_v1"] = "aspr_calibration_release_v1"
    release_id: str
    alias: Literal["prepublication_graph_v3:d5_fulltext16"] = (
        "prepublication_graph_v3:d5_fulltext16"
    )
    status: Literal["frozen"] = "frozen"
    evidence_policy: Literal["fig1_fig2_fig3_current_only"] = (
        "fig1_fig2_fig3_current_only"
    )
    deprecated_fig4_to_fig10_used: Literal[False] = False
    row_count: Literal[411490] = 411_490
    created_at_utc: datetime
    source_manifest_sha256: str
    replay: Dict[str, object]
    assets: Dict[str, ReleaseAsset]


class ReleaseRegistryEntry(StrictModel):
    manifest: str
    manifest_sha256: str
    asset_root: str


class CalibrationRegistry(StrictModel):
    contract: Literal["aspr_calibration_registry_v1"] = "aspr_calibration_registry_v1"
    active: Dict[str, str]
    releases: Dict[str, ReleaseRegistryEntry]


class LoadedCalibrationRelease:
    """Resolved, immutable view of one locally installed release."""

    def __init__(
        self,
        manifest: CalibrationReleaseManifest,
        manifest_path: Path,
        asset_root: Path,
    ) -> None:
        self.manifest = manifest
        self.manifest_path = manifest_path
        self.asset_root = asset_root

    @property
    def release_id(self) -> str:
        return self.manifest.release_id

    def path(self, asset_name: str, *, verify: bool = False) -> Path:
        if asset_name not in self.manifest.assets:
            raise KeyError(f"release has no asset named {asset_name!r}")
        asset = self.manifest.assets[asset_name]
        path = (self.asset_root / asset.file).resolve()
        _require_within(path, self.asset_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != asset.size_bytes:
            raise ValueError(f"calibration asset size mismatch: {asset_name}")
        if verify and sha256_file(path) != asset.sha256:
            raise ValueError(f"calibration asset hash mismatch: {asset_name}")
        return path

    def verify(self) -> None:
        missing = sorted(set(CORE_ASSET_NAMES) - set(self.manifest.assets))
        if missing:
            raise ValueError(f"calibration release lacks core assets: {missing}")
        for name in self.manifest.assets:
            self.path(name, verify=True)

    def core_paths(self) -> Dict[str, Path]:
        return {name: self.path(name) for name in CORE_ASSET_NAMES}


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _relative_project_path(path: Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def _require_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"unsafe calibration release path: {path}") from exc


def load_calibration_release(
    identifier: str = RELEASE_ALIAS,
    *,
    registry_path: Optional[Path] = None,
    verify: bool = False,
) -> LoadedCalibrationRelease:
    """Resolve an alias or immutable release ID through the local registry."""
    registry_file = Path(registry_path or DEFAULT_REGISTRY).resolve()
    registry = CalibrationRegistry.model_validate_json(
        registry_file.read_text(encoding="utf-8")
    )
    release_id = registry.active.get(identifier, identifier)
    entry = registry.releases.get(release_id)
    if entry is None:
        raise KeyError(f"unknown calibration release: {identifier}")
    manifest_path = _resolve_project_path(entry.manifest)
    _require_within(manifest_path, RELEASE_MANIFEST_ROOT)
    if sha256_file(manifest_path) != entry.manifest_sha256:
        raise ValueError(f"calibration release manifest hash mismatch: {release_id}")
    manifest = CalibrationReleaseManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.release_id != release_id:
        raise ValueError("calibration registry and manifest release IDs differ")
    asset_root = _resolve_project_path(entry.asset_root)
    _require_within(asset_root, RELEASE_DATA_ROOT)
    loaded = LoadedCalibrationRelease(manifest, manifest_path, asset_root)
    if verify:
        loaded.verify()
    return loaded


def _clone_or_copy(source: Path, target: Path) -> None:
    """Prefer copy-on-write cloning so a release cannot mutate with its source."""
    if fcntl is not None:
        try:
            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                fcntl.ioctl(target_handle.fileno(), 0x40049409, source_handle.fileno())
            shutil.copystat(source, target)
            return
        except OSError:
            target.unlink(missing_ok=True)
    shutil.copy2(source, target)


def promote_calibration_release(
    *,
    release_id: str,
    source_assets: Mapping[str, Path],
    replay: Mapping[str, object],
    source_manifest_sha256: str,
    registry_path: Optional[Path] = None,
) -> LoadedCalibrationRelease:
    """Atomically install and register an already validated candidate release."""
    if not release_id or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in release_id
    ):
        raise ValueError(
            "release_id may contain only lowercase letters, digits, '-' and '_'"
        )
    missing = sorted(set(CORE_ASSET_NAMES) - set(source_assets))
    if missing:
        raise ValueError(f"promotion source lacks core assets: {missing}")
    replay_row_count = replay.get("row_count")
    if not bool(replay.get("passed")) or replay_row_count != 411_490:
        raise ValueError("calibration release promotion requires a passing full replay")
    registry_file = Path(registry_path or DEFAULT_REGISTRY).resolve()
    release_root = (RELEASE_DATA_ROOT / release_id).resolve()
    manifest_path = (RELEASE_MANIFEST_ROOT / f"{release_id}.json").resolve()
    if release_root.exists() or manifest_path.exists():
        raise FileExistsError(f"calibration release already exists: {release_id}")
    RELEASE_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RELEASE_MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{release_id}-", dir=RELEASE_DATA_ROOT))
    try:
        assets: Dict[str, ReleaseAsset] = {}
        used_names: set[str] = set()
        for logical_name, raw_source in sorted(source_assets.items()):
            source = Path(raw_source).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            filename = source.name
            if filename in used_names:
                filename = f"{logical_name}-{filename}"
            used_names.add(filename)
            target = staged / filename
            _clone_or_copy(source, target)
            assets[logical_name] = ReleaseAsset(
                file=filename,
                sha256=sha256_file(target),
                size_bytes=target.stat().st_size,
                role="evaluation" if logical_name == "oof_predictions" else "runtime",
            )
        manifest = CalibrationReleaseManifest(
            release_id=release_id,
            created_at_utc=datetime.now(timezone.utc),
            source_manifest_sha256=source_manifest_sha256,
            replay=dict(replay),
            assets=assets,
        )
        staged.rename(release_root)
        manifest_path.write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        registry = (
            CalibrationRegistry.model_validate_json(
                registry_file.read_text(encoding="utf-8")
            )
            if registry_file.is_file()
            else CalibrationRegistry(active={}, releases={})
        )
        releases = dict(registry.releases)
        releases[release_id] = ReleaseRegistryEntry(
            manifest=_relative_project_path(manifest_path),
            manifest_sha256=sha256_file(manifest_path),
            asset_root=_relative_project_path(release_root),
        )
        active = dict(registry.active)
        active[RELEASE_ALIAS] = release_id
        updated = CalibrationRegistry(active=active, releases=releases)
        registry_file.write_text(
            updated.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        if release_root.exists():
            shutil.rmtree(release_root)
        manifest_path.unlink(missing_ok=True)
        raise
    return load_calibration_release(
        release_id, registry_path=registry_file, verify=True
    )


__all__ = [
    "CORE_ASSET_NAMES",
    "CalibrationReleaseManifest",
    "LoadedCalibrationRelease",
    "RELEASE_ALIAS",
    "load_calibration_release",
    "promote_calibration_release",
]
