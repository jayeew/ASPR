"""Read-only identity and availability audit for frozen local v6 sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .offline import validate_local_only_config


class LocalSourceSpec(BaseModel):
    """One immutable local raw asset required or optionally reused by v6."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    path: str = Field(min_length=1)
    role: str = Field(min_length=1)
    required: bool = True
    required_files: tuple[str, ...] = ()
    required_directories: tuple[str, ...] = ()
    identity_files: tuple[str, ...] = ()
    allow_symlink: bool = True

    @model_validator(mode="after")
    def validate_relative_members(self) -> "LocalSourceSpec":
        for member in (
            self.required_files + self.required_directories + self.identity_files
        ):
            candidate = Path(member)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("source member paths must remain inside the asset")
        return self


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _directory_inventory(path: Path) -> Dict[str, Any]:
    """Fingerprint names and sizes without hashing an entire multi-TB snapshot."""
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for member in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = member.stat()
        relative = member.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\n")
        file_count += 1
        total_bytes += int(stat.st_size)
    return {
        "relative_path": path.name,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "inventory_sha256": f"sha256:{digest.hexdigest()}",
        "identity_strength": "path_and_size_inventory; selected extracted rows require content hashes",
    }


def _resolve_source_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (project_root / path).absolute()


def audit_local_source(
    spec: LocalSourceSpec,
    *,
    project_root: Path,
    deep_hash: bool = False,
) -> Dict[str, Any]:
    """Audit one source without creating or modifying any files."""
    configured = _resolve_source_path(spec.path, project_root)
    is_symlink = configured.is_symlink()
    resolved = configured.resolve(strict=False)
    exists = configured.exists()
    blockers: List[str] = []
    if is_symlink and not spec.allow_symlink:
        blockers.append("symlink_not_allowed")
    if not exists:
        blockers.append("configured_path_missing")
        if is_symlink:
            blockers.append("broken_symlink_or_unmounted_target")

    required_rows: List[Dict[str, Any]] = []
    if exists:
        for relative in spec.required_files:
            member = configured / relative
            member_exists = member.is_file()
            required_rows.append(
                {
                    "relative_path": relative,
                    "exists": member_exists,
                    "size_bytes": member.stat().st_size if member_exists else None,
                }
            )
            if not member_exists:
                blockers.append(f"required_file_missing:{relative}")
        for relative in spec.required_directories:
            member = configured / relative
            member_exists = member.is_dir()
            required_rows.append(
                {
                    "relative_path": relative,
                    "kind": "directory",
                    "exists": member_exists,
                    "size_bytes": None,
                }
            )
            if not member_exists:
                blockers.append(f"required_directory_missing:{relative}")

    identity_names = list(spec.identity_files)
    if deep_hash:
        identity_names.extend(
            name for name in spec.required_files if name not in identity_names
        )
    identity_rows: List[Dict[str, Any]] = []
    directory_identity_rows: List[Dict[str, Any]] = []
    if exists:
        for relative in identity_names:
            member = configured / relative
            if member.is_file():
                identity_rows.append(
                    {
                        "relative_path": relative,
                        "size_bytes": member.stat().st_size,
                        "sha256": sha256_file(member),
                    }
                )
            else:
                blockers.append(f"identity_file_missing:{relative}")
        for relative in spec.required_directories:
            member = configured / relative
            if member.is_dir():
                inventory = _directory_inventory(member)
                inventory["relative_path"] = relative
                directory_identity_rows.append(inventory)

    status = "pass" if not blockers else ("blocked" if spec.required else "optional_missing")
    identity_payload = {
        "asset_id": spec.asset_id,
        "resolved_path": str(resolved),
        "identity_files": identity_rows,
        "identity_directories": directory_identity_rows,
    }
    return {
        "asset_id": spec.asset_id,
        "role": spec.role,
        "required": spec.required,
        "configured_path": str(configured),
        "resolved_path": str(resolved),
        "is_symlink": is_symlink,
        "exists": exists,
        "deep_hash": deep_hash,
        "required_files": required_rows,
        "identity_files": identity_rows,
        "identity_directories": directory_identity_rows,
        "source_identity": _canonical_hash(identity_payload)
        if exists and (identity_rows or directory_identity_rows)
        else None,
        "status": status,
        "blockers": sorted(set(blockers)),
    }


def audit_local_sources(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    deep_hash: bool = False,
) -> Dict[str, Any]:
    """Audit every configured source and return one deterministic lineage ID."""
    validate_local_only_config(config)
    if config.get("network_policy") != "forbidden":
        raise ValueError("v6 requires network_policy=forbidden")
    if config.get("raw_data_policy") != "local_frozen_only":
        raise ValueError("v6 requires raw_data_policy=local_frozen_only")
    source_payload = config.get("sources")
    if not isinstance(source_payload, Sequence) or isinstance(
        source_payload, (str, bytes)
    ):
        raise ValueError("sources must be a sequence of local source specifications")
    specs = [LocalSourceSpec.model_validate(item) for item in source_payload]
    asset_ids = [spec.asset_id for spec in specs]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("source asset_id values must be unique")
    rows = [
        audit_local_source(spec, project_root=project_root, deep_hash=deep_hash)
        for spec in specs
    ]
    required_failures = [
        row for row in rows if row["required"] and row["status"] != "pass"
    ]
    identities = {
        row["asset_id"]: row["source_identity"]
        for row in rows
        if row["source_identity"]
    }
    return {
        "audit_version": "aspr-v6-source-audit-1",
        "network_policy": "forbidden",
        "raw_data_policy": "local_frozen_only",
        "deep_hash": deep_hash,
        "overall_pass": not required_failures,
        "required_failure_count": len(required_failures),
        "source_lineage_id": _canonical_hash(identities) if identities else None,
        "assets": rows,
    }


def write_source_audit(report: Mapping[str, Any], output_path: Path) -> None:
    """Persist an audit report as a derived artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "LocalSourceSpec",
    "audit_local_source",
    "audit_local_sources",
    "sha256_file",
    "write_source_audit",
]
