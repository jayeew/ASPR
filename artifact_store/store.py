"""Publish-once local artifact storage with hash-verified resolution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import ArtifactFile, ArtifactManifest, ArtifactReference


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ArtifactStore:
    """A local store whose releases are immutable after successful publication."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def release_path(self, producer: str, artifact: str, release: str) -> Path:
        """Return the deterministic directory for a release."""
        reference = ArtifactReference(
            producer=producer,
            artifact=artifact,
            release=release,
            manifest_sha256="sha256:" + "0" * 64,
        )
        return self.root / reference.producer / reference.artifact / reference.release

    def publish_directory(
        self,
        *,
        producer: str,
        artifact: str,
        release: str,
        source: Path,
        dependencies: Sequence[ArtifactReference] = (),
        metadata: Mapping[str, str | int | float | bool | None] = (),
    ) -> ArtifactReference:
        """Copy one directory into a new immutable release and return its pinned ref."""
        source = Path(source).resolve()
        if not source.is_dir():
            raise NotADirectoryError(source)
        target = self.release_path(producer, artifact, release)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return self._existing_reference(target)
        with tempfile.TemporaryDirectory(prefix="artifact-", dir=target.parent) as temp:
            staging = Path(temp) / "release"
            shutil.copytree(source, staging, copy_function=_link_or_copy)
            files = _file_records(staging)
            manifest_body = {
                "schema_version": "aspr_artifact",
                "contract": "aspr_artifact_manifest",
                "producer": producer,
                "artifact": artifact,
                "release": release,
                "files": [item.model_dump(mode="json") for item in files],
                "dependencies": [item.model_dump(mode="json") for item in dependencies],
                "metadata": dict(metadata),
            }
            draft = ArtifactManifest(
                **manifest_body,
                manifest_sha256="sha256:" + "0" * 64,
            )
            digest = (
                "sha256:"
                + hashlib.sha256(
                    _canonical_json(
                        draft.model_dump(mode="json", exclude={"manifest_sha256"})
                    ).encode("utf-8")
                ).hexdigest()
            )
            manifest = draft.model_copy(update={"manifest_sha256": digest})
            (staging / "artifact_manifest.json").write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            try:
                os.replace(staging, target)
            except FileExistsError:
                return self._existing_reference(target)
        return ArtifactReference(
            producer=manifest.producer,
            artifact=manifest.artifact,
            release=manifest.release,
            manifest_sha256=manifest.manifest_sha256,
        )

    def resolve(self, reference: ArtifactReference) -> Path:
        """Verify a pinned release and return its immutable directory."""
        target = self.release_path(
            reference.producer, reference.artifact, reference.release
        )
        manifest_path = target / "artifact_manifest.json"
        manifest = ArtifactManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.manifest_sha256 != reference.manifest_sha256:
            raise ValueError(
                "artifact manifest hash does not match the pinned reference"
            )
        for item in manifest.files:
            path = target / item.path
            if not path.is_file() or path.stat().st_size != item.size_bytes:
                raise ValueError(f"artifact file is missing or changed: {item.path}")
            if sha256_file(path) != item.sha256:
                raise ValueError(f"artifact file hash mismatch: {item.path}")
        return target

    def _existing_reference(self, target: Path) -> ArtifactReference:
        manifest = ArtifactManifest.model_validate_json(
            (target / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.resolve(
            ArtifactReference(
                producer=manifest.producer,
                artifact=manifest.artifact,
                release=manifest.release,
                manifest_sha256=manifest.manifest_sha256,
            )
        )
        return ArtifactReference(
            producer=manifest.producer,
            artifact=manifest.artifact,
            release=manifest.release,
            manifest_sha256=manifest.manifest_sha256,
        )


def _link_or_copy(source: str, target: str) -> str:
    """Hard-link files when possible, falling back to a regular copy."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _file_records(root: Path) -> list[ArtifactFile]:
    """Build sorted records for payload files, excluding a generated manifest."""
    records: list[ArtifactFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        records.append(
            ArtifactFile(
                path=path.relative_to(root).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    if not records:
        raise ValueError("artifact releases must contain at least one file")
    return records


__all__ = ["ArtifactStore", "sha256_file"]
