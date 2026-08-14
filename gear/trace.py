"""Append-only evidence, action, and state trace storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pydantic import BaseModel

from .contracts import ActionRecord, EvidenceRecord, StateSnapshot


def canonical_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonical_payload(item) for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [canonical_payload(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_value(value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class EvidenceStore:
    """Persist records without permitting semantic overwrite."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_path = self.output_dir / "evidence_trace.jsonl"
        self.action_path = self.output_dir / "action_trace.jsonl"
        self.state_path = self.output_dir / "state_trace.jsonl"
        self._evidence: Dict[str, EvidenceRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.evidence_path.is_file():
            return
        for line in self.evidence_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = EvidenceRecord.model_validate_json(line)
            if record.payload_sha256 != sha256_value(record.payload):
                raise ValueError(f"stored evidence hash mismatch: {record.evidence_id}")
            existing = self._evidence.get(record.evidence_id)
            if existing and (
                existing.payload_sha256 != record.payload_sha256
                or existing.kind != record.kind
            ):
                raise ValueError(f"conflicting existing evidence: {record.evidence_id}")
            self._evidence[record.evidence_id] = record

    def add_evidence(
        self,
        evidence_id: str,
        kind: str,
        payload: Any,
    ) -> EvidenceRecord:
        normalized = canonical_payload(payload)
        digest = sha256_value(normalized)
        existing = self._evidence.get(evidence_id)
        if existing:
            if existing.payload_sha256 != digest or existing.kind != kind:
                raise ValueError(f"evidence overwrite rejected: {evidence_id}")
            return existing
        record = EvidenceRecord(
            evidence_id=evidence_id,
            kind=kind,
            payload=normalized,
            payload_sha256=digest,
        )
        self._append(self.evidence_path, record)
        self._evidence[evidence_id] = record
        return record

    def append_action(self, record: ActionRecord) -> None:
        self._append(self.action_path, record)

    def snapshot_state(self, state: Any) -> StateSnapshot:
        normalized = self._compact_state(state)
        snapshot = StateSnapshot(
            state_sha256=sha256_value(normalized),
            state=normalized,
        )
        self._append(self.state_path, snapshot)
        return snapshot

    @staticmethod
    def _compact_state(state: Any) -> Dict[str, Any]:
        normalized = canonical_payload(state)
        if not isinstance(normalized, dict):
            raise TypeError("state snapshots require a mapping payload")
        return normalized

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self._evidence

    def get(self, evidence_id: str) -> Optional[EvidenceRecord]:
        return self._evidence.get(evidence_id)

    def ids(self) -> Iterable[str]:
        return tuple(self._evidence)

    def write_manifest(self, payload: Mapping[str, Any]) -> Path:
        path = self.output_dir / "run_manifest.json"
        normalized = canonical_payload(payload)
        normalized["trace_files"] = {
            trace_path.name: {
                "sha256": sha256_file(trace_path),
                "line_count": sum(
                    1
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ),
            }
            for trace_path in (
                self.evidence_path,
                self.action_path,
                self.state_path,
            )
            if trace_path.is_file()
        }
        normalized["manifest_sha256"] = sha256_value(normalized)
        path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def validate_manifest(self) -> List[str]:
        """Validate the manifest, trace files, and persisted review artifacts."""
        path = self.output_dir / "run_manifest.json"
        if not path.is_file():
            return ["run_manifest_missing"]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [f"run_manifest_invalid:{type(exc).__name__}"]
        claimed_manifest_hash = str(payload.pop("manifest_sha256", ""))
        failures: List[str] = []
        if claimed_manifest_hash != sha256_value(payload):
            failures.append("run_manifest_hash_mismatch")
        for name, expected in (payload.get("trace_files") or {}).items():
            trace_path = (self.output_dir / str(name)).resolve()
            if trace_path.parent != self.output_dir or not trace_path.is_file():
                failures.append(f"trace_file_missing_or_unsafe:{name}")
                continue
            if sha256_file(trace_path) != expected.get("sha256"):
                failures.append(f"trace_file_hash_mismatch:{name}")
            line_count = sum(
                1
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if line_count != int(expected.get("line_count", -1)):
                failures.append(f"trace_file_line_count_mismatch:{name}")
        output_files = payload.get("output_files") or {}
        for artifact, expected_hash in (
            payload.get("output_file_sha256") or {}
        ).items():
            raw_path = output_files.get(artifact)
            if not raw_path:
                failures.append(f"output_file_missing_from_manifest:{artifact}")
                continue
            artifact_path = Path(raw_path).resolve()
            try:
                artifact_path.relative_to(self.output_dir)
            except ValueError:
                failures.append(f"output_file_outside_run_dir:{artifact}")
                continue
            if not artifact_path.is_file():
                failures.append(f"output_file_missing:{artifact}")
            elif sha256_file(artifact_path) != expected_hash:
                failures.append(f"output_file_hash_mismatch:{artifact}")
        return failures

    @staticmethod
    def _append(path: Path, record: BaseModel) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")


__all__ = ["EvidenceStore", "canonical_json", "sha256_file", "sha256_value"]
