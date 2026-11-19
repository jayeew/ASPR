"""Recover the exact frozen feature-set export from a Codex session log."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MARKER_START = "## feature_sets.json"
MARKER_END = "## input_snapshot.json"
EXPECTED = {
    "strict_7": (7, 4),
    "fulltext_16": (16, 10),
    "source_154": (154, 48),
    "ultrarelaxed_221": (221, 55),
}


def output_text(payload: dict[str, Any]) -> str:
    """Normalize a session tool-output payload to plain text."""
    output = payload.get("output", "")
    if isinstance(output, list):
        return "".join(
            item.get("text", "") for item in output if isinstance(item, dict)
        )
    return str(output)


def extract_feature_sets(session_log: Path) -> dict[str, Any]:
    """Extract the single complete feature-set JSON payload from a session log."""
    matches: list[dict[str, Any]] = []
    for line in session_log.read_text(encoding="utf-8", errors="replace").splitlines():
        record = json.loads(line)
        payload = record.get("payload", {})
        if payload.get("type") not in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            continue
        text = output_text(payload)
        if MARKER_START not in text or MARKER_END not in text:
            continue
        fragment = text.split(MARKER_START, 1)[1].split(MARKER_END, 1)[0].strip()
        matches.append(json.loads(fragment))
    if len(matches) != 1:
        raise ValueError(f"expected one complete feature-set payload, found {len(matches)}")
    return matches[0]


def validate(payload: dict[str, Any]) -> None:
    """Verify the known nested membership and cardinality invariants."""
    sets = payload.get("sets", {})
    previous: set[str] = set()
    for name, (feature_count, dimension_count) in EXPECTED.items():
        item = sets.get(name, {})
        features = set(item.get("feature_ids", []))
        dimensions = set(item.get("dimension_ids", []))
        if len(features) != feature_count or len(dimensions) != dimension_count:
            raise ValueError(f"unexpected {name} cardinality")
        if previous and not previous.issubset(features):
            raise ValueError(f"non-nested recovered set: {name}")
        previous = features


def main() -> None:
    """Write one validated, provenance-stamped recovery copy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = extract_feature_sets(args.session_log)
    validate(payload)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    print(f"recovered={args.output}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
