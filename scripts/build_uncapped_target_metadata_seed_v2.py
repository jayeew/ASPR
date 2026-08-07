#!/usr/bin/env python3
"""Reuse target authorship metadata embedded in the reference snapshot closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

try:
    import orjson
except ImportError:  # pragma: no cover - optional fast path
    orjson = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.corpus import normalize_openalex_id
from aspr.nature_multihorizon.openalex_controls_v6_1 import _target_metadata


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def read_target_ids(path: Path) -> set[str]:
    """Read normalized target IDs from a CSV or Parquet paper table."""
    path = Path(path)
    if path.suffix.casefold() == ".parquet":
        columns = set(pq.read_schema(path).names)
        id_column = "paper_id" if "paper_id" in columns else "id"
        values = pd.read_parquet(path, columns=[id_column])[id_column]
    else:
        header = pd.read_csv(path, nrows=0).columns
        id_column = "paper_id" if "paper_id" in header else "id"
        values = pd.read_csv(path, usecols=[id_column], dtype="string")[id_column]
    ids = {
        normalized for value in values if (normalized := normalize_openalex_id(value))
    }
    if not ids:
        raise ValueError(f"no target IDs found in {path}")
    return ids


def iter_checkpoint_works(
    paths: Iterable[Path],
) -> Iterable[tuple[Path, dict[str, Any]]]:
    """Yield valid raw OpenAlex works from line-delimited checkpoints."""
    for path in paths:
        with Path(path).open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    payload = (
                        orjson.loads(line) if orjson is not None else json.loads(line)
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                work = payload.get("work", payload)
                if not isinstance(work, dict):
                    raise TypeError(f"invalid work object at {path}:{line_number}")
                yield Path(path), work


def build_seed(
    *,
    target_works: Path,
    base_seed: Path,
    checkpoint_jsonl: Sequence[Path],
    checkpoint_provenance: Sequence[Path] | None = None,
    output: Path,
) -> dict[str, Any]:
    """Merge prior target metadata with target works found in raw closure records."""
    target_ids = read_target_ids(target_works)
    base = pd.read_parquet(base_seed)
    base["paper_id"] = base["paper_id"].map(normalize_openalex_id)
    base = base[base["paper_id"].isin(target_ids)].copy()
    base_ids = set(base["paper_id"].astype(str))
    needed = target_ids - base_ids
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    records_scanned = 0
    for source_path, work in iter_checkpoint_works(checkpoint_jsonl):
        records_scanned += 1
        paper_id = normalize_openalex_id(work.get("id"))
        if not paper_id or paper_id not in needed or paper_id in seen:
            continue
        rows.append(_target_metadata(work, source_path))
        seen.add(paper_id)
    recovered = pd.DataFrame(rows)
    combined = pd.concat([base, recovered], ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["paper_id", "openalex_updated_date", "metadata_source_file"],
        kind="stable",
    ).drop_duplicates("paper_id", keep="last")
    combined = combined[combined["paper_id"].isin(target_ids)].copy()
    if combined["paper_id"].duplicated().any():
        raise ValueError("target metadata seed contains duplicate paper IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    provenance_paths = list(checkpoint_provenance or checkpoint_jsonl)
    if len(provenance_paths) != len(checkpoint_jsonl):
        raise ValueError(
            "checkpoint provenance paths must match processing checkpoint count"
        )
    manifest = {
        "artifact_kind": "uncapped_v2_target_metadata_reference_seed",
        "target_count": len(target_ids),
        "base_seed_count": len(base_ids),
        "reference_records_scanned": records_scanned,
        "reference_targets_recovered": len(seen),
        "seed_count": len(combined),
        "remaining_for_api": len(target_ids - set(combined["paper_id"].astype(str))),
        "target_works": str(Path(target_works).resolve()),
        "base_seed": str(Path(base_seed).resolve()),
        "checkpoint_jsonl": [str(Path(path).resolve()) for path in provenance_paths],
        "processing_checkpoint_jsonl": [
            str(Path(path).resolve()) for path in checkpoint_jsonl
        ],
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse seed-construction paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-works", type=Path, required=True)
    parser.add_argument("--base-seed", type=Path, required=True)
    parser.add_argument("--checkpoint-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the audited local target-metadata seed."""
    args = parse_args(argv)
    result = build_seed(
        target_works=args.target_works,
        base_seed=args.base_seed,
        checkpoint_jsonl=args.checkpoint_jsonl,
        checkpoint_provenance=args.checkpoint_provenance,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
