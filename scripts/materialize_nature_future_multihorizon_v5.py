from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.future_citers import _normalize_citer  # noqa: E402
from scripts.nature_portfolio_v5 import (  # noqa: E402
    DEFAULT_V5_OUTPUT_DIR,
    entropy_from_counts,
    normalize_openalex_id,
    short_openalex_id,
    simpson_from_counts,
    utc_now,
)


DEFAULT_OUTPUT_DIR = DEFAULT_V5_OUTPUT_DIR / "future_multihorizon"
DEFAULT_HORIZONS = (3, 5, 8)
REQUEST_BATCH = "common_tau8_le2017_legacy_checkpoint_adapter"

FUTURE_CITER_SCHEMA = pa.schema(
    [
        pa.field("paper_id", pa.string(), nullable=False),
        pa.field("horizon", pa.int16(), nullable=False),
        pa.field("citer_id", pa.string(), nullable=False),
        pa.field("citer_year", pa.int16(), nullable=False),
        pa.field("citer_primary_field", pa.string()),
        pa.field("citer_primary_subfield", pa.string()),
        pa.field("citer_primary_topic", pa.string()),
        pa.field("referenced_works", pa.list_(pa.string())),
    ]
)

DELTA_COLUMNS = [
    "paper_id",
    "year",
    "publication_year",
    "tau",
    "horizon",
    "fetch_status",
    "fetch_valid",
    "cap_hit",
    "requested_horizon_cap_hit",
    "n_future_citers",
    "future_community_reach",
    "future_topic_reach",
    "future_field_reach",
    "future_subfield_reach",
    "future_field_entropy",
    "future_topic_entropy",
    "future_field_simpson",
    "future_topic_simpson",
    "future_field_valid_n",
    "future_subfield_valid_n",
    "future_topic_valid_n",
    "future_field_coverage",
    "future_subfield_coverage",
    "future_topic_coverage",
    "future_first_year",
    "future_last_year",
]


class FutureCiterParquetWriter:
    """Write canonical multi-horizon citer rows with a stable Arrow schema."""

    def __init__(self, path: Path, batch_size: int) -> None:
        self.path = path
        self.batch_size = int(batch_size)
        self.buffer: List[Dict[str, Any]] = []
        self.row_count = 0
        self.writer: Optional[pq.ParquetWriter] = None
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Mapping[str, Any]) -> None:
        self.buffer.append(dict(row))
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=FUTURE_CITER_SCHEMA)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.path,
                FUTURE_CITER_SCHEMA,
                compression="zstd",
            )
        self.writer.write_table(table)
        self.row_count += len(self.buffer)
        self.buffer = []

    def close(self) -> int:
        if self.closed:
            return int(self.row_count)
        self.flush()
        if self.writer is None:
            pq.write_table(
                pa.Table.from_pylist([], schema=FUTURE_CITER_SCHEMA),
                self.path,
                compression="zstd",
            )
        else:
            self.writer.close()
            self.writer = None
        self.closed = True
        return int(self.row_count)


def _read_targets(path: Path, complete_end_year: int, requested_horizon: int) -> pd.DataFrame:
    columns = set(pd.read_csv(path, nrows=0).columns)
    id_column = "id" if "id" in columns else "paper_id"
    year_column = "year" if "year" in columns else "publication_year"
    required = {id_column, year_column}
    if not required.issubset(columns):
        raise ValueError(f"Target works is missing columns: {sorted(required - columns)}")
    usecols = [id_column, year_column]
    if "short_id" in columns:
        usecols.append("short_id")
    targets = pd.read_csv(path, usecols=usecols, low_memory=False)
    targets = targets.rename(columns={id_column: "paper_id", year_column: "publication_year"})
    targets["paper_id"] = targets["paper_id"].map(normalize_openalex_id)
    targets["publication_year"] = pd.to_numeric(
        targets["publication_year"], errors="coerce"
    )
    targets = targets[
        targets["paper_id"].astype(str).str.strip().ne("")
        & targets["publication_year"].notna()
        & (
            targets["publication_year"].astype(int)
            <= int(complete_end_year) - int(requested_horizon)
        )
    ].copy()
    targets["publication_year"] = targets["publication_year"].astype(int)
    if "short_id" not in targets:
        targets["short_id"] = targets["paper_id"].map(short_openalex_id)
    else:
        targets["short_id"] = targets["short_id"].fillna("").astype(str).str.strip()
        empty = targets["short_id"].eq("")
        targets.loc[empty, "short_id"] = targets.loc[empty, "paper_id"].map(
            short_openalex_id
        )
    duplicated = targets["paper_id"].duplicated(keep=False)
    if duplicated.any():
        examples = targets.loc[duplicated, "paper_id"].head(5).tolist()
        raise ValueError(f"Eligible target paper IDs are not unique: {examples}")
    return targets.sort_values(["publication_year", "paper_id"], kind="stable").reset_index(
        drop=True
    )


def _iter_checkpoint(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Checkpoint row is not an object at {path}:{line_number}")
            yield row


def _referenced_works(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple, set)):
        return None
    normalized = [normalize_openalex_id(item) for item in value]
    return [item for item in normalized if item]


def _normalized_checkpoint_rows(
    path: Path,
    paper_id: str,
    publication_year: int,
    requested_horizon: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    rows: Dict[str, Dict[str, Any]] = {}
    raw_count = 0
    invalid_count = 0
    for wrapper in _iter_checkpoint(path):
        raw_count += 1
        wrapped_paper_id = normalize_openalex_id(wrapper.get("paper_id"))
        if wrapped_paper_id and wrapped_paper_id != paper_id:
            raise ValueError(
                f"Checkpoint paper mismatch in {path}: {wrapped_paper_id} != {paper_id}"
            )
        work = wrapper.get("work", wrapper)
        if not isinstance(work, Mapping):
            invalid_count += 1
            continue
        normalized = _normalize_citer(
            paper_id,
            requested_horizon,
            publication_year,
            work,
        )
        if normalized is None:
            invalid_count += 1
            continue
        normalized["paper_id"] = normalize_openalex_id(normalized["paper_id"])
        normalized["citer_id"] = normalize_openalex_id(normalized["citer_id"])
        normalized["referenced_works"] = _referenced_works(
            normalized.get("referenced_works")
        )
        rows.setdefault(str(normalized["citer_id"]), normalized)
    ordered = sorted(
        rows.values(), key=lambda item: (int(item["citer_year"]), str(item["citer_id"]))
    )
    return ordered, raw_count, invalid_count


def _counts(values: Sequence[str]) -> List[float]:
    return [float(value) for value in Counter(values).values()]


def _delta_row(
    paper_id: str,
    publication_year: int,
    horizon: int,
    citers: Sequence[Mapping[str, Any]],
    requested_cap_hit: bool,
    last_citer_year: Optional[int],
) -> Dict[str, Any]:
    fields = [str(row.get("citer_primary_field") or "") for row in citers]
    subfields = [str(row.get("citer_primary_subfield") or "") for row in citers]
    topics = [str(row.get("citer_primary_topic") or "") for row in citers]
    fields = [value for value in fields if value]
    subfields = [value for value in subfields if value]
    topics = [value for value in topics if value]
    years = [int(row["citer_year"]) for row in citers]
    n_future = len(citers)

    def coverage(values: Sequence[str]) -> float:
        return 1.0 if n_future == 0 else float(len(values) / n_future)

    horizon_cap_hit = bool(
        requested_cap_hit
        and (last_citer_year is None or last_citer_year <= publication_year + horizon)
    )
    field_counts = _counts(fields)
    topic_counts = _counts(topics)
    return {
        "paper_id": paper_id,
        "year": int(publication_year),
        "publication_year": int(publication_year),
        "tau": int(horizon),
        "horizon": int(horizon),
        "fetch_status": "success",
        "fetch_valid": 1,
        "cap_hit": int(horizon_cap_hit),
        "requested_horizon_cap_hit": int(requested_cap_hit),
        "n_future_citers": int(n_future),
        "future_community_reach": int(len(set(topics))),
        "future_topic_reach": int(len(set(topics))),
        "future_field_reach": int(len(set(fields))),
        "future_subfield_reach": int(len(set(subfields))),
        "future_field_entropy": (
            entropy_from_counts(field_counts) if field_counts else 0.0
        ),
        "future_topic_entropy": (
            entropy_from_counts(topic_counts) if topic_counts else 0.0
        ),
        "future_field_simpson": (
            simpson_from_counts(field_counts) if field_counts else 0.0
        ),
        "future_topic_simpson": (
            simpson_from_counts(topic_counts) if topic_counts else 0.0
        ),
        "future_field_valid_n": int(len(fields)),
        "future_subfield_valid_n": int(len(subfields)),
        "future_topic_valid_n": int(len(topics)),
        "future_field_coverage": coverage(fields),
        "future_subfield_coverage": coverage(subfields),
        "future_topic_coverage": coverage(topics),
        "future_first_year": min(years) if years else None,
        "future_last_year": max(years) if years else None,
    }


def _missing_delta_row(
    paper_id: str,
    publication_year: int,
    horizon: int,
) -> Dict[str, Any]:
    row = {column: None for column in DELTA_COLUMNS}
    row.update(
        {
            "paper_id": paper_id,
            "year": int(publication_year),
            "publication_year": int(publication_year),
            "tau": int(horizon),
            "horizon": int(horizon),
            "fetch_status": "not_requested_or_failed",
            "fetch_valid": 0,
            "cap_hit": 0,
            "requested_horizon_cap_hit": 0,
        }
    )
    return row


def _write_compatibility_outputs(
    deltas: pd.DataFrame,
    output_dir: Path,
    horizons: Sequence[int],
) -> Dict[str, Dict[str, str]]:
    paths: Dict[str, Dict[str, str]] = {}
    horizon_root = output_dir / "horizons"
    for horizon in horizons:
        subset = deltas[deltas["horizon"].eq(int(horizon))].copy()
        subset = subset[DELTA_COLUMNS].sort_values("paper_id", kind="stable")
        directory = horizon_root / f"tau{int(horizon)}"
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / "nature_future_graph_deltas.csv"
        parquet_path = directory / "nature_future_graph_deltas.parquet"
        subset.to_csv(csv_path, index=False)
        subset.to_parquet(parquet_path, index=False, compression="zstd")
        paths[f"tau{int(horizon)}"] = {
            "csv": str(csv_path.relative_to(output_dir)),
            "parquet": str(parquet_path.relative_to(output_dir)),
        }
    return paths


def _write_downstream_files(
    write_dir: Path,
    output_dir: Path,
    source_dir: Path,
    compatibility_paths: Mapping[str, Mapping[str, str]],
) -> None:
    lines = [
        f"ASPR_NATURE_MULTIHORIZON_DIR={output_dir.resolve()}",
        f"ASPR_NATURE_MULTIHORIZON_FUTURE_CITERS={(output_dir / 'future_citers.parquet').resolve()}",
        f"ASPR_NATURE_MULTIHORIZON_FETCH_STATUS={(output_dir / 'future_fetch_status.parquet').resolve()}",
        "ASPR_NATURE_MULTIHORIZON_REQUEST_MANIFEST="
        f"{(output_dir / 'future_request_manifest.parquet').resolve()}",
        "ASPR_NATURE_MULTIHORIZON_GRAPH_DELTAS="
        f"{(output_dir / 'future_graph_deltas_multihorizon.parquet').resolve()}",
    ]
    for label, paths in compatibility_paths.items():
        horizon = label.removeprefix("tau")
        lines.append(
            f"ASPR_NATURE_V5_FUTURE_GRAPH_DELTAS_TAU{horizon}="
            f"{(output_dir / paths['csv']).resolve()}"
        )
    (write_dir / "downstream_paths.env").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    usage = f"""# Nature Portfolio v5 multi-horizon outputs

The tables in this directory are derived offline from the common complete
tau8 cohort. They are label-only outcomes and must not be used as
publication-day features.

## Nature multihorizon table contract

- `future_citers.parquet`: key `(paper_id, horizon, citer_id)`
- `future_fetch_status.parquet`: key `(paper_id, requested_horizon)`
- `future_request_manifest.parquet`: common tau8 request population
- `future_graph_deltas_multihorizon.parquet`: key `(paper_id, horizon)`

## Existing v5 Fig.3 compatibility

Run a horizon directly without copying files:

```bash
python scripts/run_fig3_nature_full_v5.py \\
  --works {source_dir / 'nature_target_works.csv'} \\
  --reference-works {source_dir / 'nature_reference_works.csv'} \\
  --future-graph-deltas {output_dir / 'horizons/tau5/nature_future_graph_deltas.csv'} \\
  --out-dir outputs/nature_portfolio_v5/fig3_nature_full_v5_tau5
```

Replace `tau5` with `tau3` or `tau8` for the other windows.
"""
    (write_dir / "USAGE.md").write_text(usage, encoding="utf-8")


def materialize_multihorizon(args: argparse.Namespace) -> Dict[str, Any]:
    horizons = tuple(sorted(set(int(value) for value in args.horizons)))
    if not horizons or horizons[0] <= 0 or horizons[-1] > int(args.requested_horizon):
        raise ValueError("Horizons must be positive and no larger than requested_horizon")
    targets = _read_targets(
        args.target_works,
        complete_end_year=args.complete_end_year,
        requested_horizon=args.requested_horizon,
    )
    checkpoint_paths = {
        path.stem: path for path in args.checkpoint_dir.glob("*.jsonl")
    }
    expected_ids = set(targets["short_id"].astype(str))
    missing_ids = sorted(expected_ids - set(checkpoint_paths))
    extra_ids = sorted(set(checkpoint_paths) - expected_ids)
    if missing_ids and not args.allow_missing:
        examples = ", ".join(missing_ids[:10])
        raise RuntimeError(
            f"Missing {len(missing_ids)} eligible checkpoints; examples: {examples}"
        )

    final_output = args.output_dir
    temporary_output = final_output.with_name(
        f".{final_output.name}.tmp-{os.getpid()}"
    )
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    if final_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {final_output}; pass --overwrite to rebuild"
        )
    temporary_output.mkdir(parents=True, exist_ok=False)

    statuses: List[Dict[str, Any]] = []
    expected_rows: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    citer_writer = FutureCiterParquetWriter(
        temporary_output / "future_citers.parquet", args.batch_size
    )
    try:
        for index, paper in enumerate(targets.to_dict("records"), start=1):
            paper_id = str(paper["paper_id"])
            short_id = str(paper["short_id"])
            publication_year = int(paper["publication_year"])
            expected_rows.append(
                {
                    "paper_id": paper_id,
                    "publication_year": publication_year,
                    "requested_horizon": int(args.requested_horizon),
                    "request_batch": REQUEST_BATCH,
                }
            )
            checkpoint = checkpoint_paths.get(short_id)
            if checkpoint is None:
                statuses.append(
                    {
                        "paper_id": paper_id,
                        "requested_horizon": int(args.requested_horizon),
                        "fetch_status": "not_requested_or_failed",
                        "n_returned": None,
                        "is_zero_success": 0,
                        "cap_hit": 0,
                        "last_citer_year": None,
                        "error_type": "missing_checkpoint",
                        "attempt_count": 0,
                        "publication_year": publication_year,
                        "checkpoint_file": None,
                        "request_batch": REQUEST_BATCH,
                    }
                )
                diagnostics["missing_checkpoints"] += 1
                for horizon in horizons:
                    deltas.append(
                        _missing_delta_row(paper_id, publication_year, horizon)
                    )
                continue
            normalized, raw_count, invalid_count = _normalized_checkpoint_rows(
                checkpoint,
                paper_id,
                publication_year,
                args.requested_horizon,
            )
            diagnostics["raw_citer_rows"] += raw_count
            diagnostics["normalized_unique_citers"] += len(normalized)
            diagnostics["duplicate_or_invalid_rows"] += raw_count - len(normalized)
            diagnostics["invalid_rows"] += invalid_count
            diagnostics["zero_success"] += int(raw_count == 0)
            requested_cap_hit = raw_count >= int(args.max_citers_per_work)
            diagnostics["requested_cap_hits"] += int(requested_cap_hit)
            last_citer_year = (
                max(int(row["citer_year"]) for row in normalized)
                if normalized
                else None
            )
            statuses.append(
                {
                    "paper_id": paper_id,
                    "requested_horizon": int(args.requested_horizon),
                    "fetch_status": "success",
                    "n_returned": int(len(normalized)),
                    "is_zero_success": int(len(normalized) == 0),
                    "cap_hit": int(requested_cap_hit),
                    "last_citer_year": last_citer_year,
                    "error_type": "",
                    "attempt_count": 1,
                    "publication_year": publication_year,
                    "checkpoint_file": str(checkpoint.resolve()),
                    "request_batch": REQUEST_BATCH,
                }
            )
            for horizon in horizons:
                in_window = [
                    row
                    for row in normalized
                    if publication_year < int(row["citer_year"]) <= publication_year + horizon
                ]
                deltas.append(
                    _delta_row(
                        paper_id,
                        publication_year,
                        horizon,
                        in_window,
                        requested_cap_hit,
                        last_citer_year,
                    )
                )
                for row in in_window:
                    citer_writer.append(
                        {
                            "paper_id": paper_id,
                            "horizon": int(horizon),
                            "citer_id": str(row["citer_id"]),
                            "citer_year": int(row["citer_year"]),
                            "citer_primary_field": row.get("citer_primary_field"),
                            "citer_primary_subfield": row.get(
                                "citer_primary_subfield"
                            ),
                            "citer_primary_topic": row.get("citer_primary_topic"),
                            "referenced_works": row.get("referenced_works"),
                        }
                    )
            if not args.quiet and (
                index == 1 or index % int(args.progress_every) == 0 or index == len(targets)
            ):
                print(
                    f"[Multi-horizon v5] processed {index:,}/{len(targets):,}; "
                    f"citer_rows={citer_writer.row_count + len(citer_writer.buffer):,}",
                    flush=True,
                )
        n_citer_rows = citer_writer.close()
        status_frame = pd.DataFrame(statuses).sort_values(
            ["paper_id", "requested_horizon"], kind="stable"
        )
        expected_frame = pd.DataFrame(expected_rows).sort_values(
            ["paper_id", "requested_horizon"], kind="stable"
        )
        delta_frame = pd.DataFrame(deltas, columns=DELTA_COLUMNS).sort_values(
            ["paper_id", "horizon"], kind="stable"
        )
        status_frame.to_parquet(
            temporary_output / "future_fetch_status.parquet",
            index=False,
            compression="zstd",
        )
        expected_frame.to_parquet(
            temporary_output / "future_request_manifest.parquet",
            index=False,
            compression="zstd",
        )
        delta_frame.to_parquet(
            temporary_output / "future_graph_deltas_multihorizon.parquet",
            index=False,
            compression="zstd",
        )
        delta_frame.to_csv(
            temporary_output / "future_graph_deltas_multihorizon.csv", index=False
        )
        compatibility_paths = _write_compatibility_outputs(
            delta_frame, temporary_output, horizons
        )

        expected_delta_rows = len(targets) * len(horizons)
        key_duplicates = int(
            delta_frame.duplicated(["paper_id", "horizon"]).sum()
        )
        pivot = delta_frame.pivot(
            index="paper_id", columns="horizon", values="n_future_citers"
        )
        nested_violations = 0
        for lower, upper in zip(horizons, horizons[1:]):
            nested_violations += int((pivot[lower] > pivot[upper]).sum())
        complete_status = int(status_frame["fetch_status"].eq("success").sum())
        overall_pass = bool(
            len(delta_frame) == expected_delta_rows
            and key_duplicates == 0
            and nested_violations == 0
            and complete_status == len(targets)
            and diagnostics["invalid_rows"] == 0
        )
        quality = {
            "artifact_kind": "nature_portfolio_v5_multihorizon_quality",
            "created_at": utc_now(),
            "overall_pass": overall_pass,
            "n_common_tau8_papers": int(len(targets)),
            "horizons": list(horizons),
            "expected_delta_rows": int(expected_delta_rows),
            "actual_delta_rows": int(len(delta_frame)),
            "delta_key_duplicates": key_duplicates,
            "nested_count_violations": int(nested_violations),
            "checkpoint_coverage": float(complete_status / max(1, len(targets))),
            "missing_checkpoint_count": int(len(missing_ids)),
            "extra_checkpoint_count": int(len(extra_ids)),
            "diagnostics": dict(diagnostics),
            "horizon_cap_hits": {
                str(horizon): int(
                    delta_frame.loc[delta_frame["horizon"].eq(horizon), "cap_hit"].sum()
                )
                for horizon in horizons
            },
            "label_only_no_leakage": True,
        }
        (temporary_output / "data_quality_report.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "artifact_kind": "nature_portfolio_v5_future_multihorizon",
            "schema_version": "1.0.0",
            "created_at": utc_now(),
            "target_works": str(args.target_works.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "output_dir": str(final_output.resolve()),
            "complete_end_year": int(args.complete_end_year),
            "requested_horizon": int(args.requested_horizon),
            "derived_horizons": list(horizons),
            "max_citers_per_work": int(args.max_citers_per_work),
            "request_batch": REQUEST_BATCH,
            "n_common_tau8_papers": int(len(targets)),
            "n_future_citer_rows": int(n_citer_rows),
            "n_future_delta_rows": int(len(delta_frame)),
            "n_fetch_status_rows": int(len(status_frame)),
            "future_citers": "future_citers.parquet",
            "future_fetch_status": "future_fetch_status.parquet",
            "future_request_manifest": "future_request_manifest.parquet",
            "future_graph_deltas": "future_graph_deltas_multihorizon.parquet",
            "compatibility_outputs": compatibility_paths,
            "quality_report": "data_quality_report.json",
            "quality_overall_pass": overall_pass,
            "primary_keys": {
                "future_citers": ["paper_id", "horizon", "citer_id"],
                "future_fetch_status": ["paper_id", "requested_horizon"],
                "future_request_manifest": ["paper_id", "requested_horizon"],
                "future_graph_deltas": ["paper_id", "horizon"],
            },
            "no_leakage_contract": (
                "Future citers and graph deltas are label-only outcomes and are "
                "excluded from publication-day features."
            ),
        }
        (temporary_output / "future_multihorizon_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_output / "future_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_downstream_files(
            temporary_output,
            final_output,
            args.target_works.parent,
            compatibility_paths,
        )
        if not overall_pass and not args.allow_quality_failures:
            raise RuntimeError(
                "Multi-horizon data quality gate failed; inspect "
                f"{temporary_output / 'data_quality_report.json'}"
            )
        if final_output.exists():
            shutil.rmtree(final_output)
        os.replace(temporary_output, final_output)
        return manifest
    except Exception:
        try:
            citer_writer.close()
        except Exception:
            pass
        raise


def audit_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    """Check common-cohort/checkpoint coverage without reading checkpoint bodies."""

    targets = _read_targets(
        args.target_works,
        complete_end_year=args.complete_end_year,
        requested_horizon=args.requested_horizon,
    )
    expected_ids = set(targets["short_id"].astype(str))
    checkpoint_ids = {path.stem for path in args.checkpoint_dir.glob("*.jsonl")}
    missing = sorted(expected_ids - checkpoint_ids)
    extra = sorted(checkpoint_ids - expected_ids)
    return {
        "artifact_kind": "nature_portfolio_v5_multihorizon_input_audit",
        "target_works": str(args.target_works.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "complete_end_year": int(args.complete_end_year),
        "requested_horizon": int(args.requested_horizon),
        "n_common_tau8_papers": int(len(targets)),
        "n_checkpoint_files": int(len(checkpoint_ids)),
        "missing_checkpoint_count": int(len(missing)),
        "extra_checkpoint_count": int(len(extra)),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "ready": not missing,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive common-cohort tau3/tau5/tau8 Nature Portfolio future-citer "
            "tables from legacy tau8 checkpoints without network requests."
        )
    )
    parser.add_argument(
        "--target-works",
        type=Path,
        default=DEFAULT_V5_OUTPUT_DIR / "nature_target_works.csv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_V5_OUTPUT_DIR / "checkpoints/future_citers_tau8",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--requested-horizon", type=int, default=8)
    parser.add_argument("--complete-end-year", type=int, default=2025)
    parser.add_argument("--max-citers-per-work", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=1_000)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-quality-failures", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.audit_only:
        audit = audit_inputs(args)
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if audit["ready"] else 2
    manifest = materialize_multihorizon(args)
    if not args.quiet:
        print(
            "[Multi-horizon v5] complete: "
            f"papers={manifest['n_common_tau8_papers']:,}, "
            f"citer_rows={manifest['n_future_citer_rows']:,}, "
            f"delta_rows={manifest['n_future_delta_rows']:,}, "
            f"output={args.output_dir}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
