"""Resumable, failure-aware future-citer acquisition and horizon derivation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FutureCiterFetcher = Callable[[str, int, int, int], Iterable[Mapping[str, Any]]]

PREBUILT_FUTURE_FILENAMES: Tuple[str, ...] = (
    "future_citers.parquet",
    "future_fetch_status.parquet",
    "future_request_manifest.parquet",
    "future_graph_deltas_multihorizon.parquet",
    "future_multihorizon_manifest.json",
    "data_quality_report.json",
)

PREBUILT_FUTURE_PRIMARY_KEYS: Dict[str, Tuple[str, ...]] = {
    "future_citers": ("paper_id", "horizon", "citer_id"),
    "future_fetch_status": ("paper_id", "requested_horizon"),
    "future_request_manifest": ("paper_id", "requested_horizon"),
    "future_graph_deltas": ("paper_id", "horizon"),
}


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _parquet_rows(path: Path) -> int:
    """Read a Parquet row count from metadata without loading the table."""

    return int(pq.ParquetFile(path).metadata.num_rows)


def _assert_regular_source(path: Path) -> None:
    """Reject unfinished, missing, or symbolic-link source artifacts."""

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Prebuilt future artifact must be a regular file: {path}")
    if path.name.endswith(".tmp") or ".tmp-" in path.name:
        raise ValueError(f"Temporary future artifact is not a formal input: {path}")


def _normalize_prebuilt_status(status: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Convert explicit legacy checkpoint misses to canonical failures.

    The source row is retained through ``source_fetch_status``.  Only the
    canonical status changes; missing outcomes remain NA and can never become
    successful zero-citer observations.
    """

    required = {
        "paper_id",
        "requested_horizon",
        "fetch_status",
        "n_returned",
        "error_type",
    }
    missing = sorted(required - set(status.columns))
    if missing:
        raise ValueError(f"Prebuilt future status is missing columns: {missing}")
    output = status.copy()
    output["source_fetch_status"] = output["fetch_status"].astype(str)
    source_status = output["source_fetch_status"].str.strip()
    error_type = output["error_type"].fillna("").astype(str).str.strip()
    legacy_missing = source_status.eq("not_requested_or_failed")
    invalid_legacy = legacy_missing & ~error_type.eq("missing_checkpoint")
    if invalid_legacy.any():
        examples = output.loc[
            invalid_legacy, ["paper_id", "fetch_status", "error_type"]
        ].head(5)
        raise ValueError(
            "Only explicit missing_checkpoint rows may be adopted as failed: "
            f"{examples.to_dict('records')}"
        )
    if output.loc[legacy_missing, "n_returned"].notna().any():
        raise ValueError("Missing checkpoints must retain n_returned=NA")
    output.loc[legacy_missing, "fetch_status"] = "failed"
    output["import_normalization"] = ""
    output.loc[legacy_missing, "import_normalization"] = (
        "not_requested_or_failed_to_failed_preserving_na"
    )
    allowed = output["fetch_status"].isin(["success", "failed"])
    if not allowed.all():
        values = sorted(output.loc[~allowed, "fetch_status"].astype(str).unique())
        raise ValueError(f"Unsupported prebuilt future statuses: {values}")
    failed = output["fetch_status"].eq("failed")
    if output.loc[failed, "n_returned"].notna().any():
        raise ValueError("Canonical failed future requests must retain n_returned=NA")
    return output, int(legacy_missing.sum())


def audit_prebuilt_future_multihorizon(
    source_dir: Path,
    *,
    expected_horizons: Sequence[int] = (3, 5, 8),
    minimum_success_rate: float = 0.99,
    maximum_missing_checkpoints: int = 5,
) -> Dict[str, Any]:
    """Audit the completed offline τ3/τ5/τ8 materialization.

    ``overall_pass=false`` is preserved as upstream provenance.  It is accepted
    only when every structural check passes and the sole incomplete condition
    is a bounded set of explicit ``missing_checkpoint`` rows with NA outcomes.
    """

    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Prebuilt future directory is unavailable: {root}")
    paths = {name: root / name for name in PREBUILT_FUTURE_FILENAMES}
    for path in paths.values():
        _assert_regular_source(path)

    manifest = _read_json(paths["future_multihorizon_manifest.json"])
    quality = _read_json(paths["data_quality_report.json"])
    if manifest.get("artifact_kind") != "nature_portfolio_v5_future_multihorizon":
        raise ValueError("Unexpected prebuilt future manifest artifact_kind")
    if quality.get("artifact_kind") != "nature_portfolio_v5_multihorizon_quality":
        raise ValueError("Unexpected prebuilt future quality artifact_kind")
    horizons = tuple(sorted(set(int(value) for value in expected_horizons)))
    observed_horizons = tuple(
        sorted(int(value) for value in manifest.get("derived_horizons", ()))
    )
    if observed_horizons != horizons:
        raise ValueError(
            f"Prebuilt horizons differ; expected={horizons}, observed={observed_horizons}"
        )
    expected_keys = {
        name: list(columns) for name, columns in PREBUILT_FUTURE_PRIMARY_KEYS.items()
    }
    if manifest.get("primary_keys") != expected_keys:
        raise ValueError("Prebuilt future primary-key contract does not match V1")

    status_source = pd.read_parquet(paths["future_fetch_status.parquet"])
    requests = pd.read_parquet(paths["future_request_manifest.parquet"])
    deltas = pd.read_parquet(paths["future_graph_deltas_multihorizon.parquet"])
    status, normalized_failures = _normalize_prebuilt_status(status_source)
    status_key = ["paper_id", "requested_horizon"]
    delta_key = ["paper_id", "horizon"]
    if status.duplicated(status_key).any() or status[status_key].isna().to_numpy().any():
        raise ValueError("Prebuilt future status primary key is invalid")
    if requests.duplicated(status_key).any() or requests[status_key].isna().to_numpy().any():
        raise ValueError("Prebuilt future request primary key is invalid")
    if deltas.duplicated(delta_key).any() or deltas[delta_key].isna().to_numpy().any():
        raise ValueError("Prebuilt future delta primary key is invalid")
    joined = requests[status_key].merge(
        status[status_key + ["fetch_status"]],
        on=status_key,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Prebuilt request and status keys do not match")

    n_requests = int(len(requests))
    expected_delta_rows = n_requests * len(horizons)
    observed_delta_rows = _parquet_rows(paths["future_graph_deltas_multihorizon.parquet"])
    if observed_delta_rows != expected_delta_rows or len(deltas) != expected_delta_rows:
        raise ValueError(
            "Prebuilt future delta row count is incomplete: "
            f"expected={expected_delta_rows}, observed={observed_delta_rows}"
        )
    delta_horizons = tuple(sorted(pd.to_numeric(deltas["horizon"]).astype(int).unique()))
    if delta_horizons != horizons:
        raise ValueError(f"Prebuilt delta horizons differ: {delta_horizons}")
    horizon_counts = deltas.groupby("horizon", observed=True)["paper_id"].size()
    if any(int(horizon_counts.get(horizon, 0)) != n_requests for horizon in horizons):
        raise ValueError("Every requested paper must have one row per horizon")
    pivot = deltas.pivot(index="paper_id", columns="horizon", values="n_future_citers")
    nested_violations = sum(
        int((pivot[lower] > pivot[upper]).fillna(False).sum())
        for lower, upper in zip(horizons, horizons[1:])
    )
    if nested_violations:
        raise ValueError(f"Prebuilt future counts are not nested: {nested_violations}")

    failure_ids = set(status.loc[status["fetch_status"].eq("failed"), "paper_id"].astype(str))
    failed_deltas = deltas["paper_id"].astype(str).isin(failure_ids)
    if int(failed_deltas.sum()) != len(failure_ids) * len(horizons):
        raise ValueError("Every failed request must have one missing row per horizon")
    if deltas.loc[failed_deltas, "n_future_citers"].notna().any():
        raise ValueError("Failed future requests must not be converted to zero")
    if "fetch_valid" not in deltas or pd.to_numeric(
        deltas.loc[failed_deltas, "fetch_valid"], errors="coerce"
    ).fillna(0).ne(0).any():
        raise ValueError("Failed future rows must have fetch_valid=0")

    success_rate = float(status["fetch_status"].eq("success").mean())
    missing_count = int(len(failure_ids))
    source_overall_pass = bool(quality.get("overall_pass") is True)
    diagnostics = quality.get("diagnostics", {})
    structural_checks = {
        "delta_rows_exact": int(quality.get("actual_delta_rows", -1))
        == int(quality.get("expected_delta_rows", -2))
        == expected_delta_rows,
        "delta_keys_unique": int(quality.get("delta_key_duplicates", -1)) == 0,
        "nested_counts_valid": int(quality.get("nested_count_violations", -1)) == 0,
        "invalid_rows_zero": int(diagnostics.get("invalid_rows", -1)) == 0,
        "label_only_no_leakage": quality.get("label_only_no_leakage") is True,
        "manifest_rows_match": (
            int(manifest.get("n_common_tau8_papers", -1)) == n_requests
            and int(manifest.get("n_fetch_status_rows", -1)) == len(status)
            and int(manifest.get("n_future_delta_rows", -1)) == len(deltas)
            and int(manifest.get("n_future_citer_rows", -1))
            == _parquet_rows(paths["future_citers.parquet"])
        ),
        "bounded_missing_checkpoints": (
            missing_count == normalized_failures
            == int(quality.get("missing_checkpoint_count", -1))
            and missing_count <= int(maximum_missing_checkpoints)
        ),
        "success_rate": success_rate >= float(minimum_success_rate),
    }
    accepted = bool(all(structural_checks.values()))
    if not accepted:
        failed = sorted(name for name, passed in structural_checks.items() if not passed)
        raise ValueError(f"Prebuilt future quality contract failed: {failed}")
    return {
        "artifact_kind": "nature_multihorizon_prebuilt_future_audit",
        "source_dir": str(root),
        "overall_pass": source_overall_pass,
        "source_overall_pass": source_overall_pass,
        "accepted_for_training": accepted,
        "acceptance_policy": (
            "preserve overall_pass=false; accept only bounded explicit "
            "missing_checkpoint failures with NA outcomes"
        ),
        "n_requests": n_requests,
        "n_status_rows": int(len(status)),
        "n_future_citer_rows": _parquet_rows(paths["future_citers.parquet"]),
        "n_future_delta_rows": int(len(deltas)),
        "horizons": list(horizons),
        "source_failure_count": missing_count,
        "source_failure_rate": float(missing_count / max(1, n_requests)),
        "success_rate": success_rate,
        "normalized_failure_count": normalized_failures,
        "nested_count_violations": nested_violations,
        "structural_checks": structural_checks,
        "status": status,
        "requests": requests,
        "deltas": deltas,
        "manifest": manifest,
        "quality": quality,
        "paths": {name: str(path) for name, path in paths.items()},
    }


def _external_parquet_signature(path: Path) -> Dict[str, Any]:
    """Return a cheap identity signature for a large immutable Parquet file."""

    source = Path(path).expanduser().resolve()
    _assert_regular_source(source)
    stat = source.stat()
    sample_size = 4 * 1024 * 1024
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        digest.update(handle.read(sample_size))
        handle.seek(max(0, stat.st_size - sample_size))
        digest.update(handle.read(sample_size))
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "head_tail_sha256": f"sha256:{digest.hexdigest()}",
        "row_count": _parquet_rows(source),
        "signature_kind": "size_mtime_head_tail",
    }


def resolve_future_citers_table(stage_dir: Path) -> Path:
    """Resolve and revalidate the canonical or externally referenced big table."""

    root = Path(stage_dir)
    local = root / "future_citers.parquet"
    if local.is_file() and not local.is_symlink():
        return local
    pointer_path = root / "future_citers.external.json"
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise FileNotFoundError(
            f"future-citers stage has no table or external pointer: {root}"
        )
    expected = _read_json(pointer_path)
    source = Path(str(expected.get("path") or ""))
    observed = _external_parquet_signature(source)
    for name in (
        "size_bytes",
        "mtime_ns",
        "head_tail_sha256",
        "row_count",
        "signature_kind",
    ):
        if observed.get(name) != expected.get(name):
            raise ValueError(f"External future-citer table identity changed: {name}")
    return source


def import_prebuilt_future_multihorizon(
    source_dir: Path,
    output_dir: Path,
    *,
    expected_horizons: Sequence[int] = (3, 5, 8),
    minimum_success_rate: float = 0.99,
    maximum_missing_checkpoints: int = 5,
) -> Dict[str, Any]:
    """Adopt completed offline multi-horizon tables into an immutable stage."""

    audit = audit_prebuilt_future_multihorizon(
        source_dir,
        expected_horizons=expected_horizons,
        minimum_success_rate=minimum_success_rate,
        maximum_missing_checkpoints=maximum_missing_checkpoints,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status = audit.pop("status")
    requests = audit.pop("requests")
    deltas = audit.pop("deltas")
    source_manifest = audit.pop("manifest")
    source_quality = audit.pop("quality")
    paths = audit.pop("paths")

    # The completed V5 adapter used a longer provenance label. Normalize it
    # to the locked V1 protocol name while retaining the exact source value.
    canonical_batch = "common_tau8_le2017"
    for frame in (status, requests):
        if "request_batch" in frame:
            frame["source_request_batch"] = frame["request_batch"].astype(str)
            frame["request_batch"] = canonical_batch
    status.to_parquet(output / "future_fetch_status.parquet", index=False)
    requests.to_parquet(output / "future_request_manifest.parquet", index=False)
    deltas = deltas.copy()
    deltas["source_fetch_status"] = deltas["fetch_status"].astype(str)
    deltas.loc[
        deltas["fetch_status"].eq("not_requested_or_failed"), "fetch_status"
    ] = "failed"
    deltas.to_parquet(
        output / "future_graph_deltas_multihorizon.parquet", index=False
    )
    # NTFS/DrvFS cannot reliably hard-link this 6.7 GB table. Keep one audited
    # immutable copy and bind it into the stage through a revalidated pointer.
    future_pointer = _external_parquet_signature(
        Path(paths["future_citers.parquet"])
    )
    _atomic_json(output / "future_citers.external.json", future_pointer)
    _atomic_json(output / "source_future_multihorizon_manifest.json", source_manifest)
    _atomic_json(output / "source_data_quality_report.json", source_quality)
    import_manifest = {
        **audit,
        "schema_version": "1.0.0",
        "overall_pass": bool(audit.get("source_overall_pass") is True),
        "source_manifest_path": paths["future_multihorizon_manifest.json"],
        "source_quality_path": paths["data_quality_report.json"],
        "future_citers_materialization": "external_read_only_reference",
        "future_citers_external": future_pointer,
        "canonical_request_batch": canonical_batch,
        "failure_is_not_zero": True,
        "precomputed_deltas_are_label_only": True,
    }
    _atomic_json(output / "future_import_audit.json", import_manifest)
    _atomic_json(output / "future_manifest.json", import_manifest)
    return import_manifest


def _column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing required column; expected one of {list(candidates)}")


def _taxonomy_value(value: Any) -> Optional[str]:
    """Return a stable OpenAlex taxonomy ID (or display-name fallback)."""
    if isinstance(value, Mapping):
        candidate = value.get("id") or value.get("display_name")
    else:
        candidate = value
    text = str(candidate or "").strip()
    return text or None


def _normalize_citer(
    paper_id: str,
    requested_horizon: int,
    publication_year: int,
    work: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    citer_id = str(work.get("citer_id") or work.get("id") or "").strip()
    year_value = work.get("citer_year", work.get("publication_year"))
    year = pd.to_numeric(year_value, errors="coerce")
    if not citer_id or pd.isna(year):
        return None
    citer_year = int(year)
    if not publication_year < citer_year <= publication_year + requested_horizon:
        return None
    # OpenAlex exposes subfield and field under ``primary_topic`` rather than
    # as top-level work fields.  Persist scalar taxonomy IDs; storing the raw
    # nested dictionary would make field/subfield reach silently collapse to
    # zero and would make topic equality depend on JSON rendering.
    topic_payload = work.get("primary_topic")
    if not isinstance(topic_payload, Mapping):
        topic_payload = {}
    subfield_payload = topic_payload.get("subfield")
    if not isinstance(subfield_payload, Mapping):
        subfield_payload = {}
    # Current OpenAlex works place ``field`` beside ``subfield`` under the
    # primary topic.  Retain the older nested fallback for cached variants.
    field_payload = topic_payload.get("field") or subfield_payload.get("field")
    primary_topic = _taxonomy_value(
        work.get("citer_primary_topic") or topic_payload
    )
    primary_subfield = _taxonomy_value(
        work.get("citer_primary_subfield")
        or work.get("primary_subfield")
        or subfield_payload
    )
    primary_field = _taxonomy_value(
        work.get("citer_primary_field")
        or work.get("primary_field")
        or field_payload
    )
    return {
        "paper_id": paper_id,
        "horizon": int(requested_horizon),
        "citer_id": citer_id,
        "citer_year": citer_year,
        "citer_primary_field": primary_field,
        "citer_primary_subfield": primary_subfield,
        "citer_primary_topic": primary_topic,
        "referenced_works": work.get("referenced_works"),
    }


def fetch_future_citers(
    papers: pd.DataFrame,
    checkpoint_root: Path,
    fetcher: FutureCiterFetcher,
    *,
    requested_horizon: int = 8,
    complete_end_year: int = 2025,
    max_citers_per_work: int = 1000,
    resume: bool = True,
    retry_failed: bool = False,
    max_papers: Optional[int] = None,
    min_publication_year: Optional[int] = None,
    max_publication_year: Optional[int] = None,
    request_batch: str = "common_tau8",
) -> Dict[str, Any]:
    """Fetch one maximum-horizon checkpoint per eligible paper.

    Every paper receives an atomic status record. Successful zero-result calls
    are represented by ``fetch_status=success`` and ``is_zero_success=true``;
    exceptions are represented by ``fetch_status=failed`` and never converted
    to a zero-citer observation.
    """
    if requested_horizon <= 0 or max_citers_per_work <= 0:
        raise ValueError("requested_horizon and max_citers_per_work must be positive")
    paper_column = _column(papers, ("paper_id", "id"))
    year_column = _column(papers, ("publication_year", "year"))
    root = Path(checkpoint_root) / f"tau{requested_horizon}"
    citer_dir = root / "citers"
    status_dir = root / "status"
    citer_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    eligible = papers.copy()
    eligible[year_column] = pd.to_numeric(eligible[year_column], errors="coerce")
    eligible = eligible[
        eligible[year_column].notna()
        & (eligible[year_column].astype(int) <= int(complete_end_year) - int(requested_horizon))
    ]
    if min_publication_year is not None:
        eligible = eligible[
            eligible[year_column].astype(int) >= int(min_publication_year)
        ]
    if max_publication_year is not None:
        eligible = eligible[
            eligible[year_column].astype(int) <= int(max_publication_year)
        ]
    if max_papers is not None:
        eligible = eligible.head(int(max_papers))

    expected = eligible[[paper_column, year_column]].rename(
        columns={paper_column: "paper_id", year_column: "publication_year"}
    )
    expected["requested_horizon"] = int(requested_horizon)
    expected["request_batch"] = str(request_batch)
    expected_path = root / "expected_requests.parquet"
    temporary_expected = expected_path.with_name(f".{expected_path.name}.tmp-{os.getpid()}")
    expected.to_parquet(temporary_expected, index=False)
    os.replace(temporary_expected, expected_path)

    counters = {"eligible": int(len(eligible)), "fetched": 0, "resumed": 0, "failed": 0, "zero_success": 0}
    for row in eligible.to_dict("records"):
        paper_id = str(row.get(paper_column) or "").strip()
        publication_year = int(row[year_column])
        if not paper_id:
            continue
        key = _safe_id(paper_id)
        status_path = status_dir / f"{key}.json"
        citer_path = citer_dir / f"{key}.jsonl"
        previous: Dict[str, Any] = _read_json(status_path) if status_path.exists() else {}
        previous_status = str(previous.get("fetch_status") or "")
        completed = previous_status == "success" and citer_path.exists()
        failed_and_locked = previous_status == "failed" and not retry_failed
        if resume and (completed or failed_and_locked):
            counters["resumed"] += 1
            continue
        attempt_count = int(previous.get("attempt_count") or 0) + 1
        start_year = publication_year + 1
        end_year = publication_year + requested_horizon
        try:
            deduplicated: Dict[str, Dict[str, Any]] = {}
            for work in fetcher(paper_id, start_year, end_year, max_citers_per_work):
                normalized = _normalize_citer(paper_id, requested_horizon, publication_year, work)
                if normalized is not None:
                    deduplicated.setdefault(str(normalized["citer_id"]), normalized)
                if len(deduplicated) >= max_citers_per_work:
                    break
            rows = sorted(deduplicated.values(), key=lambda item: (int(item["citer_year"]), str(item["citer_id"])))
            _atomic_jsonl(citer_path, rows)
            status = {
                "paper_id": paper_id,
                "requested_horizon": int(requested_horizon),
                "fetch_status": "success",
                "n_returned": int(len(rows)),
                "is_zero_success": len(rows) == 0,
                "cap_hit": len(rows) >= max_citers_per_work,
                "last_citer_year": max((int(item["citer_year"]) for item in rows), default=None),
                "error_type": None,
                "attempt_count": attempt_count,
                "publication_year": publication_year,
                "checkpoint_file": str(citer_path),
                "request_batch": str(request_batch),
            }
            _atomic_json(status_path, status)
            counters["fetched"] += 1
            counters["zero_success"] += int(len(rows) == 0)
        except Exception as exc:
            status = {
                "paper_id": paper_id,
                "requested_horizon": int(requested_horizon),
                "fetch_status": "failed",
                "n_returned": None,
                "is_zero_success": False,
                "cap_hit": None,
                "last_citer_year": None,
                "error_type": type(exc).__name__,
                "attempt_count": attempt_count,
                "publication_year": publication_year,
                "checkpoint_file": None,
                "request_batch": str(request_batch),
            }
            _atomic_json(status_path, status)
            counters["failed"] += 1
    _atomic_json(
        root / "fetch_manifest.json",
        {
            "requested_horizon": int(requested_horizon),
            "complete_end_year": int(complete_end_year),
            "max_citers_per_work": int(max_citers_per_work),
            "min_publication_year": min_publication_year,
            "max_publication_year": max_publication_year,
            "request_batch": str(request_batch),
            "counters": counters,
            "failure_is_not_zero": True,
        },
    )
    return counters


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad checkpoint JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield payload


def _write_parquet_batches(path: Path, batches: Iterable[pd.DataFrame]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    writer: Optional[pq.ParquetWriter] = None
    row_count = 0
    try:
        for frame in batches:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
        if writer is None:
            pd.DataFrame().to_parquet(temporary, index=False)
        else:
            writer.close()
            writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    return int(row_count)


def materialize_future_tables(
    checkpoint_root: Path,
    output_dir: Path,
    *,
    requested_horizon: int = 8,
    derived_horizons: Sequence[int] = (3, 5, 8),
    batch_size: int = 10_000,
) -> Dict[str, Any]:
    """Stream checkpoints into canonical Parquet tables for all horizons."""
    root = Path(checkpoint_root) / f"tau{requested_horizon}"
    status_paths = sorted((root / "status").glob("*.json"))
    horizons = tuple(sorted(set(int(value) for value in derived_horizons)))
    if not horizons or horizons[-1] > requested_horizon or horizons[0] <= 0:
        raise ValueError("derived horizons must be positive and no larger than requested_horizon")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_source = root / "expected_requests.parquet"
    if not expected_source.is_file():
        raise FileNotFoundError(f"Missing expected request manifest: {expected_source}")
    expected_frame = pd.read_parquet(expected_source)
    expected_keys = set(
        zip(
            expected_frame["paper_id"].astype(str),
            pd.to_numeric(
                expected_frame["requested_horizon"], errors="coerce"
            ).astype("Int64"),
        )
    )
    statuses = [
        status
        for path in status_paths
        if (
            (status := _read_json(path))
            and (
                str(status.get("paper_id")),
                int(status.get("requested_horizon")),
            )
            in expected_keys
        )
    ]
    status_frame = pd.DataFrame(statuses)
    status_path = output_dir / "future_fetch_status.parquet"
    status_frame.to_parquet(status_path, index=False)
    expected_path = output_dir / "future_request_manifest.parquet"
    expected_frame.to_parquet(expected_path, index=False)

    def citer_batches() -> Iterator[pd.DataFrame]:
        buffer: List[Dict[str, Any]] = []
        for status in statuses:
            if status.get("fetch_status") != "success":
                continue
            checkpoint_value = status.get("checkpoint_file")
            if not checkpoint_value:
                continue
            checkpoint = Path(str(checkpoint_value))
            publication_year = int(status["publication_year"])
            for row in _iter_jsonl(checkpoint):
                citer_year = int(row["citer_year"])
                for horizon in horizons:
                    if publication_year < citer_year <= publication_year + horizon:
                        derived = dict(row)
                        derived["horizon"] = horizon
                        buffer.append(derived)
                if len(buffer) >= batch_size:
                    yield pd.DataFrame(buffer)
                    buffer = []
        if buffer:
            yield pd.DataFrame(buffer)

    citers_path = output_dir / "future_citers.parquet"
    n_citers = _write_parquet_batches(citers_path, citer_batches())
    manifest = {
        "requested_horizon": int(requested_horizon),
        "derived_horizons": list(horizons),
        "n_status_rows": int(len(status_frame)),
        "n_expected_requests": int(len(expected_frame)),
        "n_citer_rows": int(n_citers),
        "future_fetch_status": str(status_path),
        "future_request_manifest": str(expected_path),
        "future_citers": str(citers_path),
    }
    _atomic_json(output_dir / "future_manifest.json", manifest)
    return manifest


def merge_materialized_future_batches(
    batch_dirs: Sequence[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    """Merge disjoint publication-year request batches into canonical tables.

    Batches must be disjoint by paper ID. This makes the merge streaming and
    prevents an older, shorter request from silently shadowing a τ=8 result.
    """

    directories = [Path(path) for path in batch_dirs]
    if not directories:
        raise ValueError("At least one materialized future batch is required")
    statuses: List[pd.DataFrame] = []
    requests: List[pd.DataFrame] = []
    for directory in directories:
        status_path = directory / "future_fetch_status.parquet"
        request_path = directory / "future_request_manifest.parquet"
        citer_path = directory / "future_citers.parquet"
        for path in (status_path, request_path, citer_path):
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete future batch: {path}")
        status = pd.read_parquet(status_path)
        request = pd.read_parquet(request_path)
        status["materialized_batch"] = directory.name
        request["materialized_batch"] = directory.name
        statuses.append(status)
        requests.append(request)

    status_frame = pd.concat(statuses, ignore_index=True)
    request_frame = pd.concat(requests, ignore_index=True)
    request_key = ["paper_id", "requested_horizon"]
    if request_frame.duplicated(request_key).any():
        raise ValueError("Future request batches overlap on paper_id/requested_horizon")
    if request_frame["paper_id"].astype(str).duplicated().any():
        raise ValueError("Future request batches must use disjoint publication cohorts")
    if status_frame.duplicated(request_key).any():
        raise ValueError("Future status batches overlap on paper_id/requested_horizon")
    expected_keys = set(map(tuple, request_frame[request_key].astype(str).to_numpy()))
    actual_keys = set(map(tuple, status_frame[request_key].astype(str).to_numpy()))
    if expected_keys != actual_keys:
        missing = len(expected_keys - actual_keys)
        extra = len(actual_keys - expected_keys)
        raise ValueError(
            f"Future batch status/request mismatch; missing={missing}, extra={extra}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "future_fetch_status.parquet"
    request_path = output / "future_request_manifest.parquet"
    status_frame.sort_values(request_key, kind="stable").to_parquet(
        status_path, index=False
    )
    request_frame.sort_values(request_key, kind="stable").to_parquet(
        request_path, index=False
    )

    def batches() -> Iterator[pd.DataFrame]:
        for directory in directories:
            parquet = pq.ParquetFile(directory / "future_citers.parquet")
            for batch in parquet.iter_batches(batch_size=50_000):
                frame = batch.to_pandas()
                if not frame.empty:
                    frame["materialized_batch"] = directory.name
                    yield frame

    citer_path = output / "future_citers.parquet"
    n_citers = _write_parquet_batches(citer_path, batches())
    manifest = {
        "artifact_kind": "nature_multihorizon_future_batches",
        "batches": [path.name for path in directories],
        "n_expected_requests": int(len(request_frame)),
        "n_status_rows": int(len(status_frame)),
        "n_citer_rows": int(n_citers),
        "publication_cohorts_disjoint": True,
        "failure_is_not_zero": True,
        "future_fetch_status": str(status_path),
        "future_request_manifest": str(request_path),
        "future_citers": str(citer_path),
    }
    _atomic_json(output / "future_manifest.json", manifest)
    return manifest
