"""Resolve and download licensed OA manuscripts for a frozen OOF cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

OPENALEX_API = "https://api.openalex.org/works"
ALLOWED_LICENSE_TOKENS = (
    "cc-by",
    "cc0",
    "public-domain",
    "public domain",
)
PREFERRED_VERSIONS = ("submittedVersion", "acceptedVersion", "publishedVersion")


def acquire_manuscripts(
    cohort_path: Path,
    output_dir: Path,
    *,
    download: bool = True,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    """Resolve exact OpenAlex IDs, select licensed OA versions, and checkpoint PDFs."""
    output_dir = Path(output_dir).resolve()
    manuscript_dir = output_dir / "manuscripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(cohort_path)
    if "paper_id" not in cohort or cohort["paper_id"].duplicated().any():
        raise ValueError("cohort must contain unique paper_id values")
    session = requests.Session()
    session.headers["User-Agent"] = "ASPR-GEAR-research-validation/1.0"
    api_keys = _openalex_api_keys()
    paper_ids = cohort["paper_id"].astype(str).tolist()
    resolved = _resolve_many(
        session,
        paper_ids,
        api_keys=api_keys,
        timeout_seconds=timeout_seconds,
    )
    records: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        record = resolved[paper_id]
        if download and record["eligibility_status"] == "licensed_direct_pdf":
            record.update(
                _download_pdf(
                    session,
                    record,
                    manuscript_dir,
                    timeout_seconds=timeout_seconds,
                )
            )
        records.append(record)
    _write_jsonl(output_dir / "acquisition_manifest.jsonl", records)
    manifest = pd.DataFrame(records)
    manifest.to_csv(output_dir / "acquisition_manifest.csv", index=False)
    summary = _summary(manifest)
    (output_dir / "acquisition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def download_resolved_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: int = 45,
) -> dict[str, Any]:
    """Download only pre-resolved licensed rows without repeating API queries."""
    output_dir = Path(output_dir).resolve()
    manuscript_dir = output_dir / "manuscripts"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path)
    records = frame.where(pd.notna(frame), None).to_dict(orient="records")
    eligible = [
        record
        for record in records
        if record["eligibility_status"] == "licensed_direct_pdf"
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                _download_one_record,
                record,
                manuscript_dir,
                timeout_seconds,
            ): record
            for record in eligible
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            futures[future].update(future.result())
            if completed % 20 == 0 or completed == len(futures):
                _write_jsonl(output_dir / "acquisition_manifest.jsonl", records)
    updated = pd.DataFrame(records)
    updated.to_csv(output_dir / "acquisition_manifest.csv", index=False)
    summary = _summary(updated)
    (output_dir / "acquisition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _download_one_record(
    record: dict[str, Any], manuscript_dir: Path, timeout_seconds: int
) -> dict[str, Any]:
    session = requests.Session()
    session.headers["User-Agent"] = "ASPR-GEAR-research-validation/1.0"
    return _download_pdf(
        session, record, manuscript_dir, timeout_seconds=timeout_seconds
    )


def _resolve_many(
    session: requests.Session,
    paper_ids: list[str],
    *,
    api_keys: list[str],
    timeout_seconds: int,
) -> dict[str, dict[str, Any]]:
    params: dict[str, str | int] = {
        "select": (
            "id,doi,title,type,publication_date,publication_year,"
            "open_access,best_oa_location,locations"
        )
    }
    output: dict[str, dict[str, Any]] = {}
    for start in range(0, len(paper_ids), 50):
        batch = paper_ids[start : start + 50]
        short_ids = [value.rstrip("/").rsplit("/", 1)[-1] for value in batch]
        batch_params = {
            **params,
            "filter": f"openalex_id:{'|'.join(short_ids)}",
            "per_page": len(batch),
        }
        if api_keys:
            batch_params["api_key"] = api_keys[(start // 50) % len(api_keys)]
        try:
            response = session.get(
                OPENALEX_API, params=batch_params, timeout=timeout_seconds
            )
            response.raise_for_status()
            works = response.json().get("results", [])
        except (requests.RequestException, ValueError) as exc:
            for paper_id in batch:
                output[paper_id] = _base_record(
                    paper_id,
                    metadata_status=f"error:{type(exc).__name__}",
                    eligibility_status="metadata_unavailable",
                )
            continue
        by_id = {str(work.get("id")): work for work in works}
        for paper_id in batch:
            work = by_id.get(paper_id)
            output[paper_id] = (
                _record_from_work(paper_id, work)
                if work is not None
                else _base_record(
                    paper_id,
                    metadata_status="not_returned",
                    eligibility_status="metadata_unavailable",
                )
            )
        time.sleep(0.1)
    return output


def _record_from_work(paper_id: str, work: dict[str, Any]) -> dict[str, Any]:
    if str(work.get("id")) != paper_id:
        return _base_record(
            paper_id,
            metadata_status="identity_mismatch",
            eligibility_status="metadata_unavailable",
        )
    location = _select_location(work.get("locations") or [])
    record = _base_record(
        paper_id,
        metadata_status="resolved",
        eligibility_status="no_eligible_oa_location",
    )
    record.update(
        {
            "doi": work.get("doi"),
            "title": work.get("title"),
            "document_type": work.get("type"),
            "publication_year_resolved": work.get("publication_year"),
            "publication_date_resolved": work.get("publication_date"),
            "is_oa": bool((work.get("open_access") or {}).get("is_oa")),
            "metadata_sha256": _sha256_json(work),
        }
    )
    if location is None:
        return record
    license_value = str(location.get("license") or "")
    pdf_url = location.get("pdf_url")
    record.update(
        {
            "source_version": location.get("version"),
            "source_license": license_value or None,
            "source_pdf_url": pdf_url,
            "source_landing_page_url": location.get("landing_page_url"),
            "source_host_type": (location.get("source") or {}).get("type"),
        }
    )
    if not pdf_url:
        record["eligibility_status"] = "oa_without_direct_pdf"
    elif not _license_allowed(license_value):
        record["eligibility_status"] = "license_requires_manual_review"
    else:
        record["eligibility_status"] = "licensed_direct_pdf"
    return record


def _select_location(locations: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [location for location in locations if location.get("is_oa")]
    if not eligible:
        return None

    def rank(location: dict[str, Any]) -> tuple[int, int, int, str]:
        version = str(location.get("version") or "")
        try:
            version_rank = PREFERRED_VERSIONS.index(version)
        except ValueError:
            version_rank = len(PREFERRED_VERSIONS)
        license_rank = 0 if _license_allowed(str(location.get("license") or "")) else 1
        pdf_rank = 0 if location.get("pdf_url") else 1
        return (
            license_rank,
            version_rank,
            pdf_rank,
            str(location.get("landing_page_url") or ""),
        )

    return min(eligible, key=rank)


def _download_pdf(
    session: requests.Session,
    record: dict[str, Any],
    manuscript_dir: Path,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    url = str(record["source_pdf_url"])
    identifier = str(record["paper_id"]).rstrip("/").rsplit("/", 1)[-1]
    path = manuscript_dir / f"{identifier}.pdf"
    if path.is_file() and path.read_bytes()[:5] == b"%PDF-":
        return _download_result(path, "downloaded")
    try:
        response = session.get(url, timeout=timeout_seconds, allow_redirects=True)
        response.raise_for_status()
        payload = response.content
    except requests.RequestException as exc:
        return {
            "download_status": f"error:{type(exc).__name__}",
            "manuscript_path": None,
        }
    if len(payload) > 50_000_000:
        return {"download_status": "rejected:too_large", "manuscript_path": None}
    if payload[:5] != b"%PDF-":
        return {"download_status": "rejected:not_pdf", "manuscript_path": None}
    path.write_bytes(payload)
    return _download_result(path, "downloaded")


def _download_result(path: Path, status: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "download_status": status,
        "manuscript_path": str(path),
        "manuscript_bytes": len(payload),
        "manuscript_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def _base_record(
    paper_id: str, *, metadata_status: str, eligibility_status: str
) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "metadata_status": metadata_status,
        "eligibility_status": eligibility_status,
        "download_status": "not_attempted",
        "manuscript_path": None,
        "source_domain": None,
    }


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    downloaded = frame["download_status"].eq("downloaded")
    return {
        "contract": "gear_oof_manuscript_acquisition_v1",
        "papers": len(frame),
        "metadata_resolved": int(frame["metadata_status"].eq("resolved").sum()),
        "open_access": int(
            frame.get("is_oa", pd.Series(False, index=frame.index)).fillna(False).sum()
        ),
        "licensed_direct_pdf": int(
            frame["eligibility_status"].eq("licensed_direct_pdf").sum()
        ),
        "downloaded_pdfs": int(downloaded.sum()),
        "downloaded_domains": sorted(
            {
                str(urlparse(str(url)).netloc)
                for url in frame.loc[downloaded, "source_pdf_url"].dropna()
            }
        ),
        "ready_for_gear": int(downloaded.sum()),
    }


def _license_allowed(value: str) -> bool:
    normalized = value.casefold()
    return any(token in normalized for token in ALLOWED_LICENSE_TOKENS)


def _openalex_api_keys() -> list[str]:
    load_dotenv(Path(".env"), override=False)
    combined = os.environ.get("OPENALEX_API_KEYS", "")
    singular = os.environ.get("OPENALEX_API_KEY", "")
    values = [*combined.split(","), singular]
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--download-resolved", action="store_true")
    args = parser.parse_args()
    summary = (
        download_resolved_manifest(
            args.output_dir / "acquisition_manifest.csv", args.output_dir
        )
        if args.download_resolved
        else acquire_manuscripts(
            args.cohort, args.output_dir, download=not args.resolve_only
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
