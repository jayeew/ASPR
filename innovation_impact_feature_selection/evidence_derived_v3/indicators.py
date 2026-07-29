from __future__ import annotations

import json
import re
import sqlite3
from ipaddress import ip_address
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from common import (
    ROOT,
    cohen_kappa,
    gwet_ac1,
    iter_csv,
    normalize_text,
    normalize_term,
    parse_bool,
    raw_agreement,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_csv,
    write_csv_iter,
    write_json,
)
from database import (
    invalidate_stages,
    log_event,
    require_complete,
    set_stage,
    snapshot_import_file,
)
from providers import hydrate_openalex_locations
from screening import included_record_keys


RULES_PATH = ROOT / "screening_rules_v3.json"
SATURATION_PROTOCOL_PATH = ROOT / "saturation_protocol_v3.json"
OUTPUT_DIR = ROOT / "outputs"
INDICATOR_DISPOSITIONS = {
    "extracted",
    "no_indicator",
    "candidate_fulltext_missing",
    "excluded_after_fulltext",
}
SOURCE_ROLES = {
    "original_definition",
    "original_application",
    "validation",
    "review_discovery",
    "mathematical_foundation",
}
SCOPE_ROLES = {
    "direct_innovation",
    "t0_substantive",
    "t0_opportunity",
    "context_control",
    "outcome_only",
    "out_of_scope",
}
CONSTRUCT_ROLES = {
    "substantive_innovation",
    "t0_potential",
    "opportunity",
    "context_control",
    "sensitivity",
}
INDICATOR_FIELDS = (
    "record_key",
    "doi",
    "source_title",
    "source_url",
    "candidate_fulltext_url",
    "source_disposition",
    "english_fulltext_status",
    "disposition_notes",
    "disposition_decided_by",
    "mention_id",
    "raw_name_en",
    "canonical_name_en",
    "label_zh",
    "source_id",
    "research_group",
    "research_group_id",
    "research_group_evidence",
    "source_role",
    "formula_location",
    "evidence_span",
    "formula",
    "units",
    "parameters",
    "direction",
    "missing_rule",
    "required_data",
    "maximum_information_time",
    "scope_role",
    "validation_summary",
    "evidence_direction",
    "negative_evidence",
    "fulltext_source_url",
    "fulltext_local_path",
    "fulltext_sha256",
    "fulltext_license",
    "english_fulltext_verified",
    "article_level",
    "primary_or_foundational_evidence",
    "formula_reproducible",
    "t0_computable",
    "requires_future",
    "data_status",
    "bias_policy",
    "fatal_validity_concern",
    "uses_outcome_for_selection",
    "quality_audit_status",
    "nonconstant",
    "h2_approved",
    "evidence_strength",
    "stability_score",
    "stability_basis",
    "selection_priority",
    "redundancy_family",
    "extracted_by",
    "verified_by",
    "verification_notes",
    "adjudication_notes",
    "status",
)
DIMENSION_FIELDS = (
    "feature_id",
    "canonical_name_en",
    "coder_role",
    "dimension_label",
    "dimension_definition",
    "construct_role",
    "information_source",
    "t0_boundary",
    "bias_risk",
    "decision",
    "reason",
)
DIMENSION_EVIDENCE_FIELDS = (
    *DIMENSION_FIELDS,
    "formula",
    "required_data",
    "maximum_information_time_evidence",
    "scope_role_evidence",
    "research_groups_evidence",
    "mention_ids_evidence",
)
DIMENSION_H2_FIELDS = (
    *DIMENSION_EVIDENCE_FIELDS,
    *(
        f"{role}_{field}"
        for role in ("ai", "h1")
        for field in (
            "dimension_label",
            "dimension_definition",
            "construct_role",
            "information_source",
            "t0_boundary",
            "bias_risk",
            "decision",
            "reason",
        )
    ),
)
DATA_AUDIT_FIELDS = (
    "feature_id",
    "canonical_name_en",
    "data_status",
    "row_count",
    "valid_count",
    "unique_count",
    "missing_rate",
    "derivation_artifact_path",
    "input_snapshot_path",
    "derivation_hash",
    "input_snapshot_hash",
    "audit_status",
    "reviewer",
    "notes",
)
_FULLTEXT_TEXT_CACHE: Dict[tuple[str, str], str] = {}


def _safe_public_http_url(value: str) -> str:
    """Accept only public HTTP(S) URLs for automated acquisition."""
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Full-text URL is not HTTP(S): {url}")
    hostname = parsed.hostname.casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError(f"Local full-text URL is prohibited: {url}")
    try:
        address = ip_address(hostname)
    except ValueError:
        return url
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    ):
        raise ValueError(f"Non-public full-text URL is prohibited: {url}")
    return url


def _open_pdf_candidate(raw: Mapping[str, Any]) -> tuple[str, str]:
    """Return the first OpenAlex location explicitly marked open access."""
    locations: List[Mapping[str, Any]] = []
    for key in ("best_oa_location", "primary_location"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            locations.append(value)
    raw_locations = raw.get("locations")
    if isinstance(raw_locations, list):
        locations.extend(
            value for value in raw_locations if isinstance(value, Mapping)
        )
    seen: set[str] = set()
    for location in locations:
        url = str(location.get("pdf_url") or "").strip()
        if not url or url in seen or location.get("is_oa") is not True:
            continue
        seen.add(url)
        try:
            safe_url = _safe_public_http_url(url)
        except ValueError:
            continue
        license_value = str(location.get("license") or "").strip()
        access_statement = (
            license_value
            or "OpenAlex location is_oa=true; licence not reported"
        )
        return safe_url, access_statement
    return "", ""


def _download_open_pdf(
    url: str,
    timeout_seconds: int,
    maximum_bytes: int,
) -> Dict[str, Any]:
    """Download one bounded PDF and return its final response metadata."""
    request = Request(
        _safe_public_http_url(url),
        headers={
            "User-Agent": (
                "ASPR-evidence-derived-v3/3.3 "
                "(open-access evidence acquisition)"
            )
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        body = response.read(maximum_bytes + 1)
        final_url = _safe_public_http_url(str(response.geturl()))
        content_type = str(
            response.headers.get_content_type() or ""
        ).casefold()
    if len(body) > maximum_bytes:
        raise ValueError(
            f"Open full text exceeds {maximum_bytes} bytes"
        )
    if not body.lstrip().startswith(b"%PDF-"):
        raise ValueError(
            f"Open full text is not a PDF (content-type={content_type})"
        )
    return {
        "body": body,
        "final_url": final_url,
        "content_type": content_type,
    }


def _store_fulltext_acquisition(
    connection: sqlite3.Connection,
    values: Mapping[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO fulltext_acquisitions(
            record_key, candidate_url, final_url, local_path, sha256,
            access_statement, http_content_type, byte_count, status,
            error, fetched_at
        ) VALUES (
            :record_key, :candidate_url, :final_url, :local_path,
            :sha256, :access_statement, :http_content_type,
            :byte_count, :status, :error, :fetched_at
        )
        ON CONFLICT(record_key) DO UPDATE SET
            candidate_url = excluded.candidate_url,
            final_url = excluded.final_url,
            local_path = excluded.local_path,
            sha256 = excluded.sha256,
            access_statement = excluded.access_statement,
            http_content_type = excluded.http_content_type,
            byte_count = excluded.byte_count,
            status = excluded.status,
            error = excluded.error,
            fetched_at = excluded.fetched_at
        """,
        dict(values),
    )


def acquire_open_fulltexts(
    connection: sqlite3.Connection,
    output_dir: Path,
    maximum_records: int | None = None,
    retry_failed: bool = False,
    timeout_seconds: int = 60,
    maximum_bytes: int = 100_000_000,
    fetcher: Callable[[str, int, int], Mapping[str, Any]] | None = None,
    hydrate_locations: bool = False,
    location_fetcher: Callable[[str], Mapping[str, Any]] | None = None,
) -> Dict[str, int]:
    """Acquire explicitly open PDFs for included English source papers."""
    require_complete(connection, ["literature_screened"])
    if maximum_records is not None and maximum_records < 1:
        raise ValueError("maximum_records must be at least one")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be at least one")
    if maximum_bytes < 1024:
        raise ValueError("maximum_bytes must be at least 1024")
    output_dir.mkdir(parents=True, exist_ok=True)
    download = fetcher or _download_open_pdf
    included_keys = included_record_keys(connection)
    hydration = {
        "hydrated": 0,
        "resumed": 0,
        "failed": 0,
        "unconfigured": 0,
    }
    if hydrate_locations:
        hydration_kwargs: Dict[str, Any] = {
            "retry_failed": retry_failed,
        }
        if location_fetcher is not None:
            hydration_kwargs["fetcher"] = location_fetcher
        hydration = hydrate_openalex_locations(
            connection,
            included_keys,
            **hydration_kwargs,
        )
    counts = {
        "downloaded": 0,
        "resumed": 0,
        "failed": 0,
        "without_open_pdf": 0,
        "locations_hydrated": int(hydration["hydrated"]),
        "location_hydration_resumed": int(hydration["resumed"]),
        "location_hydration_failed": int(hydration["failed"]),
        "location_hydration_unconfigured": int(
            hydration["unconfigured"]
        ),
    }
    attempts = 0
    records = connection.execute(
        """
        SELECT r.record_key, r.raw_json, a.*
        FROM records r
        JOIN screening_final s USING(record_key)
        LEFT JOIN fulltext_acquisitions a USING(record_key)
        WHERE s.final_decision = 'include'
          AND s.final_language = 'en'
        ORDER BY r.record_key
        """
    )
    for record in records:
        raw = json.loads(record["raw_json"])
        candidate_url, access_statement = _open_pdf_candidate(raw)
        if not candidate_url:
            _store_fulltext_acquisition(
                connection,
                {
                    "record_key": record["record_key"],
                    "candidate_url": "",
                    "final_url": "",
                    "local_path": "",
                    "sha256": "",
                    "access_statement": (
                        "No OpenAlex location explicitly marked "
                        "is_oa=true with a PDF URL"
                    ),
                    "http_content_type": "",
                    "byte_count": 0,
                    "status": "no_open_pdf",
                    "error": "",
                    "fetched_at": utc_now(),
                },
            )
            connection.commit()
            counts["without_open_pdf"] += 1
            continue
        existing_path = Path(str(record["local_path"] or ""))
        if (
            record["status"] == "downloaded"
            and record["candidate_url"] == candidate_url
            and existing_path.is_file()
            and sha256_file(existing_path) == record["sha256"]
        ):
            counts["resumed"] += 1
            continue
        if (
            record["status"] == "failed"
            and record["candidate_url"] == candidate_url
            and not retry_failed
        ):
            counts["failed"] += 1
            continue
        if maximum_records is not None and attempts >= maximum_records:
            break
        attempts += 1
        identity_hash = sha256_bytes(
            str(record["record_key"]).encode("utf-8")
        )[:20]
        local_path = (output_dir / f"{identity_hash}.pdf").resolve()
        temporary_path = local_path.with_suffix(".pdf.part")
        try:
            result = dict(
                download(candidate_url, timeout_seconds, maximum_bytes)
            )
            body = bytes(result["body"])
            if len(body) > maximum_bytes or not body.lstrip().startswith(
                b"%PDF-"
            ):
                raise ValueError("Fetcher returned an invalid PDF payload")
            temporary_path.write_bytes(body)
            temporary_path.replace(local_path)
            digest = sha256_file(local_path)
            _store_fulltext_acquisition(
                connection,
                {
                    "record_key": record["record_key"],
                    "candidate_url": candidate_url,
                    "final_url": _safe_public_http_url(
                        str(result.get("final_url") or candidate_url)
                    ),
                    "local_path": str(local_path),
                    "sha256": digest,
                    "access_statement": access_statement,
                    "http_content_type": str(
                        result.get("content_type") or ""
                    ),
                    "byte_count": len(body),
                    "status": "downloaded",
                    "error": "",
                    "fetched_at": utc_now(),
                },
            )
            connection.execute(
                """
                INSERT INTO source_snapshots(
                    source_id, path, sha256, role, imported_at
                ) VALUES (?, ?, ?, 'candidate_open_fulltext', ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    path = excluded.path,
                    sha256 = excluded.sha256,
                    role = excluded.role,
                    imported_at = excluded.imported_at
                """,
                (
                    f"candidate_fulltext_{identity_hash}",
                    str(local_path),
                    digest,
                    utc_now(),
                ),
            )
            counts["downloaded"] += 1
        except (KeyError, OSError, TypeError, ValueError) as error:
            if temporary_path.exists():
                temporary_path.unlink()
            _store_fulltext_acquisition(
                connection,
                {
                    "record_key": record["record_key"],
                    "candidate_url": candidate_url,
                    "final_url": "",
                    "local_path": "",
                    "sha256": "",
                    "access_statement": access_statement,
                    "http_content_type": "",
                    "byte_count": 0,
                    "status": "failed",
                    "error": str(error)[:1000],
                    "fetched_at": utc_now(),
                },
            )
            counts["failed"] += 1
        connection.commit()
    return counts


def export_indicator_extraction(
    connection: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export one source row that can be duplicated for multiple indicators."""
    require_complete(connection, ["literature_screened"])
    def iter_rows() -> Iterable[Dict[str, Any]]:
        for record in connection.execute(
            """
            SELECT r.record_key, r.doi, r.title, r.source_url, r.raw_json,
                   a.candidate_url AS acquired_candidate_url,
                   a.final_url AS acquired_final_url,
                   a.local_path AS acquired_local_path,
                   a.sha256 AS acquired_sha256,
                   a.access_statement AS acquired_access_statement,
                   a.status AS acquisition_status
            FROM records r
            JOIN screening_final s USING(record_key)
            LEFT JOIN fulltext_acquisitions a USING(record_key)
            LEFT JOIN indicator_source_reviews h1
              ON h1.record_key = r.record_key
             AND h1.reviewer_role = 'H1'
            WHERE s.final_decision = 'include'
              AND s.final_language = 'en'
              AND h1.record_key IS NULL
            ORDER BY r.record_key
            """
        ):
            raw = json.loads(record["raw_json"])
            candidate_fulltext_url, access_statement = (
                _open_pdf_candidate(raw)
            )
            acquisition_downloaded = (
                record["acquisition_status"] == "downloaded"
            )
            if acquisition_downloaded:
                candidate_fulltext_url = str(
                    record["acquired_candidate_url"]
                )
                access_statement = str(
                    record["acquired_access_statement"]
                )
            row = {field: "" for field in INDICATOR_FIELDS}
            row.update(
                {
                    "record_key": record["record_key"],
                    "doi": record["doi"],
                    "source_title": record["title"],
                    "source_url": record["source_url"],
                    "candidate_fulltext_url": candidate_fulltext_url,
                    "fulltext_source_url": (
                        str(record["acquired_final_url"])
                        if acquisition_downloaded
                        else candidate_fulltext_url
                    ),
                    "fulltext_local_path": (
                        str(record["acquired_local_path"])
                        if acquisition_downloaded
                        else ""
                    ),
                    "fulltext_sha256": (
                        str(record["acquired_sha256"])
                        if acquisition_downloaded
                        else ""
                    ),
                    "fulltext_license": access_statement,
                    "source_disposition": "",
                    "english_fulltext_status": "",
                    "disposition_decided_by": "H1",
                    "extracted_by": "H1",
                    "verified_by": "H1",
                    "h2_approved": "false",
                    "status": "candidate",
                }
            )
            yield row

    return write_csv_iter(output_path, iter_rows(), INDICATOR_FIELDS)


def export_indicator_adjudication(
    connection: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export H1 evidence plus blank H2 decisions for every included source."""
    require_complete(connection, ["literature_screened"])

    def iter_rows() -> Iterable[Dict[str, Any]]:
        records = connection.execute(
            """
            SELECT r.record_key, r.doi, r.title, r.source_url,
                   a.candidate_url, h1.disposition,
                   h1.english_fulltext_status,
                   h1.notes AS disposition_notes
            FROM records r
            JOIN screening_final s USING(record_key)
            JOIN indicator_source_reviews h1
              ON h1.record_key = r.record_key
             AND h1.reviewer_role = 'H1'
            LEFT JOIN indicator_source_reviews h2
              ON h2.record_key = r.record_key
             AND h2.reviewer_role = 'H2'
            LEFT JOIN fulltext_acquisitions a USING(record_key)
            WHERE s.final_decision = 'include'
              AND s.final_language = 'en'
              AND (
                    h2.record_key IS NULL
                    OR EXISTS (
                        SELECT 1
                        FROM indicator_mention_reviews mh1
                        LEFT JOIN indicator_mention_reviews mh2
                          ON mh2.mention_id = mh1.mention_id
                         AND mh2.reviewer_role = 'H2'
                        WHERE mh1.reviewer_role = 'H1'
                          AND mh1.mention_id IN (
                              SELECT mention_id
                              FROM indicator_mentions
                              WHERE record_key = r.record_key
                          )
                          AND mh2.mention_id IS NULL
                    )
              )
            ORDER BY r.record_key
            """
        )
        boolean_fields = (
            "english_fulltext_verified",
            "article_level",
            "primary_or_foundational_evidence",
            "formula_reproducible",
            "t0_computable",
            "requires_future",
            "fatal_validity_concern",
            "uses_outcome_for_selection",
            "nonconstant",
        )
        for record in records:
            base = {field: "" for field in INDICATOR_FIELDS}
            base.update(
                {
                    "record_key": record["record_key"],
                    "doi": record["doi"],
                    "source_title": record["title"],
                    "source_url": record["source_url"],
                    "candidate_fulltext_url": (
                        record["candidate_url"] or ""
                    ),
                    "source_disposition": record["disposition"],
                    "english_fulltext_status": record[
                        "english_fulltext_status"
                    ],
                    "disposition_notes": record["disposition_notes"],
                    "disposition_decided_by": "H1|H2",
                }
            )
            mentions = connection.execute(
                """
                SELECT m.* FROM indicator_mentions m
                JOIN indicator_mention_reviews h1
                  ON h1.mention_id = m.mention_id
                 AND h1.reviewer_role = 'H1'
                WHERE m.record_key = ?
                ORDER BY m.mention_id
                """,
                (record["record_key"],),
            ).fetchall()
            if not mentions:
                yield base
                continue
            for mention in mentions:
                row = dict(base)
                for field in (
                    "mention_id",
                    "raw_name_en",
                    "canonical_name_en",
                    "label_zh",
                    "source_id",
                    "research_group",
                    "research_group_id",
                    "research_group_evidence",
                    "source_role",
                    "formula_location",
                    "evidence_span",
                    "formula",
                    "units",
                    "parameters",
                    "direction",
                    "missing_rule",
                    "maximum_information_time",
                    "scope_role",
                    "validation_summary",
                    "evidence_direction",
                    "negative_evidence",
                    "fulltext_source_url",
                    "fulltext_local_path",
                    "fulltext_sha256",
                    "fulltext_license",
                    "data_status",
                    "bias_policy",
                    "quality_audit_status",
                    "evidence_strength",
                    "stability_score",
                    "stability_basis",
                    "selection_priority",
                    "redundancy_family",
                    "extracted_by",
                    "verification_notes",
                    "status",
                ):
                    row[field] = mention[field]
                row["required_data"] = mention["required_data_json"]
                for field in boolean_fields:
                    row[field] = (
                        "true" if bool(mention[field]) else "false"
                    )
                row["h2_approved"] = ""
                verified = _review_roles(mention["verified_by"])
                verified.add("H2")
                row["verified_by"] = "|".join(sorted(verified))
                row["adjudication_notes"] = ""
                yield row

    return write_csv_iter(output_path, iter_rows(), INDICATOR_FIELDS)


def _parse_required_data(value: str) -> List[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("required_data JSON must be an array")
        return sorted({str(item).strip() for item in parsed if str(item).strip()})
    return sorted(
        {
            item.strip()
            for item in re.split(r"[|;]", stripped)
            if item.strip()
        }
    )


def _mention_id(row: Mapping[str, Any]) -> str:
    identity = "|".join(
        (
            str(row.get("record_key") or ""),
            str(row.get("canonical_name_en") or ""),
            str(row.get("formula_location") or ""),
            str(row.get("raw_name_en") or ""),
        )
    )
    return "MENTION_" + sha256_bytes(
        identity.encode("utf-8")
    )[:16].upper()


def _register_fulltext_evidence(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    mention_id: str,
    identity: str,
) -> Dict[str, str]:
    """Hash and freeze the exact lawful full-text version reviewed."""
    source_url = _required_text(
        row,
        "fulltext_source_url",
        identity,
    )
    source_url = _safe_public_http_url(source_url)
    local_path = Path(
        _required_text(row, "fulltext_local_path", identity)
    ).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Full-text evidence file does not exist: {local_path}"
        )
    license_value = _required_text(
        row,
        "fulltext_license",
        identity,
    )
    computed_hash = sha256_file(local_path)
    claimed_hash = str(row.get("fulltext_sha256") or "").strip().casefold()
    if claimed_hash and claimed_hash != computed_hash:
        raise ValueError(
            f"Full-text SHA-256 mismatch for {identity}: {local_path}"
        )
    evidence_span = _required_text(row, "evidence_span", identity)
    fulltext = _extract_fulltext_text(local_path, computed_hash)
    normalized_fulltext = normalize_text(fulltext)
    normalized_span = normalize_text(evidence_span)
    if len(normalized_span) < 8 or len(normalized_span.split()) < 2:
        raise ValueError(
            f"Full-text evidence span is too short to audit: {identity}"
        )
    if normalized_span not in normalized_fulltext:
        raise ValueError(
            f"Evidence span is not present in the frozen full text: "
            f"{identity}"
        )
    group_evidence = _required_text(
        row,
        "research_group_evidence",
        identity,
    )
    normalized_group_evidence = normalize_text(group_evidence)
    if (
        len(normalized_group_evidence) < 8
        or len(normalized_group_evidence.split()) < 2
    ):
        raise ValueError(
            f"Research-group evidence span is too short: {identity}"
        )
    if normalized_group_evidence not in normalized_fulltext:
        raise ValueError(
            "Research-group author/affiliation evidence is not present in "
            f"the frozen full text: {identity}"
        )
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, 'indicator_fulltext_evidence', ?)
        ON CONFLICT(source_id) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            role = excluded.role,
            imported_at = excluded.imported_at
        """,
        (
            f"indicator_fulltext_{mention_id}",
            str(local_path),
            computed_hash,
            utc_now(),
        ),
    )
    return {
        "fulltext_source_url": source_url,
        "fulltext_local_path": str(local_path),
        "fulltext_sha256": computed_hash,
        "fulltext_license": license_value,
    }


def _extract_fulltext_text(path: Path, digest: str) -> str:
    """Extract auditable text from a frozen text or PDF evidence file."""
    cache_key = (str(path), digest)
    cached = _FULLTEXT_TEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    header = path.read_bytes()[:5]
    if path.suffix.casefold() == ".pdf" or header == b"%PDF":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError(
                "pypdf is required to verify PDF evidence spans"
            ) from error
        try:
            reader = PdfReader(str(path))
            text = "\n".join(
                str(page.extract_text() or "") for page in reader.pages
            )
        except Exception as error:
            raise ValueError(
                f"Could not extract text from full-text PDF: {path}"
            ) from error
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Full-text evidence is neither PDF nor UTF-8 text: {path}"
            ) from error
    if not normalize_text(text):
        raise ValueError(f"Full-text evidence has no extractable text: {path}")
    _FULLTEXT_TEXT_CACHE[cache_key] = text
    return text


def _required_text(
    row: Mapping[str, Any],
    field: str,
    identity: str,
) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{identity} requires {field}")
    return value


def _review_roles(value: Any) -> set[str]:
    return {
        token.strip().upper()
        for token in re.split(r"[|;,]", str(value or ""))
        if token.strip()
    }


def _indicator_review_payload(row: Mapping[str, Any]) -> str:
    """Serialize one submitted mention row without collapsing reviewer roles."""
    payload = {
        field: str(row.get(field) or "").strip()
        for field in INDICATOR_FIELDS
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _store_source_review(
    connection: sqlite3.Connection,
    record_key: str,
    reviewer_role: str,
    disposition: str,
    fulltext_status: str,
    notes: str,
) -> None:
    """Store H1/H2 source reviews separately and refresh the resolved row."""
    if reviewer_role not in {"H1", "H2"}:
        raise ValueError(f"Invalid indicator source reviewer: {reviewer_role}")
    h1 = connection.execute(
        """
        SELECT 1 FROM indicator_source_reviews
        WHERE record_key = ? AND reviewer_role = 'H1'
        """,
        (record_key,),
    ).fetchone()
    h2 = connection.execute(
        """
        SELECT 1 FROM indicator_source_reviews
        WHERE record_key = ? AND reviewer_role = 'H2'
        """,
        (record_key,),
    ).fetchone()
    if reviewer_role == "H2" and h1 is None:
        raise ValueError(
            "H2 source adjudication requires an earlier independent H1 "
            f"review: {record_key}"
        )
    if reviewer_role == "H1" and h2 is not None:
        raise ValueError(
            "H1 source review is frozen after H2 adjudication; create a "
            f"new correction record instead of overwriting it: {record_key}"
        )
    connection.execute(
        """
        INSERT INTO indicator_source_reviews(
            record_key, reviewer_role, disposition,
            english_fulltext_status, notes, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key, reviewer_role) DO UPDATE SET
            disposition = excluded.disposition,
            english_fulltext_status = excluded.english_fulltext_status,
            notes = excluded.notes,
            reviewed_at = excluded.reviewed_at
        """,
        (
            record_key,
            reviewer_role,
            disposition,
            fulltext_status,
            notes,
            utc_now(),
        ),
    )
    resolved = connection.execute(
        """
        SELECT disposition, english_fulltext_status, notes, reviewer_role
        FROM indicator_source_reviews
        WHERE record_key = ?
        ORDER BY CASE reviewer_role WHEN 'H2' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (record_key,),
    ).fetchone()
    roles = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT reviewer_role FROM indicator_source_reviews
            WHERE record_key = ? ORDER BY reviewer_role
            """,
            (record_key,),
        )
    ]
    if resolved is None:
        raise RuntimeError(f"Source review was not stored: {record_key}")
    connection.execute(
        """
        INSERT INTO indicator_source_disposition(
            record_key, disposition, english_fulltext_status,
            notes, decided_by, decided_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key) DO UPDATE SET
            disposition = excluded.disposition,
            english_fulltext_status = excluded.english_fulltext_status,
            notes = excluded.notes,
            decided_by = excluded.decided_by,
            decided_at = excluded.decided_at
        """,
        (
            record_key,
            resolved["disposition"],
            resolved["english_fulltext_status"],
            resolved["notes"],
            "|".join(roles),
            utc_now(),
        ),
    )


def _store_mention_review(
    connection: sqlite3.Connection,
    mention_id: str,
    reviewer_role: str,
    decision: str,
    payload_json: str,
    notes: str,
) -> None:
    """Preserve H1 extraction and H2 adjudication as distinct records."""
    if reviewer_role not in {"H1", "H2"}:
        raise ValueError(f"Invalid indicator mention reviewer: {reviewer_role}")
    h1 = connection.execute(
        """
        SELECT 1 FROM indicator_mention_reviews
        WHERE mention_id = ? AND reviewer_role = 'H1'
        """,
        (mention_id,),
    ).fetchone()
    h2 = connection.execute(
        """
        SELECT 1 FROM indicator_mention_reviews
        WHERE mention_id = ? AND reviewer_role = 'H2'
        """,
        (mention_id,),
    ).fetchone()
    if reviewer_role == "H2" and h1 is None:
        raise ValueError(
            "H2 mention adjudication requires an earlier independent H1 "
            f"extraction: {mention_id}"
        )
    if reviewer_role == "H1" and h2 is not None:
        raise ValueError(
            "H1 mention extraction is frozen after H2 adjudication; create "
            f"a correction record instead of overwriting it: {mention_id}"
        )
    connection.execute(
        """
        INSERT INTO indicator_mention_reviews(
            mention_id, reviewer_role, decision, payload_json,
            notes, reviewed_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(mention_id, reviewer_role) DO UPDATE SET
            decision = excluded.decision,
            payload_json = excluded.payload_json,
            notes = excluded.notes,
            reviewed_at = excluded.reviewed_at
        """,
        (
            mention_id,
            reviewer_role,
            decision,
            payload_json,
            notes,
            utc_now(),
        ),
    )


def import_indicators(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, Any]:
    """Import source dispositions and formula-level indicator evidence."""
    require_complete(connection, ["literature_screened"])
    included = set(included_record_keys(connection))
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "indicator_extraction",
    )
    rows = iter_csv(snapshot_path)
    mentions = 0
    dispositions = 0
    submission_phase = ""
    submitted_sources: Dict[str, tuple[str, str, str, str]] = {}
    selection_rules = read_json(RULES_PATH)
    priority_by_source_role = {
        str(key): int(value)
        for key, value in selection_rules[
            "selection_priority_by_source_role"
        ].items()
    }
    allowed_evidence_strengths = {
        str(value).casefold()
        for value in selection_rules["evidence_strength_levels"]
    }
    stability_rule = selection_rules["stability_score_rule"]
    for row in rows:
        record_key = str(row.get("record_key") or "").strip()
        if not record_key:
            continue
        if record_key not in included:
            raise ValueError(
                f"Indicator extraction record is not finally included: "
                f"{record_key}"
            )
        disposition = str(
            row.get("source_disposition") or ""
        ).strip().casefold()
        if not disposition:
            raise ValueError(
                f"Every indicator row requires a source disposition: "
                f"{record_key}"
            )
        if disposition not in INDICATOR_DISPOSITIONS:
            raise ValueError(
                f"Invalid indicator source disposition: {disposition}"
            )
        fulltext_status = _required_text(
            row,
            "english_fulltext_status",
            record_key,
        )
        decided_by = _required_text(
            row,
            "disposition_decided_by",
            record_key,
        )
        decided_roles = _review_roles(decided_by)
        if decided_roles == {"H1"}:
            row_phase = "H1"
        elif decided_roles == {"H1", "H2"}:
            row_phase = "H2"
        else:
            raise ValueError(
                "Indicator source rows must be a blind H1 submission or an "
                f"H1-backed H2 adjudication: {record_key}"
            )
        if submission_phase and row_phase != submission_phase:
            raise ValueError(
                "One indicator import cannot mix H1 extraction rows and H2 "
                "adjudication rows"
            )
        submission_phase = row_phase
        disposition_notes = _required_text(
            row,
            "disposition_notes",
            record_key,
        )
        source_payload = (
            row_phase,
            disposition,
            fulltext_status,
            disposition_notes,
        )
        prior_source_payload = submitted_sources.get(record_key)
        if (
            prior_source_payload is not None
            and prior_source_payload != source_payload
        ):
            raise ValueError(
                "Duplicate rows for one source contain conflicting source "
                f"reviews: {record_key}"
            )
        if prior_source_payload is None:
            _store_source_review(
                connection,
                record_key,
                row_phase,
                disposition,
                fulltext_status,
                disposition_notes,
            )
            submitted_sources[record_key] = source_payload
            dispositions += 1
        raw_name = str(row.get("raw_name_en") or "").strip()
        if not raw_name:
            continue
        canonical_name = _required_text(
            row,
            "canonical_name_en",
            record_key,
        )
        identity = f"{record_key}/{canonical_name}"
        research_group = _required_text(
            row,
            "research_group",
            identity,
        )
        research_group_id = normalize_term(
            _required_text(row, "research_group_id", identity)
        )
        research_group_evidence = _required_text(
            row,
            "research_group_evidence",
            identity,
        )
        if not research_group_id:
            raise ValueError(f"Invalid research_group_id: {identity}")
        source_role = _required_text(row, "source_role", identity).casefold()
        if source_role not in SOURCE_ROLES:
            raise ValueError(f"Invalid source_role: {source_role}")
        evidence_strength = _required_text(
            row,
            "evidence_strength",
            identity,
        ).casefold()
        if evidence_strength not in allowed_evidence_strengths:
            raise ValueError(
                f"Invalid evidence_strength: {evidence_strength}"
            )
        stability_score = float(
            _required_text(row, "stability_score", identity)
        )
        if not (
            float(stability_rule["minimum"])
            <= stability_score
            <= float(stability_rule["maximum"])
        ):
            raise ValueError(
                f"stability_score must be between 0 and 1: {identity}"
            )
        stability_basis = _required_text(
            row,
            "stability_basis",
            identity,
        )
        selection_priority = int(
            _required_text(row, "selection_priority", identity)
        )
        expected_priority = priority_by_source_role[source_role]
        if selection_priority != expected_priority:
            raise ValueError(
                "selection_priority is frozen by source_role; expected "
                f"{expected_priority} for {source_role}: {identity}"
            )
        evidence_direction = _required_text(
            row,
            "evidence_direction",
            identity,
        ).casefold()
        if evidence_direction not in {
            "positive",
            "mixed",
            "null",
            "negative",
            "definition_only",
        }:
            raise ValueError(
                f"Invalid evidence_direction: {evidence_direction}"
            )
        scope_role = _required_text(row, "scope_role", identity).casefold()
        if scope_role not in SCOPE_ROLES:
            raise ValueError(f"Invalid scope_role: {scope_role}")
        boolean_fields = (
            "english_fulltext_verified",
            "article_level",
            "primary_or_foundational_evidence",
            "formula_reproducible",
            "t0_computable",
            "requires_future",
            "fatal_validity_concern",
            "uses_outcome_for_selection",
            "nonconstant",
            "h2_approved",
        )
        booleans = {
            field: parse_bool(row.get(field), field)
            for field in boolean_fields
        }
        if source_role == "review_discovery" and (
            booleans["primary_or_foundational_evidence"]
            or booleans["formula_reproducible"]
        ):
            raise ValueError(
                "A review_discovery mention cannot authorize primary "
                "evidence or a reproducible formula"
            )
        formula = str(row.get("formula") or "").strip()
        formula_location = str(
            row.get("formula_location") or ""
        ).strip()
        evidence_span = str(row.get("evidence_span") or "").strip()
        units = str(row.get("units") or "").strip()
        parameters = str(row.get("parameters") or "").strip()
        direction = str(row.get("direction") or "").strip()
        missing_rule = str(row.get("missing_rule") or "").strip()
        required_data = _parse_required_data(
            str(row.get("required_data") or "")
        )
        maximum_information_time = str(
            row.get("maximum_information_time") or ""
        ).strip()
        if booleans["formula_reproducible"] and not all(
            (
                formula,
                formula_location,
                evidence_span,
                units,
                parameters,
                direction,
                missing_rule,
                required_data,
                maximum_information_time,
            )
        ):
            raise ValueError(
                "Formula-authorizing mention lacks formula, provenance, "
                f"units, parameters, direction, missing rule, required data, "
                f"or information time: {identity}"
            )
        if (
            booleans["formula_reproducible"]
            and not re.search(
                r"\b(?:p(?:age)?\.?\s*\d+|table\s+\w+|"
                r"eq(?:uation)?\.?\s*\w+|appendix\s+\w+|"
                r"section\s+\w+)",
                formula_location,
                flags=re.IGNORECASE,
            )
        ):
            raise ValueError(
                "Formula location must identify a page, table, equation, "
                f"appendix, or section: {identity}"
            )
        if (
            booleans["formula_reproducible"]
            and not booleans["english_fulltext_verified"]
        ):
            raise ValueError(
                "A formula cannot be approved without verified English "
                f"full text: {identity}"
            )
        if booleans["t0_computable"] and booleans["requires_future"]:
            raise ValueError(
                f"A T0-computable indicator cannot require future data: "
                f"{identity}"
            )
        if (
            booleans["t0_computable"]
            and maximum_information_time.casefold() != "t0"
        ):
            raise ValueError(
                f"T0-computable indicators must use maximum_information_time "
                f"T0: {identity}"
            )
        verified_by = str(row.get("verified_by") or "").strip()
        verified_roles = _review_roles(verified_by)
        extracted_roles = _review_roles(
            _required_text(row, "extracted_by", identity)
        )
        if "H1" not in extracted_roles or not extracted_roles.issubset(
            {"AI", "H1"}
        ):
            raise ValueError(
                f"Indicator extraction requires H1, with optional AI "
                f"assistance: {identity}"
            )
        verification_notes = str(
            row.get("verification_notes") or ""
        ).strip()
        adjudication_notes = str(
            row.get("adjudication_notes") or ""
        ).strip()
        if booleans["english_fulltext_verified"] and (
            not verification_notes
            or "H1" not in verified_roles
        ):
            raise ValueError(
                "Verified full text requires H1 review and notes: "
                f"{identity}"
            )
        mention_status = str(
            row.get("status") or "candidate"
        ).strip().casefold()
        if mention_status not in {"candidate", "excluded"}:
            raise ValueError(
                f"Invalid indicator mention status: {mention_status}"
            )
        if not verified_roles.issubset({"H1", "H2"}):
            raise ValueError(
                f"Indicator verification roles must be H1/H2: {identity}"
            )
        if submission_phase == "H1" and verified_roles != {"H1"}:
            raise ValueError(
                "Blind H1 extraction cannot contain an H2 verification "
                f"decision: {identity}"
            )
        if submission_phase == "H1" and (
            booleans["h2_approved"] or adjudication_notes
        ):
            raise ValueError(
                "Blind H1 extraction cannot contain H2 approval or "
                f"adjudication notes: {identity}"
            )
        if submission_phase == "H2" and verified_roles != {"H1", "H2"}:
            raise ValueError(
                "H2 adjudication must retain H1 provenance and identify H2: "
                f"{identity}"
            )
        if submission_phase == "H2" and not adjudication_notes:
            raise ValueError(
                f"H2-reviewed mentions require adjudication notes: "
                f"{identity}"
            )
        if booleans["h2_approved"] and (
            "H2" not in verified_roles or not adjudication_notes
        ):
            raise ValueError(
                f"H2 approval requires named H2 review and adjudication "
                f"notes: {identity}"
            )
        if submission_phase == "H2" and (
            booleans["h2_approved"] != (mention_status == "candidate")
        ):
            raise ValueError(
                "H2 must mark an approved mention candidate and a rejected "
                f"mention excluded: {identity}"
            )
        mention_id = (
            str(row.get("mention_id") or "").strip() or _mention_id(row)
        )
        if submission_phase == "H2":
            h1_review = connection.execute(
                """
                SELECT 1 FROM indicator_mention_reviews
                WHERE mention_id = ? AND reviewer_role = 'H1'
                """,
                (mention_id,),
            ).fetchone()
            if h1_review is None:
                raise ValueError(
                    "H2 adjudication contains a mention that H1 did not "
                    f"extract: {mention_id}"
                )
        fulltext_evidence = {
            "fulltext_source_url": str(
                row.get("fulltext_source_url") or ""
            ).strip(),
            "fulltext_local_path": "",
            "fulltext_sha256": "",
            "fulltext_license": str(
                row.get("fulltext_license") or ""
            ).strip(),
        }
        if booleans["english_fulltext_verified"]:
            fulltext_evidence = _register_fulltext_evidence(
                connection,
                row,
                mention_id,
                identity,
            )
        connection.execute(
            """
            INSERT INTO indicator_mentions(
                mention_id, record_key, raw_name_en, canonical_name_en,
                label_zh, source_id, research_group, research_group_id,
                research_group_evidence, source_role,
                formula_location, evidence_span, formula, units,
                parameters, direction, missing_rule, required_data_json,
                maximum_information_time, scope_role,
                validation_summary, evidence_direction, negative_evidence,
                fulltext_source_url, fulltext_local_path,
                fulltext_sha256, fulltext_license,
                english_fulltext_verified, article_level,
                primary_or_foundational_evidence, formula_reproducible,
                t0_computable, requires_future, data_status, bias_policy,
                fatal_validity_concern, uses_outcome_for_selection,
                quality_audit_status, nonconstant, h2_approved,
                evidence_strength, stability_score, stability_basis,
                selection_priority, redundancy_family, extracted_by,
                verified_by,
                verification_notes, adjudication_notes, status
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT(mention_id) DO UPDATE SET
                raw_name_en = excluded.raw_name_en,
                canonical_name_en = excluded.canonical_name_en,
                label_zh = excluded.label_zh,
                source_id = excluded.source_id,
                research_group = excluded.research_group,
                research_group_id = excluded.research_group_id,
                research_group_evidence =
                    excluded.research_group_evidence,
                source_role = excluded.source_role,
                formula_location = excluded.formula_location,
                evidence_span = excluded.evidence_span,
                formula = excluded.formula,
                units = excluded.units,
                parameters = excluded.parameters,
                direction = excluded.direction,
                missing_rule = excluded.missing_rule,
                required_data_json = excluded.required_data_json,
                maximum_information_time =
                    excluded.maximum_information_time,
                scope_role = excluded.scope_role,
                validation_summary = excluded.validation_summary,
                evidence_direction = excluded.evidence_direction,
                negative_evidence = excluded.negative_evidence,
                fulltext_source_url = excluded.fulltext_source_url,
                fulltext_local_path = excluded.fulltext_local_path,
                fulltext_sha256 = excluded.fulltext_sha256,
                fulltext_license = excluded.fulltext_license,
                english_fulltext_verified =
                    excluded.english_fulltext_verified,
                article_level = excluded.article_level,
                primary_or_foundational_evidence =
                    excluded.primary_or_foundational_evidence,
                formula_reproducible = excluded.formula_reproducible,
                t0_computable = excluded.t0_computable,
                requires_future = excluded.requires_future,
                data_status = excluded.data_status,
                bias_policy = excluded.bias_policy,
                fatal_validity_concern =
                    excluded.fatal_validity_concern,
                uses_outcome_for_selection =
                    excluded.uses_outcome_for_selection,
                quality_audit_status = excluded.quality_audit_status,
                nonconstant = excluded.nonconstant,
                h2_approved = excluded.h2_approved,
                evidence_strength = excluded.evidence_strength,
                stability_score = excluded.stability_score,
                stability_basis = excluded.stability_basis,
                selection_priority = excluded.selection_priority,
                redundancy_family = excluded.redundancy_family,
                extracted_by = excluded.extracted_by,
                verified_by = excluded.verified_by,
                verification_notes = excluded.verification_notes,
                adjudication_notes = excluded.adjudication_notes,
                status = excluded.status
            """,
            (
                mention_id,
                record_key,
                raw_name,
                canonical_name,
                str(row.get("label_zh") or "").strip(),
                _required_text(row, "source_id", identity),
                research_group,
                research_group_id,
                research_group_evidence,
                source_role,
                formula_location,
                evidence_span,
                formula,
                units,
                parameters,
                direction,
                missing_rule,
                json.dumps(
                    required_data,
                    ensure_ascii=False,
                ),
                maximum_information_time,
                scope_role,
                str(row.get("validation_summary") or "").strip(),
                evidence_direction,
                str(row.get("negative_evidence") or "").strip(),
                fulltext_evidence["fulltext_source_url"],
                fulltext_evidence["fulltext_local_path"],
                fulltext_evidence["fulltext_sha256"],
                fulltext_evidence["fulltext_license"],
                *(int(booleans[field]) for field in boolean_fields[:6]),
                _required_text(row, "data_status", identity),
                _required_text(row, "bias_policy", identity),
                int(booleans["fatal_validity_concern"]),
                int(booleans["uses_outcome_for_selection"]),
                _required_text(row, "quality_audit_status", identity),
                int(booleans["nonconstant"]),
                int(booleans["h2_approved"]),
                evidence_strength,
                stability_score,
                stability_basis,
                selection_priority,
                _required_text(row, "redundancy_family", identity),
                "|".join(sorted(extracted_roles)),
                "|".join(sorted(verified_roles)),
                verification_notes,
                adjudication_notes,
                mention_status,
            ),
        )
        review_row = dict(row)
        review_row.update(
            {
                "mention_id": mention_id,
                "fulltext_source_url": fulltext_evidence[
                    "fulltext_source_url"
                ],
                "fulltext_local_path": fulltext_evidence[
                    "fulltext_local_path"
                ],
                "fulltext_sha256": fulltext_evidence["fulltext_sha256"],
                "fulltext_license": fulltext_evidence["fulltext_license"],
                "status": mention_status,
            }
        )
        review_decision = (
            mention_status
            if submission_phase == "H1"
            else ("approve" if booleans["h2_approved"] else "exclude")
        )
        _store_mention_review(
            connection,
            mention_id,
            submission_phase,
            review_decision,
            _indicator_review_payload(review_row),
            (
                verification_notes
                if submission_phase == "H1"
                else adjudication_notes
            ),
        )
        mentions += 1
    connection.commit()
    family_count = build_indicator_families(connection)
    invalidate_stages(
        connection,
        ("dimensions_derived", "features_selected", "audit_complete"),
        "indicator evidence changed",
    )
    source_review_roles: Dict[str, set[str]] = defaultdict(set)
    for review in connection.execute(
        """
        SELECT record_key, reviewer_role
        FROM indicator_source_reviews
        ORDER BY record_key, reviewer_role
        """
    ):
        source_review_roles[str(review["record_key"])].add(
            str(review["reviewer_role"])
        )
    missing_sources = sorted(included - set(source_review_roles))
    sources_without_h1 = sorted(
        key
        for key in included
        if "H1" not in source_review_roles.get(key, set())
    )
    sources_without_h2 = sorted(
        key
        for key in included
        if "H2" not in source_review_roles.get(key, set())
    )
    mentions_without_h1 = sorted(
        str(row["mention_id"])
        for row in connection.execute(
            """
            SELECT m.mention_id
            FROM indicator_mentions m
            LEFT JOIN indicator_mention_reviews h1
              ON h1.mention_id = m.mention_id
             AND h1.reviewer_role = 'H1'
            WHERE h1.mention_id IS NULL
            """
        )
    )
    mentions_without_h2 = sorted(
        str(row["mention_id"])
        for row in connection.execute(
            """
            SELECT m.mention_id
            FROM indicator_mentions m
            JOIN indicator_mention_reviews h1
              ON h1.mention_id = m.mention_id
             AND h1.reviewer_role = 'H1'
            LEFT JOIN indicator_mention_reviews h2
              ON h2.mention_id = m.mention_id
             AND h2.reviewer_role = 'H2'
            WHERE h1.decision != 'excluded'
              AND h2.mention_id IS NULL
            """
        )
    )
    inconsistent_source_dispositions = sorted(
        str(row["record_key"])
        for row in connection.execute(
            """
            SELECT d.record_key
            FROM indicator_source_disposition d
            WHERE (
                d.disposition = 'extracted'
                AND NOT EXISTS (
                    SELECT 1 FROM indicator_mentions m
                    WHERE m.record_key = d.record_key
                      AND m.status != 'excluded'
                      AND m.h2_approved = 1
                )
            ) OR (
                d.disposition IN ('no_indicator', 'excluded_after_fulltext')
                AND EXISTS (
                    SELECT 1 FROM indicator_mentions m
                    WHERE m.record_key = d.record_key
                      AND m.status != 'excluded'
                )
            )
            """
        )
    )
    details = {
        "submission_phase": submission_phase,
        "source_dispositions_imported": dispositions,
        "mentions_imported": mentions,
        "canonical_indicator_families": family_count,
        "unprocessed_included_sources": missing_sources,
        "sources_without_h1_review": sources_without_h1,
        "sources_without_h2_review": sources_without_h2,
        "mentions_without_h1_extraction_record": mentions_without_h1,
        "retained_mentions_without_h2_review": mentions_without_h2,
        "inconsistent_source_dispositions": (
            inconsistent_source_dispositions
        ),
    }
    extraction_complete = not any(
        (
            missing_sources,
            sources_without_h1,
            sources_without_h2,
            mentions_without_h1,
            mentions_without_h2,
            inconsistent_source_dispositions,
        )
    )
    set_stage(
        connection,
        "indicators_extracted",
        "complete" if extraction_complete else "ready",
        details,
    )
    log_event(
        connection,
        "indicator_import",
        "file",
        str(snapshot_path.resolve()),
        details,
    )
    connection.commit()
    return details


def _evidence_rank(value: str) -> int:
    ranks = {
        "systematic_review_plus_primary": 5,
        "high": 4,
        "moderate": 3,
        "limited": 2,
        "weak": 1,
        "unknown": 0,
    }
    return ranks.get(value.casefold(), 0)


def _representative_mention(
    rows: Sequence[sqlite3.Row],
) -> sqlite3.Row:
    return sorted(
        rows,
        key=lambda row: (
            int(row["selection_priority"]),
            -int(row["english_fulltext_verified"]),
            -int(row["formula_reproducible"]),
            -int(row["h2_approved"]),
            -_evidence_rank(str(row["evidence_strength"])),
            -float(row["stability_score"]),
            str(row["source_id"]),
            str(row["mention_id"]),
        ),
    )[0]


def build_indicator_families(connection: sqlite3.Connection) -> int:
    """Conservatively merge aliases and parameter variants into families."""
    mentions = connection.execute(
        """
        SELECT * FROM indicator_mentions
        WHERE status != 'excluded'
        ORDER BY canonical_name_en, mention_id
        """
    ).fetchall()
    grouped: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    display_name: Dict[str, str] = {}
    for mention in mentions:
        key = normalize_term(mention["canonical_name_en"])
        if not key:
            continue
        grouped[key].append(mention)
        display_name.setdefault(key, str(mention["canonical_name_en"]).strip())
    connection.execute("DELETE FROM dimension_coding")
    connection.execute("DELETE FROM candidate_dimensions")
    connection.execute("DELETE FROM feature_decisions")
    connection.execute("DELETE FROM dimension_decisions")
    connection.execute("DELETE FROM feature_data_audit")
    connection.execute("DELETE FROM indicator_families")
    for index, key in enumerate(sorted(grouped), start=1):
        rows = grouped[key]
        representative = _representative_mention(rows)
        feature_id = f"EF{index:04d}"
        aliases = sorted(
            {str(row["raw_name_en"]).strip() for row in rows},
            key=normalize_term,
        )
        mention_ids = sorted(str(row["mention_id"]) for row in rows)
        research_groups = sorted(
            {
                normalize_term(row["research_group_id"])
                for row in rows
                if normalize_term(row["research_group_id"])
                and row["english_fulltext_verified"]
                and row["h2_approved"]
                and str(row["research_group_evidence"]).strip()
            }
        )
        formula_evidence = [
            row
            for row in rows
            if row["formula_reproducible"]
            and row["english_fulltext_verified"]
            and row["h2_approved"]
            and str(row["formula"]).strip()
            and str(row["fulltext_source_url"]).strip()
            and str(row["fulltext_local_path"]).strip()
            and str(row["fulltext_sha256"]).strip()
            and str(row["fulltext_license"]).strip()
        ]
        formula_representative = (
            _representative_mention(formula_evidence)
            if formula_evidence
            else representative
        )
        primary_evidence = any(
            row["primary_or_foundational_evidence"] for row in rows
        )
        connection.execute(
            """
            INSERT INTO indicator_families(
                feature_id, canonical_name_en, label_zh, alias_names_json,
                mention_ids_json, formula, units, parameters, direction,
                missing_rule, required_data_json, maximum_information_time,
                scope_role, article_level,
                primary_or_foundational_evidence, formula_reproducible,
                t0_computable, requires_future, data_status, bias_policy,
                fatal_validity_concern, uses_outcome_for_selection,
                quality_audit_status, nonconstant,
                english_fulltext_verified, h2_approved, evidence_strength,
                stability_score, stability_basis, selection_priority,
                redundancy_family, research_groups_json, status
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'candidate'
            )
            """,
            (
                feature_id,
                display_name[key],
                representative["label_zh"],
                json.dumps(aliases, ensure_ascii=False),
                json.dumps(mention_ids, ensure_ascii=False),
                formula_representative["formula"],
                formula_representative["units"],
                formula_representative["parameters"],
                formula_representative["direction"],
                formula_representative["missing_rule"],
                formula_representative["required_data_json"],
                formula_representative["maximum_information_time"],
                formula_representative["scope_role"],
                int(formula_representative["article_level"]),
                int(primary_evidence),
                int(bool(formula_evidence)),
                int(formula_representative["t0_computable"]),
                int(formula_representative["requires_future"]),
                formula_representative["data_status"],
                formula_representative["bias_policy"],
                int(any(row["fatal_validity_concern"] for row in rows)),
                int(any(row["uses_outcome_for_selection"] for row in rows)),
                formula_representative["quality_audit_status"],
                int(formula_representative["nonconstant"]),
                int(bool(formula_evidence)),
                int(bool(formula_evidence)),
                max(
                    (str(row["evidence_strength"]) for row in rows),
                    key=_evidence_rank,
                ),
                float(representative["stability_score"]),
                representative["stability_basis"],
                int(representative["selection_priority"]),
                representative["redundancy_family"],
                json.dumps(research_groups, ensure_ascii=False),
            ),
        )
    connection.commit()
    return len(grouped)


def export_feature_data_audit(
    connection: sqlite3.Connection,
    output_path: Path,
) -> int:
    """Export the independent local-data quality audit worksheet."""
    require_complete(connection, ["indicators_extracted"])
    rows = [
        {
            "feature_id": row["feature_id"],
            "canonical_name_en": row["canonical_name_en"],
            "data_status": "",
            "row_count": "",
            "valid_count": "",
            "unique_count": "",
            "missing_rate": "",
            "derivation_artifact_path": "",
            "input_snapshot_path": "",
            "derivation_hash": "",
            "input_snapshot_hash": "",
            "audit_status": "",
            "reviewer": "",
            "notes": "",
        }
        for row in connection.execute(
            """
            SELECT feature_id, canonical_name_en
            FROM indicator_families ORDER BY feature_id
            """
        )
    ]
    write_csv(output_path, rows, DATA_AUDIT_FIELDS)
    return len(rows)


def _verified_data_artifact(
    connection: sqlite3.Connection,
    row: Mapping[str, Any],
    feature_id: str,
    path_field: str,
    hash_field: str,
    role: str,
) -> tuple[str, str]:
    """Verify one local derivation/input artifact and freeze its hash."""
    artifact_path = Path(
        _required_text(row, path_field, feature_id)
    ).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"{path_field} does not exist for {feature_id}: {artifact_path}"
        )
    computed_hash = sha256_file(artifact_path)
    claimed_hash = str(row.get(hash_field) or "").strip().casefold()
    if claimed_hash and claimed_hash != computed_hash:
        raise ValueError(
            f"{hash_field} does not match {path_field}: {feature_id}"
        )
    connection.execute(
        """
        INSERT INTO source_snapshots(
            source_id, path, sha256, role, imported_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            path = excluded.path,
            sha256 = excluded.sha256,
            role = excluded.role,
            imported_at = excluded.imported_at
        """,
        (
            f"{role}_{feature_id}",
            str(artifact_path),
            computed_hash,
            role,
            utc_now(),
        ),
    )
    return str(artifact_path), computed_hash


def import_feature_data_audit(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Import reproducible sample-level data checks for each family."""
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "feature_data_audit",
    )
    rows = iter_csv(snapshot_path)
    imported = 0
    allowed_data = {
        "materialized_audited",
        "derivable_from_audited_inputs",
        "unavailable",
    }
    allowed_audits = {
        "pass",
        "derivable_inputs_pass",
        "fail",
    }
    for row in rows:
        feature_id = str(row.get("feature_id") or "").strip()
        if not feature_id:
            continue
        exists = connection.execute(
            "SELECT 1 FROM indicator_families WHERE feature_id = ?",
            (feature_id,),
        ).fetchone()
        if exists is None:
            raise ValueError(f"Unknown feature for data audit: {feature_id}")
        data_status = _required_text(row, "data_status", feature_id)
        audit_status = _required_text(row, "audit_status", feature_id)
        if data_status not in allowed_data:
            raise ValueError(f"Invalid data_status: {data_status}")
        if audit_status not in allowed_audits:
            raise ValueError(f"Invalid audit_status: {audit_status}")
        if audit_status == "pass" and data_status != "materialized_audited":
            raise ValueError(
                "audit_status=pass requires materialized_audited data"
            )
        if (
            audit_status == "derivable_inputs_pass"
            and data_status != "derivable_from_audited_inputs"
        ):
            raise ValueError(
                "derivable_inputs_pass requires "
                "derivable_from_audited_inputs"
            )
        if data_status == "unavailable" and audit_status != "fail":
            raise ValueError("Unavailable data must have audit_status=fail")
        row_count = int(_required_text(row, "row_count", feature_id))
        valid_count = int(_required_text(row, "valid_count", feature_id))
        unique_count = int(_required_text(row, "unique_count", feature_id))
        missing_rate = float(
            _required_text(row, "missing_rate", feature_id)
        )
        if (
            row_count < 0
            or valid_count < 0
            or valid_count > row_count
            or unique_count < 0
            or unique_count > valid_count
            or not 0 <= missing_rate <= 1
        ):
            raise ValueError(f"Invalid data-audit counts: {feature_id}")
        expected_missing = (
            (row_count - valid_count) / row_count if row_count else 0.0
        )
        if abs(expected_missing - missing_rate) > 1e-6:
            raise ValueError(
                f"missing_rate does not match row counts: {feature_id}"
            )
        derivation_path = str(
            row.get("derivation_artifact_path") or ""
        ).strip()
        input_path_value = str(
            row.get("input_snapshot_path") or ""
        ).strip()
        derivation_hash = str(
            row.get("derivation_hash") or ""
        ).strip().casefold()
        input_hash = str(
            row.get("input_snapshot_hash") or ""
        ).strip().casefold()
        if audit_status in {"pass", "derivable_inputs_pass"}:
            derivation_path, derivation_hash = _verified_data_artifact(
                connection,
                row,
                feature_id,
                "derivation_artifact_path",
                "derivation_hash",
                "feature_derivation_artifact",
            )
            input_path_value, input_hash = _verified_data_artifact(
                connection,
                row,
                feature_id,
                "input_snapshot_path",
                "input_snapshot_hash",
                "feature_input_snapshot",
            )
        elif any(
            (derivation_path, input_path_value, derivation_hash, input_hash)
        ):
            raise ValueError(
                "Failed/unavailable audits must leave artifact fields blank"
            )
        reviewer = _required_text(row, "reviewer", feature_id)
        notes = _required_text(row, "notes", feature_id)
        connection.execute(
            """
            INSERT INTO feature_data_audit(
                feature_id, data_status, row_count, valid_count,
                unique_count, missing_rate, derivation_artifact_path,
                input_snapshot_path, derivation_hash,
                input_snapshot_hash, audit_status, reviewer, notes,
                audited_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(feature_id) DO UPDATE SET
                data_status = excluded.data_status,
                row_count = excluded.row_count,
                valid_count = excluded.valid_count,
                unique_count = excluded.unique_count,
                missing_rate = excluded.missing_rate,
                derivation_artifact_path =
                    excluded.derivation_artifact_path,
                input_snapshot_path = excluded.input_snapshot_path,
                derivation_hash = excluded.derivation_hash,
                input_snapshot_hash = excluded.input_snapshot_hash,
                audit_status = excluded.audit_status,
                reviewer = excluded.reviewer,
                notes = excluded.notes,
                audited_at = excluded.audited_at
            """,
            (
                feature_id,
                data_status,
                row_count,
                valid_count,
                unique_count,
                missing_rate,
                derivation_path,
                input_path_value,
                derivation_hash,
                input_hash,
                audit_status,
                reviewer,
                notes,
                utc_now(),
            ),
        )
        imported += 1
    if imported:
        invalidate_stages(
            connection,
            ("features_selected", "audit_complete"),
            "feature data audit changed",
        )
    connection.commit()
    return imported


def export_dimension_coding(
    connection: sqlite3.Connection,
    output_path: Path,
    reviewer_role: str | None = None,
) -> int:
    """Export independent construct coding for canonical families."""
    require_complete(connection, ["indicators_extracted"])
    if reviewer_role is None:
        raise ValueError(
            "Export one reviewer at a time to preserve independent coding"
        )
    roles = [reviewer_role]
    if any(role not in {"AI", "H1", "H2"} for role in roles):
        raise ValueError("reviewer_role must be AI, H1, or H2")
    rows: List[Dict[str, Any]] = []
    for family in connection.execute(
        """
        SELECT *
        FROM indicator_families
        WHERE status = 'candidate'
        ORDER BY feature_id
        """
    ):
        for role in roles:
            row = {
                "feature_id": family["feature_id"],
                "canonical_name_en": family["canonical_name_en"],
                "coder_role": role,
                "dimension_label": "",
                "dimension_definition": "",
                "construct_role": "",
                "information_source": "",
                "t0_boundary": "",
                "bias_risk": "",
                "decision": "",
                "reason": "",
                "formula": family["formula"],
                "required_data": family["required_data_json"],
                "maximum_information_time_evidence": family[
                    "maximum_information_time"
                ],
                "scope_role_evidence": family["scope_role"],
                "research_groups_evidence": family[
                    "research_groups_json"
                ],
                "mention_ids_evidence": family["mention_ids_json"],
            }
            if role == "H2":
                codes = {
                    str(item["coder_role"]): item
                    for item in connection.execute(
                        """
                        SELECT * FROM dimension_coding
                        WHERE feature_id = ?
                          AND coder_role IN ('AI', 'H1')
                        """,
                        (family["feature_id"],),
                    )
                }
                for source_role in ("AI", "H1"):
                    code = codes.get(source_role)
                    for field in (
                        "dimension_label",
                        "dimension_definition",
                        "construct_role",
                        "information_source",
                        "t0_boundary",
                        "bias_risk",
                        "decision",
                        "reason",
                    ):
                        row[f"{source_role.casefold()}_{field}"] = (
                            code[field] if code is not None else ""
                        )
            rows.append(row)
    write_csv(
        output_path,
        rows,
        DIMENSION_H2_FIELDS
        if reviewer_role == "H2"
        else DIMENSION_EVIDENCE_FIELDS,
    )
    return len(rows)


def import_dimension_coding(
    connection: sqlite3.Connection,
    input_path: Path,
) -> int:
    """Import AI/H1 construct coding and H2 dimension adjudication."""
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "dimension_coding",
    )
    rows = iter_csv(snapshot_path)
    imported = 0
    submission_role = ""
    for row in rows:
        feature_id = str(row.get("feature_id") or "").strip()
        decision = str(row.get("decision") or "").strip().casefold()
        if not feature_id or not decision:
            continue
        role = str(row.get("coder_role") or "").strip().upper()
        if role not in {"AI", "H1", "H2"}:
            raise ValueError(f"Invalid dimension coder: {role}")
        if submission_role and role != submission_role:
            raise ValueError(
                "One dimension-coding import cannot mix AI, H1, and H2 roles"
            )
        submission_role = role
        if role == "H1" and any(
            str(field).casefold().startswith(("ai_", "h2_"))
            for field in row
        ):
            raise ValueError(
                "Blind H1 dimension import refuses AI/H2 comparison columns"
            )
        if decision not in {"include", "exclude"}:
            raise ValueError(f"Invalid dimension decision: {decision}")
        exists = connection.execute(
            "SELECT 1 FROM indicator_families WHERE feature_id = ?",
            (feature_id,),
        ).fetchone()
        if exists is None:
            raise ValueError(f"Unknown feature family: {feature_id}")
        existing_roles = {
            str(value[0])
            for value in connection.execute(
                """
                SELECT coder_role FROM dimension_coding
                WHERE feature_id = ?
                """,
                (feature_id,),
            )
        }
        if role == "H2" and not {"AI", "H1"}.issubset(existing_roles):
            raise ValueError(
                "H2 dimension adjudication requires earlier independent AI "
                f"and H1 codes: {feature_id}"
            )
        if role in {"AI", "H1"} and "H2" in existing_roles:
            raise ValueError(
                f"{role} dimension coding is frozen after H2 adjudication: "
                f"{feature_id}"
            )
        label = str(row.get("dimension_label") or "").strip()
        definition = str(row.get("dimension_definition") or "").strip()
        construct_role = str(
            row.get("construct_role") or ""
        ).strip().casefold()
        reason = str(row.get("reason") or "").strip()
        if decision == "include":
            if not all(
                (
                    label,
                    definition,
                    construct_role,
                    str(row.get("information_source") or "").strip(),
                    str(row.get("t0_boundary") or "").strip(),
                    str(row.get("bias_risk") or "").strip(),
                    reason,
                )
            ):
                raise ValueError(
                    f"Incomplete dimension coding: {feature_id}/{role}"
                )
            if construct_role not in CONSTRUCT_ROLES:
                raise ValueError(
                    f"Invalid construct role: {construct_role}"
                )
            if re.match(r"^D(?:0?[1-9]|1[0-2])(?:_|\b)", label):
                raise ValueError(
                    "v3 dimension labels must be evidence-derived and may "
                    "not reuse v1/v2 D01-D12 labels"
                )
        elif not reason:
            raise ValueError(
                f"Excluded dimension coding requires a reason: "
                f"{feature_id}/{role}"
            )
        connection.execute(
            """
            INSERT INTO dimension_coding(
                feature_id, coder_role, dimension_label,
                dimension_definition, construct_role, information_source,
                t0_boundary, bias_risk, decision, reason, coded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(feature_id, coder_role) DO UPDATE SET
                dimension_label = excluded.dimension_label,
                dimension_definition = excluded.dimension_definition,
                construct_role = excluded.construct_role,
                information_source = excluded.information_source,
                t0_boundary = excluded.t0_boundary,
                bias_risk = excluded.bias_risk,
                decision = excluded.decision,
                reason = excluded.reason,
                coded_at = excluded.coded_at
            """,
            (
                feature_id,
                role,
                label,
                definition,
                construct_role,
                str(row.get("information_source") or "").strip(),
                str(row.get("t0_boundary") or "").strip(),
                str(row.get("bias_risk") or "").strip(),
                decision,
                reason,
                utc_now(),
            ),
        )
        imported += 1
    if imported:
        invalidate_stages(
            connection,
            ("dimensions_derived", "features_selected", "audit_complete"),
            "dimension coding changed",
        )
    connection.commit()
    return imported


def _dimension_signature(row: sqlite3.Row) -> str:
    return "|".join(
        (
            str(row["decision"]),
            normalize_term(row["dimension_label"]),
            normalize_term(row["dimension_definition"]),
            str(row["construct_role"]),
            normalize_term(row["information_source"]),
            normalize_term(row["t0_boundary"]),
            normalize_term(row["bias_risk"]),
        )
    )


def _dimension_labels(value: str) -> List[str]:
    labels = [
        item.strip()
        for item in value.replace(";", "|").split("|")
        if item.strip()
    ]
    if not labels:
        raise ValueError("Included construct coding requires a dimension")
    return sorted(set(labels), key=normalize_term)


def _require_formal_indicator_saturation(
    connection: sqlite3.Connection,
) -> None:
    """Prevent final dimensions from preceding formal-query saturation."""
    formal_pools = connection.execute(
        """
        SELECT COUNT(*) FROM discovery_queries
        WHERE query_role = 'formal_search_family' AND status = 'active'
        """
    ).fetchone()[0]
    if not formal_pools:
        return
    required = int(
        read_json(SATURATION_PROTOCOL_PATH)["sequential_review"][
            "minimum_consecutive_zero_novelty_rounds"
        ]
    )
    row = connection.execute(
        """
        SELECT 1 FROM discovery_review_rounds
        WHERE saturation_phase = 'formal_indicator_discovery'
          AND reviewer_role = 'H2'
          AND decision = 'freeze'
          AND fully_reviewed = 1
          AND consecutive_zero_rounds >= ?
          AND new_nonredundant_english_terms = 0
          AND new_canonical_indicator_families = 0
          AND iteration = (
              SELECT MAX(iteration)
              FROM discovery_review_rounds
              WHERE saturation_phase = 'formal_indicator_discovery'
          )
        LIMIT 1
        """,
        (required,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "Candidate dimensions require an H2-approved formal "
            "indicator-discovery saturation freeze"
        )


def derive_dimensions(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Derive M candidate dimensions only after family construction."""
    require_complete(connection, ["indicators_extracted"])
    _require_formal_indicator_saturation(connection)
    families = connection.execute(
        """
        SELECT * FROM indicator_families
        WHERE status = 'candidate'
        ORDER BY feature_id
        """
    ).fetchall()
    if not families:
        raise RuntimeError("No canonical indicator families are available")
    resolved: List[tuple[sqlite3.Row, sqlite3.Row]] = []
    missing: List[str] = []
    left: List[str] = []
    right: List[str] = []
    for family in families:
        codes = {
            str(row["coder_role"]): row
            for row in connection.execute(
                "SELECT * FROM dimension_coding WHERE feature_id = ?",
                (family["feature_id"],),
            )
        }
        if not {"AI", "H1", "H2"}.issubset(codes):
            missing.append(str(family["feature_id"]))
            continue
        left.append(_dimension_signature(codes["AI"]))
        right.append(_dimension_signature(codes["H1"]))
        resolved.append((family, codes["H2"]))
    if missing:
        raise RuntimeError(
            "All families require AI, H1, and H2 construct coding: "
            + ", ".join(missing[:25])
        )
    grouped: Dict[str, List[tuple[sqlite3.Row, sqlite3.Row]]] = defaultdict(
        list
    )
    for family, final in resolved:
        if final["decision"] == "exclude":
            continue
        for label in _dimension_labels(final["dimension_label"]):
            grouped[label].append((family, final))
    connection.execute("DELETE FROM dimension_decisions")
    connection.execute("DELETE FROM candidate_dimensions")
    for index, label in enumerate(
        sorted(grouped, key=normalize_term),
        start=1,
    ):
        members = grouped[label]
        roles = {str(code["construct_role"]) for _, code in members}
        if len(roles) != 1:
            raise RuntimeError(
                f"H2 assigned incompatible construct roles to {label}: "
                + ", ".join(sorted(roles))
            )
        definitions = sorted(
            {str(code["dimension_definition"]) for _, code in members},
            key=normalize_term,
        )
        feature_ids = sorted(str(family["feature_id"]) for family, _ in members)
        groups = sorted(
            {
                group
                for family, _ in members
                for group in json.loads(family["research_groups_json"])
            }
        )
        reasons = sorted(
            {str(code["reason"]) for _, code in members},
            key=normalize_term,
        )
        connection.execute(
            """
            INSERT INTO candidate_dimensions(
                dimension_id, label, definition, construct_role,
                feature_ids_json, research_groups_json, h2_approved,
                status, decision_reason
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 'candidate', ?)
            """,
            (
                f"CD{index:03d}",
                label,
                " | ".join(definitions),
                next(iter(roles)),
                json.dumps(feature_ids, ensure_ascii=False),
                json.dumps(groups, ensure_ascii=False),
                "H2 adjudicated mappings: " + " | ".join(reasons),
            ),
        )
    agreement = {
        "n": len(left),
        "raw_agreement": raw_agreement(left, right),
        "cohen_kappa": cohen_kappa(left, right),
        "gwet_ac1": gwet_ac1(left, right),
    }
    details = {"M": len(grouped), "agreement": agreement}
    set_stage(connection, "dimensions_derived", "complete", details)
    invalidate_stages(
        connection,
        ("features_selected", "audit_complete"),
        "candidate dimensions re-derived",
    )
    log_event(
        connection,
        "dimension_derivation",
        "collection",
        "canonical_indicators",
        details,
    )
    connection.commit()
    return details


def _check_gate(
    family: Mapping[str, Any],
    rule: Mapping[str, Any],
) -> bool:
    value = family[str(rule["field"])]
    if "equals" in rule:
        expected = rule["equals"]
        if isinstance(expected, bool):
            return bool(value) is expected
        return value == expected
    if "allowed" in rule:
        return value in set(rule["allowed"])
    raise ValueError(f"Unsupported gate rule: {rule}")


def _redundancy_rank(
    family: sqlite3.Row,
    audited_data_status: str,
) -> tuple[Any, ...]:
    data_rank = {
        "materialized_audited": 2,
        "derivable_from_audited_inputs": 1,
    }.get(audited_data_status, 0)
    return (
        int(family["selection_priority"]),
        -_evidence_rank(str(family["evidence_strength"])),
        -data_rank,
        -float(family["stability_score"]),
        str(family["feature_id"]),
    )


def _dimension_role(construct_role: str) -> str:
    return {
        "substantive_innovation": "predictive",
        "t0_potential": "predictive",
        "opportunity": "opportunity",
        "context_control": "control",
        "sensitivity": "sensitivity",
    }[construct_role]


def select_indicators(
    connection: sqlite3.Connection,
) -> Dict[str, Any]:
    """Apply frozen gates, redundancy rules, then dimension retention."""
    require_complete(connection, ["dimensions_derived"])
    rules = read_json(RULES_PATH)
    gates = rules["hard_gates"]
    families = connection.execute(
        "SELECT * FROM indicator_families ORDER BY feature_id"
    ).fetchall()
    connection.execute("DELETE FROM feature_decisions")
    connection.execute("DELETE FROM dimension_decisions")
    gate_passers: Dict[str, sqlite3.Row] = {}
    gate_results: Dict[str, Dict[str, bool]] = {}
    audited_data_statuses: Dict[str, str] = {}
    for family in families:
        family_values = dict(family)
        data_audit = connection.execute(
            """
            SELECT * FROM feature_data_audit WHERE feature_id = ?
            """,
            (family["feature_id"],),
        ).fetchone()
        if data_audit is None:
            family_values.update(
                {
                    "data_status": "not_audited",
                    "quality_audit_status": "not_audited",
                    "nonconstant": 0,
                }
            )
        else:
            family_values.update(
                {
                    "data_status": data_audit["data_status"],
                    "quality_audit_status": data_audit["audit_status"],
                    "nonconstant": int(data_audit["unique_count"]) > 1,
                }
            )
        audited_data_statuses[str(family["feature_id"])] = str(
            family_values["data_status"]
        )
        checks = {
            gate_id: _check_gate(family_values, rule)
            for gate_id, rule in gates.items()
        }
        gate_results[str(family["feature_id"])] = checks
        if all(checks.values()):
            gate_passers[str(family["feature_id"])] = family
    redundant_groups: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for family in gate_passers.values():
        key = str(family["redundancy_family"]).strip()
        if not key:
            key = str(family["feature_id"])
        redundant_groups[key].append(family)
    redundancy_winners: Dict[str, str] = {}
    winner_ids: set[str] = set()
    for group, members in redundant_groups.items():
        winner = sorted(
            members,
            key=lambda family: _redundancy_rank(
                family,
                audited_data_statuses[str(family["feature_id"])],
            ),
        )[0]
        winner_id = str(winner["feature_id"])
        winner_ids.add(winner_id)
        for member in members:
            redundancy_winners[str(member["feature_id"])] = winner_id
    dimensions = connection.execute(
        "SELECT * FROM candidate_dimensions ORDER BY dimension_id"
    ).fetchall()
    retained_membership: Dict[str, List[tuple[str, str]]] = defaultdict(list)
    dimension_counts = {
        "predictive": 0,
        "opportunity": 0,
        "control": 0,
        "sensitivity": 0,
    }
    for dimension in dimensions:
        members = json.loads(dimension["feature_ids_json"])
        selected_features = sorted(set(members) & winner_ids)
        groups = sorted(
            {
                group
                for feature_id in selected_features
                for group in json.loads(
                    connection.execute(
                        """
                        SELECT research_groups_json
                        FROM indicator_families WHERE feature_id = ?
                        """,
                        (feature_id,),
                    ).fetchone()[0]
                )
            }
        )
        role = _dimension_role(str(dimension["construct_role"]))
        reasons: List[str] = []
        if not selected_features:
            reasons.append("NO_INDICATOR_PASSED_ALL_HARD_GATES")
        if len(groups) < 2:
            reasons.append("FEWER_THAN_TWO_INDEPENDENT_RESEARCH_GROUPS")
        if not dimension["h2_approved"]:
            reasons.append("NO_H2_DIMENSION_APPROVAL")
        retained = not reasons
        if retained:
            dimension_counts[role] += 1
            for feature_id in selected_features:
                retained_membership[feature_id].append(
                    (str(dimension["dimension_id"]), role)
                )
        connection.execute(
            """
            INSERT INTO dimension_decisions(
                dimension_id, selected_feature_ids_json,
                independent_group_count, dimension_role, selected,
                decision_reason
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                dimension["dimension_id"],
                json.dumps(selected_features, ensure_ascii=False),
                len(groups),
                role,
                int(retained),
                "PASS" if retained else "|".join(reasons),
            ),
        )
        connection.execute(
            """
            UPDATE candidate_dimensions
            SET status = ?, decision_reason = decision_reason || ?
            WHERE dimension_id = ?
            """,
            (
                "retained" if retained else "eliminated",
                " | selection: " + ("PASS" if retained else "|".join(reasons)),
                dimension["dimension_id"],
            ),
        )
    final_roles: Dict[str, str] = {}
    for family in families:
        feature_id = str(family["feature_id"])
        checks = gate_results[feature_id]
        failed = [
            gate_id for gate_id, passed in checks.items() if not passed
        ]
        redundancy_winner = redundancy_winners.get(feature_id, "")
        reasons = list(failed)
        final_role = "excluded"
        if not failed and feature_id not in winner_ids:
            reasons.append("R_REDUNDANT_NONREPRESENTATIVE")
        elif not failed and feature_id in winner_ids:
            memberships = retained_membership.get(feature_id, [])
            if not memberships:
                reasons.append("R_NO_RETAINED_DIMENSION")
            else:
                role_priority = {
                    "predictive": 0,
                    "opportunity": 1,
                    "control": 2,
                    "sensitivity": 3,
                }
                final_role = sorted(
                    (role for _, role in memberships),
                    key=lambda value: role_priority[value],
                )[0]
                if family["bias_policy"] == "sensitivity_only":
                    final_role = "sensitivity"
                reasons.append("PASS")
                final_roles[feature_id] = final_role
        connection.execute(
            """
            INSERT INTO feature_decisions(
                feature_id, gate_checks_json, failed_gates_json,
                redundancy_winner_id, final_role, decision_reason
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                feature_id,
                json.dumps(checks, sort_keys=True),
                json.dumps(failed, ensure_ascii=False),
                redundancy_winner,
                final_role,
                "|".join(reasons),
            ),
        )
        connection.execute(
            "UPDATE indicator_families SET status = ? WHERE feature_id = ?",
            (
                "selected" if feature_id in final_roles else "excluded",
                feature_id,
            ),
        )
    summary = {
        "M": len(dimensions),
        "D": dimension_counts["predictive"],
        "F": len(final_roles),
        "final_indicator_roles": {
            role: sum(value == role for value in final_roles.values())
            for role in ("predictive", "opportunity", "control", "sensitivity")
        },
        "retained_dimensions": dimension_counts,
        "no_quota_applied": True,
    }
    set_stage(connection, "features_selected", "complete", summary)
    invalidate_stages(
        connection,
        ("audit_complete",),
        "feature selection re-executed",
    )
    log_event(
        connection,
        "feature_selection",
        "collection",
        "canonical_indicators",
        summary,
    )
    connection.commit()
    write_selection_outputs(connection, summary)
    return summary


def write_selection_outputs(
    connection: sqlite3.Connection,
    summary: Mapping[str, Any],
) -> None:
    """Write training-ready, role-separated feature manifests."""
    features = [
        {
            **dict(row),
            "alias_names": json.loads(row["alias_names_json"]),
            "mention_ids": json.loads(row["mention_ids_json"]),
            "required_data": json.loads(row["required_data_json"]),
            "research_groups": json.loads(row["research_groups_json"]),
        }
        for row in connection.execute(
            """
            SELECT f.*, d.final_role, d.decision_reason
            FROM indicator_families f
            JOIN feature_decisions d USING(feature_id)
            WHERE d.final_role != 'excluded'
            ORDER BY d.final_role, f.feature_id
            """
        )
    ]
    for row in features:
        for key in (
            "alias_names_json",
            "mention_ids_json",
            "required_data_json",
            "research_groups_json",
        ):
            row.pop(key, None)
    dimensions = [
        {
            **dict(row),
            "feature_ids": json.loads(row["feature_ids_json"]),
            "research_groups": json.loads(row["research_groups_json"]),
            "selected_feature_ids": json.loads(
                row["selected_feature_ids_json"]
            ),
        }
        for row in connection.execute(
            """
            SELECT c.*, d.selected_feature_ids_json,
                   d.independent_group_count, d.dimension_role,
                   d.selected, d.decision_reason AS selection_reason
            FROM candidate_dimensions c
            JOIN dimension_decisions d USING(dimension_id)
            ORDER BY c.dimension_id
            """
        )
    ]
    for row in dimensions:
        row.pop("feature_ids_json", None)
        row.pop("research_groups_json", None)
        row.pop("selected_feature_ids_json", None)
    write_json(
        OUTPUT_DIR / "final_feature_set_v3.json",
        {
            "schema_version": "3.4.0",
            "summary": dict(summary),
            "features": features,
        },
    )
    write_json(
        OUTPUT_DIR / "final_dimensions_v3.json",
        {
            "schema_version": "3.4.0",
            "summary": dict(summary),
            "dimensions": dimensions,
        },
    )
    decision_rows = [
        {
            **dict(row),
            "failed_gates": "|".join(
                json.loads(row["failed_gates_json"])
            ),
        }
        for row in connection.execute(
            """
            SELECT * FROM feature_decisions ORDER BY feature_id
            """
        )
    ]
    write_csv(
        OUTPUT_DIR / "feature_decisions_v3.csv",
        decision_rows,
        (
            "feature_id",
            "failed_gates",
            "redundancy_winner_id",
            "final_role",
            "decision_reason",
            "gate_checks_json",
        ),
    )
