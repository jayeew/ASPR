"""Deterministic engine for the simplified evidence-derived protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

ROOT = Path(__file__).resolve().parent
REQUIRED_OUTPUTS = (
    "protocol.json",
    "search_terms.csv",
    "search_domains.csv",
    "search_queries.csv",
    "seed_press_validation.csv",
    "literature_screening.csv",
    "prisma_flow.csv",
    "candidate_dimensions.csv",
    "indicator_library.csv",
    "indicator_decisions.csv",
    "final_feature_sets.json",
    "training_matrix_manifest.json",
    "final_training_features_strict.parquet",
    "final_training_features_primary.parquet",
    "final_training_features_expanded.parquet",
    "final_training_features_broad_t0.parquet",
    "audit_report.md",
)
AUDIT_TABLES = (
    "works",
    "citations",
    "work_publication_dates",
    "provider_cache_records",
    "search_runs",
    "terms",
    "term_families",
    "search_domains",
    "logical_queries",
    "physical_queries",
    "seed_recall",
    "seed_inputs",
    "screening_decisions",
    "screening_final",
    "formal_pool_records",
    "construct_mentions",
    "candidate_dimensions",
    "indicator_mentions",
    "indicator_families",
    "indicator_evidence",
    "indicator_data_mapping",
    "hard_gate_decisions",
    "evidence_tiers",
    "final_dimensions",
    "final_features",
    "review_sessions",
    "saturation_rounds",
    "discovery_round_records",
    "discovery_decisions",
    "discovery_final",
    "discovery_extractions",
    "discovery_term_families",
    "discovery_indicator_families",
)


class ProtocolError(RuntimeError):
    """Raised when a protocol gate would otherwise be bypassed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}_{sha256_text('|'.join(str(part) for part in parts))[:16]}"


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


@dataclass(frozen=True)
class SelectionCounts:
    k: int
    q: int
    p: int
    m: int
    d_supported: int
    d_strict: int
    f_all: int
    f_model: int
    f_strict: int

    def as_dict(self) -> dict[str, int]:
        return {
            "K": self.k,
            "Q": self.q,
            "P": self.p,
            "M": self.m,
            "D_supported": self.d_supported,
            "D_strict": self.d_strict,
            "F_all": self.f_all,
            "F_model": self.f_model,
            "F_strict": self.f_strict,
        }


class EvidenceProtocol:
    """SQLite-backed, fail-closed protocol state machine."""

    def __init__(self, database: Path, output_dir: Path | None = None) -> None:
        self.database = database.resolve()
        self.output_dir = (output_dir or database.parent).resolve()
        self.protocol_path = ROOT / "protocol.json"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        schema = (ROOT / "schema.sql").read_text(encoding="utf-8")
        self.connection.executescript(schema)
        dimension_columns = {
            row[1]
            for row in self.connection.execute(
                "PRAGMA table_info(candidate_dimensions)"
            )
        }
        if "independent_non_alias_confirmed" not in dimension_columns:
            self.connection.execute(
                "ALTER TABLE candidate_dimensions ADD COLUMN independent_non_alias_confirmed "
                "INTEGER NOT NULL DEFAULT 0"
            )
        protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.set_metadata("protocol", protocol)
        self.set_metadata("protocol_hash", file_hash(self.protocol_path))
        if self.get_metadata("stage") is None:
            self.set_metadata("stage", "initialized")
        destination = self.output_dir / "protocol.json"
        destination.write_bytes(self.protocol_path.read_bytes())
        self.connection.commit()

    def set_metadata(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key,value_json) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
            (key, canonical_json(value)),
        )

    def get_metadata(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value_json FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return _loads(row[0], default) if row else default

    def register_legacy_inventory(self, paths: Sequence[Path]) -> dict[str, str]:
        """Register legacy inputs by hash without importing their decisions."""
        inventory: dict[str, str] = {}
        for path in paths:
            resolved = path.resolve()
            if resolved.is_file():
                inventory[str(resolved)] = file_hash(resolved)
            elif resolved.is_dir():
                rows = [
                    (str(item.relative_to(resolved)), file_hash(item))
                    for item in sorted(resolved.rglob("*"))
                    if item.is_file()
                ]
                inventory[str(resolved)] = sha256_text(canonical_json(rows))
            else:
                inventory[str(resolved)] = "missing"
        self.set_metadata("legacy_inventory", inventory)
        self.set_metadata("legacy_decisions_imported", False)
        self.set_metadata("stage", "bootstrap")
        self.connection.commit()
        return inventory

    def ingest_work(self, work: Mapping[str, Any]) -> tuple[str, bool]:
        """Apply DOI → OpenAlex ID → normalized title/year deduplication."""
        title = str(work.get("title", "")).strip()
        if not title:
            raise ProtocolError("A work requires a title")
        doi = str(work.get("doi", "")).strip().lower().removeprefix("https://doi.org/")
        openalex_id = str(work.get("openalex_id", "")).strip().upper()
        normalized_title = normalize_text(title)
        publication_year = work.get("publication_year")
        existing: sqlite3.Row | None = None
        if doi:
            existing = self.connection.execute(
                "SELECT work_id FROM works WHERE doi=?", (doi,)
            ).fetchone()
        if not existing and openalex_id:
            existing = self.connection.execute(
                "SELECT work_id FROM works WHERE openalex_id=?", (openalex_id,)
            ).fetchone()
        if not existing and publication_year is not None:
            existing = self.connection.execute(
                "SELECT work_id FROM works WHERE normalized_title=? AND publication_year=? ORDER BY work_id LIMIT 1",
                (normalized_title, int(publication_year)),
            ).fetchone()
        if existing:
            work_id = str(existing[0])
            current = self.connection.execute(
                "SELECT * FROM works WHERE work_id=?", (work_id,)
            ).fetchone()
            assert current is not None
            incoming = {
                "doi": doi,
                "openalex_id": openalex_id,
                "publication_year": publication_year,
                "language": str(work.get("language", "")),
                "work_type": str(work.get("work_type", "")),
                "abstract": str(work.get("abstract", "")),
                "source_route": str(work.get("source_route", "")),
            }
            merged = {
                key: current[key] if current[key] not in (None, "") else incoming[key]
                for key in incoming
            }
            self.connection.execute(
                "UPDATE works SET doi=?,openalex_id=?,publication_year=?,language=?,"
                "work_type=?,abstract=?,source_route=? WHERE work_id=?",
                (*merged.values(), work_id),
            )
            return work_id, False
        work_id = str(
            work.get("work_id")
            or stable_id("W", doi, openalex_id, normalized_title, publication_year)
        )
        payload = {
            "doi": doi,
            "openalex_id": openalex_id,
            "title": title,
            "publication_year": publication_year,
            "language": str(work.get("language", "")),
            "work_type": str(work.get("work_type", "")),
            "abstract": str(work.get("abstract", "")),
            "source_route": str(work.get("source_route", "")),
        }
        self.connection.execute(
            "INSERT INTO works VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                work_id,
                doi,
                openalex_id,
                title,
                normalized_title,
                publication_year,
                payload["language"],
                payload["work_type"],
                payload["abstract"],
                payload["source_route"],
                sha256_text(canonical_json(payload)),
            ),
        )
        publication_date = str(work.get("publication_date", ""))
        if publication_date:
            self.connection.execute(
                "INSERT INTO work_publication_dates VALUES(?,?)",
                (work_id, publication_date),
            )
        self.connection.commit()
        return work_id, True

    def record_saturation_round(
        self,
        round_no: int,
        new_term_families: int,
        new_indicator_families: int,
        fully_reviewed: bool,
        evidence_hash: str,
    ) -> str:
        if not 1 <= round_no <= 15:
            raise ProtocolError("Saturation round must be in 1..15")
        if new_term_families < 0 or new_indicator_families < 0:
            raise ProtocolError("Saturation novelty counts cannot be negative")
        if not re.fullmatch(r"[0-9a-f]{64}", evidence_hash):
            raise ProtocolError("Saturation evidence_hash must be a SHA-256 digest")
        if self.connection.execute(
            "SELECT 1 FROM saturation_rounds WHERE round_no=?", (round_no,)
        ).fetchone():
            raise ProtocolError("A saturation round is immutable once recorded")
        previous_rounds = {
            int(row[0])
            for row in self.connection.execute(
                "SELECT round_no FROM saturation_rounds WHERE round_no<?", (round_no,)
            )
        }
        if previous_rounds != set(range(1, round_no)):
            raise ProtocolError(
                "Saturation rounds must be recorded consecutively from round 1"
            )
        incomplete_previous = self.connection.execute(
            "SELECT COUNT(*) FROM saturation_rounds WHERE round_no<? AND fully_reviewed<>1",
            (round_no,),
        ).fetchone()[0]
        if incomplete_previous:
            raise ProtocolError("Every prior saturation round must be fully reviewed")
        existing_stop = self.connection.execute(
            "SELECT round_no FROM saturation_rounds WHERE decision='stop' ORDER BY round_no LIMIT 1"
        ).fetchone()
        if existing_stop and round_no > int(existing_stop[0]):
            raise ProtocolError("Cannot add rounds after the first protocol stop")
        dual_zero = new_term_families == 0 and new_indicator_families == 0
        if fully_reviewed and dual_zero:
            decision, basis = "stop", "strict_zero_zero"
        elif fully_reviewed and round_no == 15:
            decision, basis = "stop", "maximum_round_15"
        else:
            decision, basis = "continue", "not_applicable"
        self.connection.execute(
            "INSERT INTO saturation_rounds VALUES(?,?,?,?,?,?,?)",
            (
                round_no,
                int(fully_reviewed),
                new_term_families,
                new_indicator_families,
                decision,
                basis,
                evidence_hash,
            ),
        )
        self.set_metadata("stage", "saturate")
        self.connection.commit()
        return basis

    def freeze_search(self) -> dict[str, Any]:
        stop = self.connection.execute(
            "SELECT * FROM saturation_rounds WHERE decision='stop' ORDER BY round_no LIMIT 1"
        ).fetchone()
        if not stop or not stop["fully_reviewed"]:
            raise ProtocolError(
                "Search cannot freeze before a reviewed 0/0 round or round 15"
            )
        unresolved_press = self.connection.execute(
            "SELECT COUNT(*) FROM logical_queries WHERE redundancy_status='active' AND press_status<>'pass'"
        ).fetchone()[0]
        missed = self.connection.execute(
            "SELECT COUNT(*) FROM seed_recall WHERE indexability='indexable' AND recall_status<>'recalled'"
        ).fetchone()[0]
        seed_cohorts = {
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT cohort FROM seed_recall WHERE indexability='indexable'"
            )
        }
        if unresolved_press:
            raise ProtocolError(
                f"PRESS has {unresolved_press} unresolved active queries"
            )
        if missed:
            raise ProtocolError(f"Seed recall has {missed} unresolved indexable seeds")
        if not {"development", "hidden"}.issubset(seed_cohorts):
            raise ProtocolError(
                "Both development and hidden indexable seed cohorts are required"
            )
        counts = self._search_counts()
        if any(counts[key] == 0 for key in ("K", "Q", "P")):
            raise ProtocolError("K, Q and P must all be non-zero before search freeze")
        orphan_queries = self.connection.execute(
            "SELECT COUNT(*) FROM logical_queries q LEFT JOIN search_domains d USING(domain_id) "
            "WHERE q.redundancy_status='active' AND (d.domain_id IS NULL OR d.status<>'active')"
        ).fetchone()[0]
        orphan_physical = self.connection.execute(
            "SELECT COUNT(*) FROM physical_queries p LEFT JOIN logical_queries q USING(query_id) "
            "WHERE p.active=1 AND (q.query_id IS NULL OR q.redundancy_status<>'active')"
        ).fetchone()[0]
        active_queries = {
            row[0]
            for row in self.connection.execute(
                "SELECT query_id FROM logical_queries WHERE redundancy_status='active'"
            )
        }
        invalid_seed_links = sum(
            not (links := set(_loads(row[0], []))) or not links.issubset(active_queries)
            for row in self.connection.execute(
                "SELECT matched_query_ids_json FROM seed_recall "
                "WHERE indexability='indexable' AND recall_status='recalled'"
            )
        )
        if orphan_queries or orphan_physical or invalid_seed_links:
            raise ProtocolError(
                "Invalid search-frame references: "
                f"orphan_queries={orphan_queries}, orphan_physical={orphan_physical}, "
                f"invalid_seed_links={invalid_seed_links}"
            )
        self.connection.execute(
            "UPDATE logical_queries SET frozen=1 WHERE redundancy_status='active'"
        )
        frame_hash = self._search_frame_hash(counts)
        self.set_metadata("search_frame", {**counts, "hash": frame_hash})
        self.set_metadata("stage", "freeze-search")
        self.connection.commit()
        return {**counts, "hash": frame_hash}

    def _search_frame_hash(self, counts: Mapping[str, int] | None = None) -> str:
        frame_tables = {
            name: [
                dict(row)
                for row in self.connection.execute(f"SELECT * FROM {name} ORDER BY 1")
            ]
            for name in (
                "term_families",
                "search_domains",
                "logical_queries",
                "physical_queries",
                "seed_recall",
            )
        }
        return sha256_text(
            canonical_json(
                {
                    "counts": dict(counts or self._search_counts()),
                    "tables": frame_tables,
                }
            )
        )

    def _search_counts(self) -> dict[str, int]:
        connection = self.connection
        return {
            "K": connection.execute(
                "SELECT COUNT(*) FROM search_domains WHERE status='active'"
            ).fetchone()[0],
            "Q": connection.execute(
                "SELECT COUNT(*) FROM logical_queries WHERE redundancy_status='active'"
            ).fetchone()[0],
            "P": connection.execute(
                "SELECT COUNT(*) FROM physical_queries p JOIN logical_queries q USING(query_id) "
                "WHERE p.active=1 AND q.redundancy_status='active'"
            ).fetchone()[0],
        }

    def finalize_screening(self) -> int:
        works = self.connection.execute("SELECT COUNT(*) FROM works").fetchone()[0]
        final = self.connection.execute(
            "SELECT COUNT(*) FROM screening_final"
        ).fetchone()[0]
        if works != final:
            raise ProtocolError(
                f"Formal screening incomplete: {final}/{works} works disposed"
            )
        invalid = self.connection.execute(
            "SELECT COUNT(*) FROM screening_final f JOIN works w USING(work_id) "
            "WHERE w.language <> 'en' AND NOT(f.decision='exclude' AND f.exclusion_code='E_LANGUAGE_NON_ENGLISH')"
        ).fetchone()[0]
        if invalid:
            raise ProtocolError("Non-English works require E_LANGUAGE_NON_ENGLISH")
        invalid_decisions = self.connection.execute(
            "SELECT COUNT(*) FROM screening_final WHERE decision NOT IN ('include','exclude')"
        ).fetchone()[0]
        if invalid_decisions:
            raise ProtocolError("Final screening decisions must be include or exclude")
        protocol = self.get_metadata("protocol", {})
        allowed_types = set(protocol.get("eligible_work_types", []))
        cutoff_year = int(str(protocol.get("cutoff_date", "0"))[:4])
        invalid_includes = 0
        registered_runs = {
            row[0]
            for row in self.connection.execute("SELECT run_id FROM review_sessions")
        }
        for row in self.connection.execute(
            "SELECT w.* FROM works w JOIN screening_final f USING(work_id) "
            "WHERE f.decision='include'"
        ):
            invalid_includes += int(
                row["language"] != "en"
                or row["work_type"] not in allowed_types
                or row["publication_year"] is None
                or int(row["publication_year"]) > cutoff_year
            )
            reviewer_roles = {
                decision[0]
                for decision in self.connection.execute(
                    "SELECT reviewer_role FROM screening_decisions WHERE work_id=?",
                    (row["work_id"],),
                )
            }
            invalid_includes += int(
                not {"Primary AI", "Independent Reviewer AI"}.issubset(reviewer_roles)
            )
            decisions = list(
                self.connection.execute(
                    "SELECT decision,language_evidence,eligibility_evidence,reason "
                    "FROM screening_decisions WHERE work_id=?",
                    (row["work_id"],),
                )
            )
            invalid_includes += int(
                any(
                    not decision[1].strip()
                    or not decision[2].strip()
                    or not decision[3].strip()
                    for decision in decisions
                )
            )
            final_row = self.connection.execute(
                "SELECT adjudicator_run_id FROM screening_final WHERE work_id=?",
                (row["work_id"],),
            ).fetchone()
            invalid_includes += int(
                not final_row or final_row[0] not in registered_runs
            )
            if row["publication_year"] == cutoff_year:
                date_row = self.connection.execute(
                    "SELECT publication_date FROM work_publication_dates WHERE work_id=?",
                    (row["work_id"],),
                ).fetchone()
                invalid_includes += int(
                    not date_row
                    or not date_row[0]
                    or date_row[0] > protocol.get("cutoff_date", "")
                )
        if invalid_includes:
            raise ProtocolError(
                f"{invalid_includes} included works fail type/date/language/independent-review gates"
            )
        self.set_metadata("stage", "screen")
        self.connection.commit()
        return final

    def validate_dimensions(self) -> int:
        invalid = 0
        included_works = {
            row[0]
            for row in self.connection.execute(
                "SELECT work_id FROM screening_final WHERE decision='include'"
            )
        }
        for row in self.connection.execute("SELECT * FROM candidate_dimensions"):
            sources = _loads(row["source_work_ids_json"], [])
            teams = _loads(row["independent_teams_json"], [])
            linked_mentions = self.connection.execute(
                (
                    "SELECT COUNT(*) FROM construct_mentions WHERE work_id IN "
                    f"({','.join('?' for _ in sources)})"
                    if sources
                    else "SELECT 0"
                ),
                tuple(sources),
            ).fetchone()[0]
            if not (
                row["definition"].strip()
                and row["role"].strip()
                and row["t0_boundary"].strip()
                and sources
                and teams
                and row["primary_approved"]
                and row["independent_approved"]
                and row["merge_split_log_json"].strip() not in {"", "[]"}
                and set(sources).issubset(included_works)
                and linked_mentions > 0
            ):
                invalid += 1
        if invalid:
            raise ProtocolError(
                f"{invalid} candidate dimensions lack required evidence/review"
            )
        count = self.connection.execute(
            "SELECT COUNT(*) FROM candidate_dimensions"
        ).fetchone()[0]
        self.set_metadata("stage", "derive-dimensions")
        self.connection.commit()
        return count

    def validate_indicator_census(self) -> int:
        dimensions = {
            row[0]
            for row in self.connection.execute(
                "SELECT dimension_id FROM candidate_dimensions"
            )
        }
        covered: set[str] = set()
        invalid = 0
        included_works = {
            row[0]
            for row in self.connection.execute(
                "SELECT work_id FROM screening_final WHERE decision='include'"
            )
        }
        mention_rows = {
            row["mention_id"]: row
            for row in self.connection.execute("SELECT * FROM indicator_mentions")
        }
        for row in self.connection.execute("SELECT * FROM indicator_families"):
            dimension_ids = set(_loads(row["dimension_ids_json"], []))
            mention_ids = set(_loads(row["mention_ids_json"], []))
            source_ids = set(_loads(row["definition_source_ids_json"], []))
            covered.update(dimension_ids)
            valid_mentions = all(
                mention_id in mention_rows
                and mention_rows[mention_id]["dimension_id"] in dimension_ids
                and mention_rows[mention_id]["work_id"] in included_works
                for mention_id in mention_ids
            )
            evidence_count = self.connection.execute(
                "SELECT COUNT(*) FROM indicator_evidence WHERE indicator_id=?",
                (row["indicator_id"],),
            ).fetchone()[0]
            if not (
                row["definition"].strip()
                and _loads(row["definition_source_ids_json"], [])
                and row["missing_rule"].strip()
                and row["zero_denominator_rule"].strip()
                and row["empty_set_rule"].strip()
                and row["coverage_rule"].strip()
                and row["fallback_rule"].strip()
                and dimension_ids.issubset(dimensions)
                and valid_mentions
                and source_ids
                and evidence_count > 0
            ):
                invalid += 1
        missing = dimensions - covered
        if invalid or missing:
            raise ProtocolError(
                f"Indicator census incomplete: invalid_families={invalid}, uncovered_dimensions={sorted(missing)}"
            )
        count = self.connection.execute(
            "SELECT COUNT(*) FROM indicator_families"
        ).fetchone()[0]
        self.set_metadata("stage", "census-indicators")
        self.connection.commit()
        return count

    def select_features(self) -> dict[str, list[str]]:
        self.validate_dimensions()
        self.validate_indicator_census()
        self.connection.execute("DELETE FROM final_dimensions")
        self.connection.execute("DELETE FROM final_features")
        families = list(
            self.connection.execute(
                "SELECT * FROM indicator_families ORDER BY indicator_id"
            )
        )
        gates = {
            row["indicator_id"]: row
            for row in self.connection.execute("SELECT * FROM hard_gate_decisions")
        }
        mappings = {
            row["indicator_id"]: row
            for row in self.connection.execute("SELECT * FROM indicator_data_mapping")
        }
        tiers = {
            row["indicator_id"]: row
            for row in self.connection.execute("SELECT * FROM evidence_tiers")
        }
        sets: dict[str, list[str]] = {
            "all": [],
            "model": [],
            "strict": [],
            "strict_training": [],
            "primary": [],
            "expanded": [],
            "broad_t0": [],
        }
        strict_dimensions = self._dimension_sets(families, gates, mappings, tiers)
        for family in families:
            indicator_id = family["indicator_id"]
            sets["all"].append(indicator_id)
            gate = gates.get(indicator_id)
            mapping = mappings.get(indicator_id)
            tier = tiers.get(indicator_id)
            effective = self._effective_gates(family, gate, mapping)
            all_pass = all(effective.values())
            tier_name = self._effective_tier(family, tier)
            if gate:
                self.connection.execute(
                    "UPDATE hard_gate_decisions SET h1_scope=?,h2_t0=?,h3_reproducibility=?,"
                    "h4_computability=?,h5_validity_ethics=?,h6_data_integrity=?,all_pass=?,"
                    "deterministic_evidence_json=? WHERE indicator_id=?",
                    (
                        *(int(effective[f"H{number}"]) for number in range(1, 7)),
                        int(all_pass),
                        canonical_json({"effective_gates": effective}),
                        indicator_id,
                    ),
                )
            if all_pass and tier_name:
                sets["model"].append(indicator_id)
                sets["expanded"].append(indicator_id)
                if tier_name in {"A", "B"} and family["role"] in {
                    "predictive",
                    "opportunity",
                    "control",
                }:
                    sets["primary"].append(indicator_id)
                dimensions = set(_loads(family["dimension_ids_json"], []))
                if tier_name == "A":
                    sets["strict"].append(indicator_id)
                    if dimensions & strict_dimensions and family["role"] in {
                        "predictive",
                        "opportunity",
                        "control",
                    }:
                        sets["strict_training"].append(indicator_id)
            broad_pass = bool(
                mapping
                and effective["H1"]
                and effective["H2"]
                and effective["H4"]
                and effective["H5"]
                and effective["H6"]
                and mapping["mapping_type"] in {"direct", "derivable"}
            )
            if broad_pass:
                sets["broad_t0"].append(indicator_id)
        freeze_hash = sha256_text(canonical_json(sets))
        for set_name, members in sets.items():
            for indicator_id in members:
                self.connection.execute(
                    "INSERT INTO final_features VALUES(?,?,?,?)",
                    (
                        indicator_id,
                        set_name,
                        "deterministic_protocol_rule",
                        freeze_hash,
                    ),
                )
        self.set_metadata("feature_set_freeze_hash", freeze_hash)
        self.set_metadata("stage", "select-features")
        self.connection.commit()
        payload = {
            "protocol_hash": self.get_metadata("protocol_hash"),
            "freeze_hash": freeze_hash,
            "frozen_before_model_training": True,
            "outcome_columns_used": self.get_metadata("outcome_blind_audit", {}).get(
                "outcome_columns_used"
            ),
            "sets": {
                key: value
                for key, value in sets.items()
                if key not in {"all", "model", "strict"}
            },
            "canonical_sets": {
                "F_all": sets["all"],
                "F_model": sets["model"],
                "F_strict": sets["strict"],
            },
        }
        self._write_json("final_feature_sets.json", payload)
        return sets

    def _dimension_sets(
        self,
        families: Sequence[sqlite3.Row],
        gates: Mapping[str, sqlite3.Row],
        mappings: Mapping[str, sqlite3.Row],
        tiers: Mapping[str, sqlite3.Row],
    ) -> set[str]:
        passing: dict[str, list[str]] = {}
        tier_a: dict[str, list[str]] = {}
        for family in families:
            indicator_id = family["indicator_id"]
            for dimension_id in _loads(family["dimension_ids_json"], []):
                effective = self._effective_gates(
                    family, gates.get(indicator_id), mappings.get(indicator_id)
                )
                if all(effective.values()):
                    passing.setdefault(dimension_id, []).append(indicator_id)
                    tier = tiers.get(indicator_id)
                    if self._effective_tier(family, tier) == "A":
                        tier_a.setdefault(dimension_id, []).append(indicator_id)
        strict: set[str] = set()
        for row in self.connection.execute("SELECT * FROM candidate_dimensions"):
            dimension_id = row["dimension_id"]
            if passing.get(dimension_id):
                self.connection.execute(
                    "INSERT INTO final_dimensions VALUES(?,?,?)",
                    (
                        dimension_id,
                        "supported",
                        "at_least_one_hard_gate_passing_indicator",
                    ),
                )
            teams = set(_loads(row["independent_teams_json"], []))
            if (
                tier_a.get(dimension_id)
                and len(teams) >= 2
                and row["independent_approved"]
                and row["independent_non_alias_confirmed"]
            ):
                strict.add(dimension_id)
                self.connection.execute(
                    "INSERT INTO final_dimensions VALUES(?,?,?)",
                    (
                        dimension_id,
                        "strict",
                        "tier_a_two_teams_independent_non_alias_review",
                    ),
                )
        return strict

    def _effective_gates(
        self,
        family: sqlite3.Row,
        gate: sqlite3.Row | None,
        mapping: sqlite3.Row | None,
    ) -> dict[str, bool]:
        """Combine reviewer judgments with non-overridable deterministic gates."""
        if not gate:
            return {f"H{number}": False for number in range(1, 7)}
        allowed_roles = {"predictive", "opportunity", "control", "sensitivity"}
        mapping_ready = bool(
            mapping
            and mapping["mapping_type"] in {"direct", "derivable"}
            and mapping["audit_status"] == "pass"
        )
        data_valid = bool(
            mapping_ready
            and not mapping["near_constant"]
            and (mapping["unique_count"] is None or mapping["unique_count"] > 1)
            and self.get_metadata("outcome_blind_audit", {}).get("status") == "pass"
            and self.get_metadata("outcome_blind_audit", {}).get("outcome_columns_used")
            is False
        )
        reproducible = bool(
            gate["h3_reproducibility"]
            and family["definition"].strip()
            and _loads(family["definition_source_ids_json"], [])
            and family["missing_rule"].strip()
            and family["zero_denominator_rule"].strip()
            and family["empty_set_rule"].strip()
            and family["coverage_rule"].strip()
            and family["fallback_rule"].strip()
        )
        return {
            "H1": bool(gate["h1_scope"] and family["role"] in allowed_roles),
            "H2": bool(gate["h2_t0"] and family["maximum_information_time"] == "T0"),
            "H3": reproducible,
            "H4": bool(gate["h4_computability"] and mapping_ready),
            "H5": bool(gate["h5_validity_ethics"]),
            "H6": bool(gate["h6_data_integrity"] and data_valid),
        }

    @staticmethod
    def _effective_tier(family: sqlite3.Row, tier: sqlite3.Row | None) -> str:
        if not tier or not tier["independent_approved"]:
            return ""
        teams = set(_loads(family["independent_teams_json"], []))
        requested = tier["tier"]
        if requested == "A" and len(teams) < 2:
            return "B" if teams else "C"
        if requested == "B" and not teams:
            return "C"
        return requested

    def materialize_training_sets(
        self, source: Path, id_column: str = "paper_id"
    ) -> dict[str, int]:
        import pandas as pd

        if not source.is_file():
            raise ProtocolError(f"Training source not found: {source}")
        frame = (
            pd.read_parquet(source)
            if source.suffix == ".parquet"
            else pd.read_csv(source)
        )
        if id_column not in frame.columns:
            raise ProtocolError(f"Training source lacks id column {id_column!r}")
        names = {
            row["indicator_id"]: row["canonical_name"]
            for row in self.connection.execute(
                "SELECT indicator_id,canonical_name FROM indicator_families"
            )
        }
        mapping = {
            "strict": "strict_training",
            "primary": "primary",
            "expanded": "expanded",
            "broad_t0": "broad_t0",
        }
        counts: dict[str, int] = {}
        matrix_sets: dict[str, dict[str, Any]] = {}
        for output_name, database_name in mapping.items():
            ids = [
                row[0]
                for row in self.connection.execute(
                    "SELECT indicator_id FROM final_features WHERE set_name=? ORDER BY indicator_id",
                    (database_name,),
                )
            ]
            columns = [names[item] for item in ids]
            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise ProtocolError(f"{output_name} source columns missing: {missing}")
            output = frame[[id_column, *columns]].copy()
            path = self.output_dir / f"final_training_features_{output_name}.parquet"
            output.to_parquet(path, index=False)
            counts[output_name] = len(columns)
            matrix_sets[output_name] = {
                "source_set": database_name,
                "indicator_ids": ids,
                "feature_names": columns,
                "path": str(path.resolve()),
                "sha256": file_hash(path),
                "row_count": len(output),
            }
        source_hash = file_hash(source)
        manifest = {
            "contract": "evidence_derived_training_matrices_v1",
            "protocol_hash": self.get_metadata("protocol_hash"),
            "feature_set_freeze_hash": self.get_metadata("feature_set_freeze_hash"),
            "frozen_before_model_training": True,
            "outcome_columns_used": self.get_metadata("outcome_blind_audit", {}).get(
                "outcome_columns_used"
            ),
            "id_column": id_column,
            "training_source": str(source.resolve()),
            "training_source_sha256": source_hash,
            "sets": matrix_sets,
        }
        self._write_json("training_matrix_manifest.json", manifest)
        self.set_metadata("training_source_hash", source_hash)
        self.set_metadata("training_matrix_counts", counts)
        self.connection.commit()
        return counts

    def counts(self) -> SelectionCounts:
        search = self._search_counts()
        scalar = lambda query: int(self.connection.execute(query).fetchone()[0])
        return SelectionCounts(
            k=search["K"],
            q=search["Q"],
            p=search["P"],
            m=scalar("SELECT COUNT(*) FROM candidate_dimensions"),
            d_supported=scalar(
                "SELECT COUNT(*) FROM final_dimensions WHERE set_name='supported'"
            ),
            d_strict=scalar(
                "SELECT COUNT(*) FROM final_dimensions WHERE set_name='strict'"
            ),
            f_all=scalar("SELECT COUNT(*) FROM final_features WHERE set_name='all'"),
            f_model=scalar(
                "SELECT COUNT(*) FROM final_features WHERE set_name='model'"
            ),
            f_strict=scalar(
                "SELECT COUNT(*) FROM final_features WHERE set_name='strict'"
            ),
        )

    def export_core_tables(self) -> None:
        exports = {
            "search_terms.csv": "SELECT * FROM term_families ORDER BY family_id",
            "search_domains.csv": "SELECT * FROM search_domains ORDER BY domain_id",
            "search_queries.csv": "SELECT * FROM logical_queries ORDER BY query_id",
            "seed_press_validation.csv": "SELECT * FROM seed_recall ORDER BY cohort,seed_id",
            "literature_screening.csv": "SELECT w.*,f.decision,f.exclusion_code,f.adjudication_reason FROM works w LEFT JOIN screening_final f USING(work_id) ORDER BY work_id",
            "candidate_dimensions.csv": "SELECT * FROM candidate_dimensions ORDER BY dimension_id",
            "indicator_library.csv": "SELECT * FROM indicator_families ORDER BY indicator_id",
            "indicator_decisions.csv": "SELECT g.*,t.tier,t.reason AS tier_reason,m.mapping_type,m.audit_status FROM hard_gate_decisions g LEFT JOIN evidence_tiers t USING(indicator_id) LEFT JOIN indicator_data_mapping m USING(indicator_id) ORDER BY indicator_id",
        }
        for name, query in exports.items():
            self._write_csv(name, query)
        prisma = self._prisma_counts()
        path = self.output_dir / "prisma_flow.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["stage", "count"])
            writer.writerows(prisma.items())

    def audit(self) -> dict[str, Any]:
        self.export_core_tables()
        blockers = self._audit_blockers()
        counts = self.counts().as_dict()
        artifact_hashes = {
            name: file_hash(self.output_dir / name)
            for name in REQUIRED_OUTPUTS
            if name != "audit_report.md" and (self.output_dir / name).is_file()
        }
        artifact_hashes.update(
            {
                f"implementation:{path.name}": file_hash(path)
                for path in (
                    ROOT / "core.py",
                    ROOT / "pipeline.py",
                    ROOT / "providers.py",
                    ROOT / "schema.sql",
                )
            }
        )
        artifact_hashes["frozen_metadata"] = sha256_text(
            canonical_json(
                {
                    key: self.get_metadata(key)
                    for key in (
                        "protocol_hash",
                        "legacy_inventory",
                        "legacy_decisions_imported",
                        "search_frame",
                        "outcome_blind_audit",
                        "feature_set_freeze_hash",
                        "training_source_hash",
                        "training_matrix_counts",
                    )
                }
            )
        )
        artifact_hashes.update(
            {
                f"sqlite:{table}": sha256_text(
                    canonical_json(
                        [
                            dict(row)
                            for row in self.connection.execute(
                                f"SELECT * FROM {table} ORDER BY 1"
                            )
                        ]
                    )
                )
                for table in AUDIT_TABLES
            }
        )
        deterministic_hash = sha256_text(
            canonical_json(
                {
                    "protocol": self.get_metadata("protocol_hash"),
                    "counts": counts,
                    "artifacts": artifact_hashes,
                    "blockers": blockers,
                }
            )
        )
        previous = self.connection.execute(
            "SELECT deterministic_hash FROM audit_manifest ORDER BY audit_id DESC LIMIT 1"
        ).fetchone()
        if not blockers and (not previous or previous[0] != deterministic_hash):
            blockers.append("TWO_CONSECUTIVE_AUDITS_NOT_CONFIRMED")
        status = "COMPLETE" if not blockers else "INCOMPLETE"
        report = self._audit_markdown(
            status, counts, blockers, deterministic_hash, artifact_hashes
        )
        (self.output_dir / "audit_report.md").write_text(report, encoding="utf-8")
        self.connection.execute(
            "INSERT INTO audit_manifest(status,deterministic_hash,blockers_json,counts_json,artifact_hashes_json,created_at) VALUES(?,?,?,?,?,?)",
            (
                status,
                deterministic_hash,
                canonical_json(blockers),
                canonical_json(counts),
                canonical_json(artifact_hashes),
                utc_now(),
            ),
        )
        self.set_metadata("stage", "audit")
        self.connection.commit()
        return {
            "status": status,
            "counts": counts,
            "blockers": blockers,
            "deterministic_hash": deterministic_hash,
        }

    def _audit_blockers(self) -> list[str]:
        blockers: list[str] = []
        stop = self.connection.execute(
            "SELECT 1 FROM saturation_rounds WHERE decision='stop' AND fully_reviewed=1 LIMIT 1"
        ).fetchone()
        if not stop:
            blockers.append("SEARCH_NOT_STOPPED_BY_ZERO_ZERO_OR_ROUND_15")
        if not self.get_metadata("search_frame"):
            blockers.append("SEARCH_FRAME_NOT_FROZEN")
        elif (
            self.get_metadata("search_frame", {}).get("hash")
            != self._search_frame_hash()
        ):
            blockers.append("FROZEN_SEARCH_FRAME_HASH_MISMATCH")
        incomplete_rounds = self.connection.execute(
            "SELECT COUNT(*) FROM saturation_rounds WHERE fully_reviewed<>1 "
            "OR new_term_families<0 OR new_indicator_families<0 OR length(evidence_hash)<>64"
        ).fetchone()[0]
        if incomplete_rounds:
            blockers.append(f"INVALID_SATURATION_ROUNDS:{incomplete_rounds}")
        blocked_provider_runs = self.connection.execute(
            "SELECT COUNT(*) FROM search_runs blocked WHERE blocked.status='blocked' "
            "AND NOT EXISTS (SELECT 1 FROM search_runs recovered "
            "WHERE recovered.provider=blocked.provider AND recovered.query_id=blocked.query_id "
            "AND recovered.status='complete' AND recovered.completed_at>blocked.completed_at)"
        ).fetchone()[0]
        if blocked_provider_runs:
            blockers.append(f"PROVIDER_RUNS_BLOCKED:{blocked_provider_runs}")
        works = self.connection.execute("SELECT COUNT(*) FROM works").fetchone()[0]
        finals = self.connection.execute(
            "SELECT COUNT(*) FROM screening_final"
        ).fetchone()[0]
        if works == 0 or finals != works:
            blockers.append(f"FORMAL_SCREENING_INCOMPLETE:{finals}/{works}")
        elif works:
            try:
                self.finalize_screening()
            except ProtocolError as error:
                blockers.append(f"FORMAL_SCREENING_INVALID:{error}")
        dimensions = self.connection.execute(
            "SELECT COUNT(*) FROM candidate_dimensions"
        ).fetchone()[0]
        families = self.connection.execute(
            "SELECT COUNT(*) FROM indicator_families"
        ).fetchone()[0]
        if dimensions == 0:
            blockers.append("NO_EVIDENCE_DERIVED_CANDIDATE_DIMENSIONS")
        else:
            try:
                self.validate_dimensions()
            except ProtocolError as error:
                blockers.append(f"CANDIDATE_DIMENSIONS_INVALID:{error}")
        if families == 0:
            blockers.append("INDICATOR_CENSUS_EMPTY")
        else:
            try:
                self.validate_indicator_census()
            except ProtocolError as error:
                blockers.append(f"INDICATOR_CENSUS_INVALID:{error}")
        reviewer_roles = {
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT reviewer_role FROM review_sessions"
            )
        }
        if "Primary AI" not in reviewer_roles:
            blockers.append("MISSING_PRIMARY_AI_REVIEW_SESSION")
        if "Independent Reviewer AI" not in reviewer_roles:
            blockers.append("MISSING_INDEPENDENT_REVIEW_SESSION")
        invalid_sessions = 0
        for session in self.connection.execute("SELECT * FROM review_sessions"):
            hashes_valid = bool(
                re.fullmatch(r"[0-9a-f]{64}", session["input_hash"])
                and re.fullmatch(r"[0-9a-f]{64}", session["output_hash"])
            )
            try:
                evidence = json.loads(session["evidence"])
            except json.JSONDecodeError:
                evidence = {}
            invalid_sessions += int(
                not hashes_valid
                or not session["model_label"].strip()
                or not session["reason"].strip()
                or not isinstance(evidence, dict)
                or not evidence.get("stage")
                or not evidence.get("object_ids")
            )
        if invalid_sessions:
            blockers.append(f"INVALID_REVIEW_SESSION_PROVENANCE:{invalid_sessions}")
        missing_gates = self.connection.execute(
            "SELECT COUNT(*) FROM indicator_families f LEFT JOIN hard_gate_decisions g USING(indicator_id) WHERE g.indicator_id IS NULL"
        ).fetchone()[0]
        if missing_gates:
            blockers.append(f"INDICATORS_WITHOUT_HARD_GATE_DECISION:{missing_gates}")
        missing_tiers = self.connection.execute(
            "SELECT COUNT(*) FROM indicator_families f LEFT JOIN evidence_tiers t USING(indicator_id) "
            "WHERE t.indicator_id IS NULL OR t.independent_approved<>1"
        ).fetchone()[0]
        if missing_tiers:
            blockers.append(
                f"INDICATORS_WITHOUT_APPROVED_EVIDENCE_TIER:{missing_tiers}"
            )
        missing_evidence = self.connection.execute(
            "SELECT COUNT(*) FROM indicator_families f WHERE NOT EXISTS "
            "(SELECT 1 FROM indicator_evidence e WHERE e.indicator_id=f.indicator_id)"
        ).fetchone()[0]
        if missing_evidence:
            blockers.append(f"INDICATORS_WITHOUT_SOURCE_EVIDENCE:{missing_evidence}")
        if not self.get_metadata("feature_set_freeze_hash"):
            blockers.append("FOUR_FEATURE_SETS_NOT_FROZEN")
        for name in (
            "final_training_features_strict.parquet",
            "final_training_features_primary.parquet",
            "final_training_features_expanded.parquet",
            "final_training_features_broad_t0.parquet",
        ):
            if not (self.output_dir / name).is_file():
                blockers.append(f"MISSING_TRAINING_MATRIX:{name}")
        if not (self.output_dir / "training_matrix_manifest.json").is_file():
            blockers.append("MISSING_TRAINING_MATRIX_MANIFEST")
        if not any(item.startswith("MISSING_TRAINING_MATRIX:") for item in blockers):
            try:
                self._validate_training_matrices()
            except (ProtocolError, OSError, ValueError) as error:
                blockers.append(f"TRAINING_MATRICES_INVALID:{error}")
        outcome_audit = self.get_metadata("outcome_blind_audit", {})
        if (
            outcome_audit.get("status") != "pass"
            or outcome_audit.get("outcome_columns_used") is not False
        ):
            blockers.append("OUTCOME_BLIND_AUDIT_MISSING_OR_FAILED")
        return sorted(set(blockers))

    def _validate_training_matrices(self, id_column: str = "paper_id") -> None:
        import pandas as pd

        if not self.get_metadata("training_source_hash"):
            raise ProtocolError("training source hash is missing")
        manifest_path = self.output_dir / "training_matrix_manifest.json"
        if not manifest_path.is_file():
            raise ProtocolError("training matrix manifest is missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != "evidence_derived_training_matrices_v1":
            raise ProtocolError("training matrix manifest contract is invalid")
        if manifest.get("feature_set_freeze_hash") != self.get_metadata(
            "feature_set_freeze_hash"
        ):
            raise ProtocolError("training matrix freeze hash mismatch")
        if manifest.get("protocol_hash") != self.get_metadata("protocol_hash"):
            raise ProtocolError("training matrix protocol hash mismatch")
        if manifest.get("outcome_columns_used") is not False:
            raise ProtocolError("training matrix outcome-blind audit failed")
        if manifest.get("training_source_sha256") != self.get_metadata(
            "training_source_hash"
        ):
            raise ProtocolError("training matrix source hash mismatch")
        names = {
            row["indicator_id"]: row["canonical_name"]
            for row in self.connection.execute(
                "SELECT indicator_id,canonical_name FROM indicator_families"
            )
        }
        database_sets = {
            "strict": "strict_training",
            "primary": "primary",
            "expanded": "expanded",
            "broad_t0": "broad_t0",
        }
        if set(manifest.get("sets") or {}) != set(database_sets):
            raise ProtocolError("training matrix set names are invalid")
        row_counts: set[int] = set()
        for output_name, set_name in database_sets.items():
            path = self.output_dir / f"final_training_features_{output_name}.parquet"
            frame = pd.read_parquet(path)
            expected_ids = [
                row[0]
                for row in self.connection.execute(
                    "SELECT indicator_id FROM final_features WHERE set_name=? ORDER BY indicator_id",
                    (set_name,),
                )
            ]
            expected_columns = [id_column, *(names[item] for item in expected_ids)]
            if list(frame.columns) != expected_columns:
                raise ProtocolError(
                    f"{output_name} columns do not match frozen membership"
                )
            if frame[id_column].isna().any() or frame[id_column].duplicated().any():
                raise ProtocolError(f"{output_name} paper identifiers are invalid")
            definition = manifest["sets"][output_name]
            if (
                definition.get("source_set") != set_name
                or definition.get("indicator_ids") != expected_ids
                or definition.get("feature_names") != expected_columns[1:]
                or int(definition.get("row_count", -1)) != len(frame)
                or definition.get("sha256") != file_hash(path)
            ):
                raise ProtocolError(f"{output_name} manifest lineage is invalid")
            row_counts.add(len(frame))
        if len(row_counts) != 1:
            raise ProtocolError("training matrices have inconsistent row counts")

    def _prisma_counts(self) -> dict[str, int]:
        scalar = lambda query: int(self.connection.execute(query).fetchone()[0])
        return {
            "identified": scalar("SELECT COUNT(*) FROM works"),
            "screened": scalar("SELECT COUNT(*) FROM screening_final"),
            "included": scalar(
                "SELECT COUNT(*) FROM screening_final WHERE decision='include'"
            ),
            "excluded": scalar(
                "SELECT COUNT(*) FROM screening_final WHERE decision='exclude'"
            ),
            "uncertain": scalar(
                "SELECT COUNT(*) FROM screening_final WHERE decision='uncertain'"
            ),
        }

    def _write_csv(self, name: str, query: str) -> None:
        cursor = self.connection.execute(query)
        path = self.output_dir / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([column[0] for column in cursor.description])
            writer.writerows(cursor.fetchall())

    def _write_json(self, name: str, value: Any) -> None:
        (self.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _audit_markdown(
        status: str,
        counts: Mapping[str, int],
        blockers: Sequence[str],
        deterministic_hash: str,
        artifacts: Mapping[str, str],
    ) -> str:
        count_lines = "\n".join(f"- `{key}`: {value}" for key, value in counts.items())
        blocker_lines = "\n".join(f"- `{value}`" for value in blockers) or "- None"
        artifact_lines = "\n".join(
            f"- `{key}`: `{value}`" for key, value in sorted(artifacts.items())
        )
        return f"""# Evidence-derived audit report

Status: **{status}**

## Emergent counts

{count_lines}

## Completion blockers

{blocker_lines}

## Reproducibility

- Deterministic hash: `{deterministic_hash}`
- Selection was outcome-blind and feature sets were frozen before training.

## Core artifact hashes

{artifact_lines}

## Scope limitation

Only English evidence is eligible. This restriction may underrepresent non-English research traditions, regions, venues, constructs, and validation evidence.
"""


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    raise ProtocolError(f"Expected a JSON array of objects: {path}")


def insert_rows(
    connection: sqlite3.Connection,
    table: str,
    rows: Iterable[Mapping[str, Any]],
    allowed_tables: set[str],
) -> int:
    """Strict JSON-row import used for independently reviewed artifacts."""
    if table not in allowed_tables:
        raise ProtocolError(f"Import is not allowed for table {table!r}")
    columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
    count = 0
    for row in rows:
        unknown = set(row) - set(columns)
        if unknown:
            raise ProtocolError(f"Unknown {table} columns: {sorted(unknown)}")
        names = list(row)
        values = [
            (
                canonical_json(row[name])
                if isinstance(row[name], (list, dict))
                else row[name]
            )
            for name in names
        ]
        if table == "review_sessions":
            existing = connection.execute(
                "SELECT * FROM review_sessions WHERE review_session_id=?",
                (row.get("review_session_id"),),
            ).fetchone()
            if existing:
                existing_value = {column: existing[column] for column in names}
                incoming_value = dict(zip(names, values, strict=True))
                if existing_value != incoming_value:
                    raise ProtocolError("Review sessions are immutable once registered")
                count += 1
                continue
        placeholders = ",".join("?" for _ in names)
        update = ",".join(f"{name}=excluded.{name}" for name in names)
        connection.execute(
            f"INSERT INTO {table}({','.join(names)}) VALUES({placeholders}) "
            f"ON CONFLICT DO UPDATE SET {update}",
            values,
        )
        count += 1
    connection.commit()
    return count
