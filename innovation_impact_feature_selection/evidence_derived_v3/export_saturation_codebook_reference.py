from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


TERM_FIELDS = (
    "term_id",
    "verbatim_term",
    "source_type",
    "proposed_role",
    "origin_round",
    "canonical_term",
    "term_family_label",
    "term_relation",
    "search_domain_label",
    "search_domain_definition",
    "query_family_label",
    "cross_domain",
)
INDICATOR_FIELDS = (
    "candidate_id",
    "review_round",
    "raw_name_en",
    "proposed_role",
    "canonical_family_label",
    "evidence_span",
)


def sha256_file(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash a canonical JSON representation of query rows."""
    payload = json.dumps(
        [dict(row) for row in rows],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write one deterministic UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def term_rows(
    connection: sqlite3.Connection,
    through_round: int,
) -> List[Dict[str, Any]]:
    """Return finalized H2 term assignments available before a new round."""
    rows = connection.execute(
        """
        SELECT t.term_id, r.verbatim_term, r.source_type, r.proposed_role,
               CASE
                   WHEN r.source_type IN (
                       'development_seed_hint', 'pilot_v2_indicator'
                   ) THEN 0
                   ELSE COALESCE((
                       SELECT MIN(h.review_round)
                       FROM discovery_hits h
                       WHERE h.record_key = r.source_record_key
                         AND h.review_round BETWEEN 1 AND ?
                   ), -1)
               END AS origin_round,
               t.canonical_term, t.term_family_label, t.term_relation,
               t.search_domain_label, t.search_domain_definition,
               t.query_family_label, t.cross_domain
        FROM term_coding t
        JOIN raw_terms r USING(term_id)
        WHERE t.coder_role = 'H2'
          AND t.decision = 'include'
          AND (
              r.source_type IN (
                  'development_seed_hint', 'pilot_v2_indicator'
              )
              OR EXISTS (
                  SELECT 1 FROM discovery_hits h
                  WHERE h.record_key = r.source_record_key
                    AND h.review_round BETWEEN 1 AND ?
              )
          )
        ORDER BY t.term_family_label, t.canonical_term, t.term_id
        """,
        (through_round, through_round),
    ).fetchall()
    return [dict(row) for row in rows]


def indicator_rows(
    connection: sqlite3.Connection,
    through_round: int,
) -> List[Dict[str, Any]]:
    """Return finalized H2 discovery-indicator assignments by round."""
    rows = connection.execute(
        """
        SELECT candidate_id, review_round, raw_name_en, proposed_role,
               canonical_family_label, evidence_span
        FROM discovery_indicator_candidates
        WHERE h2_decision = 'include'
          AND review_round BETWEEN 1 AND ?
        ORDER BY canonical_family_label, review_round, candidate_id
        """,
        (through_round,),
    ).fetchall()
    return [dict(row) for row in rows]


def export_codebook_reference(
    connection: sqlite3.Connection,
    through_round: int,
    term_output: Path,
    indicator_output: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Export one deterministic prior-round H2 normalization reference."""
    terms = term_rows(connection, through_round)
    indicators = indicator_rows(connection, through_round)
    write_csv(term_output, TERM_FIELDS, terms)
    write_csv(indicator_output, INDICATOR_FIELDS, indicators)
    manifest = {
        "artifact_type": "prior_round_h2_codebook_reference",
        "manifest_version": 3,
        "through_round": through_round,
        "term_output": str(term_output.resolve()),
        "term_output_sha256": sha256_file(term_output),
        "term_source_rows_sha256": rows_digest(terms),
        "term_rows": len(terms),
        "term_families": len(
            {str(row["term_family_label"]) for row in terms}
        ),
        "query_families": len(
            {str(row["query_family_label"]) for row in terms}
        ),
        "indicator_output": str(indicator_output.resolve()),
        "indicator_output_sha256": sha256_file(indicator_output),
        "indicator_source_rows_sha256": rows_digest(indicators),
        "indicator_rows": len(indicators),
        "indicator_families": len(
            {str(row["canonical_family_label"]) for row in indicators}
        ),
        "selection_rule": (
            "Final H2 include assignments with origin/review round at or "
            "before through_round; no dimensions or model outcomes."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main() -> None:
    """Export auditable prior-round codebook references."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--through-round", required=True, type=int)
    parser.add_argument("--term-output", required=True)
    parser.add_argument("--indicator-output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    database = Path(args.database).resolve()
    term_output = Path(args.term_output).resolve()
    indicator_output = Path(args.indicator_output).resolve()
    manifest_path = Path(args.manifest).resolve()
    connection = sqlite3.connect(
        f"file:{database}?mode=ro",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    manifest = export_codebook_reference(
        connection,
        args.through_round,
        term_output,
        indicator_output,
        manifest_path,
    )
    connection.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
