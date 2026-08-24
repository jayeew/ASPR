from __future__ import annotations

import json
import re
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping

import saturation
import screening
import coding
import indicators
from common import (
    json_hash,
    normalize_term,
    sha256_bytes,
    utc_now,
    write_csv,
    write_json,
)
from database import (
    invalidate_stages,
    log_event,
    require_complete,
    snapshot_import_file,
)


OLLAMA_BASE_URL = "http://localhost:11434"
SCREENING_SYSTEM_PROMPT = """
You are the AI title/abstract screener in a preregistered evidence map.
Judge whether this scholarly work is useful evidence about an individual
scientific paper's novelty, innovation, publication-time potential scholarly
impact, or T0 predictors/opportunity/context features of later scholarly
impact. Eligible evidence must define, measure, apply, predict, determine,
review, or validate an article-level construct or indicator. A study may use
future citations only as a validation outcome when it also studies T0
predictors; future citations themselves are not an eligible T0 feature.
INCLUDE studies that relate publication-time title, abstract, references,
authors, teams, topics, interdisciplinarity, networks, openness, venue, or
other paper attributes to later citations, popularity, influence, or
high-impact status. These are predictor-validation studies, even when they use
words such as correlation, association, explain, or influence rather than
predict. INCLUDE article-level metric-definition or validation studies for
novelty, diversity, interdisciplinarity, disruption, readability, openness,
network position, or related candidate constructs; later hard gates decide
whether the metric is a valid T0 feature. Exclude with E_FUTURE_OUTCOME_ONLY
only when no publication-time predictor or construct is studied.
Exclude clinical, educational, corporate, technological-product, patent-only,
or other outcomes merely using generic words such as impact, innovation,
article, feature, predictor, validation, or review. Exclude author-, journal-,
institution-, country-, or field-level studies unless they explicitly provide
an individual-paper measure or predictor. Use only the supplied title and
abstract. Do not infer missing content. Give one exact short substring copied
from the title or abstract as evidence.
""".strip()
EXCLUSION_REASON_CODES = (
    "E_LANGUAGE_NON_ENGLISH",
    "E_NOT_PAPER_LEVEL_INNOVATION_OR_POTENTIAL_IMPACT",
    "E_FUTURE_OUTCOME_ONLY",
    "E_NOT_ARTICLE_LEVEL",
    "E_NOT_INDICATOR_PREDICTOR_VALIDATION",
    "E_DUPLICATE",
    "E_INSUFFICIENT_METADATA",
)
SCREENING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "language_judgment": {
            "type": "string",
            "enum": ["en", "non_en", "uncertain"],
        },
        "decision": {
            "type": "string",
            "enum": ["include", "exclude", "uncertain"],
        },
        "exclusion_reason": {
            "type": "string",
            "enum": ["", *EXCLUSION_REASON_CODES],
        },
        "evidence_span": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "language_judgment",
        "decision",
        "exclusion_reason",
        "evidence_span",
        "rationale",
    ],
}
TERM_CODEBOOK_SYSTEM_PROMPT = """
You are the AI terminology coder in a preregistered evidence map of
scientific-paper novelty and publication-time potential scholarly impact.
Derive a concise, non-redundant set of search concept domains from the supplied
English evidence terms. Do not target a particular number and do not reuse any
pre-existing D01-D12 scheme. Merge synonyms and parameter variants, but split
terms when they measure a different theoretical object or have a different
role (substantive paper content, opportunity/attention, context/control, or
future outcome). Domain labels must be short English noun phrases. Domains are
search terminology families, not final model dimensions. Every
development_seed formula term must be covered by at least one domain. Example
terms are illustrative only and must never be treated as a whitelist.
""".strip()
TERM_CODEBOOK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "definition": {"type": "string"},
                    "example_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["label", "definition", "example_terms"],
            },
        },
        "derivation_notes": {"type": "string"},
    },
    "required": ["domains", "derivation_notes"],
}
TERM_CODING_SYSTEM_PROMPT = """
Independently code one English evidence term using the supplied AI-derived
search-domain codebook. Include it only if it is a construct, measure,
indicator, predictor, determinant, opportunity/context variable, validation
term, or outcome term useful for discovering evidence about individual-paper
novelty or potential scholarly impact. Do not treat a future outcome as a T0
feature, but it may remain a search term for validation evidence. Preserve
construct meaning when normalizing. A term family groups true synonyms,
abbreviations, historical names, and parameter/encoding variants; it must not
collapse distinct constructs. Return exact codebook labels only. Codebook
example terms are non-exhaustive. Do not exclude a term because it is absent
from the examples. Do not exclude a synonym, redundant spelling, or parameter
variant: include it and map it to the shared canonical term family.
Source-linked development_seed formula terms should normally be included.
Pilot candidate indicators should remain searchable even when their eventual
feature role is context, opportunity, sensitivity, or future validation
outcome. Exclude only terms genuinely outside individual-paper evidence or
terms so vague that no construct meaning can be preserved.
""".strip()
TERM_CODING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["include", "exclude"],
        },
        "canonical_term": {"type": "string"},
        "term_family_label": {"type": "string"},
        "term_relation": {
            "type": "string",
            "enum": sorted(coding.TERM_RELATIONS),
        },
        "domain_labels": {
            "type": "array",
            "items": {"type": "string"},
        },
        "query_family_label": {"type": "string"},
        "cross_domain": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "decision",
        "canonical_term",
        "term_family_label",
        "term_relation",
        "domain_labels",
        "query_family_label",
        "cross_domain",
        "reason",
    ],
}
DIMENSION_CODING_SYSTEM_PROMPT = """
You are the AI construct coder in a preregistered evidence map of
publication-time scientific-paper innovation and potential scholarly impact.
Independently map one canonical indicator family to an evidence-derived model
construct. Do not target a number of dimensions, reuse D01-D12 labels, infer
from another coder, or use predictive model performance. Merge only families
that measure the same theoretical object and explanatory role. Keep
substantive innovation, T0 potential, opportunity, context control, and
sensitivity roles distinct. Exclude families whose evidence cannot support a
paper-level T0 construct. Use only the supplied formula, data requirements,
information-time, scope, source-team, and mention evidence.
""".strip()
DIMENSION_CODING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["include", "exclude"],
        },
        "dimension_label": {"type": "string"},
        "dimension_definition": {"type": "string"},
        "construct_role": {
            "type": "string",
            "enum": sorted(indicators.CONSTRUCT_ROLES),
        },
        "information_source": {"type": "string"},
        "t0_boundary": {"type": "string"},
        "bias_risk": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": [
        "decision",
        "dimension_label",
        "dimension_definition",
        "construct_role",
        "information_source",
        "t0_boundary",
        "bias_risk",
        "reason",
    ],
}


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    timeout_seconds: int = 240,
) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout_seconds,
    ) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Local model response is not a JSON object")
    return value


def _model_digest(model: str) -> str:
    with urllib.request.urlopen(
        f"{OLLAMA_BASE_URL}/api/tags",
        timeout=10,
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for item in payload.get("models", []):
        if item.get("name") == model or item.get("model") == model:
            return str(item.get("digest") or "")
    raise RuntimeError(f"Local Ollama model is unavailable: {model}")


def _chat_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: Mapping[str, Any],
    retries: int = 3,
    num_predict: int = 500,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            payload = {
                "model": model,
                "stream": False,
                "think": False,
                "format": dict(schema),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": 0,
                    "seed": 20_260_729,
                    "num_predict": num_predict,
                },
            }
            response = _post_json(
                f"{OLLAMA_BASE_URL}/api/chat",
                payload,
            )
            message = response.get("message")
            if not isinstance(message, dict):
                raise ValueError("Local model response has no message")
            content = json.loads(str(message.get("content") or ""))
            if not isinstance(content, dict):
                raise ValueError("Local model did not return a JSON object")
            return content
        except (
            OSError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(float(2**attempt))
    raise RuntimeError(
        f"Local model failed after {retries} attempts: "
        f"{type(last_error).__name__ if last_error else 'unknown'}"
    )


def _exact_evidence_span(
    proposed: str,
    title: str,
    abstract: str,
) -> str:
    source = f"{title}\n{abstract}".strip()
    candidate = proposed.strip().strip("\"'")
    if candidate:
        index = source.casefold().find(candidate.casefold())
        if index >= 0:
            return source[index : index + len(candidate)]
    if title.strip():
        return title.strip()
    return abstract.strip()[:240]


def _automatic_non_english(record: sqlite3.Row) -> Dict[str, str] | None:
    language = str(record["language"] or "").casefold()
    if language in {"", "unknown", "en"}:
        return None
    return {
        "language_judgment": "non_en",
        "language_evidence": _exact_evidence_span(
            "",
            str(record["title"] or ""),
            str(record["abstract"] or ""),
        ),
        "decision": "exclude",
        "exclusion_reason": "E_LANGUAGE_NON_ENGLISH",
        "evidence_span": str(record["title"] or "")[:240],
        "rationale": f"OpenAlex language field is {language}.",
    }


def _screen_one(record: sqlite3.Row, model: str) -> Dict[str, str]:
    automatic = _automatic_non_english(record)
    if automatic is not None:
        return automatic
    title = str(record["title"] or "")
    abstract = str(record["abstract"] or "")
    response = _chat_json(
        model,
        SCREENING_SYSTEM_PROMPT,
        f"TITLE:\n{title}\n\nABSTRACT:\n{abstract}",
        SCREENING_SCHEMA,
    )
    language = str(response.get("language_judgment") or "uncertain")
    decision = str(response.get("decision") or "uncertain")
    reason = str(response.get("exclusion_reason") or "")
    if language == "non_en":
        decision = "exclude"
        reason = "E_LANGUAGE_NON_ENGLISH"
    elif decision == "exclude" and reason not in EXCLUSION_REASON_CODES:
        reason = "E_NOT_PAPER_LEVEL_INNOVATION_OR_POTENTIAL_IMPACT"
    elif decision != "exclude":
        reason = ""
    return {
        "language_judgment": language,
        "language_evidence": _exact_evidence_span("", title, abstract),
        "decision": decision,
        "exclusion_reason": reason,
        "evidence_span": _exact_evidence_span(
            str(response.get("evidence_span") or ""),
            title,
            abstract,
        ),
        "rationale": str(response.get("rationale") or "").strip(),
    }


def _upsert_ai_decision(
    connection: sqlite3.Connection,
    record_key: str,
    value: Mapping[str, str],
    model: str,
    prompt_hash: str,
) -> None:
    connection.execute(
        """
        INSERT INTO screening_decisions(
            record_key, reviewer_role, language_judgment,
            language_evidence, decision, exclusion_reason,
            evidence_span, notes, decided_at
        ) VALUES (?, 'AI', ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(record_key, reviewer_role) DO UPDATE SET
            language_judgment = excluded.language_judgment,
            language_evidence = excluded.language_evidence,
            decision = excluded.decision,
            exclusion_reason = excluded.exclusion_reason,
            evidence_span = excluded.evidence_span,
            notes = excluded.notes,
            decided_at = excluded.decided_at
        """,
        (
            record_key,
            value["language_judgment"],
            value["language_evidence"],
            value["decision"],
            value["exclusion_reason"],
            value["evidence_span"],
            value["rationale"],
            utc_now(),
        ),
    )


def _completed_ai_rows(
    connection: sqlite3.Connection,
    iteration: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in saturation._round_records(connection, iteration):
        decision = connection.execute(
            """
            SELECT * FROM screening_decisions
            WHERE record_key = ? AND reviewer_role = 'AI'
            """,
            (record["record_key"],),
        ).fetchone()
        if decision is None:
            continue
        rows.append(
            {
                "record_key": record["record_key"],
                "doi": record["doi"],
                "title": record["title"],
                "abstract": record["abstract"],
                "openalex_language": record["language"],
                "publication_year": record["publication_year"],
                "work_type": record["work_type"],
                "reviewer_role": "AI",
                "language_judgment": decision["language_judgment"],
                "language_evidence": decision["language_evidence"],
                "decision": decision["decision"],
                "exclusion_reason": decision["exclusion_reason"],
                "evidence_span": decision["evidence_span"],
                "notes": decision["notes"],
                "review_round": iteration,
                "discovery_query_ids": record["discovery_query_ids"],
            }
        )
    return rows


def normalize_ai_language_evidence(
    connection: sqlite3.Connection,
    output_dir: Path,
) -> Dict[str, Any]:
    """Replace legacy AI provenance prose with an exact frozen source span."""
    changes: List[Dict[str, str]] = []
    for row in connection.execute(
        """
        SELECT d.record_key, d.language_evidence, r.title, r.abstract
        FROM screening_decisions d
        JOIN records r USING(record_key)
        WHERE d.reviewer_role = 'AI'
        ORDER BY d.record_key
        """
    ):
        source = f"{row['title']}\n{row['abstract']}"
        old_value = str(row["language_evidence"] or "")
        if old_value and old_value in source:
            continue
        h2_exists = connection.execute(
            """
            SELECT 1 FROM screening_decisions
            WHERE record_key = ? AND reviewer_role = 'H2'
            """,
            (row["record_key"],),
        ).fetchone()
        if h2_exists is not None:
            raise RuntimeError(
                "AI provenance normalization must precede H2 adjudication: "
                f"{row['record_key']}"
            )
        replacement = _exact_evidence_span(
            "",
            str(row["title"] or ""),
            str(row["abstract"] or ""),
        )
        if not replacement or replacement not in source:
            raise ValueError(
                "Cannot derive an exact AI language-evidence span: "
                f"{row['record_key']}"
            )
        changes.append(
            {
                "record_key": str(row["record_key"]),
                "old_value": old_value,
                "new_value": replacement,
            }
        )
    if not changes:
        return {
            "changed": 0,
            "iterations_rewritten": [],
            "change_hash": json_hash({"changes": []}),
        }
    changed_keys = {item["record_key"] for item in changes}
    iterations = sorted(
        {
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT review_round FROM discovery_hits
                WHERE review_round > 0
                """
            )
            if any(
                str(value[0]) in changed_keys
                for value in connection.execute(
                    """
                    SELECT record_key FROM discovery_hits
                    WHERE review_round = ?
                    """,
                    (row[0],),
                )
            )
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for iteration in iterations:
        output_path = (
            output_dir
            / (
                f"discovery_round_{iteration}_"
                "screening_ai_completed_v3.csv"
            )
        )
        if output_path.is_file():
            snapshot_import_file(
                connection,
                output_path,
                "ai_screening_pre_exact_language_evidence",
            )
    for item in changes:
        connection.execute(
            """
            UPDATE screening_decisions
            SET language_evidence = ?, decided_at = ?
            WHERE record_key = ? AND reviewer_role = 'AI'
            """,
            (
                item["new_value"],
                utc_now(),
                item["record_key"],
            ),
        )
    fields = list(screening.SCREENING_FIELDS) + [
        "review_round",
        "discovery_query_ids",
    ]
    for iteration in iterations:
        write_csv(
            output_dir
            / (
                f"discovery_round_{iteration}_"
                "screening_ai_completed_v3.csv"
            ),
            _completed_ai_rows(connection, iteration),
            fields,
        )
    change_hash = json_hash({"changes": changes})
    log_event(
        connection,
        "ai_language_evidence_normalized",
        "collection",
        "screening_decisions",
        {
            "changed": len(changes),
            "iterations_rewritten": iterations,
            "change_hash": change_hash,
            "rule": (
                "Legacy model/provenance prose was replaced by the exact "
                "title, or abstract fallback, without changing any language "
                "or eligibility decision."
            ),
        },
    )
    invalidate_stages(
        connection,
        ("audit_complete",),
        "AI language-evidence provenance normalized to exact source spans",
    )
    connection.commit()
    return {
        "changed": len(changes),
        "iterations_rewritten": iterations,
        "change_hash": change_hash,
    }


def ai_screen_discovery_round(
    connection: sqlite3.Connection,
    iteration: int,
    model: str,
    output_path: Path,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the declared AI reviewer locally and persist resumable decisions."""
    prompt_hash = sha256_bytes(SCREENING_SYSTEM_PROMPT.encode("utf-8"))
    records = saturation._round_records(connection, iteration)
    if force:
        record_keys = [str(row["record_key"]) for row in records]
        if record_keys:
            placeholders = ",".join("?" for _ in record_keys)
            h2_rows = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM screening_decisions
                    WHERE reviewer_role = 'H2'
                      AND record_key IN ({placeholders})
                    """,
                    record_keys,
                ).fetchone()[0]
            )
            if h2_rows:
                raise RuntimeError(
                    "AI screening is frozen after H2 adjudication for this "
                    "round"
                )
            removed_rows = connection.execute(
                f"""
                SELECT COUNT(*) FROM screening_decisions
                WHERE reviewer_role = 'AI'
                  AND record_key IN ({placeholders})
                """,
                record_keys,
            ).fetchone()[0]
            connection.execute(
                f"""
                DELETE FROM screening_decisions
                WHERE reviewer_role = 'AI'
                  AND record_key IN ({placeholders})
                """,
                record_keys,
            )
            log_event(
                connection,
                "ai_screening_replaced",
                "discovery_round",
                str(iteration),
                {
                    "removed_rows": removed_rows,
                    "reason": (
                        "Explicit --force after an AI prompt, input, or "
                        "source-provenance revision"
                    ),
                },
            )
            connection.commit()
    digest = _model_digest(model)
    input_hash = json_hash(
        {
            "records": [
                {
                    "record_key": row["record_key"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "language": row["language"],
                }
                for row in records
            ]
        }
    )
    run_id = "AIRUN_" + sha256_bytes(
        f"screen|{model}|{prompt_hash}|{input_hash}".encode("utf-8")
    )[:16].upper()
    connection.execute(
        """
        INSERT INTO ai_assistance_runs(
            run_id, task, reviewer_role, model, model_digest,
            prompt_hash, input_hash, output_path, completed_items,
            failed_items, status, started_at, completed_at
        ) VALUES (?, 'discovery_screening', 'AI', ?, ?, ?, ?, ?, 0, 0,
                  'running', ?, '')
        ON CONFLICT(run_id) DO UPDATE SET
            output_path = excluded.output_path,
            status = 'running'
        """,
        (
            run_id,
            model,
            digest,
            prompt_hash,
            input_hash,
            str(output_path.resolve()),
            utc_now(),
        ),
    )
    connection.commit()
    completed = 0
    failed = 0
    for index, record in enumerate(records, start=1):
        existing = connection.execute(
            """
            SELECT 1 FROM screening_decisions
            WHERE record_key = ? AND reviewer_role = 'AI'
            """,
            (record["record_key"],),
        ).fetchone()
        if existing is not None:
            completed += 1
            continue
        try:
            value = _screen_one(record, model)
            _upsert_ai_decision(
                connection,
                str(record["record_key"]),
                value,
                model,
                prompt_hash,
            )
            completed += 1
        except Exception as error:
            failed += 1
            fallback = {
                "language_judgment": (
                    "en"
                    if str(record["language"] or "").casefold() == "en"
                    else "uncertain"
                ),
                "language_evidence": _exact_evidence_span(
                    "",
                    str(record["title"] or ""),
                    str(record["abstract"] or ""),
                ),
                "decision": "uncertain",
                "exclusion_reason": "",
                "evidence_span": _exact_evidence_span(
                    "",
                    str(record["title"] or ""),
                    str(record["abstract"] or ""),
                ),
                "rationale": (
                    "Local AI structured output failed validation; "
                    "record is conservatively routed to human adjudication."
                ),
            }
            _upsert_ai_decision(
                connection,
                str(record["record_key"]),
                fallback,
                model,
                prompt_hash,
            )
            completed += 1
            log_event(
                connection,
                "ai_screening_error",
                "record",
                str(record["record_key"]),
                {"error_type": type(error).__name__},
            )
        if index % 10 == 0:
            connection.execute(
                """
                UPDATE ai_assistance_runs
                SET completed_items = ?, failed_items = ?
                WHERE run_id = ?
                """,
                (completed, failed, run_id),
            )
            connection.commit()
            print(
                f"[AI screening] {index}/{len(records)} "
                f"complete={completed} failed={failed}",
                flush=True,
            )
    fields = list(screening.SCREENING_FIELDS) + [
        "review_round",
        "discovery_query_ids",
    ]
    write_csv(
        output_path,
        _completed_ai_rows(connection, iteration),
        fields,
    )
    status = (
        "complete"
        if failed == 0 and completed == len(records)
        else (
            "complete_with_human_fallback"
            if completed == len(records)
            else "partial"
        )
    )
    connection.execute(
        """
        UPDATE ai_assistance_runs
        SET completed_items = ?, failed_items = ?, status = ?,
            completed_at = ?
        WHERE run_id = ?
        """,
        (completed, failed, status, utc_now(), run_id),
    )
    invalidate_stages(
        connection,
        (
            "literature_screened",
            "indicators_extracted",
            "dimensions_derived",
            "features_selected",
            "audit_complete",
        ),
        "AI discovery screening changed",
    )
    connection.commit()
    return {
        "run_id": run_id,
        "model": model,
        "model_digest": digest,
        "prompt_hash": prompt_hash,
        "records": len(records),
        "completed": completed,
        "failed": failed,
        "status": status,
        "output": str(output_path),
    }


def _active_term_rows(connection: sqlite3.Connection) -> List[sqlite3.Row]:
    return connection.execute(
        """
        SELECT term_id, verbatim_term, source_type, evidence_span,
               proposed_role
        FROM raw_terms
        WHERE status = 'active'
        ORDER BY term_id
        """
    ).fetchall()


def _derive_ai_term_codebook(
    terms: List[sqlite3.Row],
    model: str,
) -> Dict[str, Any]:
    compact_terms = [
        {
            "term_id": row["term_id"],
            "term": row["verbatim_term"],
            "source_type": row["source_type"],
            "proposed_role": row["proposed_role"],
        }
        for row in terms
    ]
    codebook = _chat_json(
        model,
        TERM_CODEBOOK_SYSTEM_PROMPT,
        json.dumps(compact_terms, ensure_ascii=False),
        TERM_CODEBOOK_SCHEMA,
        retries=3,
        num_predict=3000,
    )
    domains = codebook.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("AI term codebook contains no domains")
    seen: set[str] = set()
    cleaned: List[Dict[str, Any]] = []
    for item in domains:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        definition = str(item.get("definition") or "").strip()
        key = normalize_term(label)
        if not label or not definition or key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "label": label,
                "definition": definition,
                "example_terms": [
                    str(value).strip()
                    for value in item.get("example_terms", [])
                    if str(value).strip()
                ],
            }
        )
    if not cleaned:
        raise ValueError("AI term codebook has no valid unique domain")
    return {
        "domains": cleaned,
        "derivation_notes": str(
            codebook.get("derivation_notes") or ""
        ).strip(),
    }


def _code_one_term(
    term: sqlite3.Row,
    codebook: Mapping[str, Any],
    model: str,
) -> Dict[str, Any]:
    response = _chat_json(
        model,
        TERM_CODING_SYSTEM_PROMPT,
        json.dumps(
            {
                "codebook": codebook,
                "term": {
                    "term_id": term["term_id"],
                    "verbatim_term": term["verbatim_term"],
                    "source_type": term["source_type"],
                    "evidence_span": term["evidence_span"],
                    "proposed_role": term["proposed_role"],
                },
            },
            ensure_ascii=False,
        ),
        TERM_CODING_SCHEMA,
        retries=3,
    )
    decision = str(response.get("decision") or "").casefold()
    reason = str(response.get("reason") or "").strip()
    if decision not in {"include", "exclude"} or not reason:
        raise ValueError("AI term coding lacks decision or reason")
    if decision == "exclude":
        return {
            "decision": "exclude",
            "canonical_term": "",
            "term_family_label": "",
            "term_relation": "",
            "search_domain_label": "",
            "search_domain_definition": "",
            "query_family_label": "",
            "cross_domain": False,
            "reason": reason,
        }
    domains = {
        normalize_term(str(item["label"])): item
        for item in codebook["domains"]
    }
    labels = [
        str(value).strip()
        for value in response.get("domain_labels", [])
        if str(value).strip()
    ]
    if not labels or any(normalize_term(label) not in domains for label in labels):
        raise ValueError("AI term coding used an unknown domain label")
    definitions = {
        str(domains[normalize_term(label)]["definition"])
        for label in labels
    }
    canonical = str(response.get("canonical_term") or "").strip()
    family = str(response.get("term_family_label") or "").strip()
    relation = str(response.get("term_relation") or "").strip().casefold()
    query_family = str(response.get("query_family_label") or "").strip()
    if not all((canonical, family, relation, query_family)):
        raise ValueError("AI included term coding lacks normalization fields")
    return {
        "decision": "include",
        "canonical_term": canonical,
        "term_family_label": family,
        "term_relation": relation,
        "search_domain_label": "|".join(sorted(set(labels))),
        "search_domain_definition": " | ".join(sorted(definitions)),
        "query_family_label": query_family,
        "cross_domain": len(set(labels)) > 1,
        "reason": reason,
    }


def _upsert_ai_term_code(
    connection: sqlite3.Connection,
    term_id: str,
    value: Mapping[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO term_coding(
            term_id, coder_role, canonical_term, term_family_label,
            term_relation, search_domain_label,
            search_domain_definition, query_family_label,
            cross_domain, decision, reason, coded_at
        ) VALUES (?, 'AI', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(term_id, coder_role) DO UPDATE SET
            canonical_term = excluded.canonical_term,
            term_family_label = excluded.term_family_label,
            term_relation = excluded.term_relation,
            search_domain_label = excluded.search_domain_label,
            search_domain_definition = excluded.search_domain_definition,
            query_family_label = excluded.query_family_label,
            cross_domain = excluded.cross_domain,
            decision = excluded.decision,
            reason = excluded.reason,
            coded_at = excluded.coded_at
        """,
        (
            term_id,
            value["canonical_term"],
            value["term_family_label"],
            value["term_relation"],
            value["search_domain_label"],
            value["search_domain_definition"],
            value["query_family_label"],
            int(bool(value["cross_domain"])),
            value["decision"],
            value["reason"],
            utc_now(),
        ),
    )


def _ai_term_coding_rows(
    connection: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    return [
        {
            "term_id": row["term_id"],
            "verbatim_term": row["verbatim_term"],
            "source_type": row["source_type"],
            "coder_role": "AI",
            "canonical_term": row["canonical_term"],
            "term_family_label": row["term_family_label"],
            "term_relation": row["term_relation"],
            "search_domain_label": row["search_domain_label"],
            "search_domain_definition": row[
                "search_domain_definition"
            ],
            "query_family_label": row["query_family_label"],
            "cross_domain": bool(row["cross_domain"]),
            "decision": row["decision"],
            "reason": row["reason"],
        }
        for row in connection.execute(
            """
            SELECT r.term_id, r.verbatim_term, r.source_type, c.*
            FROM raw_terms r
            JOIN term_coding c USING(term_id)
            WHERE r.status = 'active' AND c.coder_role = 'AI'
            ORDER BY r.term_id
            """
        )
    ]


def ai_code_terms(
    connection: sqlite3.Connection,
    model: str,
    output_path: Path,
    codebook_path: Path,
    force: bool = False,
) -> Dict[str, Any]:
    """Derive an AI-only codebook and independently code every active term."""
    h2_rows = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM term_coding
            WHERE coder_role = 'H2'
            """
        ).fetchone()[0]
    )
    if force and h2_rows:
        raise RuntimeError(
            "Forced AI term replacement is forbidden after H2 "
            "adjudication; a reopened frame may add only uncoded new terms"
        )
    digest = _model_digest(model)
    terms = _active_term_rows(connection)
    if force:
        removed_rows = connection.execute(
            """
            SELECT COUNT(*) FROM term_coding
            WHERE coder_role = 'AI'
            """
        ).fetchone()[0]
        connection.execute(
            "DELETE FROM term_coding WHERE coder_role = 'AI'"
        )
        log_event(
            connection,
            "ai_term_coding_replaced",
            "collection",
            "AI",
            {
                "removed_rows": removed_rows,
                "reason": (
                    "Explicit --force after an AI prompt, input, or "
                    "source-provenance revision"
                ),
            },
        )
        connection.commit()
    input_hash = json_hash(
        {
            "terms": [
                {
                    "term_id": row["term_id"],
                    "term": row["verbatim_term"],
                    "source_type": row["source_type"],
                    "evidence_span": row["evidence_span"],
                }
                for row in terms
            ]
        }
    )
    codebook = _derive_ai_term_codebook(terms, model)
    codebook_payload = {
        "model": model,
        "model_digest": digest,
        "prompt_hash": sha256_bytes(
            TERM_CODEBOOK_SYSTEM_PROMPT.encode("utf-8")
        ),
        "input_hash": input_hash,
        **codebook,
    }
    write_json(codebook_path, codebook_payload)
    prompt_hash = sha256_bytes(
        (
            TERM_CODEBOOK_SYSTEM_PROMPT
            + "\n"
            + TERM_CODING_SYSTEM_PROMPT
        ).encode("utf-8")
    )
    run_id = "AIRUN_" + sha256_bytes(
        f"term_code|{model}|{prompt_hash}|{input_hash}".encode("utf-8")
    )[:16].upper()
    connection.execute(
        """
        INSERT INTO ai_assistance_runs(
            run_id, task, reviewer_role, model, model_digest,
            prompt_hash, input_hash, output_path, completed_items,
            failed_items, status, started_at, completed_at
        ) VALUES (?, 'term_coding', 'AI', ?, ?, ?, ?, ?, 0, 0,
                  'running', ?, '')
        ON CONFLICT(run_id) DO UPDATE SET
            output_path = excluded.output_path, status = 'running'
        """,
        (
            run_id,
            model,
            digest,
            prompt_hash,
            input_hash,
            str(output_path.resolve()),
            utc_now(),
        ),
    )
    connection.commit()
    completed = 0
    failed = 0
    for index, term in enumerate(terms, start=1):
        existing = connection.execute(
            """
            SELECT 1 FROM term_coding
            WHERE term_id = ? AND coder_role = 'AI'
            """,
            (term["term_id"],),
        ).fetchone()
        if existing is not None:
            completed += 1
            continue
        try:
            value = _code_one_term(term, codebook, model)
            _upsert_ai_term_code(
                connection,
                str(term["term_id"]),
                value,
            )
            completed += 1
        except Exception as error:
            failed += 1
            log_event(
                connection,
                "ai_term_coding_error",
                "term",
                str(term["term_id"]),
                {"error_type": type(error).__name__},
            )
        if index % 10 == 0:
            connection.execute(
                """
                UPDATE ai_assistance_runs
                SET completed_items = ?, failed_items = ?
                WHERE run_id = ?
                """,
                (completed, failed, run_id),
            )
            connection.commit()
            print(
                f"[AI term coding] {index}/{len(terms)} "
                f"complete={completed} failed={failed}",
                flush=True,
            )
    write_csv(
        output_path,
        _ai_term_coding_rows(connection),
        coding.TERM_CODING_FIELDS,
    )
    status = (
        "complete"
        if failed == 0 and completed == len(terms)
        else "partial"
    )
    connection.execute(
        """
        UPDATE ai_assistance_runs
        SET completed_items = ?, failed_items = ?, status = ?,
            completed_at = ?
        WHERE run_id = ?
        """,
        (completed, failed, status, utc_now(), run_id),
    )
    invalidate_stages(
        connection,
        coding.SEARCH_FRAME_DOWNSTREAM_STAGES,
        "AI term coding changed",
    )
    connection.commit()
    return {
        "run_id": run_id,
        "model": model,
        "model_digest": digest,
        "active_terms": len(terms),
        "codebook_domains": len(codebook["domains"]),
        "completed": completed,
        "failed": failed,
        "status": status,
        "output": str(output_path),
        "codebook": str(codebook_path),
    }


def _code_one_dimension(
    family: sqlite3.Row,
    model: str,
) -> Dict[str, str]:
    evidence = {
        "feature_id": family["feature_id"],
        "canonical_name_en": family["canonical_name_en"],
        "aliases": json.loads(family["alias_names_json"]),
        "formula": family["formula"],
        "required_data": json.loads(family["required_data_json"]),
        "maximum_information_time": family["maximum_information_time"],
        "scope_role": family["scope_role"],
        "research_groups": json.loads(family["research_groups_json"]),
        "mention_ids": json.loads(family["mention_ids_json"]),
    }
    response = _chat_json(
        model,
        DIMENSION_CODING_SYSTEM_PROMPT,
        json.dumps(evidence, ensure_ascii=False),
        DIMENSION_CODING_SCHEMA,
        retries=3,
        num_predict=800,
    )
    decision = str(response.get("decision") or "").strip().casefold()
    reason = str(response.get("reason") or "").strip()
    if decision not in {"include", "exclude"} or not reason:
        raise ValueError("AI dimension coding lacks decision or reason")
    if decision == "exclude":
        return {
            "decision": decision,
            "dimension_label": "",
            "dimension_definition": "",
            "construct_role": "",
            "information_source": "",
            "t0_boundary": "",
            "bias_risk": "",
            "reason": reason,
        }
    values = {
        field: str(response.get(field) or "").strip()
        for field in (
            "dimension_label",
            "dimension_definition",
            "construct_role",
            "information_source",
            "t0_boundary",
            "bias_risk",
        )
    }
    if not all(values.values()):
        raise ValueError("AI included dimension coding is incomplete")
    values["construct_role"] = values["construct_role"].casefold()
    if values["construct_role"] not in indicators.CONSTRUCT_ROLES:
        raise ValueError("AI dimension coding used an invalid construct role")
    if re.match(
        r"^D(?:0?[1-9]|1[0-2])(?:_|\b)",
        values["dimension_label"],
    ):
        raise ValueError("AI dimension coding reused a legacy D01-D12 label")
    return {"decision": decision, "reason": reason, **values}


def _upsert_ai_dimension_code(
    connection: sqlite3.Connection,
    feature_id: str,
    value: Mapping[str, str],
) -> None:
    connection.execute(
        """
        INSERT INTO dimension_coding(
            feature_id, coder_role, dimension_label,
            dimension_definition, construct_role, information_source,
            t0_boundary, bias_risk, decision, reason, coded_at
        ) VALUES (?, 'AI', ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            value["dimension_label"],
            value["dimension_definition"],
            value["construct_role"],
            value["information_source"],
            value["t0_boundary"],
            value["bias_risk"],
            value["decision"],
            value["reason"],
            utc_now(),
        ),
    )


def ai_code_dimensions(
    connection: sqlite3.Connection,
    model: str,
    output_path: Path,
    force: bool = False,
) -> Dict[str, Any]:
    """Independently code canonical indicator families as the AI reviewer."""
    require_complete(connection, ["indicators_extracted"])
    h2_rows = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM dimension_coding
            WHERE coder_role = 'H2'
            """
        ).fetchone()[0]
    )
    if force and h2_rows:
        raise RuntimeError(
            "Forced AI dimension replacement is forbidden after H2 "
            "adjudication"
        )
    digest = _model_digest(model)
    families = connection.execute(
        """
        SELECT * FROM indicator_families
        WHERE status = 'candidate' ORDER BY feature_id
        """
    ).fetchall()
    if force:
        connection.execute(
            "DELETE FROM dimension_coding WHERE coder_role = 'AI'"
        )
        connection.commit()
    input_hash = json_hash(
        {
            "families": [
                {
                    "feature_id": row["feature_id"],
                    "canonical_name_en": row["canonical_name_en"],
                    "formula": row["formula"],
                    "required_data_json": row["required_data_json"],
                    "maximum_information_time": row[
                        "maximum_information_time"
                    ],
                    "scope_role": row["scope_role"],
                    "research_groups_json": row["research_groups_json"],
                    "mention_ids_json": row["mention_ids_json"],
                }
                for row in families
            ]
        }
    )
    prompt_hash = sha256_bytes(
        DIMENSION_CODING_SYSTEM_PROMPT.encode("utf-8")
    )
    run_id = "AIRUN_" + sha256_bytes(
        f"dimension_code|{model}|{prompt_hash}|{input_hash}".encode("utf-8")
    )[:16].upper()
    connection.execute(
        """
        INSERT INTO ai_assistance_runs(
            run_id, task, reviewer_role, model, model_digest,
            prompt_hash, input_hash, output_path, completed_items,
            failed_items, status, started_at, completed_at
        ) VALUES (?, 'dimension_coding', 'AI', ?, ?, ?, ?, ?, 0, 0,
                  'running', ?, '')
        ON CONFLICT(run_id) DO UPDATE SET
            output_path = excluded.output_path, status = 'running'
        """,
        (
            run_id,
            model,
            digest,
            prompt_hash,
            input_hash,
            str(output_path.resolve()),
            utc_now(),
        ),
    )
    connection.commit()
    completed = 0
    failed = 0
    for index, family in enumerate(families, start=1):
        existing = connection.execute(
            """
            SELECT 1 FROM dimension_coding
            WHERE feature_id = ? AND coder_role = 'AI'
            """,
            (family["feature_id"],),
        ).fetchone()
        if existing is not None:
            completed += 1
            continue
        try:
            value = _code_one_dimension(family, model)
            _upsert_ai_dimension_code(
                connection,
                str(family["feature_id"]),
                value,
            )
            completed += 1
        except Exception as error:
            failed += 1
            log_event(
                connection,
                "ai_dimension_coding_error",
                "feature",
                str(family["feature_id"]),
                {"error_type": type(error).__name__},
            )
        if index % 10 == 0:
            connection.execute(
                """
                UPDATE ai_assistance_runs
                SET completed_items = ?, failed_items = ?
                WHERE run_id = ?
                """,
                (completed, failed, run_id),
            )
            connection.commit()
    export_rows: List[Dict[str, Any]] = []
    for family in families:
        code = connection.execute(
            """
            SELECT * FROM dimension_coding
            WHERE feature_id = ? AND coder_role = 'AI'
            """,
            (family["feature_id"],),
        ).fetchone()
        if code is None:
            continue
        export_rows.append(
            {
                "feature_id": family["feature_id"],
                "canonical_name_en": family["canonical_name_en"],
                "coder_role": "AI",
                "dimension_label": code["dimension_label"],
                "dimension_definition": code["dimension_definition"],
                "construct_role": code["construct_role"],
                "information_source": code["information_source"],
                "t0_boundary": code["t0_boundary"],
                "bias_risk": code["bias_risk"],
                "decision": code["decision"],
                "reason": code["reason"],
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
        )
    write_csv(
        output_path,
        export_rows,
        indicators.DIMENSION_EVIDENCE_FIELDS,
    )
    status = (
        "complete"
        if completed == len(families) and failed == 0
        else "partial"
    )
    connection.execute(
        """
        UPDATE ai_assistance_runs
        SET completed_items = ?, failed_items = ?, status = ?,
            completed_at = ?
        WHERE run_id = ?
        """,
        (completed, failed, status, utc_now(), run_id),
    )
    invalidate_stages(
        connection,
        ("dimensions_derived", "features_selected", "audit_complete"),
        "AI dimension coding changed",
    )
    connection.commit()
    return {
        "run_id": run_id,
        "model": model,
        "model_digest": digest,
        "families": len(families),
        "completed": completed,
        "failed": failed,
        "status": status,
        "output": str(output_path),
    }
