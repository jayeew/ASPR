from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import coding
import local_ai
from common import (
    OUTPUT_DIR,
    json_hash,
    read_csv,
    sha256_bytes,
    write_csv,
    write_json,
)


DRAFT_DIR = OUTPUT_DIR / "unreviewed_automated_h2_drafts_20260729"
RECONCILIATION_PROMPT = """
You are generating an UNREVIEWED H2 terminology-reconciliation draft for a
systematic evidence map of individual-paper innovation and publication-time
potential scholarly impact. Review all source-linked English terms and both
role-separated primary code streams. Derive a concise, non-redundant search
concept-domain codebook without targeting any number. The AI labels may be
too broad and the H1 labels may be too granular and came from a provisional
28-label dictionary; neither list is authoritative. Do not copy either
primary codebook wholesale.

Merge only labels that measure the same theoretical object and explanatory
role. Split substantive content/novelty, T0 substantive potential,
opportunity/attention, context/control, and future validation outcomes when
their roles differ. Search domains are terminology families, not final model
dimensions. Labels and definitions must be English and source-grounded.
Every domain must name a specific theoretical object, reconcile named
primary labels, and cite at least two supplied supporting term_ids. Generic
catch-all domains such as "candidate only", "specialized/contextual", or
"technical metrics" are forbidden. Preserve distinct evidence roles present
in the source terms instead of assigning every domain the same role.
""".strip()
TERM_DRAFT_PROMPT = """
Code each supplied English evidence term against the reconciled codebook.
Do not target a domain or query count. Include useful paper-level construct,
measure, indicator, predictor, opportunity/context, validation, and outcome
search terms. Future outcomes may support validation retrieval but are never
T0 predictors. Preserve true synonyms and parameter variants in one term
family, but do not collapse different constructs. Use only exact codebook
domain labels. Return one result for every term_id.
""".strip()
CODEBOOK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "domains": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "definition": {"type": "string"},
                    "evidence_role": {
                        "type": "string",
                        "enum": [
                            "substantive_innovation",
                            "t0_substantive_potential",
                            "opportunity_attention",
                            "context_control",
                            "validation_outcome",
                            "mixed_search_role",
                        ],
                    },
                    "reconciliation_reason": {"type": "string"},
                    "primary_labels_reconciled": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "support_term_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "label",
                    "definition",
                    "evidence_role",
                    "reconciliation_reason",
                    "primary_labels_reconciled",
                    "support_term_ids",
                ],
            },
        },
        "derivation_notes": {"type": "string"},
    },
    "required": ["domains", "derivation_notes"],
}
TERM_BATCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term_id": {"type": "string"},
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
                    "term_id",
                    "decision",
                    "canonical_term",
                    "term_family_label",
                    "term_relation",
                    "domain_labels",
                    "query_family_label",
                    "cross_domain",
                    "reason",
                ],
            },
        }
    },
    "required": ["results"],
}


def _safe_output(path: Path) -> Path:
    """Restrict every draft to the import-blocked quarantine directory."""
    resolved = path.resolve()
    draft_root = DRAFT_DIR.resolve()
    if not resolved.is_relative_to(draft_root):
        raise ValueError(
            f"Draft output must be inside the quarantine: {draft_root}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _append_draft_fields(fields: Sequence[str]) -> List[str]:
    additions = (
        "draft_method",
        "draft_model",
        "draft_prompt_hash",
        "human_review_status",
    )
    return [*fields, *(field for field in additions if field not in fields)]


def draft_screening(
    input_path: Path,
    output_path: Path,
    model: str,
) -> Dict[str, Any]:
    """Prefill H2 with the reviewed H1 decision, without importing it."""
    rows = read_csv(input_path)
    if not rows:
        raise ValueError("H2 screening worksheet is empty")
    fields = _append_draft_fields(tuple(rows[0]))
    uncertain = 0
    model_resolved = 0
    model_failures = 0
    prompt_hash = sha256_bytes(
        local_ai.SCREENING_SYSTEM_PROMPT.encode("utf-8")
    )
    for row in rows:
        row.update(
            {
                "reviewer_role": "H2",
                "language_judgment": row["h1_language_judgment"],
                "language_evidence": row["h1_evidence_span"],
                "decision": row["h1_decision"],
                "exclusion_reason": row["h1_exclusion_reason"],
                "evidence_span": row["h1_evidence_span"],
                "notes": (
                    "UNREVIEWED AUTOMATED H2 DRAFT seeded from the "
                    "human-reviewed H1 decision. Human H2 must inspect the "
                    "source and both primary codes before adoption. "
                    + row["h1_notes"]
                ).strip(),
                "draft_method": "seed_from_human_reviewed_H1",
                "draft_model": "",
                "draft_prompt_hash": "",
                "human_review_status": "unreviewed",
            }
        )
        needs_model = (
            row["decision"] == "uncertain"
            or row["language_judgment"] == "uncertain"
        )
        if needs_model:
            try:
                suggestion = local_ai._screen_one(
                    {
                        "language": row["openalex_language"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                    },
                    model,
                )
                row.update(
                    {
                        "language_judgment":
                            suggestion["language_judgment"],
                        "language_evidence":
                            suggestion["language_evidence"],
                        "decision": suggestion["decision"],
                        "exclusion_reason":
                            suggestion["exclusion_reason"],
                        "evidence_span": suggestion["evidence_span"],
                        "notes": (
                            "UNREVIEWED AUTOMATED H2 DRAFT. A fresh model "
                            "suggestion resolved an H1-uncertain row; human "
                            "H2 must inspect the source and both primary "
                            "codes before adoption. "
                            + suggestion["rationale"]
                        ).strip(),
                        "draft_method":
                            "fresh_model_suggestion_for_H1_uncertain",
                        "draft_model": model,
                        "draft_prompt_hash": prompt_hash,
                    }
                )
                model_resolved += int(
                    row["decision"] != "uncertain"
                    and row["language_judgment"] != "uncertain"
                )
            except Exception:
                model_failures += 1
        uncertain += int(
            row["decision"] == "uncertain"
            or row["language_judgment"] == "uncertain"
        )
    output = _safe_output(output_path)
    write_csv(output, rows, fields)
    return {
        "rows": len(rows),
        "uncertain_requiring_manual_resolution": uncertain,
        "model_resolved": model_resolved,
        "model_failures": model_failures,
        "output": str(output),
        "status": "unreviewed_not_importable",
    }


def _compact_term_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "term_id": row["term_id"],
            "term": row["verbatim_term"],
            "source_type": row["source_type"],
            "proposed_role": row["proposed_role"],
            "source_evidence_span": row["source_evidence_span"],
            "ai_decision": row["ai_decision"],
            "ai_domain": row["ai_search_domain_label"],
            "ai_family": row["ai_query_family_label"],
            "h1_decision": row["h1_decision"],
            "h1_domain": row["h1_search_domain_label"],
            "h1_family": row["h1_query_family_label"],
        }
        for row in rows
    ]


def _primary_codebooks(
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Deduplicate both primary codebooks without treating either as final."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for role in ("ai", "h1"):
        labels: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            raw_labels = row[f"{role}_search_domain_label"]
            raw_definitions = row[f"{role}_search_domain_definition"]
            definitions = raw_definitions.split(" | ")
            for index, label in enumerate(raw_labels.split("|")):
                label = label.strip()
                if not label:
                    continue
                item = labels.setdefault(
                    label,
                    {
                        "label": label,
                        "definition": (
                            definitions[index].strip()
                            if index < len(definitions)
                            else raw_definitions
                        ),
                        "term_count": 0,
                    },
                )
                item["term_count"] += 1
        result[role] = sorted(labels.values(), key=lambda item: item["label"])
    return result


def _clean_codebook(
    value: Mapping[str, Any],
    valid_term_ids: set[str],
    primary_labels: Mapping[str, set[str]],
) -> Dict[str, Any]:
    domains: List[Dict[str, Any]] = []
    seen: set[str] = set()
    forbidden = (
        "candidate only",
        "specialized",
        "contextual metrics",
        "technical & computational metrics",
        "technical and computational metrics",
    )
    for item in value.get("domains", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        key = label.casefold()
        if (
            not label
            or key in seen
            or any(token in key for token in forbidden)
        ):
            continue
        seen.add(key)
        support_ids = sorted(
            {
                str(term_id).strip()
                for term_id in item.get("support_term_ids", [])
                if str(term_id).strip() in valid_term_ids
            }
        )
        primary = sorted(
            {
                str(primary_label).strip()
                for primary_label in item.get(
                    "primary_labels_reconciled", []
                )
                if str(primary_label).strip()
            }
        )
        domain = {
            "label": label,
            "definition": str(item.get("definition") or "").strip(),
            "evidence_role": str(item.get("evidence_role") or "").strip(),
            "reconciliation_reason": str(
                item.get("reconciliation_reason") or ""
            ).strip(),
            "primary_labels_reconciled": primary,
            "support_term_ids": support_ids,
        }
        if all(domain.values()) and len(support_ids) >= 2:
            domains.append(domain)
    if not domains:
        raise ValueError("Reconciled codebook has incomplete domains")
    normalized = {domain["label"].casefold() for domain in domains}
    for role in ("ai", "h1"):
        primary_normalized = {
            label.casefold() for label in primary_labels[role]
        }
        union = normalized | primary_normalized
        overlap = len(normalized & primary_normalized) / max(len(union), 1)
        if overlap >= 0.8:
            raise ValueError(
                f"Reconciled codebook substantially copied {role.upper()}"
            )
    roles = {domain["evidence_role"] for domain in domains}
    if len(roles) < 3:
        raise ValueError("Reconciled codebook collapsed evidence roles")
    return {
        "domains": domains,
        "derivation_notes": str(
            value.get("derivation_notes") or ""
        ).strip(),
    }


def _derive_reconciled_codebook(
    rows: Sequence[Mapping[str, str]],
    compact: Sequence[Mapping[str, str]],
    model: str,
) -> Dict[str, Any]:
    primary = _primary_codebooks(rows)
    valid_ids = {str(item["term_id"]) for item in compact}
    primary_label_sets = {
        role: {str(item["label"]) for item in items}
        for role, items in primary.items()
    }
    last_error = ""
    for attempt in range(1, 4):
        payload = {
            "attempt": attempt,
            "previous_validation_error": last_error,
            "primary_codebooks": primary,
            "source_terms": list(compact),
        }
        raw = local_ai._chat_json(
            model,
            RECONCILIATION_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            CODEBOOK_SCHEMA,
            retries=3,
            num_predict=7000,
        )
        attempt_path = DRAFT_DIR / (
            f"term_codebook_reconciliation_attempt_{attempt:02d}.json"
        )
        attempt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            attempt_path,
            {
                "attempt": attempt,
                "previous_validation_error": last_error,
                "raw_model_output": raw,
            },
        )
        try:
            return _clean_codebook(
                raw,
                valid_ids,
                primary_label_sets,
            )
        except ValueError as error:
            last_error = str(error)
    raise RuntimeError(
        "Reconciled codebook failed source/role validation: " + last_error
    )


def _term_batch_result(
    batch: Sequence[Mapping[str, str]],
    codebook: Mapping[str, Any],
    model: str,
) -> Dict[str, Mapping[str, Any]]:
    payload = {
        "codebook": codebook["domains"],
        "terms": list(batch),
    }
    value = local_ai._chat_json(
        model,
        TERM_DRAFT_PROMPT,
        json.dumps(payload, ensure_ascii=False),
        TERM_BATCH_SCHEMA,
        retries=3,
        num_predict=4000,
    )
    results = value.get("results")
    if not isinstance(results, list):
        raise ValueError("Term draft lacks results")
    by_id = {
        str(item.get("term_id") or ""): item
        for item in results
        if isinstance(item, dict)
    }
    expected = {str(item["term_id"]) for item in batch}
    if set(by_id) != expected:
        raise ValueError("Term draft IDs do not match the requested batch")
    return by_id


def _fallback_term(row: Mapping[str, str]) -> Dict[str, Any]:
    """Preserve H1 as a visible fallback that remains unreviewed."""
    domains = [
        value.strip()
        for value in row["h1_search_domain_label"].split("|")
        if value.strip()
    ]
    return {
        "decision": row["h1_decision"],
        "canonical_term": row["h1_canonical_term"],
        "term_family_label": row["h1_term_family_label"],
        "term_relation": row["h1_term_relation"] or "canonical",
        "domain_labels": domains,
        "query_family_label": row["h1_query_family_label"],
        "cross_domain": row["h1_cross_domain"].casefold() == "true",
        "reason": "Model draft failed; H1 copied as an unreviewed fallback.",
    }


def _allowed_with_h1(
    allowed_domains: Mapping[str, str],
    row: Mapping[str, str],
) -> Dict[str, str]:
    """Add provisional H1 definitions only for a visible fallback row."""
    expanded = dict(allowed_domains)
    labels = [
        value.strip()
        for value in row["h1_search_domain_label"].split("|")
        if value.strip()
    ]
    definitions = [
        value.strip()
        for value in row["h1_search_domain_definition"].split(" | ")
    ]
    for index, label in enumerate(labels):
        definition = (
            definitions[index]
            if index < len(definitions) and definitions[index]
            else row["h1_search_domain_definition"]
        )
        expanded.setdefault(label, definition)
    return expanded


def _apply_term_result(
    row: Dict[str, str],
    result: Mapping[str, Any],
    allowed_domains: Mapping[str, str],
) -> None:
    decision = str(result.get("decision") or "").strip().casefold()
    reason = str(result.get("reason") or "").strip()
    if decision not in {"include", "exclude"} or not reason:
        raise ValueError(f"Invalid draft decision for {row['term_id']}")
    if decision == "exclude":
        row.update(
            {
                "decision": "exclude",
                "canonical_term": "",
                "term_family_label": "",
                "term_relation": "",
                "search_domain_label": "",
                "search_domain_definition": "",
                "query_family_label": "",
                "cross_domain": "false",
                "reason": reason,
            }
        )
        return
    labels = [str(value).strip() for value in result["domain_labels"]]
    if not labels or any(label not in allowed_domains for label in labels):
        raise ValueError(f"Unknown reconciled domain for {row['term_id']}")
    relation = str(result.get("term_relation") or "").strip().casefold()
    if relation not in coding.TERM_RELATIONS:
        raise ValueError(f"Invalid term relation for {row['term_id']}")
    row.update(
        {
            "decision": "include",
            "canonical_term": str(result["canonical_term"]).strip(),
            "term_family_label": str(result["term_family_label"]).strip(),
            "term_relation": relation,
            "search_domain_label": "|".join(labels),
            "search_domain_definition": " | ".join(
                allowed_domains[label] for label in labels
            ),
            "query_family_label": str(
                result["query_family_label"]
            ).strip(),
            "cross_domain": str(len(labels) > 1).casefold(),
            "reason": reason,
        }
    )


def draft_terms(
    input_path: Path,
    output_path: Path,
    model: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Generate an import-blocked reconciliation draft without a count goal."""
    rows = read_csv(input_path)
    compact = _compact_term_rows(rows)
    codebook = _derive_reconciled_codebook(rows, compact, model)
    codebook_payload = {
        "status": "unreviewed_not_importable",
        "model": model,
        "model_digest": local_ai._model_digest(model),
        "prompt_hash": sha256_bytes(
            RECONCILIATION_PROMPT.encode("utf-8")
        ),
        "input_hash": json_hash({"terms": compact}),
        **codebook,
    }
    output = _safe_output(output_path)
    codebook_path = output.with_suffix(".codebook.json")
    write_json(codebook_path, codebook_payload)
    allowed = {
        item["label"]: item["definition"] for item in codebook["domains"]
    }
    failures = 0
    for offset in range(0, len(rows), batch_size):
        source_batch = compact[offset : offset + batch_size]
        try:
            by_id = _term_batch_result(source_batch, codebook, model)
        except Exception:
            by_id = {}
            failures += len(source_batch)
        for index, source in enumerate(source_batch, start=offset):
            result = by_id.get(source["term_id"])
            row_allowed = allowed
            if result is None:
                result = _fallback_term(rows[index])
                row_allowed = _allowed_with_h1(allowed, rows[index])
            try:
                _apply_term_result(rows[index], result, row_allowed)
            except Exception:
                failures += int(bool(by_id))
                _apply_term_result(
                    rows[index],
                    _fallback_term(rows[index]),
                    _allowed_with_h1(allowed, rows[index]),
                )
        print(
            f"[H2 term draft] {min(offset + batch_size, len(rows))}/"
            f"{len(rows)} fallback_rows={failures}",
            flush=True,
        )
    fields = _append_draft_fields(tuple(rows[0]))
    prompt_hash = sha256_bytes(TERM_DRAFT_PROMPT.encode("utf-8"))
    for row in rows:
        row.update(
            {
                "coder_role": "H2",
                "draft_method": "model_reconcile_AI_H1_without_count_target",
                "draft_model": model,
                "draft_prompt_hash": prompt_hash,
                "human_review_status": "unreviewed",
            }
        )
    write_csv(output, rows, fields)
    return {
        "rows": len(rows),
        "draft_domains": len(codebook["domains"]),
        "fallback_rows": failures,
        "output": str(output),
        "codebook": str(codebook_path),
        "status": "unreviewed_not_importable",
    }


def draft_terms_provisional(
    input_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Seed H2 from H1 and expose domain support for manual merge/split."""
    rows = read_csv(input_path)
    if not rows:
        raise ValueError("H2 term worksheet is empty")
    domains: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row.update(
            {
                "coder_role": "H2",
                "canonical_term": row["h1_canonical_term"],
                "term_family_label": row["h1_term_family_label"],
                "term_relation": row["h1_term_relation"],
                "search_domain_label": row["h1_search_domain_label"],
                "search_domain_definition":
                    row["h1_search_domain_definition"],
                "query_family_label": row["h1_query_family_label"],
                "cross_domain": row["h1_cross_domain"].casefold(),
                "decision": row["h1_decision"],
                "reason": (
                    "UNREVIEWED AUTOMATED H2 DRAFT seeded from the "
                    "human-reviewed H1 proposal. H2 must review the domain "
                    "support summary and explicitly merge, split, relabel, "
                    "or exclude before adoption."
                ),
                "draft_method": "seed_from_provisional_H1_term_code",
                "draft_model": "",
                "draft_prompt_hash": "",
                "human_review_status": "unreviewed",
            }
        )
        labels = [
            value.strip()
            for value in row["h1_search_domain_label"].split("|")
            if value.strip()
        ]
        for label in labels:
            item = domains.setdefault(
                label,
                {
                    "h1_domain_label": label,
                    "h1_domain_definition":
                        row["h1_search_domain_definition"],
                    "term_ids": set(),
                    "verbatim_terms": set(),
                    "source_ids": set(),
                    "source_types": set(),
                    "ai_domain_labels": set(),
                    "direct_authorizing_term_ids": set(),
                },
            )
            item["term_ids"].add(row["term_id"])
            item["verbatim_terms"].add(row["verbatim_term"])
            item["source_ids"].add(row["source_id"])
            item["source_types"].add(row["source_type"])
            item["ai_domain_labels"].update(
                value.strip()
                for value in row["ai_search_domain_label"].split("|")
                if value.strip()
            )
            if row["source_type"] not in {
                "development_seed_hint",
                "pilot_v2_indicator",
            }:
                item["direct_authorizing_term_ids"].add(row["term_id"])
    output = _safe_output(output_path)
    fields = _append_draft_fields(tuple(rows[0]))
    write_csv(output, rows, fields)
    summary_rows = []
    for label, item in sorted(domains.items()):
        summary_rows.append(
            {
                "h1_domain_label": label,
                "h1_domain_definition": item["h1_domain_definition"],
                "term_count": len(item["term_ids"]),
                "direct_authorizing_term_count": len(
                    item["direct_authorizing_term_ids"]
                ),
                "term_ids": ";".join(sorted(item["term_ids"])),
                "verbatim_terms": " | ".join(
                    sorted(item["verbatim_terms"])
                ),
                "source_ids": ";".join(sorted(item["source_ids"])),
                "source_types": ";".join(sorted(item["source_types"])),
                "ai_domain_labels": " | ".join(
                    sorted(item["ai_domain_labels"])
                ),
                "h2_action": "",
                "h2_final_domain_label": "",
                "h2_final_domain_definition": "",
                "h2_merge_split_reason": "",
                "human_review_status": "unreviewed",
            }
        )
    summary_path = output.with_name(
        output.stem + "_DOMAIN_RECONCILIATION.csv"
    )
    write_csv(
        summary_path,
        summary_rows,
        (
            "h1_domain_label",
            "h1_domain_definition",
            "term_count",
            "direct_authorizing_term_count",
            "term_ids",
            "verbatim_terms",
            "source_ids",
            "source_types",
            "ai_domain_labels",
            "h2_action",
            "h2_final_domain_label",
            "h2_final_domain_definition",
            "h2_merge_split_reason",
            "human_review_status",
        ),
    )
    return {
        "rows": len(rows),
        "provisional_h1_domains": len(summary_rows),
        "direct_authorizing_terms": sum(
            row["direct_authorizing_term_count"] for row in summary_rows
        ),
        "output": str(output),
        "domain_reconciliation_summary": str(summary_path),
        "status": "unreviewed_not_importable",
    }


def _crossref_suggestion(row: Mapping[str, str]) -> tuple[str, str]:
    reason = row["conflict_reason"]
    similarity = float(row["title_match"] or 0)
    if reason == "crossref_doi_not_found":
        return (
            "manual_bibliographic_resolution",
            "DRAFT: Crossref has no DOI endpoint. Verify the DOI against the "
            "publisher record before retaining the OpenAlex metadata.",
        )
    if "doi" in reason or ("title" in reason and similarity < 0.85):
        return (
            "exclude_mapping_error",
            "DRAFT: DOI/title identity is materially inconsistent between "
            "OpenAlex and Crossref; verify the publisher record.",
        )
    return (
        "manual_bibliographic_resolution",
        "DRAFT: DOI identity is plausible but type/title metadata differ; "
        "verify the publisher record and select the authoritative metadata.",
    )


def draft_crossref(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """Add non-final bibliographic suggestions for human verification."""
    rows = read_csv(input_path)
    fields = _append_draft_fields(tuple(rows[0]))
    counts: Dict[str, int] = {}
    for row in rows:
        resolution, notes = _crossref_suggestion(row)
        row.update(
            {
                "reviewer_role": "H2",
                "resolution": resolution,
                "resolution_notes": notes,
                "draft_method": "deterministic_conflict_triage",
                "draft_model": "",
                "draft_prompt_hash": "",
                "human_review_status": "unreviewed",
            }
        )
        counts[resolution] = counts.get(resolution, 0) + 1
    output = _safe_output(output_path)
    write_csv(output, rows, fields)
    return {
        "rows": len(rows),
        "suggestions": counts,
        "output": str(output),
        "status": "unreviewed_not_importable",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create import-blocked H2 assistance drafts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("screening", "crossref"):
        child = subparsers.add_parser(command)
        child.add_argument("--input", required=True)
        child.add_argument("--output", required=True)
        if command == "screening":
            child.add_argument("--model", default="qwen3:8b")
    terms = subparsers.add_parser("terms")
    terms.add_argument("--input", required=True)
    terms.add_argument("--output", required=True)
    terms.add_argument("--model", default="qwen3:8b")
    terms.add_argument("--batch-size", type=int, default=8)
    provisional = subparsers.add_parser("terms-provisional")
    provisional.add_argument("--input", required=True)
    provisional.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    if args.command == "screening":
        result = draft_screening(input_path, output_path, args.model)
    elif args.command == "terms":
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least one")
        result = draft_terms(
            input_path,
            output_path,
            args.model,
            args.batch_size,
        )
    elif args.command == "terms-provisional":
        result = draft_terms_provisional(input_path, output_path)
    else:
        result = draft_crossref(input_path, output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
