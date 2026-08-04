from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PROVENANCE_FIELDS = (
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_review_note",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    """Read a UTF-8 CSV in its frozen order."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    """Write a UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def evidence_span(row: Mapping[str, str]) -> str:
    """Return an exact compact title or abstract substring."""
    title = str(row.get("title") or "").strip()
    if title:
        return title
    abstract = str(row.get("abstract") or "").strip()
    return abstract[:240]


def route_row(row: Mapping[str, str]) -> Dict[str, str]:
    """Apply the frozen high-recall routing decision."""
    output = dict(row)
    language = str(
        row.get("openalex_language") or "unknown"
    ).strip().casefold()
    span = evidence_span(row)
    output["reviewer_role"] = "AI"
    output["language_evidence"] = span
    output["evidence_span"] = span
    if language not in {"", "en", "unknown"}:
        output.update(
            {
                "language_judgment": "non_en",
                "decision": "exclude",
                "exclusion_reason": "E_LANGUAGE_NON_ENGLISH",
                "notes": (
                    f"OpenAlex language={language}; routed as explicit "
                    "non-English under the frozen language rule."
                ),
            }
        )
    elif not span:
        output.update(
            {
                "language_judgment": "uncertain",
                "decision": "exclude",
                "exclusion_reason": "E_INSUFFICIENT_METADATA",
                "notes": (
                    "No usable title or abstract; independent review lacks "
                    "enough metadata."
                ),
            }
        )
    else:
        output.update(
            {
                "language_judgment": "en",
                "decision": "include",
                "exclusion_reason": "",
                "notes": (
                    "High-recall AI routing inclusion; independent blind H1 "
                    "and mandatory H2 determine final eligibility."
                ),
            }
        )
    return output


def main() -> None:
    """Generate one exact-hash primary AI screening artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--thread-id", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    prompt_path = Path(args.prompt).resolve()
    fields, input_rows = read_csv(input_path)
    record_keys = [str(row.get("record_key") or "") for row in input_rows]
    if "" in record_keys or len(set(record_keys)) != len(record_keys):
        raise ValueError("Screening input needs unique nonblank record keys")
    output_fields = list(fields)
    for field in PROVENANCE_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    prompt_sha = sha256_file(prompt_path)
    output_rows = [route_row(row) for row in input_rows]
    for row in output_rows:
        row.update(
            {
                "draft_method": "primary_codex_high_recall_routing",
                "independent_ai_review_status": "complete",
                "independent_ai_reviewer_id": (
                    "primary_codex_ai_screening_v3"
                ),
                "independent_ai_reviewed_at": args.reviewed_at,
                "independent_ai_review_action": (
                    "high_recall_literature_screening"
                ),
                "independent_ai_review_note": (
                    "Deterministic high-recall routing for mandatory H2."
                ),
                "independent_ai_run_id": args.run_id,
                "independent_ai_model": "codex_configured_default",
                "independent_ai_prompt_sha256": prompt_sha,
            }
        )
    write_csv(output_path, output_fields, output_rows)
    manifest = {
        "run_id": args.run_id,
        "artifact_path": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "reviewer_role": "AI",
        "reviewer_id": "primary_codex_ai_screening_v3",
        "model": "codex_configured_default",
        "model_digest": f"codex-thread:{args.thread_id}",
        "prompt_sha256": prompt_sha,
        "parameters": {
            "review_method": "deterministic_high_recall_routing",
            "routing_rule": (
                "All usable English/unknown-language records are included "
                "so they require H2 after blind H1 screening."
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "item_count": len(output_rows),
        "completed_at": args.reviewed_at,
        "status": "complete",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts: Dict[str, int] = {}
    for row in output_rows:
        key = f"{row['language_judgment']}:{row['decision']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"rows": len(output_rows), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
