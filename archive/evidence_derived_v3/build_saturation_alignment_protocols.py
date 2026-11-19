from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from common import ROOT, read_json, sha256_file


INDICATOR_BASE_PROTOCOL = (
    ROOT / "INDEPENDENT_CODEX_INDICATOR_ADJUDICATION_PROTOCOL_V3.json"
)
TERM_BASE_BRIEF = ROOT / "INDEPENDENT_CODEX_TERM_CODING_BRIEF_V3.md"
REGISTRATION_MANIFEST_FIELDS = [
    "run_id",
    "artifact_path",
    "artifact_sha256",
    "input_path",
    "input_sha256",
    "reviewer_role",
    "reviewer_id",
    "model",
    "model_digest",
    "prompt_sha256",
    "parameters",
    "item_count",
    "completed_at",
    "status",
]


def _resolved_artifact(
    manifest_path: Path,
    value: Any,
) -> Path:
    """Resolve an artifact path recorded by a codebook manifest."""
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return path.resolve()


def _validated_reference(
    manifest_path: Path,
    through_round: int,
) -> Dict[str, Any]:
    """Validate and return one deterministic prior-round codebook manifest."""
    payload = read_json(manifest_path)
    if int(payload.get("through_round", -1)) != through_round:
        raise ValueError(
            "Codebook through_round does not match the requested prior round"
        )
    if int(payload.get("manifest_version", 0)) < 3:
        raise ValueError(
            "Generated alignment protocols require deterministic codebook "
            "manifest_version >= 3"
        )
    term_path = _resolved_artifact(manifest_path, payload["term_output"])
    indicator_path = _resolved_artifact(
        manifest_path,
        payload["indicator_output"],
    )
    for label, path, expected in (
        ("term", term_path, payload["term_output_sha256"]),
        ("indicator", indicator_path, payload["indicator_output_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"{label} codebook hash does not match manifest")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "term_path": str(term_path),
        "term_sha256": sha256_file(term_path),
        "term_rows": int(payload["term_rows"]),
        "term_families": int(payload["term_families"]),
        "query_families": int(payload["query_families"]),
        "indicator_path": str(indicator_path),
        "indicator_sha256": sha256_file(indicator_path),
        "indicator_rows": int(payload["indicator_rows"]),
        "indicator_families": int(payload["indicator_families"]),
    }


def _write_protocol(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one deterministic composite protocol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_alignment_protocols(
    current_round: int,
    codebook_manifest: Path,
    output_dir: Path,
    indicator_base_protocol: Path = INDICATOR_BASE_PROTOCOL,
    term_base_brief: Path = TERM_BASE_BRIEF,
) -> Dict[str, Any]:
    """Build deterministic, round-versioned H2 alignment protocols."""
    if current_round < 2:
        raise ValueError("Alignment protocols require current_round >= 2")
    prior_round = current_round - 1
    reference = _validated_reference(codebook_manifest, prior_round)
    indicator_base = indicator_base_protocol.resolve()
    term_brief = term_base_brief.resolve()
    indicator_base_sha = sha256_file(indicator_base)
    term_brief_sha = sha256_file(term_brief)
    common_rule = (
        "First adjudicate the current evidence's inclusion, construct, role, "
        "and T0 or validation boundary. Only then consult the frozen "
        "prior-round reference for label alignment."
    )
    alignment_rule = (
        "Reuse an existing label only when theoretical object, explanatory "
        "or validation role, and T0 boundary are substantively identical. "
        "Lexical, abbreviation, data-source, parameter, threshold, "
        "transformation, or time-window variants alone are not new. A new "
        "label requires an explicit non-redundancy rationale."
    )
    protection_rule = (
        "The historical reference cannot copy or change inclusion, role, "
        "source evidence, or future-information judgments and contains no "
        "downstream dimensions, selected features, or model outcomes."
    )
    manifest_rule = {
        "required_top_level_fields": REGISTRATION_MANIFEST_FIELDS,
        "parameters_type": "nonempty JSON object",
        "status": "complete",
        "reviewer_role": "H2",
        "item_count_rule": (
            "Must equal the exact number of reviewed output rows."
        ),
        "forbidden_model_identity_tokens": ["qwen", "ollama"],
    }
    validation_rule = (
        "Before sealing, run the deterministic "
        "validate-saturation-alignment command for the matching kind. "
        "A review is not importable until this validation and independent "
        "review registration both pass."
    )
    output_root = output_dir.resolve()
    indicator_output = (
        output_root
        / f"round_{current_round:02d}_indicator_alignment_protocol_v3.json"
    )
    term_output = (
        output_root
        / f"round_{current_round:02d}_term_alignment_protocol_v3.json"
    )
    indicator_payload = {
        "protocol_id": (
            "INDEPENDENT_CODEX_INDICATOR_ALIGNMENT_PROTOCOL_V3_"
            f"ROUND_{current_round:02d}"
        ),
        "current_round": current_round,
        "purpose": (
            "Strict-T0 H2 discovery-indicator adjudication with "
            "decision-protected prior-round label alignment."
        ),
        "components": [
            {
                "path": str(indicator_base),
                "sha256": indicator_base_sha,
                "role": "strict_t0_indicator_adjudication_protocol",
            }
        ],
        "required_row_role": {"field": "reviewer_role", "value": "H2"},
        "prior_round_reference": {
            "through_round": prior_round,
            "path": reference["indicator_path"],
            "sha256": reference["indicator_sha256"],
            "rows": reference["indicator_rows"],
            "indicator_families": reference["indicator_families"],
            "export_manifest_path": reference["manifest_path"],
            "export_manifest_sha256": reference["manifest_sha256"],
        },
        "independence_order": common_rule,
        "alignment_rule": alignment_rule,
        "exact_role_rule": (
            "If a normalized canonical label exists in the frozen reference "
            "but the current proposed_role has no prior exemplar for that "
            "label, do not reuse it; use a semantically explicit new "
            "exact-role label and document non-redundancy."
        ),
        "decision_protection": protection_rule,
        "novelty_rule": (
            "A renamed synonym or parameter variant is not a new indicator "
            "family; novelty is computed only after H2 alignment."
        ),
        "provenance_rule": (
            "Every row and the manifest use this JSON's SHA-256 and report "
            "mapped-to-existing and genuinely-new row/family counts."
        ),
        "registration_manifest_schema": manifest_rule,
        "deterministic_post_review_validation": validation_rule,
        "local_models_forbidden": True,
    }
    term_payload = {
        "protocol_id": (
            "INDEPENDENT_CODEX_TERM_ADJUDICATION_PROTOCOL_V3_"
            f"ROUND_{current_round:02d}"
        ),
        "current_round": current_round,
        "purpose": (
            "H2 term adjudication with decision-protected prior-round "
            "term/domain/query-family label alignment."
        ),
        "components": [
            {
                "path": str(term_brief),
                "sha256": term_brief_sha,
                "role": "term_coding_and_adjudication_brief",
            }
        ],
        "required_row_roles": [
            {"field": "coder_role", "value": "H2"},
            {"field": "reviewer_role", "value": "H2"},
        ],
        "prior_round_reference": {
            "through_round": prior_round,
            "path": reference["term_path"],
            "sha256": reference["term_sha256"],
            "rows": reference["term_rows"],
            "term_families": reference["term_families"],
            "query_families": reference["query_families"],
            "export_manifest_path": reference["manifest_path"],
            "export_manifest_sha256": reference["manifest_sha256"],
        },
        "independence_order": common_rule,
        "alignment_rule": alignment_rule,
        "exact_assignment_rule": (
            "A prior term family may be reused only with its frozen "
            "search-domain and query-family assignment and a same-role "
            "exemplar. If the family name exists but that exact assignment "
            "does not, create a semantically explicit new family label."
        ),
        "decision_protection": protection_rule,
        "novelty_rule": (
            "A renamed synonym or parameter variant is not a new term or "
            "query family; novelty is computed only after H2 alignment."
        ),
        "provenance_rule": (
            "Every row and the manifest use this JSON's SHA-256 and report "
            "mapped-to-existing and genuinely-new row/family counts."
        ),
        "registration_manifest_schema": manifest_rule,
        "deterministic_post_review_validation": validation_rule,
        "local_models_forbidden": True,
    }
    _write_protocol(indicator_output, indicator_payload)
    _write_protocol(term_output, term_payload)
    return {
        "current_round": current_round,
        "prior_round": prior_round,
        "indicator_protocol": str(indicator_output),
        "indicator_protocol_sha256": sha256_file(indicator_output),
        "term_protocol": str(term_output),
        "term_protocol_sha256": sha256_file(term_output),
        "reference_manifest": reference["manifest_path"],
        "reference_manifest_sha256": reference["manifest_sha256"],
    }


def main() -> None:
    """Build round-versioned alignment protocols from one frozen reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-round", required=True, type=int)
    parser.add_argument("--codebook-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--indicator-base-protocol",
        default=str(INDICATOR_BASE_PROTOCOL),
    )
    parser.add_argument("--term-base-brief", default=str(TERM_BASE_BRIEF))
    args = parser.parse_args()
    result = build_alignment_protocols(
        args.current_round,
        Path(args.codebook_manifest),
        Path(args.output_dir),
        Path(args.indicator_base_protocol),
        Path(args.term_base_brief),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
