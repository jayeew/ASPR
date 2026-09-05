"""Graph- and retrieval-blind full-text LLM innovation baseline."""

from __future__ import annotations

import json
from pathlib import Path

from gear.config import GearConfig
from gear.contracts import PaperIR

from gear.review_contracts import GearBranchResult
from gear.artifacts import read_model, write_json
from gear.model_client import LazyRoleClient


def run_direct_baseline(run_dir: Path, config: GearConfig) -> Path:
    gear = read_model(run_dir / "gear" / "gear_branch_result.json", GearBranchResult)
    paper = read_model(run_dir / "gear" / "paper_ir.json", PaperIR)
    client = LazyRoleClient(config, "relation_fusion")
    raw = client.generate_json(
        system="Evaluate only the apparent innovation of supplied claims from the manuscript text. You have no retrieval and no graph. Express uncertainty explicitly. Return one record per input claim with claim_id, claim_text, stance (recognized, incremental_or_limited, challenged, or unresolved), and assessment.",
        user=json.dumps({"claims": [x.model_dump(mode="json") for x in gear.claims], "support_spans": [{"span_id": x.span_id, "text": x.text} for x in paper.spans]}, ensure_ascii=False),
        response_schema={"type": "object", "properties": {"claims": {"type": "array", "items": {
            "type": "object", "properties": {
                "claim_id": {"type": "string"}, "claim_text": {"type": "string"},
                "stance": {"type": "string", "enum": ["recognized", "incremental_or_limited", "challenged", "unresolved"]},
                "assessment": {"type": "string"},
            }, "required": ["claim_id", "claim_text", "stance", "assessment"], "additionalProperties": False,
        }}}, "required": ["claims"], "additionalProperties": False},
    )
    path = run_dir / "baselines" / "direct_fulltext_llm.json"
    write_json(path, raw)
    return path
