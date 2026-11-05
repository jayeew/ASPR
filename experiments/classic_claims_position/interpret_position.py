"""Interpret saved facts using the configured graph role."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ConfigDict

from gear.config import load_config
from gear.innovation.usage import usage_log
from gear.model_client import LazyRoleClient, resolve_role_model

from .graph_position import DEST, save


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation: str
    interpretation: str
    evidence_keys: list[str]


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[Finding]
    limits: str


PROMPT = "根据以下唯一允许的图谱事实，用中文解释局部知识位置。区分实际观测和解释，指出具体研究对象、方法或机制的相近与差异。每项发现绑定所给证据键，讨论组时点明成员、共享或分离内容。仅凭语义邻近不能断言实际采用、因果、历史传承或影响。零邻居明确受限。引用近邻原文中的短语。"


def run(i: str) -> None:
    f = json.loads((DEST / i / "facts.json").read_text())
    evidence = {"structure": f["structure"]}
    for c in f.get("members", [f]):
        evidence[c["claim_id"]] = {
            k: c[k]
            for k in ("text_en", "cutoff_exclusive", "graph_coverage_end", "metrics")
        }
        for n in c["neighbors"]:
            evidence[c["claim_id"] + "/" + n["claim_id"]] = n
    data = json.dumps(evidence, ensure_ascii=False)
    sig = hashlib.sha256(
        (data + PROMPT + str(Position.model_json_schema())).encode()
    ).hexdigest()
    out = DEST / i / "analysis.json"
    if out.exists() and json.loads(out.read_text())["fingerprint"] == sig:
        return
    client = LazyRoleClient(load_config(), "graph_analysis")
    for attempt in range(1, 4):
        try:
            with usage_log(DEST / i / "model_usage.jsonl"):
                result = Position.model_validate(
                    client.generate_json(
                        system=PROMPT,
                        user=data,
                        response_schema=Position.model_json_schema(),
                    )
                )
            invalid = [
                x
                for x in result.findings
                if not x.evidence_keys or not set(x.evidence_keys) <= evidence.keys()
            ]
            if invalid:
                save(i + "/excluded_findings.json", [x.model_dump() for x in invalid])
                result.findings = [x for x in result.findings if x not in invalid]
            if not result.findings:
                raise ValueError("No evidence-bound findings")
            save(
                i + "/analysis.json",
                dict(fingerprint=sig, attempt=attempt, **result.model_dump()),
            )
            return
        except (ValueError, RuntimeError, AssertionError) as e:
            save(f"{i}/attempt_{attempt}.json", {"error": str(e)})
    raise RuntimeError(i)


def main() -> None:
    manifest = DEST / "run_manifest.json"
    if manifest.exists():
        expected = json.loads(manifest.read_text())["role"]
        if list(resolve_role_model(load_config(), "graph_analysis")) != expected:
            raise ValueError(
                "Model configuration changed; use a separate result directory"
            )
    ids = sorted(p.parent.name for p in DEST.glob("*/facts.json"))
    assert len(ids) == 24
    with ThreadPoolExecutor(max_workers=2) as pool:
        for batch in [
            ["C01", "C02"],
            ["G01"],
            [i for i in ids if i not in {"C01", "C02", "G01"}],
        ]:
            list(pool.map(run, batch))


if __name__ == "__main__":
    main()
