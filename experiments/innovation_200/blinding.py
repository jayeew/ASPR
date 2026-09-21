"""Remove producer metadata from judge inputs without rewriting evidence text."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from experiments.innovation_200.contracts import HumanMatch, ReportBundle, ReportSource
from gear.contracts import StrictModel

EVALUATION_VERSION = "identity_blind_v2"
BLINDING_LIMITATION = (
    "Producer metadata and native source identifiers are withheld. Report wording, "
    "evidence provenance types and available content can still reveal the method; "
    "this is metadata blinding, not guaranteed semantic anonymity."
)


class BlindHumanEvaluation(StrictModel):
    matches: list[HumanMatch]


def evaluation_root(study: Path) -> Path:
    return study / "evaluations" / EVALUATION_VERSION


def summary_evaluation_root(study: Path) -> Path:
    """Choose one evaluation condition for the whole summary, never mix rows."""
    current = evaluation_root(study)
    paths = [
        *(current / "human").glob("*/*.json"),
        *(current / "pairwise").glob("*.json"),
    ]
    if any(not p.name.endswith((".mapping.json", ".request.json")) for p in paths):
        return current
    return study


def summary_human_root(study: Path) -> Path:
    root = summary_evaluation_root(study)
    return root / ("human_evaluation" if root == study else "human")


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_content(source: ReportSource) -> dict[str, Any]:
    return source.model_dump(mode="json", exclude={"source_id", "passage_id"})


def blind_reports(
    reports: dict[str, ReportBundle],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Give identical evidence the same alias across reports, regardless of origin.

    Source text/title/DOI/provenance are left verbatim. Only source IDs in report
    prose are replaced; the original source/passage IDs stay in the external map.
    """
    digests = sorted(
        {
            payload_hash(_source_content(source))
            for report in reports.values()
            for source in report.references
        }
    )
    aliases = {digest: f"S{index:04d}" for index, digest in enumerate(digests, 1)}
    visible: dict[str, dict[str, Any]] = {}
    external: dict[str, Any] = {}
    for label, report in reports.items():
        references, mapping = [], {}
        for source in report.references:
            alias = aliases[payload_hash(_source_content(source))]
            if source.source_id in mapping:
                raise ValueError("Duplicate source_id in report references")
            mapping[source.source_id] = alias
            references.append(
                {
                    "source_id": alias,
                    "passage_id": f"P{alias[1:]}",
                    **_source_content(source),
                }
            )
        alternatives = "|".join(
            re.escape(key) for key in sorted(mapping, key=len, reverse=True)
        )
        pattern = rf"(?<![\w:])(?:{alternatives})(?![\w:])" if alternatives else ""
        body = (
            re.sub(pattern, lambda match, ids=mapping: ids[match[0]], report.body)
            if pattern
            else report.body
        )
        visible[label] = {"body": body, "references": references}
        external[label] = {
            "paper_id": report.paper_id,
            "system": report.system,
            "original_report_sha256": payload_hash(report.model_dump(mode="json")),
            "source_mapping": [
                {
                    "source_id": source.source_id,
                    "passage_id": source.passage_id,
                    "blind_source_id": mapping[source.source_id],
                    "blind_passage_id": f"P{mapping[source.source_id][1:]}",
                }
                for source in report.references
            ],
        }
    return visible, {
        "evaluation_version": EVALUATION_VERSION,
        "limitation": BLINDING_LIMITATION,
        "reports": external,
    }
