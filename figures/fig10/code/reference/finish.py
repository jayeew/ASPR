"""Validate completed case branches and write the joint-aware whole-paper report."""

from __future__ import annotations
from figure_pipeline.paths import PROJECT_ROOT

import json
import re
from pathlib import Path

from pydantic import Field, field_validator

from experiments.innovation_200 import reporting
from experiments.innovation_200.contracts import ReportBundle, ReportDraft
from experiments.innovation_200.reporting import generate_report, report_markdown
from gear.innovation.usage import usage_log
from gear.innovation.validation import validate_run

ROOT = PROJECT_ROOT
OUT = ROOT / "outputs/fig10_reference"


class CaseReportDraft(ReportDraft):
    """Allow all eight contributions and joint interpretation without truncation."""

    body: str = Field(min_length=1000, max_length=16000)


class CaseReportBundle(ReportBundle):
    @field_validator("body")
    @classmethod
    def prose_length(cls, value: str) -> str:
        prose = re.sub(r"\[((?:M|G|W|WORK|FULLTEXT):[^\]\n]+)\]", "", value)
        if len(prose) > 16000:
            raise ValueError("Case report exceeds 16000 prose characters")
        return value


def main() -> None:
    reporting.ReportDraft = CaseReportDraft
    reporting.ReportBundle = CaseReportBundle
    run = OUT / "analysis"
    validation = validate_run(run)
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2))
    if not validation["valid"]:
        raise ValueError(validation)
    for name in ("facts.json", "analysis.json", "status.json"):
        if not (run / "graph/joint" / name).is_file():
            raise ValueError("Missing joint Graph artifact: " + name)
    target = OUT / "report.json"
    if not target.exists():
        with usage_log(OUT / "report_model_usage.jsonl"):
            report = generate_report("s41586-024-07487-w", "fusion", run)
        target.write_text(report.model_dump_json(indent=2))
        (OUT / "innovation_report.md").write_text(report_markdown(report))


if __name__ == "__main__":
    main()
