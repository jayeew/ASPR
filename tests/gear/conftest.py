from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

import pytest
from reportlab.pdfgen import canvas

from gear.config import GearConfig, load_config
from gear.contracts import (
    CalibrationCutoff,
    CalibrationForecast,
    CalibrationMeasurement,
    CalibrationMode,
    CalibrationPacketV3,
    CalibrationReliability,
    PaperMetadata,
    ReviewRequest,
)
from gear.paper_compiler import PaperCompiler


@pytest.fixture
def gear_config(tmp_path: Path) -> GearConfig:
    return load_config(
        overrides={
            "allow_external_retrieval": False,
            "cache_dir": str(tmp_path / "cache"),
            "output_root": str(tmp_path / "outputs"),
            "minimum_pdf_characters": 100,
            "minimum_nonempty_page_ratio": 0.25,
            "max_claims": 8,
        }
    )


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    document = canvas.Canvas(str(path))
    pages = [
        [
            "ASPR-GEAR Evidence Grounded Review Study",
            "Abstract",
            "We introduce a novel evidence controller for bounded scientific paper review.",
            "The method uses claim-level spans, deterministic hashes, and explicit budgets.",
            "We evaluate the framework on a dataset of one thousand manuscripts.",
            "The proposed method improves trace accuracy by twelve percent over the baseline.",
            "This result suggests the approach may generalize across multiple scientific domains.",
        ],
        [
            "Methods",
            "The model algorithm compares each claim with method and result evidence.",
            "Results",
            "Ablation analysis reports robustness under removal of the graph signal.",
            "Limitations",
            "The study is limited to English language manuscripts and retrospective evaluation.",
            "References",
            "Prior Study. Evidence systems for review. 10.1000/example.2001.",
        ],
    ]
    for page_lines in pages:
        y = 800
        for repeat in range(4):
            for line in page_lines:
                document.drawString(48, y, f"{line} Replication block {repeat + 1}.")
                y -= 18
        document.showPage()
    document.save()
    return path


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    blocks = [
        "# ASPR-GEAR Evidence Grounded Review Study",
        "## Abstract\nWe introduce a novel evidence controller for bounded scientific paper review.",
        "## Methods\nThe model algorithm compares each claim with method and result evidence.",
        "## Results\nThe proposed method improves trace accuracy by twelve percent over the baseline.",
        "## Limitations\nThe study is limited to English language manuscripts.",
        "## References\nPrior Study. Evidence systems for review. 10.1000/example.2001.",
    ]
    path.write_text("\n\n".join(blocks * 3), encoding="utf-8")
    return path


@pytest.fixture
def paper_request(sample_md: Path) -> ReviewRequest:
    return ReviewRequest(
        paper_path=sample_md,
        metadata=PaperMetadata(
            title="ASPR-GEAR Evidence Grounded Review Study",
            publication_date=date(2010, 1, 2),
        ),
    )


@pytest.fixture
def paper_ir(gear_config: GearConfig, paper_request: ReviewRequest):
    return PaperCompiler(gear_config).compile(paper_request)


@pytest.fixture
def calibration_factory() -> Callable[..., CalibrationPacketV3]:
    def build(
        paper_id: str,
        *,
        score: float | None = None,
        mode: CalibrationMode = CalibrationMode.EXACT_LOOKUP,
        high_profile: bool = False,
    ) -> CalibrationPacketV3:
        return CalibrationPacketV3(
            paper_id=paper_id,
            cutoff=CalibrationCutoff(
                publication_year=2010,
                source_max_year=2009,
                granularity="year",
            ),
            measurement=CalibrationMeasurement(
                substantive_innovation={"EF0017": 0.5, "EF0052": 4.0, "EF0240": 0.7},
                t0_potential={"EF0309": 0.3},
                opportunity={"EF0197": "venue"},
                context_control={"EF0038": 4.0},
                historical_bands={
                    "EF0017": "high_extreme" if high_profile else "typical"
                },
            ),
            forecast=CalibrationForecast(
                p_uptake=0.5 if score is not None else None,
                conditional_diffusion=0.4 if score is not None else None,
                raw_expected_diffusion=0.2 if score is not None else None,
                aspr_score_0_100=score,
            ),
            reliability=CalibrationReliability(
                mode=mode,
                domain="test",
                feature_coverage=1.0 if mode == CalibrationMode.EXACT_LOOKUP else 0.0,
                quality_flags=(
                    [] if mode == CalibrationMode.EXACT_LOOKUP else ["unavailable"]
                ),
            ),
        )

    return build
