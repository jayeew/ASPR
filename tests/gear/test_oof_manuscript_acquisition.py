from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.gear.evaluation.acquire_oof_manuscripts import _select_location
from experiments.gear.evaluation.prepare_stage_a_gear_manifest import prepare_manifest


def test_location_selection_prefers_licensed_submitted_pdf() -> None:
    locations = [
        {
            "is_oa": True,
            "license": None,
            "version": "publishedVersion",
            "pdf_url": "https://publisher.example/paper.pdf",
        },
        {
            "is_oa": True,
            "license": "cc-by-4.0",
            "version": "submittedVersion",
            "pdf_url": "https://repository.example/paper.pdf",
        },
    ]

    selected = _select_location(locations)

    assert selected is not None
    assert selected["license"] == "cc-by-4.0"
    assert selected["version"] == "submittedVersion"


def test_manifest_is_blinded_and_uses_publication_date(tmp_path: Path) -> None:
    cohort = pd.DataFrame(
        {
            "paper_id": [f"https://openalex.org/W{index}" for index in range(6)],
            "score_decile": [0, 2, 4, 6, 8, 9],
            "domain12": ["physics", "chemistry"] * 3,
            "realized_diffusion_target": [999.0] * 6,
        }
    )
    acquired = pd.DataFrame(
        {
            "paper_id": cohort["paper_id"],
            "download_status": ["downloaded"] * 6,
            "publication_date_resolved": ["2020-01-02"] * 6,
            "manuscript_path": [str(tmp_path / f"paper-{i}.pdf") for i in range(6)],
            "title": [f"Paper {i}" for i in range(6)],
            "doi": [None] * 6,
        }
    )
    cohort_path = tmp_path / "cohort.csv"
    acquisition_path = tmp_path / "acquisition.csv"
    output_path = tmp_path / "benchmark.json"
    cohort.to_csv(cohort_path, index=False)
    acquired.to_csv(acquisition_path, index=False)

    summary = prepare_manifest(cohort_path, acquisition_path, output_path, limit=6)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["selected"] == 6
    assert payload["selection_uses_future_outcomes"] is False
    assert all(case["cutoff"] == "2020-01-02" for case in payload["cases"])
    assert "realized_diffusion_target" not in output_path.read_text(encoding="utf-8")
