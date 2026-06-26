from __future__ import annotations

import pandas as pd

from scripts import build_landmark_registry_v3 as mod


def test_build_registry_limits_three_landmarks_and_excludes_failed_rows() -> None:
    rows = [
        {
            "domain": "field_a",
            "label": f"Landmark {idx}",
            "doi": f"10.1000/{idx}",
            "year": 2000 + idx,
            "validation_status": "passed",
        }
        for idx in range(4)
    ]
    rows.append(
        {
            "domain": "field_b",
            "label": "Failed",
            "doi": "10.1000/failed",
            "year": 2001,
            "validation_status": "failed",
        }
    )

    registry = mod.build_registry(rows)

    assert registry["domain"].tolist() == ["field_a", "field_a", "field_a"]
    assert registry["label"].tolist() == ["Landmark 0", "Landmark 1", "Landmark 2"]
    assert "field_b" not in set(registry["domain"])


def test_domain_coverage_blocks_main_domain_without_v3_landmark_and_keeps_magnetic_candidate() -> None:
    registry = pd.DataFrame(
        [
            {"domain": "crispr", "label": "Jinek 2012", "doi": "10.1126/science.1225829"},
            {
                "domain": "magnetic_properties_of_thin_films",
                "label": "Baibich 1988",
                "doi": "10.1103/physrevlett.61.2472",
            },
        ]
    )
    roster = [
        {"domain_id": "crispr", "status": "main_ready", "family": "biology_biomedicine"},
        {
            "domain_id": "spectroscopy_and_quantum_chemical_studies",
            "status": "main_ready",
            "family": "materials_chemistry",
        },
    ]

    coverage = mod.build_domain_coverage(
        registry,
        roster_rows=roster,
        candidate_domains=["magnetic_properties_of_thin_films"],
    )

    by_domain = coverage.set_index("domain").to_dict("index")
    assert by_domain["crispr"]["v3_status"] == "main_v3_covered"
    assert by_domain["spectroscopy_and_quantum_chemical_studies"]["v3_status"] == "blocked_missing_v3_landmark"
    assert by_domain["spectroscopy_and_quantum_chemical_studies"]["eligible_for_main_roster"] == 0
    assert by_domain["magnetic_properties_of_thin_films"]["roster_status"] == "performance_gated_candidate"
    assert by_domain["magnetic_properties_of_thin_films"]["eligible_for_v3_graph"] == 1
