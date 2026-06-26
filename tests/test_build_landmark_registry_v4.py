from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts import build_landmark_registry_v4 as mod


def test_build_registry_v4_rejects_legacy_missing_doi_recent_and_limits_three(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    pd.DataFrame(
        [
            {
                "slug": "field_a",
                "display_name": "Field A",
                "field_name": "Biology and Medicine",
                "query": "field a",
            }
        ]
    ).to_csv(base / "domains.csv", index=False)
    pd.DataFrame(
        [
            {
                "domain": "field_a",
                "label": "Legacy",
                "title": "Legacy anchor",
                "year": 2000,
                "doi": "10.1000/legacy",
                "landmark_source": "fig1_anchor",
                "include_main": 1,
            }
        ]
    ).to_csv(base / "landmarks.csv", index=False)
    v3 = tmp_path / "v3.csv"
    pd.DataFrame().to_csv(v3, index=False)
    seed = tmp_path / "seed.csv"
    pd.DataFrame(
        [
            {
                "domain": "field_a",
                "display_name": "Field A",
                "field_name": "Biology and Medicine",
                "query": "field a",
                "label": f"Valid {idx}",
                "title": f"Valid landmark {idx}",
                "year": 2000 + idx,
                "doi": f"10.1000/{idx}",
                "openalex_id": f"https://openalex.org/W{idx}",
                "evidence_type": "authority_review",
                "evidence_url": f"https://doi.org/10.1000/{idx}",
                "evidence_note": "authority reviewed DOI",
                "include_main": 1,
            }
            for idx in range(4)
        ]
        + [
            {
                "domain": "field_a",
                "display_name": "Field A",
                "field_name": "Biology and Medicine",
                "query": "field a",
                "label": "No DOI",
                "title": "No DOI landmark",
                "year": 2001,
                "doi": "",
                "evidence_type": "authority_review",
                "evidence_url": "",
                "evidence_note": "missing DOI must not pass",
                "include_main": 1,
            },
            {
                "domain": "field_b",
                "display_name": "Field B",
                "field_name": "Methods",
                "query": "field b",
                "label": "Too recent",
                "title": "Recent landmark",
                "year": 2018,
                "doi": "10.1000/recent",
                "evidence_type": "authority_review",
                "evidence_url": "https://doi.org/10.1000/recent",
                "evidence_note": "after cutoff",
                "include_main": 1,
            },
        ]
    ).to_csv(seed, index=False)

    args = argparse.Namespace(
        base_corpus_dir=base,
        v3_registry_csv=v3,
        seed_csv=seed,
        out_csv=tmp_path / "landmark_registry_v4.csv",
        out_json=tmp_path / "landmark_registry_v4.json",
        domain_seed_csv=tmp_path / "publication_candidate_domains_v4.csv",
        report_dir=tmp_path / "report",
        max_main_year=2015,
        max_landmarks_per_domain=3,
        validate_openalex=False,
        openalex_email="",
        timeout_seconds=5,
        sleep_seconds=0.0,
    )

    manifest = mod.build_registry(args)
    registry = pd.read_csv(args.out_csv)
    domain_seed = pd.read_csv(args.domain_seed_csv)
    payload = json.loads(args.out_json.read_text(encoding="utf-8"))

    assert manifest["n_domains"] == 1
    assert registry["domain"].tolist() == ["field_a", "field_a", "field_a"]
    assert registry["doi"].tolist() == ["10.1000/0", "10.1000/1", "10.1000/2"]
    assert "10.1000/legacy" not in set(registry["doi"])
    assert "field_b" not in set(registry["domain"])
    assert domain_seed["slug"].tolist() == ["field_a"]
    assert len(payload) == 3
