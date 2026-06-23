from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.corpus import apply_strict_anchor_policy  # noqa: E402


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def test_strict_anchor_policy_cleans_noisy_domain_and_recovers_manual_match() -> None:
    works = pd.DataFrame(
        [
            {
                "id": "https://openalex.org/W1",
                "doi": "10.1000/noisy.1",
                "title": "Noisy canonical paper",
                "year": 2001,
                "domain": "noisy_domain",
                "primary_field": "Biology",
                "display_community": 1,
                "display_topic_label": "topic",
                "is_landmark": 1,
                "anchor_label": "",
            },
            {
                "id": "https://openalex.org/W2",
                "doi": "10.1000/noisy.2",
                "title": "Noisy ordinary paper",
                "year": 2002,
                "domain": "noisy_domain",
                "primary_field": "Biology",
                "display_community": 1,
                "display_topic_label": "topic",
                "is_landmark": 1,
                "anchor_label": "",
            },
            {
                "id": "https://openalex.org/W3",
                "doi": "10.1021/ja809598r",
                "title": "Organometal Halide Perovskites as Visible-Light Sensitizers for Photovoltaic Cells",
                "year": 2009,
                "domain": "perovskite_solar_cells",
                "primary_field": "Materials",
                "display_community": 2,
                "display_topic_label": "topic",
                "is_landmark": 0,
                "anchor_label": "",
            },
        ]
    )
    landmarks = pd.DataFrame(
        [
            {
                "domain": "noisy_domain",
                "id": "https://openalex.org/W1",
                "doi": "10.1000/noisy.1",
                "title": "Noisy canonical paper",
                "year": 2001,
                "label": "Canonical 2001",
                "include_main": 1,
            },
            {
                "domain": "perovskite_solar_cells",
                "id": "",
                "doi": "10.1021/ja809598r",
                "title": "Organometal Halide Perovskites as Visible-Light Sensitizers for Photovoltaic Cells",
                "year": 2009,
                "label": "Kojima 2009",
                "include_main": 1,
            },
        ]
    )

    strict = apply_strict_anchor_policy(works, landmarks)

    assert int(strict["legacy_is_landmark"].sum()) == 2
    assert int(strict[strict["domain"] == "noisy_domain"]["is_landmark"].sum()) == 1
    assert strict.loc[strict["id"] == "https://openalex.org/W1", "anchor_label"].iloc[0] == "Canonical 2001"
    assert int(strict[strict["domain"] == "perovskite_solar_cells"]["is_landmark"].sum()) == 1
    assert strict.loc[strict["domain"] == "perovskite_solar_cells", "anchor_label"].iloc[0] == "Kojima 2009"


def test_strict_anchor_policy_uses_title_only_when_no_exact_identifier() -> None:
    works = pd.DataFrame(
        [
            {
                "id": "https://openalex.org/W1",
                "doi": "10.1000/exact",
                "title": "The ubiquitin system",
                "year": 1998,
                "domain": "ubiquitin_and_proteasome_pathways",
                "primary_field": "Biology",
                "display_community": 1,
                "display_topic_label": "topic",
                "is_landmark": 0,
                "anchor_label": "",
            },
            {
                "id": "https://openalex.org/W2",
                "doi": "10.1000/other",
                "title": "The ubiquitin system",
                "year": 2000,
                "domain": "ubiquitin_and_proteasome_pathways",
                "primary_field": "Biology",
                "display_community": 1,
                "display_topic_label": "topic",
                "is_landmark": 0,
                "anchor_label": "",
            },
        ]
    )
    landmarks = pd.DataFrame(
        [
            {
                "domain": "ubiquitin_and_proteasome_pathways",
                "id": "https://openalex.org/W1",
                "doi": "10.1000/exact",
                "title": "The ubiquitin system",
                "year": 1998,
                "label": "Exact 1998",
                "include_main": 1,
            }
        ]
    )

    strict = apply_strict_anchor_policy(works, landmarks)

    assert strict["is_landmark"].tolist() == [1, 0]
    assert strict["anchor_label"].tolist() == ["Exact 1998", ""]


def test_offline_corpus_build_creates_views() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_corpus_") as tmp:
        root = Path(tmp)
        fig1_root = root / "fig1"
        fig3_auto_root = root / "fig3_auto"
        crispr_dir = fig1_root / "crispr"
        crispr_dir.mkdir(parents=True)
        fig3_auto_root.mkdir(parents=True)
        records = [
            {
                "id": "https://openalex.org/W100",
                "doi": "10.1000/aspr.100",
                "title": "Synthetic CRISPR anchor",
                "year": 2012,
                "primary_topic": "CRISPR",
                "type": "article",
                "cited_by_count": 100,
                "referenced_works": [],
            },
            {
                "id": "https://openalex.org/W101",
                "doi": "10.1000/aspr.101",
                "title": "Synthetic CRISPR citer",
                "year": 2014,
                "primary_topic": "CRISPR",
                "type": "article",
                "cited_by_count": 50,
                "referenced_works": ["https://openalex.org/W100"],
            },
            {
                "id": "https://openalex.org/W102",
                "doi": "10.1000/aspr.102",
                "title": "Synthetic CRISPR control",
                "year": 2015,
                "primary_topic": "Genome editing",
                "type": "article",
                "cited_by_count": 25,
                "referenced_works": ["https://openalex.org/W100"],
            },
        ]
        crispr_dir.joinpath("works_raw.jsonl").write_text(
            "\n".join(json.dumps(item) for item in records) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "id": "https://openalex.org/W100",
                    "doi": "10.1000/aspr.100",
                    "title": "Synthetic CRISPR anchor",
                    "year": 2012,
                    "primary_topic": "CRISPR",
                    "display_community": 1,
                    "display_label": "CRISPR",
                    "anchor_label": "Synthetic 2012",
                },
                {
                    "id": "https://openalex.org/W101",
                    "doi": "10.1000/aspr.101",
                    "title": "Synthetic CRISPR citer",
                    "year": 2014,
                    "primary_topic": "CRISPR",
                    "display_community": 1,
                    "display_label": "CRISPR",
                    "anchor_label": "",
                },
                {
                    "id": "https://openalex.org/W102",
                    "doi": "10.1000/aspr.102",
                    "title": "Synthetic CRISPR control",
                    "year": 2015,
                    "primary_topic": "Genome editing",
                    "display_community": 2,
                    "display_label": "Genome editing",
                    "anchor_label": "",
                },
            ]
        ).to_csv(crispr_dir / "works_selected.csv", index=False)
        out_dir = Path(tmp) / "v1_test"
        run_cmd(
            [
                sys.executable,
                "-m",
                "aspr.corpus",
                "build",
                "--offline",
                "--out-dir",
                str(out_dir),
                "--fig1-root",
                str(fig1_root),
                "--fig3-auto-root",
                str(fig3_auto_root),
                "--max-domains",
                "6",
                "--papers-per-domain",
                "100",
                "--min-papers-per-domain",
                "1",
                "--quiet",
            ]
        )
        for rel in [
            "manifest.json",
            "quality_report.json",
            "works.csv",
            "citations.csv",
            "topics.csv",
            "views/fig2/multi_domain/works.csv",
            "views/fig3/multi_domain/works.csv",
            "views/fig5/multi_domain/topics.csv",
        ]:
            assert (out_dir / rel).exists(), rel

        works = pd.read_csv(out_dir / "works.csv")
        fig3_works = pd.read_csv(out_dir / "views" / "fig3" / "multi_domain" / "works.csv")
        assert {"id", "year", "domain", "primary_field", "display_community", "is_landmark"}.issubset(works.columns)
        assert {"reference_count", "source_dataset", "legacy_is_landmark"}.issubset(fig3_works.columns)
        assert fig3_works["id"].astype(str).str.contains("::").any()
        assert int(works["year"].max()) <= 2025


def test_derive_strict_creates_strict_views() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_strict_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        out_dir = root / "strict"
        source_dir.mkdir()
        works = pd.DataFrame(
            [
                {
                    "id": f"https://openalex.org/W{i}",
                    "short_id": f"W{i}",
                    "doi": f"10.1000/test.{i}",
                    "title": f"Test paper {i}",
                    "year": 2000 + i,
                    "domain": "noisy_domain",
                    "primary_field": "Biology",
                    "display_community": 1,
                    "display_topic_id": "",
                    "display_topic_label": "topic",
                    "is_landmark": 1,
                    "anchor_label": "",
                    "document_type": "article",
                    "cited_by_count": 100,
                    "reference_count": 10,
                    "source_provider": "test",
                    "source_dataset": "unit",
                    "fetched_at": "",
                    "referenced_works": "[]",
                    "partial_2026": 0,
                }
                for i in range(6)
            ]
        )
        works.to_csv(source_dir / "works.csv", index=False)
        pd.DataFrame([{"source": "https://openalex.org/W1", "target": "https://openalex.org/W0"}]).to_csv(
            source_dir / "citations.csv",
            index=False,
        )
        pd.DataFrame([{"slug": "noisy_domain", "display_name": "Noisy Domain"}]).to_csv(source_dir / "domains.csv", index=False)
        pd.DataFrame(
            [
                {
                    "domain": "noisy_domain",
                    "id": "https://openalex.org/W0",
                    "doi": "10.1000/test.0",
                    "title": "Test paper 0",
                    "year": 2000,
                    "label": "Canonical 2000",
                    "include_main": 1,
                }
            ]
        ).to_csv(source_dir / "landmarks.csv", index=False)
        pd.DataFrame(columns=["community", "label", "x", "y", "domain", "topic_id"]).to_csv(source_dir / "topics.csv", index=False)
        pd.DataFrame(columns=["source_community", "target_community", "weight"]).to_csv(source_dir / "topic_edges.csv", index=False)
        (source_dir / "manifest.json").write_text("{}", encoding="utf-8")

        run_cmd(
            [
                sys.executable,
                "-m",
                "aspr.corpus",
                "derive-strict",
                "--source-dir",
                str(source_dir),
                "--out-dir",
                str(out_dir),
                "--min-papers-per-domain",
                "1",
                "--quiet",
            ]
        )

        fig3_works = pd.read_csv(out_dir / "views" / "fig3" / "noisy_domain" / "works.csv")
        assert int(fig3_works["legacy_is_landmark"].sum()) == 6
        assert int(fig3_works["is_landmark"].sum()) == 1
        assert (out_dir / "strict_view_audit.csv").exists()


if __name__ == "__main__":
    test_strict_anchor_policy_cleans_noisy_domain_and_recovers_manual_match()
    test_offline_corpus_build_creates_views()
    test_derive_strict_creates_strict_views()
    print("test_corpus passed")
