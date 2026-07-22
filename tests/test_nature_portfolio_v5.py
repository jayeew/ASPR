from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nature_portfolio_v5 import (  # noqa: E402
    coverage_quality_summary,
    target_reference_edges,
)


def _target_works() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        year = 2000 + idx
        refs = [f"https://openalex.org/R{idx}_{j}" for j in range(3)]
        rows.append(
            {
                "id": f"https://openalex.org/W{idx}",
                "short_id": f"W{idx}",
                "doi": f"10.1038/test.{idx}",
                "title": f"Nature test paper {idx}",
                "year": year,
                "domain": "machine_learning_foundations" if idx % 2 else "plant_genomics",
                "broad_category": "computer_science_ai_data_science" if idx % 2 else "biology_life_sciences",
                "journal_family": "nature_research",
                "source_id": "https://openalex.org/S1",
                "source_display_name": "Nature",
                "source_issn_l": "0000-0000",
                "primary_field": "Computer Science" if idx % 2 else "Biology",
                "openalex_primary_field": "Computer Science" if idx % 2 else "Biochemistry, Genetics and Molecular Biology",
                "openalex_primary_subfield": "Artificial Intelligence" if idx % 2 else "Genetics",
                "display_community": 100 + idx % 4,
                "display_topic_id": f"https://openalex.org/T{idx % 4}",
                "display_topic_label": "Machine learning" if idx % 2 else "Plant genomics",
                "primary_topic": "Machine learning" if idx % 2 else "Plant genomics",
                "legacy_is_landmark": 0,
                "is_landmark": 0,
                "anchor_label": "",
                "reliable_anchor_source": "",
                "anchor_policy": "venue_driven_v5",
                "document_type": "article",
                "cited_by_count": 0,
                "reference_count": len(refs),
                "source_provider": "openalex",
                "source_dataset": "test",
                "fetched_at": "2026-01-01T00:00:00+00:00",
                "referenced_works": json.dumps(refs),
                "partial_2026": 0,
                "is_target_work": 1,
            }
        )
    return pd.DataFrame(rows)


def _reference_works(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in targets.to_dict("records"):
        for ref in json.loads(target["referenced_works"]):
            rows.append(
                {
                    "id": ref,
                    "short_id": ref.rsplit("/", 1)[-1],
                    "doi": "",
                    "title": f"Reference {ref}",
                    "year": int(target["year"]) - 2,
                    "domain": target["domain"],
                    "broad_category": target["broad_category"],
                    "source_id": "",
                    "source_display_name": "",
                    "primary_field": target["primary_field"],
                    "openalex_primary_field": target["openalex_primary_field"],
                    "openalex_primary_subfield": target["openalex_primary_subfield"],
                    "display_community": target["display_community"],
                    "display_topic_id": target["display_topic_id"],
                    "display_topic_label": target["display_topic_label"],
                    "document_type": "article",
                    "cited_by_count": 0,
                    "reference_count": 0,
                    "source_provider": "openalex",
                    "source_dataset": "test",
                    "fetched_at": "2026-01-01T00:00:00+00:00",
                    "is_target_work": 0,
                }
            )
    return pd.DataFrame(rows)


def _future_deltas(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(targets.to_dict("records")):
        rows.append(
            {
                "paper_id": row["id"],
                "year": row["year"],
                "tau": 8,
                "n_future_citers": idx + 1,
                "future_community_reach": 1 + idx % 4,
                "future_field_reach": 1 + idx % 3,
                "future_subfield_reach": 1 + idx % 2,
                "future_field_entropy": float(idx) / 10.0,
                "future_topic_entropy": float(idx) / 9.0,
                "future_field_simpson": float(idx) / 12.0,
                "future_topic_simpson": float(idx) / 11.0,
            }
        )
    return pd.DataFrame(rows)


def test_offline_roster_has_broad_and_fine_coverage(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_nature_portfolio_source_roster.py",
            "--offline",
            "--out-dir",
            str(tmp_path),
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    taxonomy = pd.read_csv(tmp_path / "nature_subject_taxonomy.csv")
    assert taxonomy.loc[taxonomy["row_kind"] == "broad_category", "broad_category"].nunique() >= 10
    assert taxonomy.loc[taxonomy["row_kind"] == "fine_domain", "domain"].nunique() >= 80
    assert not pd.read_csv(tmp_path / "nature_source_roster.csv").empty


def test_reference_edges_expand_target_references() -> None:
    targets = _target_works().head(2)
    edges = target_reference_edges(targets)
    assert len(edges) == 6
    assert set(edges["relation"]) == {"reference"}


def test_materialize_and_run_fig3_v5_smoke(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    corpus_dir = tmp_path / "corpus"
    out_dir = tmp_path / "fig3"
    source_dir.mkdir()
    targets = _target_works()
    refs = _reference_works(targets)
    edges = target_reference_edges(targets)
    future = _future_deltas(targets)
    targets.to_csv(source_dir / "nature_target_works.csv", index=False)
    refs.to_csv(source_dir / "nature_reference_works.csv", index=False)
    edges.to_csv(source_dir / "nature_reference_edges.csv", index=False)
    future.to_csv(source_dir / "nature_future_graph_deltas.csv", index=False)
    pd.DataFrame(columns=["paper_id", "citer_id"]).to_csv(source_dir / "nature_future_citers.csv", index=False)
    pd.DataFrame(
        [
            {"source_display_name": "Nature", "journal_family": "nature_flagship", "broad_category": "multidisciplinary"},
        ]
    ).to_csv(source_dir / "nature_source_roster.csv", index=False)
    pd.DataFrame(
        [
            {"row_kind": "broad_category", "broad_category": "biology_life_sciences", "domain": ""},
            {"row_kind": "broad_category", "broad_category": "computer_science_ai_data_science", "domain": ""},
            {"row_kind": "fine_domain", "broad_category": "biology_life_sciences", "domain": "plant_genomics", "domain_display_name": "Plant genomics", "query": ""},
            {"row_kind": "fine_domain", "broad_category": "computer_science_ai_data_science", "domain": "machine_learning_foundations", "domain_display_name": "Machine learning foundations", "query": ""},
        ]
    ).to_csv(source_dir / "nature_subject_taxonomy.csv", index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/materialize_nature_full_corpus_v5.py",
            "--target-works",
            str(source_dir / "nature_target_works.csv"),
            "--reference-works",
            str(source_dir / "nature_reference_works.csv"),
            "--reference-edges",
            str(source_dir / "nature_reference_edges.csv"),
            "--future-citers",
            str(source_dir / "nature_future_citers.csv"),
            "--future-graph-deltas",
            str(source_dir / "nature_future_graph_deltas.csv"),
            "--source-roster",
            str(source_dir / "nature_source_roster.csv"),
            "--subject-taxonomy",
            str(source_dir / "nature_subject_taxonomy.csv"),
            "--corpus-dir",
            str(corpus_dir),
            "--min-broad-categories",
            "2",
            "--min-fine-domains",
            "2",
            "--min-broad-eligible",
            "1",
            "--min-domain-eligible",
            "1",
            "--min-papers-per-domain",
            "1",
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    assert (corpus_dir / "v5_nature_portfolio_full_manifest.json").exists()
    quality = json.loads((corpus_dir / "data_quality_report.json").read_text(encoding="utf-8"))
    assert quality["checks"]["no_partial_2026"] == 1

    subprocess.run(
        [
            sys.executable,
            "scripts/run_fig3_nature_full_v5.py",
            "--corpus-dir",
            str(corpus_dir),
            "--out-dir",
            str(out_dir),
            "--n-folds",
            "3",
            "--max-pairs",
            "200",
            "--epochs",
            "5",
            "--quiet",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    assert not pd.read_csv(out_dir / "fig3_v5_nature_full_oof_score_table.csv").empty
    assert not pd.read_csv(out_dir / "fig3_v5_nature_full_target_sensitivity.csv").empty


def test_quality_summary_flags_partial_2026() -> None:
    targets = _target_works()
    targets.loc[0, "year"] = 2026
    summary = coverage_quality_summary(
        targets,
        future=_future_deltas(_target_works()),
        min_broad_categories=1,
        min_fine_domains=1,
        min_broad_eligible=1,
        min_domain_eligible=1,
    )
    assert summary["checks"]["no_partial_2026"] == 0
