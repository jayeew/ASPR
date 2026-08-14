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

import scripts.publication_corpus_v2 as publication_corpus_v2  # noqa: E402
from gear.corpus import namespace_multi_domain  # noqa: E402
from scripts.publication_corpus_v2 import (  # noqa: E402
    FIGURE_LOGIC_POLICY,
    FIGURE_VIEW_CONTRACT,
    augment_corpus_with_global_landmark_repairs,
    augment_corpus_with_external_landmark_records,
    augment_corpus_with_landmark_reference_support,
    augment_corpus_with_topup_work_records,
    build_global_landmark_repair_plan,
    build_domain_diagnostics,
    build_fig3_readiness_diagnostics,
    build_domain_status_table,
    build_publication_target_roster,
    clean_landmark_registry,
    deduplicate_domain_dois,
    materialize_candidate_corpus,
    openalex_search_query_from_domain_query,
    repair_corpus_metadata,
)


def _write_minimal_corpus(root: Path) -> None:
    works_rows = []
    citation_rows = []
    for idx in range(12):
        works_rows.append(
            {
                "id": f"https://openalex.org/WGOOD{idx}",
                "doi": f"10.1000/good.{idx}",
                "title": f"High quality field paper {idx}",
                "year": 2000 + idx,
                "domain": "high_quality_field",
                "primary_field": "Biology",
                "display_community": idx % 3,
                "display_topic_label": "clean topic",
                "is_landmark": 1 if idx in {2, 7} else 0,
                "anchor_label": "Good 2002" if idx == 2 else "Good 2007" if idx == 7 else "",
                "cited_by_count": 100 - idx,
                "reference_count": 5,
            }
        )
        if idx > 0:
            citation_rows.append(
                {
                    "source": f"https://openalex.org/WGOOD{idx}",
                    "target": f"https://openalex.org/WGOOD{idx - 1}",
                    "relation": "reference",
                }
            )

    for idx in range(10):
        works_rows.append(
            {
                "id": f"https://openalex.org/WNOISY{idx}",
                "doi": f"10.1000/noisy.{idx}",
                "title": f"Noisy inherited paper {idx}",
                "year": 2000 + idx,
                "domain": "noisy_legacy_field",
                "primary_field": "Computer Science",
                "display_community": idx % 2,
                "display_topic_label": "alphafold protein folding" if idx < 3 else "clean topic",
                "is_landmark": 1,
                "anchor_label": "",
                "cited_by_count": 50 - idx,
                "reference_count": 3,
            }
        )
        if idx > 0:
            citation_rows.append(
                {
                    "source": f"https://openalex.org/WNOISY{idx}",
                    "target": f"https://openalex.org/WNOISY{idx - 1}",
                    "relation": "reference",
                }
            )

    pd.DataFrame(works_rows).to_csv(root / "works.csv", index=False)
    pd.DataFrame(citation_rows).to_csv(root / "citations.csv", index=False)
    pd.DataFrame(
        [
            {"slug": "high_quality_field", "display_name": "High Quality Field", "field_name": "Biology"},
            {"slug": "noisy_legacy_field", "display_name": "Noisy Legacy Field", "field_name": "Computer Science"},
        ]
    ).to_csv(root / "domains.csv", index=False)
    pd.DataFrame(
        [
            {
                "domain": "high_quality_field",
                "landmark_source": "manual",
                "source_id": "good:2002",
                "label": "Good 2002",
                "id": "https://openalex.org/WGOOD2",
                "doi": "10.1000/good.2",
                "title": "High quality field paper 2",
                "year": 2002,
                "match_confidence": 1.0,
                "include_main": 1,
            },
            {
                "domain": "high_quality_field",
                "landmark_source": "fig1_anchor",
                "source_id": "high_quality_field:Good 2007",
                "label": "Good 2007",
                "id": "https://openalex.org/WGOOD7",
                "doi": "10.1000/good.7",
                "title": "High quality field paper 7",
                "year": 2007,
                "match_confidence": 1.0,
                "include_main": 1,
            },
            {
                "domain": "noisy_legacy_field",
                "landmark_source": "fig1_anchor",
                "source_id": "noisy_legacy_field:nan",
                "label": "",
                "id": "https://openalex.org/WNOISY0",
                "doi": "10.1000/noisy.0",
                "title": "Noisy inherited paper 0",
                "year": 2000,
                "match_confidence": 1.0,
                "include_main": 1,
            },
        ]
    ).to_csv(root / "landmarks.csv", index=False)
    pd.DataFrame(columns=["community", "label", "x", "y", "domain", "topic_id"]).to_csv(root / "topics.csv", index=False)
    pd.DataFrame(columns=["source_community", "target_community", "weight"]).to_csv(root / "topic_edges.csv", index=False)


def _add_misplaced_landmark_case(root: Path) -> None:
    works = pd.read_csv(root / "works.csv")
    target = works[works["domain"] == "high_quality_field"].copy()
    target["domain"] = "misplaced_landmark_field"
    target["id"] = [f"https://openalex.org/WMISPLACED{i}" for i in range(len(target))]
    target["doi"] = [f"10.1000/misplaced.{i}" for i in range(len(target))]
    target["is_landmark"] = 0
    target["anchor_label"] = ""
    source_landmark = target.iloc[[0]].copy()
    source_landmark["domain"] = "neighbor_source_field"
    source_landmark["id"] = "https://openalex.org/WGLOBAL999"
    source_landmark["doi"] = "10.1000/global.landmark"
    source_landmark["title"] = "Globally misplaced landmark"
    source_landmark["year"] = 2004
    source_landmark["display_topic_label"] = "neighbor topic"
    pd.concat([works, target, source_landmark], ignore_index=True).to_csv(root / "works.csv", index=False)

    citations = pd.read_csv(root / "citations.csv")
    target_citations = [
        {
            "source": f"https://openalex.org/WMISPLACED{i}",
            "target": f"https://openalex.org/WMISPLACED{i - 1}",
            "relation": "reference",
        }
        for i in range(1, len(target))
    ]
    target_citations.append(
        {
            "source": "https://openalex.org/WGLOBAL999",
            "target": "https://openalex.org/WMISPLACED0",
            "relation": "reference",
        }
    )
    pd.concat([citations, pd.DataFrame(target_citations)], ignore_index=True).to_csv(root / "citations.csv", index=False)

    domains = pd.read_csv(root / "domains.csv")
    pd.concat(
        [
            domains,
            pd.DataFrame(
                [
                    {"slug": "misplaced_landmark_field", "display_name": "Misplaced Landmark Field", "field_name": "Biology"},
                    {"slug": "neighbor_source_field", "display_name": "Neighbor Source Field", "field_name": "Biology"},
                ]
            ),
        ],
        ignore_index=True,
    ).to_csv(root / "domains.csv", index=False)

    landmarks = pd.read_csv(root / "landmarks.csv")
    pd.concat(
        [
            landmarks,
            pd.DataFrame(
                [
                    {
                        "domain": "misplaced_landmark_field",
                        "landmark_source": "manual",
                        "source_id": "misplaced:global",
                        "label": "Global 2004",
                        "id": "https://openalex.org/WGLOBAL999",
                        "doi": "10.1000/global.landmark",
                        "title": "Globally misplaced landmark",
                        "year": 2004,
                        "match_confidence": 1.0,
                        "include_main": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).to_csv(root / "landmarks.csv", index=False)


def test_clean_landmark_registry_drops_blank_legacy_fig1_anchors() -> None:
    landmarks = pd.DataFrame(
        [
            {
                "domain": "manual_domain",
                "landmark_source": "manual",
                "source_id": "manual:paper",
                "label": "Manual 2001",
                "id": "",
                "doi": "10.1000/manual",
                "title": "Manual landmark",
                "year": 2001,
                "include_main": 1,
            },
            {
                "domain": "legacy_domain",
                "landmark_source": "fig1_anchor",
                "source_id": "legacy:nan",
                "label": "",
                "id": "https://openalex.org/W1",
                "doi": "10.1000/legacy",
                "title": "Blank inherited anchor",
                "year": 2002,
                "include_main": 1,
            },
            {
                "domain": "legacy_domain",
                "landmark_source": "fig1_anchor",
                "source_id": "legacy:Named 2003",
                "label": "Named 2003",
                "id": "https://openalex.org/W2",
                "doi": "10.1000/named",
                "title": "Named inherited anchor",
                "year": 2003,
                "include_main": 1,
            },
        ]
    )

    clean = clean_landmark_registry(landmarks, max_landmarks_per_domain=5)

    assert clean["label"].tolist() == ["Manual 2001", "Named 2003"]
    assert clean.loc[clean["label"] == "Manual 2001", "needs_manual_confirmation"].iloc[0] == 0
    assert clean.loc[clean["label"] == "Named 2003", "needs_manual_confirmation"].iloc[0] == 1
    assert "Blank inherited anchor" not in clean["title"].tolist()


def test_namespace_multi_domain_keeps_topic_communities_unique_by_domain() -> None:
    works = pd.DataFrame(
        [
            {
                "id": "https://openalex.org/WA1",
                "year": 2001,
                "title": "Alpha paper",
                "domain": "alpha",
                "primary_field": "Biology",
                "display_community": 7,
                "display_topic_label": "Alpha topic",
            },
            {
                "id": "https://openalex.org/WB1",
                "year": 2001,
                "title": "Beta paper",
                "domain": "beta",
                "primary_field": "Physics",
                "display_community": 7,
                "display_topic_label": "Beta topic",
            },
        ]
    )
    topics = pd.DataFrame(
        [
            {"community": 7, "label": "Alpha topic", "x": 0.1, "y": 0.2, "domain": "alpha", "topic_id": "T7"},
            {"community": 7, "label": "Beta topic", "x": 0.3, "y": 0.4, "domain": "beta", "topic_id": "T7"},
        ]
    )

    out_works, _, out_topics, _ = namespace_multi_domain(
        works,
        pd.DataFrame(columns=["source", "target"]),
        topics,
        pd.DataFrame(columns=["source_community", "target_community", "weight"]),
    )

    assert len(out_topics) == 2
    assert out_topics["community"].is_unique
    assert out_works["display_community"].is_unique
    assert set(out_topics["domain"]) == {"alpha", "beta"}


def test_clean_landmark_registry_deduplicates_same_labeled_event() -> None:
    landmarks = pd.DataFrame(
        [
            {
                "domain": "topological_insulators",
                "landmark_source": "manual",
                "source_id": "topological:doi",
                "label": "Kane/Mele 2005",
                "id": "",
                "doi": "10.1103/physrevlett.95.146802",
                "title": "Quantum Spin Hall Effect in Graphene",
                "year": 2005,
                "include_main": 1,
            },
            {
                "domain": "topological_insulators",
                "landmark_source": "manual",
                "source_id": "topological:openalex",
                "label": "Kane/Mele 2005",
                "id": "https://openalex.org/W2030164271",
                "doi": "10.1103/physrevlett.95.226801",
                "title": "Quantum Spin Hall Effect in Graphene",
                "year": 2005,
                "include_main": 1,
            },
        ]
    )

    clean = clean_landmark_registry(landmarks, max_landmarks_per_domain=5)

    assert len(clean) == 1
    assert clean["label"].tolist() == ["Kane/Mele 2005"]


def test_build_domain_diagnostics_ranks_clean_fields_above_noisy_legacy_fields() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_") as tmp:
        root = Path(tmp)
        _write_minimal_corpus(root)

        diagnostics = build_domain_diagnostics(
            corpus_dir=root,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )

        high = diagnostics[diagnostics["domain"] == "high_quality_field"].iloc[0]
        noisy = diagnostics[diagnostics["domain"] == "noisy_legacy_field"].iloc[0]
        assert high["recommended_role"] == "main_candidate"
        assert noisy["recommended_role"] in {"rebuild_needed", "exclude_for_now"}
        assert "legacy_landmark_inflation" in str(noisy["failure_reasons"])
        assert high["publication_score"] > noisy["publication_score"]


def test_build_domain_diagnostics_marks_reference_closure_gaps_as_rebuild_needed() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_closure_") as tmp:
        root = Path(tmp)
        _write_minimal_corpus(root)
        citations = pd.read_csv(root / "citations.csv")
        citations.loc[citations["source"].astype(str).str.contains("WGOOD"), "target"] = "https://openalex.org/WOUTSIDE"
        citations.to_csv(root / "citations.csv", index=False)

        diagnostics = build_domain_diagnostics(
            corpus_dir=root,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )

        high = diagnostics[diagnostics["domain"] == "high_quality_field"].iloc[0]
        assert high["recommended_role"] == "rebuild_needed"
        assert "reference_closure" in str(high["failure_reasons"])


def test_build_domain_diagnostics_requires_clean_landmark_to_match_domain_works() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_unmatched_") as tmp:
        root = Path(tmp)
        _write_minimal_corpus(root)
        works = pd.read_csv(root / "works.csv")
        extra = works[works["domain"] == "high_quality_field"].copy()
        extra["domain"] = "unmatched_landmark_field"
        extra["id"] = [f"https://openalex.org/WUNMATCHED{i}" for i in range(len(extra))]
        extra["doi"] = [f"10.1000/unmatched.{i}" for i in range(len(extra))]
        extra["is_landmark"] = 0
        extra["anchor_label"] = ""
        pd.concat([works, extra], ignore_index=True).to_csv(root / "works.csv", index=False)

        domains = pd.read_csv(root / "domains.csv")
        pd.concat(
            [
                domains,
                pd.DataFrame(
                    [
                        {
                            "slug": "unmatched_landmark_field",
                            "display_name": "Unmatched Landmark Field",
                            "field_name": "Biology",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ).to_csv(root / "domains.csv", index=False)

        landmarks = pd.read_csv(root / "landmarks.csv")
        pd.concat(
            [
                landmarks,
                pd.DataFrame(
                    [
                        {
                            "domain": "unmatched_landmark_field",
                            "landmark_source": "manual",
                            "source_id": "unmatched:2002",
                            "label": "Missing 2002",
                            "id": "https://openalex.org/WMISSING2002",
                            "doi": "10.1000/missing.2002",
                            "title": "Missing landmark paper",
                            "year": 2002,
                            "match_confidence": 1.0,
                            "include_main": 1,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ).to_csv(root / "landmarks.csv", index=False)

        diagnostics = build_domain_diagnostics(
            corpus_dir=root,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )

        unmatched = diagnostics[diagnostics["domain"] == "unmatched_landmark_field"].iloc[0]
        assert unmatched["recommended_role"] == "rebuild_needed"
        assert unmatched["matched_clean_landmark_rows"] == 0
        assert "unmatched_clean_landmark" in str(unmatched["failure_reasons"])


def test_global_landmark_repair_plan_finds_exact_match_in_neighbor_domain() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_repair_plan_") as tmp:
        root = Path(tmp)
        _write_minimal_corpus(root)
        _add_misplaced_landmark_case(root)

        works = pd.read_csv(root / "works.csv")
        clean_landmarks = clean_landmark_registry(pd.read_csv(root / "landmarks.csv"))
        plan = build_global_landmark_repair_plan(works, clean_landmarks)

        misplaced = plan[plan["domain"] == "misplaced_landmark_field"].iloc[0]
        assert misplaced["repair_status"] == "global_source_match_available"
        assert misplaced["source_domain"] == "neighbor_source_field"
        assert misplaced["matched_work_id"] == "https://openalex.org/WGLOBAL999"
        assert misplaced["matched_by"] == "id"


def test_global_landmark_repair_plan_marks_absent_landmark_for_external_fetch() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_repair_absent_") as tmp:
        root = Path(tmp)
        _write_minimal_corpus(root)
        works = pd.read_csv(root / "works.csv")
        absent = pd.DataFrame(
            [
                {
                    "domain": "absent_landmark_field",
                    "landmark_source": "manual",
                    "accepted_landmark_source": "manual",
                    "source_id": "absent:2001",
                    "label": "Absent 2001",
                    "id": "https://openalex.org/WABSENT2001",
                    "doi": "10.1000/absent.2001",
                    "title": "Absent landmark paper",
                    "year": 2001,
                    "match_confidence": 1.0,
                    "include_main": 1,
                    "needs_manual_confirmation": 0,
                    "evidence_key": "https://openalex.org/WABSENT2001",
                }
            ]
        )

        plan = build_global_landmark_repair_plan(works, absent)

        row = plan[plan["domain"] == "absent_landmark_field"].iloc[0]
        assert row["repair_status"] == "external_fetch_required"
        assert row["matched_work_id"] == ""


def test_augment_corpus_with_global_landmark_repairs_adds_audited_target_copy() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_repair_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        repaired_dir = root / "repaired"
        source_dir.mkdir()
        _write_minimal_corpus(source_dir)
        _add_misplaced_landmark_case(source_dir)

        manifest = augment_corpus_with_global_landmark_repairs(
            source_corpus_dir=source_dir,
            target_corpus_dir=repaired_dir,
            domains=["misplaced_landmark_field"],
        )

        assert manifest["n_repaired_landmarks"] == 1
        works = pd.read_csv(repaired_dir / "works.csv")
        repaired = works[
            (works["domain"] == "misplaced_landmark_field")
            & (works["id"] == "https://openalex.org/WGLOBAL999")
        ].iloc[0]
        assert repaired["landmark_repair_source_domain"] == "neighbor_source_field"
        assert repaired["source_dataset"] == "v2_global_landmark_repair"

        diagnostics = build_domain_diagnostics(
            repaired_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        row = diagnostics[diagnostics["domain"] == "misplaced_landmark_field"].iloc[0]
        assert row["matched_clean_landmark_rows"] == 1
        assert "unmatched_clean_landmark" not in str(row["failure_reasons"])


def test_publication_corpus_v2_cli_writes_global_landmark_repair_source() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_repair_cli_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        repaired_dir = root / "repaired"
        source_dir.mkdir()
        _write_minimal_corpus(source_dir)
        _add_misplaced_landmark_case(source_dir)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "repair-global-landmarks",
                "--source-corpus-dir",
                str(source_dir),
                "--target-corpus-dir",
                str(repaired_dir),
                "--domains",
                "misplaced_landmark_field",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        assert (repaired_dir / "global_landmark_repair_manifest.json").exists()
        assert (repaired_dir / "landmark_repair_plan.csv").exists()
        works = pd.read_csv(repaired_dir / "works.csv")
        assert (
            (works["domain"] == "misplaced_landmark_field")
            & (works["id"] == "https://openalex.org/WGLOBAL999")
        ).any()


def _add_external_landmark_case(root: Path) -> None:
    works = pd.read_csv(root / "works.csv")
    target = works[works["domain"] == "high_quality_field"].copy()
    target["domain"] = "external_landmark_field"
    target["id"] = [f"https://openalex.org/WEXTERNALTARGET{i}" for i in range(len(target))]
    target["doi"] = [f"10.1000/external.target.{i}" for i in range(len(target))]
    target["is_landmark"] = 0
    target["anchor_label"] = ""
    pd.concat([works, target], ignore_index=True).to_csv(root / "works.csv", index=False)

    citations = pd.read_csv(root / "citations.csv")
    target_citations = [
        {
            "source": f"https://openalex.org/WEXTERNALTARGET{i}",
            "target": f"https://openalex.org/WEXTERNALTARGET{i - 1}",
            "relation": "reference",
        }
        for i in range(1, len(target))
    ]
    pd.concat([citations, pd.DataFrame(target_citations)], ignore_index=True).to_csv(root / "citations.csv", index=False)

    domains = pd.read_csv(root / "domains.csv")
    pd.concat(
        [
            domains,
            pd.DataFrame(
                [
                    {"slug": "external_landmark_field", "display_name": "External Landmark Field", "field_name": "Biology"},
                ]
            ),
        ],
        ignore_index=True,
    ).to_csv(root / "domains.csv", index=False)

    landmarks = pd.read_csv(root / "landmarks.csv")
    pd.concat(
        [
            landmarks,
            pd.DataFrame(
                [
                    {
                        "domain": "external_landmark_field",
                        "landmark_source": "manual",
                        "source_id": "external:2004",
                        "label": "External 2004",
                        "id": "https://openalex.org/WEXTERNAL2004",
                        "doi": "10.1000/external.2004",
                        "title": "Externally fetched landmark",
                        "year": 2004,
                        "match_confidence": 1.0,
                        "include_main": 1,
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).to_csv(root / "landmarks.csv", index=False)


def _fake_external_record() -> dict:
    return {
        "domain": "external_landmark_field",
        "label": "External 2004",
        "matched_by": "doi",
        "work": {
            "id": "https://openalex.org/WEXTERNAL2004",
            "doi": "https://doi.org/10.1000/external.2004",
            "display_name": "Externally fetched landmark",
            "publication_year": 2004,
            "type": "article",
            "cited_by_count": 500,
            "referenced_works": [
                "https://openalex.org/WEXTERNALTARGET0",
                "https://openalex.org/WOUTSIDEEXTERNAL",
            ],
            "primary_topic": {
                "id": "https://openalex.org/T1",
                "display_name": "External topic",
                "field": {"display_name": "Biology"},
            },
        },
    }


def test_augment_corpus_with_external_landmark_records_adds_fetched_landmark() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_external_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "external"
        source_dir.mkdir()
        _write_minimal_corpus(source_dir)
        _add_external_landmark_case(source_dir)

        manifest = augment_corpus_with_external_landmark_records(
            source_corpus_dir=source_dir,
            target_corpus_dir=target_dir,
            records=[_fake_external_record()],
        )

        assert manifest["n_external_landmarks_added"] == 1
        works = pd.read_csv(target_dir / "works.csv")
        added = works[
            (works["domain"] == "external_landmark_field")
            & (works["id"] == "https://openalex.org/WEXTERNAL2004")
        ].iloc[0]
        assert added["source_dataset"] == "v2_external_landmark_fetch"
        assert added["anchor_label"] == "External 2004"
        citations = pd.read_csv(target_dir / "citations.csv")
        assert (
            (citations["source"] == "https://openalex.org/WEXTERNAL2004")
            & (citations["target"] == "https://openalex.org/WOUTSIDEEXTERNAL")
        ).any()

        diagnostics = build_domain_diagnostics(
            target_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        row = diagnostics[diagnostics["domain"] == "external_landmark_field"].iloc[0]
        assert row["matched_clean_landmark_rows"] == 1
        assert "unmatched_clean_landmark" not in str(row["failure_reasons"])


def test_publication_corpus_v2_cli_uses_cached_external_landmark_records() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_external_cli_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "external"
        records_path = root / "records.jsonl"
        source_dir.mkdir()
        _write_minimal_corpus(source_dir)
        _add_external_landmark_case(source_dir)
        records_path.write_text(json.dumps(_fake_external_record()) + "\n", encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "fetch-external-landmarks",
                "--source-corpus-dir",
                str(source_dir),
                "--target-corpus-dir",
                str(target_dir),
                "--fetched-records-jsonl",
                str(records_path),
                "--domains",
                "external_landmark_field",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        assert (target_dir / "external_landmark_fetch_manifest.json").exists()
        works = pd.read_csv(target_dir / "works.csv")
        assert (
            (works["domain"] == "external_landmark_field")
            & (works["id"] == "https://openalex.org/WEXTERNAL2004")
        ).any()


def _write_near_threshold_corpus(root: Path) -> None:
    works_rows = []
    citation_rows = []
    for idx in range(7):
        works_rows.append(
            {
                "id": f"https://openalex.org/WNEAR{idx}",
                "doi": f"10.1000/near.{idx}",
                "title": f"Near threshold paper {idx}",
                "year": 2000 + idx,
                "domain": "near_threshold_field",
                "primary_field": "Biology",
                "display_community": idx % 2,
                "display_topic_label": "near topic",
                "is_landmark": 1 if idx == 2 else 0,
                "anchor_label": "Near 2002" if idx == 2 else "",
                "cited_by_count": 100 - idx,
                "reference_count": 2,
            }
        )
        if idx > 0:
            citation_rows.append(
                {
                    "source": f"https://openalex.org/WNEAR{idx}",
                    "target": f"https://openalex.org/WNEAR{idx - 1}",
                    "relation": "reference",
                }
            )
    pd.DataFrame(works_rows).to_csv(root / "works.csv", index=False)
    pd.DataFrame(citation_rows).to_csv(root / "citations.csv", index=False)
    pd.DataFrame(
        [{"slug": "near_threshold_field", "display_name": "Near Threshold Field", "field_name": "Biology"}]
    ).to_csv(root / "domains.csv", index=False)
    pd.DataFrame(
        [
            {
                "domain": "near_threshold_field",
                "landmark_source": "manual",
                "source_id": "near:2002",
                "label": "Near 2002",
                "id": "https://openalex.org/WNEAR2",
                "doi": "10.1000/near.2",
                "title": "Near threshold paper 2",
                "year": 2002,
                "match_confidence": 1.0,
                "include_main": 1,
            }
        ]
    ).to_csv(root / "landmarks.csv", index=False)
    pd.DataFrame(columns=["community", "label", "x", "y", "domain", "topic_id"]).to_csv(root / "topics.csv", index=False)
    pd.DataFrame(columns=["source_community", "target_community", "weight"]).to_csv(root / "topic_edges.csv", index=False)


def _fake_topup_record() -> dict:
    return {
        "domain": "near_threshold_field",
        "work": {
            "id": "https://openalex.org/WTOPUP1",
            "doi": "https://doi.org/10.1000/topup.1",
            "display_name": "Top-up ordinary paper",
            "publication_year": 2008,
            "type": "article",
            "cited_by_count": 42,
            "referenced_works": ["https://openalex.org/WNEAR6"],
            "primary_topic": {
                "id": "https://openalex.org/T2",
                "display_name": "near topic",
                "field": {"display_name": "Biology"},
            },
        },
    }


def test_augment_corpus_with_topup_work_records_can_rescue_near_threshold_domain() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_topup_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "topup"
        source_dir.mkdir()
        _write_near_threshold_corpus(source_dir)

        before = build_domain_diagnostics(
            source_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        assert "n_works" in before[before["domain"] == "near_threshold_field"]["failure_reasons"].iloc[0]

        manifest = augment_corpus_with_topup_work_records(
            source_corpus_dir=source_dir,
            target_corpus_dir=target_dir,
            records=[_fake_topup_record()],
        )

        assert manifest["n_topup_works_added"] == 1
        after = build_domain_diagnostics(
            target_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        row = after[after["domain"] == "near_threshold_field"].iloc[0]
        assert row["recommended_role"] == "main_candidate"
        assert row["n_works"] == 8


def test_topup_work_records_can_keep_only_local_reference_edges() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_topup_local_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "topup"
        source_dir.mkdir()
        _write_near_threshold_corpus(source_dir)
        record = _fake_topup_record()
        record["work"]["referenced_works"] = [
            "https://openalex.org/WNEAR6",
            "https://openalex.org/WOUTSIDE",
        ]

        manifest = augment_corpus_with_topup_work_records(
            source_corpus_dir=source_dir,
            target_corpus_dir=target_dir,
            records=[record],
            min_local_refs=1,
            local_references_only=True,
        )

        assert manifest["n_topup_works_added"] == 1
        assert manifest["n_topup_reference_edges_added"] == 1
        citations = pd.read_csv(target_dir / "citations.csv")
        topup_edges = citations[citations["source"].astype(str).eq("https://openalex.org/WTOPUP1")]
        assert topup_edges["target"].tolist() == ["https://openalex.org/WNEAR6"]
        works = pd.read_csv(target_dir / "works.csv")
        refs = json.loads(works.loc[works["id"].eq("https://openalex.org/WTOPUP1"), "referenced_works"].iloc[0])
        assert refs == ["https://openalex.org/WNEAR6"]


def test_topup_work_records_skip_candidates_without_enough_local_refs() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_topup_local_skip_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "topup"
        source_dir.mkdir()
        _write_near_threshold_corpus(source_dir)

        manifest = augment_corpus_with_topup_work_records(
            source_corpus_dir=source_dir,
            target_corpus_dir=target_dir,
            records=[_fake_topup_record()],
            min_local_refs=2,
            local_references_only=True,
        )

        assert manifest["n_topup_works_added"] == 0
        assert pd.read_csv(target_dir / "works.csv")["id"].astype(str).str.contains("WTOPUP1").sum() == 0


def test_augment_corpus_with_landmark_reference_support_adds_prior_reference_targets() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_ref_support_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "supported"
        source_dir.mkdir()
        _write_near_threshold_corpus(source_dir)

        works = pd.read_csv(source_dir / "works.csv")
        works.loc[works["id"].eq("https://openalex.org/WNEAR2"), "referenced_works"] = json.dumps(
            ["https://openalex.org/WSUPPORT1", "https://openalex.org/WSUPPORT2"]
        )
        works.to_csv(source_dir / "works.csv", index=False)

        fake_works = {
            "https://openalex.org/WSUPPORT1": {
                "id": "https://openalex.org/WSUPPORT1",
                "doi": "https://doi.org/10.1000/support.1",
                "display_name": "Prior support one",
                "publication_year": 1998,
                "type": "article",
                "cited_by_count": 5,
                "referenced_works": [],
                "primary_topic": {
                    "id": "https://openalex.org/TREF1",
                    "display_name": "prior support",
                    "field": {"display_name": "Biology"},
                },
            },
            "https://openalex.org/WSUPPORT2": {
                "id": "https://openalex.org/WSUPPORT2",
                "doi": "https://doi.org/10.1000/support.2",
                "display_name": "Prior support two",
                "publication_year": 1999,
                "type": "article",
                "cited_by_count": 4,
                "referenced_works": [],
                "primary_topic": {
                    "id": "https://openalex.org/TREF2",
                    "display_name": "prior support",
                    "field": {"display_name": "Biology"},
                },
            },
        }

        def fake_fetch(identifier: object, timeout_seconds: int = 60):
            return fake_works.get(str(identifier))

        original = publication_corpus_v2.fetch_openalex_work
        publication_corpus_v2.fetch_openalex_work = fake_fetch
        try:
            manifest = augment_corpus_with_landmark_reference_support(
                source_corpus_dir=source_dir,
                target_corpus_dir=target_dir,
                domains=["near_threshold_field"],
                min_internal_refs=3,
                max_support_refs_per_landmark=3,
            )
        finally:
            publication_corpus_v2.fetch_openalex_work = original

        assert manifest["n_support_works_added"] == 2
        assert manifest["n_landmarks_repaired"] == 1
        report = pd.read_csv(target_dir / "landmark_reference_support_report.csv")
        assert report["refs_before"].iloc[0] == 1
        assert report["refs_after"].iloc[0] == 3
        works_after = pd.read_csv(target_dir / "works.csv")
        assert set(works_after[works_after["source_dataset"].eq("v2_landmark_reference_support")]["id"]) == {
            "https://openalex.org/WSUPPORT1",
            "https://openalex.org/WSUPPORT2",
        }
        citations_after = pd.read_csv(target_dir / "citations.csv")
        landmark_edges = citations_after[citations_after["source"].eq("https://openalex.org/WNEAR2")]
        assert {"https://openalex.org/WSUPPORT1", "https://openalex.org/WSUPPORT2"}.issubset(
            set(landmark_edges["target"])
        )


def test_openalex_search_query_from_domain_query_flattens_boolean_seed_query() -> None:
    query = '"microbiome" OR "metagenomics" OR "human microbiome"'

    sanitized = openalex_search_query_from_domain_query(query)

    assert sanitized == "microbiome metagenomics human microbiome"


def test_fetch_openalex_works_for_query_paginates_until_enough_complete_records() -> None:
    payloads = [
        {
            "results": [
                {
                    "id": "https://openalex.org/WPAGE1",
                    "doi": "https://doi.org/10.1000/page.1",
                    "display_name": "Page one",
                    "publication_year": 2001,
                },
                {
                    "id": "https://openalex.org/WPAGE2",
                    "doi": "https://doi.org/10.1000/page.2",
                    "display_name": "Page two",
                    "publication_year": 2002,
                },
            ],
            "meta": {"next_cursor": "cursor-two"},
        },
        {
            "results": [
                {
                    "id": "https://openalex.org/WPAGE3",
                    "doi": "https://doi.org/10.1000/page.3",
                    "display_name": "Page three",
                    "publication_year": 2003,
                }
            ],
            "meta": {},
        },
    ]
    calls = []

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout=60):
        calls.append(request.full_url)
        return FakeResponse(payloads[len(calls) - 1])

    original = publication_corpus_v2.urllib.request.urlopen
    publication_corpus_v2.urllib.request.urlopen = fake_urlopen
    try:
        works = publication_corpus_v2.fetch_openalex_works_for_query("test query", max_records=3)
    finally:
        publication_corpus_v2.urllib.request.urlopen = original

    assert [work["id"] for work in works] == [
        "https://openalex.org/WPAGE1",
        "https://openalex.org/WPAGE2",
        "https://openalex.org/WPAGE3",
    ]
    assert len(calls) == 2
    assert "cursor=cursor-two" in calls[1]


def test_publication_corpus_v2_cli_uses_cached_topup_work_records() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_topup_cli_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "topup"
        records_path = root / "topup.jsonl"
        source_dir.mkdir()
        _write_near_threshold_corpus(source_dir)
        records_path.write_text(json.dumps(_fake_topup_record()) + "\n", encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "topup-openalex-works",
                "--source-corpus-dir",
                str(source_dir),
                "--target-corpus-dir",
                str(target_dir),
                "--topup-records-jsonl",
                str(records_path),
                "--domains",
                "near_threshold_field",
                "--target-papers",
                "8",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        assert (target_dir / "openalex_topup_manifest.json").exists()
        works = pd.read_csv(target_dir / "works.csv")
        assert (works["id"] == "https://openalex.org/WTOPUP1").any()


def test_topup_records_fetch_uses_max_extra_even_when_one_paper_short() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_topup_fetch_") as tmp:
        root = Path(tmp)
        _write_near_threshold_corpus(root)
        captured = []

        def fake_fetch(query: object, max_records: int, timeout_seconds: int = 60) -> list[dict]:
            captured.append(max_records)
            return [_fake_topup_record()["work"]]

        original = publication_corpus_v2.fetch_openalex_works_for_query
        publication_corpus_v2.fetch_openalex_works_for_query = fake_fetch
        try:
            records = publication_corpus_v2._topup_records_from_openalex(
                root,
                ["near_threshold_field"],
                target_papers=8,
                max_extra_per_domain=17,
            )
        finally:
            publication_corpus_v2.fetch_openalex_works_for_query = original

        assert captured == [17]
        assert len(records) == 1


def test_deduplicate_domain_dois_keeps_landmark_duplicate_and_filters_dropped_sources() -> None:
    works = pd.DataFrame(
        [
            {
                "id": "https://openalex.org/WKEEP",
                "doi": "10.1000/dup",
                "domain": "dedup_field",
                "title": "Landmark duplicate",
                "year": 2000,
                "is_landmark": 1,
                "cited_by_count": 10,
            },
            {
                "id": "https://openalex.org/WDROP",
                "doi": "10.1000/dup",
                "domain": "dedup_field",
                "title": "Non-landmark duplicate",
                "year": 2001,
                "is_landmark": 0,
                "cited_by_count": 100,
            },
            {
                "id": "https://openalex.org/WOTHER",
                "doi": "10.1000/other",
                "domain": "dedup_field",
                "title": "Other",
                "year": 2002,
                "is_landmark": 0,
                "cited_by_count": 1,
            },
        ]
    )
    citations = pd.DataFrame(
        [
            {"source": "https://openalex.org/WDROP", "target": "https://openalex.org/WOTHER"},
            {"source": "https://openalex.org/WOTHER", "target": "https://openalex.org/WDROP"},
        ]
    )

    dedup_works, dedup_citations, report = deduplicate_domain_dois(works, citations)

    assert dedup_works["id"].tolist() == ["https://openalex.org/WKEEP", "https://openalex.org/WOTHER"]
    assert dedup_citations["source"].tolist() == ["https://openalex.org/WOTHER"]
    assert dedup_citations["target"].tolist() == ["https://openalex.org/WDROP"]
    assert report["dropped_duplicate_works"] == 1


def test_publication_corpus_v2_cli_dedupes_domain_dois() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_dedup_cli_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "dedup"
        source_dir.mkdir()
        _write_minimal_corpus(source_dir)
        works = pd.read_csv(source_dir / "works.csv")
        duplicate = works.iloc[[0]].copy()
        duplicate["id"] = "https://openalex.org/WGOOD_DUP"
        duplicate["is_landmark"] = 0
        works = pd.concat([works, duplicate], ignore_index=True)
        works.to_csv(source_dir / "works.csv", index=False)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "dedupe-dois",
                "--source-corpus-dir",
                str(source_dir),
                "--target-corpus-dir",
                str(target_dir),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        assert (target_dir / "doi_dedupe_manifest.json").exists()
        deduped = pd.read_csv(target_dir / "works.csv")
        assert len(deduped) == len(works) - 1


def test_publication_corpus_v2_cli_writes_seed_bundle() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_cli_") as tmp:
        root = Path(tmp)
        corpus_dir = root / "corpus"
        out_dir = root / "out"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "diagnose",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
                "--min-papers",
                "8",
                "--topic-coverage-target",
                "0.80",
                "--reference-closure-target",
                "0.70",
                "--min-controls-per-landmark",
                "2",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        assert (out_dir / "domain_diagnostics.csv").exists()
        assert (out_dir / "candidate_domains.csv").exists()
        assert (out_dir / "domain_status.csv").exists()
        assert (out_dir / "domain_inclusion_table.csv").exists()
        assert (out_dir / "publication_target_domains.json").exists()
        assert (out_dir / "landmark_registry_v2_seed.csv").exists()
        assert (out_dir / "rebuild_queue.csv").exists()
        assert (out_dir / "v2_publication_seed_manifest.json").exists()
        candidates = pd.read_csv(out_dir / "candidate_domains.csv")
        status = pd.read_csv(out_dir / "domain_status.csv")
        clean_landmarks = pd.read_csv(out_dir / "landmark_registry_v2_seed.csv")
        rebuild_queue = pd.read_csv(out_dir / "rebuild_queue.csv")
        manifest = json.loads((out_dir / "v2_publication_seed_manifest.json").read_text())
        assert candidates["domain"].tolist() == ["high_quality_field"]
        assert "status" in status.columns
        assert clean_landmarks["label"].tolist() == ["Good 2002", "Good 2007"]
        assert rebuild_queue["domain"].tolist() == ["noisy_legacy_field"]
        assert "legacy_landmark_inflation" in rebuild_queue["failure_reasons"].iloc[0]
        assert manifest["figure_logic_policy"] == "fixed_consumer_contract"
        assert "domain_status.csv" in manifest["outputs"]


def test_expected_figure_view_contract_is_documented() -> None:
    assert FIGURE_LOGIC_POLICY == "fixed_consumer_contract"
    assert set(FIGURE_VIEW_CONTRACT["fig1"]) >= {"works.csv", "citations.csv"}
    assert set(FIGURE_VIEW_CONTRACT["fig3"]) >= {"works.csv", "citations.csv"}


def test_build_domain_status_table_creates_main_ready_and_repair_statuses() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_status_") as tmp:
        corpus_dir = Path(tmp) / "corpus"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        diagnostics = build_domain_diagnostics(
            corpus_dir=corpus_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        fig3_ready = build_fig3_readiness_diagnostics(
            corpus_dir=corpus_dir,
            analysis_end_year=2020,
            tau=10,
            min_refs=1,
            min_metric_papers=8,
            min_metric_landmarks=2,
        )
        clean_landmarks = clean_landmark_registry(pd.read_csv(corpus_dir / "landmarks.csv"))

        status = build_domain_status_table(diagnostics, fig3_ready, clean_landmarks)
        roster = build_publication_target_roster(status, top_domains=12)

        high = status[status["domain_id"] == "high_quality_field"].iloc[0]
        noisy = status[status["domain_id"] == "noisy_legacy_field"].iloc[0]
        assert high["status"] == "main_ready"
        assert high["family"] == "biology_biomedicine"
        assert noisy["status"] == "repair_landmark"
        assert roster["figure_logic_policy"] == "fixed_consumer_contract"
        assert [row["domain_id"] for row in roster["domains"]] == ["high_quality_field"]


def test_publication_corpus_v2_cli_excludes_main_candidate_domains() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_exclude_cli_") as tmp:
        root = Path(tmp)
        corpus_dir = root / "corpus"
        out_dir = root / "out"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "diagnose",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
                "--min-papers",
                "8",
                "--topic-coverage-target",
                "0.80",
                "--reference-closure-target",
                "0.70",
                "--min-controls-per-landmark",
                "2",
                "--exclude-domains",
                "high_quality_field",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        candidates = pd.read_csv(out_dir / "candidate_domains.csv")
        excluded = pd.read_csv(out_dir / "excluded_candidate_domains.csv")
        clean_landmarks = pd.read_csv(out_dir / "landmark_registry_v2_seed.csv")
        manifest = json.loads((out_dir / "v2_publication_seed_manifest.json").read_text())

        assert candidates.empty
        assert excluded["domain"].tolist() == ["high_quality_field"]
        assert clean_landmarks.empty
        assert manifest["excluded_domains"] == ["high_quality_field"]
        assert manifest["n_candidate_domains"] == 0
        assert manifest["n_excluded_candidate_domains"] == 1


def test_publication_corpus_v2_cli_excludes_domain_prefixes() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_exclude_prefix_cli_") as tmp:
        root = Path(tmp)
        corpus_dir = root / "corpus"
        out_dir = root / "out"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "diagnose",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
                "--min-papers",
                "8",
                "--topic-coverage-target",
                "0.80",
                "--reference-closure-target",
                "0.70",
                "--min-controls-per-landmark",
                "2",
                "--exclude-domain-prefix",
                "high",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        candidates = pd.read_csv(out_dir / "candidate_domains.csv")
        excluded = pd.read_csv(out_dir / "excluded_candidate_domains.csv")
        manifest = json.loads((out_dir / "v2_publication_seed_manifest.json").read_text())
        assert candidates.empty
        assert excluded["domain"].tolist() == ["high_quality_field"]
        assert manifest["excluded_domains"] == ["high_quality_field"]
        assert manifest["thresholds"]["excluded_domain_prefixes"] == ["high"]


def test_publication_corpus_v2_cli_excludes_domain_families() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_exclude_family_cli_") as tmp:
        root = Path(tmp)
        corpus_dir = root / "corpus"
        out_dir = root / "out"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "diagnose",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
                "--min-papers",
                "8",
                "--topic-coverage-target",
                "0.80",
                "--reference-closure-target",
                "0.70",
                "--min-controls-per-landmark",
                "2",
                "--exclude-family",
                "biology_biomedicine",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        candidates = pd.read_csv(out_dir / "candidate_domains.csv")
        status = pd.read_csv(out_dir / "domain_status.csv")
        excluded = pd.read_csv(out_dir / "excluded_candidate_domains.csv")
        manifest = json.loads((out_dir / "v2_publication_seed_manifest.json").read_text())
        high = status[status["domain_id"] == "high_quality_field"].iloc[0]
        assert candidates.empty
        assert excluded["domain"].tolist() == ["high_quality_field"]
        assert high["status"] == "drop"
        assert high["reason_for_inclusion_or_exclusion"] == "excluded_family"
        assert manifest["excluded_families"] == ["biology_biomedicine"]


def test_publication_corpus_v2_cli_writes_target_roster_path() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_roster_cli_") as tmp:
        root = Path(tmp)
        corpus_dir = root / "corpus"
        out_dir = root / "out"
        roster_path = root / "data" / "publication_target_domains.json"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "diagnose",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
                "--min-papers",
                "8",
                "--topic-coverage-target",
                "0.80",
                "--reference-closure-target",
                "0.70",
                "--min-controls-per-landmark",
                "2",
                "--require-fig3-ready",
                "--fig3-analysis-end-year",
                "2020",
                "--fig3-tau",
                "10",
                "--fig3-min-refs",
                "1",
                "--fig3-min-metric-papers",
                "8",
                "--fig3-min-metric-landmarks",
                "2",
                "--target-roster-path",
                str(roster_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        out_roster = json.loads((out_dir / "publication_target_domains.json").read_text(encoding="utf-8"))
        assert roster == out_roster
        assert roster["n_domains"] == 1
        assert roster["domains"][0]["domain_id"] == "high_quality_field"


def test_build_fig3_readiness_diagnostics_counts_eligible_metric_landmarks() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_fig3_ready_") as tmp:
        corpus_dir = Path(tmp) / "corpus"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        ready = build_fig3_readiness_diagnostics(
            corpus_dir=corpus_dir,
            analysis_end_year=2020,
            tau=10,
            min_refs=1,
            min_metric_papers=8,
            min_metric_landmarks=2,
        )
        high = ready[ready["domain"] == "high_quality_field"].iloc[0]

        assert int(high["fig3_cutoff_year"]) == 2010
        assert int(high["fig3_eligible_metric_papers"]) == 10
        assert int(high["fig3_eligible_metric_landmarks"]) == 2
        assert bool(high["fig3_ready"])
        assert high["fig3_readiness_failures"] == ""


def test_build_fig3_readiness_diagnostics_requires_local_citation_rows() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_fig3_refs_") as tmp:
        corpus_dir = Path(tmp) / "corpus"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)
        pd.DataFrame(columns=["source", "target", "relation"]).to_csv(corpus_dir / "citations.csv", index=False)

        ready = build_fig3_readiness_diagnostics(
            corpus_dir=corpus_dir,
            analysis_end_year=2020,
            tau=10,
            min_refs=1,
            min_metric_papers=1,
            min_metric_landmarks=1,
        )
        high = ready[ready["domain"] == "high_quality_field"].iloc[0]

        assert int(high["fig3_eligible_metric_papers"]) == 0
        assert int(high["fig3_eligible_metric_landmarks"]) == 0
        assert not bool(high["fig3_ready"])
        assert "fig3_metric_papers" in high["fig3_readiness_failures"]


def test_publication_corpus_v2_cli_can_require_fig3_ready_domains() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_fig3_ready_cli_") as tmp:
        root = Path(tmp)
        corpus_dir = root / "corpus"
        out_dir = root / "out"
        corpus_dir.mkdir()
        _write_minimal_corpus(corpus_dir)

        subprocess.run(
            [
                sys.executable,
                "scripts/publication_corpus_v2.py",
                "diagnose",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
                "--min-papers",
                "8",
                "--topic-coverage-target",
                "0.80",
                "--reference-closure-target",
                "0.70",
                "--min-controls-per-landmark",
                "2",
                "--require-fig3-ready",
                "--fig3-analysis-end-year",
                "2020",
                "--fig3-tau",
                "10",
                "--fig3-min-refs",
                "1",
                "--fig3-min-metric-papers",
                "20",
                "--fig3-min-metric-landmarks",
                "2",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        candidates = pd.read_csv(out_dir / "candidate_domains.csv")
        fig3_ready = pd.read_csv(out_dir / "fig3_readiness_diagnostics.csv")
        high = fig3_ready[fig3_ready["domain"] == "high_quality_field"].iloc[0]
        manifest = json.loads((out_dir / "v2_publication_seed_manifest.json").read_text())

        assert candidates.empty
        assert not bool(high["fig3_ready"])
        assert "fig3_metric_papers" in high["fig3_readiness_failures"]
        assert manifest["require_fig3_ready"] is True


def test_repair_corpus_metadata_fills_topic_labels_and_cleans_legacy_landmarks() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_metadata_repair_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        target_dir = root / "repaired"
        source_dir.mkdir()
        _write_minimal_corpus(source_dir)

        works = pd.read_csv(source_dir / "works.csv")
        works.loc[works["domain"].eq("high_quality_field"), "display_topic_label"] = ""
        works.to_csv(source_dir / "works.csv", index=False)

        landmarks = pd.read_csv(source_dir / "landmarks.csv")
        noisy = pd.DataFrame(
            [
                {
                    "domain": "high_quality_field",
                    "landmark_source": "fig1_anchor",
                    "source_id": "high_quality_field:nan",
                    "label": "",
                    "id": "https://openalex.org/WGOOD0",
                    "doi": "10.1000/good.0",
                    "title": "High quality field paper 0",
                    "year": 2000,
                    "match_confidence": 1.0,
                    "include_main": 1,
                }
                for _ in range(4)
            ]
        )
        pd.concat([landmarks, noisy], ignore_index=True).to_csv(source_dir / "landmarks.csv", index=False)

        before = build_domain_diagnostics(
            corpus_dir=source_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        before_high = before[before["domain"] == "high_quality_field"].iloc[0]
        assert "topic_coverage" in before_high["failure_reasons"]
        assert "legacy_landmark_inflation" in before_high["failure_reasons"]

        manifest = repair_corpus_metadata(
            source_corpus_dir=source_dir,
            target_corpus_dir=target_dir,
            domains=["high_quality_field"],
        )
        after = build_domain_diagnostics(
            corpus_dir=target_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        after_high = after[after["domain"] == "high_quality_field"].iloc[0]
        repaired_works = pd.read_csv(target_dir / "works.csv")
        repaired_landmarks = pd.read_csv(target_dir / "landmarks.csv")

        assert manifest["n_topic_labels_filled"] == 12
        assert manifest["n_landmarks_input"] > manifest["n_landmarks_output"]
        assert repaired_works[repaired_works["domain"].eq("high_quality_field")]["display_topic_label"].ne("").all()
        assert repaired_landmarks["label"].fillna("").astype(str).str.strip().ne("").all()
        assert after_high["recommended_role"] == "main_candidate"


def test_materialize_candidate_corpus_writes_strict_views_and_preserves_closure_evidence() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_pub_v2_materialize_") as tmp:
        root = Path(tmp)
        source_dir = root / "source"
        seed_dir = root / "seed"
        target_dir = root / "v2_publication"
        source_dir.mkdir()
        seed_dir.mkdir()
        _write_minimal_corpus(source_dir)

        citations = pd.read_csv(source_dir / "citations.csv")
        citations.loc[citations["source"].astype(str).eq("https://openalex.org/WGOOD3"), "target"] = (
            "https://openalex.org/WOUTSIDE"
        )
        citations.to_csv(source_dir / "citations.csv", index=False)

        diagnostics = build_domain_diagnostics(
            corpus_dir=source_dir,
            min_papers=8,
            topic_coverage_target=0.80,
            duplicate_doi_max=0.015,
            reference_closure_target=0.70,
            min_controls_per_landmark=2,
        )
        diagnostics.to_csv(seed_dir / "domain_diagnostics.csv", index=False)
        diagnostics[diagnostics["recommended_role"] == "main_candidate"].to_csv(
            seed_dir / "candidate_domains.csv",
            index=False,
        )
        clean_landmark_registry(pd.read_csv(source_dir / "landmarks.csv")).to_csv(
            seed_dir / "landmark_registry_v2_seed.csv",
            index=False,
        )

        manifest = materialize_candidate_corpus(
            source_corpus_dir=source_dir,
            seed_dir=seed_dir,
            target_corpus_dir=target_dir,
            min_papers_per_domain=8,
        )

        assert manifest["n_domains"] == 1
        assert manifest["domains"] == ["high_quality_field"]
        assert (target_dir / "works.csv").exists()
        assert (target_dir / "views" / "fig3" / "multi_domain" / "works.csv").exists()
        assert (target_dir / "strict_view_audit.json").exists()

        works = pd.read_csv(target_dir / "works.csv")
        landmarks = pd.read_csv(target_dir / "landmarks.csv")
        root_citations = pd.read_csv(target_dir / "citations.csv")
        view_citations = pd.read_csv(target_dir / "views" / "fig3" / "high_quality_field" / "citations.csv")

        assert works["domain"].unique().tolist() == ["high_quality_field"]
        assert int(works["is_landmark"].sum()) == 2
        assert landmarks["label"].tolist() == ["Good 2002", "Good 2007"]
        assert "https://openalex.org/WOUTSIDE" in root_citations["target"].astype(str).tolist()
        assert "https://openalex.org/WOUTSIDE" not in view_citations["target"].astype(str).tolist()
