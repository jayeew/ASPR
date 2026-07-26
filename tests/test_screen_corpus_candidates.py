from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fig03.old.screen_corpus_candidates import (  # noqa: E402
    annotate_citations,
    annotate_reliable_anchors,
    complete_works,
    event_metrics_for_anchor,
    parse_fig3_run,
)


def _work_rows(domain: str, years: List[int]) -> pd.DataFrame:
    rows = []
    for idx, year in enumerate(years):
        rows.append(
            {
                "id": f"https://openalex.org/W{domain.replace('_', '')}{idx}",
                "doi": f"10.0000/{domain}.{idx}",
                "title": f"{domain} paper {idx}",
                "year": year,
                "domain": domain,
                "primary_field": "Biology",
                "display_community": 1 if year < 2005 else 2 + (idx % 2),
                "display_topic_id": f"T{idx % 3}",
                "display_topic_label": f"topic {idx % 3}",
                "is_landmark": 0,
                "anchor_label": "",
                "document_type": "article",
                "cited_by_count": 1000 - idx,
                "reference_count": 20 + idx,
                "source_provider": "test",
                "partial_2026": 0,
            }
        )
    return pd.DataFrame(rows)


def test_noisy_landmark_flags_require_labeled_match() -> None:
    works = _work_rows("noisy_domain", list(range(2000, 2008)))
    works["is_landmark"] = 1
    landmarks = works[["domain", "id", "doi", "title", "year"]].copy()
    landmarks["label"] = ""
    landmarks.loc[0, "label"] = "Canonical 2000"
    landmarks["include_main"] = 1

    annotated = annotate_reliable_anchors(works, landmarks, complete_end_year=2025, noisy_ratio=0.25)

    assert annotated["domain_anchor_flags_noisy"].all()
    assert int(annotated["reliable_anchor"].sum()) == 1
    assert int(annotated.loc[0, "reliable_anchor"]) == 1
    assert annotated.loc[0, "reliable_anchor_source"] == "landmarks_csv_labeled"


def test_complete_works_excludes_2026_partial_rows() -> None:
    works = _work_rows("partial_domain", [2024, 2025, 2026])
    works.loc[2, "partial_2026"] = 1

    complete = complete_works(works, complete_end_year=2025)

    assert complete["year"].tolist() == [2024, 2025]
    assert not complete["partial_2026"].astype(int).any()


def test_event_proxy_detects_topic_and_citation_shock() -> None:
    works = _work_rows("shock_domain", list(range(2000, 2011)))
    works.loc[5, "anchor_label"] = "Shock 2005"
    works.loc[5, "is_landmark"] = 1
    works.loc[5, "cited_by_count"] = 5000
    landmark = pd.DataFrame(
        [
            {
                "domain": "shock_domain",
                "id": works.loc[5, "id"],
                "doi": works.loc[5, "doi"],
                "title": works.loc[5, "title"],
                "year": 2005,
                "label": "Shock 2005",
                "include_main": 1,
            }
        ]
    )
    annotated = annotate_reliable_anchors(works, landmark, complete_end_year=2025)
    citations = pd.DataFrame(
        [
            {"source": works.loc[idx, "id"], "target": works.loc[5, "id"]}
            for idx in range(6, 11)
        ]
        + [
            {"source": works.loc[idx, "id"], "target": works.loc[idx - 1, "id"]}
            for idx in range(1, 11)
        ]
    )
    annotated_citations = annotate_citations(citations, annotated)
    metrics = event_metrics_for_anchor(annotated.loc[5].to_dict(), annotated, annotated_citations, window=5)

    assert metrics["pre_papers"] == 5
    assert metrics["post_papers"] == 5
    assert metrics["topic_shift"] > 0.0
    assert metrics["anchor_citers"] == 5
    assert metrics["event_proxy_raw"] > 0.0


def test_proxy_cli_writes_nonempty_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus_dir = root / "corpus"
        out_dir = root / "out"
        corpus_dir.mkdir()

        works = pd.concat(
            [
                _work_rows("domain_one", list(range(2000, 2012))),
                _work_rows("domain_two", list(range(2000, 2012))),
            ],
            ignore_index=True,
        )
        works.loc[works.groupby("domain").head(1).index, "is_landmark"] = 1
        works.loc[works.groupby("domain").head(1).index, "anchor_label"] = ["One 2000", "Two 2000"]
        works.to_csv(corpus_dir / "works.csv", index=False)

        landmarks = works.groupby("domain").head(1)[["domain", "id", "doi", "title", "year", "anchor_label"]].copy()
        landmarks = landmarks.rename(columns={"anchor_label": "label"})
        landmarks["include_main"] = 1
        landmarks.to_csv(corpus_dir / "landmarks.csv", index=False)

        citations = []
        for _, group in works.groupby("domain"):
            ids = group["id"].tolist()
            anchor_id = ids[0]
            citations.extend({"source": src, "target": anchor_id} for src in ids[1:])
            citations.extend({"source": ids[i], "target": ids[i - 1]} for i in range(1, len(ids)))
        pd.DataFrame(citations).to_csv(corpus_dir / "citations.csv", index=False)

        pd.DataFrame(
            [
                {"slug": "domain_one", "display_name": "Domain One", "field_name": "Biology"},
                {"slug": "domain_two", "display_name": "Domain Two", "field_name": "Physics"},
            ]
        ).to_csv(corpus_dir / "domains.csv", index=False)
        pd.DataFrame(columns=["id", "domain"]).to_csv(corpus_dir / "topics.csv", index=False)
        (corpus_dir / "quality_report.json").write_text(
            json.dumps(
                {
                    "domains": [
                        {"domain": "domain_one", "passes": True, "topic_coverage": 1.0},
                        {"domain": "domain_two", "passes": True, "topic_coverage": 1.0},
                    ]
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "experiments/fig03/old/screen_corpus_candidates.py",
                "--stage",
                "proxy",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
                "--candidate-set-sizes",
                "2",
                "--top-domains",
                "2",
                "--beam-width",
                "2",
                "--top-papers-per-domain",
                "2",
                "--min-domain-papers",
                "5",
                "--min-citations-per-work",
                "0",
                "--quiet",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        assert not pd.read_csv(out_dir / "domain_proxy_scores.csv").empty
        assert not pd.read_csv(out_dir / "paper_event_candidates.csv").empty
        assert not pd.read_csv(out_dir / "candidate_domain_sets.csv").empty
        assert not pd.read_csv(out_dir / "recommended_domain_sets.csv").empty


def test_parse_fig3_run_reads_existing_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "fig3_diagnostics_summary.json").write_text(
            json.dumps(
                {
                    "learned_oof_spearman": 0.24,
                    "score_iqr": 0.42,
                    "overall_pass": True,
                    "status_label": "pass",
                    "n_contributing_graph_deltas": 5,
                    "active_delta_z_cap_hit_rate_max": 0.02,
                    "mean_delta_reliability": 0.31,
                    "delta_variant": "matched_control_v3",
                    "data_profile": {
                        "total_papers": 5000,
                        "min_papers_per_domain": 500,
                        "relaxed_control_tier_rate_max_by_domain": 0.40,
                    },
                    "checks": {"oof_spearman": 1, "contributing_graph_deltas": 1},
                    "data_checks": {"total_papers": 1, "papers_per_domain": 1},
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"model": "equal_weights", "oof_spearman": 0.15},
                {"model": "best_single_indicator", "oof_spearman": 0.20},
            ]
        ).to_csv(run_dir / "fig3_baseline_comparison.csv", index=False)
        pd.DataFrame({"rgpm_matched_percentile": [80.0], "sw_matched_percentile": [90.0]}).to_csv(
            run_dir / "fig3_landmark_validation.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"stat": "top_vs_bottom_score_decile_rgpm_top20_enrichment", "value": 4.5},
                {"stat": "high_vs_low_tertile_median_rgpm_lift_pp", "value": 25.0},
            ]
        ).to_csv(run_dir / "fig3_effect_summary.csv", index=False)

        parsed = parse_fig3_run(
            run_dir,
            "set02_rank01",
            {
                "mean_data_quality_score": 0.8,
                "max_event_proxy": 0.7,
                "category_diversity_score": 0.5,
            },
        )

        assert parsed["candidate_id"] == "set02_rank01"
        assert parsed["learned_oof_spearman"] == 0.24
        assert round(parsed["delta_vs_equal"], 2) == 0.09
        assert round(parsed["landmark_percentile_mean"], 2) == 0.85
        assert parsed["n_contributing_graph_deltas"] == 5
        assert parsed["total_effective_papers"] == 5000
        assert parsed["top20_enrichment"] == 4.5
        assert parsed["fig3_screening_score"] > 0.0


if __name__ == "__main__":
    test_noisy_landmark_flags_require_labeled_match()
    test_complete_works_excludes_2026_partial_rows()
    test_event_proxy_detects_topic_and_citation_shock()
    test_proxy_cli_writes_nonempty_outputs()
    test_parse_fig3_run_reads_existing_outputs()
    print("screen_corpus_candidates tests passed")
