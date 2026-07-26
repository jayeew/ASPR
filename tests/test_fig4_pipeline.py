from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fig04.old.main_fig4 import (  # noqa: E402
    Fig4ArgsForAgent,
    audit_markdown_inputs,
    build_fig3_weighted_prior_rows,
    build_paper_dossier,
    candidate_records_for_peer_aspect,
    coerce_semantic_match_refinement_payload,
    controlled_sample,
    draw_fig4,
    filter_prior_art_candidates,
    heuristic_structured_consistency,
    load_fig3_weight_config,
    load_fig3_weights,
    normalize_innovation_label_payload,
    parse_article_markdown,
    parse_peer_review_markdown,
    read_csv_records,
    run_aspr_agent_for_row,
    run_graph_prior_stage,
    run_metrics_stage,
    screen_peer_review_label,
    semantic_match_one_point,
    write_csv,
    write_json,
    write_jsonl,
)
from experiments.fig04.old.draw_fig4_publication_summary import (  # noqa: E402
    draw_figure as draw_publication_summary,
)


ARTICLE_MD = """
Article

https://doi.org/10.1038/s41467-023-35783-y

## Bottom-up evolution of perovskite clusters into high-activity rhodium nanoparticles toward alkaline hydrogen evolution

Received: 8 April 2022 Accepted: 2 January 2023

Gaoxin Lin, Zhuang Zhang and colleagues

Self-reconstruction has been considered an efficient means to prepare efficient electrocatalysts in various energy transformation process for bond activation and breaking. However, developing nano-sized electrocatalysts through complete in-situ reconstruction with improved activity remains challenging. Herein, we report a bottom-up evolution route of electrochemically reducing Cs3Rh2I9 halide-perovskite clusters on N-doped carbon to prepare ultrafine Rh nanoparticles with large lattice spacings and grain boundaries. Various in-situ and ex-situ characterizations elucidate the Cs and I extraction and Rh reduction during the electrochemical reduction. These Rh nanoparticles show enhanced mass and area activity toward hydrogen evolution reaction.

Hydrogen, as an energy carrier, is critical to utilize renewable energy.
"""


REVIEW_MD = """
## Peer Review File

Open Access This file is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

## REVIEWER COMMENTS

Reviewer #1 (Remarks to the Author):

The paper is significant and novel because it proposes a bottom-up route to generate active rhodium nanoparticles. The evidence is promising, but the rigor of grain-boundary characterization needs improvement. The authors should address limitations in microscopy evidence and compare against additional benchmarks.

## Author Response

We thank the reviewer for the helpful comments and have revised the manuscript.
"""


def test_article_parser_extracts_doi_title_abstract_and_year() -> None:
    parsed = parse_article_markdown(ARTICLE_MD, "s41467-023-35783-y")

    assert parsed["doi"] == "10.1038/s41467-023-35783-y"
    assert parsed["year"] == 2023
    assert parsed["title"].startswith("Bottom-up evolution")
    assert "bottom-up evolution route" in parsed["abstract"]
    assert parsed["word_count"] > 50


def test_article_parser_skips_generic_article_heading_and_author_list() -> None:
    text = """
https://doi.org/10.1038/s41467-023-35924-3

## Article

## Tautomerism unveils a self-inhibition mechanism of crystallization

Received: 14 November 2022 Accepted: 9 January 2023

Weiwei Tang 1,2, Taimin Yang 3, Cristian A. Morales-Rivera4, Xi Geng1, Vijay K. Srirambhatla[5,6] , Xiang Kang[2] , Vraj P. Chauhan[1] , Sungil Hong[4] , Qing Tu[7] , Alastair J. Florence 5,6, Huaping Mo8, Hector A. Calderon9,10 & Jeffrey D. Rimer 1

Modifiers are commonly used in natural, biological, and synthetic crystallization to tailor the growth of diverse materials. Here, we identify tautomers as a new class of modifiers where the dynamic interconversion between solute and its corresponding tautomer produces native crystal growth inhibitors. The macroscopic and microscopic effects imposed by inhibitor-crystal interactions reveal dual mechanisms of inhibition. These findings offer potential routes in crystal engineering to strategically control the mechanical or physicochemical properties of tautomeric materials.
"""
    parsed = parse_article_markdown(text, "s41467-023-35924-3")

    assert parsed["title"] == "Tautomerism unveils a self-inhibition mechanism of crystallization"
    assert parsed["abstract"].startswith("Modifiers are commonly used")
    assert "Weiwei Tang" not in parsed["abstract"]


def test_peer_review_parser_keeps_review_comments_and_excludes_author_response() -> None:
    parsed = parse_peer_review_markdown(REVIEW_MD)

    assert "The paper is significant and novel" in parsed["peer_review_text"]
    assert "We thank the reviewer" not in parsed["peer_review_text"]
    assert "author_response" in parsed["excluded_sections"]
    assert "reviewer_comments" in parsed["included_sections"]


def test_controlled_sample_caps_nature_communications() -> None:
    rows = []
    for idx in range(40):
        rows.append({"paper_id": f"nc_{idx}", "journal_id": "41467", "year": "2023", "included_in_audit": True})
    for idx in range(40):
        rows.append({"paper_id": f"sub_{idx}", "journal_id": "41564", "year": "2024", "included_in_audit": True})

    sampled = controlled_sample(rows, sample_size=20, seed=7, cap_41467=0.5)

    assert len(sampled) == 20
    assert sum(1 for row in sampled if row["journal_id"] == "41467") <= 10
    assert all(row["included_in_main"] for row in sampled)


def test_audit_markdown_inputs_writes_parsed_cache() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_audit_") as tmp:
        root = Path(tmp)
        markdown_root = root / "markdown"
        out_dir = root / "out"
        (markdown_root / "paper").mkdir(parents=True)
        (markdown_root / "peer_review").mkdir()
        (markdown_root / "paper" / "s41467-023-35783-y.md").write_text(ARTICLE_MD, encoding="utf-8")
        (markdown_root / "peer_review" / "s41467-023-35783-y_r.md").write_text(REVIEW_MD, encoding="utf-8")
        write_jsonl(
            markdown_root / "manifest.jsonl",
            [
                {
                    "article_id": "s41467-023-35783-y",
                    "year": 2023,
                    "article_pdf_path": "/tmp/article.pdf",
                    "peer_review_pdf_path": "/tmp/review.pdf",
                }
            ],
        )

        rows = audit_markdown_inputs(markdown_root, out_dir, journal_scope="all", quiet=True)

        assert len(rows) == 1
        assert rows[0]["included_in_audit"]
        assert Path(rows[0]["parsed_text_cache"]).exists()
        parsed = json.loads(Path(rows[0]["parsed_text_cache"]).read_text(encoding="utf-8"))
        assert "peer_review_text" in parsed
        dossier_path = Path(rows[0]["paper_dossier_cache"])
        assert dossier_path.exists()
        dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
        assert dossier["leakage_guard"] == "peer_review_text_excluded"
        assert "The paper is significant and novel" not in json.dumps(dossier)


def test_paper_dossier_excludes_peer_review_text() -> None:
    parsed = parse_article_markdown(ARTICLE_MD, "s41467-023-35783-y")
    dossier = build_paper_dossier(parsed, {"paper_id": "paper", "journal": "Nature Communications"})

    serialized = json.dumps(dossier, ensure_ascii=False)
    assert "bottom-up evolution route" in serialized
    assert "Reviewer #1" not in serialized
    assert "author response" not in serialized.lower()


def test_prior_art_filter_removes_target_and_future_papers() -> None:
    papers = [
        {"paperId": "target", "title": "Test Paper", "doi": "10.1038/test", "year": 2023},
        {"paperId": "future", "title": "Future related work", "doi": "10.1038/future", "year": 2025},
        {"paperId": "prior", "title": "Prior related work", "doi": "10.1038/prior", "year": 2020},
    ]
    kept, excluded = filter_prior_art_candidates(
        papers,
        target_title="Test Paper",
        target_doi="10.1038/test",
        cutoff_year=2023,
    )

    assert [paper["paperId"] for paper in kept] == ["prior"]
    reasons = {row["paperId"]: row["reasons"] for row in excluded}
    assert "target_doi_match" in reasons["target"]
    assert "target_title_match" in reasons["target"]
    assert "post_publication_year" in reasons["future"]


def test_innovation_label_normalization_requires_quotes() -> None:
    text = "The paper is significant and novel because it proposes a bottom-up route."
    payload = {
        "overall_innovation_stance": {
            "score_1_5": 4,
            "label": "positive",
            "quote": "significant and novel",
            "confidence": 0.9,
        },
        "aspects": {
            "novelty": {
                "score_1_5": 4,
                "points": ["bottom-up route"],
                "quotes": ["bottom-up route"],
                "confidence": 0.8,
            }
        },
    }

    normalized = normalize_innovation_label_payload(payload, "paper", "peer_review", text)
    assert normalized["success"]
    assert normalized["aspects"]["novelty"]["quotes"] == ["bottom-up route"]

    payload["aspects"]["novelty"]["quotes"] = []
    missing = normalize_innovation_label_payload(payload, "paper", "peer_review", text)
    assert not missing["success"]
    assert missing["failure_reason"] == "missing_required_quotes"


def test_point_records_require_exact_quotes_and_filter_revision_only_points() -> None:
    text = (
        "Reviewer #1: The work is novel because it introduces a bottom-up route. "
        "The authors have addressed all concerns and no further comments remain."
    )
    payload = {
        "overall_innovation_stance": {
            "score_1_5": 4,
            "label": "positive",
            "quote": "work is novel",
            "confidence": 0.9,
        },
        "aspects": {
            "novelty": {
                "score_1_5": 4,
                "point_records": [
                    {
                        "point_id": "n1",
                        "point": "The paper introduces a bottom-up route.",
                        "quote": "introduces a bottom-up route",
                        "polarity": "positive",
                        "evidence_type": "novelty_claim",
                        "confidence": 0.8,
                        "source_role": "reviewer",
                    },
                    {
                        "point_id": "n2",
                        "point": "The authors addressed all concerns.",
                        "quote": "addressed all concerns",
                        "polarity": "neutral",
                        "evidence_type": "novelty_claim",
                        "confidence": 0.8,
                        "source_role": "reviewer",
                    },
                    {
                        "point_id": "n3",
                        "point": "Invented novelty point.",
                        "quote": "not present in source",
                        "polarity": "positive",
                        "evidence_type": "novelty_claim",
                        "confidence": 0.8,
                        "source_role": "reviewer",
                    },
                ],
            }
        },
    }

    normalized = normalize_innovation_label_payload(payload, "paper", "peer_review", text)

    novelty = normalized["aspects"]["novelty"]
    assert normalized["success"]
    assert novelty["points"] == ["The paper introduces a bottom-up route."]
    assert novelty["quotes"] == ["introduces a bottom-up route"]
    assert len(novelty["point_records"]) == 1
    assert novelty["point_records"][0]["evidence_type"] == "novelty_claim"
    assert any("revision_only_point_dropped" in warning for warning in normalized["warnings"])
    assert any("point_record_quote_not_exact" in warning for warning in normalized["warnings"])


def test_peer_review_screen_requires_explicit_innovation_points() -> None:
    review = "The work is novel and significant. It compares well with prior art and raises useful evidence concerns."
    label = {
        "success": True,
        "overall_innovation_stance": {"score_1_5": 4, "quote": "novel and significant"},
        "aspects": {
            "novelty": {"score_1_5": 4, "points": ["novel method"], "quotes": ["novel"], "confidence": 0.9},
            "significance": {"score_1_5": 4, "points": ["significant result"], "quotes": ["significant"], "confidence": 0.9},
            "prior_art_comparison": {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0},
            "evidence_rigor": {"score_1_5": 3, "points": ["evidence concerns"], "quotes": ["evidence"], "confidence": 0.8},
            "limitations": {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0},
            "future_work": {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0},
        },
    }

    screen = screen_peer_review_label(label, review, min_core_aspects=2, min_peer_label_points=2)
    assert screen["screen_pass"]

    revision = screen_peer_review_label(label, "The authors have addressed all of my concerns. No further comments.", 2, 2)
    assert not revision["screen_pass"]
    assert "revision_only" in revision["screen_reason"]

    weak_label = json.loads(json.dumps(label))
    weak_label["overall_innovation_stance"]["score_1_5"] = None
    weak = screen_peer_review_label(weak_label, review, min_core_aspects=2, min_peer_label_points=2)
    assert not weak["screen_pass"]
    assert "missing_innovation_stance" in weak["screen_reason"]


def test_semantic_match_heuristic_relations() -> None:
    entailed = semantic_match_one_point(
        "T",
        "novelty",
        "bottom-up route",
        "bottom-up route",
        ["The paper proposes a bottom-up route."],
        client=None,
    )
    assert entailed["relation"] in {"entailed", "related"}
    no_match = semantic_match_one_point(
        "T",
        "future_work",
        "benchmark against RhCl3 route",
        "benchmark",
        [],
        client=None,
    )
    assert no_match["relation"] == "no_match"
    rhcl3 = semantic_match_one_point(
        "T",
        "prior_art_comparison",
        "Lack of benchmarking against traditional RhCl3 route for Rh nanoparticle synthesis.",
        "RhCl3 route",
        ["Explicitly contrasts the bottom-up approach with conventional top-down bulk reconstruction."],
        client=None,
    )
    assert rhcl3["relation"] in {"related", "no_match"}


def test_semantic_refinement_payload_requires_real_candidate() -> None:
    candidates = ["Agent discusses donor-specific safety assessment.", "Agent notes missing siRNA off-target controls."]
    valid = coerce_semantic_match_refinement_payload(
        {"relation": "related", "best_candidate_id": 2, "confidence": 0.8, "rationale": "same control concern"},
        candidates,
        min_confidence=0.55,
    )
    assert valid["relation"] == "related"
    assert valid["best_agent_point"] == candidates[1]

    invented = coerce_semantic_match_refinement_payload(
        {"relation": "related", "best_agent_point": "Invented candidate text.", "confidence": 0.9},
        candidates,
        min_confidence=0.55,
    )
    assert invented["relation"] == "no_match"
    assert invented["best_agent_point"] == ""

    low_confidence = coerce_semantic_match_refinement_payload(
        {"relation": "entailed", "best_candidate_id": 1, "confidence": 0.2},
        candidates,
        min_confidence=0.55,
    )
    assert low_confidence["relation"] == "no_match"


def test_cross_aspect_candidate_fallback_is_controlled() -> None:
    agent_label = {
        "aspects": {
            "limitations": {
                "points": ["The agent notes missing siRNA off-target controls."],
                "quotes": ["missing siRNA off-target controls"],
            },
            "future_work": {
                "points": ["Future work should test larger cohorts."],
                "quotes": ["larger cohorts"],
            },
            "significance": {
                "points": ["The finding is clinically important."],
                "quotes": ["clinically important"],
            },
        }
    }

    evidence_records = candidate_records_for_peer_aspect(
        agent_label,
        "evidence_rigor",
        "Reviewer asks for off-target controls.",
    )
    assert any(record["aspect"] == "limitations" for record in evidence_records)

    unrelated_records = candidate_records_for_peer_aspect(
        agent_label,
        "significance",
        "Reviewer says the result is important.",
    )
    assert all(record["aspect"] == "significance" for record in unrelated_records)

    future_gap_records = candidate_records_for_peer_aspect(
        agent_label,
        "limitations",
        "The reviewer says additional cohort testing is needed.",
    )
    assert any(record["aspect"] == "future_work" for record in future_gap_records)

    ordinary_limitation_records = candidate_records_for_peer_aspect(
        agent_label,
        "limitations",
        "The reviewer says the current sample is small.",
    )
    assert all(record["aspect"] != "future_work" for record in ordinary_limitation_records)


def test_prompt_calibration_rules_are_present() -> None:
    from aspr.prompts import (
        FINAL_INNOVATION_REPORT_PROMPT,
        INNOVATION_GENERATION_PROMPT,
        INNOVATION_REFLECTION_PROMPT,
    )

    assert "默认从“中等/不确定创新性”开始判断" in INNOVATION_GENERATION_PROMPT
    assert "clear prior-art contrast" in INNOVATION_GENERATION_PROMPT
    assert "Prior-art comparison" in INNOVATION_GENERATION_PROMPT
    assert "Evidence and rigor" in INNOVATION_GENERATION_PROMPT
    assert "overclaiming_check" in INNOVATION_REFLECTION_PROMPT
    assert "prior_art_section_check" in INNOVATION_REFLECTION_PROMPT
    assert "校准后的创新性立场" in FINAL_INNOVATION_REPORT_PROMPT
    assert "Limitations and uncertainty" in FINAL_INNOVATION_REPORT_PROMPT


def test_structured_consistency_heuristic_scores_and_overclaiming() -> None:
    peer_label = {
        "overall_innovation_stance": {"score_1_5": 3},
        "aspects": {
            "novelty": {"score_1_5": 3},
            "significance": {"score_1_5": 4},
        },
    }
    agent_label = {
        "overall_innovation_stance": {"score_1_5": 5},
        "aspects": {
            "novelty": {"score_1_5": 5},
            "significance": {"score_1_5": 4},
        },
    }

    payload = heuristic_structured_consistency(peer_label, agent_label)

    assert payload["stance_consistency_1_5"] == 3.0
    assert payload["novelty_consistency_1_5"] == 3.0
    assert payload["significance_consistency_1_5"] == 5.0
    assert payload["overclaiming_score_1_5"] == 3.0


def test_fig3_weights_load_and_normalize() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_weights_") as tmp:
        path = Path(tmp) / "weights.csv"
        write_csv(path, [{"metric": "DeltaQ0", "weight": 3}, {"metric": "PDE", "weight": 1}])
        weights = load_fig3_weights(path)

    assert round(sum(weights.values()), 6) == 1.0
    assert weights["DeltaQ0"] == 0.75
    assert weights["PDE"] == 0.25


def test_fig3_weight_config_missing_path_reports_warning() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_missing_weights_") as tmp:
        config = load_fig3_weight_config(Path(tmp) / "missing_weights.csv")

    assert round(sum(config["weights"].values()), 6) == 1.0
    assert config["warning"].startswith("missing_fig3_weights")
    assert config["weights_source"] == "equal_weight_fallback"


def test_fig3_weighted_prior_uses_z_style_scores_not_raw_product() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_sw_") as tmp:
        root = Path(tmp)
        weights_path = root / "fig3_best_weights.csv"
        reference_path = root / "fig3_publication_day_indicators.csv"
        write_csv(weights_path, [{"metric": "DeltaQ0", "weight": 1.0}])
        write_csv(
            reference_path,
            [
                {"paper_id": "r0", "DeltaQ0": 0.0},
                {"paper_id": "r1", "DeltaQ0": 1.0},
                {"paper_id": "r2", "DeltaQ0": 2.0},
                {"paper_id": "r3", "DeltaQ0": 3.0},
            ],
        )

        rows = build_fig3_weighted_prior_rows(
            [{"paper_id": "paper", "DeltaQ0": 2.5}],
            weights_path=weights_path,
            reference_indicators_path=reference_path,
            reference_scores_path=root / "missing_score_table.csv",
        )

    assert len(rows) == 1
    assert rows[0]["fig3_sw"] != 2.5
    assert rows[0]["fig3_sw_normalization"] == "fig3_reference_rank_normal"
    assert rows[0]["fig3_sw_percentile_source"] == "fig3_reference_distribution"
    assert float(rows[0]["fig3_sw_percentile"]) > 0.5
    assert rows[0]["fig3_sw_tier"] in {"middle", "high"}


def test_graph_prior_prompt_hides_individual_metric_values() -> None:
    from experiments.fig04.old.main_fig4 import graph_metric_prompt_block

    prompt = graph_metric_prompt_block(
        {
            "fig3_sw": 0.73,
            "fig3_sw_percentile": 0.82,
            "fig3_sw_tier": "high",
            "fig3_sw_quality_flag": "ok",
            "B": 0.11,
            "RS": 0.22,
            "DeltaQ0": 0.33,
            "Uzzi": 0.44,
            "RTD": 0.55,
            "BurtIP": 0.66,
            "PDE": 0.77,
        }
    )

    assert "Fig.3-weighted graph innovation prior" in prompt
    assert "0.730" in prompt
    assert "82%" in prompt
    assert "high" in prompt
    for metric in ["B", "RS", "DeltaQ0", "Uzzi", "RTD", "BurtIP", "PDE"]:
        assert metric not in prompt
    for value in ["0.110", "0.220", "0.330", "0.440", "0.550", "0.660", "0.770"]:
        assert value not in prompt


def test_graph_prior_stage_writes_one_row_per_manifest_entry() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_graph_prior_") as tmp:
        root = Path(tmp)
        weights_path = root / "fig3_best_weights.csv"
        reference_path = root / "fig3_publication_day_indicators.csv"
        write_csv(root / "fig4_manifest.csv", [{"paper_id": "paper_a"}, {"paper_id": "paper_b"}])
        write_csv(
            root / "fig4_graph_metrics.csv",
            [
                {"paper_id": "paper_a", "DeltaQ0": 0.25, "RTD": 0.75},
                {"paper_id": "paper_b", "DeltaQ0": 0.75, "RTD": 0.25},
            ],
        )
        write_csv(weights_path, [{"metric": "DeltaQ0", "weight": 1.0}])
        write_csv(
            reference_path,
            [
                {"paper_id": "r0", "DeltaQ0": 0.0},
                {"paper_id": "r1", "DeltaQ0": 0.5},
                {"paper_id": "r2", "DeltaQ0": 1.0},
            ],
        )

        rows = run_graph_prior_stage(
            root,
            weights_path=weights_path,
            reference_indicators_path=reference_path,
            reference_scores_path=root / "missing_score_table.csv",
            quiet=True,
        )

        assert len(rows) == 2
        assert {row["paper_id"] for row in rows} == {"paper_a", "paper_b"}
        assert all(row["fig3_weights_hash"] for row in rows)
        assert (root / "fig4_graph_prior.csv").exists()


def test_s2_keyed_403_falls_back_to_anonymous() -> None:
    import aspr.open_scholar as open_scholar

    class Args:
        s2_api_key = "bad-key"
        and_search = False

    class Response:
        def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self) -> dict:
            return self._payload

    calls = []
    original_get = open_scholar.requests.get

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append({"headers": headers or {}, "url": url})
        if headers and headers.get("x-api-key"):
            return Response(403, text='{"message":"Forbidden"}')
        return Response(200, payload={"data": [{"paperId": "p1", "title": "Prior", "year": 2020}]})

    open_scholar.requests.get = fake_get
    try:
        scholar = open_scholar.OpenScholar(Args())
        papers = scholar.search_semantic_scholar(["CRISPR"])
    finally:
        open_scholar.requests.get = original_get

    assert papers[0]["paperId"] == "p1"
    assert scholar.s2_key_status == "s2_key_rejected"
    assert scholar.last_retrieval_source == "semantic_scholar_anonymous"
    assert len(calls) == 2


def test_neural_retrieval_load_failure_falls_back_to_tfidf() -> None:
    import aspr.open_scholar as open_scholar

    refs = [
        "Paper A studies CRISPR gene editing with single cell sequencing.",
        "Paper B is about catalyst design and rhodium chloride synthesis.",
    ]
    original_get_recall = open_scholar._get_recall_model
    original_get_reranker = open_scholar._get_reranker
    original_recall_failure = open_scholar._RECALL_MODEL_FAILURE
    original_reranker_failure = open_scholar._RERANKER_MODEL_FAILURE
    original_reported = set(open_scholar._FALLBACK_MESSAGES_REPORTED)

    def failing_recall_model():
        raise OSError("no cached bge model")

    def failing_reranker_model():
        raise OSError("no cached reranker model")

    open_scholar._get_recall_model = failing_recall_model
    open_scholar._get_reranker = failing_reranker_model
    open_scholar._RECALL_MODEL_FAILURE = ""
    open_scholar._RERANKER_MODEL_FAILURE = ""
    open_scholar._FALLBACK_MESSAGES_REPORTED.clear()
    try:
        recalled, _ = open_scholar.retrieval_recall("CRISPR gene editing", refs)
        reranked, _ = open_scholar.retrieval_rerank("CRISPR gene editing", refs)
        status = open_scholar.retrieval_backend_status()
    finally:
        open_scholar._get_recall_model = original_get_recall
        open_scholar._get_reranker = original_get_reranker
        open_scholar._RECALL_MODEL_FAILURE = original_recall_failure
        open_scholar._RERANKER_MODEL_FAILURE = original_reranker_failure
        open_scholar._FALLBACK_MESSAGES_REPORTED.clear()
        open_scholar._FALLBACK_MESSAGES_REPORTED.update(original_reported)

    assert recalled[0].startswith("Paper A")
    assert reranked[0].startswith("Paper A")
    assert status["recall_backend"] == "tfidf_fallback"
    assert status["reranker_backend"] == "tfidf_fallback"
    assert "OSError" in status["recall_failure"]
    assert "OSError" in status["reranker_failure"]


def test_retrieval_recall_retries_smaller_batch_after_cuda_oom() -> None:
    import aspr.open_scholar as open_scholar

    class FakeRecallModel:
        def __init__(self) -> None:
            self.calls = []

        def compute_score(self, pairs, max_passage_length=None, weights_for_different_modes=None, batch_size=None):
            self.calls.append(
                {
                    "batch_size": batch_size,
                    "max_passage_length": max_passage_length,
                    "pair_count": len(pairs),
                }
            )
            if batch_size == 8:
                raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB.")
            return {"colbert+sparse+dense": [0.2, 0.8]}

    fake = FakeRecallModel()
    refs = ["weak reference", "strong CRISPR gene editing reference"]
    original_get_recall = open_scholar._get_recall_model
    original_env = {key: os.environ.get(key) for key in ["ASPR_RECALL_BATCH_SIZE", "ASPR_RECALL_RETRY_BATCHES"]}
    original_recall_failure = open_scholar._RECALL_MODEL_FAILURE
    try:
        open_scholar._get_recall_model = lambda: fake
        open_scholar._RECALL_MODEL_FAILURE = ""
        os.environ["ASPR_RECALL_BATCH_SIZE"] = "8"
        os.environ["ASPR_RECALL_RETRY_BATCHES"] = "8,4,2,1"
        ranked, _ = open_scholar.retrieval_recall("CRISPR gene editing", refs)
        status = open_scholar.retrieval_backend_status()
    finally:
        open_scholar._get_recall_model = original_get_recall
        open_scholar._RECALL_MODEL_FAILURE = original_recall_failure
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert ranked[0] == "strong CRISPR gene editing reference"
    assert [call["batch_size"] for call in fake.calls] == [8, 4]
    assert all(call["max_passage_length"] == 2048 for call in fake.calls)
    assert status["recall_backend"] == "bge_m3"
    assert status["recall_batch_size_used"] == "4"
    assert status["retrieval_failure_stage"] == ""


def test_retrieval_recall_all_oom_falls_back_to_tfidf() -> None:
    import aspr.open_scholar as open_scholar

    class AlwaysOOMRecallModel:
        def compute_score(self, pairs, max_passage_length=None, weights_for_different_modes=None, batch_size=None):
            raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB.")

    refs = ["Paper A studies CRISPR gene editing.", "Paper B studies unrelated catalysts."]
    original_get_recall = open_scholar._get_recall_model
    original_env = {key: os.environ.get(key) for key in ["ASPR_RECALL_BATCH_SIZE", "ASPR_RECALL_RETRY_BATCHES"]}
    original_recall_failure = open_scholar._RECALL_MODEL_FAILURE
    original_reported = set(open_scholar._FALLBACK_MESSAGES_REPORTED)
    try:
        open_scholar._get_recall_model = lambda: AlwaysOOMRecallModel()
        open_scholar._RECALL_MODEL_FAILURE = ""
        open_scholar._FALLBACK_MESSAGES_REPORTED.clear()
        os.environ["ASPR_RECALL_BATCH_SIZE"] = "8"
        os.environ["ASPR_RECALL_RETRY_BATCHES"] = "8,4"
        ranked, _ = open_scholar.retrieval_recall("CRISPR gene editing", refs)
        status = open_scholar.retrieval_backend_status()
    finally:
        open_scholar._get_recall_model = original_get_recall
        open_scholar._RECALL_MODEL_FAILURE = original_recall_failure
        open_scholar._FALLBACK_MESSAGES_REPORTED.clear()
        open_scholar._FALLBACK_MESSAGES_REPORTED.update(original_reported)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert ranked[0].startswith("Paper A")
    assert status["recall_backend"] == "tfidf_fallback"
    assert status["retrieval_failure_stage"] == "retrieval_recall_oom"
    assert "CUDA out of memory" in status["recall_failure"]


def test_agent_cache_reuse_does_not_import_or_run_aspr() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_agent_") as tmp:
        cache_dir = Path(tmp) / "cache" / "paper"
        cache_dir.mkdir(parents=True)
        write_json(
            cache_dir / "agent_eval.json",
            {
                "paper_id": "paper",
                "success": True,
                "innovation_evaluation": "cached evaluation",
                "agent_runtime_seconds": 1.2,
            },
        )

        result = run_aspr_agent_for_row(
            {"paper_id": "paper", "title": "T", "abstract": "A"},
            cache_dir,
            Fig4ArgsForAgent(s2_api_key="", and_search=False, top_n=10),
            reuse_agent=True,
        )

        assert result["cache_reused"]
        assert result["innovation_evaluation"] == "cached evaluation"


def test_metrics_same_text_similarity_and_coverage() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_metrics_") as tmp:
        out_dir = Path(tmp)
        cache_dir = out_dir / "cache" / "paper"
        cache_dir.mkdir(parents=True)
        write_csv(
            out_dir / "fig4_manifest.csv",
            [
                {
                    "paper_id": "paper",
                    "journal": "Nature Communications",
                    "journal_id": "41467",
                    "year": "2023",
                    "title": "Test paper",
                    "abstract": "Test abstract",
                    "parsed_text_cache": str(cache_dir / "parsed_text.json"),
                }
            ],
        )
        shared_text = "Novel method improves rigor and significance while noting limitations and future work."
        write_json(cache_dir / "parsed_text.json", {"peer_review_text": shared_text, "word_count": 12})
        write_jsonl(
            out_dir / "fig4_agent_outputs.jsonl",
            [
                {
                    "paper_id": "paper",
                    "success": True,
                    "innovation_evaluation": shared_text,
                    "agent_runtime_seconds": 60.0,
                }
            ],
        )
        aspects = {
            "significance": ["improves significance"],
            "novelty": ["novel method"],
            "rigor": ["improves rigor"],
            "limitations": ["noting limitations"],
            "future_work": ["future work"],
        }
        write_jsonl(
            out_dir / "fig4_rating_judgements.jsonl",
            [
                {
                    "paper_id": "paper",
                    "kind": "peer_review",
                    "success": True,
                    "overall_score_1_5": 4,
                    "novelty": 4,
                    "significance": 4,
                    "rigor": 4,
                    "limitations": 4,
                    "future_work": 4,
                    "aspects": aspects,
                },
                {
                    "paper_id": "paper",
                    "kind": "agent",
                    "success": True,
                    "overall_score_1_5": 4,
                    "novelty": 4,
                    "significance": 4,
                    "rigor": 4,
                    "limitations": 4,
                    "future_work": 4,
                    "aspects": aspects,
                },
            ],
        )
        label_aspects = {
            "novelty": {
                "score_1_5": 4,
                "points": ["novel method"],
                "quotes": ["Novel method"],
                "confidence": 0.9,
            },
            "significance": {
                "score_1_5": 4,
                "points": ["improves significance"],
                "quotes": ["significance"],
                "confidence": 0.9,
            },
            "prior_art_comparison": {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0},
            "evidence_rigor": {
                "score_1_5": 4,
                "points": ["improves rigor"],
                "quotes": ["rigor"],
                "confidence": 0.9,
            },
            "limitations": {
                "score_1_5": 4,
                "points": ["noting limitations"],
                "quotes": ["limitations"],
                "confidence": 0.9,
            },
            "future_work": {
                "score_1_5": 4,
                "points": ["future work"],
                "quotes": ["future work"],
                "confidence": 0.9,
            },
        }
        write_jsonl(
            out_dir / "fig4_innovation_label_judgements.jsonl",
            [
                {
                    "paper_id": "paper",
                    "kind": "peer_review",
                    "success": True,
                    "overall_innovation_stance": {
                        "score_1_5": 4,
                        "label": "positive",
                        "quote": "Novel method",
                        "confidence": 0.9,
                    },
                    "aspects": label_aspects,
                },
                {
                    "paper_id": "paper",
                    "kind": "agent",
                    "success": True,
                    "overall_innovation_stance": {
                        "score_1_5": 4,
                        "label": "positive",
                        "quote": "Novel method",
                        "confidence": 0.9,
                    },
                    "aspects": label_aspects,
                },
            ],
        )
        write_csv(
            out_dir / "fig4_peer_review_screen.csv",
            [
                {
                    "paper_id": "paper",
                    "screen_pass": True,
                    "screen_reason": "",
                    "core_aspect_count": 2,
                    "peer_point_count": 5,
                }
            ],
        )
        write_csv(
            out_dir / "fig4_graph_metrics.csv",
            [
                {
                    "paper_id": "paper",
                    "graph_metric_valid": True,
                    "weighted_score_fig3": 0.7,
                    "graph_confidence": 0.9,
                    "metric_source": "graph_metrics_table",
                    "B": 0.1,
                    "RS": 0.2,
                    "DeltaQ0": 0.3,
                    "Uzzi": 0.4,
                    "RTD": 0.5,
                    "BurtIP": 0.6,
                    "PDE": 0.7,
                }
            ],
        )
        write_csv(
            out_dir / "fig4_graph_prior.csv",
            [
                {
                    "paper_id": "paper",
                    "fig3_sw": 0.42,
                    "fig3_sw_percentile": 0.80,
                    "fig3_sw_tier": "high",
                    "fig3_weights_source": "synthetic_weights.csv",
                    "fig3_weights_hash": "abc123",
                    "fig3_sw_quality_flag": "ok",
                    "graph_prior_prompt_mode": "fig3_sw_only",
                    "fallback_percentile_source": "",
                }
            ],
        )
        write_csv(
            out_dir / "fig4_retrieval_diagnostics.csv",
            [{"paper_id": "paper", "retrieval_source": "semantic_scholar_anonymous", "s2_key_status": "s2_key_rejected"}],
        )
        semantic_matches = []
        for aspect, item in label_aspects.items():
            for point in item["points"]:
                semantic_matches.append(
                    {
                        "paper_id": "paper",
                        "aspect": aspect,
                        "peer_point": point,
                        "peer_quote": (item["quotes"] or [""])[0],
                        "best_agent_point": point,
                        "agent_quote": (item["quotes"] or [""])[0],
                        "relation": "entailed",
                        "score": 1.0,
                        "rationale": "same point",
                    }
                )
        write_jsonl(out_dir / "fig4_semantic_claim_matches.jsonl", semantic_matches)
        write_jsonl(
            out_dir / "fig4_structured_consistency_judgements.jsonl",
            [
                {
                    "paper_id": "paper",
                    "success": True,
                    "stance_consistency_1_5": 5,
                    "novelty_consistency_1_5": 5,
                    "significance_consistency_1_5": 5,
                    "prior_art_consistency_1_5": None,
                    "evidence_rigor_consistency_1_5": 5,
                    "limitations_consistency_1_5": 5,
                    "future_work_consistency_1_5": 5,
                    "overclaiming_score_1_5": 1,
                    "missing_key_points": [],
                    "contradictions": [],
                }
            ],
        )

        rows, matches = run_metrics_stage(out_dir, human_hours=5.0, judge_backend="none", quiet=True)

        assert rows[0]["consistency_cosine"] > 0.99
        assert rows[0]["coverage_score"] == 1.0
        assert rows[0]["innovation_stance_agreement"] == 1.0
        assert rows[0]["stance_exact_agreement"] == 1.0
        assert rows[0]["stance_within_one_agreement"] == 1.0
        assert rows[0]["semantic_claim_alignment"] == 1.0
        assert rows[0]["strict_claim_recall"] == 1.0
        assert rows[0]["soft_claim_recall"] == 1.0
        assert rows[0]["structured_semantic_consistency_mean"] == 5.0
        assert rows[0]["overclaiming_score_1_5"] == 1.0
        assert rows[0]["overclaiming_flag"] == 0.0
        assert rows[0]["claim_validation_pass"] == 1.0
        assert rows[0]["novelty_alignment"] == 1.0
        assert rows[0]["phrase_claim_coverage_supplementary"] == 1.0
        assert rows[0]["included_in_main"]
        assert rows[0]["fig3_sw"] == 0.42
        assert rows[0]["fig3_sw_percentile"] == 0.80
        assert rows[0]["fig3_sw_tier"] == "high"
        assert rows[0]["graph_prior_prompt_mode"] == "fig3_sw_only"
        assert len([match for match in matches if match["match_method"] == "normalized_phrase_overlap"]) == 5
        assert len([match for match in matches if match["match_method"] == "innovation_label_point_overlap"]) == 6
        metrics_csv = read_csv_records(out_dir / "fig4_metrics_summary.csv")
        assert metrics_csv[0]["embedding_backend"] in {"bge-m3", "lexical_fallback"}
        assert metrics_csv[0]["fig3_sw_tier"] == "high"
        aspect_summary = read_csv_records(out_dir / "fig4_aspect_relation_summary.csv")
        assert aspect_summary
        assert sum(int(float(row["total_points"])) for row in aspect_summary) == len(semantic_matches)
        examples = json.loads((out_dir / "fig4_claim_examples.json").read_text(encoding="utf-8"))
        assert examples["examples"]


def test_draw_fig4_smoke_outputs_all_formats() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_draw_") as tmp:
        out_dir = Path(tmp)
        metric_rows = []
        semantic_rows = []
        aspects = ["significance", "novelty", "evidence_rigor", "limitations", "future_work"]
        for idx in range(12):
            paper_id = f"paper_{idx}"
            metric_rows.append(
                {
                    "paper_id": paper_id,
                    "included_in_main": True,
                    "agent_success": True,
                    "retrieval_source": "semantic_scholar",
                    "graph_metric_valid": True,
                    "agent_innovation_stance_1_5": 3 + (idx % 3),
                    "peer_innovation_stance_1_5": 3 + ((idx + 1) % 3),
                    "agent_overall_score_1_5": 3 + (idx % 3),
                    "peer_overall_score_1_5": 3 + ((idx + 1) % 3),
                    "quadratic_weighted_kappa": 0.62,
                    "innovation_stance_agreement": 0.66,
                    "consistency_cosine": 0.70 + 0.02 * (idx % 4),
                    "structured_semantic_consistency_mean": 3.2 + 0.1 * (idx % 5),
                    "overclaiming_score_1_5": 1.4,
                    "semantic_claim_alignment": 0.72,
                    "missing_peer_point_rate": 0.18,
                    "claim_evidence_coverage": 0.78,
                    "contradiction_rate": 0.03,
                    "agent_runtime_seconds": 80 + idx,
                    "speedup_vs_human": 18000 / (80 + idx),
                    "peer_tense_errors_per_5000": 25,
                    "agent_tense_errors_per_5000": 5,
                    "peer_grammar_errors_per_5000": 18,
                    "agent_grammar_errors_per_5000": 9,
                    "peer_spelling_errors_per_5000": 100,
                    "agent_spelling_errors_per_5000": 2,
                    "readability_available": True,
                    "peer_flesch_reading_ease": 42,
                    "agent_flesch_reading_ease": 62,
                    "peer_flesch_kincaid_grade": 13,
                    "agent_flesch_kincaid_grade": 9,
                    "peer_significance": 2.2,
                    "agent_significance": 3.5,
                    "peer_novelty": 2.0,
                    "agent_novelty": 3.2,
                    "peer_rigor": 2.1,
                    "agent_rigor": 3.3,
                    "peer_limitations": 1.7,
                    "agent_limitations": 2.9,
                    "peer_future_work": 1.8,
                    "agent_future_work": 3.1,
                    "total_peer_aspects": 9,
                    "covered_peer_aspects": 7,
                    "fig3_sw": -0.8 + idx * 0.15,
                    "fig3_sw_percentile": (idx + 1) / 13,
                    "fig3_sw_tier": "low" if idx < 4 else ("middle" if idx < 8 else "high"),
                }
            )
            for aspect in aspects:
                semantic_rows.append(
                    {
                        "paper_id": paper_id,
                        "aspect": aspect,
                        "relation": "entailed" if idx % 3 else "related",
                        "score": 1.0 if idx % 3 else 0.5,
                    }
                )
        write_csv(out_dir / "fig4_metrics_summary.csv", metric_rows)
        write_jsonl(out_dir / "fig4_semantic_claim_matches.jsonl", semantic_rows)

        panel_data = draw_fig4(out_dir, human_hours=5.0, quiet=True)

        assert panel_data["n_main"] == 12
        assert (out_dir / "fig4_full.png").exists()
        assert (out_dir / "fig4_full.pdf").exists()
        assert (out_dir / "fig4_full.svg").exists()
        assert (out_dir / "fig4_system_dashboard.png").exists()
        assert panel_data["sw_tier_counts"]["low"] == 4
        assert panel_data["sw_tier_counts"]["middle"] == 4
        assert panel_data["sw_tier_counts"]["high"] == 4


def test_publication_summary_emphasizes_claim_validation_metrics() -> None:
    with tempfile.TemporaryDirectory(prefix="aspr_fig4_pub_summary_") as tmp:
        root = Path(tmp)
        metrics_path = root / "fig4_metrics_summary.csv"
        out_dir = root / "summary"
        rows = []
        for idx in range(4):
            rows.append(
                {
                    "paper_id": f"paper_{idx}",
                    "included_in_main": True,
                    "screen_pass": True,
                    "agent_success": True,
                    "graph_metric_valid": True,
                    "peer_innovation_stance_1_5": 4,
                    "agent_innovation_stance_1_5": 4,
                    "stance_exact_agreement": 1.0,
                    "stance_within_one_agreement": 1.0,
                    "quadratic_weighted_kappa": 1.0,
                    "semantic_claim_alignment": 0.875,
                    "strict_claim_recall": 0.75,
                    "soft_claim_recall": 1.0,
                    "overclaiming_score_1_5": 1.0,
                    "overclaiming_flag": 0.0,
                    "claim_validation_pass": 1.0,
                    "missing_peer_point_rate": 0.0,
                    "contradiction_rate": 0.0,
                    "consistency_cosine": 0.68,
                    "novelty_semantic_coverage": 1.0,
                    "significance_semantic_coverage": 1.0,
                    "prior_art_semantic_coverage": 0.5,
                    "evidence_rigor_semantic_coverage": 1.0,
                    "limitations_semantic_coverage": 0.5,
                    "future_work_semantic_coverage": 1.0,
                    "entailed_points": 3,
                    "related_points": 1,
                    "no_match_points": 0,
                    "contradicted_points": 0,
                }
            )
        write_csv(metrics_path, rows)

        result = draw_publication_summary(pd.read_csv(metrics_path), out_dir, dpi=120)

        assert result["summary"]["strict_claim_recall_mean"] == 0.75
        assert result["summary"]["soft_claim_recall_mean"] == 1.0
        assert result["summary"]["low_overclaiming_rate_mean"] == 1.0
        assert (out_dir / "fig4_claim_validation_summary.png").exists()
        assert (out_dir / "fig4_claim_validation_summary.svg").exists()


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("fig4 pipeline tests passed")
