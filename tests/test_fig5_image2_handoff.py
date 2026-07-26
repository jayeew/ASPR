from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fig05.old.build_fig5_image2_handoff import build_handoff, strict_ai_frontier_diagnostics
from experiments.fig05.old.build_fig5_plot_data import (
    DEFAULT_FIG3_RUN_DIR,
    PROJECT_ROOT as FIG5_PROJECT_ROOT,
    resolve_default_input_dir,
    resolve_default_run_dir,
)


def write_fixture(root: Path) -> None:
    """Write a minimal Fig. 5 plot-data fixture."""
    derived = root / "derived"
    config = root / "config"
    derived.mkdir(parents=True)
    config.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "forecast_rank": 1,
                "focus_id": "crispr::1",
                "focus_label": "Precision genome editing",
                "short_label": "Precision genome editing",
                "forecast_score": 0.93,
                "display_color": "#2563EB",
                "domain": "crispr",
            },
            {
                "forecast_rank": 2,
                "focus_id": "crispr::2",
                "focus_label": "RNA-targeted editors",
                "short_label": "RNA-targeted editors",
                "forecast_score": 0.86,
                "display_color": "#F97316",
                "domain": "crispr",
            },
        ]
    ).to_csv(derived / "fig5_panel_b_top_focus.csv", index=False)

    pd.DataFrame(
        [
            {
                "focus_id": "crispr::1",
                "focus_label": "Precision genome editing",
                "short_label": "Precision genome editing",
                "forecast_rank": 1,
                "plot_x": 0.2,
                "plot_y": 0.6,
                "forecast_score": 0.93,
                "hist_size": 120,
                "display_size": 420,
                "display_color": "#2563EB",
                "domain": "crispr",
            }
        ]
    ).to_csv(derived / "fig5_panel_c_focus_positions.csv", index=False)

    pd.DataFrame(
        [
            {
                "display_rank": 1,
                "innovation_id": "innovation_1",
                "innovation_label": "A programmable precision editing platform",
                "short_label": "Precision editing platform",
                "predicted_role": "key enabling innovation",
                "short_reason": "High-scoring seed for Precision genome editing.",
                "linked_focus_label": "Precision genome editing",
                "icon_type": "method",
                "display_color": "#2563EB",
                "seed_year": 2018,
            }
        ]
    ).to_csv(derived / "fig5_panel_d_cards.csv", index=False)

    (derived / "fig5_panel_a_meta.json").write_text(
        json.dumps(
            {
                "historical_start_year": 1950,
                "historical_end_year": 2020,
                "future_start_year": 2021,
                "future_end_year": 2026,
                "future_end_year_actual": 2025,
                "n_hist_papers": 2500,
                "n_hist_topics": 32,
                "n_predicted_focus": 12,
                "n_predicted_innovations": 4,
                "score_column": "S_w_oof",
            }
        ),
        encoding="utf-8",
    )
    (config / "fig5_config.json").write_text(
        json.dumps({"domain_filter": ["crispr"], "domains": ["crispr"]}),
        encoding="utf-8",
    )
    base = root / "base"
    base.mkdir(parents=True)
    pd.DataFrame([{"topic_id": "crispr::1"}, {"topic_id": "crispr::2"}]).to_csv(base / "topic_nodes.csv", index=False)
    pd.DataFrame([{"source_topic_id": "crispr::1", "target_topic_id": "crispr::2"}]).to_csv(base / "topic_edges.csv", index=False)
    pd.DataFrame([{"source_paper_id": "p1", "target_paper_id": "p2"}]).to_csv(base / "citation_edges.csv", index=False)
    pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "title": "A deep learning model for materials discovery",
                "year": 2019,
                "domain": "materials",
                "topic_id": "materials::ml",
                "topic_label": "Machine Learning in Materials Science",
                "selected_score": 0.91,
                "cited_by_count": 42,
            },
            {
                "paper_id": "p2",
                "title": "A programmable precision editing platform",
                "year": 2018,
                "domain": "crispr",
                "topic_id": "crispr::1",
                "topic_label": "Precision genome editing",
                "selected_score": 0.82,
                "cited_by_count": 36,
            },
        ]
    ).to_csv(base / "papers_master.csv", index=False)
    pd.DataFrame(
        [
            {
                "forecast_rank": 1,
                "focus_id": "crispr::1",
                "focus_label": "Precision genome editing",
                "short_label": "Precision genome editing",
                "forecast_score": 0.93,
                "display_color": "#2563EB",
                "domain": "crispr",
                "domain_label": "CRISPR",
                "historical_size": 120,
                "x": 0.2,
                "y": 0.6,
                "keyword_list": '["precision", "genome", "editing"]',
                "description": "Precision genome editing supported by 120 pre-cutoff papers.",
            },
            {
                "forecast_rank": 7,
                "focus_id": "materials::ml",
                "focus_label": "Machine Learning in Materials Science",
                "short_label": "Machine Learning in Materials",
                "forecast_score": 0.77,
                "display_color": "#7C3AED",
                "domain": "materials",
                "domain_label": "Materials",
                "historical_size": 18,
                "x": 0.7,
                "y": 0.3,
                "keyword_list": '["machine", "learning", "materials"]',
                "description": "Machine Learning in Materials Science supported by 18 pre-cutoff papers.",
            },
        ]
    ).to_csv(derived / "forecast_focus.csv", index=False)
    pd.DataFrame(
        [
            {
                "innovation_id": "innovation_ai_1",
                "innovation_label": "Deep learning model for materials discovery",
                "short_label": "Deep learning model",
                "forecast_rank": 7,
                "predicted_role": "computational discovery seed",
                "short_reason": "High-scoring seed for Machine Learning in Materials Science.",
                "linked_focus_id": "materials::ml",
                "linked_focus_label": "Machine Learning in Materials Science",
                "linked_topic_ids": '["materials::ml"]',
                "representative_papers": '["p1"]',
                "icon_type": "computation",
                "description": "Deep learning model anchors the predicted focus Machine Learning in Materials Science.",
                "seed_year": 2019,
                "seed_score": 0.91,
                "color_group": "computational",
                "display_color": "#7C3AED",
            }
        ]
    ).to_csv(derived / "forecast_innovations.csv", index=False)


def test_build_handoff_writes_prompt_text_and_draft() -> None:
    """The handoff builder writes all files and preserves exact fixture text."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "plot_data"
        out_dir = Path(tmp) / "image2_handoff"
        write_fixture(root)

        build_handoff(root, out_dir)

        panel_text = json.loads((out_dir / "fig5_panel_text.json").read_text(encoding="utf-8"))
        assert panel_text["title"].startswith("Fig. 5 | Forecasting future research focus")
        assert panel_text["subtitle"] == "Case study: CRISPR-Cas genome editing"
        assert panel_text["panel_b"]["top_foci"][0]["focus_label"] == "Precision genome editing"
        assert panel_text["panel_b"]["top_foci"][0]["forecast_score"] == 0.93
        assert panel_text["panel_d"]["cards"][0]["predicted_role"] == "key enabling innovation"

        prompt = (out_dir / "fig5_image2_prompt.md").read_text(encoding="utf-8")
        assert "Do not invent new scientific claims" in prompt
        assert "Precision genome editing" in prompt
        assert "four-panel publication figure" in prompt

        notes = (out_dir / "fig5_visual_reference_notes.md").read_text(encoding="utf-8")
        assert "Panel a" in notes
        assert (out_dir / "fig5_layout_draft.png").exists()
        assert (out_dir / "fig5_layout_draft.png").stat().st_size > 0


def test_build_handoff_can_write_ai_cross_domain_lens() -> None:
    """The handoff builder can create an explicitly labelled AI fusion lens."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "plot_data"
        out_dir = Path(tmp) / "ai_handoff"
        write_fixture(root)

        build_handoff(root, out_dir, theme="ai_cross_domain")

        panel_text = json.loads((out_dir / "fig5_panel_text.json").read_text(encoding="utf-8"))
        assert panel_text["theme_mode"] == "ai_cross_domain_lens"
        assert panel_text["subtitle"] == "Multi-domain frontier lens: AI-enabled scientific discovery"
        assert panel_text["panel_a"]["network_data_basis"]["topic_nodes"] == 2
        assert panel_text["panel_a"]["future_bubbles"][0]["label"] == "AI-enabled discovery"
        assert panel_text["panel_b"]["display_mode"] == "word_cloud_only"
        assert panel_text["panel_b"]["top_foci"][0]["focus_label"] == "AI-enabled scientific discovery"
        assert panel_text["panel_d"]["cards"][0]["predicted_role"] == "general-purpose discovery engine"


def test_build_handoff_can_write_strict_ai_filtered_theme_from_real_tables() -> None:
    """Strict AI mode filters real focus and seed tables without thematic replacements."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "plot_data"
        out_dir = Path(tmp) / "strict_ai_handoff"
        write_fixture(root)

        build_handoff(root, out_dir, theme="strict_ai_filtered")

        panel_text = json.loads((out_dir / "fig5_panel_text.json").read_text(encoding="utf-8"))
        assert panel_text["theme_mode"] == "strict_ai_filtered"
        assert panel_text["panel_b"]["display_mode"] == "word_cloud_only"
        assert panel_text["panel_b"]["top_foci"][0]["focus_label"] == "Machine Learning in Materials Science"
        assert panel_text["panel_b"]["top_foci"][0]["source_table"] == "derived/forecast_focus.csv"
        assert panel_text["panel_c"]["foci"][0]["source_table"] == "derived/forecast_focus.csv"
        assert panel_text["panel_d"]["cards"][0]["source_table"] == "derived/forecast_innovations.csv"
        assert panel_text["panel_d"]["cards"][0]["representative_papers"] == '["p1"]'
        assert "AI-enabled scientific discovery" not in json.dumps(panel_text, ensure_ascii=False)
        assert panel_text["strict_filter"]["source_tables"]["panel_b"] == "derived/forecast_focus.csv"
        assert panel_text["strict_filter"]["claim_gate"] == "blocked"
        assert panel_text["strict_filter"]["data_diagnostics"]["source_backed_ai_frontier_ready"] == 0
        assert "does not pass the source-backed AI frontier gate" in panel_text["take_home"]


def test_strict_ai_frontier_diagnostics_pass_when_source_rows_are_dense() -> None:
    """The source-backed AI gate requires dense AI evidence, not a single keyword hit."""
    forecast_rows = []
    for rank in range(1, 31):
        is_ai = rank <= 20
        forecast_rows.append(
            {
                "forecast_rank": rank,
                "forecast_score": 1.0 - rank / 100.0,
                "focus_label": f"{'Deep learning' if is_ai else 'Protein'} focus {rank}",
            }
        )
    forecast_focus = pd.DataFrame(forecast_rows)
    ai_focus = forecast_focus[forecast_focus["focus_label"].str.contains("Deep learning")].copy()

    diagnostics = strict_ai_frontier_diagnostics(forecast_focus, ai_focus)

    assert diagnostics["source_backed_ai_frontier_ready"] == 1
    assert diagnostics["strict_ai_rows_in_top20_forecast"] == 20
    assert diagnostics["strict_ai_positive_score_rows"] == 20


def test_default_fig5_data_paths_prefer_local_redraw_v6a_best_fig3() -> None:
    """Default Fig. 5 data inputs prefer the local validated Fig. 3 redraw run."""
    expected_run = FIG5_PROJECT_ROOT / "outputs" / "fig03/old" / "multi_domain"
    expected_input = FIG5_PROJECT_ROOT / "outputs" / "fig03/old" / "fig3_input" / "multi_domain"
    if expected_run.exists() and expected_input.exists() and not DEFAULT_FIG3_RUN_DIR.exists():
        assert resolve_default_run_dir() == expected_run
        assert resolve_default_input_dir(expected_run) == expected_input


if __name__ == "__main__":
    test_build_handoff_writes_prompt_text_and_draft()
    test_build_handoff_can_write_ai_cross_domain_lens()
    test_build_handoff_can_write_strict_ai_filtered_theme_from_real_tables()
    test_strict_ai_frontier_diagnostics_pass_when_source_rows_are_dense()
    test_default_fig5_data_paths_prefer_local_redraw_v6a_best_fig3()
    print("test_fig5_image2_handoff passed")
