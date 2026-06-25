from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig5.build_fig5_image2_handoff import build_handoff
from experiments.kg_perturbation_fig5.build_fig5_plot_data import (
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
        assert panel_text["panel_b"]["top_foci"][0]["focus_label"] == "AI-enabled scientific discovery"
        assert panel_text["panel_d"]["cards"][0]["predicted_role"] == "general-purpose discovery engine"


def test_default_fig5_data_paths_prefer_local_redraw_v6a_best_fig3() -> None:
    """Default Fig. 5 data inputs prefer the local validated Fig. 3 redraw run."""
    expected_run = FIG5_PROJECT_ROOT / "outputs" / "redraw_v6a_best_fig3" / "multi_domain"
    expected_input = FIG5_PROJECT_ROOT / "outputs" / "redraw_v6a_best_fig3" / "fig3_input" / "multi_domain"
    if expected_run.exists() and expected_input.exists() and not DEFAULT_FIG3_RUN_DIR.exists():
        assert resolve_default_run_dir() == expected_run
        assert resolve_default_input_dir(expected_run) == expected_input


if __name__ == "__main__":
    test_build_handoff_writes_prompt_text_and_draft()
    test_build_handoff_can_write_ai_cross_domain_lens()
    test_default_fig5_data_paths_prefer_local_redraw_v6a_best_fig3()
    print("test_fig5_image2_handoff passed")
