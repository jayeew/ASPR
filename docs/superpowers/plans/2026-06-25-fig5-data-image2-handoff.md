# Fig. 5 Data Package and Image-2 Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Fig. 5 data package and image-2 handoff package for a four-panel publication-style figure.

**Architecture:** Keep `build_fig5_plot_data.py` responsible for evidence tables and add a focused `build_fig5_image2_handoff.py` module that only reads those generated tables. The handoff module writes exact figure text, a complete image-2 prompt, visual reference notes, and a deterministic low-fidelity draft without recomputing rankings.

**Tech Stack:** Python 3.11+, pandas, Pillow, pathlib/json/argparse, inline Python tests under `tests/`.

---

### Task 1: Add Tests For Image-2 Handoff Outputs

**Files:**
- Create: `tests/test_fig5_image2_handoff.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fig5_image2_handoff.py` with:

```python
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from experiments.kg_perturbation_fig5.build_fig5_image2_handoff import build_handoff


def write_fixture(root: Path) -> None:
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


if __name__ == "__main__":
    test_build_handoff_writes_prompt_text_and_draft()
    print("test_fig5_image2_handoff passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe tests/test_fig5_image2_handoff.py
```

Expected: FAIL with `ModuleNotFoundError` for `build_fig5_image2_handoff`.

### Task 2: Implement Image-2 Handoff Builder

**Files:**
- Create: `experiments/kg_perturbation_fig5/build_fig5_image2_handoff.py`

- [ ] **Step 1: Write the module**

Create `experiments/kg_perturbation_fig5/build_fig5_image2_handoff.py` with:

```python
#!/usr/bin/env python3
"""Build image-2 handoff assets for Fig. 5."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PLOT_DATA_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig5" / "plot_data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig5" / "image2_handoff"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def clean_float(value: object, digits: int = 2) -> float:
    return round(float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]), digits)


def clean_int(value: object) -> int:
    return int(float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]))


def infer_subtitle(config: Dict[str, Any]) -> str:
    domains = [str(item) for item in config.get("domain_filter") or config.get("domains") or []]
    if len(domains) == 1 and domains[0] == "crispr":
        return "Case study: CRISPR-Cas genome editing"
    if len(domains) == 1:
        return f"Case study: {domains[0].replace('_', ' ').title()}"
    return "Multi-domain forecast validation"


def records_for_panel_b(panel_b: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in panel_b.sort_values("forecast_rank").iterrows():
        rows.append(
            {
                "rank": clean_int(row["forecast_rank"]),
                "focus_id": str(row["focus_id"]),
                "focus_label": str(row["focus_label"]),
                "short_label": str(row.get("short_label", row["focus_label"])),
                "forecast_score": clean_float(row["forecast_score"]),
                "display_color": str(row.get("display_color", "#2563EB")),
                "domain": str(row.get("domain", "")),
            }
        )
    return rows
```

Add the remaining helpers with these exact public function names and responsibilities:

```python
def records_for_panel_c(panel_c: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return rank, label, position, score, historical size, color, and domain records."""


def records_for_panel_d(panel_d: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return card rank, label, role, reason, focus label, icon type, color, and seed year records."""


def build_panel_text(plot_data_dir: Path) -> Dict[str, Any]:
    """Read generated Fig. 5 data tables and return the exact figure text JSON."""


def render_prompt(panel_text: Dict[str, Any]) -> str:
    """Return a complete image-2 prompt embedding the exact panel_text JSON."""


def render_visual_notes(panel_text: Dict[str, Any]) -> str:
    """Return concise visual rules for matching the reference Fig. 5 layout."""


def draw_layout_draft(panel_text: Dict[str, Any], out_path: Path) -> None:
    """Draw the low-fidelity four-panel draft PNG from panel_text."""


def build_handoff(plot_data_dir: Path, out_dir: Path) -> Dict[str, Path]:
    """Write fig5_panel_text.json, fig5_image2_prompt.md, fig5_visual_reference_notes.md, and fig5_layout_draft.png."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse --plot-data-dir and --out-dir."""


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point that calls build_handoff and prints written paths."""
```

- [ ] **Step 2: Include the prompt contract**

In `render_prompt`, include these exact strings:

```text
Create a four-panel publication figure closely matching the supplied Fig. 5 reference.
Do not invent new scientific claims, focus names, rankings, scores, or card text.
Use the exact text from the panel text JSON below.
```

- [ ] **Step 3: Include the draft contract**

In `draw_layout_draft`, use Pillow to draw a 1536 x 1024 style low-fidelity layout with four rounded panel boxes, the title/subtitle, panel labels a/b/c/d, a bottom take-home strip, and actual top focus labels from `panel_text`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe tests/test_fig5_image2_handoff.py
```

Expected: `test_fig5_image2_handoff passed`.

### Task 3: Point Fig. 5 Data Builder At The Local Fig. 3 Run

**Files:**
- Modify: `experiments/kg_perturbation_fig5/build_fig5_plot_data.py`

- [ ] **Step 1: Update default candidates**

Modify `resolve_default_run_dir()` so `outputs/redraw_v6a_best_fig3/multi_domain` is the first fallback candidate after `DEFAULT_FIG3_RUN_DIR`.

Modify `resolve_default_input_dir()` so `outputs/redraw_v6a_best_fig3/fig3_input/multi_domain` is checked before older Fig. 3 locations.

- [ ] **Step 2: Run the data builder**

Run:

```bash
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m experiments.kg_perturbation_fig5.build_fig5_plot_data \
  --out-dir outputs/kg_perturbation_fig5/plot_data
```

Expected: output includes `[fig5-data] wrote` and nonzero paper/topic/focus counts.

### Task 4: Generate Handoff Package From Real Data

**Files:**
- Generated output under ignored `outputs/kg_perturbation_fig5/image2_handoff/`

- [ ] **Step 1: Run the handoff builder**

Run:

```bash
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m experiments.kg_perturbation_fig5.build_fig5_image2_handoff \
  --plot-data-dir outputs/kg_perturbation_fig5/plot_data \
  --out-dir outputs/kg_perturbation_fig5/image2_handoff
```

Expected: writes `fig5_image2_prompt.md`, `fig5_panel_text.json`, `fig5_visual_reference_notes.md`, and `fig5_layout_draft.png`.

- [ ] **Step 2: Verify panel text consistency**

Run:

```bash
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("outputs/kg_perturbation_fig5")
panel_b = pd.read_csv(root / "plot_data" / "derived" / "fig5_panel_b_top_focus.csv")
panel_text = json.loads((root / "image2_handoff" / "fig5_panel_text.json").read_text(encoding="utf-8"))
assert panel_text["panel_b"]["top_foci"][0]["focus_label"] == str(panel_b.sort_values("forecast_rank").iloc[0]["focus_label"])
assert round(panel_text["panel_b"]["top_foci"][0]["forecast_score"], 2) == round(float(panel_b.sort_values("forecast_rank").iloc[0]["forecast_score"]), 2)
print("fig5 panel text consistency passed")
PY
```

Expected: `fig5 panel text consistency passed`.

### Task 5: Document The Workflow

**Files:**
- Modify: `experiments/kg_perturbation_fig5/README.md`

- [ ] **Step 1: Add commands**

Add a short section with the exact data-builder and handoff-builder commands from Tasks 3 and 4.

- [ ] **Step 2: Run a syntax check**

Run:

```bash
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m py_compile \
  experiments/kg_perturbation_fig5/build_fig5_plot_data.py \
  experiments/kg_perturbation_fig5/build_fig5_image2_handoff.py \
  tests/test_fig5_image2_handoff.py
```

Expected: exit code 0 with no output.

### Task 6: Final Verification

**Files:**
- Review: all changed files

- [ ] **Step 1: Run focused verification**

Run:

```bash
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe tests/test_fig5_image2_handoff.py
/mnt/c/Users/jayee/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -m py_compile experiments/kg_perturbation_fig5/build_fig5_plot_data.py experiments/kg_perturbation_fig5/build_fig5_image2_handoff.py tests/test_fig5_image2_handoff.py
```

Expected: test prints `test_fig5_image2_handoff passed`; `py_compile` exits with no output.

- [ ] **Step 2: Inspect git diff**

Run:

```bash
git diff -- experiments/kg_perturbation_fig5/build_fig5_plot_data.py experiments/kg_perturbation_fig5/build_fig5_image2_handoff.py experiments/kg_perturbation_fig5/README.md tests/test_fig5_image2_handoff.py
```

Expected: only Fig. 5 handoff-related changes.
