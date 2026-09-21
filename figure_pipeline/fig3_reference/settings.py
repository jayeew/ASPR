"""Fixed study population, information conditions and drawing coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/innovation_200_20260907"
OUTPUT = ROOT / "outputs/fig3_reference"
REFERENCE = Path("/mnt/c/Users/jayee/.codex/attachments/24faa840-f900-4a5a-ad5e-fff798256814/image-1.png")
DESIGN = Path("/mnt/c/Users/jayee/Downloads/Fig3_final_drawing_spec.md")
METHODS = ("direct_a", "direct_b", "direct_c", "gear", "graph", "fusion")
LABELS = ("Direct-A", "Direct-B", "Direct-C", "GEAR-only", "Graph-only", "GEAR–Graph")
MODELS = dict(zip(METHODS, ("gpt-5.6-luna", "gpt-5.5", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-luna", "gpt-5.6-luna")))
COLORS = dict(zip(METHODS, ("#65676f", "#92959f", "#b5b9c3", "#ef782e", "#2379ed", "#8b2be2")))
JUDGES = {"extract": "gpt-5.6-sol", "support": "gpt-5.6-sol", "quality": "gpt-5.6-sol", "preference": "gpt-6-astra", "reviews": "gpt-5.6-sol"}
DIMENSIONS = ("contribution_fidelity", "historical_increment", "knowledge_relation", "scope_calibration", "whole_paper_synthesis")
PANELS = {
    "a": (30, 90, 2040, 404), "b": (30, 512, 2040, 433),
    "c": (30, 963, 2040, 433), "d": (30, 1414, 2040, 433),
    "e": (30, 1865, 2040, 462), "f": (30, 2345, 1260, 721),
    "g": (1308, 2345, 762, 721),
}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def roster() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (STUDY / "papers.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 200 or len({row["paper_id"] for row in rows}) != 200:
        raise ValueError("Figure 3 requires the original 200 distinct study papers")
    return rows


def missing_path(stage: str, ident: str, method: str = "") -> Path:
    return OUTPUT / "data/incomplete" / stage / f"{ident}{'__'+method if method else ''}.json"
