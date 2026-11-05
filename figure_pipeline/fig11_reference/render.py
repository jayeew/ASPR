"""Render an editable taxonomy table using the Fig.1–8 SVG/export helpers."""

from __future__ import annotations

import ast
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from figure_pipeline.fig1_reference.export import convert, fonts, outline
from figure_pipeline.fig1_reference.svg import Scene, el

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig11_reference"
HISTORICAL = "scripts/claim_graph/04_extract_claims.py"
FULL_TEXT = "gear/grounding.py"
INK, BLUE, MUTED = "#101D64", "#006BFF", "#466487"
W, H = 1680, 785
ROWS = [
    ("METHOD", "method_or_system",
     "A reusable procedure, algorithm, model, system, material or intervention design itself. The observed outcome of applying it is FINDING.",
     "We developed a cell-imaging method that suppresses background fluorescence."),
    ("FINDING", "finding",
     "An observed, measured, estimated, comparative, structural, performance, association or intervention outcome. An observed effect alone does not explain its mechanism.",
     "The imaging method achieved higher cell-detection accuracy than the comparison method."),
    ("MECHANISM", "mechanism",
     "An evidence-supported causal account of how or why an effect occurs. Correlation, predictive association or inferred connectivity alone is FINDING.",
     "Controlled experiments showed that suppression of background fluorescence caused the gain in cell-detection accuracy."),
    ("RESOURCE", "resource_or_synthesis",
     "A dataset, atlas, database, catalog, benchmark, genome assembly or curated synthesis delivered for reuse. A protocol is METHOD; an ordinary result is FINDING.",
     "We released an annotated cell-image dataset for reuse in training and evaluating detection methods."),
    ("THEORY", "theory",
     "A newly proposed formal principle, law or conceptual/mathematical framework. Theoretical computation or an inferred historical explanation alone does not qualify.",
     "We proposed a mathematical framework relating background noise to the limits of cell-detection accuracy."),
]


def constants(relative: str, names: tuple[str, ...]) -> dict[str, str]:
    """Read prompt literals without importing model clients or invoking models."""
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != set(names):
        raise ValueError(f"Missing prompt constants in {relative}")
    return result


def taxonomy(f: Scene) -> None:
    f.rect(28, 123, 1624, 48, "#EAF6FF", "#BDDEFA", radius=4)
    for x, title in [(46, "Type / prompt label"), (304, "Classification criteria and distinctions"),
                     (1064, "Hypothetical example")]:
        f.text(x, 155, title, 23, INK, True)
    for i, (kind, alias, criterion, example) in enumerate(ROWS):
        y = 171 + i * 118
        f.rect(28, y, 1624, 118, "#F7FBFF" if i % 2 == 0 else "#FFFFFF")
        accent = ["#E4AE4A", "#4C78A8", "#D87562", "#B43E54", "#4C78A8"][i]
        f.node(57, y + 33, kind, accent, 10, accent, 1.2)
        f.text(80, y + 41, kind, 27, INK, True)
        f.text(46, y + 76, alias, 20, MUTED)
        f.paragraph(304, y + 30, criterion, 720, 23, 27, max_lines=4)
        f.paragraph(1064, y + 30, example, 568, 23, 27, max_lines=4)
        f.line(28, y + 118, 1652, y + 118, "#BDDEFA", .8)
    for x in (284, 1044):
        f.line(x, 123, x, 761, "#BDDEFA", .8)


def validate(f: Scene) -> dict[str, object]:
    outside = [b for b in f.text_boxes if b["x"] < 0 or b["y"] < 0
               or b["x"] + b["width"] > W or b["y"] + b["height"] > H]
    overlaps = []
    for i, left in enumerate(f.text_boxes):
        for right in f.text_boxes[i + 1:]:
            dx = min(left["x"] + left["width"], right["x"] + right["width"]) - max(left["x"], right["x"])
            dy = min(left["y"] + left["height"], right["y"] + right["height"]) - max(left["y"], right["y"])
            if dx > 1 and dy > 1:
                overlaps.append([left["text"], right["text"]])
    if outside or overlaps:
        raise ValueError({"outside": outside, "overlaps": overlaps})
    return {"outside_canvas": outside, "text_overlaps": overlaps, "text_elements": len(f.text_boxes)}


def main() -> None:
    historical = constants(HISTORICAL, ("SYSTEM_PROMPT", "PROMPT_TEMPLATE"))
    full_text = constants(FULL_TEXT, ("MINER_SYSTEM", "CONSOLIDATOR_SYSTEM", "VERIFIER_SYSTEM"))
    final = OUT / "final"
    final.mkdir(parents=True, exist_ok=True)
    fonts(OUT)
    f = Scene(W, H, "fig11-claim-taxonomy")
    f.rect(0, 0, W, H, "white")
    f.text(28, 46, "Fig. 11 | Claim types: classification criteria and key distinctions", 37, INK, True)
    f.text(28, 84, "Five contribution roles, distinguished by what the claim asserts — not by novelty, discipline or graph community", 25, MUTED)
    taxonomy(f)
    qa = validate(f)
    root = el("svg", width="297mm", height=f"{297 * H / W:.3f}mm", viewBox=f"0 0 {W} {H}")
    root.append(f.root)
    svg = final / "Fig11.svg"
    ET.ElementTree(root).write(svg, encoding="utf-8", xml_declaration=True)
    outline(svg, final / "Fig11_outlined.svg")
    convert(svg, scale=2)
    sources = [HISTORICAL, FULL_TEXT, "gear/claim_graph/contracts.py"]
    qa["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    (OUT / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (OUT / "prompts.json").write_text(json.dumps({HISTORICAL: historical, FULL_TEXT: full_text}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(final), **qa}, indent=2))


if __name__ == "__main__":
    main()
