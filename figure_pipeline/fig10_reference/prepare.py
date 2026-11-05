"""Prepare the AlphaFold 3 case from its preserved publisher JATS XML."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig10_reference"


def text(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()


def blocks(node: ET.Element, level: int = 2) -> list[str]:
    result: list[str] = []
    for child in node:
        if child.tag == "title":
            result.append("#" * min(level, 6) + " " + text(child))
        elif child.tag in ("p", "fig", "table-wrap", "disp-formula", "ref"):
            result.append(text(child))
        else:
            result.extend(blocks(child, level + (child.tag == "sec")))
    return result


def main() -> None:
    source = OUT / "source"
    root = ET.parse(source / "paper.xml").getroot()
    title = root.find(".//article-title")
    abstract = root.find(".//abstract")
    body = root.find("body")
    assert title is not None and abstract is not None and body is not None
    parts = ["# " + text(title), "## Abstract", text(abstract)]
    parts.extend(blocks(body))
    manuscript = "\n\n".join(parts)
    assert len(manuscript) > 50000
    (source / "paper_structured.md").write_text(manuscript)
    item = json.loads((source / "input.json").read_text())
    item["paper_path"] = str(source / "paper_structured.md")
    item["abstract_text"] = text(abstract)
    item["abstract_source"] = "publisher"
    (source / "input_structured.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2)
    )
    manifest = json.loads((source / "provenance.json").read_text())
    manifest["structured_files"] = {
        n: hashlib.sha256((source / n).read_bytes()).hexdigest()
        for n in ("paper_structured.md", "input_structured.json")
    }
    (source / "provenance.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
