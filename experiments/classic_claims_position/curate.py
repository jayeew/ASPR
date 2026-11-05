"""Select verbatim passages from archived web captures, without model inference."""

from __future__ import annotations

import hashlib
import json
import re

from .preflight import OUT


def select(
    doi: str, filename: str, numbers: list[int], locator: str, material: str
) -> None:
    raw = OUT / "web_scientific" / filename
    capture = json.loads(raw.read_text())["web_result"]
    lines = []
    for line in capture.splitlines():
        match = re.match(r"L(\d+)(?:@P\d+)?: (.*)", line)
        if match and int(match[1]) in numbers:
            lines.append(match[2])
    if not numbers:
        lines = [capture.split("## Abstract\n\n", 1)[1].split("\n\nFig.", 1)[0]]
    assert lines, filename
    item = {
        "text": "\n".join(lines),
        "locator": locator,
        "raw_file": str(raw.relative_to(OUT)),
        "material_type": material,
    }
    directory = (
        OUT / "sources" / ("S-" + hashlib.sha256(doi.lower().encode()).hexdigest()[:12])
    )
    (directory / "curated_excerpt.json").write_text(
        json.dumps(item, ensure_ascii=False, indent=2)
    )


def main() -> None:
    select(
        "10.1103/PhysRevLett.55.1543",
        "10.1103_PhysRevLett.55.1543.json",
        list(range(8, 12))
        + list(range(21, 25))
        + list(range(43, 52))
        + list(range(195, 201)),
        "University-hosted published PDF; pp.1543,1545; web lines 8–11,21–24,43–51,195–200 (OCR retained)",
        "fulltext_excerpt",
    )
    select(
        "10.1002/anie.199717251",
        "kitagawa1997.json",
        [128, 129],
        "Publisher graphical abstract, web lines 128–129",
        "abstract",
    )
    select(
        "10.1088/1748-0221/12/03/P03012",
        "icecube_sections.json",
        [101, 105, 108, 113],
        "arXiv:1612.05093v3 (2024-02-06), sections 1 and 1.1; web lines 101,105,108,113; journal 2017",
        "fulltext_excerpt",
    )
    select(
        "10.1038/s41929-020-0441-1",
        "geiger_search.json",
        [],
        "Publisher abstract returned by indexed web search; publication 2020-03-16",
        "abstract",
    )


if __name__ == "__main__":
    main()
