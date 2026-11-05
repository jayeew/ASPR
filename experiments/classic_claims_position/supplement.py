"""Retrieve additional open primary sources and preserve acquisition failures."""

from __future__ import annotations

import hashlib
import json

import pymupdf
from bs4 import BeautifulSoup

from .acquire import OUT, acquire, fetch


def main() -> None:
    acquire(("10.1021/ja00197a079", [3], "original"))
    acquire(("10.1021/ja00100a004", [13], "followup"))
    acquire(("10.1038/s41567-020-0829-5", [1, 2], "followup"))
    urls = [
        (
            "10.1103/PhysRevLett.55.1543",
            "https://www.physics.umd.edu/courses/Phys798C/AnlageFall25/Martinis-1985-Energy-Level%20Quantization%20in%20the.pdf",
        ),
        ("10.1088/1748-0221/12/03/P03012", "https://arxiv.org/pdf/1612.05093"),
    ]
    for doi, url in urls:
        sid = "S-" + hashlib.sha256(doi.lower().encode()).hexdigest()[:12]
        directory = OUT / "sources" / sid
        path = directory / "supplement.pdf"
        meta = fetch(url, path)
        if meta["status"] == 200 and path.read_bytes().startswith(b"%PDF"):
            with pymupdf.open(path) as doc:
                pages = [p.get_text() for p in doc]
            (directory / "pdf_pages.json").write_text(
                json.dumps(pages, ensure_ascii=False, indent=2)
            )
    path = OUT / "sources/physics_author_index.html"
    meta = fetch("https://qulab.eng.yale.edu/publications_page/", path)
    if meta["status"] == 200:
        soup = BeautifulSoup(path.read_text(), "html.parser")
        links = [
            (a.get_text(), a.get("href"))
            for a in soup.find_all("a")
            if any(
                w in (a.get_text() + " " + str(a.get("href"))).lower()
                for w in ["1985", "1987", "tunneling", "quantization"]
            )
        ]
        print(json.dumps(links, indent=2), flush=True)


if __name__ == "__main__":
    main()
