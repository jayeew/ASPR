"""Acquire traceable scientific metadata and text without model calls."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from .catalog import SOURCES

OUT = (
    Path(__file__).resolve().parents[2]
    / "outputs/classic_claims_position_2025_2026/preparation"
)


def fetch(url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    meta = {"url": url, "acquired_at": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(
            url, timeout=50, headers={"User-Agent": "ScientificEvidencePreparation/1.0"}
        )
        path.write_bytes(response.content)
        meta.update(
            status=response.status_code,
            final_url=response.url,
            sha256=hashlib.sha256(response.content).hexdigest(),
        )
    except requests.RequestException as exc:
        meta.update(status=0, error=str(exc))
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def acquire(entry: tuple) -> dict:
    doi, claims, role = entry
    sid = "S-" + hashlib.sha256(doi.lower().encode()).hexdigest()[:12]
    directory = OUT / "sources" / sid
    directory.mkdir(parents=True, exist_ok=True)
    attempts = []
    cross = directory / "crossref.json"
    attempts.append(
        fetch("https://api.crossref.org/works/" + quote(doi, safe=""), cross)
    )
    data = (
        json.loads(cross.read_text()).get("message", {})
        if attempts[-1]["status"] == 200
        else {}
    )
    epmc = directory / "europepmc.json"
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&query="
        + quote('DOI:"' + doi + '"')
    )
    attempts.append(fetch(url, epmc))
    records = (
        json.loads(epmc.read_text()).get("resultList", {}).get("result", [])
        if attempts[-1]["status"] == 200
        else []
    )
    record = next((r for r in records if r.get("doi", "").lower() == doi.lower()), {})
    abstracts = []
    for origin, value in [
        ("crossref", data.get("abstract", "")),
        ("europepmc", record.get("abstractText", "")),
    ]:
        if value:
            text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
            abstracts.append(
                {
                    "text": text,
                    "locator": f"{origin}.json: abstract",
                    "material_type": "abstract",
                }
            )
    if record.get("pmcid"):
        xml = directory / "fulltext.xml"
        attempts.append(
            fetch(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                + record["pmcid"]
                + "/fullTextXML",
                xml,
            )
        )
        if attempts[-1]["status"] == 200:
            soup = BeautifulSoup(xml.read_text(), "xml")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.select("body p")]
            (directory / "fulltext_paragraphs.json").write_text(
                json.dumps(paragraphs, ensure_ascii=False, indent=2)
            )
    if not abstracts:
        html = directory / "publisher.html"
        attempts.append(fetch("https://doi.org/" + doi, html))
        if attempts[-1]["status"] == 200:
            soup = BeautifulSoup(html.read_text(), "html.parser")
            for selector in [
                "#Abs1-content",
                ".abstract",
                "#abstract",
                ".article__abstract",
            ]:
                node = soup.select_one(selector)
                if node and len(node.get_text()) > 80:
                    abstracts.append(
                        {
                            "text": node.get_text(" ", strip=True),
                            "locator": "publisher.html CSS " + selector,
                            "material_type": "abstract",
                        }
                    )
                    break
    dateparts = []
    for key in ["published-online", "published-print", "published"]:
        parts = data.get(key, {}).get("date-parts", [[]])[0]
        if parts:
            dateparts.append({"field": key, "parts": parts})
    result = {
        "source_id": sid,
        "doi": doi,
        "url": "https://doi.org/" + doi,
        "title": data.get("title", [record.get("title", "")])[0],
        "authors": data.get("author", []),
        "date_candidates": dateparts,
        "claim_numbers": claims,
        "role": role,
        "abstracts": abstracts,
        "attempts": attempts,
        "raw_directory": str(directory.relative_to(OUT)),
        "fulltext_available": (directory / "fulltext_paragraphs.json").exists(),
    }
    (directory / "record.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(doi, result["title"][:55], "abstracts", len(abstracts), flush=True)
    return result


def main() -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(acquire, SOURCES))
    (OUT / "sources.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


if __name__ == "__main__":
    main()
