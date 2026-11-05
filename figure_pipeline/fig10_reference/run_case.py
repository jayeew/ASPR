"""Resume the case with audited exclusions and its pre-publication graph."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from gear.artifacts import read_model
from gear.config import load_config
from gear.contracts import RetrievedWork
from gear.prior_art import PriorArtService
from gear.review_contracts import InnovationPaperInput
from gear.review_pipeline import review_paper

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fig10_reference"
EXCLUDED = {
    "W4398774111": {
        "doi": "10.1158/2159-8290.cd-nw2024-0034",
        "reason": "Reporting on the target study, not independent prior art; Crossref only gives year 2024, not January 1.",
        "source": "https://api.crossref.org/works/10.1158/2159-8290.CD-NW2024-0034",
    },
    "W4402956197": {
        "doi": "10.5376/bm.2024.15.0010",
        "reason": "Target-study news published 2024-05-25, after cutoff.",
        "source": "https://www.bioscipublisher.com/index.php/bm/article/view/3818",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["gear", "graph", "fusion", "all"], default="all"
    )
    args = parser.parse_args()
    original = PriorArtService._eligible_before_cutoff

    def eligible(work: RetrievedWork, cutoff: date) -> bool:
        ident = str(work.work_id).rstrip("/").split("/")[-1]
        return ident not in EXCLUDED and original(work, cutoff)

    PriorArtService._eligible_before_cutoff = staticmethod(eligible)
    (OUT / "source/exclusions.json").write_text(
        json.dumps(EXCLUDED, ensure_ascii=False, indent=2)
    )
    item = read_model(OUT / "source/input_structured.json", InnovationPaperInput)
    result = review_paper(
        item,
        output_dir=OUT / "analysis",
        config=load_config(None),
        stage=args.stage,
        graph_root=OUT / "snapshot",
        embedding_model=ROOT / "data/models/Qwen3-Embedding-4B",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
