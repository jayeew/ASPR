"""Paper-cluster bootstrap and durable Markdown reporting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Mapping


def bootstrap_papers(
    paper_values: Mapping[str, float],
    *,
    iterations: int = 5_000,
    seed: int = 17,
) -> Dict[str, float | int | None]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    values = list(paper_values.values())
    if not values:
        return {"n_papers": 0, "estimate": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(iterations)
    )
    return {
        "n_papers": len(values),
        "estimate": sum(values) / len(values),
        "ci_low": samples[int(0.025 * (iterations - 1))],
        "ci_high": samples[int(0.975 * (iterations - 1))],
    }


def render_markdown_report(sections: Mapping[str, Mapping[str, object]]) -> str:
    lines = ["# ASPR-ESR Human Alignment Report", ""]
    for title, metrics in sections.items():
        lines.extend([f"## {title}", ""])
        for name, value in metrics.items():
            lines.append(f"- `{name}`: {value}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def availability_summary(
    *, total_papers: int, available_papers: int, valid_review_papers: int
) -> Dict[str, float | int | None]:
    return {
        "total_papers": total_papers,
        "available_papers": available_papers,
        "valid_review_papers": valid_review_papers,
        "availability_rate": (
            available_papers / total_papers if total_papers else None
        ),
        "valid_review_rate": (
            valid_review_papers / total_papers if total_papers else None
        ),
        "conditional_valid_rate": (
            valid_review_papers / available_papers if available_papers else None
        ),
    }


def write_report(
    output_dir: Path, sections: Mapping[str, Mapping[str, object]]
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "human_alignment_report.json"
    markdown_path = output_dir / "human_alignment_report.md"
    json_path.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(sections), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "availability_summary",
    "bootstrap_papers",
    "render_markdown_report",
    "write_report",
]
