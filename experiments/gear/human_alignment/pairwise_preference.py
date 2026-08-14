"""Blind pairwise preference summaries."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Literal

Preference = Literal["A", "B", "TIE", "UNCLEAR"]


def preference_summary(values: Iterable[Preference]) -> Dict[str, int | float]:
    counts = Counter(values)
    decided = counts["A"] + counts["B"] + counts["TIE"]
    return {
        "win": counts["A"],
        "loss": counts["B"],
        "tie": counts["TIE"],
        "unclear": counts["UNCLEAR"],
        "win_rate_excluding_unclear": counts["A"] / decided if decided else 0.0,
    }


__all__ = ["preference_summary"]
