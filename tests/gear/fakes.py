from __future__ import annotations

from typing import Any


class EmptyPriorArt:
    def __init__(self) -> None:
        self.families: list[str] = []
        self.claim_texts: list[str] = []

    def retrieve(self, claim: Any, cutoff: Any, budget: Any, *, family: str = "normal"):
        self.families.append(family)
        self.claim_texts.append(str(claim.text))
        if family == "contrastive":
            budget.contrastive_used += 1
        else:
            budget.normal_used = budget.normal_max
        return []

    def expand_neighbors(self, seed: Any, claim: Any, cutoff: Any, budget: Any):
        budget.citation_expansion_used += 1
        return []


class UnusedRelationClassifier:
    def classify(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no work should reach the relation classifier")
