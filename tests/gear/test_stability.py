from __future__ import annotations

from gear.review_verifier import GRAPH_SEMANTIC_TERMS


def test_aspr_brand_is_not_graph_semantic_leakage() -> None:
    assert GRAPH_SEMANTIC_TERMS.search("ASPR-Qwen was unavailable") is None
    assert GRAPH_SEMANTIC_TERMS.search("The ASPR score proves novelty") is not None
