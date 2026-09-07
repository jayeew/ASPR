import sqlite3
from datetime import date
from pathlib import Path

from gear.claim_attribution import ClaimGraphRuntime
from gear.claim_graph.contracts import InnovationClaimType
from gear.contracts import PaperMetadata, RetrievedWork
from gear.work_identity import version_identity


def test_undefined_graph_values_have_no_percentile():
    runtime = ClaimGraphRuntime(Path("/nonexistent"), Path("/nonexistent"))
    assert runtime._pair_surprisal([1]) is None
    assert runtime._new_pair_share([1]) is None
    fact = runtime._metric_fact(
        "community_pair_mean_surprisal", None, InnovationClaimType.FINDING
    )
    assert fact.value is None and fact.global_percentile is None
    assert all(
        x.value is None for x in runtime._metrics([], InnovationClaimType.FINDING)
    )


def test_two_hop_normalizes_identifiers():
    runtime = ClaimGraphRuntime(Path("."), Path("."))
    runtime._paper_db = sqlite3.connect(":memory:")
    runtime._paper_db.execute(
        "CREATE TABLE paper_edges(citing_work_id TEXT,cited_work_id TEXT)"
    )
    runtime._paper_db.execute("INSERT INTO paper_edges VALUES ('W1','W2')")
    assert (
        runtime._paper_path(["https://openalex.org/W1"], "W2")["two_hop_path_count"]
        == 1
    )
    runtime.close()


def test_version_suspicion_is_not_author_overlap():
    work = RetrievedWork(
        work_id="W1",
        target_claim_id="c",
        title="Same research",
        authors=["A"],
        retrieval_query_id="q",
        retrieval_source="test",
    )
    assert (
        version_identity(work, PaperMetadata(title="Same research"))
        == "suspected_target_version"
    )
    assert (
        version_identity(work, PaperMetadata(title="Different research", authors=["A"]))
        is None
    )


def test_cross_boundary_uses_weight_not_count():
    from gear.review_contracts import GraphNeighbor

    neighbors = []
    for i, (community, weight) in enumerate([(1, 0.1), (1, 0.1), (2, 0.9)]):
        neighbors.append(
            GraphNeighbor(
                claim_id=str(i),
                parent_paper_id="p",
                claim_type=InnovationClaimType.FINDING,
                claim_text="x",
                publication_date=date(2025, 1, 1),
                cosine_similarity=weight,
                semantic_rank=i + 1,
                community_id=community,
            )
        )
    assert abs(ClaimGraphRuntime._cross_boundary_share(neighbors) - 0.2 / 1.1) < 1e-8


def test_partial_pair_statistics_are_not_complete_rarity():
    runtime = ClaimGraphRuntime(Path("."), Path("."))
    runtime._stats_db = sqlite3.connect(":memory:")
    runtime._stats_db.execute(
        "CREATE TABLE community_pair_history(community_a INTEGER,community_b INTEGER,pair_connector_count INTEGER,community_a_claim_count INTEGER,community_b_claim_count INTEGER,historical_claim_count INTEGER)"
    )
    runtime._stats_db.execute(
        "INSERT INTO community_pair_history VALUES(1,2,3,10,10,100)"
    )
    assert runtime._pair_surprisal([1, 2]) is not None
    assert runtime._pair_surprisal([1, 2, 3]) is None
    runtime.close()
