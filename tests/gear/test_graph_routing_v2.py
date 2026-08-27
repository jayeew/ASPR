from datetime import date

import pytest

from gear.contracts import ScientificSearchFrame
from gear.graph_guidance import (
    _assign_seeds,
    _citation_topology_mission,
    routing_weights,
    weighted_interleave,
)
from gear.graph_prior_contracts import (
    GraphRuntimePacket,
    InfluenceForecast,
    TopologySeed,
)
from gear.review_contracts import CanonicalReviewPoint, PointSeverity, ReviewAspect


def forecast(score: float, coverage: float = 1.0) -> InfluenceForecast:
    return InfluenceForecast(
        status="available",
        prospective_5y_diffusion_percentile=score,
        uptake_probability=0.5,
        conditional_diffusion=0.4,
        expected_diffusion=0.2,
        feature_coverage=coverage,
        release_id="release",
        model_sha256="sha256:model",
        percentile_reference_sha256="sha256:reference",
    )


def test_routing_requires_reliable_noncentral_forecast() -> None:
    local, remote, effective = routing_weights(forecast(100.0, 0.5), use_score=True)
    assert (local, remote, effective) == pytest.approx((0.5, 0.5, 0.5))
    strong = routing_weights(forecast(100.0), use_score=True)
    assert strong == pytest.approx((0.25, 0.75, 1.0))
    central = routing_weights(forecast(55.0), use_score=True)
    assert central == pytest.approx((0.5, 0.5, 0.5))
    neutral = routing_weights(None, use_score=True)
    assert neutral == pytest.approx((0.5, 0.5, 0.5))


def test_weighted_interleave_changes_order_not_pool_or_budget() -> None:
    local = ["l1", "l2", "l3", "l4"]
    remote = ["r1", "r2", "r3", "r4"]
    neutral = weighted_interleave(
        local, remote, local_weight=0.5, remote_weight=0.5, limit=8
    )
    scored = weighted_interleave(
        local, remote, local_weight=0.25, remote_weight=0.75, limit=8
    )
    assert len(neutral) == len(scored) == 8
    assert set(neutral) == set(scored) == set(local + remote)
    assert neutral != scored
    assert [row for row in scored if row.startswith("l")] == local
    assert [row for row in scored if row.startswith("r")] == remote


def test_claim_citation_is_an_explicit_topology_mission() -> None:
    mission = _citation_topology_mission("claim")

    assert mission.origin == "topology"
    assert mission.query_roles == ["author_citation"]
    assert mission.traversal == "none"


def test_runtime_packet_rejects_same_year_unknown_date_seed() -> None:
    cutoff = date(2021, 6, 1)
    with pytest.raises(ValueError, match="year-only topology"):
        GraphRuntimePacket(
            paper_id="paper",
            cutoff_date=cutoff,
            forecast=forecast(50.0),
            topology_seeds=[
                TopologySeed(
                    work_id="W1",
                    title="candidate",
                    publication_year=2020,
                    as_of_date=date(2020, 12, 31),
                    source_snapshot_id="snapshot",
                    source_snapshot_sha256="sha256:snapshot",
                    source_max_year=2021,
                    cutoff_valid=True,
                )
            ],
        )


def test_runtime_packet_accepts_conservative_year_only_seed() -> None:
    packet = GraphRuntimePacket(
        paper_id="paper",
        cutoff_date=date(2021, 6, 1),
        forecast=forecast(50.0),
        topology_seeds=[
            TopologySeed(
                work_id="W1",
                title="candidate",
                publication_year=2019,
                as_of_date=date(2020, 12, 31),
                source_snapshot_id="snapshot",
                source_snapshot_sha256="sha256:snapshot",
                source_max_year=2020,
                cutoff_valid=True,
            )
        ],
    )
    assert packet.topology_seeds[0].work_id == "W1"


def test_topology_seed_must_link_to_claim_citation_when_available() -> None:
    point = CanonicalReviewPoint(
        point_id="CP",
        section="novelty_limit",
        initial_section="novelty_limit",
        aspect=ReviewAspect.NOVELTY_PRIOR_ART,
        severity=PointSeverity.MINOR,
        proposition="Chromosome locus motion near the replisome is reduced.",
        agent_support=True,
        retained=True,
    )
    frame = ScientificSearchFrame(
        target_object=["chromosome locus"],
        task_problem=["replisome motion"],
        citation_seed_ids=["R-claim"],
        source_span_ids=["S-claim"],
    )
    common = {
        "title": "Chromosome locus and replisome motion",
        "publication_year": 2019,
        "as_of_date": date(2020, 12, 31),
        "source_snapshot_id": "snapshot",
        "source_snapshot_sha256": "sha256:snapshot",
        "source_max_year": 2020,
        "cutoff_valid": True,
    }
    unrelated = TopologySeed(
        work_id="W-unrelated",
        shared_reference_ids=["R-other", "R-other-2"],
        **common,
    )
    linked = TopologySeed(
        work_id="W-linked",
        shared_reference_ids=["R-claim", "R-other"],
        **common,
    )

    selected, relevance = _assign_seeds(
        [unrelated, linked], point, frame, seed_usage={}
    )

    assert [seed.work_id for seed in selected] == ["W-linked"]
    assert relevance > 0.0
