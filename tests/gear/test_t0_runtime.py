from __future__ import annotations

import pandas as pd
import pytest

from gear.nature_multihorizon.features_v6 import (
    field_disparity_mean,
    rao_stirling_integration,
)
from gear.nature_multihorizon.features_v6_1 import field_gini_balance
from gear.nature_multihorizon.t0_runtime_v3 import (
    ContextSnapshot,
    ReferenceT0,
    TargetT0Record,
    build_context_snapshot,
    coerce_fulltext16_storage_schema,
    compute_reference_diversity,
    context_snapshot_sha256,
    load_context_snapshot,
    materialize_fulltext16,
    save_context_snapshot,
    validate_materialization_replay,
)


def test_reference_diversity_matches_frozen_formulas():
    fields = ["A", "A", "B", "C"]
    distances = {("A", "B"): 0.2, ("A", "C"): 0.8, ("B", "C"): 0.5}
    observed = compute_reference_diversity(fields, distances)
    assert observed["reference_balance"] == pytest.approx(field_gini_balance(fields))
    assert observed["reference_disparity"] == pytest.approx(
        field_disparity_mean(fields, distances)
    )
    assert observed["rao_stirling_diversity"] == pytest.approx(
        rao_stirling_integration(fields, distances)
    )


def test_context_snapshot_must_be_strictly_prior_and_materializes_16_fields():
    target = TargetT0Record(
        paper_id="W-target",
        publication_year=2020,
        title="A New Evidence Controller",
        author_ids=("A1", "A2"),
        country_codes=("US", "CN"),
        source_id="S1",
        references=(
            ReferenceT0("R1", 2010, "A"),
            ReferenceT0("R2", 2015, "B"),
        ),
    )
    invalid = ContextSnapshot(source_max_year=2020)
    with pytest.raises(ValueError, match="before the target"):
        materialize_fulltext16(target, invalid)
    context = ContextSnapshot(
        source_max_year=2019,
        field_distances={("A", "B"): 0.4},
    )
    values = materialize_fulltext16(target, context)
    assert len(values) == 16
    assert values["EF0197"] == "S1"
    submission = materialize_fulltext16(target, context, include_journal_identity=False)
    assert submission["EF0197"] is None


def test_frozen_author_and_country_missingness_semantics():
    context = ContextSnapshot(source_max_year=2019)
    target = TargetT0Record(
        paper_id="W-target",
        publication_year=2020,
        title="Target",
        author_ids=("A1",),
        author_count=3,
    )
    values = materialize_fulltext16(target, context)
    assert values["EF0038"] == 3.0
    assert values["EF0186"] == 0.0
    assert values["EF0188"] is None
    unavailable = materialize_fulltext16(
        target.__class__(
            paper_id="W-unavailable",
            publication_year=2020,
            title="Unavailable",
            metadata_observed=False,
        ),
        context,
    )
    assert unavailable["EF0186"] is None


def test_fulltext16_storage_schema_uses_frozen_float32_contract():
    names = [
        "EF0017",
        "EF0038",
        "EF0052",
        "EF0083",
        "EF0186",
        "EF0188",
        "EF0238",
        "EF0240",
        "EF0307",
        "EF0309",
        "EF0312",
        "EF0314",
        "EF0315",
        "EF0318",
        "EF0319",
    ]
    frame = pd.DataFrame([{**{name: 1.0 / 3.0 for name in names}, "EF0197": "S1"}])
    coerced = coerce_fulltext16_storage_schema(frame)
    assert all(str(coerced[name].dtype) == "float32" for name in names)
    assert str(coerced["EF0197"].dtype) == "string"


def test_snapshot_resolves_reference_metadata_and_excludes_nonprior_references():
    prior = TargetT0Record(
        paper_id="W-prior",
        publication_year=2019,
        title="Prior evidence graph",
        references=(
            ReferenceT0("R1", 2010, "A"),
            ReferenceT0("R2", 2011, "B"),
        ),
    )
    context = build_context_snapshot(
        [prior],
        target_year=2020,
        field_distances={("A", "B"): 0.4},
    )
    target = TargetT0Record(
        paper_id="W-target",
        publication_year=2020,
        title="Target evidence graph",
        references=(
            ReferenceT0("R1"),
            ReferenceT0("R2"),
            ReferenceT0("R-same-year", 2020, "C"),
            ReferenceT0("R-future", 2021, "D"),
        ),
    )

    values = materialize_fulltext16(target, context)

    assert context.prior_titles == ["Prior evidence graph"]
    assert values["EF0314"] == 2.0
    assert values["EF0318"] == 2.0
    assert values["EF0238"] == 0.5
    # The formal backward-age feature admits same-year age zero but rejects future work.
    assert values["EF0052"] == pytest.approx((10 + 9 + 0) / 3)


def test_snapshot_roundtrip_and_replay_gate(tmp_path):
    context = ContextSnapshot(
        source_max_year=2019,
        seen_title_bigrams={("evidence", "review")},
        prior_author_adjacency={"A": {"B"}, "B": {"A"}},
        prior_coauthor_weights={("A", "B"): 2},
        prior_paper_reference_ids=[{"R1", "R2"}],
        field_distances={("A", "B"): 0.4},
    )
    path = save_context_snapshot(context, tmp_path / "context.json")
    restored = load_context_snapshot(path)
    assert context_snapshot_sha256(restored) == context_snapshot_sha256(context)

    names = [
        "EF0017",
        "EF0038",
        "EF0052",
        "EF0083",
        "EF0186",
        "EF0188",
        "EF0197",
        "EF0238",
        "EF0240",
        "EF0307",
        "EF0309",
        "EF0312",
        "EF0314",
        "EF0315",
        "EF0318",
        "EF0319",
    ]
    row = {"paper_id": "W1", **{name: 0.1 for name in names}}
    row["EF0197"] = "S1"
    frozen = pd.DataFrame([row])
    report = validate_materialization_replay(
        frozen.copy(),
        frozen,
        prediction_func=lambda frame: frame.drop(columns=["EF0197"]).sum(axis=1),
    )
    assert report.eligible_inference is True

    changed = frozen.copy()
    changed.loc[0, "EF0017"] = 9.0
    failed = validate_materialization_replay(
        changed,
        frozen,
        prediction_func=lambda frame: frame.drop(columns=["EF0197"]).sum(axis=1),
    )
    assert failed.eligible_inference is False
