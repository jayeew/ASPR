"""Contract tests for the v6.1 candidate universe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from aspr.nature_multihorizon.candidate_registry_v6_1 import (
    EXPECTED_ANGLES,
    candidate_registry_sha256,
    load_candidate_registry_v6_1,
    verify_search_log,
)
from aspr.nature_multihorizon.screening_v6_1 import (
    freeze_registry_from_screening,
)
from aspr.nature_multihorizon.modeling_v6_1 import load_simple_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = (
    PROJECT_ROOT / "configs/innovation_candidate_catalog_v6_1.json"
)


def test_candidate_catalog_covers_five_angles_and_required_families() -> None:
    """The candidate catalog contains the promised non-cherry-picked scope."""
    registry = load_candidate_registry_v6_1(CATALOG_PATH)
    assert set(registry.observation_angles) == EXPECTED_ANGLES
    assert len(registry.candidates) == 50
    assert len(registry.sources) == 34
    families = {
        candidate.mathematical_family
        for candidate in registry.candidates.values()
    }
    assert {
        "commonness_lower_tail",
        "prior_reference_overlap_mean",
        "z_distribution_left_tail",
        "first_pair_incidence",
        "category_variety",
        "category_balance",
        "unweighted_field_distance",
        "share_weighted_integration_composite",
        "multiplicative_div_composite",
        "network_coherence",
        "semantic_reference_distance",
        "controlled_term_pair_volume_novelty",
        "content_context_surprise",
        "topic_cloud_novelty",
        "language_model_token_surprise",
        "new_component_incidence",
        "question_method_semantic_combination",
        "semantic_local_outlier",
        "contextual_embedding_semantic_gain",
    }.issubset(families)


def test_search_log_hash_and_oof_exclusion() -> None:
    """The literature log is frozen and no candidate used OOF selection."""
    registry = load_candidate_registry_v6_1(CATALOG_PATH)
    search_path = verify_search_log(registry, PROJECT_ROOT)
    assert search_path.is_file()
    search = json.loads(search_path.read_text(encoding="utf-8"))
    for record in search["search_records"]:
        assert {
            "search_id",
            "database",
            "query",
            "date",
            "page_or_sort",
            "result_titles_screened",
            "result_dois_screened",
            "deduplication",
            "decision",
        }.issubset(record)
        assert len(record["result_titles_screened"]) == len(
            record["result_dois_screened"]
        )
    assert all(
        candidate.oof_used_for_selection is False
        for candidate in registry.candidates.values()
    )


def test_every_k1_k2_control_has_evidence_and_role_boundary() -> None:
    """Every auxiliary feature is registered and separated from innovation."""
    config = load_simple_config(
        PROJECT_ROOT / "configs/nature_multihorizon/v6_1_simple.json"
    )
    control_registry = json.loads(
        (
            PROJECT_ROOT
            / config["paths"]["control_registry"]
        ).read_text(encoding="utf-8")
    )
    expected = set(config["k1_controls"]) | set(
        config["k2_additional_controls"]
    )
    assert set(control_registry["features"]) == expected
    assert "never be interpreted as paper-innovation indicators" in (
        control_registry["role_boundary"]
    )
    assert all(
        definition["source_ids"]
        for definition in control_registry["features"].values()
    )


def test_future_reuse_and_semantic_external_candidate_are_excluded() -> None:
    """Known leakage and unfrozen-model candidates cannot enter prediction."""
    registry = load_candidate_registry_v6_1(CATALOG_PATH)
    assert registry.candidates["A3.FUTURE_REUSED"].final_role == "excluded"
    assert registry.candidates["A5.SEMANTIC_DISTANCE"].final_role == "excluded"
    assert registry.candidates["A1.LLM_TOKEN_NLL"].final_role == "excluded"
    assert (
        registry.candidates["A3.QUESTION_METHOD_COMBINATION"].final_role
        == "excluded"
    )


def test_first_pair_distance_debt_is_excluded_until_coverage_passes() -> None:
    """The explicit technical-debt rule is stricter than sensitivity status."""
    registry = load_candidate_registry_v6_1(CATALOG_PATH)
    for candidate_id in (
        "A3.FIRST_DISTANCE_MEAN",
        "A3.FIRST_DISTANCE_SUM",
    ):
        candidate = registry.candidates[candidate_id]
        assert candidate.final_role == "excluded"
        assert "coverage" in candidate.decision_reason


def test_outcome_blind_screen_can_freeze_zero_error_metrics(
    tmp_path: Path,
) -> None:
    """The generated registry accepts exact zero stability/fidelity error."""
    catalog = load_candidate_registry_v6_1(CATALOG_PATH)
    primary_ids = {
        "A1.REFERENCE_OVERLAP",
        "A2.HYPERGEOM_MEDIAN",
        "A3.FIRST_SHARE",
        "A4.VARIETY",
        "A4.OTHER_FIELD_SHARE",
        "A4.GINI_BALANCE",
        "A5.MEAN_DISTANCE",
        "A5.RAO_STIRLING",
    }
    rows = []
    for candidate_id, candidate in catalog.candidates.items():
        rows.append(
            {
                "candidate_id": candidate_id,
                "total_n": 1000,
                "eligible_n": 800,
                "coverage_denominator_policy": (
                    "eligible_by_metric_family"
                ),
                "raw_overall_coverage": 0.8,
                "raw_minimum_domain_coverage": 0.6,
                "overall_coverage": 0.99,
                "minimum_domain_coverage": 0.95,
                "stability_spearman": 0.99,
                "stability_median_relative_error": 0.0,
                "relative_error_denominator_policy": (
                    "median_absolute_floor"
                ),
                "relative_error_scale_floor": 1.0,
                "approximation_spearman": (
                    0.99 if candidate_id in primary_ids else float("nan")
                ),
                "approximation_median_relative_error": (
                    0.0 if candidate_id in primary_ids else float("nan")
                ),
                "coverage_pass": 1,
                "stability_pass": 1,
                "approximation_pass": 1,
                "toy_test_pass": int(
                    candidate.empirical_screen.toy_test_pass
                ),
                "temporal_test_pass": int(
                    candidate.empirical_screen.temporal_test_pass
                ),
                "nondegenerate_test_pass": int(
                    candidate.implementation_name is not None
                ),
                "outcome_used": 0,
                "proposed_final_role": (
                    "primary"
                    if candidate_id in primary_ids
                    else candidate.final_role
                ),
                "proposed_decision_reason": (
                    "Synthetic outcome-blind contract test."
                ),
            }
        )
    decisions_path = tmp_path / "candidate_decisions.csv"
    pd.DataFrame(rows).to_csv(decisions_path, index=False)
    decisions_hash = (
        "sha256:"
        + hashlib.sha256(decisions_path.read_bytes()).hexdigest()
    )
    manifest = {
        "artifact_kind": (
            "aspr_v6_1_outcome_blind_candidate_screening"
        ),
        "artifact_id": "sha256:" + "1" * 64,
        "lineage": {
            "candidate_catalog_sha256": candidate_registry_sha256(
                catalog
            ),
            "future_influence_outcomes_used": False,
            "network_used": False,
        },
        "summary": {"future_influence_outcomes_used": False},
        "outputs": {
            "decisions": {
                "path": str(decisions_path),
                "sha256": decisions_hash,
            }
        },
    }
    manifest_path = tmp_path / "screening_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    output_path = tmp_path / "frozen_registry.json"
    frozen = freeze_registry_from_screening(
        project_root=PROJECT_ROOT,
        catalog_path=CATALOG_PATH,
        screening_manifest_path=manifest_path,
        output_path=output_path,
    )
    assert frozen.registry_stage.endswith("frozen_before_oof")
    assert set(frozen.primary_feature_names) == {
        catalog.candidates[item].code_name for item in primary_ids
    }
