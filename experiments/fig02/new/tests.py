"""Contract tests for the evidence-derived v3 Fig.2."""

from pathlib import Path

from experiments.common.new.adapters.builders import build_new_bundle
from experiments.common.new.adapters.runtime import load_figure_context
from experiments.common.new.adapters.tests_support import run_tests


def _assert_fig2_contract(config_path: Path) -> None:
    """Check the outcome-free v3 evidence contract before rendering."""
    config, paths, _ = load_figure_context(config_path)
    bundle = build_new_bundle(2, config, paths)
    assert bundle.status == "complete_evidence_derived_tuned_recovery"
    forbidden = ("oof", "future", "known_group", "measurement_scene")
    assert not [
        table
        for table in bundle.tables
        if any(token in table.lower() for token in forbidden)
    ]
    process = bundle.tables["fig2_process_stages"]
    assert dict(zip(process["label"], process["count"].astype(int))) == {
        "Raw English terms": 3615,
        "H2 retained": 3170,
        "Canonical terms": 1102,
        "Term families": 367,
        "Search domains": 42,
        "Candidate logical queries": 381,
        "Frozen search frame": 336,
        "Seed + PRESS validation": 62,
        "Formal query records": 30332,
        "Final title/abstract dispositions": 9515,
        "Indicator mentions": 1685,
        "Canonical indicator families": 432,
        "Mapped families": 428,
        "Candidate dimensions": 66,
    }
    rounds = bundle.tables["fig2_round_yields"].sort_values("iteration")
    assert len(rounds) == 12
    assert int(rounds.iloc[-1]["new_nonredundant_english_terms"]) == 10
    assert int(rounds.iloc[-1]["new_canonical_indicator_families"]) == 9
    assert str(rounds.iloc[-1]["stop_basis"]) == "retrospective_owner_pragmatic_stop"
    query = bundle.tables["fig2_query_audit"].iloc[0]
    assert [
        int(query["candidate_logical_queries"]),
        int(query["active_logical_queries"]),
        int(query["physical_openalex_requests"]),
        int(query["zero_hit_archived"]),
        int(query["press_unsupported_archived"]),
        int(query["redundant_archived"]),
        int(query["archived_queries"]),
        int(query["press_unresolved_active"]),
    ] == [381, 336, 367, 13, 9, 23, 45, 0]
    recall = bundle.tables["fig2_recall_audit"].iloc[0]
    assert [
        int(recall["initial_recalled_seed_count"]),
        int(recall["initial_seed_denominator"]),
        int(recall["source_grounded_repairs"]),
        int(recall["recalled_seed_count"]),
        int(recall["indexable_seed_count"]),
        int(recall["press_unresolved"]),
    ] == [51, 62, 10, 62, 62, 0]
    screening = process.loc[
        process["label"].eq("Final title/abstract dispositions")
    ].iloc[0]
    assert [
        int(screening["count"]),
        int(screening["included_count"]),
        int(screening["excluded_count"]),
    ] == [9515, 363, 9152]
    mentions = process.loc[process["label"].eq("Indicator mentions")].iloc[0]
    assert int(mentions["count"]) == 1685
    assert "+13 targeted formula completions" in str(mentions["detail"])
    mapping = bundle.tables["fig2_indicator_family_mapping"]
    assert len(mapping) == 432
    assert int(mapping["dimension_id"].notna().sum()) == 428
    assert mapping["tier"].value_counts().to_dict() == {
        "excluded": 213,
        "source_only": 137,
        "broad_t0_only": 66,
        "fulltext_only": 9,
        "strict_core": 7,
    }
    assert int(mapping["tier"].value_counts().sum()) == 432
    strict = bundle.tables["fig2_strict_mapping"]
    assert len(strict) == 4
    assert int(strict["indicator_count"].sum()) == 7
    assert strict["final_role"].value_counts().to_dict() == {
        "control": 2,
        "predictive": 1,
        "opportunity": 1,
    }
    sets = bundle.tables["fig2_feature_sets"].sort_values("set_order")
    assert sets["feature_count"].astype(int).tolist() == [7, 16, 153, 219]
    assert sets["dimension_count"].astype(int).tolist() == [4, 10, 48, 55]
    source_tiers = bundle.tables["fig2_operationalization_tiers"]
    source_tiers = source_tiers.loc[source_tiers["set_id"].eq("source_154")]
    assert source_tiers.sort_values("tier_order")["feature_count"].astype(int).tolist() == [12, 4, 78, 59]
    broad_tiers = bundle.tables["fig2_operationalization_tiers"]
    broad_tiers = broad_tiers.loc[broad_tiers["set_id"].eq("ultrarelaxed_221")]
    assert broad_tiers.sort_values("tier_order")["feature_count"].astype(int).tolist() == [12, 4, 88, 115]
    review = bundle.tables["fig2_review_coverage"].iloc[0]
    assert int(review["excluded_local_qwen_artifact_count"]) == 3
    assert int(review["human_attested_worksheet_count"]) == 7
    assert int(review["independent_ai_run_count"]) == 119
    membership_sources = {
        Path(path).name
        for path in bundle.source_paths
        if Path(path).name in {"run_manifest.json", "recovery_manifest_v3.json"}
    }
    assert membership_sources == {"run_manifest.json", "recovery_manifest_v3.json"}
    assert bundle.chart_contract["future_data_used"] is False
    assert bundle.chart_contract["oof_data_used"] is False
    assert bundle.chart_contract["outcome_used_for_indicator_selection"] is False


if __name__ == "__main__":
    figure_config = Path(__file__).with_name("config.json")
    _assert_fig2_contract(figure_config)
    run_tests(2, figure_config)
