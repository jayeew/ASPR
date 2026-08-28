from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from gear.claim_attribution import (
    FEATURE_SCHEMA_VERSION,
    T0_FEATURE_NAMES,
    ClaimAttributionLinearHead,
    deterministic_t0_attribution,
    learned_t0_attribution,
    load_claim_attribution_release,
    promote_claim_attribution_release,
)
from gear.graph_prior_contracts import (
    ClaimInventoryEntry,
    ForecastAnatomy,
    GraphRuntimePacket,
    GraphSignalBundle,
    InfluenceForecast,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _packet(paper_id: str) -> GraphRuntimePacket:
    roles = {
        "substantive_innovation": 0.6,
        "t0_potential": 0.2,
        "opportunity": 0.1,
        "context": 0.1,
    }
    return GraphRuntimePacket(
        paper_id=paper_id,
        cutoff_date=date(2020, 1, 1),
        forecast=InfluenceForecast(
            status="available",
            prospective_5y_diffusion_percentile=60.0,
            uptake_probability=0.75,
            conditional_diffusion=0.8,
            expected_diffusion=0.6,
            feature_coverage=1.0,
            field_year_base=0.1,
            release_id="test-release",
            model_sha256="sha256:model",
            percentile_reference_sha256="sha256:reference",
        ),
        forecast_anatomy=ForecastAnatomy(
            paper_id=paper_id,
            uptake_role_contributions=roles,
            conditional_role_contributions=dict.fromkeys(roles, 0.0),
            role_coverage=dict.fromkeys(roles, 1.0),
        ),
    )


def _bundle(paper_id: str) -> GraphSignalBundle:
    return GraphSignalBundle(
        paper_id=paper_id,
        expected_diffusion=0.6,
        field_year_base=0.1,
        reliability=1.0,
        shrunk_diffusion=0.6,
    )


def _inventory() -> list[ClaimInventoryEntry]:
    return [
        ClaimInventoryEntry(
            claim_id="method",
            claim_type="method",
            text="We introduce a reusable framework.",
            manuscript_evidence_keys=["P:1"],
            centrality=1.0,
        ),
        ClaimInventoryEntry(
            claim_id="scope",
            claim_type="scope",
            text="The analysis is limited in scope.",
            manuscript_evidence_keys=["P:2"],
            centrality=0.3,
        ),
    ]


def _write_release_inputs(root: Path) -> tuple[Path, Path, list[Path]]:
    coefficients = [0.0] * len(T0_FEATURE_NAMES)
    coefficients[0] = 1.0
    model = ClaimAttributionLinearHead(
        feature_names=list(T0_FEATURE_NAMES), coefficients=coefficients
    )
    model_path = root / "model.json"
    model_path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    replay_path = root / "replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "paper_id": "P",
                        "features": [1.0] + [0.0] * (len(T0_FEATURE_NAMES) - 1),
                        "expected_raw_score": 1.0,
                        "expected_attribution_weight": 0.8,
                    },
                    {
                        "paper_id": "P",
                        "features": [0.25] + [0.0] * (len(T0_FEATURE_NAMES) - 1),
                        "expected_raw_score": 0.25,
                        "expected_attribution_weight": 0.2,
                    },
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    binding = {
        "model_sha256": _sha256(model_path),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "future_contexts_used_at_inference": False,
        "development_only": True,
        "training_split": "development",
        "sealed_holdout_labels_used": False,
        "holdout_labels_used_for_model_selection": False,
        "fold_local_target_fit": True,
        "future_features_used": False,
    }
    reports = []
    for name in ("temporal", "domain"):
        path = root / f"gate1_{name}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "claim_allowed": True,
                    "claim_attribution_runtime_candidate": {
                        **binding,
                        "evaluation_axis": name,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        reports.append(path)
    return model_path, replay_path, reports


def test_deterministic_t0_is_explicit_and_conserves_signal() -> None:
    inventory = _inventory()
    priors, audit = deterministic_t0_attribution(inventory, _bundle("P"), _packet("P"))

    assert audit.status == "available"
    assert audit.applied_mode == "deterministic_t0"
    assert audit.future_contexts_used_at_inference is False
    assert all(prior.attribution_method == "deterministic_t0" for prior in priors)
    assert priors[0].attribution_weight > priors[1].attribution_weight
    assert sum(prior.attribution_weight for prior in priors) == pytest.approx(1.0)
    assert sum(prior.diffusion_prior for prior in priors) == pytest.approx(0.6)


def test_learned_mode_without_promoted_asset_fails_closed() -> None:
    priors, audit = learned_t0_attribution(
        _inventory(), _bundle("P"), _packet("P"), None
    )

    assert priors == []
    assert audit.status == "limited"
    assert audit.applied_mode == "unavailable"
    assert "learned_head_unavailable" in audit.diagnostics[0]


def test_promoted_release_is_hash_bound_and_replayed(tmp_path: Path) -> None:
    model, replay, reports = _write_release_inputs(tmp_path)
    manifest = tmp_path / "release.json"
    promoted = promote_claim_attribution_release(
        model_path=model,
        replay_path=replay,
        gate1_report_paths=reports,
        output_path=manifest,
        release_id="claim-attribution-test-v1",
    )

    loaded, _ = load_claim_attribution_release(manifest)
    assert loaded == promoted
    assert loaded.sealed_holdout_labels_used is False
    model.write_text(model.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_claim_attribution_release(manifest)


def test_learned_runtime_uses_only_promoted_t0_head(tmp_path: Path) -> None:
    model, replay, reports = _write_release_inputs(tmp_path)
    manifest = tmp_path / "release.json"
    promote_claim_attribution_release(
        model_path=model,
        replay_path=replay,
        gate1_report_paths=reports,
        output_path=manifest,
        release_id="claim-attribution-test-v1",
    )

    priors, audit = learned_t0_attribution(
        _inventory(), _bundle("P"), _packet("P"), manifest
    )

    assert audit.status == "available"
    assert audit.applied_mode == "learned_t0"
    assert audit.release_id == "claim-attribution-test-v1"
    assert all(prior.attribution_method == "learned_t0" for prior in priors)
    assert sum(prior.attribution_weight for prior in priors) == pytest.approx(1.0)
    assert priors[0].attribution_weight == pytest.approx(1.0 / 1.3)


def test_promotion_rejects_unbound_gate1_result(tmp_path: Path) -> None:
    model, replay, reports = _write_release_inputs(tmp_path)
    reports[0].write_text(
        json.dumps({"status": "passed", "claim_allowed": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not bound to candidate model"):
        promote_claim_attribution_release(
            model_path=model,
            replay_path=replay,
            gate1_report_paths=reports,
            output_path=tmp_path / "release.json",
            release_id="invalid",
        )
