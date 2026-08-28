"""Selective Graph action policy with an explicit no-uplift abstention rule."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import StrictModel
from .graph_prior_contracts import GraphActionDecision

GraphAction = Literal[
    "antecedent_falsification",
    "remote_mechanism_analogue",
    "cross_field_pathway",
    "topology_expansion",
    "opportunity_attribution_audit",
]
GRAPH_ACTIONS: tuple[GraphAction, ...] = (
    "antecedent_falsification",
    "remote_mechanism_analogue",
    "cross_field_pathway",
    "topology_expansion",
    "opportunity_attribution_audit",
)
RandomizedGraphAction = Literal[
    "baseline",
    "antecedent_falsification",
    "remote_mechanism_analogue",
    "cross_field_pathway",
    "topology_expansion",
    "opportunity_attribution_audit",
]
ALL_ACTIONS: tuple[RandomizedGraphAction, ...] = ("baseline", *GRAPH_ACTIONS)
ACTION_POLICY_FEATURE_SCHEMA: Literal["gear_graph_action_policy_t0_features_v1"] = (
    "gear_graph_action_policy_t0_features_v1"
)
ACTION_POLICY_FEATURES = (
    "claim_count",
    "mean_claim_centrality",
    "publication_year",
    "graph_shrunk_diffusion",
    "graph_reliability",
    "graph_structural_share",
    "graph_opportunity_share",
    "graph_perturbation_potential",
    "graph_prediction_uncertainty",
)


class ActionPolicyRule(StrictModel):
    uplift_margin: float = Field(ge=0.0, allow_inf_nan=False)
    development_rows: Literal[15]
    development_average_uplift: float = Field(allow_inf_nan=False)
    development_average_uplift_lcb: float = Field(allow_inf_nan=False)
    development_positive_uplift_pass: bool
    wrong_correction_pass: bool
    unsupported_claim_pass: bool
    cost_pass: bool

    @property
    def guardrails_pass(self) -> bool:
        return bool(
            self.wrong_correction_pass
            and self.unsupported_claim_pass
            and self.cost_pass
            and self.development_positive_uplift_pass
        )


class GraphActionQModel(StrictModel):
    """Portable Q heads; loading cannot execute serialized Python."""

    contract: Literal["gear_graph_action_q_linear_v1"] = "gear_graph_action_q_linear_v1"
    feature_schema_version: Literal["gear_graph_action_policy_t0_features_v1"]
    feature_family: Literal["graph_features"]
    feature_names: list[str]
    intercepts: dict[str, float]
    coefficients: dict[str, list[float]]
    rules: dict[str, ActionPolicyRule]
    selection_rule: Literal["max_positive_q_minus_baseline_minus_uplift_margin_v1"]
    tie_break: Literal["uplift_lcb_then_uplift_then_action_lexicographic_v1"]
    fallback_action: Literal["abstain"]
    future_features_used: Literal[False]
    training_rows: Literal[90]
    training_scope: Literal["development_only"]
    sealed_holdout_used_for_fitting: Literal[False]
    gear_evidence_gap_status: Literal[
        "phase_one_excluded_not_available_at_pre_retrieval_decision"
    ]

    @model_validator(mode="after")
    def exact_runtime_contract(self) -> GraphActionQModel:
        if tuple(self.feature_names) != ACTION_POLICY_FEATURES:
            raise ValueError("action policy does not use the frozen T0 feature schema")
        if set(self.intercepts) != set(ALL_ACTIONS):
            raise ValueError("action policy intercepts must cover A0-A5 exactly")
        if set(self.coefficients) != set(ALL_ACTIONS):
            raise ValueError("action policy Q heads must cover A0-A5 exactly")
        if set(self.rules) != set(GRAPH_ACTIONS):
            raise ValueError("action policy rules must cover A1-A5 exactly")
        numeric = [*self.intercepts.values()]
        for values in self.coefficients.values():
            if len(values) != len(self.feature_names):
                raise ValueError("action policy coefficient count mismatch")
            numeric.extend(values)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("action policy model contains non-finite values")
        return self

    def predict(self, features: Sequence[float]) -> dict[str, float]:
        if len(features) != len(self.feature_names) or not all(
            math.isfinite(float(value)) for value in features
        ):
            raise ValueError("action policy runtime feature vector is invalid")
        return {
            action: float(self.intercepts[action])
            + sum(
                coefficient * float(value)
                for coefficient, value in zip(
                    self.coefficients[action], features, strict=True
                )
            )
            for action in ALL_ACTIONS
        }

    def decision(self, features: Sequence[float]) -> GraphActionDecision:
        q_values = self.predict(features)
        baseline = q_values["baseline"]
        eligible: list[tuple[float, float, GraphAction]] = []
        safe_action_exists = False
        for action in GRAPH_ACTIONS:
            rule = self.rules[action]
            if not rule.guardrails_pass:
                continue
            safe_action_exists = True
            uplift = q_values[action] - baseline
            lcb = uplift - rule.uplift_margin
            if lcb > 0.0:
                eligible.append((lcb, uplift, action))
        if not eligible:
            return GraphActionDecision(
                action="abstain",
                predicted_uplift=max(
                    (q_values[action] - baseline for action in GRAPH_ACTIONS),
                    default=0.0,
                ),
                uplift_lcb=0.0,
                selected=False,
                reason=(
                    "uplift_lcb_nonpositive"
                    if safe_action_exists
                    else "guardrail_failed"
                ),
                policy_status="available",
            )
        lcb, uplift, selected = max(eligible)
        return GraphActionDecision(
            action=selected,
            predicted_uplift=uplift,
            uplift_lcb=lcb,
            selected=True,
            reason="positive_uplift_lcb_and_guardrails_passed",
            policy_status="available",
        )


class GraphActionPolicyRelease(StrictModel):
    """Hash-bound release admitted only after complete Stage C and Gate 2."""

    contract: Literal["gear_graph_action_policy_release_v1"] = (
        "gear_graph_action_policy_release_v1"
    )
    release_id: str
    status: Literal["promoted"]
    feature_schema_version: Literal["gear_graph_action_policy_t0_features_v1"]
    feature_family: Literal["graph_features"]
    feature_names: list[str]
    q_model_family: Literal["linear_t0_v1"]
    model_path: str
    model_sha256: str
    replay_path: str
    replay_sha256: str
    development_data_path: str
    development_data_sha256: str
    development_rows: Literal[90]
    randomized_data_path: str
    randomized_data_sha256: str
    randomized_rows: Literal[150]
    graph_policy_path: str
    graph_policy_sha256: str
    no_graph_policy_path: str
    no_graph_policy_sha256: str
    gate2_report_path: str
    gate2_report_sha256: str
    frozen_replay_manifest_path: str
    frozen_replay_manifest_sha256: str
    source_fingerprint_audit_path: str
    source_fingerprint_audit_sha256: str
    stage_a_runtime_audit_path: str
    stage_a_runtime_audit_sha256: str
    stage_b_runtime_audit_path: str
    stage_b_runtime_audit_sha256: str
    stage_c_runtime_audit_path: str
    stage_c_runtime_audit_sha256: str
    future_features_used: Literal[False]
    future_outcomes_used_at_inference: Literal[False]
    sealed_holdout_used_for_fitting: Literal[False]
    gear_evidence_gap_status: Literal[
        "phase_one_excluded_not_available_at_pre_retrieval_decision"
    ]

    @model_validator(mode="after")
    def exact_schema(self) -> GraphActionPolicyRelease:
        if tuple(self.feature_names) != ACTION_POLICY_FEATURES:
            raise ValueError("action policy release has the wrong feature schema")
        references = (
            self.model_path,
            self.replay_path,
            self.development_data_path,
            self.randomized_data_path,
            self.graph_policy_path,
            self.no_graph_policy_path,
            self.gate2_report_path,
            self.frozen_replay_manifest_path,
            self.source_fingerprint_audit_path,
            self.stage_a_runtime_audit_path,
            self.stage_b_runtime_audit_path,
            self.stage_c_runtime_audit_path,
        )
        if len(set(references)) != len(references):
            raise ValueError("action policy release references must be distinct")
        return self


class FrozenGraphActionSelector:
    """Lazy production selector that abstains when its release cannot verify."""

    def __init__(self, manifest_path: Path | None) -> None:
        self.manifest_path = manifest_path
        self._loaded: tuple[GraphActionPolicyRelease, GraphActionQModel] | None = None
        self._failure: str | None = None

    def decide(self, state: object) -> GraphActionDecision:
        try:
            release, model = self._load()
            features = action_policy_t0_features(state)
            decision = model.decision(features)
            return decision.model_copy(
                update={
                    "policy_release_id": release.release_id,
                    "policy_manifest_sha256": _sha256(self.manifest_path),
                }
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._failure = f"{type(exc).__name__}:{exc}"
            return GraphActionDecision(
                action="abstain",
                predicted_uplift=0.0,
                uplift_lcb=0.0,
                selected=False,
                reason=f"action_policy_unavailable:{type(exc).__name__}",
                policy_status="limited",
            )

    def _load(self) -> tuple[GraphActionPolicyRelease, GraphActionQModel]:
        if self._loaded is not None:
            return self._loaded
        if self._failure is not None:
            raise ValueError(self._failure)
        if self.manifest_path is None:
            raise ValueError("action_policy_manifest_unconfigured")
        self._loaded = load_graph_action_policy_release(self.manifest_path)
        return self._loaded


class GraphActionPolicy:
    """Choose an action only when its conservative uplift clears baseline."""

    def decide(
        self,
        uplift: Mapping[str, float],
        uplift_lcb: Mapping[str, float],
        *,
        guardrails_pass: bool,
        propensities: Mapping[str, float] | None = None,
    ) -> GraphActionDecision:
        unknown = sorted(set(uplift) - set(GRAPH_ACTIONS))
        if unknown:
            raise ValueError(f"unknown Graph actions: {unknown}")
        if not guardrails_pass:
            return GraphActionDecision(
                action="abstain",
                predicted_uplift=0.0,
                uplift_lcb=0.0,
                selected=False,
                reason="guardrail_failed",
            )
        eligible = [
            action
            for action in GRAPH_ACTIONS
            if float(uplift_lcb.get(action, float("-inf"))) > 0.0
        ]
        if not eligible:
            return GraphActionDecision(
                action="abstain",
                predicted_uplift=max(
                    (float(value) for value in uplift.values()), default=0.0
                ),
                uplift_lcb=max(
                    (float(value) for value in uplift_lcb.values()), default=0.0
                ),
                selected=False,
                reason="uplift_lcb_nonpositive",
            )
        selected = max(
            eligible,
            key=lambda action: (
                float(uplift_lcb[action]),
                float(uplift.get(action, 0.0)),
                action,
            ),
        )
        propensity = None if propensities is None else propensities.get(selected)
        return GraphActionDecision(
            action=selected,
            predicted_uplift=float(uplift.get(selected, 0.0)),
            uplift_lcb=float(uplift_lcb[selected]),
            propensity=propensity,
            selected=True,
            reason="positive_uplift_lcb_and_guardrails_passed",
        )


class RandomizedGraphActionSelector:
    """Inject one preassigned A0-A5 action with its known logging propensity."""

    def __init__(self, action: RandomizedGraphAction, propensity: float) -> None:
        if not 0.0 < propensity <= 1.0:
            raise ValueError("randomized action propensity must be in (0, 1]")
        self.action = action
        self.propensity = propensity

    def decide(self, state: object) -> GraphActionDecision:
        del state
        return GraphActionDecision(
            action=self.action,
            predicted_uplift=0.0,
            uplift_lcb=0.0,
            propensity=self.propensity,
            selected=self.action != "baseline",
            reason="preassigned_randomized_action",
            policy_status="experimental",
        )


def action_policy_t0_features(state: object) -> list[float]:
    """Project only predeclared manuscript-time and frozen Graph values."""
    inventory = list(getattr(state, "claim_inventory", []) or [])
    bundle = getattr(state, "graph_signal_bundle", None)
    cutoff = getattr(state, "cutoff_date", None)
    if bundle is None or cutoff is None:
        raise ValueError("action policy T0 state is incomplete")
    centrality = [float(item.centrality) for item in inventory]
    values = [
        float(len(inventory)),
        sum(centrality) / len(centrality) if centrality else 0.0,
        float(cutoff.year),
        _required_float(bundle.shrunk_diffusion, "graph_shrunk_diffusion"),
        _required_float(bundle.reliability, "graph_reliability"),
        _required_float(bundle.structural_contribution_share, "graph_structural_share"),
        _required_float(bundle.opportunity_context_share, "graph_opportunity_share"),
        _required_float(bundle.perturbation_potential, "graph_perturbation_potential"),
        _prediction_uncertainty(state, bundle),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("action policy T0 features are non-finite")
    return values


def load_graph_action_policy_release(
    manifest_path: Path,
) -> tuple[GraphActionPolicyRelease, GraphActionQModel]:
    """Verify complete Stage C, Gate 2, hashes, and replay before use."""
    path = Path(manifest_path).resolve()
    release = GraphActionPolicyRelease.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    references = _release_paths(path, release)
    expected = {
        "model": release.model_sha256,
        "replay": release.replay_sha256,
        "development": release.development_data_sha256,
        "randomized": release.randomized_data_sha256,
        "graph_policy": release.graph_policy_sha256,
        "no_graph_policy": release.no_graph_policy_sha256,
        "gate2": release.gate2_report_sha256,
        "frozen_replay": release.frozen_replay_manifest_sha256,
        "source_audit": release.source_fingerprint_audit_sha256,
        "stage_a_runtime_audit": release.stage_a_runtime_audit_sha256,
        "stage_b_runtime_audit": release.stage_b_runtime_audit_sha256,
        "stage_c_runtime_audit": release.stage_c_runtime_audit_sha256,
    }
    for key, target in references.items():
        _require_hash(target, expected[key])
    model = GraphActionQModel.model_validate_json(
        references["model"].read_text(encoding="utf-8")
    )
    _verify_stage_c_data(references["development"], references["randomized"])
    _verify_policy_pair(
        references["graph_policy"],
        references["no_graph_policy"],
        development_sha256=release.development_data_sha256,
        model=model,
    )
    _verify_gate2_report(
        references["gate2"],
        model_sha256=release.model_sha256,
        development_sha256=release.development_data_sha256,
        randomized_sha256=release.randomized_data_sha256,
        graph_policy_sha256=release.graph_policy_sha256,
        no_graph_policy_sha256=release.no_graph_policy_sha256,
    )
    _verify_frozen_runtime(
        references["frozen_replay"],
        references["source_audit"],
        references["stage_a_runtime_audit"],
        references["stage_b_runtime_audit"],
        references["stage_c_runtime_audit"],
    )
    validate_graph_action_policy_replay(path, release=release, model=model)
    return release, model


def validate_graph_action_policy_replay(
    manifest_path: Path,
    *,
    release: GraphActionPolicyRelease | None = None,
    model: GraphActionQModel | None = None,
) -> dict[str, Any]:
    """Replay frozen Q values and action decisions exactly."""
    path = Path(manifest_path).resolve()
    if release is None or model is None:
        release, model = load_graph_action_policy_release(path)
        return {
            "contract": "gear_graph_action_policy_replay_validation_v1",
            "passed": True,
            "release_id": release.release_id,
            "rows": _replay_row_count(path, release),
        }
    replay_path = _bound_path(path, release.replay_path)
    _require_hash(replay_path, release.replay_sha256)
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("action policy replay is empty")
    max_error = 0.0
    for row in rows:
        features = row.get("features")
        if not isinstance(features, list):
            raise TypeError("action policy replay features are invalid")
        observed_q = model.predict(features)
        expected_q = row.get("expected_q_values")
        if not isinstance(expected_q, dict) or set(expected_q) != set(ALL_ACTIONS):
            raise ValueError("action policy replay Q values do not cover A0-A5")
        max_error = max(
            max_error,
            max(
                abs(observed_q[action] - float(expected_q[action]))
                for action in ALL_ACTIONS
            ),
        )
        expected_decision = GraphActionDecision.model_validate(
            row.get("expected_decision")
        )
        observed_decision = model.decision(features)
        if observed_decision != expected_decision:
            raise ValueError("action policy replay decision mismatch")
    if max_error > 1e-12:
        raise ValueError("action policy replay Q-value mismatch")
    return {
        "contract": "gear_graph_action_policy_replay_validation_v1",
        "passed": True,
        "release_id": release.release_id,
        "rows": len(rows),
        "max_abs_q_error": max_error,
    }


def promote_graph_action_policy_release(
    *,
    model_path: Path,
    replay_path: Path,
    development_data_path: Path,
    randomized_data_path: Path,
    graph_policy_path: Path,
    no_graph_policy_path: Path,
    gate2_report_path: Path,
    frozen_replay_manifest_path: Path,
    source_fingerprint_audit_path: Path,
    stage_a_runtime_audit_path: Path,
    stage_b_runtime_audit_path: Path,
    stage_c_runtime_audit_path: Path,
    output_path: Path,
    release_id: str,
) -> GraphActionPolicyRelease:
    """Publish only after the exact complete Stage C and paired Gate 2 pass."""
    if output_path.parent.exists():
        raise FileExistsError(
            f"immutable action policy release directory exists: {output_path.parent}"
        )
    model = GraphActionQModel.model_validate_json(
        model_path.read_text(encoding="utf-8")
    )
    model_hash = _sha256(model_path)
    development_hash = _sha256(development_data_path)
    randomized_hash = _sha256(randomized_data_path)
    graph_policy_hash = _sha256(graph_policy_path)
    no_graph_policy_hash = _sha256(no_graph_policy_path)
    _verify_stage_c_data(development_data_path, randomized_data_path)
    _verify_policy_pair(
        graph_policy_path,
        no_graph_policy_path,
        development_sha256=development_hash,
        model=model,
    )
    _verify_gate2_report(
        gate2_report_path,
        model_sha256=model_hash,
        development_sha256=development_hash,
        randomized_sha256=randomized_hash,
        graph_policy_sha256=graph_policy_hash,
        no_graph_policy_sha256=no_graph_policy_hash,
    )
    _verify_frozen_runtime(
        frozen_replay_manifest_path,
        source_fingerprint_audit_path,
        stage_a_runtime_audit_path,
        stage_b_runtime_audit_path,
        stage_c_runtime_audit_path,
    )
    sources = {
        "model": model_path,
        "replay": replay_path,
        "development": development_data_path,
        "randomized": randomized_data_path,
        "graph_policy": graph_policy_path,
        "no_graph_policy": no_graph_policy_path,
        "gate2": gate2_report_path,
        "frozen_replay": frozen_replay_manifest_path,
        "source_audit": source_fingerprint_audit_path,
        "stage_a_runtime_audit": stage_a_runtime_audit_path,
        "stage_b_runtime_audit": stage_b_runtime_audit_path,
        "stage_c_runtime_audit": stage_c_runtime_audit_path,
    }
    output_path.parent.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_path.parent.with_name(
        f".{output_path.parent.name}.tmp-{uuid.uuid4().hex}"
    )
    staging_dir.mkdir()
    staging_output = staging_dir / output_path.name
    try:
        published = _publish_release_artifacts(staging_output, sources)
        release = _build_release(
            staging_output,
            published,
            release_id=release_id,
            model_hash=model_hash,
            development_hash=development_hash,
            randomized_hash=randomized_hash,
            graph_policy_hash=graph_policy_hash,
            no_graph_policy_hash=no_graph_policy_hash,
        )
        staging_output.write_text(
            release.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        load_graph_action_policy_release(staging_output)
        staging_dir.rename(output_path.parent)
    except (OSError, TypeError, ValueError):
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return release


def _build_release(
    output_path: Path,
    published: Mapping[str, Path],
    *,
    release_id: str,
    model_hash: str,
    development_hash: str,
    randomized_hash: str,
    graph_policy_hash: str,
    no_graph_policy_hash: str,
) -> GraphActionPolicyRelease:
    return GraphActionPolicyRelease(
        release_id=release_id,
        status="promoted",
        feature_schema_version=ACTION_POLICY_FEATURE_SCHEMA,
        feature_family="graph_features",
        feature_names=list(ACTION_POLICY_FEATURES),
        q_model_family="linear_t0_v1",
        model_path=_relative_reference(output_path, published["model"]),
        model_sha256=model_hash,
        replay_path=_relative_reference(output_path, published["replay"]),
        replay_sha256=_sha256(published["replay"]),
        development_data_path=_relative_reference(
            output_path, published["development"]
        ),
        development_data_sha256=development_hash,
        development_rows=90,
        randomized_data_path=_relative_reference(output_path, published["randomized"]),
        randomized_data_sha256=randomized_hash,
        randomized_rows=150,
        graph_policy_path=_relative_reference(output_path, published["graph_policy"]),
        graph_policy_sha256=graph_policy_hash,
        no_graph_policy_path=_relative_reference(
            output_path, published["no_graph_policy"]
        ),
        no_graph_policy_sha256=no_graph_policy_hash,
        gate2_report_path=_relative_reference(output_path, published["gate2"]),
        gate2_report_sha256=_sha256(published["gate2"]),
        frozen_replay_manifest_path=_relative_reference(
            output_path, published["frozen_replay"]
        ),
        frozen_replay_manifest_sha256=_sha256(published["frozen_replay"]),
        source_fingerprint_audit_path=_relative_reference(
            output_path, published["source_audit"]
        ),
        source_fingerprint_audit_sha256=_sha256(published["source_audit"]),
        stage_a_runtime_audit_path=_relative_reference(
            output_path, published["stage_a_runtime_audit"]
        ),
        stage_a_runtime_audit_sha256=_sha256(published["stage_a_runtime_audit"]),
        stage_b_runtime_audit_path=_relative_reference(
            output_path, published["stage_b_runtime_audit"]
        ),
        stage_b_runtime_audit_sha256=_sha256(published["stage_b_runtime_audit"]),
        stage_c_runtime_audit_path=_relative_reference(
            output_path, published["stage_c_runtime_audit"]
        ),
        stage_c_runtime_audit_sha256=_sha256(published["stage_c_runtime_audit"]),
        future_features_used=False,
        future_outcomes_used_at_inference=False,
        sealed_holdout_used_for_fitting=False,
        gear_evidence_gap_status=(
            "phase_one_excluded_not_available_at_pre_retrieval_decision"
        ),
    )


def _publish_release_artifacts(
    output_path: Path, sources: Mapping[str, Path]
) -> dict[str, Path]:
    names = {
        "model": "graph_action_q_model.json",
        "replay": "graph_action_policy_replay.json",
        "development": "development90.parquet",
        "randomized": "randomized150.parquet",
        "graph_policy": "graph_policy_holdout60.parquet",
        "no_graph_policy": "no_graph_policy_holdout60.parquet",
        "gate2": "gate2_report.json",
        "frozen_replay": "frozen_replay_manifest.json",
        "source_audit": "source_fingerprint_audit.json",
        "stage_a_runtime_audit": "stage_a_runtime_audit.json",
        "stage_b_runtime_audit": "stage_b_runtime_audit.json",
        "stage_c_runtime_audit": "stage_c_runtime_audit.json",
    }
    targets = {key: output_path.parent / name for key, name in names.items()}
    for key, source in sources.items():
        shutil.copy2(source, targets[key])
    return targets


def _verify_stage_c_data(development_path: Path, randomized_path: Path) -> None:
    development = _read_frame(development_path)
    randomized = _read_frame(randomized_path)
    if len(randomized) != 150 or len(development) != 90:
        raise ValueError("action policy promotion requires complete 150/90 data")
    required = {
        "paper_id",
        "context_id",
        "experiment_split",
        "logged_action",
        "propensity",
        "matched_budget",
        "policy_fold_id",
    }
    for label, frame in (("randomized", randomized), ("development", development)):
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"{label} action data columns are missing: {missing}")
        if (
            frame["paper_id"].astype(str).duplicated().any()
            or frame["context_id"].astype(str).duplicated().any()
        ):
            raise ValueError(f"{label} action data contains duplicate identities")
    splits = randomized["experiment_split"].astype(str).value_counts().to_dict()
    if splits != {"development": 90, "confirmatory_holdout": 60}:
        raise ValueError(f"randomized action split counts are invalid: {splits}")
    if not development["experiment_split"].astype(str).eq("development").all():
        raise ValueError("development action data contains holdout rows")
    if not randomized["propensity"].astype(float).eq(1.0 / 6.0).all():
        raise ValueError("randomized action propensities are invalid")
    if not randomized["matched_budget"].astype(float).eq(20.0).all():
        raise ValueError("randomized action budgets are invalid")
    holdout = randomized[
        randomized["experiment_split"].astype(str).eq("confirmatory_holdout")
    ]
    if not holdout["policy_fold_id"].astype(str).eq("holdout").all():
        raise ValueError("randomized holdout policy folds are invalid")
    if development["policy_fold_id"].astype(str).eq("holdout").any():
        raise ValueError("development90 contains the holdout policy fold")
    development_folds = (
        development["policy_fold_id"].astype(str).value_counts().to_dict()
    )
    if development_folds != {"0": 30, "1": 30, "2": 30}:
        raise ValueError(f"development90 policy folds are invalid: {development_folds}")
    _require_balanced_actions(randomized, development)
    randomized_development = randomized[
        randomized["experiment_split"].astype(str).eq("development")
    ]
    for identity in ("paper_id", "context_id"):
        if set(development[identity].astype(str)) != set(
            randomized_development[identity].astype(str)
        ):
            raise ValueError("development90 is not the frozen randomized subset")


def _require_balanced_actions(randomized: Any, development: Any) -> None:
    random_counts = (
        randomized.groupby("experiment_split")["logged_action"].value_counts().to_dict()
    )
    expected = {
        **{("development", action): 15 for action in ALL_ACTIONS},
        **{("confirmatory_holdout", action): 10 for action in ALL_ACTIONS},
    }
    if random_counts != expected:
        raise ValueError(f"randomized A0-A5 counts are invalid: {random_counts}")
    if development["logged_action"].astype(str).value_counts().to_dict() != {
        action: 15 for action in ALL_ACTIONS
    }:
        raise ValueError("development90 A0-A5 counts are invalid")


def _verify_policy_pair(
    graph_path: Path,
    no_graph_path: Path,
    *,
    development_sha256: str,
    model: GraphActionQModel,
) -> None:
    graph = _read_frame(graph_path)
    no_graph = _read_frame(no_graph_path)
    for label, frame, family in (
        ("graph", graph, "graph_features"),
        ("no_graph", no_graph, "no_graph_features"),
    ):
        if len(frame) != 60:
            raise ValueError(f"{label} policy holdout must contain 60 rows")
        required = {
            "paper_id",
            "context_id",
            "experiment_split",
            "logged_action",
            "target_action",
            "outcome",
            "propensity",
            "matched_budget",
            "policy_fold_id",
            "q_logged",
            "q_target",
            "q_baseline",
            "wrong_correction",
            "unsupported_claim",
            "realized_cost",
            "policy_feature_set",
            "policy_development_input_sha256",
            "policy_holdout_input_sha256",
        }
        if required - set(frame):
            raise ValueError(f"{label} policy holdout columns are incomplete")
        if not frame["experiment_split"].astype(str).eq("confirmatory_holdout").all():
            raise ValueError(f"{label} policy contains non-holdout rows")
        if set(frame["policy_feature_set"].astype(str)) != {family}:
            raise ValueError(f"{label} policy feature family mismatch")
        if frame["logged_action"].astype(str).value_counts().to_dict() != {
            action: 10 for action in ALL_ACTIONS
        }:
            raise ValueError(f"{label} policy logged actions are unbalanced")
        if not frame["propensity"].astype(float).eq(1.0 / 6.0).all():
            raise ValueError(f"{label} policy propensities are invalid")
        if not frame["matched_budget"].astype(float).eq(20.0).all():
            raise ValueError(f"{label} policy budgets are invalid")
        if not frame["policy_fold_id"].astype(str).eq("holdout").all():
            raise ValueError(f"{label} policy fold is not sealed holdout")
        numeric = frame[
            [
                "outcome",
                "propensity",
                "q_logged",
                "q_target",
                "q_baseline",
                "realized_cost",
            ]
        ].apply(lambda column: column.astype(float))
        if not all(math.isfinite(value) for value in numeric.to_numpy().ravel()):
            raise ValueError(f"{label} policy contains non-finite values")
        if set(frame["policy_development_input_sha256"].astype(str)) != {
            development_sha256
        }:
            raise ValueError(f"{label} policy is not bound to development90")
    for column in (
        "paper_id",
        "context_id",
        "policy_development_input_sha256",
        "policy_holdout_input_sha256",
    ):
        if set(graph[column].astype(str)) != set(no_graph[column].astype(str)):
            raise ValueError(f"paired policy mismatch: {column}")
    _verify_graph_policy_predictions(graph, model)


def _verify_graph_policy_predictions(graph: Any, model: GraphActionQModel) -> None:
    required = {*ACTION_POLICY_FEATURES, *(f"q_{action}" for action in ALL_ACTIONS)}
    missing = sorted(required - set(graph))
    if missing:
        raise ValueError(f"graph policy runtime columns are missing: {missing}")
    for _, row in graph.iterrows():
        features = [float(row[name]) for name in ACTION_POLICY_FEATURES]
        predicted = model.predict(features)
        if any(
            abs(predicted[action] - float(row[f"q_{action}"])) > 1e-12
            for action in ALL_ACTIONS
        ):
            raise ValueError("graph policy Q values do not replay from candidate model")
        logged_action = str(row["logged_action"])
        target_action = str(row["target_action"])
        if (
            abs(float(row["q_baseline"]) - predicted["baseline"]) > 1e-12
            or abs(float(row["q_logged"]) - predicted[logged_action]) > 1e-12
            or abs(float(row["q_target"]) - predicted[target_action]) > 1e-12
        ):
            raise ValueError("graph policy selected Q columns do not replay")
        decision = model.decision(features)
        expected_action = decision.action if decision.selected else "baseline"
        if str(row["target_action"]) != expected_action:
            raise ValueError(
                "graph policy target action does not replay candidate rules"
            )


def _verify_gate2_report(
    path: Path,
    *,
    model_sha256: str,
    development_sha256: str,
    randomized_sha256: str,
    graph_policy_sha256: str,
    no_graph_policy_sha256: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("contract") != "gear_gate2_dual_holdout_and_paired_policy_v2"
        or payload.get("status") != "passed"
        or payload.get("claim_allowed") is not True
    ):
        raise ValueError("action policy Gate 2 did not pass")
    checks = payload.get("checks") or {}
    guardrails = payload.get("guardrails") or {}
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("action policy Gate 2 checks are incomplete")
    for key in ("wrong_correction_pass", "unsupported_claim_pass", "cost_pass"):
        if guardrails.get(key) is not True:
            raise ValueError(f"action policy Gate 2 guardrail failed: {key}")
    paired = payload.get("graph_vs_no_graph_policy") or {}
    if "lcb_95" not in paired or "paired_switch_dr_sensitivity" not in paired:
        raise ValueError("Gate 2 lacks paired Graph-vs-noGraph evidence")
    paired_lcb = float(paired["lcb_95"])
    if not math.isfinite(paired_lcb) or not (
        paired_lcb > 0.0 or paired.get("both_abstain") is True
    ):
        raise ValueError("Gate 2 paired Graph-vs-noGraph criterion failed")
    binding = payload.get("action_policy_runtime_candidate") or {}
    expected = {
        "model_sha256": model_sha256,
        "q_model_family": "linear_t0_v1",
        "feature_schema_version": ACTION_POLICY_FEATURE_SCHEMA,
        "feature_family": "graph_features",
        "development_data_sha256": development_sha256,
        "randomized_data_sha256": randomized_sha256,
        "graph_policy_sha256": graph_policy_sha256,
        "no_graph_policy_sha256": no_graph_policy_sha256,
        "future_features_used": False,
        "future_outcomes_used_at_inference": False,
        "sealed_holdout_used_for_fitting": False,
        "training_rows": 90,
        "training_scope": "development_only",
        "gear_evidence_gap_status": (
            "phase_one_excluded_not_available_at_pre_retrieval_decision"
        ),
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Gate 2 is not hash-bound to the runtime policy candidate")


def _verify_frozen_runtime(
    frozen_path: Path,
    source_audit_path: Path,
    stage_a_audit_path: Path,
    stage_b_audit_path: Path,
    stage_c_audit_path: Path,
) -> None:
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("contract") != "gear_graph_rescue_frozen_replay_v1":
        raise ValueError("action policy lacks a formal frozen replay manifest")
    code = _validated_sha(frozen.get("runtime_code_sha256"), "runtime code")
    source = _validated_sha(frozen.get("rescue_source_sha256"), "rescue source")
    stage_ab_config = _validated_sha(
        frozen.get("stage_ab_runtime_config_sha256"), "Stage A/B config"
    )
    stage_c_config = _validated_sha(
        frozen.get("stage_c_runtime_config_sha256"), "Stage C config"
    )
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if (
        source_audit.get("contract") != "gear_rescue_source_fingerprint_audit_v1"
        or source_audit.get("passed") is not True
        or source_audit.get("source_sha256", source_audit.get("rescue_source_sha256"))
        != source
    ):
        raise ValueError("rescue source fingerprint audit is not frozen")
    frozen_count = int(
        frozen.get("rescue_source_file_count", frozen.get("source_file_count", 0))
    )
    if (
        frozen_count <= 0
        or int(source_audit.get("source_file_count", 0)) != frozen_count
    ):
        raise ValueError("rescue source fingerprint file count mismatch")
    audits = (
        (stage_a_audit_path, stage_ab_config, None, "Stage A"),
        (stage_b_audit_path, stage_ab_config, None, "Stage B"),
        (stage_c_audit_path, stage_c_config, 150, "Stage C"),
    )
    for path, expected_config, expected_cases, label in audits:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (
            audit.get("contract") != "gear_runtime_cohort_fingerprint_audit_v1"
            or audit.get("passed") is not True
            or audit.get("runtime_code_sha256") != code
            or audit.get("runtime_config_sha256") != expected_config
            or int(audit.get("runtime_source_file_count", 0)) <= 0
            or int(audit.get("cases", 0)) <= 0
        ):
            raise ValueError(f"{label} runtime cohort is not bound to frozen replay")
        if expected_cases is not None and int(audit["cases"]) != expected_cases:
            raise ValueError(f"{label} runtime cohort is incomplete")


def _validated_sha(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 71 or not text.startswith("sha256:"):
        raise ValueError(f"{label} fingerprint is invalid")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{label} fingerprint is invalid") from exc
    return text


def _release_paths(
    manifest_path: Path, release: GraphActionPolicyRelease
) -> dict[str, Path]:
    return {
        "model": _bound_path(manifest_path, release.model_path),
        "replay": _bound_path(manifest_path, release.replay_path),
        "development": _bound_path(manifest_path, release.development_data_path),
        "randomized": _bound_path(manifest_path, release.randomized_data_path),
        "graph_policy": _bound_path(manifest_path, release.graph_policy_path),
        "no_graph_policy": _bound_path(manifest_path, release.no_graph_policy_path),
        "gate2": _bound_path(manifest_path, release.gate2_report_path),
        "frozen_replay": _bound_path(
            manifest_path, release.frozen_replay_manifest_path
        ),
        "source_audit": _bound_path(
            manifest_path, release.source_fingerprint_audit_path
        ),
        "stage_a_runtime_audit": _bound_path(
            manifest_path, release.stage_a_runtime_audit_path
        ),
        "stage_b_runtime_audit": _bound_path(
            manifest_path, release.stage_b_runtime_audit_path
        ),
        "stage_c_runtime_audit": _bound_path(
            manifest_path, release.stage_c_runtime_audit_path
        ),
    }


def _read_frame(path: Path) -> Any:
    import pandas as pd

    return (
        pd.read_csv(path) if path.suffix.casefold() == ".csv" else pd.read_parquet(path)
    )


def _bound_path(manifest_path: Path, reference: str) -> Path:
    candidate = (manifest_path.parent / reference).resolve()
    if not candidate.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("action policy release reference escapes release directory")
    return candidate


def _relative_reference(output_path: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(output_path.parent.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "promotion inputs must be inside the release directory"
        ) from exc


def _required_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"action policy feature unavailable: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"action policy feature is non-finite: {name}")
    return result


def _prediction_uncertainty(state: object, bundle: Any) -> float:
    graph_result = getattr(state, "graph_result", None)
    forecast = getattr(graph_result, "forecast", None)
    width = getattr(forecast, "prediction_interval_width", None)
    if width is not None:
        return _required_float(width, "graph_prediction_uncertainty")
    reliability = _required_float(bundle.reliability, "graph_reliability")
    return 1.0 - reliability


def _replay_row_count(path: Path, release: GraphActionPolicyRelease) -> int:
    replay = json.loads(
        _bound_path(path, release.replay_path).read_text(encoding="utf-8")
    )
    return len(replay.get("rows") or [])


def _sha256(path: Path | None) -> str:
    if path is None:
        raise ValueError("action policy manifest is unavailable")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"action policy artifact hash mismatch: {path}")


__all__ = [
    "ACTION_POLICY_FEATURES",
    "ACTION_POLICY_FEATURE_SCHEMA",
    "ALL_ACTIONS",
    "GRAPH_ACTIONS",
    "ActionPolicyRule",
    "FrozenGraphActionSelector",
    "GraphActionPolicy",
    "GraphActionPolicyRelease",
    "GraphActionQModel",
    "RandomizedGraphActionSelector",
    "action_policy_t0_features",
    "load_graph_action_policy_release",
    "promote_graph_action_policy_release",
    "validate_graph_action_policy_replay",
]
