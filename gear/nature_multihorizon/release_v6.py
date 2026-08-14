"""Pre-holdout release validation and fail-closed v6 promotion evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .contracts_v6 import ImplementationFidelity, RegistryStatus
from .evidence_registry import load_evidence_registry, registry_sha256
from .modeling_v6 import safe_spearman
from .promotion_v6 import (
    PromotionGateEvidence,
    PromotionStatus,
    build_promotion_report,
)
from .source_audit_v6 import sha256_file


RELEASE_VALIDATION_VERSION = "aspr-v6-release-validation-1"
REGISTERED_TEST_FILES: Tuple[str, ...] = (
    "tests/nature_multihorizon/test_v6_evidence_framework.py",
    "tests/nature_multihorizon/test_v6_feature_materializer.py",
    "tests/nature_multihorizon/test_v6_local_work_view.py",
    "tests/nature_multihorizon/test_v6_modeling.py",
    "tests/nature_multihorizon/test_v6_offline_source_audit.py",
    "tests/nature_multihorizon/test_v6_prediction_registry.py",
    "tests/nature_multihorizon/test_v6_construct_validation.py",
    "tests/nature_multihorizon/test_v6_sealed.py",
)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _current_development_runs(
    output_root: Path,
    *,
    config_sha256: str,
    registry_sha256_value: str,
) -> Mapping[int, Tuple[Mapping[str, Any], Path]]:
    matches: Dict[int, List[Tuple[Mapping[str, Any], Path]]] = {
        3: [],
        5: [],
        8: [],
    }
    for path in Path(output_root).glob(
        "development_D*/development_run_manifest.json"
    ):
        manifest = _load_json(path)
        lineage = manifest.get("lineage", {})
        horizon = int(lineage.get("horizon", -1))
        if (
            horizon in matches
            and lineage.get("config_sha256") == config_sha256
            and lineage.get("innovation_registry_sha256")
            == registry_sha256_value
            and not manifest.get("sealed_holdout_accessed", True)
            and lineage.get("development_gate_pass") is True
        ):
            matches[horizon].append((manifest, path.parent))
    selected = {}
    for horizon, candidates in matches.items():
        if len(candidates) != 1:
            raise ValueError(
                f"expected one current passing D{horizon} run, "
                f"found {len(candidates)}"
            )
        selected[horizon] = candidates[0]
    return selected


def _current_construct_run(
    output_root: Path,
    *,
    config_sha256: str,
    registry_sha256_value: str,
) -> Tuple[Mapping[str, Any], Path]:
    matches = []
    for path in Path(output_root).glob(
        "construct_validation_*/construct_validation_manifest.json"
    ):
        manifest = _load_json(path)
        lineage = manifest.get("lineage", {})
        if (
            lineage.get("config_sha256") == config_sha256
            and lineage.get("innovation_registry_sha256")
            == registry_sha256_value
            and manifest.get("summary", {}).get(
                "c1_measurement_gate_pass"
            )
            is True
            and not manifest.get("summary", {}).get(
                "sealed_holdout_accessed", True
            )
        ):
            matches.append((manifest, path.parent))
    if len(matches) != 1:
        raise ValueError(
            f"expected one current construct audit, found {len(matches)}"
        )
    return matches[0]


def run_registered_implementation_tests(
    *,
    project_root: Path,
    output_dir: Path,
    config_sha256: str,
    registry_sha256_value: str,
) -> Mapping[str, Any]:
    """Run the exact v6 test roster and hash both command and output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "registered_implementation_tests.txt"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        *REGISTERED_TEST_FILES,
        "-q",
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output_path.write_text(completed.stdout, encoding="utf-8")
    code_paths = (
        project_root / "gear/nature_multihorizon/contracts_v6.py",
        project_root / "gear/nature_multihorizon/features_v6.py",
        project_root / "gear/nature_multihorizon/modeling_v6.py",
        project_root / "gear/nature_multihorizon/targets_v6.py",
        project_root / "gear/nature_multihorizon/promotion_v6.py",
    )
    manifest = {
        "artifact_kind": "aspr_v6_registered_implementation_tests",
        "command": command,
        "test_files": list(REGISTERED_TEST_FILES),
        "exit_code": int(completed.returncode),
        "passed": completed.returncode == 0,
        "config_sha256": config_sha256,
        "innovation_registry_sha256": registry_sha256_value,
        "network_data_acquisition": "forbidden_and_not_used",
        "code_sha256": {
            path.name: sha256_file(path) for path in code_paths
        },
        "output": {
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path = output_dir / "implementation_test_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"registered implementation tests failed; see {output_path}"
        )
    return manifest


def predictive_stability_audit(
    development_dir: Path,
    *,
    min_domain_rows: int,
) -> Tuple[pd.DataFrame, Mapping[str, Any]]:
    """Check primary direction in every development time block and domain."""
    predictions = pd.read_parquet(
        Path(development_dir) / "development_oof_predictions.parquet"
    )
    baseline = predictions[
        predictions["model_id"].eq("controls_only")
    ][
        [
            "paper_id",
            "outer_fold_id",
            "publication_year",
            "domain12",
            "realized_diffusion_target",
            "expected_diffusion_score",
        ]
    ].rename(columns={"expected_diffusion_score": "baseline_prediction"})
    primary = predictions[
        predictions["model_id"].eq("innovation_plus_controls")
    ][["paper_id", "outer_fold_id", "expected_diffusion_score"]].rename(
        columns={"expected_diffusion_score": "primary_prediction"}
    )
    paired = baseline.merge(
        primary,
        on=["paper_id", "outer_fold_id"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    group_specs: Iterable[Tuple[str, Any, pd.DataFrame]] = [
        ("domain", value, group)
        for value, group in paired.groupby("domain12", sort=True)
    ]
    group_specs = [
        *group_specs,
        *[
            ("outer_time_fold", int(value), group)
            for value, group in paired.groupby("outer_fold_id", sort=True)
        ],
    ]
    for scope, value, group in group_specs:
        baseline_rho = safe_spearman(
            group["realized_diffusion_target"],
            group["baseline_prediction"],
        )
        primary_rho = safe_spearman(
            group["realized_diffusion_target"],
            group["primary_prediction"],
        )
        rows.append(
            {
                "scope": scope,
                "group": str(value),
                "n_rows": len(group),
                "publication_year_min": int(
                    group["publication_year"].min()
                ),
                "publication_year_max": int(
                    group["publication_year"].max()
                ),
                "baseline_spearman": baseline_rho,
                "primary_spearman": primary_rho,
                "gain_over_controls": primary_rho - baseline_rho,
                "reportable": int(
                    scope != "domain"
                    or len(group) >= int(min_domain_rows)
                ),
            }
        )
    output = pd.DataFrame(rows)
    reportable = output[output["reportable"].eq(1)]
    domain = reportable[reportable["scope"].eq("domain")]
    temporal = reportable[reportable["scope"].eq("outer_time_fold")]
    summary = {
        "n_reportable_domains": int(len(domain)),
        "all_reportable_domain_primary_directions_positive": bool(
            domain["primary_spearman"].gt(0).all()
        ),
        "all_temporal_fold_primary_directions_positive": bool(
            temporal["primary_spearman"].gt(0).all()
        ),
        "domains_with_nonpositive_incremental_gain": domain.loc[
            domain["gain_over_controls"].le(0), "group"
        ].tolist(),
        "all_temporal_fold_incremental_gains_positive": bool(
            temporal["gain_over_controls"].gt(0).all()
        ),
    }
    summary["p7_direction_stability_pass"] = bool(
        summary["n_reportable_domains"] == 12
        and summary[
            "all_reportable_domain_primary_directions_positive"
        ]
        and summary["all_temporal_fold_primary_directions_positive"]
        and summary["all_temporal_fold_incremental_gains_positive"]
    )
    return output, summary


def _gate_evidence(
    gate_id: str,
    *,
    passed: bool,
    artifact_ids: Tuple[str, ...],
    detail: str,
) -> PromotionGateEvidence:
    return PromotionGateEvidence(
        gate_id=gate_id,
        passed=bool(passed),
        evidence_artifact_ids=artifact_ids if passed else (),
        detail=detail,
    )


def prepare_release_candidate(
    *,
    project_root: Path,
    config_path: Path,
    dataset_dir: Path,
    output_root: Path,
) -> Tuple[Mapping[str, Any], Path]:
    """Validate development evidence and freeze promotion before holdout."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_root = Path(output_root).resolve()
    config = _load_json(config_path)
    registry_path = project_root / str(config["evidence_registry_path"])
    registry = load_evidence_registry(registry_path)
    config_hash = sha256_file(config_path)
    registry_hash = registry_sha256(registry)
    development = _current_development_runs(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
    )
    construct_manifest, construct_dir = _current_construct_run(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
    )
    pre_spec = {
        "release_validation_version": RELEASE_VALIDATION_VERSION,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": config_hash,
        "innovation_registry_sha256": registry_hash,
        "development_artifact_ids": {
            str(horizon): manifest["artifact_id"]
            for horizon, (manifest, _) in development.items()
        },
        "construct_artifact_id": construct_manifest["artifact_id"],
        "sealed_holdout_accessed": False,
    }
    run_hash = _canonical_hash(pre_spec)
    output_dir = output_root / (
        f"release_candidate_{run_hash.removeprefix('sha256:')[:12]}"
    )
    manifest_path = output_dir / "release_candidate_manifest.json"
    if manifest_path.is_file():
        return _load_json(manifest_path), output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    implementation = run_registered_implementation_tests(
        project_root=project_root,
        output_dir=output_dir,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
    )
    stability, stability_summary = predictive_stability_audit(
        development[5][1],
        min_domain_rows=int(
            config["quality_protocol"]["min_reportable_domain_rows"]
        ),
    )
    stability_path = output_dir / "predictive_stability.csv"
    stability.to_csv(stability_path, index=False)
    stability_artifact = {
        "artifact_kind": "aspr_v6_predictive_stability",
        "development_artifact_id": development[5][0]["artifact_id"],
        "summary": stability_summary,
        "output_sha256": sha256_file(stability_path),
    }
    stability_artifact["artifact_id"] = _canonical_hash(
        stability_artifact
    )
    framework_path = output_root / "framework_audit.json"
    source_path = output_root / "source_audit.json"
    framework = _load_json(framework_path)
    source = _load_json(source_path)
    publication_manifest = _load_json(
        dataset_dir / "publication_features_manifest.json"
    )
    amendment_path = output_root / (
        "protocol_amendment_001_dimension_merge.json"
    )
    amendment = _load_json(amendment_path)
    all_development_pass = all(
        manifest["lineage"].get("development_gate_pass") is True
        and not manifest.get("sealed_holdout_accessed", True)
        for manifest, _ in development.values()
    )
    exact_c1 = all(
        metric.fidelity is ImplementationFidelity.EXACT_SOURCE
        for metric in registry.metrics.values()
        if metric.dimension_id == "C1_KNOWLEDGE_DIVERSITY"
        and metric.status is RegistryStatus.CANDIDATE_CONFIRMATORY
    )
    p8_pass = bool(
        amendment.get("sealed_holdout_accessed") is False
        and amendment["decision"]["metrics_added"] == []
        and amendment["decision"]["metrics_removed"] == []
        and amendment["decision"]["samples_added_or_removed"] == 0
        and amendment["decision"]["outcome_or_gate_change"] is False
        and all_development_pass
    )
    evidence_artifacts = {
        "P1": (implementation["artifact_id"],),
        "P2": (publication_manifest["artifact_id"],),
        "P3": (
            sha256_file(source_path),
            sha256_file(framework_path),
        ),
        "P4": (construct_manifest["artifact_id"],),
        "P5": (
            implementation["artifact_id"],
            construct_manifest["artifact_id"],
        ),
        "P6": (construct_manifest["artifact_id"],),
        "P7": (
            stability_artifact["artifact_id"],
            development[3][0]["artifact_id"],
            development[5][0]["artifact_id"],
            development[8][0]["artifact_id"],
        ),
        "P8": (
            sha256_file(amendment_path),
            development[5][0]["artifact_id"],
        ),
    }
    pass_by_gate = {
        "P1": bool(implementation["passed"])
        and bool(
            framework["audits"]["innovation_implementation"][
                "overall_pass"
            ]
        )
        and bool(
            framework["audits"]["prediction_implementation"][
                "overall_pass"
            ]
        ),
        "P2": publication_manifest["counts"]["strict_prior_violations"] == 0,
        "P3": bool(source["overall_pass"])
        and bool(framework["source_data_audit_pass"]),
        "P4": bool(
            construct_manifest["summary"][
                "all_c1_reference_stability_pass"
            ]
        ),
        "P5": bool(exact_c1 and implementation["passed"]),
        "P6": bool(
            construct_manifest["summary"]["c1_measurement_gate_pass"]
        ),
        "P7": bool(
            stability_summary["p7_direction_stability_pass"]
            and all_development_pass
        ),
        "P8": p8_pass,
    }
    detail_by_gate = {
        "P1": "All registered implementations resolve and the frozen v6 test roster passes.",
        "P2": "Materialized C1 rows have zero source_max_year violations.",
        "P3": "Frozen source, taxonomy, reference, and cohort audits pass without outcome-based exclusions.",
        "P4": "Twenty outcome-blind 80% reference-subsampling repetitions pass the worst-repeat stability gates.",
        "P5": "C1 formulas are exact-source implementations with hand-calculated tests; project parameters and limitations remain explicit.",
        "P6": "The merged source-defined construct passes noncollinearity/discriminant and measurement audits; the 22-paper heuristic review set remains supportive only.",
        "P7": "Primary direction is positive in all 12 reportable domains and every expanding time fold; D3/D5/D8 development gates pass.",
        "P8": "The conservative label-only amendment retained every metric and paper, changed no gate, was versioned before the final rerun, and the sealed holdout remains locked.",
    }
    eligible_entities = {
        "C1_KNOWLEDGE_DIVERSITY",
        *{
            metric_id
            for metric_id, metric in registry.metrics.items()
            if metric.dimension_id == "C1_KNOWLEDGE_DIVERSITY"
            and metric.status is RegistryStatus.CANDIDATE_CONFIRMATORY
        },
    }
    evidence_by_entity = {
        entity_id: {
            gate_id: _gate_evidence(
                gate_id,
                passed=pass_by_gate[gate_id],
                artifact_ids=evidence_artifacts[gate_id],
                detail=detail_by_gate[gate_id],
            )
            for gate_id in sorted(pass_by_gate)
        }
        for entity_id in sorted(eligible_entities)
    }
    promotion = build_promotion_report(
        registry,
        evidence_by_entity,
        report_id=(
            "aspr-v6-promotion-"
            f"{run_hash.removeprefix('sha256:')[:12]}"
        ),
        evaluated_artifact_id=development[5][0]["artifact_id"],
        sealed_holdout_inspected=False,
    )
    promotion_path = output_dir / "promotion_report.json"
    promotion_path.write_text(
        json.dumps(
            promotion.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    promoted = set(promotion.promoted_entity_ids)
    expected_promoted = eligible_entities
    summary = {
        "implementation_tests_pass": bool(implementation["passed"]),
        "construct_gate_pass": bool(
            construct_manifest["summary"]["c1_measurement_gate_pass"]
        ),
        "predictive_stability": stability_summary,
        "development_gates_pass": all_development_pass,
        "promotion_gate_pass": promoted == expected_promoted,
        "promoted_entity_ids": sorted(promoted),
        "nonpromoted_conditional_or_exploratory_entities": sorted(
            entity_id
            for entity_id, decision in promotion.decisions.items()
            if decision.promotion_status is PromotionStatus.NOT_ELIGIBLE
        ),
        "release_candidate_ready_before_sealed": bool(
            implementation["passed"]
            and all_development_pass
            and stability_summary["p7_direction_stability_pass"]
            and promoted == expected_promoted
        ),
        "sealed_holdout_accessed": False,
    }
    summary_path = output_dir / "release_candidate_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    outputs = {
        "implementation_test_manifest": output_dir
        / "implementation_test_manifest.json",
        "implementation_test_output": output_dir
        / "registered_implementation_tests.txt",
        "predictive_stability": stability_path,
        "promotion_report": promotion_path,
        "summary": summary_path,
    }
    manifest = {
        "artifact_kind": "aspr_v6_release_candidate",
        "release_validation_version": RELEASE_VALIDATION_VERSION,
        "lineage": pre_spec,
        "promotion_gate_evidence": pass_by_gate,
        "summary": summary,
        "outputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if not summary["release_candidate_ready_before_sealed"]:
        raise RuntimeError("pre-holdout release candidate gates did not pass")
    return manifest, output_dir


__all__ = [
    "REGISTERED_TEST_FILES",
    "RELEASE_VALIDATION_VERSION",
    "predictive_stability_audit",
    "prepare_release_candidate",
    "run_registered_implementation_tests",
]
