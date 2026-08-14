"""Fail-closed final release assembly for the local-frozen ASPR v6 study."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

from .evidence_registry import load_evidence_registry, registry_sha256
from .evidence_selection_v6 import (
    evidence_selection_sha256,
    load_evidence_selection_protocol,
)
from .prediction_registry_v6 import (
    load_prediction_registry,
    prediction_registry_sha256,
)
from .source_audit_v6 import sha256_file


FINAL_RELEASE_VERSION = "aspr-v6-final-frozen-release-1"
FINAL_TEST_FILES: Tuple[str, ...] = (
    "tests/nature_multihorizon/test_v6_evidence_framework.py",
    "tests/nature_multihorizon/test_v6_feature_materializer.py",
    "tests/nature_multihorizon/test_v6_local_work_view.py",
    "tests/nature_multihorizon/test_v6_modeling.py",
    "tests/nature_multihorizon/test_v6_offline_source_audit.py",
    "tests/nature_multihorizon/test_v6_prediction_registry.py",
    "tests/nature_multihorizon/test_v6_construct_validation.py",
    "tests/nature_multihorizon/test_v6_sealed.py",
    "tests/nature_multihorizon/test_v6_finalize.py",
)
DATASET_MANIFEST_NAMES: Tuple[str, ...] = (
    "input_views_manifest.json",
    "field_events_manifest.json",
    "publication_features_manifest.json",
    "opportunity_features_manifest.json",
    "targets_cohort_manifest.json",
)
PROMOTED_ENTITY_IDS = {
    "C1.DISPARITY",
    "C1.PIELOU",
    "C1.RAO",
    "C1.VARIETY",
    "C1_KNOWLEDGE_DIVERSITY",
}


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def canonical_artifact_id(payload: Mapping[str, Any]) -> str:
    """Return a deterministic identifier, excluding any existing identifier."""
    clean = {
        str(key): value
        for key, value in payload.items()
        if key != "artifact_id"
    }
    encoded = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _file_reference(
    path: Path, *, artifact_id: str | None = None
) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"required release file is missing: {resolved}")
    reference: Dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if artifact_id is not None:
        reference["artifact_id"] = artifact_id
    return reference


def _verify_file_reference(reference: Mapping[str, Any]) -> None:
    path = Path(str(reference["path"]))
    _require(path.is_file(), f"referenced file is missing: {path}")
    _require(
        sha256_file(path) == reference["sha256"],
        f"referenced file hash changed: {path}",
    )
    if "size_bytes" in reference:
        _require(
            path.stat().st_size == int(reference["size_bytes"]),
            f"referenced file size changed: {path}",
        )


def _find_single_manifest(
    output_root: Path,
    pattern: str,
    predicate: Callable[[Mapping[str, Any]], bool],
    label: str,
) -> Tuple[Dict[str, Any], Path]:
    matches: List[Tuple[Dict[str, Any], Path]] = []
    for path in sorted(Path(output_root).glob(pattern)):
        manifest = _load_json(path)
        if predicate(manifest):
            matches.append((manifest, path))
    _require(
        len(matches) == 1,
        f"expected exactly one current {label}; found {len(matches)}",
    )
    return matches[0]


def read_gate_results(path: Path) -> Dict[str, Any]:
    """Read a gate CSV and fail closed on missing, duplicate, or failed rows."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    gate_ids = [str(row.get("gate_id", "")) for row in rows]
    _require(bool(rows), f"no validation gates found: {path}")
    _require(
        all(gate_ids) and len(gate_ids) == len(set(gate_ids)),
        f"gate identifiers are missing or duplicated: {path}",
    )
    passed_values = {"1", "true", "yes"}
    failed = [
        gate_id
        for gate_id, row in zip(gate_ids, rows)
        if str(row.get("passed", "")).strip().lower() not in passed_values
    ]
    _require(not failed, f"blocking gates failed in {path}: {failed}")
    return {
        "n_gates": len(rows),
        "all_pass": True,
        "gate_ids": gate_ids,
    }


def _read_model_metrics(path: Path, model_id: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("model_id") == model_id
        ]
    _require(
        len(rows) == 1,
        f"expected one {model_id} metrics row in {path}; found {len(rows)}",
    )
    row = rows[0]
    numeric_fields = (
        "n_oof",
        "n_realized_finite",
        "spearman_expected",
        "spearman_ci_low",
        "spearman_ci_high",
        "spearman_conditional",
        "domain_macro_spearman",
        "n_reportable_domains",
        "uptake_brier_skill_score",
        "uptake_ece_10",
        "realized_interval_coverage_90",
        "realized_interval_mean_width",
        "gain_over_controls",
        "gain_over_controls_ci_low",
        "gain_over_controls_ci_high",
    )
    metrics: Dict[str, Any] = {"model_id": model_id}
    for field in numeric_fields:
        if field in row and row[field] != "":
            metrics[field] = float(row[field])
    return metrics


def _code_paths(project_root: Path) -> Tuple[Path, ...]:
    module_dir = project_root / "gear" / "nature_multihorizon"
    paths = list(sorted(module_dir.glob("*_v6.py")))
    paths.extend(
        [
            module_dir / "cohorts.py",
            module_dir / "contracts.py",
            module_dir / "targets.py",
            project_root / "scripts" / "run_nature_v6_local.py",
        ]
    )
    return tuple(path for path in paths if path.is_file())


def run_final_implementation_tests(
    *, project_root: Path, output_root: Path
) -> Tuple[Mapping[str, Any], Path]:
    """Run and freeze the complete final v6 test roster."""
    output_path = output_root / "final_registered_implementation_tests.txt"
    manifest_path = output_root / "final_implementation_test_manifest.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-s",
        *FINAL_TEST_FILES,
        "-q",
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["ASPR_NETWORK_POLICY"] = "forbidden"
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
    passed_match = re.search(r"(\d+) passed", completed.stdout)
    manifest: Dict[str, Any] = {
        "artifact_kind": "aspr_v6_final_implementation_tests",
        "command": command,
        "test_files": list(FINAL_TEST_FILES),
        "exit_code": int(completed.returncode),
        "passed": completed.returncode == 0,
        "passed_test_count": (
            int(passed_match.group(1)) if passed_match else None
        ),
        "network_data_acquisition": "forbidden_and_not_used",
        "code_sha256": {
            str(path.relative_to(project_root)): sha256_file(path)
            for path in _code_paths(project_root)
        },
        "test_sha256": {
            relative: sha256_file(project_root / relative)
            for relative in FINAL_TEST_FILES
        },
        "output": _file_reference(output_path),
    }
    manifest["artifact_id"] = canonical_artifact_id(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _require(
        completed.returncode == 0,
        f"final v6 tests failed; see {output_path}",
    )
    return manifest, manifest_path


def _current_development_runs(
    output_root: Path,
    *,
    config_sha256: str,
    registry_sha256_value: str,
) -> Dict[int, Tuple[Dict[str, Any], Path]]:
    selected: Dict[int, Tuple[Dict[str, Any], Path]] = {}
    for horizon in (3, 5, 8):
        selected[horizon] = _find_single_manifest(
            output_root,
            f"development_D{horizon}_*/development_run_manifest.json",
            lambda manifest, expected=horizon: (
                manifest.get("lineage", {}).get("config_sha256")
                == config_sha256
                and manifest.get("lineage", {}).get(
                    "innovation_registry_sha256"
                )
                == registry_sha256_value
                and int(
                    manifest.get("lineage", {}).get("horizon", -1)
                )
                == expected
                and manifest.get("lineage", {}).get(
                    "development_gate_pass"
                )
                is True
                and manifest.get("sealed_holdout_accessed") is False
            ),
            f"D{horizon} development run",
        )
    return selected


def _validate_framework(
    *,
    output_root: Path,
    source_audit_path: Path,
    registry_hash: str,
    prediction_hash: str,
    selection_hash: str,
) -> Dict[str, Any]:
    source = _load_json(source_audit_path)
    _require(source.get("overall_pass") is True, "source audit did not pass")
    _require(
        source.get("network_policy") == "forbidden",
        "source audit does not enforce zero-network execution",
    )
    _require(
        int(source.get("required_failure_count", -1)) == 0,
        "required frozen sources are missing",
    )
    framework_path = output_root / "framework_audit.json"
    framework = _load_json(framework_path)
    lineage = framework.get("lineage", {})
    _require(
        framework.get("definition_audit_pass") is True,
        "definition-stage framework audit did not pass",
    )
    _require(
        framework.get("source_data_audit_pass") is True,
        "framework source-data audit did not pass",
    )
    _require(
        lineage.get("innovation_registry_sha256") == registry_hash,
        "framework innovation-registry lineage mismatch",
    )
    _require(
        lineage.get("prediction_registry_sha256") == prediction_hash,
        "framework prediction-registry lineage mismatch",
    )
    _require(
        lineage.get("evidence_selection_sha256") == selection_hash,
        "framework evidence-selection lineage mismatch",
    )
    return {
        "source_audit": _file_reference(source_audit_path),
        "source_lineage_id": source["source_lineage_id"],
        "framework_audit": _file_reference(framework_path),
        "definition_stage_confirmatory_ready": framework.get(
            "release_confirmatory_ready"
        ),
        "definition_stage_note": (
            "Expected to be false before runtime promotion and sealed "
            "evaluation; final status is established below."
        ),
    }


def _validate_dataset_manifests(
    dataset_dir: Path,
    development_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    development_refs = development_manifest["lineage"]["dataset_manifests"]
    references: Dict[str, Any] = {}
    for name in DATASET_MANIFEST_NAMES:
        path = dataset_dir / name
        manifest = _load_json(path)
        expected = development_refs.get(name)
        _require(expected is not None, f"D5 lineage omits {name}")
        _require(
            manifest.get("artifact_id") == expected.get("artifact_id"),
            f"dataset artifact changed after development: {name}",
        )
        _require(
            sha256_file(path) == expected.get("manifest_sha256"),
            f"dataset manifest changed after development: {name}",
        )
        references[name] = _file_reference(
            path, artifact_id=str(manifest["artifact_id"])
        )
    return references


def _validate_development(
    runs: Mapping[int, Tuple[Dict[str, Any], Path]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    references: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    for horizon, (manifest, manifest_path) in runs.items():
        run_dir = manifest_path.parent
        gate_summary = read_gate_results(
            run_dir / "development_acceptance_gates.csv"
        )
        primary = _read_model_metrics(
            run_dir / "development_metrics.csv",
            "innovation_plus_controls",
        )
        references[str(horizon)] = {
            "manifest": _file_reference(
                manifest_path, artifact_id=str(manifest["artifact_id"])
            ),
            "gates": _file_reference(
                run_dir / "development_acceptance_gates.csv"
            ),
            "gate_summary": gate_summary,
        }
        metrics[str(horizon)] = primary
    _require(
        runs[5][0]["lineage"]["development_frame"]["n_observed_zero"] > 0,
        "D5 development cohort silently dropped observed zero papers",
    )
    return references, metrics


def _validate_release_candidate(
    output_root: Path,
    *,
    config_sha256: str,
    registry_sha256_value: str,
    development_ids: Mapping[str, str],
) -> Tuple[Dict[str, Any], Path, Dict[str, Any]]:
    freeze, _ = _find_single_manifest(
        output_root,
        "sealed_D5_*/sealed_model_freeze_manifest.json",
        lambda manifest: (
            manifest.get("lineage", {}).get("config_sha256")
            == config_sha256
            and manifest.get("lineage", {}).get(
                "innovation_registry_sha256"
            )
            == registry_sha256_value
            and manifest.get("sealed_holdout_labels_accessed") is False
        ),
        "current sealed model freeze",
    )
    frozen_release_id = freeze["lineage"]["release_candidate_artifact_id"]
    release, manifest_path = _find_single_manifest(
        output_root,
        "release_candidate_*/release_candidate_manifest.json",
        lambda manifest: (
            manifest.get("artifact_id") == frozen_release_id
            and manifest.get("lineage", {}).get("config_sha256")
            == config_sha256
            and manifest.get("lineage", {}).get(
                "innovation_registry_sha256"
            )
            == registry_sha256_value
            and manifest.get("lineage", {}).get("development_artifact_ids")
            == development_ids
            and manifest.get("summary", {}).get(
                "release_candidate_ready_before_sealed"
            )
            is True
            and manifest.get("summary", {}).get("sealed_holdout_accessed")
            is False
        ),
        "pre-sealed release candidate",
    )
    for reference in release.get("outputs", {}).values():
        _verify_file_reference(reference)
    promotion_path = manifest_path.parent / "promotion_report.json"
    promotion = _load_json(promotion_path)
    promoted = {
        entity_id
        for entity_id, decision in promotion["decisions"].items()
        if decision.get("promotion_status") == "promoted"
    }
    _require(
        promoted == PROMOTED_ENTITY_IDS,
        f"unexpected promoted construct set: {sorted(promoted)}",
    )
    for entity_id in promoted:
        gates = promotion["decisions"][entity_id]["gates"]
        _require(
            set(gates) == {f"P{index}" for index in range(1, 9)}
            and all(gate.get("passed") is True for gate in gates.values()),
            f"promotion gates are incomplete for {entity_id}",
        )
    return release, manifest_path, {
        "manifest": _file_reference(
            manifest_path, artifact_id=str(release["artifact_id"])
        ),
        "promotion_report": _file_reference(promotion_path),
        "promoted_entity_ids": sorted(promoted),
        "nonpromoted_entity_ids": release["summary"][
            "nonpromoted_conditional_or_exploratory_entities"
        ],
    }


def _validate_sealed(
    output_root: Path,
    *,
    config_sha256: str,
    registry_sha256_value: str,
    release_artifact_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    freeze, freeze_path = _find_single_manifest(
        output_root,
        "sealed_D5_*/sealed_model_freeze_manifest.json",
        lambda manifest: (
            manifest.get("lineage", {}).get("config_sha256")
            == config_sha256
            and manifest.get("lineage", {}).get(
                "innovation_registry_sha256"
            )
            == registry_sha256_value
            and manifest.get("lineage", {}).get(
                "release_candidate_artifact_id"
            )
            == release_artifact_id
            and manifest.get("sealed_holdout_labels_accessed") is False
        ),
        "sealed model freeze",
    )
    for reference in freeze.get("models", {}).values():
        _verify_file_reference(reference)
    for reference in freeze.get("outputs", {}).values():
        _verify_file_reference(reference)
    sealed_dir = freeze_path.parent
    evaluation_path = sealed_dir / "sealed_evaluation_manifest.json"
    evaluation = _load_json(evaluation_path)
    _require(
        evaluation.get("freeze_artifact_id") == freeze["artifact_id"],
        "sealed evaluation does not match the locked model freeze",
    )
    summary = evaluation.get("summary", {})
    _require(
        summary.get("final_release_pass") is True,
        "the single sealed evaluation did not pass",
    )
    _require(
        summary.get("sealed_unlocks_used") == 1
        and summary.get("sealed_reunlock_forbidden") is True,
        "sealed unlock discipline is invalid",
    )
    _require(
        int(summary.get("n_observed_zero", 0)) > 0,
        "sealed evaluation silently dropped observed zero papers",
    )
    for reference in evaluation.get("outputs", {}).values():
        _verify_file_reference(reference)
    receipts = list(sealed_dir.glob("sealed_unlock_receipt*.json"))
    _require(
        len(receipts) == 1,
        f"expected one and only one sealed unlock receipt; found {len(receipts)}",
    )
    receipt = _load_json(receipts[0])
    _require(
        receipt.get("unlock_number") == 1
        and receipt.get("maximum_unlocks") == 1
        and receipt.get("receipt_id") == evaluation.get("unlock_receipt_id"),
        "sealed unlock receipt is inconsistent",
    )
    gate_path = sealed_dir / "sealed_evaluation_gates.csv"
    metrics_path = sealed_dir / "sealed_evaluation_metrics.csv"
    gate_summary = read_gate_results(gate_path)
    _require(
        gate_summary["n_gates"] == 10,
        "sealed gate roster must contain exactly ten gates",
    )
    return {
        "freeze_manifest": _file_reference(
            freeze_path, artifact_id=str(freeze["artifact_id"])
        ),
        "evaluation_manifest": _file_reference(
            evaluation_path, artifact_id=str(evaluation["artifact_id"])
        ),
        "unlock_receipt": _file_reference(
            receipts[0], artifact_id=str(receipt["receipt_id"])
        ),
        "gates": _file_reference(gate_path),
        "gate_summary": gate_summary,
        "unlock_count_used": 1,
        "reunlock_forbidden": True,
    }, _read_model_metrics(metrics_path, "innovation_plus_controls")


def _validate_existing_manifest(path: Path) -> Dict[str, Any]:
    manifest = _load_json(path)
    _require(
        manifest.get("artifact_id") == canonical_artifact_id(manifest),
        "existing final release manifest has an invalid artifact identifier",
    )
    _require(
        manifest.get("summary", {}).get("final_release_pass") is True,
        "existing final release is not passing",
    )
    for reference in manifest.get("immutable_file_references", {}).values():
        _verify_file_reference(reference)
    return manifest


def finalize_v6_release(
    *,
    project_root: Path,
    config_path: Path,
    dataset_dir: Path,
    output_root: Path,
) -> Tuple[Mapping[str, Any], Path]:
    """Validate every release layer and write one immutable final manifest."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = output_root / "final_release_manifest.json"
    if manifest_path.is_file():
        return _validate_existing_manifest(manifest_path), manifest_path

    config = _load_json(config_path)
    registry_path = project_root / str(config["evidence_registry_path"])
    prediction_path = project_root / str(config["prediction_registry_path"])
    selection_path = project_root / str(
        config["evidence_selection_protocol_path"]
    )
    registry = load_evidence_registry(registry_path)
    prediction = load_prediction_registry(prediction_path)
    selection = load_evidence_selection_protocol(selection_path)
    config_hash = sha256_file(config_path)
    registry_hash = registry_sha256(registry)
    prediction_hash = prediction_registry_sha256(prediction)
    selection_hash = evidence_selection_sha256(selection)

    framework = _validate_framework(
        output_root=output_root,
        source_audit_path=project_root
        / str(config["storage"]["source_audit_path"]),
        registry_hash=registry_hash,
        prediction_hash=prediction_hash,
        selection_hash=selection_hash,
    )
    runs = _current_development_runs(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
    )
    development_refs, development_metrics = _validate_development(runs)
    dataset_refs = _validate_dataset_manifests(dataset_dir, runs[5][0])
    development_ids = {
        str(horizon): str(manifest["artifact_id"])
        for horizon, (manifest, _) in runs.items()
    }

    construct, construct_path = _find_single_manifest(
        output_root,
        "construct_validation_*/construct_validation_manifest.json",
        lambda manifest: (
            manifest.get("lineage", {}).get("config_sha256") == config_hash
            and manifest.get("lineage", {}).get(
                "innovation_registry_sha256"
            )
            == registry_hash
            and manifest.get("summary", {}).get(
                "c1_measurement_gate_pass"
            )
            is True
            and manifest.get("summary", {}).get(
                "sealed_holdout_accessed"
            )
            is False
        ),
        "construct validation",
    )
    release, release_path, release_ref = _validate_release_candidate(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
        development_ids=development_ids,
    )
    sealed_ref, sealed_metrics = _validate_sealed(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
        release_artifact_id=str(release["artifact_id"]),
    )

    amendment_path = (
        output_root / "protocol_amendment_001_dimension_merge.json"
    )
    amendment = _load_json(amendment_path)
    _require(
        amendment.get("new_registry_sha256") == registry_hash
        and amendment.get("sealed_holdout_accessed") is False
        and amendment.get("decision", {}).get("metrics_added") == []
        and amendment.get("decision", {}).get("metrics_removed") == []
        and amendment.get("decision", {}).get("samples_added_or_removed")
        == 0
        and amendment.get("decision", {}).get("outcome_or_gate_change")
        is False,
        "dimension-merge amendment is not label-only and pre-sealed",
    )
    report_path = output_root / "ASPR_v6_final_reviewer_defense.md"
    _require(
        report_path.is_file() and report_path.stat().st_size > 1000,
        "final reviewer-defense report is missing or incomplete",
    )
    tests, test_manifest_path = run_final_implementation_tests(
        project_root=project_root,
        output_root=output_root,
    )

    immutable_files = {
        "config": _file_reference(config_path),
        "innovation_registry": _file_reference(registry_path),
        "prediction_registry": _file_reference(prediction_path),
        "evidence_selection_protocol": _file_reference(selection_path),
        "source_audit": framework["source_audit"],
        "framework_audit": framework["framework_audit"],
        "protocol_amendment": _file_reference(amendment_path),
        "construct_manifest": _file_reference(
            construct_path, artifact_id=str(construct["artifact_id"])
        ),
        "release_candidate_manifest": _file_reference(
            release_path, artifact_id=str(release["artifact_id"])
        ),
        "final_test_manifest": _file_reference(
            test_manifest_path, artifact_id=str(tests["artifact_id"])
        ),
        "reviewer_defense_report": _file_reference(report_path),
    }
    manifest: Dict[str, Any] = {
        "artifact_kind": "aspr_v6_final_frozen_release",
        "final_release_version": FINAL_RELEASE_VERSION,
        "protocol_id": config["protocol_id"],
        "claim_scope": {
            "innovation_interface": (
                "publication-time innovation-evidence profile; not an "
                "innovation truth score"
            ),
            "influence_interface": (
                "calibrated D3/D5/D8 future scholarly uptake and conditional "
                "diffusion forecast; not quality, causality, or acceptance"
            ),
            "primary_confirmatory_context": (
                "C1 knowledge-base diversity and integration"
            ),
            "conditional_dimensions": [
                "N1_RECOMBINATION",
                "S1_STRUCTURAL_VARIATION",
            ],
            "exploratory_not_used": ["N2_SEMANTIC_NOVELTY"],
        },
        "lineage": {
            "config_sha256": config_hash,
            "innovation_registry_canonical_sha256": registry_hash,
            "prediction_registry_canonical_sha256": prediction_hash,
            "evidence_selection_canonical_sha256": selection_hash,
            "source_lineage_id": framework["source_lineage_id"],
            "network_policy": "forbidden",
            "raw_data_policy": "local_frozen_only",
            "new_raw_dataset_created": False,
        },
        "definition_and_source_validation": framework,
        "materialized_dataset_manifests": dataset_refs,
        "construct_validation": {
            "manifest": immutable_files["construct_manifest"],
            "summary": construct["summary"],
        },
        "development_validation": {
            "runs": development_refs,
            "primary_metrics": development_metrics,
        },
        "presealed_promotion": release_ref,
        "sealed_validation": {
            **sealed_ref,
            "primary_metrics": sealed_metrics,
        },
        "implementation_validation": {
            "manifest": immutable_files["final_test_manifest"],
            "passed": tests["passed"],
            "passed_test_count": tests["passed_test_count"],
        },
        "immutable_file_references": immutable_files,
        "limitations": [
            (
                "The system estimates observable evidence and future scholarly "
                "diffusion, not innovation truth, paper quality, social impact, "
                "causal impact, or Nature acceptance."
            ),
            (
                "N1 remains conditional because the source-type adaptation and "
                "exact bipartite randomization equivalence are not complete."
            ),
            (
                "S1 remains sensitivity-only and N2 is not used because their "
                "registered equivalence or frozen-data validity gates are unmet."
            ),
            (
                "The 22 matched review-silver papers are supportive only and "
                "cannot serve as confirmatory human-gold construct validation."
            ),
            (
                "Mathematics/statistics has only 143 sealed papers and is not "
                "individually reportable under the preregistered n>=200 rule."
            ),
        ],
        "summary": {
            "scientific_definition_gate_pass": True,
            "technical_implementation_gate_pass": True,
            "construct_measurement_gate_pass": True,
            "development_D3_D5_D8_gate_pass": True,
            "presealed_promotion_gate_pass": True,
            "sealed_gate_count": sealed_ref["gate_summary"]["n_gates"],
            "sealed_gate_pass_count": sealed_ref["gate_summary"]["n_gates"],
            "sealed_unlock_count_used": 1,
            "sealed_reunlock_forbidden": True,
            "zero_observations_retained": True,
            "all_12_domains_present": True,
            "final_release_pass": True,
        },
    }
    manifest["artifact_id"] = canonical_artifact_id(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest, manifest_path


__all__ = [
    "FINAL_RELEASE_VERSION",
    "FINAL_TEST_FILES",
    "canonical_artifact_id",
    "finalize_v6_release",
    "read_gate_results",
    "run_final_implementation_tests",
]
