"""Prepare the minimum source-backed evaluation data for Fig. 4new.

This module does not render a figure and does not define new runtime or
Pydantic contracts.  JSONL review tasks reuse the existing GEAR evaluation
formats; all other products are derived CSV/JSON audit files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments.gear.evaluation.reconstruct_claim_attribution_folds import (
    reconstruct_strict_claim_attribution_folds,
)
from experiments.gear.review_reconstruction.evaluation import (
    build_blind_match_package,
    wrong_paper_shuffle,
)
from gear.review_contracts import ReviewPoint, StructuredReview

SEED = 20260829
BOOTSTRAP_REPLICATES = 5000
CLAIM_C_TOP_K = 3


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _stable_digest(*parts: object) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _spearman(frame: pd.DataFrame, score: str) -> float:
    left = rankdata(frame[score].to_numpy(float), method="average")
    right = rankdata(
        frame["future_structural_outcome"].to_numpy(float), method="average"
    )
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _top_decile_mean(frame: pd.DataFrame, score: str) -> float:
    count = max(1, math.ceil(len(frame) * 0.1))
    return float(frame.nlargest(count, score)["future_structural_outcome"].mean())


def _bootstrap_metric(
    frame: pd.DataFrame,
    score: str,
    metric: str,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        indexes = generator.integers(0, len(frame), len(frame))
        sample = frame.iloc[indexes]
        value = (
            _spearman(sample, score)
            if metric == "spearman"
            else _top_decile_mean(sample, score)
        )
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _bootstrap_contrast(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        indexes = generator.integers(0, len(frame), len(frame))
        sample = frame.iloc[indexes]
        value = _spearman(sample, left) - _spearman(sample, right)
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def _integration_outputs(
    root: Path, output_dir: Path, source_paths: dict[str, Path]
) -> dict[str, int]:
    cohort_sources = {
        "overall_241": source_paths["stage_a_scores"],
        "temporal_49": source_paths["gate2_temporal"],
        "domain_68": source_paths["gate2_domain"],
    }
    cohorts: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    paper_rows: list[pd.DataFrame] = []
    required = [
        "paper_id",
        "domain12",
        "publication_year",
        "future_structural_outcome",
        "gear_evidence_score",
        "joint_structural_score",
        "shuffled_structural_score",
    ]
    expected = {"overall_241": 241, "temporal_49": 49, "domain_68": 68}
    for cohort, path in cohort_sources.items():
        frame = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
        missing_columns = sorted(set(required) - set(frame))
        if missing_columns:
            raise ValueError(f"{cohort} missing columns: {missing_columns}")
        selected = frame[
            required + (["integration_split"] if "integration_split" in frame else [])
        ].copy()
        selected.insert(0, "cohort", cohort)
        cohorts[cohort] = selected
        paper_rows.append(selected)
        null_rows = int(selected[required].isna().any(axis=1).sum())
        duplicates = int(selected["paper_id"].duplicated().sum())
        audit_rows.append(
            {
                "dataset": cohort,
                "grain": "paper",
                "rows": len(selected),
                "unique_papers": selected["paper_id"].nunique(),
                "expected_papers": expected[cohort],
                "duplicate_paper_ids": duplicates,
                "rows_missing_required_values": null_rows,
                "ready": len(selected) == expected[cohort]
                and duplicates == 0
                and null_rows == 0,
                "source": str(path.relative_to(root)),
            }
        )
    pd.concat(paper_rows, ignore_index=True).to_csv(
        output_dir / "paper_level_scores.csv", index=False
    )
    pd.DataFrame(audit_rows).to_csv(output_dir / "cohort_audit.csv", index=False)

    arm_columns = {
        "GEAR-only": "gear_evidence_score",
        "GEAR+Graph": "joint_structural_score",
        "GEAR+shuffled-Graph": "shuffled_structural_score",
    }
    validity: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for cohort_index, (cohort, frame) in enumerate(cohorts.items()):
        clean = frame.dropna(subset=required).copy()
        overall_mean = float(clean["future_structural_outcome"].mean())
        for arm_index, (arm, score) in enumerate(arm_columns.items()):
            rho = _spearman(clean, score)
            top_mean = _top_decile_mean(clean, score)
            rho_ci = _bootstrap_metric(
                clean,
                score,
                "spearman",
                seed=SEED + cohort_index * 100 + arm_index,
                replicates=BOOTSTRAP_REPLICATES,
            )
            validity.append(
                {
                    "cohort": cohort,
                    "arm": arm,
                    "papers": len(clean),
                    "spearman_rho": rho,
                    "spearman_ci95_low": rho_ci[0],
                    "spearman_ci95_high": rho_ci[1],
                    "top_decile_outcome_mean": top_mean,
                    "overall_outcome_mean": overall_mean,
                    "top_decile_lift": (
                        top_mean / overall_mean if overall_mean else np.nan
                    ),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": SEED,
                }
            )
        for contrast_index, (left_name, left, right_name, right) in enumerate(
            [
                (
                    "GEAR+Graph",
                    "joint_structural_score",
                    "GEAR-only",
                    "gear_evidence_score",
                ),
                (
                    "GEAR+Graph",
                    "joint_structural_score",
                    "GEAR+shuffled-Graph",
                    "shuffled_structural_score",
                ),
            ]
        ):
            interval = _bootstrap_contrast(
                clean,
                left,
                right,
                seed=SEED + 2000 + cohort_index * 100 + contrast_index,
                replicates=BOOTSTRAP_REPLICATES,
            )
            contrasts.append(
                {
                    "cohort": cohort,
                    "contrast": f"{left_name} minus {right_name}",
                    "delta_spearman_rho": _spearman(clean, left)
                    - _spearman(clean, right),
                    "delta_ci95_low": interval[0],
                    "delta_ci95_high": interval[1],
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": SEED,
                }
            )
    pd.DataFrame(validity).to_csv(output_dir / "integration_validity.csv", index=False)
    pd.DataFrame(contrasts).to_csv(
        output_dir / "integration_contrasts.csv", index=False
    )
    frozen = _json(source_paths["stage_a_validation"])
    frozen_integration = frozen["integration"]
    recomputed = pd.DataFrame(validity)
    reconciliation = []
    frozen_values = {
        "GEAR-only": frozen_integration["real_hgb"]["gear_only_spearman"],
        "GEAR+Graph": frozen_integration["real_hgb"]["joint_spearman"],
        "GEAR+shuffled-Graph": frozen_integration["shuffled_hgb"]["joint_spearman"],
    }
    for arm, frozen_value in frozen_values.items():
        row_value = float(
            recomputed.loc[
                recomputed["cohort"].eq("overall_241") & recomputed["arm"].eq(arm),
                "spearman_rho",
            ].iloc[0]
        )
        reconciliation.append(
            {
                "metric": f"overall_241 {arm} spearman_rho",
                "frozen_report_value": frozen_value,
                "row_recomputed_value": row_value,
                "absolute_difference": abs(frozen_value - row_value),
                "exact_match": bool(np.isclose(frozen_value, row_value, atol=1e-12)),
                "interpretation": (
                    "reconciled"
                    if np.isclose(frozen_value, row_value, atol=1e-12)
                    else "small source/report discrepancy; retain both values and do not silently substitute"
                ),
            }
        )
    pd.DataFrame(reconciliation).to_csv(
        output_dir / "metric_reconciliation.csv", index=False
    )
    graph = frozen["graph_predictive_validity"]
    pd.DataFrame(
        [
            {
                "papers": graph["papers"],
                "spearman_rho": graph["spearman"],
                "top_decile_lift": graph["top_decile_lift"],
                "uptake_brier": graph["uptake_brier"],
                "worst_domain_spearman": graph["worst_domain_spearman"],
                "worst_fold_spearman": graph["worst_fold_spearman"],
                "claim_scope": graph["claim_scope"],
                "source": str(source_paths["stage_a_validation"].relative_to(root)),
            }
        ]
    ).to_csv(output_dir / "graph_predictive_validity.csv", index=False)
    return {"paper_rows": sum(len(frame) for frame in cohorts.values())}


def _claim_adoption_outputs(
    root: Path, output_dir: Path, source_paths: dict[str, Path]
) -> dict[str, int]:
    labels = pd.read_parquet(source_paths["claim_adoption_labels"])
    temporal = pd.read_parquet(source_paths["gate1_temporal"])
    domain = pd.read_parquet(source_paths["gate1_domain"])
    keys = ["paper_id", "claim_id"]
    if labels.duplicated(keys).any():
        raise ValueError("claim adoption labels contain duplicate paper/claim keys")
    selected = labels[
        [
            "paper_id",
            "claim_id",
            "claim_text",
            "claim_type",
            "pathway_hypothesis",
            "claim_centrality",
            "attribution_weight",
            "future_adoption",
            "adopting_paper_count",
            "adopting_context_count",
            "context_observation_status",
            "verification_passed",
            "blinded_to_future_outcome",
            "gear_run_path",
        ]
    ].copy()
    for axis, frame in (("temporal", temporal), ("domain", domain)):
        axis_columns = keys + [
            "domain12",
            "publication_year",
            "integration_split",
            "graph_percentile",
            "structural_score_at_zero",
            "structural_innovation_score",
            "shuffled_structural_score",
        ]
        rename = {
            column: f"{axis}_{column}"
            for column in axis_columns
            if column not in keys
            and column
            not in {
                "domain12",
                "publication_year",
                "integration_split",
                "graph_percentile",
            }
        }
        merge = frame[axis_columns].rename(columns=rename)
        if axis == "domain":
            merge = merge.drop(
                columns=[
                    "domain12",
                    "publication_year",
                    "integration_split",
                    "graph_percentile",
                ]
            )
        selected = selected.merge(merge, on=keys, how="left", validate="one_to_one")
    selected.to_csv(output_dir / "claim_level_adoption.csv", index=False)

    top_claim = reconstruct_strict_claim_attribution_folds(
        source_paths["gate1_temporal"],
        source_paths["gate1_domain"],
        source_paths["claim_attribution_release"],
        output_dir,
    ).set_index("axis")
    summary_rows: list[dict[str, Any]] = []
    for axis in ("temporal", "domain"):
        report = _json(source_paths[f"claim_attribution_{axis}"])
        metrics = report["metrics"]
        summary_rows.append(
            {
                "axis": axis,
                "protocol": report["claim_attribution_runtime_candidate"][
                    "evaluation_protocol"
                ],
                "papers": metrics["papers"],
                "claims": metrics["rows"],
                "spearman_rho": metrics["spearman_rho"],
                "within_paper_permutation_rho": metrics["within_paper_permutation_rho"],
                "advantage_over_permutation": metrics["advantage_over_permutation"],
                "advantage_ci95_low": metrics["paired_advantage_bootstrap_ci95"][0],
                "advantage_ci95_high": metrics["paired_advantage_bootstrap_ci95"][1],
                "strict_prediction_rows_available": True,
                "top_claim_accuracy_available": False,
                "top3_retrieval_available": True,
                "top3_retrieval_metric": "paper_level_recall_at_3_any_important_claim",
                "ndcg_available": False,
                "adoption_lift_available": False,
                "limitation": "The deprecated top-1 accuracy endpoint is not reported. Recall@3 is the frozen primary; Precision@3, NDCG@3, and MRR are secondary.",
                "top_claim_accuracy": top_claim.loc[axis, "learned_recall_at_3"],
                "uniform_top_claim_accuracy": top_claim.loc[
                    axis, "uniform_random_recall_at_3"
                ],
                "top_claim_advantage": top_claim.loc[axis, "advantage_over_uniform"],
                "top_claim_advantage_ci95_low": top_claim.loc[
                    axis, "advantage_over_uniform_ci95_low"
                ],
                "top_claim_advantage_ci95_high": top_claim.loc[
                    axis, "advantage_over_uniform_ci95_high"
                ],
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "claim_adoption_validity.csv", index=False
    )
    audit = pd.DataFrame(
        [
            {
                "dataset": "claim_level_adoption",
                "grain": "paper_claim",
                "rows": len(selected),
                "unique_papers": selected["paper_id"].nunique(),
                "duplicate_keys": int(selected.duplicated(keys).sum()),
                "unresolved_context_rows": int(
                    (~selected["context_observation_status"].eq("resolved")).sum()
                ),
                "source": str(source_paths["claim_adoption_labels"].relative_to(root)),
            }
        ]
    )
    audit.to_csv(output_dir / "claim_adoption_audit.csv", index=False)
    return {
        "claim_rows": len(selected),
        "claim_papers": selected["paper_id"].nunique(),
        "top_claim_axes": len(top_claim),
    }


def _existing_review_outputs(output_dir: Path, pack_dir: Path) -> dict[str, int]:
    b_tasks = _jsonl(pack_dir / "claim_b_tasks.jsonl")
    b_annotations = _jsonl(pack_dir / "claim_b_annotations.jsonl")
    c_tasks = _jsonl(pack_dir / "claim_c_tasks.jsonl")
    c_annotations = _jsonl(pack_dir / "claim_c_annotations.jsonl")
    b_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in b_annotations:
        b_by_task[str(row["task_id"])].append(row)
    exact_tasks = 0
    compared_tasks = 0
    field_equal = 0
    field_total = 0
    fields = [
        "inventory_valid",
        "relation",
        "residual_novelty",
        "manuscript_support",
        "trace_complete",
    ]
    for rows in b_by_task.values():
        if len(rows) != 2:
            continue
        compared_tasks += 1
        left, right = rows
        left_assessments = {row["claim_alias"]: row for row in left["assessments"]}
        right_assessments = {row["claim_alias"]: row for row in right["assessments"]}
        inventory_equal = left["inventory_complete"] == right["inventory_complete"]
        comparisons = [inventory_equal]
        field_equal += int(inventory_equal)
        field_total += 1
        for alias in sorted(set(left_assessments) & set(right_assessments)):
            for field in fields:
                equal = (
                    left_assessments[alias][field] == right_assessments[alias][field]
                )
                comparisons.append(equal)
                field_equal += int(equal)
                field_total += 1
        exact_tasks += int(all(comparisons))
    c_preferences = Counter(str(row["preference"]) for row in c_annotations)
    same_content = 0
    for task in c_tasks:
        left = sorted(
            (
                claim["claim_text"],
                sorted(e["evidence_key"] for e in claim["manuscript_evidence"]),
            )
            for claim in task["left"]["claims"]
        )
        right = sorted(
            (
                claim["claim_text"],
                sorted(e["evidence_key"] for e in claim["manuscript_evidence"]),
            )
            for claim in task["right"]["claims"]
        )
        same_content += int(left == right)
    summary = [
        {
            "review_set": "existing_claim_b",
            "metric": "task_exact_agreement",
            "value": exact_tasks / compared_tasks if compared_tasks else np.nan,
            "numerator": exact_tasks,
            "denominator": compared_tasks,
            "usable_for_final_result": False,
            "reason": "Antecedent relation evidence is absent, so residual-novelty validity is not measurable.",
        },
        {
            "review_set": "existing_claim_b",
            "metric": "assessment_field_agreement",
            "value": field_equal / field_total if field_total else np.nan,
            "numerator": field_equal,
            "denominator": field_total,
            "usable_for_final_result": False,
            "reason": "Agreement is descriptive only while relation evidence coverage is zero.",
        },
        {
            "review_set": "existing_claim_c",
            "metric": "tie_rate",
            "value": c_preferences.get("TIE", 0) / len(c_annotations),
            "numerator": c_preferences.get("TIE", 0),
            "denominator": len(c_annotations),
            "usable_for_final_result": False,
            "reason": "The existing arms contain identical claims and differ only in order.",
        },
        {
            "review_set": "existing_claim_c",
            "metric": "identical_arm_content_rate",
            "value": same_content / len(c_tasks),
            "numerator": same_content,
            "denominator": len(c_tasks),
            "usable_for_final_result": False,
            "reason": "Replacement tasks must compare content-different top-k sets.",
        },
    ]
    pd.DataFrame(summary).to_csv(
        output_dir / "independent_review_summary.csv", index=False
    )
    readiness = []
    for task in b_tasks:
        claims = task["claims"]
        covered = sum(bool(claim.get("relation_evidence")) for claim in claims)
        readiness.append(
            {
                "task_id": task["task_id"],
                "paper_alias": task["paper_alias"],
                "claims": len(claims),
                "claims_with_relation_evidence": covered,
                "relation_evidence_coverage": covered / len(claims),
                "ready_for_residual_novelty_review": covered == len(claims),
                "required_data": "Closest antecedent excerpts and stable relation evidence keys for every claim.",
            }
        )
    pd.DataFrame(readiness).to_csv(output_dir / "claim_b_readiness.csv", index=False)
    return {
        "existing_claim_b_tasks": len(b_tasks),
        "existing_claim_c_tasks": len(c_tasks),
    }


def _claim_b_completion_outputs(output_dir: Path) -> dict[str, int]:
    """Summarise the claim-level completion endpoint without inferring missing art."""
    path = output_dir / "claim_b_evidence_completion.csv"
    if not path.is_file():
        return {"claim_b_evaluable_claims": 0, "claim_b_evaluable_papers": 0}
    frame = pd.read_csv(path)
    required = {
        "claim_id",
        "paper_alias",
        "manuscript_evidence_keys",
        "prior_work_identifier",
        "prior_work_excerpt",
        "prior_work_location",
        "cutoff_verified",
        "relation_evidence_key",
        "relation_rationale",
        "relation_status",
        "residual_novelty_eligible",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Claim B completion table missing columns: {missing}")
    eligible = frame[frame["residual_novelty_eligible"].astype(bool)].copy()
    evidence_columns = list(
        required - {"claim_id", "paper_alias", "residual_novelty_eligible"}
    )
    incomplete = eligible[evidence_columns].isna().any(axis=1)
    if incomplete.any():
        raise ValueError("Claim B marks claims evaluable without complete evidence")
    summary = pd.DataFrame(
        [
            {
                "metric": "claims_in_original_pack",
                "value": frame["claim_id"].nunique(),
            },
            {
                "metric": "claims_evaluable_for_residual_support",
                "value": eligible["claim_id"].nunique(),
            },
            {
                "metric": "papers_with_evaluable_claims",
                "value": eligible["paper_alias"].nunique(),
            },
            {
                "metric": "claims_excluded_for_incomplete_relation_evidence",
                "value": frame["claim_id"].nunique() - eligible["claim_id"].nunique(),
            },
        ]
    )
    summary.to_csv(output_dir / "claim_b_evidence_coverage.csv", index=False)
    return {
        "claim_b_evaluable_claims": int(eligible["claim_id"].nunique()),
        "claim_b_evaluable_papers": int(eligible["paper_alias"].nunique()),
    }


def _trace_index(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["evidence_id"]): row for row in _jsonl(path) if row.get("evidence_id")
    }


def _collect_text(value: Any, key: str = "") -> list[str]:
    forbidden = {"relation", "relation_label", "confidence", "score", "rationale"}
    if key.casefold() in forbidden:
        return []
    if isinstance(value, str) and key.casefold() in {
        "text",
        "title",
        "claim",
        "target_claim",
        "candidate_excerpt",
        "shared_base",
        "residual_delta",
        "abstract",
    }:
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [
            text
            for child_key, child in value.items()
            for text in _collect_text(child, str(child_key))
        ]
    if isinstance(value, list):
        return [text for child in value for text in _collect_text(child, key)]
    return []


def _excerpt(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload") or {}
    texts = _collect_text(payload)
    page = payload.get("page") if isinstance(payload, dict) else None
    return {
        "evidence_key": str(record["evidence_id"]),
        "evidence_kind": str(record.get("kind") or "unknown"),
        "excerpt": " | ".join(dict.fromkeys(texts))[:4000]
        or "Evidence record available; inspect the source trace.",
        "page": int(page) if isinstance(page, int) and page >= 0 else None,
    }


def _claim_evidence(
    group: pd.DataFrame, selected_claim_ids: list[str]
) -> list[dict[str, Any]]:
    run_path = Path(str(group.iloc[0]["gear_run_path"]))
    trace = _trace_index(run_path / "evidence_trace.jsonl")
    bundle = _json(run_path / "review_bundle.json")
    inventory = bundle.get("state", {}).get("claim_inventory") or []
    inventory_by_id = {str(row.get("claim_id")): row for row in inventory}
    relation_keys = sorted(
        {
            str(key)
            for point in (
                bundle.get("state", {}).get("canonical_points") or {}
            ).values()
            for key in point.get("relation_evidence_keys", [])
        }
    )
    relation = [_excerpt(trace[key]) for key in relation_keys if key in trace][:1]
    aliases = {
        claim_id: f"CL-{index:02d}"
        for index, claim_id in enumerate(sorted(group["claim_id"].astype(str)), start=1)
    }
    claims: list[dict[str, Any]] = []
    for claim_id in selected_claim_ids:
        row = group[group["claim_id"].astype(str).eq(claim_id)].iloc[0]
        keys = inventory_by_id.get(claim_id, {}).get("manuscript_evidence_keys", [])
        manuscript = [_excerpt(trace[str(key)]) for key in keys if str(key) in trace][
            :1
        ]
        if not manuscript:
            raise ValueError(
                f"missing manuscript evidence for {row['paper_id']}:{claim_id}"
            )
        claims.append(
            {
                "claim_alias": aliases[claim_id],
                "claim_text": str(row["claim_text"]),
                "manuscript_evidence": manuscript,
                "relation_evidence": relation,
            }
        )
    return claims


def _claim_c_replacement(
    output_dir: Path, source_paths: dict[str, Path]
) -> dict[str, int]:
    labels = pd.read_parquet(source_paths["claim_adoption_labels"])[
        ["paper_id", "claim_id", "claim_text", "gear_run_path"]
    ]
    gate = pd.read_parquet(source_paths["gate1_temporal"])
    frame = gate.merge(
        labels, on=["paper_id", "claim_id"], how="inner", validate="one_to_one"
    )
    eligible: list[tuple[str, list[str], list[str]]] = []
    for paper_id, group in frame.groupby("paper_id", sort=True):
        evidence_only = (
            group.sort_values(
                ["structural_score_at_zero", "claim_id"], ascending=[False, True]
            )
            .head(CLAIM_C_TOP_K)["claim_id"]
            .astype(str)
            .tolist()
        )
        joint = (
            group.sort_values(
                ["structural_innovation_score", "claim_id"], ascending=[False, True]
            )
            .head(CLAIM_C_TOP_K)["claim_id"]
            .astype(str)
            .tolist()
        )
        if set(evidence_only) != set(joint):
            eligible.append((str(paper_id), evidence_only, joint))
    tasks: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    seal: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for index, (paper_id, evidence_ids, joint_ids) in enumerate(eligible, start=1):
        group = frame[frame["paper_id"].astype(str).eq(paper_id)]
        evidence_claims = _claim_evidence(group, evidence_ids)
        joint_claims = _claim_evidence(group, joint_ids)
        joint_left = int(_stable_digest(SEED, "claim-c-side", paper_id), 16) % 2 == 0
        left, right = (
            (joint_claims, evidence_claims)
            if joint_left
            else (evidence_claims, joint_claims)
        )
        task_id = "CC-F4-" + _stable_digest(SEED, paper_id)[:14]
        task = {
            "contract": "gear_claim_c_blind_pairwise_task_v1",
            "task_id": task_id,
            "paper_alias": f"PC-F4-{index:03d}",
            "left": {"side": "LEFT", "claims": left},
            "right": {"side": "RIGHT", "claims": right},
        }
        tasks.append(task)
        for slot in (1, 2):
            templates.append(
                {
                    "contract": "gear_claim_c_independent_review_v1",
                    "task_id": task_id,
                    "annotation_slot": slot,
                    "annotator_id": None,
                    "preference": None,
                    "confidence": None,
                    "rationale": None,
                    "evidence_keys": [],
                }
            )
        domain = str(group.iloc[0]["domain12"])
        percentile = float(group.iloc[0]["graph_percentile"])
        tier = (
            "low" if percentile < 33.333 else "mid" if percentile < 66.667 else "high"
        )
        seal.append(
            {
                "task_id": task_id,
                "paper_alias": task["paper_alias"],
                "paper_id": paper_id,
                "left_arm": "GEAR+Graph" if joint_left else "GEAR-only",
                "right_arm": "GEAR-only" if joint_left else "GEAR+Graph",
                "gear_only_claim_ids": evidence_ids,
                "gear_graph_claim_ids": joint_ids,
                "domain12": domain,
                "graph_percentile": percentile,
                "graph_tier": tier,
            }
        )
        audit.append(
            {
                "task_id": task_id,
                "domain12": domain,
                "graph_tier": tier,
                "source_claims": len(group),
                "claims_per_arm": CLAIM_C_TOP_K,
                "different_selected_claims": set(evidence_ids) != set(joint_ids),
                "left_manuscript_excerpts": sum(
                    len(row["manuscript_evidence"]) for row in left
                ),
                "right_manuscript_excerpts": sum(
                    len(row["manuscript_evidence"]) for row in right
                ),
                "left_relation_excerpts": sum(
                    len(row["relation_evidence"]) for row in left
                ),
                "right_relation_excerpts": sum(
                    len(row["relation_evidence"]) for row in right
                ),
                "equal_claim_budget": len(left) == len(right) == CLAIM_C_TOP_K,
                "equal_manuscript_evidence_cap": all(
                    len(row["manuscript_evidence"]) <= 1 for row in [*left, *right]
                ),
                "future_outcome_in_task": False,
                "graph_scores_in_task": False,
            }
        )
    _write_jsonl(output_dir / "claim_c_replacement_tasks.jsonl", tasks)
    _write_jsonl(output_dir / "claim_c_replacement_review_templates.jsonl", templates)
    _write_json(output_dir / "claim_c_replacement_sealed_key.json", seal)
    pd.DataFrame(audit).to_csv(
        output_dir / "claim_c_replacement_audit.csv", index=False
    )
    distribution = (
        pd.DataFrame(audit)
        .groupby(["domain12", "graph_tier"], observed=True)
        .size()
        .rename("tasks")
        .reset_index()
    )
    distribution.to_csv(
        output_dir / "claim_c_replacement_distribution.csv", index=False
    )
    return {"claim_c_replacement_tasks": len(tasks)}


def _load_candidate_runs(
    run_root: Path, reference_ids: set[str]
) -> tuple[dict[str, StructuredReview], list[dict[str, Any]]]:
    lanes = [
        "deepseek_batch",
        "smoke_deepseek_fixed",
        "codex_batch",
        "codex_batch2",
        "codex_retry1",
        "codex_retry2",
        "codex_retry3",
        "codex_retry4",
        "codex_fig4_increment_20260829_corrected",
    ]
    selected: dict[str, tuple[int, StructuredReview, Path, str, str | None]] = {}
    for priority, lane in enumerate(lanes):
        directory = run_root / lane
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*/review.json")):
            try:
                review = StructuredReview.model_validate(_json(path))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            if review.paper_id not in reference_ids:
                continue
            bundle_path = path.with_name("review_bundle.json")
            bundle = _json(bundle_path) if bundle_path.is_file() else {}
            status = bundle.get("status", "unknown")
            model_id = (bundle.get("agent_review") or {}).get("model_id")
            if status not in {"complete", "limited"}:
                continue
            if (
                review.paper_id not in selected
                or priority >= selected[review.paper_id][0]
            ):
                selected[review.paper_id] = (
                    priority,
                    review,
                    path,
                    str(status),
                    str(model_id) if model_id else None,
                )
    reviews = {paper_id: item[1] for paper_id, item in selected.items()}
    audit = [
        {
            "paper_id": paper_id,
            "candidate_source_lane": item[2].parents[1].name,
            "candidate_status": item[3],
            "candidate_model_id": item[4],
            "candidate_review_path": str(item[2]),
        }
        for paper_id, item in sorted(selected.items())
    ]
    return reviews, audit


def _point_metadata(review: StructuredReview) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    sections: list[tuple[str, list[ReviewPoint]]] = [
        (
            "novelty",
            [
                *review.novelty.supporting_points,
                *review.novelty.limiting_points,
                *review.novelty.uncertain_points,
            ],
        ),
        ("strengths", review.strengths),
        ("weaknesses", review.weaknesses),
        ("questions", review.questions),
    ]
    for section, points in sections:
        for point in points:
            result[point.point_id] = (section, str(point.aspect.value))
    return result


def _alignment_outputs(
    root: Path, output_dir: Path, source_paths: dict[str, Path]
) -> dict[str, int]:
    reference_path = source_paths["human_structured_reviews"]
    references = [
        StructuredReview.model_validate(row) for row in _jsonl(reference_path)
    ]
    references.sort(key=lambda row: row.paper_id)
    reference_by_id = {row.paper_id: row for row in references}
    if len(reference_by_id) != len(references):
        raise ValueError("human structured reviews contain duplicate paper IDs")
    candidates, candidate_audit = _load_candidate_runs(
        source_paths["candidate_run_root"], set(reference_by_id)
    )
    missing = sorted(set(reference_by_id) - set(candidates))
    aligned_ids = sorted(set(reference_by_id) & set(candidates))
    aligned_references = [reference_by_id[paper_id] for paper_id in aligned_ids]
    aligned_candidates = [candidates[paper_id] for paper_id in aligned_ids]
    correct_packages = [
        build_blind_match_package(reference, candidate, top_k=5)
        for reference, candidate in zip(
            aligned_references, aligned_candidates, strict=True
        )
    ]
    wrong_pairs = wrong_paper_shuffle(aligned_references, aligned_candidates)
    wrong_packages = [
        build_blind_match_package(reference, candidate, top_k=5)
        for reference, candidate in wrong_pairs
    ]
    _write_jsonl(
        output_dir / "reviewer_alignment_correct_tasks.jsonl",
        [package.model_dump(mode="json") for package in correct_packages],
    )
    _write_jsonl(
        output_dir / "reviewer_alignment_wrong_paper_tasks.jsonl",
        [package.model_dump(mode="json") for package in wrong_packages],
    )
    label_rows: list[dict[str, Any]] = []
    for condition, packages, refs, cands in (
        ("correct_pair", correct_packages, aligned_references, aligned_candidates),
        (
            "wrong_paper",
            wrong_packages,
            aligned_references,
            [pair[1] for pair in wrong_pairs],
        ),
    ):
        for package, reference, candidate in zip(packages, refs, cands, strict=True):
            reference_meta = _point_metadata(reference)
            candidate_meta = _point_metadata(candidate)
            candidate_aspects = [
                candidate_meta[point.point_id][1] for point in candidate.all_points()
            ]
            shuffled = (
                candidate_aspects[1:] + candidate_aspects[:1]
                if candidate_aspects
                else []
            )
            shuffled_aspect = {
                point.point_id: aspect
                for point, aspect in zip(candidate.all_points(), shuffled, strict=True)
            }
            for reference_id, candidate_id in package.candidate_pairs:
                label_rows.append(
                    {
                        "condition": condition,
                        "task_id": package.task_id,
                        "paper_id_hash": package.paper_id_hash,
                        "reference_point_id": reference_id,
                        "candidate_point_id": candidate_id,
                        "reference_section": reference_meta[reference_id][0],
                        "candidate_section": candidate_meta[candidate_id][0],
                        "reference_aspect": reference_meta[reference_id][1],
                        "candidate_aspect": candidate_meta[candidate_id][1],
                        "shuffled_candidate_aspect": shuffled_aspect[candidate_id],
                        "label": None,
                        "confidence": None,
                        "rationale": None,
                    }
                )
    pd.DataFrame(label_rows).to_csv(
        output_dir / "reviewer_alignment_label_template.csv", index=False
    )
    audit_by_id = {row["paper_id"]: row for row in candidate_audit}
    audit_rows = []
    for paper_id, reference in sorted(reference_by_id.items()):
        row = audit_by_id.get(paper_id, {})
        audit_rows.append(
            {
                "paper_id": paper_id,
                "reference_available": True,
                "candidate_available": paper_id in candidates,
                "candidate_source_lane": row.get("candidate_source_lane"),
                "candidate_status": row.get("candidate_status"),
                "candidate_model_id": row.get("candidate_model_id"),
                "candidate_review_path": row.get("candidate_review_path"),
                "reference_release_path": str(reference_path.relative_to(root)),
                "alignment_labels_complete": False,
                "soft_metrics_ready": False,
            }
        )
    pd.DataFrame(audit_rows).to_csv(output_dir / "reviewer_run_audit.csv", index=False)
    controls = [
        {
            "condition": "correct_pair",
            "tasks": len(correct_packages),
            "candidate_pairs": sum(
                len(row.candidate_pairs) for row in correct_packages
            ),
            "labels_complete": False,
            "purpose": "Primary GEAR versus published-review reconstruction alignment.",
        },
        {
            "condition": "wrong_paper",
            "tasks": len(wrong_packages),
            "candidate_pairs": sum(len(row.candidate_pairs) for row in wrong_packages),
            "labels_complete": False,
            "purpose": "Negative control for paper-specific alignment.",
        },
        {
            "condition": "within_paper_aspect_shuffle",
            "tasks": len(correct_packages),
            "candidate_pairs": sum(
                len(row.candidate_pairs) for row in correct_packages
            ),
            "labels_complete": False,
            "purpose": "Negative control for aspect-label agreement using the same judged pairs.",
        },
    ]
    pd.DataFrame(controls).to_csv(
        output_dir / "reviewer_alignment_controls.csv", index=False
    )
    status = pd.DataFrame(
        [
            {
                "papers_expected": len(reference_by_id),
                "papers_paired": len(aligned_ids),
                "papers_missing_candidate": len(missing),
                "missing_paper_ids": ";".join(missing),
                "soft_alignment_status": "pending_independent_session_labels",
                "required_before_claim": "Complete SAME_POINT/PARTIAL_POINT/CONTRADICTORY/NO_MATCH labels; then require correct-pair soft F1 above the 95th percentile of wrong-paper controls.",
            }
        ]
    )
    status.to_csv(output_dir / "reviewer_soft_alignment.csv", index=False)
    return {
        "reviewer_reference_papers": len(reference_by_id),
        "reviewer_paired_papers": len(aligned_ids),
        "reviewer_pair_rows": len(label_rows),
    }


def _source_paths(root: Path) -> dict[str, Path]:
    stage_b = root / "outputs/gear/stage_b_targeted_expansion_20260828"
    replication = root / "outputs/gear/graph_rescue_replication_20260828"
    return {
        "stage_a_scores": replication / "stage_a/stage_a_three_arm_scores.csv",
        "stage_a_validation": replication / "stage_a/stage_a_validation.json",
        "gate2_temporal": stage_b / "gate2_temporal/gate2_integration_frame.parquet",
        "gate2_domain": stage_b / "gate2_domain/gate2_integration_frame.parquet",
        "claim_adoption_labels": stage_b
        / "claim_adoption_labels/claim_adoption_labels.parquet",
        "gate1_temporal": stage_b / "gate1_temporal/gate1_mechanism_dataset.parquet",
        "gate1_domain": stage_b / "gate1_domain/gate1_mechanism_dataset.parquet",
        "claim_attribution_temporal": stage_b
        / "claim_attribution_release/gate1_temporal.json",
        "claim_attribution_domain": stage_b
        / "claim_attribution_release/gate1_domain.json",
        "claim_attribution_release": stage_b / "claim_attribution_release",
        "expert_pack": replication / "expert_annotation_pack",
        "human_structured_reviews": root
        / "outputs/gear/human_review_reconstruction/nature_dev100_human_v2_20260824/human_structured_reviews.jsonl",
        "revision_issue_labels": root
        / "outputs/gear/human_review_reconstruction/nature_dev100_human_v2_20260824/revision_issue_labels.jsonl",
        "candidate_run_root": root / "outputs/gear/agent_runs/nature_dev100_v1",
    }


def prepare(root: Path, output_dir: Path) -> dict[str, Any]:
    """Build all currently identifiable Fig. 4new data products."""
    sources = _source_paths(root)
    required_files = [
        path
        for key, path in sources.items()
        if key not in {"expert_pack", "candidate_run_root", "claim_attribution_release"}
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Fig. 4new inputs: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    counts.update(_integration_outputs(root, output_dir, sources))
    counts.update(_claim_adoption_outputs(root, output_dir, sources))
    counts.update(_existing_review_outputs(output_dir, sources["expert_pack"]))
    counts.update(_claim_b_completion_outputs(output_dir))
    counts.update(_claim_c_replacement(output_dir, sources))
    counts.update(_alignment_outputs(root, output_dir, sources))
    required_data = [
        {
            "evaluation": "Claim B residual contribution validity",
            "status": "locally_completed_subset_only",
            "data_needed": "For every reviewed claim: closest antecedent excerpt, stable evidence key, publication date before cutoff, and claim-to-antecedent relation trace.",
            "why": "Only claims in claim_b_evidence_completion.csv marked eligible may enter residual-support metrics; excluded claims remain unassessed, not novel by default.",
        },
        {
            "evaluation": "Claim C independent-session preference",
            "status": "tasks_ready_labels_pending",
            "data_needed": "At least one independent AI review per prepared task; additional reviews are optional.",
            "why": "The replacement tasks now compare equal-budget, content-different top-3 claim sets.",
        },
        {
            "evaluation": "Published-review soft alignment",
            "status": "tasks_ready_labels_pending",
            "data_needed": "Independent-session pair labels for correct and wrong-paper packages using existing four-way match labels.",
            "why": "Soft precision/recall/F1 and aspect agreement cannot be computed before pair labels exist.",
        },
        {
            "evaluation": "Additional claim-adoption ranking metrics",
            "status": "not_identifiable_from_frozen_release",
            "data_needed": "Strict temporal/domain fold-level claim predictions with paper_id and claim_id.",
            "why": "The frozen release exposes aggregate rho and permutation advantages only; using all 1,442 rows would mix evaluation and non-evaluation rows.",
        },
    ]
    pd.DataFrame(required_data).to_csv(output_dir / "required_data.csv", index=False)
    file_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "data_manifest.json":
            file_rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "generated_at_date": "2026-08-29",
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "scope": "minimum data preparation and evaluation data only; no figure rendering",
        "new_contracts_added": False,
        "counts": counts,
        "sources": {
            key: str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for key, path in sources.items()
        },
        "outputs": file_rows,
    }
    _write_json(output_dir / "data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fig04/new/data_20260829"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    manifest = prepare(root, output_dir)
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
