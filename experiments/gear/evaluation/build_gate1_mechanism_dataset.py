"""Assemble the leakage-safe claim-level dataset used by Gate 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gear.claim_attribution import (
    CLAIM_TYPES,
    FEATURE_SCHEMA_VERSION,
    FORECAST_ROLES,
    PATHWAYS,
    T0_FEATURE_NAMES,
)
from gear.graph_calibration import load_forecast_analog_index

from .claim_attribution_training import (
    fit_claim_attribution_development_holdout,
    run_claim_attribution_oof,
)
from .stage_a_dataset import OOF_PATH, SCORE_PATH

ANATOMY_MANIFEST_PATH = Path(
    "data/calibration/graph_calibration/primary16_forecast_anatomy_v1/manifest.json"
)

ATTRIBUTION_FEATURES = list(T0_FEATURE_NAMES)


def build_gate1_dataset(
    claim_adoption_path: Path,
    perturbation_predictions_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260828,
    epsilon: float = 0.1,
    split_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Cross-fit claim attribution and join only OOF future targets/predictions."""
    labels = pd.read_parquet(claim_adoption_path)
    perturbation = pd.read_parquet(perturbation_predictions_path)
    oof = pd.read_parquet(OOF_PATH)
    score = pd.read_parquet(
        SCORE_PATH,
        columns=["paper_id", "prospective_5y_diffusion_percentile"],
    )
    anatomy_index = load_forecast_analog_index(ANATOMY_MANIFEST_PATH.resolve())
    anatomy = _forecast_anatomy_shares(anatomy_index.table())
    frame = labels.merge(
        oof[
            [
                "paper_id",
                "outer_fold_id",
                "domain12",
                "publication_year",
                "future_uptake",
                "realized_diffusion_target",
                "expected_diffusion_score",
            ]
        ],
        on="paper_id",
        how="inner",
        validate="many_to_one",
    )
    frame = frame.merge(score, on="paper_id", how="inner", validate="many_to_one")
    frame = frame.merge(anatomy, on="paper_id", how="inner", validate="many_to_one")
    frame = _materialize_exact_t0_features(frame)
    frame = frame.merge(
        perturbation[
            [
                "paper_id",
                "perturbation_target_fold",
                "perturbation_head_p",
                "shuffled_perturbation_head_p",
            ]
        ],
        on="paper_id",
        how="inner",
        validate="many_to_one",
    )
    if split_manifest_path is not None:
        frame = frame.merge(
            _integration_splits(split_manifest_path),
            on="paper_id",
            how="inner",
            validate="many_to_one",
        )
    else:
        frame["integration_split"] = "development"
    frame["future_claim_adoption_breadth"] = pd.to_numeric(
        frame["future_adoption"], errors="coerce"
    )
    development = frame[frame["integration_split"].eq("development")]
    holdout = frame[frame["integration_split"].ne("development")]
    if holdout.empty:
        attribution, attribution_report = run_claim_attribution_oof(
            development,
            feature_columns=ATTRIBUTION_FEATURES,
            fold_column="outer_fold_id",
            seed=seed,
        )
    else:
        development_attribution, holdout_attribution, attribution_report = (
            fit_claim_attribution_development_holdout(
                development,
                holdout,
                feature_columns=ATTRIBUTION_FEATURES,
                fold_column="outer_fold_id",
                seed=seed,
            )
        )
        attribution = pd.concat(
            [development_attribution, holdout_attribution], ignore_index=True
        )
    frame = frame.drop(columns=["attribution_weight"], errors="ignore").merge(
        attribution[["paper_id", "claim_id", "attribution_weight"]],
        on=["paper_id", "claim_id"],
        how="inner",
        validate="one_to_one",
    )
    shuffled_diffusion = _shuffle_within_field_year(frame, seed)
    frame["diffusion_potential"] = _joint_graph_signal(
        frame["expected_diffusion_score"], frame["perturbation_head_p"]
    )
    shuffled_signal = _joint_graph_signal(
        shuffled_diffusion, frame["shuffled_perturbation_head_p"]
    )
    frame["evidence_gate"] = _evidence_gate(frame)
    frame["structural_innovation_score"] = _fusion_score(
        frame, frame["diffusion_potential"], epsilon
    )
    frame["shuffled_structural_score"] = _fusion_score(frame, shuffled_signal, epsilon)
    frame["structural_score_at_zero"] = _fusion_score(
        frame, pd.Series(0.0, index=frame.index), epsilon
    )
    frame["structural_score_at_one"] = _fusion_score(
        frame, pd.Series(1.0, index=frame.index), epsilon
    )
    frame["future_structural_outcome"] = pd.to_numeric(
        frame["future_uptake"], errors="coerce"
    ) * _joint_graph_signal(
        frame["realized_diffusion_target"], frame["perturbation_target_fold"]
    )
    frame["graph_percentile"] = frame["prospective_5y_diffusion_percentile"]
    output_columns = [
        "paper_id",
        "claim_id",
        "outer_fold_id",
        "domain12",
        "publication_year",
        "integration_split",
        "context_observation_status",
        "claim_type",
        "pathway_hypothesis",
        "claim_t0_schema_version",
        *T0_FEATURE_NAMES,
        "attribution_weight",
        "future_adoption",
        "evidence_gate",
        "diffusion_potential",
        "structural_innovation_score",
        "shuffled_structural_score",
        "structural_score_at_zero",
        "structural_score_at_one",
        "future_structural_outcome",
        "graph_percentile",
        "structural_contribution_share",
        "opportunity_context_share",
        "anatomy_limited",
    ]
    output = frame[output_columns].dropna().reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gate1_mechanism_dataset.parquet"
    output.to_parquet(output_path, index=False)
    report = {
        "contract": "gear_gate1_mechanism_dataset_v1",
        "rows": len(output),
        "papers": int(output["paper_id"].nunique()),
        "claims": int(output[["paper_id", "claim_id"]].drop_duplicates().shape[0]),
        "score_deciles": int(
            np.floor(output["graph_percentile"] / 10).clip(0, 9).nunique()
        ),
        "anatomy_profile_counts": {
            "structural_driven": int(
                output["structural_contribution_share"].gt(0.5).sum()
            ),
            "opportunity_driven": int(
                output["opportunity_context_share"].ge(0.5).sum()
            ),
            "limited": int(output["anatomy_limited"].astype(bool).sum()),
        },
        "split_papers": {
            str(split): int(group["paper_id"].nunique())
            for split, group in output.groupby("integration_split", observed=True)
        },
        "claim_attribution": attribution_report,
        "claim_attribution_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "claim_attribution_feature_names": list(T0_FEATURE_NAMES),
        "future_features_used_for_training": False,
        "future_labels_role": "development_training_target_and_evaluation_target",
        "split_manifest_applied": split_manifest_path is not None,
        "split_manifest_sha256": (
            "sha256:" + hashlib.sha256(split_manifest_path.read_bytes()).hexdigest()
            if split_manifest_path is not None
            else None
        ),
        "output_sha256": "sha256:"
        + hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (output_dir / "gate1_mechanism_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _evidence_gate(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["manuscript_validity"]).clip(0.0, 1.0)
        * pd.to_numeric(frame["evidence_coverage"]).clip(0.0, 1.0)
        * (1.0 - pd.to_numeric(frame["antecedent_risk"]).clip(0.0, 1.0))
        * pd.to_numeric(frame["residual_novelty"]).clip(0.0, 1.0)
    )


def _forecast_anatomy_shares(anatomy: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, pd.Series] = {}
    for role in FORECAST_ROLES:
        values[role] = (
            pd.to_numeric(anatomy[f"uptake_contribution__{role}"]).abs()
            + pd.to_numeric(anatomy[f"conditional_contribution__{role}"]).abs()
        )
    denominator = sum(values.values())
    structural = (values["substantive_innovation"] + values["t0_potential"]).div(
        denominator.replace(0.0, np.nan)
    )
    output = anatomy[["paper_id", "anatomy_limited"]].copy()
    for role in FORECAST_ROLES:
        output[f"anatomy_role__{role}"] = (
            values[role].div(denominator.replace(0.0, np.nan)).fillna(0.0)
        )
    output["structural_contribution_share"] = structural.fillna(0.0)
    output["opportunity_context_share"] = 1.0 - output["structural_contribution_share"]
    return output


def _materialize_exact_t0_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail closed unless every claim has the exact runtime T0 representation."""
    required = {
        "claim_centrality",
        "claim_type",
        "pathway_hypothesis",
        "claim_t0_schema_version",
        *(f"anatomy_role__{role}" for role in FORECAST_ROLES),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"claim attribution T0 columns missing: {missing}")
    if frame[list(required)].isna().any().any():
        raise ValueError("claim attribution T0 columns contain null values")
    if not frame["claim_t0_schema_version"].eq(FEATURE_SCHEMA_VERSION).all():
        raise ValueError("claim attribution T0 schema version mismatch")
    if not frame["claim_type"].isin(CLAIM_TYPES).all():
        raise ValueError("claim attribution contains unknown formal claim_type")
    if not frame["pathway_hypothesis"].isin(PATHWAYS).all():
        raise ValueError("claim attribution contains unknown pathway_hypothesis")
    output = frame.copy()
    for value in CLAIM_TYPES:
        output[f"claim_type__{value}"] = output["claim_type"].eq(value).astype(float)
    for value in PATHWAYS:
        output[f"pathway__{value}"] = (
            output["pathway_hypothesis"].eq(value).astype(float)
        )
    numeric = output[list(T0_FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("claim attribution T0 features are not finite numeric values")
    if not (numeric.ge(0.0) & numeric.le(1.0)).all().all():
        raise ValueError("claim attribution T0 features must be in [0, 1]")
    output[list(T0_FEATURE_NAMES)] = numeric
    return output


def _integration_splits(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_uses_future_outcomes") is not False:
        raise ValueError("integration split manifest must be outcome-blind")
    rows = [
        {
            "paper_id": str(case["paper_id"]),
            "integration_split": str(case["integration_split"]),
        }
        for case in payload.get("cases", [])
    ]
    frame = pd.DataFrame(rows)
    if frame.empty or frame["paper_id"].duplicated().any():
        raise ValueError("integration split manifest must contain unique cases")
    return frame


def _fusion_score(
    frame: pd.DataFrame, graph_signal: pd.Series, epsilon: float
) -> pd.Series:
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    attributed = pd.to_numeric(graph_signal).clip(0.0, 1.0) * pd.to_numeric(
        frame["attribution_weight"]
    ).clip(0.0, 1.0)
    mechanism = pd.to_numeric(frame["mechanism_validity"]).clip(0.0, 1.0)
    return (
        frame["evidence_gate"]
        * (epsilon + (1.0 - epsilon) * attributed)
        * np.sqrt(mechanism)
    )


def _joint_graph_signal(first: pd.Series, second: pd.Series) -> pd.Series:
    return 0.5 * pd.to_numeric(first).clip(0.0, 1.0) + 0.5 * pd.to_numeric(second).clip(
        0.0, 1.0
    )


def _shuffle_within_field_year(frame: pd.DataFrame, seed: int) -> pd.Series:
    paper = frame[
        ["paper_id", "domain12", "publication_year", "expected_diffusion_score"]
    ].drop_duplicates("paper_id")
    generator = np.random.default_rng(seed)
    shuffled = paper["expected_diffusion_score"].copy()
    year_bin = pd.to_numeric(paper["publication_year"]) // 5 * 5
    groups = pd.DataFrame(
        {"domain12": paper["domain12"].astype(str), "year_bin": year_bin},
        index=paper.index,
    )
    for indexes in groups.groupby(["domain12", "year_bin"]).groups.values():
        indexes = list(indexes)
        shuffled.loc[indexes] = generator.permutation(
            paper.loc[indexes, "expected_diffusion_score"].to_numpy()
        )
    if shuffled.equals(paper["expected_diffusion_score"]):
        shuffled.loc[:] = generator.permutation(
            paper["expected_diffusion_score"].to_numpy()
        )
    mapping = dict(zip(paper["paper_id"], shuffled, strict=True))
    return frame["paper_id"].map(mapping)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-adoption", type=Path, required=True)
    parser.add_argument("--perturbation-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    result = build_gate1_dataset(
        args.claim_adoption,
        args.perturbation_predictions,
        args.output_dir,
        seed=args.seed,
        split_manifest_path=args.split_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_gate1_dataset"]
