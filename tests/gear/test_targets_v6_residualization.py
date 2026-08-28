from __future__ import annotations

import numpy as np
import pandas as pd

from gear.nature_multihorizon.targets_v6 import (
    PERTURBATION_COMPONENTS,
    FoldLocalExcessDiffusionTransformer,
    FoldLocalPerturbationTargetTransformer,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rgpm_d_fold": [0.1, 0.2, 0.4, 0.5, 0.7, 0.9],
            "n_future_citers": [1, 2, 4, 8, 16, 32],
            "domain12": ["a", "a", "a", "b", "b", "b"],
            "publication_year": [2010, 2011, 2012, 2010, 2011, 2012],
            "opportunity_score": [0.1, 0.2, 0.2, 0.4, 0.7, 0.9],
        }
    )


def test_fold_local_target_residualization_and_no_future_leakage() -> None:
    training = _training_frame()
    transformer = FoldLocalExcessDiffusionTransformer().fit(training)
    beta = transformer.beta_.copy()
    validation = training.iloc[:2].copy()
    first = transformer.transform(validation)
    validation["rgpm_d_fold"] = [0.95, 0.05]
    second = transformer.transform(validation)

    assert np.array_equal(beta, transformer.beta_)
    assert np.allclose(
        first["expected_diffusion_null_fold"],
        second["expected_diffusion_null_fold"],
    )
    assert set(first["excess_null_fit_scope"]) == {"outer_training_fold_only"}


def test_perturbation_heads_are_preserved() -> None:
    frame = pd.DataFrame(
        {component: [0.0, 0.5, 1.0] for component in PERTURBATION_COMPONENTS}
    )
    result = FoldLocalPerturbationTargetTransformer().fit_transform(frame)
    assert all(f"{component}_fold" in result for component in PERTURBATION_COMPONENTS)
    assert "perturbation_fold" in result
