from __future__ import annotations

import math
import unittest

from pydantic import ValidationError

from gear.nature_multihorizon.contracts import (
    AUXILIARY_FEATURES,
    CORE_FEATURES,
    CohortSpec,
    FeatureSpec,
    HorizonSpec,
    MECHANISM_FEATURES,
    ScorePacket,
    SplitSpec,
    TargetSpec,
)


class ContractTests(unittest.TestCase):
    def test_default_feature_contract_has_five_mechanisms_and_eighteen_inputs(self) -> None:
        spec = FeatureSpec()

        self.assertEqual(CORE_FEATURES, spec.core_features)
        self.assertEqual(AUXILIARY_FEATURES, spec.auxiliary_features)
        self.assertEqual(5, len(spec.mechanisms))
        self.assertEqual(18, len(spec.prediction_features))
        flattened = [feature for group in spec.mechanisms.values() for feature in group]
        self.assertEqual(set(CORE_FEATURES), set(flattened))

    def test_feature_contract_rejects_unknown_or_duplicate_mechanism_members(self) -> None:
        with self.assertRaises(ValidationError):
            FeatureSpec(
                mechanisms={
                    "one": CORE_FEATURES,
                    "two": (CORE_FEATURES[0],),
                    "three": (),
                    "four": (),
                    "five": (),
                }
            )

    def test_horizon_requires_separate_development_and_holdout_periods(self) -> None:
        valid = HorizonSpec(
            tau=5,
            complete_publication_end_year=2020,
            development_end_year=2016,
            sealed_test_start_year=2017,
            sealed_test_end_year=2020,
            target_name="RGPM-D5",
        )
        self.assertEqual(10, valid.min_future_citers)

        with self.assertRaises(ValidationError):
            HorizonSpec(
                tau=5,
                complete_publication_end_year=2020,
                development_end_year=2018,
                sealed_test_start_year=2017,
                sealed_test_end_year=2020,
                target_name="RGPM-D5",
            )

    def test_target_weights_are_fixed_convex_combination(self) -> None:
        spec = TargetSpec(horizon=5, target_name="RGPM-D5")
        self.assertAlmostEqual(1.0, spec.breadth_weight + spec.evenness_weight)
        with self.assertRaises(ValidationError):
            TargetSpec(
                horizon=5,
                target_name="bad",
                breadth_weight=0.8,
                evenness_weight=0.8,
            )

    def test_cohort_and_split_defaults_lock_the_protocol(self) -> None:
        cohort = CohortSpec()
        split = SplitSpec()

        self.assertEqual((3, 5, 8), cohort.horizons)
        self.assertEqual(5, cohort.primary_horizon)
        self.assertEqual(10, cohort.min_future_citers)
        self.assertEqual(5, split.outer_folds)
        self.assertEqual(4, split.inner_folds)
        self.assertEqual((2017, 2020), split.sealed_holdout_years[5])
        self.assertEqual(2000, split.bootstrap_iterations)

    def test_score_packet_rejects_nonfinite_values_and_bad_percentile(self) -> None:
        channels = {name: 0.2 for name in MECHANISM_FEATURES}
        packet = ScorePacket(
            paper_id="W1",
            horizon=5,
            mechanism_channels=channels,
            score_mechanism=0.4,
            score_performance_raw=1.2,
            score_performance_calibrated=1.4,
            score_performance_percentile=0.91,
            model_version="model-v1",
            feature_version="feature-v1",
        )
        self.assertEqual(0.91, packet.score_performance_percentile)

        with self.assertRaises(ValidationError):
            ScorePacket(
                paper_id="W1",
                horizon=5,
                mechanism_channels=channels,
                score_mechanism=math.nan,
                model_version="model-v1",
                feature_version="feature-v1",
            )
        with self.assertRaises(ValidationError):
            ScorePacket(
                paper_id="W1",
                horizon=5,
                mechanism_channels=channels,
                score_performance_percentile=1.1,
                model_version="model-v1",
                feature_version="feature-v1",
            )


if __name__ == "__main__":
    unittest.main()
