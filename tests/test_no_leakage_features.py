from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.nature_ready_checks import detect_no_leakage_feature_violations  # noqa: E402


class NoLeakageFeatureTests(unittest.TestCase):
    def test_future_or_outcome_columns_are_blocked_from_publication_day_features(self) -> None:
        frame = pd.DataFrame(
            {
                "B_z": [0.1],
                "reference_count": [12],
                "future_citations": [99],
                "RGPM": [1.2],
                "venue_future_impact": [0.4],
                "cited_by_count": [300],
                "n_future_citers": [12],
            }
        )

        violations = detect_no_leakage_feature_violations(frame, feature_columns=list(frame.columns))

        self.assertIn("future_citations", violations)
        self.assertIn("RGPM", violations)
        self.assertIn("venue_future_impact", violations)
        self.assertIn("cited_by_count", violations)
        self.assertIn("n_future_citers", violations)
        self.assertNotIn("B_z", violations)
        self.assertNotIn("reference_count", violations)


if __name__ == "__main__":
    unittest.main()
