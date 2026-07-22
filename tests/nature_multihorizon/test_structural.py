from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import pandas as pd

from aspr.nature_multihorizon.structural import (
    annotate_future_reference_coverage,
    annotate_future_reference_coverage_from_parquet,
    read_future_citers_for_subset,
)


class StructuralTests(unittest.TestCase):
    def test_large_future_table_is_filtered_while_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.parquet"
            pd.DataFrame(
                [
                    {
                        "paper_id": "W1",
                        "horizon": 5,
                        "citer_id": "C1",
                        "citer_year": 2012,
                        "referenced_works": ["R1"],
                    },
                    {
                        "paper_id": "W1",
                        "horizon": 5,
                        "citer_id": "C2",
                        "citer_year": 2013,
                        "referenced_works": [],
                    },
                    {
                        "paper_id": "W2",
                        "horizon": 5,
                        "citer_id": "C3",
                        "citer_year": 2014,
                        "referenced_works": ["R2"],
                    },
                ]
            ).to_parquet(path, index=False)
            membership = pd.DataFrame([{"paper_id": "W1", "horizon": 5}])
            annotated = annotate_future_reference_coverage_from_parquet(
                membership, path, batch_size=1
            )
            self.assertEqual(0.5, annotated.loc[0, "future_citer_reference_coverage"])
            selected = read_future_citers_for_subset(path, membership, batch_size=1)
            self.assertEqual({"C1", "C2"}, set(selected["citer_id"]))

    def test_future_reference_coverage_is_computed_per_horizon(self) -> None:
        membership = pd.DataFrame(
            [{"paper_id": "W1", "horizon": 5}, {"paper_id": "W1", "horizon": 8}]
        )
        citers = pd.DataFrame(
            [
                {"paper_id": "W1", "horizon": 5, "citer_id": "C1", "referenced_works": ["R1"]},
                {"paper_id": "W1", "horizon": 5, "citer_id": "C2", "referenced_works": []},
                {"paper_id": "W1", "horizon": 8, "citer_id": "C1", "referenced_works": ["R1"]},
            ]
        )
        output = annotate_future_reference_coverage(membership, citers)
        coverage = output.set_index("horizon")["future_citer_reference_coverage"]
        self.assertEqual(0.5, coverage.loc[5])
        self.assertEqual(1.0, coverage.loc[8])


if __name__ == "__main__":
    unittest.main()
