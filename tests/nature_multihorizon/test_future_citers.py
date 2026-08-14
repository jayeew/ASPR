from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List
import unittest

import pandas as pd

from gear.nature_multihorizon.future_citers import (
    audit_prebuilt_future_multihorizon,
    fetch_future_citers,
    import_prebuilt_future_multihorizon,
    materialize_future_tables,
    merge_materialized_future_batches,
)
from gear.nature_multihorizon.targets import build_diffusion_targets_from_deltas


def _write_prebuilt_fixture(root: Path) -> None:
    status = pd.DataFrame(
        [
            {
                "paper_id": "W1",
                "requested_horizon": 8,
                "fetch_status": "success",
                "n_returned": 1.0,
                "error_type": "",
                "request_batch": "common_tau8_le2017_legacy_checkpoint_adapter",
            },
            {
                "paper_id": "W2",
                "requested_horizon": 8,
                "fetch_status": "not_requested_or_failed",
                "n_returned": float("nan"),
                "error_type": "missing_checkpoint",
                "request_batch": "common_tau8_le2017_legacy_checkpoint_adapter",
            },
        ]
    )
    requests = status[
        ["paper_id", "requested_horizon", "request_batch"]
    ].copy()
    rows = []
    for paper_id, fetch_status in (
        ("W1", "success"),
        ("W2", "not_requested_or_failed"),
    ):
        for horizon in (3, 5, 8):
            valid = paper_id == "W1"
            rows.append(
                {
                    "paper_id": paper_id,
                    "publication_year": 2010,
                    "horizon": horizon,
                    "fetch_status": fetch_status,
                    "fetch_valid": int(valid),
                    "cap_hit": 0,
                    "requested_horizon_cap_hit": 0,
                    "n_future_citers": 1.0 if valid else float("nan"),
                    "future_field_reach": 1.0 if valid else float("nan"),
                    "future_subfield_reach": 1.0 if valid else float("nan"),
                    "future_topic_reach": 1.0 if valid else float("nan"),
                    "future_field_simpson": 0.0 if valid else float("nan"),
                    "future_topic_simpson": 0.0 if valid else float("nan"),
                    "future_field_coverage": 1.0 if valid else float("nan"),
                    "future_subfield_coverage": 1.0 if valid else float("nan"),
                    "future_topic_coverage": 1.0 if valid else float("nan"),
                }
            )
    deltas = pd.DataFrame(rows)
    citers = pd.DataFrame(
        [
            {
                "paper_id": "W1",
                "horizon": horizon,
                "citer_id": "C1",
                "citer_year": 2011,
                "referenced_works": ["R1"],
            }
            for horizon in (3, 5, 8)
        ]
    )
    status.to_parquet(root / "future_fetch_status.parquet", index=False)
    requests.to_parquet(root / "future_request_manifest.parquet", index=False)
    deltas.to_parquet(root / "future_graph_deltas_multihorizon.parquet", index=False)
    citers.to_parquet(root / "future_citers.parquet", index=False)
    manifest = {
        "artifact_kind": "nature_portfolio_v5_future_multihorizon",
        "derived_horizons": [3, 5, 8],
        "n_common_tau8_papers": 2,
        "n_fetch_status_rows": 2,
        "n_future_citer_rows": 3,
        "n_future_delta_rows": 6,
        "primary_keys": {
            "future_citers": ["paper_id", "horizon", "citer_id"],
            "future_fetch_status": ["paper_id", "requested_horizon"],
            "future_request_manifest": ["paper_id", "requested_horizon"],
            "future_graph_deltas": ["paper_id", "horizon"],
        },
    }
    quality = {
        "artifact_kind": "nature_portfolio_v5_multihorizon_quality",
        "overall_pass": False,
        "actual_delta_rows": 6,
        "expected_delta_rows": 6,
        "delta_key_duplicates": 0,
        "nested_count_violations": 0,
        "missing_checkpoint_count": 1,
        "label_only_no_leakage": True,
        "diagnostics": {"invalid_rows": 0},
    }
    (root / "future_multihorizon_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "data_quality_report.json").write_text(
        json.dumps(quality), encoding="utf-8"
    )


class FutureCiterTests(unittest.TestCase):
    def test_prebuilt_false_overall_is_adopted_without_turning_failure_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            _write_prebuilt_fixture(source)
            audit = audit_prebuilt_future_multihorizon(
                source, minimum_success_rate=0.5, maximum_missing_checkpoints=1
            )
            self.assertFalse(audit["source_overall_pass"])
            self.assertTrue(audit["accepted_for_training"])
            manifest = import_prebuilt_future_multihorizon(
                source,
                output,
                minimum_success_rate=0.5,
                maximum_missing_checkpoints=1,
            )
            self.assertFalse(manifest["source_overall_pass"])
            imported = pd.read_parquet(output / "future_fetch_status.parquet")
            missing = imported.set_index("paper_id").loc["W2"]
            self.assertEqual("failed", missing["fetch_status"])
            self.assertEqual("not_requested_or_failed", missing["source_fetch_status"])
            self.assertTrue(pd.isna(missing["n_returned"]))
            self.assertEqual(
                {"common_tau8_le2017"}, set(imported["request_batch"])
            )

            papers = pd.DataFrame(
                [
                    {"paper_id": "W1", "domain12": "chemistry"},
                    {"paper_id": "W2", "domain12": "chemistry"},
                ]
            )
            targets = build_diffusion_targets_from_deltas(
                papers,
                pd.read_parquet(
                    output / "future_graph_deltas_multihorizon.parquet"
                ),
                min_future_citers=1,
            )
            tau5 = targets[targets["horizon"].eq(5)].set_index("paper_id")
            self.assertEqual(1, tau5.loc["https://openalex.org/W1", "target_rank_eligible"])
            self.assertEqual(0, tau5.loc["https://openalex.org/W2", "target_rank_eligible"])
            self.assertTrue(
                pd.isna(tau5.loc["https://openalex.org/W2", "n_future_citers"])
            )

    def test_prebuilt_missing_checkpoint_bound_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_prebuilt_fixture(root)
            with self.assertRaisesRegex(ValueError, "bounded_missing_checkpoints"):
                audit_prebuilt_future_multihorizon(
                    root, minimum_success_rate=0.5, maximum_missing_checkpoints=0
                )

    def test_failure_is_not_zero_and_horizons_are_derived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            papers = pd.DataFrame(
                [
                    {"paper_id": "W1", "publication_year": 2010},
                    {"paper_id": "W2", "publication_year": 2010},
                    {"paper_id": "W3", "publication_year": 2010},
                ]
            )

            def fetcher(paper_id: str, start: int, end: int, cap: int) -> Iterable[Dict[str, Any]]:
                del start, end, cap
                if paper_id == "W1":
                    return [
                        {
                            "id": "C1",
                            "publication_year": 2012,
                            "primary_topic": {
                                "id": "T1",
                                "display_name": "Quantum optics",
                                "subfield": {
                                    "id": "S1",
                                    "display_name": "Atomic physics",
                                },
                                "field": {
                                    "id": "F1",
                                    "display_name": "Physics and Astronomy",
                                },
                            },
                        },
                        {"id": "C2", "publication_year": 2016, "primary_field": "Physics"},
                    ]
                if paper_id == "W2":
                    return []
                raise RuntimeError("temporary API failure")

            counts = fetch_future_citers(papers, tmp_path / "checkpoints", fetcher)
            self.assertEqual(1, counts["failed"])
            self.assertEqual(1, counts["zero_success"])
            manifest = materialize_future_tables(tmp_path / "checkpoints", tmp_path / "tables")
            statuses = pd.read_parquet(manifest["future_fetch_status"])
            self.assertTrue(bool(statuses.set_index("paper_id").loc["W2", "is_zero_success"]))
            self.assertEqual("failed", statuses.set_index("paper_id").loc["W3", "fetch_status"])
            self.assertTrue(pd.isna(statuses.set_index("paper_id").loc["W3", "n_returned"]))
            citers = pd.read_parquet(manifest["future_citers"])
            self.assertEqual({3, 5, 8}, set(citers[citers["citer_id"] == "C1"]["horizon"]))
            self.assertEqual({8}, set(citers[citers["citer_id"] == "C2"]["horizon"]))
            nested = citers[citers["citer_id"] == "C1"].iloc[0]
            self.assertEqual("T1", nested["citer_primary_topic"])
            self.assertEqual("S1", nested["citer_primary_subfield"])
            self.assertEqual("F1", nested["citer_primary_field"])

    def test_common_and_recent_batches_merge_without_refetch_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            papers = pd.DataFrame(
                [
                    {"paper_id": "W2010", "publication_year": 2010},
                    {"paper_id": "W2018", "publication_year": 2018},
                    {"paper_id": "W2021", "publication_year": 2021},
                ]
            )

            def fetcher(
                paper_id: str, start: int, end: int, cap: int
            ) -> Iterable[Dict[str, Any]]:
                del cap
                return [
                    {
                        "id": f"C-{paper_id}",
                        "publication_year": min(start + 1, end),
                        "primary_topic": {
                            "id": "T1",
                            "subfield": {"id": "S1"},
                            "field": {"id": "F1"},
                        },
                    }
                ]

            specifications = (
                ("common", 8, None, 2017, (3, 5, 8)),
                ("recent5", 5, 2018, 2020, (3, 5)),
                ("recent3", 3, 2021, 2022, (3,)),
            )
            outputs = []
            for name, horizon, minimum, maximum, derived in specifications:
                checkpoint = root / "checkpoints" / name
                output = root / "batches" / name
                fetch_future_citers(
                    papers,
                    checkpoint,
                    fetcher,
                    requested_horizon=horizon,
                    min_publication_year=minimum,
                    max_publication_year=maximum,
                    request_batch=name,
                )
                materialize_future_tables(
                    checkpoint,
                    output,
                    requested_horizon=horizon,
                    derived_horizons=derived,
                )
                outputs.append(output)
            manifest = merge_materialized_future_batches(outputs, root / "merged")
            requests = pd.read_parquet(manifest["future_request_manifest"])
            self.assertEqual(
                {"W2010": 8, "W2018": 5, "W2021": 3},
                requests.set_index("paper_id")["requested_horizon"].to_dict(),
            )
            citers = pd.read_parquet(manifest["future_citers"])
            self.assertEqual({3, 5, 8}, set(citers[citers["paper_id"] == "W2010"]["horizon"]))
            self.assertEqual({3, 5}, set(citers[citers["paper_id"] == "W2018"]["horizon"]))
            self.assertEqual({3}, set(citers[citers["paper_id"] == "W2021"]["horizon"]))


if __name__ == "__main__":
    unittest.main()
