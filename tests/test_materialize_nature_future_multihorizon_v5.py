from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_nature_future_multihorizon_v5 import materialize_multihorizon
from aspr.nature_multihorizon.targets import build_diffusion_targets


def _work(citer_id: str, year: int, topic_index: int) -> dict:
    return {
        "id": f"https://openalex.org/{citer_id}",
        "publication_year": year,
        "primary_topic": {
            "id": f"https://openalex.org/T{topic_index}",
            "subfield": {"id": f"https://openalex.org/S{topic_index}"},
            "field": {"id": f"https://openalex.org/F{topic_index % 2}"},
        },
        "referenced_works": [f"https://openalex.org/R{topic_index}"],
    }


def _write_checkpoint(path: Path, paper_id: str, works: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for work in works:
            handle.write(
                json.dumps(
                    {
                        "paper_id": f"https://openalex.org/{paper_id}",
                        "work": work,
                    }
                )
                + "\n"
            )


def _args(root: Path, *, max_citers: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        target_works=root / "nature_target_works.csv",
        checkpoint_dir=root / "checkpoints/future_citers_tau8",
        output_dir=root / "future_multihorizon",
        horizons=[3, 5, 8],
        requested_horizon=8,
        complete_end_year=2025,
        max_citers_per_work=max_citers,
        batch_size=2,
        progress_every=1000,
        allow_missing=False,
        allow_quality_failures=False,
        overwrite=False,
        quiet=True,
    )


class MultiHorizonMaterializationTests(unittest.TestCase):
    def test_derives_nested_windows_and_compatibility_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [
                    {"id": "https://openalex.org/W1", "short_id": "W1", "year": 2010},
                    {"id": "https://openalex.org/W2", "short_id": "W2", "year": 2010},
                    {"id": "https://openalex.org/W3", "short_id": "W3", "year": 2018},
                ]
            ).to_csv(root / "nature_target_works.csv", index=False)
            checkpoint_dir = root / "checkpoints/future_citers_tau8"
            _write_checkpoint(
                checkpoint_dir / "W1.jsonl",
                "W1",
                [
                    _work("C1", 2012, 1),
                    _work("C2", 2014, 2),
                    _work("C3", 2017, 3),
                ],
            )
            _write_checkpoint(checkpoint_dir / "W2.jsonl", "W2", [])

            manifest = materialize_multihorizon(_args(root))
            output = root / "future_multihorizon"
            self.assertEqual(2, manifest["n_common_tau8_papers"])
            self.assertEqual(6, manifest["n_future_delta_rows"])
            self.assertTrue(manifest["quality_overall_pass"])

            citers = pd.read_parquet(output / "future_citers.parquet")
            self.assertEqual(
                {3, 5, 8}, set(citers[citers["citer_id"].str.endswith("C1")]["horizon"])
            )
            self.assertEqual(
                {5, 8}, set(citers[citers["citer_id"].str.endswith("C2")]["horizon"])
            )
            self.assertEqual(
                {8}, set(citers[citers["citer_id"].str.endswith("C3")]["horizon"])
            )
            self.assertEqual(6, len(citers))

            status = pd.read_parquet(output / "future_fetch_status.parquet").set_index(
                "paper_id"
            )
            self.assertEqual("success", status.loc["https://openalex.org/W2", "fetch_status"])
            self.assertEqual(1, status.loc["https://openalex.org/W2", "is_zero_success"])

            deltas = pd.read_parquet(
                output / "future_graph_deltas_multihorizon.parquet"
            )
            w1 = deltas[deltas["paper_id"].eq("https://openalex.org/W1")].set_index(
                "horizon"
            )
            self.assertEqual([1, 2, 3], w1.loc[[3, 5, 8], "n_future_citers"].tolist())
            self.assertEqual([0, 0, 1], w1.loc[[3, 5, 8], "cap_hit"].tolist())
            w2 = deltas[deltas["paper_id"].eq("https://openalex.org/W2")]
            self.assertEqual([0, 0, 0], w2["n_future_citers"].tolist())

            for horizon in (3, 5, 8):
                compatibility = (
                    output
                    / f"horizons/tau{horizon}/nature_future_graph_deltas.csv"
                )
                self.assertTrue(compatibility.is_file())
                frame = pd.read_csv(compatibility)
                self.assertEqual(2, len(frame))
                self.assertTrue(frame["tau"].eq(horizon).all())
            downstream = (output / "downstream_paths.env").read_text(encoding="utf-8")
            self.assertNotIn(".tmp-", downstream)
            quality = json.loads(
                (output / "data_quality_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(0, quality["nested_count_violations"])

            downstream_papers = pd.DataFrame(
                [
                    {
                        "paper_id": "https://openalex.org/W1",
                        "publication_year": 2010,
                        "domain12": "chemistry",
                    },
                    {
                        "paper_id": "https://openalex.org/W2",
                        "publication_year": 2010,
                        "domain12": "chemistry",
                    },
                ]
            )
            targets = build_diffusion_targets(
                downstream_papers,
                citers,
                status.reset_index(),
                horizons=(3, 5, 8),
                min_future_citers=1,
                min_taxonomy_coverage=0.0,
            )
            self.assertEqual(6, len(targets))
            self.assertEqual({"RGPM-D3", "RGPM-D5", "RGPM-D8"}, set(targets["target_name"]))

    def test_missing_checkpoint_blocks_atomic_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [{"id": "https://openalex.org/W1", "short_id": "W1", "year": 2010}]
            ).to_csv(root / "nature_target_works.csv", index=False)
            (root / "checkpoints/future_citers_tau8").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "Missing 1 eligible checkpoints"):
                materialize_multihorizon(_args(root))
            self.assertFalse((root / "future_multihorizon").exists())

    def test_explicit_missing_mode_preserves_rows_and_marks_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame(
                [{"id": "https://openalex.org/W1", "short_id": "W1", "year": 2010}]
            ).to_csv(root / "nature_target_works.csv", index=False)
            (root / "checkpoints/future_citers_tau8").mkdir(parents=True)
            args = _args(root)
            args.allow_missing = True
            args.allow_quality_failures = True
            manifest = materialize_multihorizon(args)
            self.assertFalse(manifest["quality_overall_pass"])
            deltas = pd.read_parquet(
                root
                / "future_multihorizon/future_graph_deltas_multihorizon.parquet"
            )
            self.assertEqual(3, len(deltas))
            self.assertTrue(deltas["fetch_status"].eq("not_requested_or_failed").all())
            self.assertTrue(deltas["n_future_citers"].isna().all())


if __name__ == "__main__":
    unittest.main()
