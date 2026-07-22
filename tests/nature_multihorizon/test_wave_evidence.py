from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from experiments.kg_perturbation_v2.materialize_wave_evidence import (
    ABLATION_IDS,
    package_table,
)
from scripts.run_nature_multihorizon import _figure_evidence_sources


class WaveEvidenceTests(unittest.TestCase):
    def test_fig10_materializer_binds_results_and_protocol_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw_ablation_results.json"
            protocol = root / "locked_ablation_protocol.json"
            raw.write_text('{"result": "locked"}', encoding="utf-8")
            protocol.write_text('{"protocol": "v1"}', encoding="utf-8")
            input_table = root / "ablation_summary.csv"
            pd.DataFrame(
                [
                    {
                        "evidence_id": evidence_id,
                        "metric": "rho_global_calibrated",
                        "value": 0.4,
                        "ci_low": 0.3,
                        "ci_high": 0.5,
                        "n": 100,
                    }
                    for evidence_id in ABLATION_IDS
                ]
            ).to_csv(input_table, index=False)
            output_dir = root / "evidence"
            result = package_table(
                Namespace(
                    artifact="fig10_registered_ablations",
                    input=input_table,
                    source=[raw],
                    protocol=protocol,
                    output_dir=output_dir,
                )
            )
            self.assertTrue(Path(result["path"]).is_file())
            sources = _figure_evidence_sources(
                Namespace(figure_evidence_dir=output_dir)
            )
            self.assertIn("fig10_registered_ablations", sources)
            self.assertEqual(
                2,
                len(
                    [
                        name
                        for name in sources
                        if name.startswith("figure_evidence_asset__")
                    ]
                ),
            )
            with_extra = pd.read_csv(input_table)
            with_extra.loc[len(with_extra)] = {
                "evidence_id": "post_hoc_extra",
                "metric": "rho_global_calibrated",
                "value": 0.9,
                "ci_low": 0.8,
                "ci_high": 1.0,
                "n": 100,
            }
            extra_input = root / "ablation_with_extra.csv"
            with_extra.to_csv(extra_input, index=False)
            with self.assertRaisesRegex(ValueError, "exactly match"):
                package_table(
                    Namespace(
                        artifact="fig10_registered_ablations",
                        input=extra_input,
                        source=[raw],
                        protocol=protocol,
                        output_dir=root / "extra-evidence",
                    )
                )


if __name__ == "__main__":
    unittest.main()
