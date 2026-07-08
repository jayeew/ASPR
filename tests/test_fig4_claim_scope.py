from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.kg_perturbation_fig4.build_fig4_claim_scope import build_claim_scope_decision


class Fig4ClaimScopeTests(unittest.TestCase):
    def test_external_validation_blocked_quality_report_is_demoted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aspr_fig4_scope_") as tmp:
            fig4_dir = Path(tmp)
            (fig4_dir / "figure_quality_report.json").write_text(
                json.dumps(
                    {
                        "overall_pass": False,
                        "status_label": "external_validation_blocked",
                        "quality_gates": {
                            "checks": {
                                "fixed_sample_size_50": 1,
                                "fig3_reference_tier_range_present": 0,
                                "fig3_peer_novelty_positive": 0,
                                "fig3_peer_significance_positive": 0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            decision = build_claim_scope_decision(fig4_dir)

        self.assertEqual(1, decision["claim_scope_gate_pass"])
        self.assertEqual("extended", decision["main_or_extended_data"])
        self.assertEqual("demote_to_range_restricted_peer_review_audit", decision["claim_scope_action"])
        self.assertIn("Do not claim global external validation", decision["forbidden_claim"])


if __name__ == "__main__":
    unittest.main()
