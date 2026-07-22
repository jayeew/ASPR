from __future__ import annotations

import unittest

import pandas as pd

from aspr.nature_multihorizon.quality import audit_pipeline_tables


class QualityGateTests(unittest.TestCase):
    def test_api_failure_is_not_accepted_as_zero(self) -> None:
        report = audit_pipeline_tables(
            {
                "future_fetch_status": pd.DataFrame(
                    [{"paper_id": "W1", "fetch_status": "failed", "n_returned": 0}]
                )
            }
        )
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual("fail", checks["future_failures_are_na"]["status"])
        self.assertFalse(report["go_for_training"])

    def test_status_coverage_uses_expected_request_antijoin(self) -> None:
        report = audit_pipeline_tables(
            {
                "future_request_manifest": pd.DataFrame(
                    [
                        {"paper_id": "W1", "requested_horizon": 8},
                        {"paper_id": "W2", "requested_horizon": 8},
                    ]
                ),
                "future_fetch_status": pd.DataFrame(
                    [
                        {
                            "paper_id": "W1",
                            "requested_horizon": 8,
                            "fetch_status": "success",
                            "n_returned": 0,
                        }
                    ]
                ),
            }
        )
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual(0.5, checks["future_status_coverage"]["value"])
        self.assertEqual("fail", checks["future_status_coverage"]["status"])

    def test_legacy_columns_are_rejected(self) -> None:
        report = audit_pipeline_tables({"features_raw": pd.DataFrame({"B_z": [0.0]})})
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual("fail", checks["no_legacy_columns"]["status"])

    def test_expanded_future_coverage_requires_recent_short_windows(self) -> None:
        common = pd.DataFrame(
            [
                {
                    "paper_id": f"W{year}",
                    "publication_year": year,
                    "requested_horizon": 8,
                }
                for year in range(2010, 2018)
            ]
        )
        common_report = audit_pipeline_tables(
            {"future_request_manifest": common}
        )
        common_check = {
            item["name"]: item for item in common_report["checks"]
        }["expanded_future_horizon_coverage"]
        self.assertEqual("fail", common_check["status"])

        expanded = pd.concat(
            [
                common,
                pd.DataFrame(
                    [
                        {
                            "paper_id": f"W{year}",
                            "publication_year": year,
                            "requested_horizon": 5,
                        }
                        for year in range(2018, 2021)
                    ]
                ),
                pd.DataFrame(
                    [
                        {
                            "paper_id": f"W{year}",
                            "publication_year": year,
                            "requested_horizon": 3,
                        }
                        for year in range(2021, 2023)
                    ]
                ),
            ],
            ignore_index=True,
        )
        expanded_report = audit_pipeline_tables(
            {"future_request_manifest": expanded}
        )
        expanded_check = {
            item["name"]: item for item in expanded_report["checks"]
        }["expanded_future_horizon_coverage"]
        self.assertEqual("pass", expanded_check["status"])

    def test_future_success_gate_is_enforced_per_request_batch(self) -> None:
        requests = pd.DataFrame(
            [
                {
                    "paper_id": f"C{index}",
                    "requested_horizon": 8,
                    "request_batch": "common_tau8_le2017",
                }
                for index in range(1_000)
            ]
            + [
                {
                    "paper_id": f"R{index}",
                    "requested_horizon": 3,
                    "request_batch": "recent_tau3_2021_2022",
                }
                for index in range(10)
            ]
        )
        status = requests[["paper_id", "requested_horizon"]].copy()
        status["fetch_status"] = "success"
        status["n_returned"] = 10.0
        recent_failure = status["paper_id"].isin(["R8", "R9"])
        status.loc[recent_failure, "fetch_status"] = "failed"
        status.loc[recent_failure, "n_returned"] = pd.NA
        report = audit_pipeline_tables(
            {
                "future_request_manifest": requests,
                "future_fetch_status": status,
            }
        )
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual("pass", checks["future_fetch_success"]["status"])
        self.assertEqual(
            "fail",
            checks[
                "future_fetch_success_batch:recent_tau3_2021_2022"
            ]["status"],
        )

    def test_target_validity_gate_is_enforced_in_sealed_period(self) -> None:
        targets = pd.DataFrame(
            [
                {
                    "paper_id": f"D{index}",
                    "horizon": 3,
                    "publication_year": 2010,
                    "fetch_valid": True,
                    "target_valid": True,
                    "n_future_citers": 10,
                }
                for index in range(1_000)
            ]
            + [
                {
                    "paper_id": f"S{index}",
                    "horizon": 3,
                    "publication_year": 2019,
                    "fetch_valid": True,
                    "target_valid": index < 5,
                    "n_future_citers": 10,
                }
                for index in range(10)
            ]
        )
        report = audit_pipeline_tables({"targets": targets})
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual(
            "pass", checks["future_taxonomy_target_valid_coverage"]["status"]
        )
        self.assertEqual(
            "fail", checks["sealed_target_valid_coverage_tau3"]["status"]
        )

    def test_structural_gate_requires_one_row_with_positive_ci(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "horizon": 5,
                    "model_id": "nested_selector",
                    "scope": "structural_validation_subset",
                    "metric": "rho_rgpm_s5",
                    "sensitivity": "main",
                    "value": 0.2,
                    "ci_low": -0.1,
                },
                {
                    "horizon": 5,
                    "model_id": "nested_selector",
                    "scope": "structural_validation_subset",
                    "metric": "rho_rgpm_s5",
                    "sensitivity": "main",
                    "value": -0.1,
                    "ci_low": 0.1,
                },
            ]
        )
        report = audit_pipeline_tables({"evaluation_metrics": metrics})
        check = {item["name"]: item for item in report["checks"]}[
            "tau5_structural_validation"
        ]
        self.assertEqual("fail", check["status"])

    def test_cap_hit_rate_and_uncapped_oof_are_frozen_gates(self) -> None:
        cohorts = pd.DataFrame(
            [
                {
                    "paper_id": f"W{index}",
                    "horizon": 5,
                    "cohort_member": 1,
                    "cap_hit": int(index < 3),
                    "domain12": "chemistry",
                }
                for index in range(100)
            ]
        )
        metrics = pd.DataFrame(
            [
                {
                    "horizon": 5,
                    "model_id": "nested_selector",
                    "scope": "development_oof",
                    "metric": "rho_global_calibrated",
                    "sensitivity": "main",
                    "value": 0.50,
                },
                {
                    "horizon": 5,
                    "model_id": "nested_selector",
                    "scope": "sensitivity_uncapped_future_citers",
                    "metric": "rho_global_calibrated",
                    "sensitivity": "uncapped_cohort_member",
                    "value": 0.49,
                },
            ]
        )
        report = audit_pipeline_tables(
            {"cohort_membership": cohorts, "evaluation_metrics": metrics}
        )
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual("fail", checks["cap_hit_rate_tau5"]["status"])
        self.assertEqual(
            "pass",
            checks["tau5_uncapped_future_citer_sensitivity"]["status"],
        )


if __name__ == "__main__":
    unittest.main()
