from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import EvidenceProtocol, ProtocolError, normalize_text, sha256_text
from providers import OpenAlexClient

EVIDENCE_HASH = "a" * 64


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(str(self.status_code))

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.params: list[dict] = []

    def get(self, _url: str, **kwargs: object) -> FakeResponse:
        self.params.append(dict(kwargs["params"]))
        return self.responses.pop(0)


class ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.engine = EvidenceProtocol(
            self.root / "evidence.sqlite3", self.root / "outputs"
        )
        self.engine.initialize()

    def tearDown(self) -> None:
        self.engine.close()
        self.temporary.cleanup()

    def _search_fixture(self, press: str = "pass", hidden: bool = True) -> None:
        connection = self.engine.connection
        connection.execute(
            "INSERT INTO search_domains VALUES(?,?,?,?,?,?,?,?)",
            (
                "D1",
                "Novelty",
                "Novelty evidence",
                '["T1"]',
                '["W1"]',
                "include",
                "include",
                "active",
            ),
        )
        connection.execute(
            "INSERT INTO logical_queries VALUES(?,?,?,?,?,?,?,?)",
            (
                "Q1",
                "D1",
                "novelty AND paper",
                "novelty AND paper",
                '["W1"]',
                press,
                "active",
                0,
            ),
        )
        connection.execute(
            "INSERT INTO physical_queries VALUES(?,?,?,?,?,?)",
            ("P1", "Q1", "OpenAlex", "search=novelty", "provider_syntax", 1),
        )
        seeds = [("S1", "development", "W1", "indexable", "recalled", "", '["Q1"]')]
        if hidden:
            seeds.append(("S2", "hidden", "W2", "indexable", "recalled", "", '["Q1"]'))
        connection.executemany("INSERT INTO seed_recall VALUES(?,?,?,?,?,?,?)", seeds)
        connection.commit()

    def _selection_fixture(self) -> None:
        connection = self.engine.connection
        connection.executemany(
            "INSERT INTO works VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    f"W{number}",
                    "",
                    f"OA{number}",
                    f"Paper {number}",
                    f"paper {number}",
                    2020,
                    "en",
                    "article",
                    "",
                    "formal",
                    f"hash{number}",
                )
                for number in range(1, 4)
            ],
        )
        connection.executemany(
            "INSERT INTO screening_final VALUES(?,?,?,?,?)",
            [
                (f"W{number}", "include", "", "adjudicated", "R1")
                for number in range(1, 4)
            ],
        )
        connection.executemany(
            "INSERT INTO construct_mentions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    "C1",
                    "W1",
                    "recombination",
                    "predictive",
                    "references",
                    "T0",
                    "low",
                    "all",
                    '["M1"]',
                    "A",
                    "evidence",
                ),
                (
                    "C2",
                    "W2",
                    "team size",
                    "control",
                    "metadata",
                    "T0",
                    "low",
                    "all",
                    '["M2"]',
                    "A",
                    "evidence",
                ),
            ],
        )
        dimension_rows = [
            (
                "D1",
                "Novel recombination",
                "Combines prior knowledge",
                "predictive",
                "T0",
                '["W1"]',
                '["A","B"]',
                '[{"decision":"distinct construct"}]',
                1,
                1,
                1,
            ),
            (
                "D2",
                "Team size",
                "Number of authors",
                "control",
                "T0",
                '["W2"]',
                '["A"]',
                '[{"decision":"distinct control"}]',
                1,
                1,
                1,
            ),
        ]
        connection.executemany(
            "INSERT INTO candidate_dimensions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            dimension_rows,
        )
        self.engine.set_metadata(
            "outcome_blind_audit",
            {"status": "pass", "outcome_columns_used": False, "input_hash": "audit"},
        )
        families = [
            (
                "F1",
                "atypicality",
                '["novel combination"]',
                '["D1"]',
                '["M1"]',
                "z score",
                "z",
                '["W1"]',
                '["A","B"]',
                "predictive",
                "T0",
                "missing",
                "missing",
                "missing",
                "minimum 1",
                "missing",
                "candidate",
            ),
            (
                "F2",
                "author_count",
                '["team size"]',
                '["D2"]',
                '["M2"]',
                "count authors",
                "n",
                '["W2"]',
                '["A"]',
                "control",
                "T0",
                "missing",
                "missing",
                "zero",
                "complete",
                "missing",
                "candidate",
            ),
            (
                "F3",
                "future_citations",
                "[]",
                '["D1"]',
                '["M3"]',
                "future citations",
                "n",
                '["W3"]',
                '["A"]',
                "outcome",
                "T+5",
                "missing",
                "missing",
                "zero",
                "complete",
                "missing",
                "candidate",
            ),
        ]
        connection.executemany(
            "INSERT INTO indicator_families VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            families,
        )
        connection.executemany(
            "INSERT INTO indicator_mentions VALUES(?,?,?,?,?,?)",
            [
                ("M1", "W1", "D1", "atypicality", "definition", "original_definition"),
                (
                    "M2",
                    "W2",
                    "D2",
                    "author count",
                    "definition",
                    "original_application",
                ),
                ("M3", "W3", "D1", "future citations", "definition", "outcome"),
            ],
        )
        connection.executemany(
            "INSERT INTO indicator_evidence VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (
                    f"E{number}",
                    f"F{number}",
                    f"W{number}",
                    "definition",
                    "quote",
                    "p1",
                    f"source{number}",
                    1,
                    "A",
                )
                for number in range(1, 4)
            ],
        )
        mappings = [
            (
                "F1",
                "derivable",
                '["refs"]',
                "deterministic SQL",
                "hash",
                1.0,
                0.0,
                20,
                0,
                "pass",
            ),
            ("F2", "direct", '["authors"]', "count", "hash", 1.0, 0.0, 20, 0, "pass"),
            ("F3", "direct", '["future"]', "count", "hash", 1.0, 0.0, 20, 0, "pass"),
        ]
        connection.executemany(
            "INSERT INTO indicator_data_mapping VALUES(?,?,?,?,?,?,?,?,?,?)", mappings
        )
        gates = [
            ("F1", 1, 1, 1, 1, 1, 1, "ok", "ok", "{}", 1),
            ("F2", 1, 1, 1, 1, 1, 1, "ok", "ok", "{}", 1),
            ("F3", 1, 0, 1, 1, 1, 1, "future", "future", "{}", 0),
        ]
        connection.executemany(
            "INSERT INTO hard_gate_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?)", gates
        )
        connection.executemany(
            "INSERT INTO evidence_tiers VALUES(?,?,?,?)",
            [
                ("F1", "A", "two teams", 1),
                ("F2", "B", "one team", 1),
                ("F3", "C", "future", 1),
            ],
        )
        connection.commit()

    def test_unicode_case_punctuation_normalization(self) -> None:
        self.assertEqual(normalize_text("Novelty—METRICS"), "novelty metrics")

    def test_three_level_work_deduplication(self) -> None:
        first, created = self.engine.ingest_work(
            {"doi": "10.1/X", "title": "A Novel Paper", "publication_year": 2020}
        )
        self.assertTrue(created)
        self.assertEqual(
            self.engine.ingest_work(
                {
                    "doi": "https://doi.org/10.1/x",
                    "title": "Changed",
                    "publication_year": 2020,
                }
            ),
            (first, False),
        )
        self.assertEqual(
            self.engine.ingest_work(
                {"title": "A novel paper!", "publication_year": 2020}
            ),
            (first, False),
        )

    def test_openalex_429_retry_and_key_rotation(self) -> None:
        session = FakeSession(
            [
                FakeResponse(429),
                FakeResponse(
                    200, {"results": [{"id": "W1"}], "meta": {"next_cursor": "next"}}
                ),
            ]
        )
        with patch.dict(
            "os.environ",
            {"OPENALEX_API_KEY_A": "secret-a", "OPENALEX_API_KEY_B": "secret-b"},
        ):
            client = OpenAlexClient(
                session=session, sleep=lambda _: None, minimum_interval=0
            )
            page = client.fetch_page("title.search:novelty")
        self.assertEqual(page.key_slot, "B")
        self.assertEqual(page.next_cursor, "next")
        self.assertEqual(session.params[0]["api_key"], "secret-a")
        self.assertEqual(session.params[1]["api_key"], "secret-b")
        self.assertNotIn("secret", repr(page))

    def test_openalex_search_page_keeps_search_separate_from_filter(self) -> None:
        session = FakeSession(
            [FakeResponse(200, {"results": [], "meta": {"next_cursor": ""}})]
        )
        client = OpenAlexClient(
            session=session, sleep=lambda _: None, minimum_interval=0
        )

        client.fetch_search_page(
            'paper AND "citation impact"',
            "to_publication_date:2026-07-28,language:en",
        )

        self.assertEqual(session.params[0]["search"], 'paper AND "citation impact"')
        self.assertEqual(
            session.params[0]["filter"],
            "to_publication_date:2026-07-28,language:en",
        )

    def test_openalex_http_error_does_not_expose_request_or_key(self) -> None:
        session = FakeSession([FakeResponse(400)])
        with patch.dict("os.environ", {"OPENALEX_API_KEY_A": "secret-a"}):
            client = OpenAlexClient(
                session=session, sleep=lambda _: None, minimum_interval=0
            )
            with self.assertRaises(ProtocolError) as raised:
                client.fetch_search_page("bad query", "language:en")

        message = str(raised.exception)
        self.assertIn("HTTP_400", message)
        self.assertNotIn("secret-a", message)
        self.assertNotIn("api.openalex.org", message)

    def test_first_reviewed_zero_zero_round_stops(self) -> None:
        basis = self.engine.record_saturation_round(1, 0, 0, True, EVIDENCE_HASH)
        self.assertEqual(basis, "strict_zero_zero")
        with self.assertRaises(ProtocolError):
            self.engine.record_saturation_round(2, 1, 0, True, EVIDENCE_HASH)

    def test_round_15_stops_without_claiming_saturation(self) -> None:
        for round_no in range(1, 15):
            self.engine.record_saturation_round(round_no, 1, 0, True, EVIDENCE_HASH)
        basis = self.engine.record_saturation_round(15, 2, 1, True, EVIDENCE_HASH)
        self.assertEqual(basis, "maximum_round_15")

    def test_round_15_cannot_skip_prior_rounds_or_overwrite(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "consecutively"):
            self.engine.record_saturation_round(15, 1, 1, True, EVIDENCE_HASH)
        self.engine.record_saturation_round(1, 1, 1, True, EVIDENCE_HASH)
        with self.assertRaisesRegex(ProtocolError, "immutable"):
            self.engine.record_saturation_round(1, 0, 0, True, "b" * 64)

    def test_press_and_seed_recall_are_freeze_gates(self) -> None:
        self._search_fixture(press="pending")
        self.engine.record_saturation_round(1, 0, 0, True, EVIDENCE_HASH)
        with self.assertRaisesRegex(ProtocolError, "PRESS"):
            self.engine.freeze_search()

    def test_hidden_seed_cohort_is_required(self) -> None:
        self._search_fixture(hidden=False)
        self.engine.record_saturation_round(1, 0, 0, True, EVIDENCE_HASH)
        with self.assertRaisesRegex(ProtocolError, "hidden"):
            self.engine.freeze_search()

    def test_physical_splits_do_not_increase_q(self) -> None:
        self._search_fixture()
        self.engine.connection.execute(
            "INSERT INTO physical_queries VALUES(?,?,?,?,?,?)",
            ("P2", "Q1", "OpenAlex", "search=novelty&page=2", "URL length", 1),
        )
        counts = self.engine._search_counts()
        self.assertEqual(counts, {"K": 1, "Q": 1, "P": 2})

    def test_api_failure_retry_state_contains_no_secret(self) -> None:
        self.engine.connection.execute(
            "INSERT INTO search_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "R1",
                "saturate",
                "Q1",
                "OpenAlex",
                "A",
                "cursor",
                "retryable",
                2,
                "",
                "HTTP_429",
                "now",
                "",
            ),
        )
        row = self.engine.connection.execute("SELECT * FROM search_runs").fetchone()
        self.assertEqual(row["key_slot"], "A")
        self.assertEqual(row["cursor"], "cursor")
        self.assertEqual(row["error_code"], "HTTP_429")

    def test_non_english_requires_fixed_exclusion(self) -> None:
        connection = self.engine.connection
        connection.execute(
            "INSERT INTO works VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            ("W1", "", "", "标题", "标题", 2020, "zh", "article", "", "formal", "hash"),
        )
        connection.execute(
            "INSERT INTO screening_final VALUES(?,?,?,?,?)",
            ("W1", "include", "", "wrong", "R1"),
        )
        connection.commit()
        with self.assertRaisesRegex(ProtocolError, "Non-English"):
            self.engine.finalize_screening()

    def test_uncertain_is_not_a_final_screening_disposition(self) -> None:
        connection = self.engine.connection
        connection.execute(
            "INSERT INTO works VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                "W1",
                "",
                "",
                "Paper",
                "paper",
                2020,
                "en",
                "article",
                "",
                "formal",
                "hash",
            ),
        )
        connection.execute(
            "INSERT INTO screening_final VALUES(?,?,?,?,?)",
            ("W1", "uncertain", "", "unresolved", "R1"),
        )
        connection.commit()
        with self.assertRaisesRegex(ProtocolError, "include or exclude"):
            self.engine.finalize_screening()

    def test_legacy_inventory_never_imports_decisions(self) -> None:
        legacy = self.root / "legacy.json"
        legacy.write_text('{"decision":"include"}', encoding="utf-8")
        inventory = self.engine.register_legacy_inventory([legacy])
        self.assertEqual(
            inventory[str(legacy.resolve())],
            sha256_text(legacy.read_text().encode("utf-8").decode()),
        )
        self.assertFalse(self.engine.get_metadata("legacy_decisions_imported"))

    def test_data_mapping_vocabulary_is_closed(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.engine.connection.execute(
                "INSERT INTO indicator_data_mapping VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("X", "proxy", "[]", "", "", None, None, None, 0, "pending"),
            )

    def test_supported_and_strict_dimension_rules(self) -> None:
        self._selection_fixture()
        sets = self.engine.select_features()
        self.assertEqual(sets["model"], ["F1", "F2"])
        self.assertEqual(sets["strict_training"], ["F1"])
        self.assertEqual(sets["primary"], ["F1", "F2"])
        self.assertNotIn("F3", sets["broad_t0"])
        supported = self.engine.connection.execute(
            "SELECT dimension_id FROM final_dimensions WHERE set_name='supported' ORDER BY dimension_id"
        ).fetchall()
        strict = self.engine.connection.execute(
            "SELECT dimension_id FROM final_dimensions WHERE set_name='strict'"
        ).fetchall()
        self.assertEqual([row[0] for row in supported], ["D1", "D2"])
        self.assertEqual([row[0] for row in strict], ["D1"])

    def test_missing_independent_tier_approval_fails_closed(self) -> None:
        self._selection_fixture()
        self.engine.connection.execute(
            "UPDATE evidence_tiers SET independent_approved=0 WHERE indicator_id='F1'"
        )
        sets = self.engine.select_features()
        self.assertNotIn("F1", sets["strict"])
        self.assertNotIn("F1", sets["primary"])
        self.assertNotIn("F1", sets["expanded"])

    def test_sensitivity_role_is_excluded_from_primary(self) -> None:
        self._selection_fixture()
        self.engine.connection.execute(
            "UPDATE indicator_families SET role='sensitivity' WHERE indicator_id='F1'"
        )
        sets = self.engine.select_features()
        self.assertIn("F1", sets["strict"])
        self.assertIn("F1", sets["expanded"])
        self.assertNotIn("F1", sets["strict_training"])
        self.assertNotIn("F1", sets["primary"])

    def test_unavailable_mapping_fails_model_and_broad_sets(self) -> None:
        self._selection_fixture()
        self.engine.connection.execute(
            "UPDATE indicator_data_mapping SET mapping_type='unavailable' WHERE indicator_id='F2'"
        )
        sets = self.engine.select_features()
        self.assertNotIn("F2", sets["model"])
        self.assertNotIn("F2", sets["broad_t0"])

    def test_external_source_family_without_formal_mention_remains_in_census(self) -> None:
        self._selection_fixture()
        self.engine.connection.execute(
            "UPDATE indicator_families SET mention_ids_json='[]', "
            "definition_source_ids_json='[\"doi:external-definition\"]' "
            "WHERE indicator_id='F1'"
        )
        self.assertEqual(self.engine.validate_indicator_census(), 3)

    def test_audit_hash_covers_non_exported_evidence_tables(self) -> None:
        first = self.engine.audit()["deterministic_hash"]
        self.engine.connection.execute(
            "INSERT INTO citations VALUES(?,?,?)", ("W1", "W2", "backward")
        )
        self.engine.connection.commit()
        second = self.engine.audit()["deterministic_hash"]
        self.assertNotEqual(first, second)

    def test_near_constant_must_fail_data_integrity_gate(self) -> None:
        self._selection_fixture()
        self.engine.connection.execute(
            "UPDATE indicator_data_mapping SET near_constant=1 WHERE indicator_id='F2'"
        )
        self.assertNotIn("F2", self.engine.select_features()["model"])
        gate = self.engine.connection.execute(
            "SELECT h6_data_integrity,all_pass FROM hard_gate_decisions WHERE indicator_id='F2'"
        ).fetchone()
        self.assertEqual(tuple(gate), (0, 0))

    def test_synonym_and_parameter_variants_share_one_family(self) -> None:
        self._selection_fixture()
        with self.assertRaises(sqlite3.IntegrityError):
            self.engine.connection.execute(
                "INSERT INTO indicator_families VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "F4",
                    "atypicality",
                    '["z-score atypicality"]',
                    '["D1"]',
                    '["M4"]',
                    "variant",
                    "z",
                    '["W4"]',
                    '["C"]',
                    "predictive",
                    "T0",
                    "missing",
                    "missing",
                    "missing",
                    "minimum 2",
                    "missing",
                    "candidate",
                ),
            )

    def test_four_set_freeze_is_deterministic_and_outcome_blind(self) -> None:
        self._selection_fixture()
        first = self.engine.select_features()
        first_hash = self.engine.get_metadata("feature_set_freeze_hash")
        second = self.engine.select_features()
        self.assertEqual(first, second)
        self.assertEqual(
            first_hash, self.engine.get_metadata("feature_set_freeze_hash")
        )
        payload = json.loads(
            (self.root / "outputs" / "final_feature_sets.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(payload["outcome_columns_used"])


if __name__ == "__main__":
    unittest.main()
