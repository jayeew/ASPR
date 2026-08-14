from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from gear.nature_multihorizon.artifact_store import (
    ArtifactStore,
    hash_file,
    hash_json,
)
from scripts.run_nature_multihorizon import (
    ANALYSIS_STAGES,
    DATASET_STAGES,
    DEFAULT_CONFIG,
    REUSABLE_PUBLICATION_STAGES,
    STAGE_DEPENDENCIES,
    Runtime,
    _audit_pipeline_identity,
    _figure_evidence_sources,
    _release_analysis_id,
    _validate_config,
    build_runtime,
    command_publish,
    parse_args,
)


class CliContractTests(unittest.TestCase):
    def test_figure_evidence_content_derives_new_release_analysis_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            source = assets / "source.json"
            protocol = assets / "protocol.json"
            source.write_text('{"source": 1}', encoding="utf-8")
            protocol.write_text('{"protocol": 1}', encoding="utf-8")
            table = root / "fig04_peer_review_validation.csv"
            base_row = {
                "value": 0.1,
                "source_artifact_sha256": hash_file(source),
                "protocol_hash": hash_file(protocol),
            }
            pd.DataFrame([{"evidence_id": "v1", **base_row}]).to_csv(
                table, index=False
            )
            sources = _figure_evidence_sources(
                SimpleNamespace(figure_evidence_dir=root)
            )
            first, first_hashes = _release_analysis_id("analysis-base", sources)
            self.assertTrue(first.startswith("analysis-base-ev"))
            pd.DataFrame(
                [{"evidence_id": "v2", **base_row, "value": 0.2}]
            ).to_csv(table, index=False)
            sources = _figure_evidence_sources(
                SimpleNamespace(figure_evidence_dir=root)
            )
            second, second_hashes = _release_analysis_id("analysis-base", sources)
            self.assertNotEqual(first, second)
            self.assertNotEqual(first_hashes, second_hashes)

    def test_publish_identity_rejects_model_trained_on_another_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "store")
            source_hash = hash_json("source")
            config_hash = hash_json("config")
            code_hash = hash_json("code")
            publication_hash = hash_json("publication-code")
            dirty_hash = hash_json("dirty")

            outputs: dict[tuple[str, str], str] = {}

            def build_stage(identifier: str, stage: str, inputs: tuple[str, ...]) -> None:
                with store.stage(
                    identifier,
                    stage,
                    input_artifact_ids=inputs,
                    source_snapshot_id=source_hash,
                    config_hash=config_hash,
                    code_hash=(
                        publication_hash
                        if stage in REUSABLE_PUBLICATION_STAGES
                        else code_hash
                    ),
                    dirty_diff_hash=dirty_hash,
                ) as handle:
                    handle.artifact_path("payload/value.bin").write_bytes(
                        f"{identifier}:{stage}".encode("utf-8")
                    )
                outputs[(identifier, stage)] = handle.result.manifest.output_sha256

            for dataset in ("D1", "D2"):
                for stage in DATASET_STAGES:
                    build_stage(
                        dataset,
                        stage,
                        tuple(
                            outputs[(dataset, dependency)]
                            for dependency in STAGE_DEPENDENCIES[stage]
                        ),
                    )
            for stage in ANALYSIS_STAGES:
                wrong_inputs = tuple(
                    outputs[("A", dependency)]
                    if dependency in ANALYSIS_STAGES
                    else outputs[("D1", dependency)]
                    for dependency in STAGE_DEPENDENCIES[stage]
                )
                build_stage("A", stage, wrong_inputs)

            runtime = Runtime(
                config_path=root / "config.json",
                config={},
                source_dir=root / "source",
                snapshot_dir=root / "snapshot",
                store=store,
                dataset_id="D2",
                analysis_id="A",
                source_snapshot_id=source_hash,
                config_hash=config_hash,
                code_hash=code_hash,
                publication_code_hash=publication_hash,
                dirty_diff_hash=dirty_hash,
                release_root=root / "releases",
                dataset_entry_root=root / "datasets",
            )
            with self.assertRaisesRegex(ValueError, "upstream lineage mismatch"):
                _audit_pipeline_identity(runtime)

    def test_repository_config_matches_typed_contracts(self) -> None:
        config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
        _validate_config(config)

    def test_taxonomy_drift_is_rejected(self) -> None:
        config = json.loads(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["domains"][0] = "legacy-domain"
        with self.assertRaises(ValueError):
            _validate_config(changed)

    def test_options_work_after_subcommand(self) -> None:
        args = parse_args(
            [
                "recover-v5-reference-closure",
                "--resume",
                "--dry-run",
                "--workers",
                "3",
            ]
        )
        self.assertTrue(args.resume)
        self.assertTrue(args.dry_run)
        self.assertEqual(3, args.workers)

    def test_data_affecting_future_overrides_change_dataset_id(self) -> None:
        left_args = parse_args(
            [
                "audit-source",
                "--complete-observation-year",
                "2025",
                "--max-citers-per-work",
                "1000",
            ]
        )
        right_args = parse_args(
            [
                "audit-source",
                "--complete-observation-year",
                "2024",
                "--max-citers-per-work",
                "500",
            ]
        )
        self.assertNotEqual(
            build_runtime(left_args).dataset_id,
            build_runtime(right_args).dataset_id,
        )

    def test_frozen_publish_requires_explicit_candidate_even_for_dry_run(self) -> None:
        args = parse_args(["publish-release", "--channel", "frozen", "--dry-run"])
        with self.assertRaisesRegex(ValueError, "requires --release"):
            command_publish(build_runtime(args), args)


if __name__ == "__main__":
    unittest.main()
