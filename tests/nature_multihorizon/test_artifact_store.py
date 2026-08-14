from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gear.nature_multihorizon.artifact_store import (
    ArtifactExistsError,
    ArtifactStore,
    IncompleteStageError,
    audit_stage,
    hash_json,
)


class ArtifactStoreTests(unittest.TestCase):
    def test_stage_is_atomically_published_with_manifest_and_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            with store.stage("dataset-v1", "ingest") as stage:
                output = stage.artifact_path("tables/papers.parquet")
                output.write_bytes(b"not-a-real-parquet-yet")
                stage.record_table("tables/papers.parquet", 3, ("paper_id",))

            result = stage.result
            self.assertTrue((result.path / "_SUCCESS").is_file())
            self.assertTrue((result.path / "manifest.json").is_file())
            self.assertFalse(stage.path.exists())
            self.assertEqual(3, result.manifest.row_counts["tables/papers.parquet"])
            self.assertEqual(
                ("paper_id",),
                result.manifest.artifacts["tables/papers.parquet"].primary_key,
            )
            self.assertTrue(audit_stage(result.path).ok)

            with self.assertRaises(ArtifactExistsError):
                with store.stage("dataset-v1", "ingest"):
                    pass

    def test_interrupted_stage_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                with store.stage("dataset-v1", "references") as stage:
                    stage.artifact_path("part-000.jsonl").write_text("one\n", encoding="utf-8")
                    raise RuntimeError("interrupt")

            self.assertTrue(store.staging_path("dataset-v1", "references").is_dir())
            with self.assertRaises(IncompleteStageError):
                with store.stage("dataset-v1", "references"):
                    pass

            with store.stage("dataset-v1", "references", resume=True) as resumed:
                self.assertTrue(resumed.resumed)
                resumed.artifact_path("part-001.jsonl").write_text("two\n", encoding="utf-8")
            self.assertTrue(audit_stage(resumed.result.path).ok)

    def test_audit_detects_post_publication_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            with store.stage("dataset-v1", "features") as stage:
                stage.artifact_path("features.parquet").write_bytes(b"original")
            completed = stage.result.path
            (completed / "features.parquet").write_bytes(b"tampered")

            audit = audit_stage(completed)
            self.assertFalse(audit.ok)
            self.assertTrue(any("mismatch" in error for error in audit.errors))

    def test_success_marker_binds_stage_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            with store.stage(
                "dataset-v1",
                "features",
                code_hash=hash_json("original-code"),
            ) as stage:
                stage.artifact_path("features.parquet").write_bytes(b"original")
            manifest_path = stage.result.path / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["code_hash"] = hash_json("tampered-code")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            audit = audit_stage(stage.result.path)
            self.assertFalse(audit.ok)
            self.assertTrue(
                any("manifest identity" in error for error in audit.errors)
            )

    def test_hash_json_is_order_independent_for_mappings(self) -> None:
        self.assertEqual(hash_json({"a": 1, "b": 2}), hash_json({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
