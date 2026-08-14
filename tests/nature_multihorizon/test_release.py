from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gear.nature_multihorizon.artifact_store import ArtifactExistsError, hash_json
from gear.nature_multihorizon.contracts import ReleaseChannel
from gear.nature_multihorizon.release import (
    REQUIRED_FROZEN_ARTIFACTS,
    ReleaseAuditError,
    ReleaseError,
    audit_release,
    build_release_manifest,
    freeze_candidate,
    load_release,
    publish_release,
    release_directory,
    validate_candidate_for_freeze,
)


class ReleaseTests(unittest.TestCase):
    def _publish_candidate(self, root: Path) -> tuple[Path, object]:
        source = root / "source"
        source.mkdir()
        scores = source / "paper_scores.parquet"
        scores.write_bytes(b"score-table")
        sources = {"paper_scores": scores}
        manifest = build_release_manifest(
            source_snapshot_id="openalex-snapshot-v1",
            dataset_id="nature-multihorizon-v1",
            analysis_id="analysis-001",
            channel=ReleaseChannel.CANDIDATE,
            config_hash=hash_json({"config": 1}),
            code_hash=hash_json({"code": 1}),
            dirty_diff_hash=hash_json("clean"),
            source_artifacts=sources,
            artifact_paths={"paper_scores": "paper_scores.parquet"},
            row_counts={"paper_scores": 2},
            primary_keys={"paper_scores": ("paper_id", "horizon")},
        )
        return scores, publish_release(root / "releases", manifest, sources)

    def _publish_freezable_candidate(self, root: Path) -> object:
        source = root / "full-source"
        source.mkdir()
        sources = {}
        artifact_paths = {}
        for name in sorted(REQUIRED_FROZEN_ARTIFACTS):
            if name == "graph_snapshots":
                continue
            suffix = (
                ".json"
                if name.endswith("registry")
                or name in {"quality_report", "run_protocol"}
                else ".bin"
            )
            path = source / f"{name}{suffix}"
            if name == "quality_report":
                path.write_text(json.dumps({"go_for_frozen_release": True}), encoding="utf-8")
            elif name == "run_protocol":
                path.write_text(
                    json.dumps(
                        {
                            "dataset_id": "nature-multihorizon-v1",
                            "analysis_id": "analysis-002",
                            "source_snapshot_id": "openalex-snapshot-v1",
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                path.write_bytes(name.encode("utf-8"))
            sources[name] = path
            artifact_paths[name] = path.name
        graph_files = {
            "prior-2000-v1.nodes.parquet": pd.DataFrame(
                [{"node_id": "W1", "degree": 0}]
            ),
            "prior-2000-v1.edges.parquet": pd.DataFrame(
                columns=["left_id", "right_id"]
            ),
            "prior-2000-v1.pairs.parquet": pd.DataFrame(
                columns=["left_id", "right_id", "pair_count"]
            ),
        }
        for filename, frame in graph_files.items():
            path = source / filename
            frame.to_parquet(path, index=False)
            name = f"graph_asset__{filename}"
            sources[name] = path
            artifact_paths[name] = f"graph_snapshots/{filename}"
        catalog = source / "graph_snapshots.parquet"
        pd.DataFrame(
            [
                {
                    "cutoff_year": 2000,
                    "graph_id": "prior-2000-v1",
                    "node_path": "prior-2000-v1.nodes.parquet",
                    "edge_path": "prior-2000-v1.edges.parquet",
                    "pair_path": "prior-2000-v1.pairs.parquet",
                }
            ]
        ).to_parquet(catalog, index=False)
        sources["graph_snapshots"] = catalog
        artifact_paths["graph_snapshots"] = (
            "graph_snapshots/graph_snapshots.parquet"
        )
        for index in range(1, 11):
            figure_id = f"fig{index:02d}"
            view_files = {
                "_SUCCESS": "ok\n",
                "view_manifest.json": json.dumps(
                    {"analysis_id": "analysis-002", "figure_id": figure_id}
                ),
                "panel_spec.json": json.dumps(
                    {"analysis_id": "analysis-002", "figure_id": figure_id}
                ),
                "caption_stats.json": json.dumps(
                    {"analysis_id": "analysis-002", "figure_id": figure_id}
                ),
                "data/plot.csv": "x,y\n1,1\n",
            }
            for relative, content in view_files.items():
                safe_name = relative.replace("/", "__")
                name = f"view_{figure_id}_{safe_name}"
                path = source / f"{name}.txt"
                path.write_text(content, encoding="utf-8")
                sources[name] = path
                artifact_paths[name] = f"figure_views/{figure_id}/{relative}"
        manifest = build_release_manifest(
            source_snapshot_id="openalex-snapshot-v1",
            dataset_id="nature-multihorizon-v1",
            analysis_id="analysis-002",
            channel=ReleaseChannel.CANDIDATE,
            config_hash=hash_json({"config": 1}),
            code_hash=hash_json({"code": 1}),
            dirty_diff_hash=hash_json("clean"),
            source_artifacts=sources,
            artifact_paths=artifact_paths,
        )
        return publish_release(root / "releases", manifest, sources)

    def test_candidate_is_audited_and_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, loaded = self._publish_candidate(root)

            self.assertEqual(ReleaseChannel.CANDIDATE, loaded.manifest.channel)
            self.assertEqual(b"score-table", loaded.artifact("paper_scores").read_bytes())
            self.assertTrue(audit_release(loaded.path).ok)
            with self.assertRaises(ReleaseAuditError):
                load_release(loaded.path, require_frozen=True)
            with self.assertRaisesRegex(ReleaseError, "missing required"):
                validate_candidate_for_freeze(loaded)

            sources = {"paper_scores": source}
            with self.assertRaises(ArtifactExistsError):
                publish_release(root / "releases", loaded.manifest, sources)

    def test_candidate_promotion_creates_separate_immutable_frozen_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._publish_candidate(root)
            with self.assertRaises(ReleaseError):
                freeze_candidate(root / "releases", "analysis-001")
            candidate = self._publish_freezable_candidate(root)
            frozen = freeze_candidate(root / "releases", "analysis-002")

            self.assertEqual(ReleaseChannel.FROZEN, frozen.manifest.channel)
            self.assertNotEqual(candidate.path, frozen.path)
            self.assertEqual(
                release_directory(root / "releases", "analysis-002", "frozen"),
                frozen.path,
            )
            self.assertEqual(b"paper_scores", frozen.artifact("paper_scores").read_bytes())
            self.assertEqual(
                "analysis-002",
                load_release(frozen.path, require_frozen=True).manifest.analysis_id,
            )

            with self.assertRaises(ArtifactExistsError):
                freeze_candidate(root / "releases", "analysis-002")

    def test_release_audit_detects_changed_or_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, loaded = self._publish_candidate(root)
            loaded.artifact("paper_scores").write_bytes(b"changed")
            (loaded.path / "untracked.txt").write_text("extra", encoding="utf-8")

            audit = audit_release(loaded.path)
            self.assertFalse(audit.ok)
            self.assertIn("untracked.txt", audit.unexpected_files)
            with self.assertRaises(ReleaseAuditError):
                load_release(loaded.path)

    def test_success_marker_binds_release_channel_and_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, loaded = self._publish_candidate(root)
            manifest_path = loaded.path / "release.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["channel"] = "frozen"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            audit = audit_release(loaded.path)
            self.assertFalse(audit.ok)
            self.assertTrue(
                any(
                    "manifest identity" in error or "namespace" in error
                    for error in audit.errors
                )
            )
            with self.assertRaises(ReleaseAuditError):
                load_release(loaded.path, require_frozen=True)

    def test_publish_staging_directory_is_not_a_consumable_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, loaded = self._publish_candidate(root)
            staging = loaded.path.with_name(
                f".{loaded.manifest.analysis_id}.publishing-999999"
            )
            shutil.copytree(loaded.path, staging)

            audit = audit_release(staging)
            self.assertFalse(audit.ok)
            self.assertTrue(any("namespace" in error for error in audit.errors))
            with self.assertRaises(ReleaseAuditError):
                load_release(staging)


if __name__ == "__main__":
    unittest.main()
