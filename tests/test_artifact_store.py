from __future__ import annotations

import pytest

from artifact_store import ArtifactStore
from artifact_store.catalog import validate_dependency


def test_publish_once_and_resolve_hash_verified_release(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text('{"value": 1}\n', encoding="utf-8")
    store = ArtifactStore(tmp_path / "artifacts")

    reference = store.publish_directory(
        producer="datasets",
        artifact="dataset_release",
        release="sample",
        source=source,
    )

    resolved = store.resolve(reference)
    assert (resolved / "result.json").read_text(encoding="utf-8") == '{"value": 1}\n'
    assert (
        store.publish_directory(
            producer="datasets",
            artifact="dataset_release",
            release="sample",
            source=source,
        )
        == reference
    )

    (resolved / "result.json").write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        store.resolve(reference)


def test_catalog_allows_only_declared_module_dependencies():
    validate_dependency("gear_agent", "datasets")
    validate_dependency("consistency_evaluation", "review_reconstruction")
    with pytest.raises(ValueError, match="cannot consume"):
        validate_dependency("figures", "gear_agent")
