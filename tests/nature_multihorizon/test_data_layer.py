from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import TracebackType
from typing import Callable, Optional, Type

import numpy as np
import pandas as pd

from gear.nature_multihorizon.cohorts import build_cohort_membership
from gear.nature_multihorizon.contracts import AUXILIARY_FEATURES, CORE_FEATURES
from gear.nature_multihorizon.features import build_feature_table
from gear.nature_multihorizon.graph_snapshots import (
    build_graph_snapshots,
    load_prior_graph,
    load_snapshot_catalog,
)
from gear.nature_multihorizon.targets import (
    DIFFUSION_TARGET_COMPONENTS,
    FoldLocalDiffusionTarget,
    build_diffusion_targets,
    build_future_fetch_status,
)
from gear.nature_multihorizon.taxonomy import build_taxonomy_table, map_domain12
from gear.nature_multihorizon.v5_adapter import (
    audit_reference_recovery,
    audit_v5_source,
    ingest_v5,
    iter_jsonl_records,
    recover_v5_reference_closure,
)


class _Raises:
    def __init__(self, exception: Type[BaseException], match: str) -> None:
        self.exception = exception
        self.match = match

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        del traceback
        if exception_type is None or exception is None:
            raise AssertionError(f"Expected {self.exception.__name__}")
        if not issubclass(exception_type, self.exception):
            return False
        if self.match and re.search(self.match, str(exception)) is None:
            raise AssertionError(f"Exception {exception!r} does not match {self.match!r}")
        return True


class _PytestCompat:
    @staticmethod
    def raises(exception: Type[BaseException], *, match: str = "") -> _Raises:
        return _Raises(exception, match)


pytest = _PytestCompat()


def _work(work_id: str, year: int, refs: list[str] | None = None) -> dict:
    return {
        "id": f"https://openalex.org/{work_id}",
        "display_name": work_id,
        "publication_year": year,
        "type": "article",
        "referenced_works": [f"https://openalex.org/{item}" for item in (refs or [])],
        "primary_topic": {
            "id": "https://openalex.org/T1",
            "display_name": "Molecular biology",
            "domain": {"display_name": "Life Sciences"},
            "field": {"display_name": "Biochemistry, Genetics and Molecular Biology"},
            "subfield": {"display_name": "Molecular Biology"},
        },
    }


def test_jsonl_streaming_is_strict_and_can_audit_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"work":{"id":"W1"}}\nnot-json\n', encoding="utf-8")
    diagnostics: dict[str, int] = {}
    rows = list(iter_jsonl_records(path, unwrap_key="work", strict=False, diagnostics=diagnostics))
    assert rows == [{"id": "W1"}]
    assert diagnostics["bad_json_records"] == 1
    with pytest.raises(ValueError, match="Malformed JSONL"):
        list(iter_jsonl_records(path, strict=True))


def test_recovery_has_file_ledger_resume_and_atomic_formal_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    snapshot = tmp_path / "snapshot" / "data" / "works"
    (source / "checkpoints").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    pd.DataFrame(
        [
            {"source": "P1", "target": "R1"},
            {"source": "P1", "target": "R2"},
            {"source": "P2", "target": "R3"},
        ]
    ).to_csv(source / "nature_reference_edges.csv", index=False)
    (source / "checkpoints" / "reference_works.jsonl").write_text(
        json.dumps({"work": _work("R1", 1990)}) + "\n", encoding="utf-8"
    )
    for name, work in (("a", _work("R2", 1991, ["R1"])), ("b", _work("R3", 1992))):
        with gzip.open(snapshot / f"{name}.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(work) + "\n")

    dry = recover_v5_reference_closure(source, tmp_path / "snapshot", dry_run=True)
    assert dry["stage_status"] == "dry_run"
    assert not (source / "reference_closure_recovery").exists()

    partial = recover_v5_reference_closure(
        source, tmp_path / "snapshot", max_snapshot_files=1, workers=1
    )
    assert partial["stage_status"] == "partial"
    assert not (source / "nature_reference_works.csv").exists()

    complete = recover_v5_reference_closure(source, tmp_path / "snapshot", resume=True)
    assert complete["stage_status"] == "complete"
    assert complete["n_files_scanned_this_run"] == 1
    formal = pd.read_csv(source / "nature_reference_works.csv")
    assert formal["id"].is_unique
    assert set(formal["short_id"]) == {"R1", "R2", "R3"}
    assert "referenced_works" in formal
    assert (source / "reference_closure_recovery" / "_SUCCESS").exists()
    assert audit_reference_recovery(source)["ok"]

    manifest_path = source / "reference_closure_recovery" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"] = 0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered = audit_reference_recovery(source, verify_reference_hash=False)
    assert not tampered["ok"]
    assert any("manifest identity" in error for error in tampered["errors"])


def _write_ingest_fixture(root: Path) -> None:
    (root / "checkpoints" / "future_citers_tau8").mkdir(parents=True)
    targets = pd.DataFrame(
        [
            {
                "id": f"https://openalex.org/P{index}",
                "year": 2000,
                "title": f"Paper {index}",
                "journal_family": "nature_research",
                "document_type": "article",
                "openalex_primary_field": "Chemistry",
                "openalex_primary_subfield": "Organic Chemistry",
                "primary_topic": "Organic chemistry",
            }
            for index in range(1, 4)
        ]
    )
    targets.to_csv(root / "nature_target_works.csv", index=False)
    pd.DataFrame(
        [
            {"source": "P1", "target": "R1"},
            {"source": "P1", "target": "R1"},
            {"source": "P2", "target": "R2"},
        ]
    ).to_csv(root / "nature_reference_edges.csv", index=False)
    references = pd.DataFrame(
        [
            {
                "id": "https://openalex.org/R1",
                "year": 1990,
                "openalex_primary_field": "Chemistry",
                "referenced_works": json.dumps(["https://openalex.org/R0"]),
            },
            {
                "id": "https://openalex.org/R2",
                "year": 1991,
                "openalex_primary_field": "Chemistry",
                "referenced_works": "[]",
            },
        ]
    )
    references.to_csv(root / "nature_reference_works.csv", index=False)
    pd.DataFrame(
        [
            {
                "paper_id": "https://openalex.org/P1",
                "citer_id": "https://openalex.org/C1",
                "citer_year": 2001,
                "citer_primary_field": "Chemistry",
                "citer_primary_subfield": "Organic Chemistry",
                "citer_primary_topic": "Catalysis",
                "fetch_status": "fetched",
            }
        ]
    ).to_csv(root / "nature_future_citers.csv", index=False)
    pd.DataFrame(
        [
            {"paper_id": f"https://openalex.org/P{index}", "tau": 8, "n_future_citers": count}
            for index, count in ((1, 1), (2, 0), (3, 0))
        ]
    ).to_csv(root / "nature_future_graph_deltas.csv", index=False)
    (root / "checkpoints" / "future_citers_tau8" / "P1.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (root / "checkpoints" / "future_citers_tau8" / "P2.jsonl").write_text(
        "", encoding="utf-8"
    )


def test_ingest_rejects_tmp_and_globally_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_ingest_fixture(source)
    formal = source / "nature_reference_works.csv"
    formal.rename(source / "nature_reference_works.csv.tmp")
    with pytest.raises(FileNotFoundError, match="unfinished file"):
        ingest_v5(source, tmp_path / "bad")
    (source / "nature_reference_works.csv.tmp").rename(formal)

    manifest = ingest_v5(source, tmp_path / "ingested")
    paper_refs = pd.read_parquet(tmp_path / "ingested" / "paper_references.parquet")
    ref_edges = pd.read_parquet(tmp_path / "ingested" / "reference_edges.parquet")
    statuses = pd.read_parquet(tmp_path / "ingested" / "future_fetch_status.parquet")
    assert len(paper_refs) == 2
    assert paper_refs[["paper_id", "reference_id"]].duplicated().sum() == 0
    assert ref_edges[["source_reference_id", "target_reference_id"]].duplicated().sum() == 0
    assert manifest["duplicates_removed"]["paper_references"] == 1
    status = statuses.set_index("paper_id")["fetch_status"].to_dict()
    assert status["https://openalex.org/P2"] == "zero_success"
    assert status["https://openalex.org/P3"] == "not_requested_or_failed"


def test_ingest_resumes_with_existing_sqlite_schema(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_ingest_fixture(source)
    output = tmp_path / "resume-ingest"
    output.mkdir()
    with sqlite3.connect(output / ".ingest_dedup.sqlite") as database:
        database.executescript(
            """
            CREATE TABLE papers (paper_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE paper_references (
                paper_id TEXT NOT NULL, reference_id TEXT NOT NULL,
                PRIMARY KEY(paper_id, reference_id));
            CREATE TABLE reference_works (reference_id TEXT PRIMARY KEY, row_json TEXT NOT NULL);
            CREATE TABLE reference_edges (
                source_reference_id TEXT NOT NULL, target_reference_id TEXT NOT NULL,
                edge_year INTEGER, PRIMARY KEY(source_reference_id, target_reference_id));
            CREATE TABLE future_citers (
                paper_id TEXT NOT NULL, requested_horizon INTEGER NOT NULL,
                citer_id TEXT NOT NULL, row_json TEXT NOT NULL,
                PRIMARY KEY(paper_id, requested_horizon, citer_id));
            """
        )
    manifest = ingest_v5(source, output)
    assert manifest["row_counts"]["papers"] == 3
    assert (output / "papers.parquet").is_file()


def test_taxonomy_uses_specific_topic_and_excludes_nonnatural_scope() -> None:
    papers = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "primary_topic": "Quantum many-body physics",
                "openalex_primary_subfield": "Condensed Matter Physics",
                "openalex_primary_field": "Physics and Astronomy",
            },
            {
                "paper_id": "P2",
                "primary_topic": "Exoplanet atmospheres",
                "openalex_primary_subfield": "Astronomy and Astrophysics",
                "openalex_primary_field": "Physics and Astronomy",
            },
            {
                "paper_id": "P3",
                "primary_topic": "Education policy",
                "openalex_primary_field": "Social Sciences",
            },
            {
                "paper_id": "P4",
                "primary_topic": "Perovskite solar cell stability",
                "openalex_primary_field": "Materials Science",
            },
            {
                "paper_id": "P5",
                "primary_topic": "Cellular network resource allocation",
                "openalex_primary_field": "Computer Science",
            },
            {
                "paper_id": "P6",
                "primary_topic": "Graph neural networks for molecules",
                "openalex_primary_field": "Computer Science",
            },
            {
                "paper_id": "P7",
                "primary_topic": "Energy management optimization",
                "openalex_primary_field": "Engineering",
            },
        ]
    )
    mapped = map_domain12(papers)
    assert mapped.set_index("paper_id").loc["P1", "domain12"] == "physics"
    assert mapped.set_index("paper_id").loc["P2", "domain12"] == "astronomy_space"
    assert mapped.set_index("paper_id").loc["P3", "domain12"] == "out_of_scope_nonnatural"
    assert mapped.set_index("paper_id").loc["P4", "domain12"] == "materials_nanoscience"
    assert mapped.set_index("paper_id").loc["P5", "domain12"] == "computer_science_ai"
    assert mapped.set_index("paper_id").loc["P6", "domain12"] == "computer_science_ai"
    assert mapped.set_index("paper_id").loc["P7", "domain12"] == "engineering_energy"
    _, coverage, audit = build_taxonomy_table(papers, min_discussion_size=1)
    assert len(coverage) == 12
    assert audit["n_out_of_scope_nonnatural"] == 1


def _graph_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    papers = pd.DataFrame(
        [
            {"paper_id": "P0", "publication_year": 2000, "domain12": "chemistry", "venue_family": "nature_research"},
            {"paper_id": "P1", "publication_year": 2005, "domain12": "chemistry", "venue_family": "nature_research"},
        ]
    )
    paper_refs = pd.DataFrame(
        [
            {"paper_id": paper, "reference_id": f"R{reference}"}
            for paper, references in (("P0", range(1, 6)), ("P1", range(1, 7)))
            for reference in references
        ]
    )
    reference_works = pd.DataFrame(
        [
            {
                "reference_id": f"R{index}",
                "publication_year": 1985 + index,
                "openalex_primary_field": "Chemistry" if index <= 3 else "Materials Science",
            }
            for index in range(1, 7)
        ]
    )
    reference_edges = pd.DataFrame(
        [
            {"source_reference_id": "R1", "target_reference_id": "R2", "edge_year": 1995},
            {"source_reference_id": "R2", "target_reference_id": "R3", "edge_year": 1996},
            {"source_reference_id": "R4", "target_reference_id": "R5", "edge_year": 1997},
            # Same-year edge must not appear in the 2005 focal snapshot.
            {"source_reference_id": "R3", "target_reference_id": "R4", "edge_year": 2005},
        ]
    )
    return papers, paper_refs, reference_works, reference_edges


def test_graph_and_features_are_strictly_prior_and_relocation_safe(tmp_path: Path) -> None:
    papers, paper_refs, references, reference_edges = _graph_fixture()
    staging = tmp_path / "staging"
    catalog = build_graph_snapshots(
        papers, paper_refs, references, reference_edges, staging, interval=5
    )
    final = tmp_path / "final"
    os.replace(staging, final)
    persisted = load_snapshot_catalog(final / "graph_snapshots.parquet")
    graph = load_prior_graph(persisted, 2005)
    assert graph.source_max_year == 2004
    assert ("https://openalex.org/R3", "https://openalex.org/R4") not in graph.edges

    focal = papers[papers["paper_id"] == "P1"]
    features = build_feature_table(
        focal,
        paper_refs,
        references,
        final / "graph_snapshots.parquet",
    )
    assert set(CORE_FEATURES + AUXILIARY_FEATURES).issubset(features.columns)
    assert (features["source_max_year"] < features["publication_year"]).all()
    assert features.loc[0, "valid_reference_count"] == 6
    assert features.loc[0, "valid_pair_count"] > 0


def test_targets_and_cohort_keep_failure_distinct_from_true_zero() -> None:
    papers = pd.DataFrame(
        [
            {
                "paper_id": f"P{index}",
                "publication_year": 2010,
                "domain12": "chemistry",
                "work_type": "article",
                "venue_family": "nature_research",
            }
            for index in range(1, 5)
        ]
    )
    citer_rows = []
    for paper_id, count in (("P1", 12), ("P4", 10)):
        for index in range(count):
            citer_rows.append(
                {
                    "paper_id": paper_id,
                    "citer_id": f"{paper_id}C{index}",
                    "citer_year": 2011 + index % 3,
                    "citer_primary_field": "Chemistry" if index % 2 else "Materials Science",
                    "citer_primary_subfield": f"Subfield {index % 3}",
                    "citer_primary_topic": f"Topic {index % 4}",
                }
            )
    citers = pd.DataFrame(citer_rows)
    explicit = pd.DataFrame(
        [
            {"paper_id": "P1", "requested_horizon": 8, "fetch_status": "success", "n_returned": 12},
            {"paper_id": "P2", "requested_horizon": 8, "fetch_status": "zero_success", "n_returned": 0},
            {"paper_id": "P3", "requested_horizon": 8, "fetch_status": "failed", "n_returned": np.nan},
            {"paper_id": "P4", "requested_horizon": 8, "fetch_status": "success", "n_returned": 10},
        ]
    )
    status = build_future_fetch_status(papers, citers, explicit_status=explicit)
    targets = build_diffusion_targets(papers, citers, status, horizons=(3, 5, 8))
    tau5 = targets[targets["horizon"] == 5].set_index("paper_id")
    assert tau5.loc["https://openalex.org/P2", "target_valid"] == 1
    assert tau5.loc["https://openalex.org/P2", "n_future_citers"] == 0
    assert tau5.loc["https://openalex.org/P3", "target_valid"] == 0
    assert np.isnan(tau5.loc["https://openalex.org/P3", "n_future_citers"])

    feature_rows = []
    for paper_id in papers["paper_id"]:
        row = {
            "paper_id": f"https://openalex.org/{paper_id}",
            "valid_reference_count": 12,
            "reference_metadata_coverage": 0.9,
        }
        row.update({feature: 0.5 for feature in CORE_FEATURES})
        feature_rows.append(row)
    membership = build_cohort_membership(
        papers.assign(paper_id=papers["paper_id"].map(lambda value: f"https://openalex.org/{value}")),
        pd.DataFrame(feature_rows),
        targets,
    )
    tau5_members = membership[membership["horizon"] == 5].set_index("paper_id")
    assert tau5_members.loc["https://openalex.org/P1", "cohort_member"] == 1
    assert tau5_members.loc["https://openalex.org/P4", "cohort_member"] == 1
    assert tau5_members.loc["https://openalex.org/P2", "cohort_member"] == 0
    assert tau5_members.loc["https://openalex.org/P3", "cohort_member"] == 0


def test_fold_local_target_uses_average_tie_ranks() -> None:
    values = [1.0, 1.0, 3.0, 4.0]
    frame = pd.DataFrame(
        {name: values for name in DIFFUSION_TARGET_COMPONENTS}
    )
    transformed = FoldLocalDiffusionTarget().fit_transform(frame)
    expected = pd.Series(values).rank(method="average", pct=True).to_numpy(float)
    assert np.allclose(transformed, expected)


def test_short_horizon_is_not_marked_capped_when_tau8_cap_occurs_later() -> None:
    papers = pd.DataFrame(
        [{"paper_id": "P1", "publication_year": 2000, "domain12": "chemistry"}]
    )
    future = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "citer_id": f"C{index}",
                "citer_year": 2001 + index % 6,
                "citer_primary_field": "Chemistry",
                "citer_primary_subfield": "Organic Chemistry",
                "citer_primary_topic": "Catalysis",
                "requested_horizon": 8,
            }
            for index in range(12)
        ]
    )
    status = pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "requested_horizon": 8,
                "fetch_status": "success",
                "cap_hit": 1,
                "last_citer_year": 2006,
            }
        ]
    )
    targets = build_diffusion_targets(
        papers,
        future,
        status,
        horizons=(3, 5, 8),
        min_future_citers=1,
    ).set_index("horizon")
    assert int(targets.loc[3, "cap_hit"]) == 0
    assert int(targets.loc[5, "cap_hit"]) == 0
    assert int(targets.loc[8, "cap_hit"]) == 1


class DataLayerTests(unittest.TestCase):
    def _with_tmp(self, function: Callable[[Path], None]) -> None:
        with tempfile.TemporaryDirectory() as directory:
            function(Path(directory))

    def test_jsonl_streaming(self) -> None:
        self._with_tmp(test_jsonl_streaming_is_strict_and_can_audit_bad_rows)

    def test_recovery_resume(self) -> None:
        self._with_tmp(test_recovery_has_file_ledger_resume_and_atomic_formal_output)

    def test_ingest_deduplication(self) -> None:
        self._with_tmp(test_ingest_rejects_tmp_and_globally_deduplicates)

    def test_ingest_resume(self) -> None:
        self._with_tmp(test_ingest_resumes_with_existing_sqlite_schema)

    def test_taxonomy(self) -> None:
        test_taxonomy_uses_specific_topic_and_excludes_nonnatural_scope()

    def test_graph_features(self) -> None:
        self._with_tmp(test_graph_and_features_are_strictly_prior_and_relocation_safe)

    def test_targets_cohort(self) -> None:
        test_targets_and_cohort_keep_failure_distinct_from_true_zero()

    def test_fold_local_target_ties(self) -> None:
        test_fold_local_target_uses_average_tie_ranks()

    def test_horizon_specific_cap(self) -> None:
        test_short_horizon_is_not_marked_capped_when_tau8_cap_occurs_later()


if __name__ == "__main__":
    unittest.main()
