from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
OUTPUT_DIR = ROOT / "outputs"


def load_module(name: str, path: Path) -> ModuleType:
    """Load a local module without changing the Python path."""
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def test_query_compilation(search: ModuleType) -> None:
    """The frozen matrix must compile to 72 distinct exact searches."""
    config = read_json(ROOT / "search_queries_v2.json")
    queries = search.compile_queries(config)
    assert len(config["domains"]) == 12
    assert len(queries) == 72
    assert len({row["query_id"] for row in queries}) == 72
    assert len({row["query_hash"] for row in queries}) == 72
    assert all(" AND " in row["expression"] for row in queries)
    assert all(row["concept_terms"] for row in queries)
    assert all(row["context_terms"] for row in queries)


def test_record_normalization(search: ModuleType) -> None:
    """DOI identity and abstract reconstruction must be deterministic."""
    assert (
        search.normalize_doi("https://doi.org/10.1000/ABC")
        == "10.1000/abc"
    )
    abstract = search.reconstruct_abstract(
        {"reproducible": [2], "is": [1], "This": [0]}
    )
    assert abstract == "This is reproducible"
    first = search.record_key("10.1000/x", "A", "Title", 2020)
    second = search.record_key("10.1000/x", "B", "Other", 2021)
    assert first == second


def test_env_parser_does_not_execute(search: ModuleType) -> None:
    """The key loader must be a literal parser, never a sourced shell file."""
    source = (ROOT / "expanded_search.py").read_text(encoding="utf-8")
    assert "source .env" not in source
    assert "local_environment_value" in source


def test_crossref_exact_matching(search: ModuleType) -> None:
    """Local snapshot matching must obey both concept and context clauses."""
    config = read_json(ROOT / "search_queries_v2.json")
    queries = search.compile_queries(config)
    item = {
        "DOI": "10.1000/example",
        "title": [
            "Combinatorial novelty and citation impact of scientific papers"
        ],
    }
    hits = search.matching_query_ids(item, queries)
    assert any("combinatorial_novelty" in value for value in hits)
    irrelevant = {
        "title": ["Combinatorial novelty in a commercial puzzle game"]
    }
    assert not search.matching_query_ids(irrelevant, queries)
    assert search.crossref_in_scope(
        {
            "type": "journal-article",
            "published": {"date-parts": [[2020, 1, 1]]},
        },
        config,
    )
    assert not search.crossref_in_scope(
        {
            "type": "book",
            "published": {"date-parts": [[2020, 1, 1]]},
        },
        config,
    )


def test_database_deduplication(search: ModuleType) -> None:
    """One work can link to several queries without duplicate work rows."""
    with tempfile.TemporaryDirectory() as temporary:
        database = Path(temporary) / "test.sqlite3"
        connection = search.connect_database(database)
        item = {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1000/example",
            "display_name": "Example",
            "publication_year": 2020,
            "type": "article",
            "abstract_inverted_index": {},
            "primary_location": {},
        }
        record = search.openalex_record(item)
        search.insert_work(connection, record)
        search.insert_work(connection, record)
        connection.commit()
        count = connection.execute(
            "SELECT COUNT(*) FROM works"
        ).fetchone()[0]
        connection.close()
        assert count == 1


def test_expanded_registry(build: ModuleType) -> None:
    """The expansion must be additive and preserve all baseline rows."""
    assert sum(len(values) for values in build.CATALOG.values()) == 133
    baseline = read_json(PARENT / "feature_registry.json")["features"]
    expanded = read_json(OUTPUT_DIR / "feature_registry_v2.json")["features"]
    assert len(baseline) == 64
    assert len(expanded) == 197
    assert expanded[:64] == baseline
    additions = expanded[64:]
    assert all(
        row["candidate_stage"]
        == "review_mapped_pending_primary_verification"
        for row in additions
    )
    assert all(not row["formula_reproducible"] for row in additions)
    assert all(not row["peer_reviewed_evidence"] for row in additions)
    assert all(row["source_ids"] for row in additions)


def test_screening_outcome() -> None:
    """Unverified concepts must never leak into the training feature sets."""
    summary = read_json(OUTPUT_DIR / "selection_summary_v2.json")
    assert summary["new_review_mapped_concepts"] == 133
    assert summary["selected_new_feature_families"] == 0
    assert summary["selected_feature_families"] == 26
    assert summary["selected_predictor_dimensions"] == 8
    assert summary["selected_context_dimensions"] == 1
    assert not summary["formal_review_complete"]
    assert not summary["manual_coding"]["dual_coding_complete"]


def main() -> None:
    """Run all lightweight offline checks."""
    search = load_module("expanded_search_v2", ROOT / "expanded_search.py")
    build = load_module(
        "build_indicator_registry_v2",
        ROOT / "build_indicator_registry_v2.py",
    )
    test_query_compilation(search)
    test_record_normalization(search)
    test_env_parser_does_not_execute(search)
    test_crossref_exact_matching(search)
    test_database_deduplication(search)
    test_expanded_registry(build)
    test_screening_outcome()
    print("All expanded-review v2 tests passed.")


if __name__ == "__main__":
    main()
