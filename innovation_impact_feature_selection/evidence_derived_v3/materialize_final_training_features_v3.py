from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

from common import (
    DATABASE_PATH,
    OUTPUT_DIR,
    json_hash,
    read_json,
    sha256_file,
    write_json,
)
from database import connect, require_complete


ROOT = Path(__file__).resolve().parent
DEFAULT_MAPPING = (
    OUTPUT_DIR
    / "targeted_operationalizations_v3"
    / "targeted_feature_data_mapping_v3.json"
)
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "final_training_features_v3"
MATRIX_NAME = "final_training_features_v3.parquet"
SCHEMA_NAME = "final_training_features_schema_v3.json"


def _selected_features(
    connection: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT f.feature_id, f.canonical_name_en, f.scope_role,
                   f.redundancy_family, f.research_groups_json,
                   d.final_role, d.decision_reason
            FROM feature_decisions d
            JOIN indicator_families f USING(feature_id)
            WHERE d.final_role != 'excluded'
            ORDER BY CASE d.final_role
                       WHEN 'predictive' THEN 0
                       WHEN 'opportunity' THEN 1
                       WHEN 'control' THEN 2
                       ELSE 3
                     END,
                     f.feature_id
            """
        )
    ]
    if not rows:
        raise RuntimeError("No final selected feature is available")
    memberships: Dict[str, List[Dict[str, str]]] = {
        str(row["feature_id"]): [] for row in rows
    }
    for dimension in connection.execute(
        """
        SELECT c.dimension_id, c.label, d.dimension_role,
               d.selected_feature_ids_json
        FROM candidate_dimensions c
        JOIN dimension_decisions d USING(dimension_id)
        WHERE d.selected = 1
        ORDER BY c.dimension_id
        """
    ):
        for feature_id in json.loads(
            str(dimension["selected_feature_ids_json"])
        ):
            if feature_id in memberships:
                memberships[feature_id].append(
                    {
                        "dimension_id": str(dimension["dimension_id"]),
                        "dimension_label": str(dimension["label"]),
                        "dimension_role": str(dimension["dimension_role"]),
                    }
                )
    for row in rows:
        row["research_groups"] = json.loads(
            str(row.pop("research_groups_json"))
        )
        row["dimensions"] = memberships[str(row["feature_id"])]
    return rows


def _mapped_sources(
    mapping: Mapping[str, Any],
    selected: List[Dict[str, Any]],
) -> tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    sources = mapping.get("sources")
    features = mapping.get("features")
    if not isinstance(sources, dict) or not isinstance(features, dict):
        raise ValueError("Feature mapping requires sources and features")
    source_frames: Dict[str, pd.DataFrame] = {}
    resolved: Dict[str, Dict[str, str]] = {}
    output: pd.DataFrame | None = None
    for row in selected:
        feature_id = str(row["feature_id"])
        feature_mapping = features.get(feature_id)
        if not isinstance(feature_mapping, dict):
            raise ValueError(f"Final feature lacks mapping: {feature_id}")
        source_id = str(feature_mapping.get("source") or "")
        column = str(feature_mapping.get("column") or "")
        source_path = Path(str(sources.get(source_id) or "")).resolve()
        if not source_id or not column or not source_path.is_file():
            raise ValueError(f"Invalid mapping for {feature_id}")
        if source_id not in source_frames:
            frame = pd.read_parquet(source_path)
            if (
                "paper_id" not in frame
                or frame["paper_id"].isna().any()
                or frame["paper_id"].duplicated().any()
            ):
                raise ValueError(f"Invalid paper_id in {source_path}")
            source_frames[source_id] = frame
        frame = source_frames[source_id]
        if column not in frame:
            raise ValueError(f"{column} is absent from {source_path}")
        values = frame[["paper_id", column]].rename(
            columns={column: feature_id}
        )
        output = (
            values
            if output is None
            else output.merge(
                values,
                on="paper_id",
                how="outer",
                validate="one_to_one",
            )
        )
        resolved[feature_id] = {
            "source_id": source_id,
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source_column": column,
        }
    if output is None:
        raise RuntimeError("No mapped final features")
    return output.sort_values("paper_id", kind="stable"), resolved


def _feature_summary(
    frame: pd.DataFrame,
    feature_id: str,
) -> Dict[str, Any]:
    series = frame[feature_id]
    valid = series.notna()
    if int(valid.sum()) == 0 or int(series[valid].nunique()) <= 1:
        raise ValueError(f"Final feature is empty or constant: {feature_id}")
    return {
        "dtype": str(series.dtype),
        "row_count": int(len(series)),
        "valid_count": int(valid.sum()),
        "missing_count": int((~valid).sum()),
        "missing_rate": float((~valid).mean()),
        "unique_count": int(series[valid].nunique(dropna=True)),
    }


def materialize(
    connection: sqlite3.Connection,
    mapping_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    require_complete(connection, ["features_selected"])
    mapping = read_json(mapping_path)
    selected = _selected_features(connection)
    frame, resolved = _mapped_sources(mapping, selected)
    feature_ids = [str(row["feature_id"]) for row in selected]
    frame = frame[["paper_id", *feature_ids]]
    summaries = {
        feature_id: _feature_summary(frame, feature_id)
        for feature_id in feature_ids
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = (output_dir / MATRIX_NAME).resolve()
    frame.to_parquet(matrix_path, index=False)
    decision_basis = {
        "features": selected,
        "mapping_sha256": sha256_file(mapping_path),
        "source_columns": resolved,
    }
    schema_path = (output_dir / SCHEMA_NAME).resolve()
    write_json(
        schema_path,
        {
            "schema_version": "final_training_features_v3",
            "decision_basis_hash": json_hash(decision_basis),
            "matrix_path": str(matrix_path),
            "matrix_sha256": sha256_file(matrix_path),
            "mapping_path": str(mapping_path),
            "mapping_sha256": sha256_file(mapping_path),
            "row_count": int(len(frame)),
            "feature_count": len(feature_ids),
            "feature_ids": feature_ids,
            "features": [
                {
                    **row,
                    **resolved[str(row["feature_id"])],
                    **summaries[str(row["feature_id"])],
                }
                for row in selected
            ],
            "contains_outcomes": False,
            "uses_future_information": False,
            "target_count_is_not_a_selection_quota": True,
        },
    )
    return {
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "schema_path": str(schema_path),
        "schema_sha256": sha256_file(schema_path),
        "row_count": int(len(frame)),
        "feature_count": len(feature_ids),
        "feature_ids": feature_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    connection = connect(args.database.resolve())
    try:
        result = materialize(
            connection,
            args.mapping.resolve(),
            args.output_dir.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
