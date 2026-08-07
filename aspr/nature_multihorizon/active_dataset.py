"""Resolve and validate the active Nature multi-horizon dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ACTIVE_DATASET_REGISTRY = (
    Path("configs") / "nature_multihorizon" / "active_dataset.json"
)


def load_active_dataset(project_root: Path) -> Dict[str, Any]:
    """Load the active dataset registry and require a passing quality audit."""
    root = Path(project_root).resolve()
    override = os.environ.get("ASPR_ACTIVE_DATASET_REGISTRY", "").strip()
    registry_path = Path(override) if override else root / ACTIVE_DATASET_REGISTRY
    if not registry_path.is_absolute():
        registry_path = root / registry_path
    registry_path = registry_path.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dataset_dir = root / registry["dataset_dir"]
    contract_path = root / registry["contract"]
    quality_path = root / registry["quality_report"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if registry["active_dataset_version"] != contract["dataset_version"]:
        raise ValueError("active dataset registry and contract versions differ")
    if not quality.get("overall_pass"):
        raise ValueError("active dataset quality audit has not passed")
    return {
        **registry,
        "registry_path": registry_path,
        "dataset_dir": dataset_dir,
        "feature_dataset_dir": root / registry["feature_dataset_dir"],
        "indicator_matrix_dir": root / registry["indicator_matrix_dir"],
        "contract_path": contract_path,
        "quality_report_path": quality_path,
        "contract_payload": contract,
        "quality_payload": quality,
    }


def active_horizon_end_year(project_root: Path, horizon: int) -> int:
    """Return the registered mature publication-year cap for a horizon."""
    active = load_active_dataset(project_root)
    return int(active["horizon_publication_year_max"][str(int(horizon))])


__all__ = [
    "ACTIVE_DATASET_REGISTRY",
    "active_horizon_end_year",
    "load_active_dataset",
]
