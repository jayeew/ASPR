#!/usr/bin/env python3
"""Re-render a completed suite from frozen experiment tables without retraining."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.nature_multihorizon.source_audit_v6 import sha256_file  # noqa: E402
from experiments.common.old.v6_1_figures_r1.analysis import hash_payload  # noqa: E402
from experiments.common.old.v6_1_figures_r1.render import render_all  # noqa: E402
from experiments.common.old.v6_1_figures_r1.run_all import _create_contact_sheet  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "aspr_v6_1_figures.json"


def _load_tables(output_dir: Path) -> Dict[int, Dict[str, pd.DataFrame]]:
    experiments: Dict[int, Dict[str, pd.DataFrame]] = {}
    for index in range(1, 11):
        data_dir = output_dir / f"experiment_{index:02d}" / "data"
        paths = sorted(data_dir.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No frozen tables for experiment {index}")
        experiments[index] = {
            path.stem: pd.read_csv(path) for path in paths
        }
    return experiments


def rerender(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Refresh only visual artifacts and bind their new hashes."""
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tables = _load_tables(output_dir)
    figure_paths = render_all(
        tables,
        output_dir / "figures",
        formats=list(config["render"]["formats"]),
        dpi=int(config["render"]["dpi"]),
    )
    contact_sheet = _create_contact_sheet(
        {index: paths["png"] for index, paths in figure_paths.items()},
        output_dir / "fig1_fig10_contact_sheet.png",
    )
    output_records = manifest["outputs"]
    output_records["contact_sheet"] = {
        "path": str(contact_sheet.resolve()),
        "sha256": sha256_file(contact_sheet),
        "size_bytes": contact_sheet.stat().st_size,
    }
    for index, paths in figure_paths.items():
        for file_format, path in paths.items():
            output_records[f"figure_{index:02d}_{file_format}"] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest["renderer_sha256"] = sha256_file(
        PROJECT_ROOT / "experiments" / "aspr_v6_1_figures" / "render.py"
    )
    manifest["rerendered_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("artifact_id", None)
    manifest["artifact_id"] = hash_payload(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "_SUCCESS").write_text(
        f"{manifest['artifact_id']}\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "contact_sheet": str(contact_sheet),
        "artifact_id": manifest["artifact_id"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            rerender(args.config.resolve(), args.output_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
