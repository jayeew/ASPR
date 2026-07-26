"""Shared local test entry point for figure-specific ``tests.py`` files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from experiments.common.new.adapters.audit import smoke_test_figure
from experiments.common.new.adapters.io import sha256_file
from experiments.common.new.adapters.runtime import load_figure_context, run_figure


def run_tests(
    figure_id: int,
    config_path: Path,
    argv: Sequence[str] | None = None,
) -> None:
    """Run fast contract tests, optionally followed by the full figure audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Render the figure and execute all scientific acceptance checks.",
    )
    args = parser.parse_args(argv)
    smoke_test_figure(figure_id)
    config_path = config_path.resolve()
    first_hash = sha256_file(config_path)
    second_hash = sha256_file(config_path)
    assert first_hash == second_hash
    _, paths, local = load_figure_context(config_path)
    assert paths.output_root.name == "new"
    assert paths.output_root.parent.name == f"fig{figure_id:02d}"
    assert local["output_dir"].endswith(f"/fig{figure_id:02d}/new")
    if args.full:
        first = run_figure(figure_id, config_path, stage="all")
        first_hashes = {
            str(path.relative_to(paths.output_root)): sha256_file(path)
            for path in sorted(
                (paths.output_root / "panel_data").glob("*")
            )
            if path.is_file()
        }
        first_hashes["figure_full.png"] = sha256_file(
            paths.output_root / "figure_full.png"
        )
        second = run_figure(figure_id, config_path, stage="all")
        second_hashes = {
            str(path.relative_to(paths.output_root)): sha256_file(path)
            for path in sorted(
                (paths.output_root / "panel_data").glob("*")
            )
            if path.is_file()
        }
        second_hashes["figure_full.png"] = sha256_file(
            paths.output_root / "figure_full.png"
        )
        assert first_hashes == second_hashes
        first_sources = {
            source["path"]: source["sha256"]
            for source in first["sources"]
        }
        second_sources = {
            source["path"]: source["sha256"]
            for source in second["sources"]
        }
        assert first_sources == second_sources
        audit = second.get("audit")
        assert audit is not None
        failed = [
            check for check in audit["checks"] if not bool(check["passed"])
        ]
        assert not failed, failed
    print(f"Fig.{figure_id} tests passed (full={args.full})")
