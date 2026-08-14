from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from gear.config import load_config

BANNED = {
    "gear.lats",
    "gear.review_committee",
    "gear.graph_innovation_scorer",
    "gear.graph_rag",
}


def test_gear_ast_has_no_legacy_imports_or_score_packet():
    root = Path(__file__).resolve().parents[2] / "gear"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "ScorePacket" not in source
        tree = ast.parse(source, filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            imported == banned or imported.startswith(f"{banned}.")
            for imported in imports
            for banned in BANNED
        ), path


def test_importing_gear_has_no_legacy_model_side_effects():
    program = (
        "import sys; import gear; "
        f"banned={BANNED!r}; "
        "assert not (set(sys.modules) & banned), set(sys.modules) & banned"
    )
    subprocess.run([sys.executable, "-c", program], check=True)


def test_aspr_runtime_package_was_removed():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "aspr").exists()


def test_current_fig1_to_fig3_policy_cannot_be_disabled():
    with pytest.raises(ValidationError, match="current_fig1_3_only"):
        load_config(overrides={"current_fig1_3_only": False})
