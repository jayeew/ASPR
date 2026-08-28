from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_full_runner_builds_only_ready_expert_pack_after_gate() -> None:
    script = (ROOT / "scripts/run_gear_graph_rescue_full.sh").read_text(
        encoding="utf-8"
    )
    gate = script.index('if not report["overall_claim_allowed"]')
    build = script.index("expert_annotation_pack build")
    ready_validation = script.index("expert_annotation_pack validate", build)
    assert gate < build < ready_validation
    ready_command = script[ready_validation : ready_validation + 300]
    assert "--require-completed" not in ready_command
    assert "expert_annotation_pack_ready_validation.json" in ready_command


def test_finalize_requires_completed_experts_before_completion_audit() -> None:
    script = (ROOT / "scripts/finalize_gear_graph_rescue.sh").read_text(
        encoding="utf-8"
    )
    completed = script.index("--require-completed")
    audit = script.index("audit_rescue_completion")
    assert completed < audit
    assert "expert_annotation_pack_completed_validation.json" in script
    assert '--expert-pack-validation "${completed_validation}"' in script


def test_finalize_shell_is_syntactically_valid() -> None:
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/finalize_gear_graph_rescue.sh")],
        check=True,
    )
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/run_gear_graph_rescue_full.sh")],
        check=True,
    )


def test_finalize_fails_before_validation_without_frozen_manifest() -> None:
    environment = dict(os.environ)
    environment.pop("GEAR_FROZEN_REPLAY_MANIFEST", None)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/finalize_gear_graph_rescue.sh")],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "GEAR_FROZEN_REPLAY_MANIFEST" in result.stderr
