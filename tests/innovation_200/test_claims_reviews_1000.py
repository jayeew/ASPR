from __future__ import annotations

from pathlib import Path

import pytest

from experiments.innovation_200.common import experiment_config
from experiments.innovation_200.run_claims_reviews_1000 import stage_environment
from gear.codex_cli import CodexCliJsonClient
from gear.model_client import LazyRoleClient, resolve_role_model


@pytest.mark.parametrize(
    "role",
    [
        "claim_miner",
        "claim_consolidator",
        "internal_verifier",
        "reference_extract",
        "reference_check",
    ],
)
def test_child_role_commands_use_luna_low_fast(
    role: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inherited = {
        "ASPR_GEAR_MODEL_BACKEND": "openai_compatible",
        "ASPR_GEAR_CODEX_MODEL": "gpt-6-astra",
        "ASPR_GEAR_CODEX_REASONING_EFFORT": "max",
        "GEAR_STUDY_MODEL": "gpt-6-astra",
        "GEAR_STUDY_REASONING_EFFORT": "max",
        "GEAR_CODEX_SERVICE_TIER": "default",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    env = stage_environment()
    for name in inherited:
        monkeypatch.setenv(name, env[name])

    config = experiment_config()
    assert config.model_backend == "codex_cli"
    assert config.codex_cli.model == "gpt-5.6-luna"
    assert config.codex_cli.reasoning_effort == "low"
    assert resolve_role_model(config, role) == ("gpt-5.6-luna", "low")
    assert not config.model_cache_enabled
    assert not config.resume_fingerprint_checks_enabled

    # Construct the real adapter command without executing any model request.
    client = LazyRoleClient(config, role)._get()
    assert isinstance(client, CodexCliJsonClient)
    command = client._command(tmp_path, None)
    assert command[command.index("--model") + 1] == "gpt-5.6-luna"
    assert 'model_reasoning_effort="low"' in command
    assert 'service_tier="fast"' in command
