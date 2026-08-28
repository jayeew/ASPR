from experiments.gear.evaluation.source_fingerprint import (
    audit_expected_source,
    rescue_source_fingerprint,
)


def test_rescue_source_fingerprint_is_stable_and_auditable() -> None:
    first, first_count = rescue_source_fingerprint()
    second, second_count = rescue_source_fingerprint()
    assert first == second
    assert first.startswith("sha256:")
    assert first_count == second_count
    assert first_count > 100
    assert audit_expected_source(first)["passed"] is True
