from gear.review_pipeline import runtime_code_fingerprint


def test_runtime_code_fingerprint_is_stable_and_complete() -> None:
    first, first_count = runtime_code_fingerprint()
    second, second_count = runtime_code_fingerprint()
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64
    assert first_count == second_count
    assert first_count >= 20
