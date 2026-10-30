from types import SimpleNamespace

import pytest

from experiments.fig8_extension import pmc_cloud_versions as cloud


def test_comparison_journals_dispatch_first_without_excluding_other_journals() -> None:
    assert cloud.journal_priority('https://openalex.org/S3880285', 'Science') == 0
    assert cloud.journal_priority('https://openalex.org/S110447773', 'Cell') == 1
    assert cloud.journal_priority('https://openalex.org/S125754415', 'PNAS') == 2
    assert cloud.journal_priority('other', 'Nature Communications') == 3
    assert cloud.journal_priority('other', 'Scientific Reports', nature_portfolio=True) == 3
    assert cloud.journal_priority('other', 'Other journal') == 4


@pytest.fixture
def fake_source(tmp_path, monkeypatch):
    folder = tmp_path / 'cloud'; folder.mkdir()
    monkeypatch.setattr(cloud, 'ROOT', tmp_path)
    monkeypatch.setattr(cloud, 'FOLDER', folder)
    monkeypatch.setattr(cloud.time, 'sleep', lambda _: None)
    cloud.STOP.clear()
    calls = []
    def install(status: int) -> None:
        def get(url: str, **kwargs):
            calls.append((url, kwargs))
            return SimpleNamespace(status_code=status, headers={}, content=b'{}')
        monkeypatch.setattr(cloud.LOCAL, 'session', SimpleNamespace(get=get), raising=False)
    yield install, calls
    cloud.STOP.clear()


def test_permission_failure_not_retried_and_cache_reused(fake_source) -> None:
    install, calls = fake_source; install(403)
    first = cloud.fetch_metadata('metadata/PMC1.1.json')
    assert first['http_status'] == 403
    assert cloud.STOP.is_set()
    assert cloud.fetch_metadata('metadata/PMC1.1.json') == first
    assert len(calls) == 1
    assert cloud.fetch_metadata('metadata/PMC2.1.json')['status'] == 'not_requested_after_source_stop'
    assert len(calls) == 1


def test_transient_retries_are_bounded(fake_source) -> None:
    install, calls = fake_source; install(503)
    result = cloud.fetch_metadata('metadata/PMC1.1.json')
    assert len(calls) == len(result['attempts']) == 3
    assert cloud.STOP.is_set()


def test_success_is_not_requested_twice(fake_source) -> None:
    install, calls = fake_source; install(200)
    assert cloud.fetch_metadata('metadata/PMC1.1.json')['http_status'] == 200
    assert cloud.fetch_metadata('metadata/PMC1.1.json')['http_status'] == 200
    assert len(calls) == 1
