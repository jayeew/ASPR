import gzip
from pathlib import Path

from experiments.fig8_extension.pmc_journal_harvest import retain_front


def test_resume_reuses_unparsed_raw_without_overwriting_conflict(tmp_path: Path) -> None:
    path = tmp_path / 'front.xml.gz'
    raw = b'<article><front/></article>'
    assert retain_front(path, raw)
    saved = path.read_bytes()
    assert retain_front(path, raw)
    assert path.read_bytes() == saved
    assert not retain_front(path, b'<article><front>different</front></article>')
    assert gzip.decompress(path.read_bytes()) == raw
