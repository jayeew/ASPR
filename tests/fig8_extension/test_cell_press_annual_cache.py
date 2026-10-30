from pathlib import Path

import pytest

from experiments.fig8_extension.cell_press_annual_cache import apply_proposals, title_rows


def test_out_of_scope_year_is_rejected_before_any_write(tmp_path: Path) -> None:
    for year in (2022, 2026):
        with pytest.raises(ValueError, match='restricted to 2023'):
            apply_proposals([{'year': year}], tmp_path, [])
    assert not list(tmp_path.iterdir())


def test_dated_title_tables_preserve_issn_title_pairing():
    assert title_rows('2451-9456 Cell Chemical Biology\nISSN Journal Title\n1535-6108 Cancer Cell', 2024) == [
        ('2451-9456', 'Cell Chemical Biology'), ('1535-6108', 'Cancer Cell')]
    assert title_rows('10Chem 2451-9294 HYBRID CP/TL\n11Chem Catalysis 2667-1093 HYBRID CP/TL', 2025) == [
        ('2451-9294', 'Chem'), ('2667-1093', 'Chem Catalysis')]
