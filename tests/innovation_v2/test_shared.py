from datetime import date

import pytest

from gear.artifacts import write_model
from gear.config import GearConfig
from gear.innovation.contracts import ClaimSet
from gear.innovation.shared import prepare_shared
from gear.review_contracts import InnovationPaperInput


def test_resume_refuses_changed_input(tmp_path):
    paper = tmp_path / "paper.md"
    paper.write_text("Changed manuscript")
    item = InnovationPaperInput(
        paper_id="p",
        paper_path=paper,
        title="t",
        publication_date=date(2026, 1, 1),
        cutoff_date=date(2026, 1, 1),
        abstract_text="a",
        abstract_source="test",
    )
    write_model(
        tmp_path / "shared" / "claims.json",
        ClaimSet(paper_id="p", input_fingerprint="old", claims=[]),
    )
    with pytest.raises(ValueError, match="changed"):
        prepare_shared(item, tmp_path, GearConfig())
