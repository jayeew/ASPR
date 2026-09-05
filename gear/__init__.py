"""ASPR-GEAR: Claim Graph plus evidence-traceable innovation evaluation."""

from __future__ import annotations

from pathlib import Path

from .config import GearConfig
from .review_contracts import InnovationPaperInput


def review_paper(
    input_contract: Path | InnovationPaperInput,
    *,
    output_dir: Path,
    config: GearConfig | None = None,
    stage: str = "all",
    fusion_mode: str = "passive",
) -> dict[str, str]:
    """Lazily invoke the current innovation-only runtime."""
    from .review_pipeline import review_paper as run

    return run(
        input_contract,
        output_dir=output_dir,
        config=config,
        stage=stage,
        fusion_mode=fusion_mode,
    )


__all__ = ["review_paper"]
