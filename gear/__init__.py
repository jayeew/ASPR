"""ASPR-GEAR: evidence-traceable, five-part paper review."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import ReviewRequest
from .review_contracts import ReviewBundle

if TYPE_CHECKING:
    from .config import GearConfig
    from .review_pipeline import ServiceRegistry


def review_paper(
    request: ReviewRequest,
    *,
    output_dir: Path | None = None,
    config: GearConfig | None = None,
    services: ServiceRegistry | None = None,
) -> ReviewBundle:
    """Lazily import the current pipeline to keep package import side-effect free."""
    from .review_pipeline import review_paper

    return review_paper(
        request,
        output_dir=output_dir,
        config=config,
        services=services,
    )


__all__ = ["review_paper"]
