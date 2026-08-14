"""ASPR-GEAR: evidence-traceable, five-part paper review."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .contracts import ReviewRequest
from .review_contracts import ReviewBundle

if TYPE_CHECKING:
    from .config import GearConfig
    from .review_pipeline import ServiceRegistry


def review_paper(
    request: ReviewRequest,
    *,
    output_dir: Optional[Path] = None,
    config: Optional[GearConfig] = None,
    services: Optional[ServiceRegistry] = None,
) -> ReviewBundle:
    """Lazily import the current pipeline to keep package import side-effect free."""
    from .review_pipeline import review_paper

    return review_paper(
        request,
        output_dir=output_dir,
        config=config,
        services=services,
    )


def load_calibration_release(identifier: str = "prepublication_graph_v3:d5_fulltext16"):
    """Load a frozen local calibration release for GEAR or another experiment."""
    from .calibration_assets import load_calibration_release as _load_release

    return _load_release(identifier)


__all__ = ["load_calibration_release", "review_paper"]
