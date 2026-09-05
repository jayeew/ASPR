"""Primary runtime for Claim Graph + full-text GEAR innovation evaluation.

The old paper-level Graph attribution and generic reviewer/fusion runtime are no
longer reachable from this module. Stages exchange only fixed disk artifacts.
"""

from __future__ import annotations

from pathlib import Path

from .config import GearConfig, load_config
from .review_contracts import InnovationPaperInput
from .review_fusion import BranchFusion
from .review_compiler import run_gear_branch
from .claim_attribution import run_graph_branch
from .artifacts import read_model


DEFAULT_GRAPH_ROOT = Path("data/claim_graph")
DEFAULT_EMBEDDING_MODEL = Path("data/models/Qwen3-Embedding-4B")


def review_paper(
    input_contract: Path | InnovationPaperInput,
    *,
    output_dir: Path,
    config: GearConfig | None = None,
    stage: str = "all",
    fusion_mode: str = "passive",
    graph_root: Path = DEFAULT_GRAPH_ROOT,
    embedding_model: Path = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, str]:
    """Run all stages or one independently addressable stage."""
    resolved = config or load_config()
    item = (
        input_contract
        if isinstance(input_contract, InnovationPaperInput)
        else read_model(input_contract, InnovationPaperInput)
    )
    outputs: dict[str, str] = {}
    if stage in {"all", "graph"}:
        run_graph_branch(item, output_dir, resolved, graph_root, embedding_model)
        outputs["graph"] = str(output_dir / "graph" / "graph_branch_result.json")
    if stage in {"all", "gear"}:
        run_gear_branch(item, output_dir, resolved)
        outputs["gear"] = str(output_dir / "gear" / "gear_branch_result.json")
    if stage in {"all", "fusion"}:
        BranchFusion(resolved).run(output_dir, fusion_mode)
        outputs["fusion"] = str(output_dir / "fusion" / "fusion_result.json")
    return outputs


run_innovation_review = review_paper

__all__ = ["review_paper", "run_innovation_review"]
