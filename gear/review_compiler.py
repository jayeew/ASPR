"""Compile the independent full-text GEAR innovation branch."""

from __future__ import annotations

from pathlib import Path

from gear.config import GearConfig
from gear.contracts import PaperMetadata, ReviewRequest
from gear.paper_compiler import PaperCompiler
from gear.trace import EvidenceStore

from .review_contracts import BranchStatus, GearBranchResult, InnovationPaperInput
from .evidence_supervisor import EvidenceSupervisor
from .grounding import FullTextClaimMiner
from gear.artifacts import write_jsonl, write_model


def run_gear_branch(
    item: InnovationPaperInput,
    output_dir: Path,
    config: GearConfig,
    *,
    graph_seed_work_ids: dict[str, list[str]] | None = None,
) -> GearBranchResult:
    """Compile full text, mine claims and verify prior art; never reads Graph facts by default."""
    branch_dir = output_dir / "gear"
    branch_dir.mkdir(parents=True, exist_ok=True)
    for name in ("evidence_trace.jsonl", "action_trace.jsonl", "state_trace.jsonl"):
        (branch_dir / name).unlink(missing_ok=True)
    write_model(output_dir / "innovation_input.json", item)
    try:
        request = ReviewRequest(
            paper_path=item.paper_path,
            metadata=PaperMetadata(
                title=item.title, doi=item.doi, openalex_id=item.openalex_work_id,
                publication_date=item.publication_date, venue=item.venue,
            ),
            evaluation_date=item.cutoff_date,
        )
        paper = PaperCompiler(config).compile(request)
        write_model(branch_dir / "paper_ir.json", paper)
        claims = FullTextClaimMiner(config).extract(paper)
        store = EvidenceStore(branch_dir)
        supervisor = EvidenceSupervisor(config, store)
        cards = [
            supervisor.evaluate(
                claim, paper, item.cutoff_date,
                seed_work_ids=(graph_seed_work_ids or {}).get(claim.claim_id, []),
            )
            for claim in claims
        ]
        write_jsonl(branch_dir / "gear_claims.jsonl", claims)
        write_jsonl(branch_dir / "gear_claim_cards.jsonl", cards)
        status = BranchStatus.LIMITED if any(card.limitations for card in cards) else BranchStatus.COMPLETE
        result = GearBranchResult(paper_id=item.paper_id, status=status, claims=claims, claim_cards=cards)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        result = GearBranchResult(paper_id=item.paper_id, status=BranchStatus.LIMITED, limitations=[str(exc)])
    result.output_files = {
        "claims": str(branch_dir / "gear_claims.jsonl"),
        "claim_cards": str(branch_dir / "gear_claim_cards.jsonl"),
        "evidence_trace": str(branch_dir / "evidence_trace.jsonl"),
    }
    write_model(branch_dir / "gear_branch_result.json", result)
    return result
