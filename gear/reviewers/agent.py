"""Required primary scientific Agent Reviewer implementations."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import ValidationError

from ..codex_critic import CodexCliCritic, limited_review
from ..config import GearConfig, load_config
from ..contracts import PaperIR
from ..model_client import (
    JsonModelClient,
    ModelClientUnavailableError,
    build_json_model_client,
)
from ..review_contracts import (
    BranchReview,
    PaperSpecificRubric,
    ReviewSource,
    StructuredReview,
)
from ..trace import sha256_value
from .base import build_graph_blind_payload

AGENT_REVIEW_PROMPT = """You are the required paper-internal ASPR-ESR Agent Reviewer.
Read only the supplied manuscript spans and paper-specific rubric. Return one
StructuredReview JSON object with summary, novelty, strengths, weaknesses, and
questions. Every major point must cite P:S-* evidence. Mark novelty hypotheses
external_verification_required=true. Do not use model memory as prior-art proof.
Do not output acceptance, rejection, ratings, Graph information, human reviews,
or claims that a work is definitively first."""


class CodexAgentReviewer:
    def __init__(
        self,
        config: Optional[GearConfig] = None,
        *,
        generator: Optional[Callable[[str, str], Mapping[str, Any]]] = None,
        client: Optional[JsonModelClient] = None,
    ) -> None:
        self.config = config or load_config()
        self.client = client or build_json_model_client(self.config)
        self.generator = generator
        self.model_name = self.config.codex_cli.model
        if self.config.model_backend == "openai_compatible":
            endpoint = self.config.openai_compatible
            self.model_name = (
                endpoint.model if endpoint is not None else self.model_name
            )
        self.last_failures: List[str] = []
        self.last_payload: Dict[str, Any] = {}

    def review(self, paper_ir: PaperIR, rubric: PaperSpecificRubric) -> BranchReview:
        self.last_failures = []
        self.last_payload = build_graph_blind_payload(paper_ir, rubric)
        prompt_hash = _hash_text(AGENT_REVIEW_PROMPT)
        input_hash = sha256_value(self.last_payload)
        user = json.dumps(
            {
                "context": self.last_payload,
                "output_schema": StructuredReview.model_json_schema(),
            },
            ensure_ascii=False,
        )
        try:
            payload = self._generate(AGENT_REVIEW_PROMPT, user)
            structured = StructuredReview.model_validate(payload.get("review", payload))
            CodexCliCritic._force_novelty_external_verification(structured)
            CodexCliCritic._validate_review(structured, paper_ir)
        except (
            ModelClientUnavailableError,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            self.last_failures.append(f"agent_reviewer_unavailable_or_invalid:{exc}")
            structured = limited_review(paper_ir)
        return BranchReview.from_structured(
            structured,
            source=ReviewSource.AGENT,
            model_id=self.model_name,
            prompt_sha256=prompt_hash,
            input_sha256=input_hash,
            failures=self.last_failures,
        )

    def _generate(self, system: str, user: str) -> Mapping[str, Any]:
        if self.generator is not None:
            return self.generator(system, user)
        return self.client.generate_json(
            system=system,
            user=user,
            response_schema=StructuredReview.model_json_schema(),
        )


class OpenAICompatibleAgentReviewer(CodexAgentReviewer):
    """Named adapter for deployments selecting an OpenAI-compatible endpoint."""


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "AGENT_REVIEW_PROMPT",
    "CodexAgentReviewer",
    "OpenAICompatibleAgentReviewer",
]
