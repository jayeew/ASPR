from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


OVERCLAIM_TERMS = [
    "breakthrough",
    "paradigm",
    "revolutionary",
    "disruptive",
    "颠覆",
    "范式",
    "革命",
    "开创",
    "突破",
]

MECHANISM_MAP = {
    "DeltaQ0": "社区边界扰动 / boundary perturbation",
    "Uzzi": "非典型组合 / atypical recombination",
    "RS": "跨学科知识广度 / interdisciplinary breadth",
    "PDE": "潜在扩散广度 / prospective diffusion",
    "B": "跨社区桥接 / bridge position",
    "RTD": "参考目标社区多样性 / reference target diversity",
    "BurtIP": "结构洞潜力 / structural-hole potential",
}


class ClaimCard(BaseModel):
    """A claim-level contract linking novelty claims to evidence and caveats."""

    claim: str = Field(description="Candidate novelty claim.")
    supporting_references: List[int] = Field(default_factory=list, description="Related paper indices supporting comparison.")
    evidence_summary: str = Field(default="", description="Textual evidence and differences against related work.")
    graph_support: str = Field(default="", description="Graph-mechanism support or limitation.")
    counterarguments: List[str] = Field(default_factory=list, description="Skeptical counterclaims or missing evidence.")
    uncertainty: str = Field(default="medium", description="low | medium | high")

    @field_validator("uncertainty", mode="before")
    @classmethod
    def normalize_uncertainty(cls, value: Any) -> str:
        text = str(value or "medium").lower()
        if text in {"low", "medium", "high"}:
            return text
        return "medium"


class AgentReview(BaseModel):
    """One committee member's structured review."""

    agent_name: str
    score: float = Field(ge=0.0, le=10.0)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    required_revisions: List[str] = Field(default_factory=list)

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, value: Any) -> float:
        try:
            return max(0.0, min(float(value), 10.0))
        except (TypeError, ValueError):
            return 0.0


class CommitteeReport(BaseModel):
    """Final reviewer committee output."""

    claim_cards: List[ClaimCard] = Field(default_factory=list)
    agent_reviews: List[AgentReview] = Field(default_factory=list)
    disagreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    meta_review_summary: str = ""
    recommended_tone: str = Field(default="balanced", description="conservative | balanced | assertive")

    @field_validator("recommended_tone", mode="before")
    @classmethod
    def normalize_tone(cls, value: Any) -> str:
        text = str(value or "balanced").lower()
        if text in {"conservative", "balanced", "assertive"}:
            return text
        return "balanced"

    def to_prompt_block(self) -> str:
        """Render a compact committee evidence block for prompts."""
        claim_lines = []
        for idx, card in enumerate(self.claim_cards, start=1):
            refs = ", ".join(f"[{item}]" for item in card.supporting_references) or "无"
            counters = "; ".join(card.counterarguments) or "暂无强反驳"
            claim_lines.append(
                f"{idx}. Claim: {card.claim}\n"
                f"   References: {refs}\n"
                f"   Evidence: {card.evidence_summary}\n"
                f"   Graph: {card.graph_support}\n"
                f"   Counterarguments: {counters}\n"
                f"   Uncertainty: {card.uncertainty}"
            )
        review_lines = [
            f"- {review.agent_name}: score={review.score:.1f}; "
            f"strengths={'; '.join(review.strengths) or '无'}; "
            f"weaknesses={'; '.join(review.weaknesses) or '无'}"
            for review in self.agent_reviews
        ]
        return (
            "【审稿委员会证据 / Reviewer Committee Evidence】\n"
            f"Recommended tone: {self.recommended_tone}\n"
            f"Disagreement score: {self.disagreement_score:.3f} / 1.000\n"
            f"Meta-review: {self.meta_review_summary}\n"
            "Claim-Evidence-Graph-Counterclaim cards:\n"
            + ("\n".join(claim_lines) if claim_lines else "无可用 claim cards")
            + "\nAgent reviews:\n"
            + ("\n".join(review_lines) if review_lines else "无 agent reviews")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return self.model_dump()


class ClaimDecomposer:
    """Extract candidate novelty claims from title and abstract."""

    claim_patterns = [
        r"\bwe (?:propose|present|introduce|develop|show|demonstrate|find|identify|report)\b",
        r"\bthis (?:paper|study|work) (?:proposes|presents|introduces|develops|shows|demonstrates|reports)\b",
    ]

    def run(self, paper_title: str, paper_abstract: str) -> tuple[List[str], AgentReview]:
        sentences = self._split_sentences(paper_abstract)
        claims = [
            sentence
            for sentence in sentences
            if any(re.search(pattern, sentence, flags=re.I) for pattern in self.claim_patterns)
        ]
        if not claims:
            claims = sentences[:3]
        claims = [self._clean_claim(sentence) for sentence in claims if self._clean_claim(sentence)]
        if paper_title and len(claims) < 2:
            claims.insert(0, f"论文围绕“{paper_title.strip()}”提出或验证新的研究贡献。")
        claims = list(dict.fromkeys(claims))[:5]
        if len(claims) > 2:
            score = 8.0
        elif claims:
            score = 6.5
        else:
            score = 2.0
        return claims, AgentReview(
            agent_name="ClaimDecomposer",
            score=score,
            strengths=[f"识别到 {len(claims)} 个候选创新声明。"] if claims else [],
            weaknesses=[] if claims else ["摘要中缺少明确创新声明。"],
            required_revisions=[] if claims else ["需要人工补充论文的核心创新声明。"],
        )

    def _split_sentences(self, text: str) -> List[str]:
        chunks = re.split(r"(?<=[.!?。！？])\s+", str(text or "").strip())
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]

    def _clean_claim(self, sentence: str) -> str:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        return sentence[:420].rstrip()


class EvidenceMapper:
    """Map each claim to related papers through token overlap."""

    def run(self, claims: List[str], related_papers: List[Dict[str, Any]]) -> tuple[List[ClaimCard], AgentReview]:
        cards: List[ClaimCard] = []
        for claim in claims:
            ranked = self._rank_references(claim, related_papers)
            selected = [idx for idx, score in ranked[:3] if score > 0.0]
            summary = self._summary_for_claim(claim, selected, related_papers)
            uncertainty = "high" if not selected else ("medium" if len(selected) < 2 else "low")
            counterarguments = []
            if not selected:
                counterarguments.append("当前检索结果中没有找到与该 claim 明显相关的对比文献。")
            cards.append(
                ClaimCard(
                    claim=claim,
                    supporting_references=selected,
                    evidence_summary=summary,
                    counterarguments=counterarguments,
                    uncertainty=uncertainty,
                )
            )
        mapped = sum(1 for card in cards if card.supporting_references)
        score = 2.0 + 8.0 * (mapped / max(len(cards), 1))
        return cards, AgentReview(
            agent_name="EvidenceMapper",
            score=score,
            strengths=[f"{mapped}/{len(cards)} 个 claim 已绑定相关文献证据。"] if cards else [],
            weaknesses=[] if mapped == len(cards) and cards else ["部分 claim 缺少明确对比文献。"],
            required_revisions=[] if mapped == len(cards) and cards else ["补充检索或扩大相关工作集合。"],
        )

    def _rank_references(self, claim: str, related_papers: List[Dict[str, Any]]) -> List[tuple[int, float]]:
        claim_tokens = self._tokens(claim)
        ranked = []
        for idx, paper in enumerate(related_papers):
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')} {' '.join(paper.get('fieldsOfStudy') or [])}"
            score = self._jaccard(claim_tokens, self._tokens(paper_text))
            ranked.append((idx, score))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _summary_for_claim(self, claim: str, selected: List[int], related_papers: List[Dict[str, Any]]) -> str:
        if not selected:
            return "未能从当前相关论文集合中建立稳定的 claim-reference 对应关系。"
        titles = [str(related_papers[idx].get("title") or f"Reference {idx}")[:120] for idx in selected]
        return f"该 claim 与 {', '.join(f'[{idx}] {title}' for idx, title in zip(selected, titles))} 存在主题重叠，应重点比较方法、对象或应用场景差异。"

    def _tokens(self, text: str) -> set[str]:
        stopwords = {"the", "and", "for", "with", "from", "that", "this", "paper", "study", "work", "using"}
        return {
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(text).lower())
            if token not in stopwords
        }

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)


class GraphAnalyst:
    """Explain how graph metrics support or limit claim-level novelty."""

    def run(self, cards: List[ClaimCard], graph_metric_result: Dict[str, Any]) -> tuple[List[ClaimCard], AgentReview]:
        metrics = graph_metric_result.get("metrics") or {}
        confidence = float(graph_metric_result.get("confidence") or 0.0)
        top_metrics = sorted(metrics.items(), key=lambda item: float(item[1] or 0.0), reverse=True)[:3]
        mechanism_text = self._mechanism_text(top_metrics, confidence)
        for card in cards:
            card.graph_support = mechanism_text
            if confidence < 0.45 and card.uncertainty != "high":
                card.uncertainty = "medium"
                card.counterarguments.append("图谱证据置信度偏低，结构性创新解释需保守。")
        score = min(10.0, 2.0 + 8.0 * confidence)
        if any(key in {"DeltaQ0", "Uzzi"} and float(value or 0.0) >= 0.35 for key, value in top_metrics):
            strengths = ["图谱信号支持讨论边界扰动或非典型组合。"]
        elif top_metrics:
            strengths = ["图谱信号可作为创新性判断的辅助证据。"]
        else:
            strengths = []
        weaknesses = [] if confidence >= 0.45 else ["图谱证据置信度偏低。"]
        return cards, AgentReview(
            agent_name="GraphAnalyst",
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            required_revisions=[] if confidence >= 0.45 else ["需要更完整的 reference graph 或 DOI/OpenAlex 图谱验证。"],
        )

    def _mechanism_text(self, top_metrics: List[tuple[str, float]], confidence: float) -> str:
        if not top_metrics:
            return "七维图谱指标不可用，不能提供结构性支持。"
        parts = [
            f"{key}={float(value or 0.0):.2f} 对应 {MECHANISM_MAP.get(key, key)}"
            for key, value in top_metrics
            if float(value or 0.0) >= 0.10
        ]
        if not parts:
            return "七维指标整体较弱，图谱证据不支持强创新性表述。"
        prefix = "图谱证据较可靠" if confidence >= 0.60 else "图谱证据需保守使用"
        return f"{prefix}；主要机制为：" + "；".join(parts) + "。"


class SkepticReviewer:
    """Adversarial reviewer focused on overclaiming and missing controls."""

    def run(self, cards: List[ClaimCard], related_papers: List[Dict[str, Any]], graph_metric_result: Dict[str, Any]) -> tuple[List[ClaimCard], AgentReview]:
        weaknesses = []
        revisions = []
        graph_confidence = float(graph_metric_result.get("confidence") or 0.0)
        for card in cards:
            claim_lower = card.claim.lower()
            if any(term.lower() in claim_lower for term in OVERCLAIM_TERMS):
                card.counterarguments.append("该 claim 含强创新措辞，需要更强对比证据支撑。")
            if not card.supporting_references:
                card.counterarguments.append("缺少直接相关参考文献，无法排除已有工作已覆盖类似贡献。")
            if graph_confidence < 0.45:
                card.counterarguments.append("当前图谱证据不足以支撑强机制结论。")
            if card.counterarguments:
                weaknesses.append(f"Claim 需要审慎处理：{card.claim[:80]}")
        if not related_papers:
            revisions.append("没有相关论文输入，不能形成可信审稿委员会判断。")
        if graph_confidence < 0.45:
            revisions.append("补充更多相关文献或正式 OpenAlex reference graph 后再声称结构性创新。")
        skeptic_load = sum(len(card.counterarguments) for card in cards)
        score = max(0.0, 9.0 - skeptic_load * 0.8 - (2.0 if not related_papers else 0.0))
        return cards, AgentReview(
            agent_name="SkepticReviewer",
            score=score,
            strengths=["已完成反方审查。"],
            weaknesses=weaknesses[:5],
            required_revisions=revisions,
        )


class MetaReviewer:
    """Aggregate committee members and calibrate the final tone."""

    def run(
        self,
        cards: List[ClaimCard],
        agent_reviews: List[AgentReview],
        graph_metric_result: Dict[str, Any],
    ) -> CommitteeReport:
        disagreement = self._disagreement(agent_reviews, cards, graph_metric_result)
        tone = self._recommended_tone(disagreement, cards, agent_reviews, graph_metric_result)
        summary = self._summary(cards, tone, disagreement, graph_metric_result)
        meta_score = max(0.0, min(10.0, 8.5 - disagreement * 4.0))
        agent_reviews = agent_reviews + [
            AgentReview(
                agent_name="MetaReviewer",
                score=meta_score,
                strengths=[f"完成多 agent 仲裁，建议语气为 {tone}。"],
                weaknesses=[] if disagreement < 0.45 else ["委员会分歧较高，最终报告需保守。"],
                required_revisions=[] if tone != "conservative" else ["最终报告必须显式说明证据限制和不确定性。"],
            )
        ]
        return CommitteeReport(
            claim_cards=cards,
            agent_reviews=agent_reviews,
            disagreement_score=disagreement,
            meta_review_summary=summary,
            recommended_tone=tone,
        )

    def _disagreement(
        self,
        agent_reviews: List[AgentReview],
        cards: List[ClaimCard],
        graph_metric_result: Dict[str, Any],
    ) -> float:
        scores = [review.score for review in agent_reviews]
        if len(scores) < 2:
            score_spread = 0.0
        else:
            score_spread = (max(scores) - min(scores)) / 10.0
        high_uncertainty = sum(1 for card in cards if card.uncertainty == "high") / max(len(cards), 1)
        counter_density = sum(len(card.counterarguments) for card in cards) / max(len(cards) * 3, 1)
        graph_penalty = 1.0 - float(graph_metric_result.get("confidence") or 0.0)
        disagreement = 0.35 * score_spread + 0.25 * high_uncertainty + 0.25 * min(counter_density, 1.0) + 0.15 * graph_penalty
        return max(0.0, min(disagreement, 1.0))

    def _recommended_tone(
        self,
        disagreement: float,
        cards: List[ClaimCard],
        agent_reviews: List[AgentReview],
        graph_metric_result: Dict[str, Any],
    ) -> str:
        graph_score = float(graph_metric_result.get("weighted_score") or 0.0)
        confidence = float(graph_metric_result.get("confidence") or 0.0)
        skeptic = next((review for review in agent_reviews if review.agent_name == "SkepticReviewer"), None)
        skeptic_score = skeptic.score if skeptic else 5.0
        if disagreement >= 0.45 or confidence < 0.45 or any(card.uncertainty == "high" for card in cards):
            return "conservative"
        if graph_score >= 0.55 and confidence >= 0.65 and skeptic_score >= 7.0:
            return "assertive"
        return "balanced"

    def _summary(
        self,
        cards: List[ClaimCard],
        tone: str,
        disagreement: float,
        graph_metric_result: Dict[str, Any],
    ) -> str:
        n_claims = len(cards)
        mapped = sum(1 for card in cards if card.supporting_references)
        confidence = float(graph_metric_result.get("confidence") or 0.0)
        return (
            f"委员会识别 {n_claims} 个创新声明，其中 {mapped} 个绑定了相关文献证据；"
            f"图谱证据置信度为 {confidence:.2f}，委员会分歧为 {disagreement:.2f}。"
            f"最终建议采用 {tone} 语气，并按 claim 逐条呈现证据、图谱机制、反方质疑和不确定性。"
        )


class ReviewerCommittee:
    """Coordinate the five reviewer agents into one committee report."""

    def __init__(self) -> None:
        self.claim_decomposer = ClaimDecomposer()
        self.evidence_mapper = EvidenceMapper()
        self.graph_analyst = GraphAnalyst()
        self.skeptic = SkepticReviewer()
        self.meta_reviewer = MetaReviewer()

    def run(
        self,
        paper_title: str,
        paper_abstract: str,
        related_papers: List[Dict[str, Any]],
        graph_metric_result: Optional[Dict[str, Any]] = None,
    ) -> CommitteeReport:
        graph_metric_result = graph_metric_result or {}
        claims, claim_review = self.claim_decomposer.run(paper_title, paper_abstract)
        cards, evidence_review = self.evidence_mapper.run(claims, related_papers)
        cards, graph_review = self.graph_analyst.run(cards, graph_metric_result)
        cards, skeptic_review = self.skeptic.run(cards, related_papers, graph_metric_result)
        return self.meta_reviewer.run(
            cards=cards,
            agent_reviews=[claim_review, evidence_review, graph_review, skeptic_review],
            graph_metric_result=graph_metric_result,
        )


def run_reviewer_committee(
    paper_title: str,
    paper_abstract: str,
    related_papers: List[Dict[str, Any]],
    graph_metric_result: Optional[Dict[str, Any]] = None,
) -> CommitteeReport:
    """Convenience entrypoint for the graph-grounded reviewer committee."""
    return ReviewerCommittee().run(
        paper_title=paper_title,
        paper_abstract=paper_abstract,
        related_papers=related_papers,
        graph_metric_result=graph_metric_result,
    )
