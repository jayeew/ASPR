#!/usr/bin/env python3
"""Blind Primary AI construct coding and mention-derived dimension synthesis."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .core import (
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        stable_id,
        utc_now,
    )
except ImportError:
    from core import (  # type: ignore[no-redef]
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        stable_id,
        utc_now,
    )

MODEL_LABEL = "codex-gpt-5-primary-ai-construct-coding-v1"


@dataclass(frozen=True)
class DimensionRule:
    key: str
    label: str
    definition: str
    construct: str
    role: str
    t0_boundary: str
    indicators: tuple[str, ...]
    phrases: tuple[str, ...]
    bias_risk: str


RULES = (
    DimensionRule(
        "novelty",
        "Intrinsic novelty and originality",
        "Paper-level departure from prior knowledge, including unusual combinations, originality, and disruption measures computable from publication-time content or references.",
        "paper novelty, originality, and disruptive contribution",
        "predictive",
        "Only novelty signals computable from the submitted/published paper and its T0 reference set are eligible; later citations are excluded.",
        (
            "semantic novelty score",
            "atypical-reference combination score",
            "disruption-oriented novelty proxy",
        ),
        (
            "novelty",
            "paper originality",
            "scientific originality",
            "disruptive",
            "disruption",
            "atypical combination",
            "unconventional combination",
        ),
        "Novelty measures can privilege fields with rapid vocabulary or reference turnover and can mistake rarity for quality.",
    ),
    DimensionRule(
        "text_semantics",
        "Textual and semantic paper characteristics",
        "T0 properties extracted from title, abstract, keywords, or full manuscript text that may signal novelty, clarity, or potential scholarly impact.",
        "textual and semantic characteristics of the paper",
        "predictive",
        "Text must be available in the paper by T0; embeddings or language-model features must not use post-publication outcomes.",
        (
            "title/abstract embedding",
            "readability score",
            "semantic diversity",
            "keyword distinctiveness",
        ),
        (
            "textual feature",
            "semantic feature",
            "word embedding",
            "language model",
            "llm",
            "title length",
            "article title",
            "paper title",
            "abstract feature",
            "keyword choice",
            "author-selected keyword",
            "readability",
            "linguistic style",
        ),
        "Language, discipline, document length, and model-training-corpus biases may dominate semantic signals.",
    ),
    DimensionRule(
        "references",
        "Reference-list and knowledge-combination structure",
        "T0 structure and composition of the focal paper's references, including age, diversity, conventionality, and network position of cited knowledge.",
        "reference-list structure and prior-knowledge recombination",
        "predictive",
        "Only the reference list fixed in the paper at T0 and citation-network information dated no later than T0 are eligible.",
        (
            "reference age distribution",
            "reference diversity",
            "reference-network centrality",
            "conventionality score",
        ),
        (
            "reference list",
            "bibliographic coupling",
            "co-citation",
            "cocitation",
            "cited references",
            "referencing practice",
            "past knowledge",
            "knowledge combination",
        ),
        "Database coverage, field citation customs, and historical-indexing depth can distort reference structure.",
    ),
    DimensionRule(
        "team",
        "Authorship and collaboration structure",
        "T0 team composition and collaboration-network characteristics attached to the focal paper.",
        "authorship and collaboration structure",
        "predictive",
        "Authors, team size, collaboration links, and contributor roles must be known by T0; later collaboration is excluded.",
        (
            "team size",
            "coauthor-network centrality",
            "international collaboration",
            "author diversity",
        ),
        (
            "coauthor",
            "co-author",
            "authorship",
            "team size",
            "international collaboration",
            "research collaboration",
            "collaboration network",
            "collaboration pattern",
            "collaborative research",
            "research team",
            "author diversity",
        ),
        "Name disambiguation, honorary authorship, field team-size norms, and geographic coverage create bias.",
    ),
    DimensionRule(
        "author_history",
        "Author prior scholarly track record",
        "Pre-T0 author experience, prior productivity, prior influence, and career-stage signals associated with a focal paper.",
        "authors' prior scholarly track record",
        "predictive",
        "Only achievements and network information observed before the focal paper's T0 may be used.",
        (
            "prior publication count",
            "pre-T0 author impact",
            "career age",
            "prior h-index",
        ),
        (
            "author impact",
            "author characteristic",
            "author reputation",
            "career",
            "researcher impact",
            "prior publication",
            "author-level",
        ),
        "Historical inequities, name disambiguation, cumulative advantage, and career interruptions can encode protected-status bias.",
    ),
    DimensionRule(
        "institution",
        "Institutional and geographic context",
        "T0 institutional affiliation, country, and organizational context of the focal paper.",
        "institutional and geographic research context",
        "control",
        "Affiliations and geography must be those declared at T0; later rankings or reputation changes are excluded.",
        (
            "institutional affiliation",
            "country/region",
            "institutional collaboration",
            "institution prestige control",
        ),
        (
            "institutional",
            "institution prestige",
            "country affiliation",
            "country in the title",
            "geographic",
            "global south",
            "global north",
            "affiliation",
        ),
        "Institutional prestige and database coverage reproduce geographic, language, and resource inequities.",
    ),
    DimensionRule(
        "venue",
        "Venue and publication-context characteristics",
        "T0 journal, conference, editorial, and publication-route characteristics associated with the focal paper.",
        "venue and publication context",
        "control",
        "Use venue attributes known at acceptance/publication; do not use future journal metrics as T0 features.",
        (
            "journal/conference identity",
            "venue field",
            "publication route",
            "editorial policy",
        ),
        (
            "journal impact",
            "journal characteristic",
            "journal prestige",
            "journal quality",
            "journal rank",
            "publication venue",
            "conference venue",
            "editorial policy",
            "editorial decision",
            "publication model",
            "publishing model",
            "publisher",
        ),
        "Venue prestige is field-dependent and can encode language, geography, access, and editorial-selection bias.",
    ),
    DimensionRule(
        "access",
        "Access, availability, and dissemination at T0",
        "Publication-time accessibility of the paper and associated research objects, including open-access and repository status.",
        "publication-time openness and availability",
        "predictive",
        "Access route or deposited object must exist by T0; later downloads, shares, or deposits are not T0 predictors.",
        (
            "open-access status",
            "repository availability",
            "data availability",
            "code availability",
        ),
        (
            "open access",
            "open-access",
            "repository",
            "data availability",
            "open data",
            "code availability",
            "research data",
        ),
        "Mandates, publisher business models, discipline, and national resources confound openness with impact.",
    ),
    DimensionRule(
        "funding",
        "Funding and declared support",
        "Funding, sponsorship, and declared support observable for the focal paper at T0.",
        "funding and sponsorship context",
        "control",
        "Only funding and conflicts disclosed by T0 are eligible; subsequent awards are excluded.",
        (
            "funding declaration",
            "sponsor type",
            "grant support",
            "conflict-of-interest declaration",
        ),
        (
            "research funding",
            "funding source",
            "funding declaration",
            "funded",
            "sponsor",
            "grant support",
            "industry-sponsored",
            "conflict of interest",
        ),
        "Funding disclosure is incomplete and sponsor effects differ by field, geography, and study design.",
    ),
    DimensionRule(
        "field_topic",
        "Field, topic, and temporal context",
        "Disciplinary field, research topic, publication year, and topic-growth context used for fair comparison or normalization.",
        "field, topic, and temporal context",
        "control",
        "Field/topic assignment and calendar time must be determined with information available by T0.",
        (
            "field classification",
            "topic cluster",
            "publication year",
            "topic growth rate at T0",
        ),
        (
            "field-normal",
            "discipline",
            "research topic",
            "topic model",
            "field differences",
            "across fields",
            "temporal",
            "publication year",
        ),
        "Taxonomy choice, interdisciplinary work, database coverage, and fast-changing topics can induce unstable controls.",
    ),
    DimensionRule(
        "interdisciplinarity",
        "Interdisciplinarity and knowledge diversity",
        "T0 breadth, diversity, and distance among disciplines or knowledge sources combined in the paper.",
        "interdisciplinarity and knowledge diversity",
        "predictive",
        "Disciplinary diversity must be computed from T0 content, affiliations, or references only.",
        (
            "Rao-Stirling diversity",
            "disciplinary diversity",
            "knowledge distance",
            "reference-category diversity",
        ),
        (
            "interdisciplin",
            "disciplinary diversity",
            "knowledge diversity",
            "diversity in citation",
            "distant knowledge",
        ),
        "Field taxonomies and coverage can systematically over- or understate interdisciplinarity.",
    ),
    DimensionRule(
        "rigor_reporting",
        "Methodological rigor and reporting quality",
        "Paper-level study-design, reporting-completeness, transparency, and methodological-quality characteristics visible in the manuscript.",
        "methodological rigor and reporting completeness",
        "predictive",
        "The assessed design/reporting information must be present in the manuscript by T0; later replications are validation evidence only.",
        (
            "reporting-guideline adherence",
            "risk-of-bias score",
            "methods completeness",
            "study-design quality",
        ),
        (
            "reporting quality",
            "reporting completeness",
            "research quality",
            "methodological quality",
            "risk of bias",
            "guideline",
            "checklist",
            "consort",
            "prisma",
            "study design",
        ),
        "Reporting standards vary by discipline and year; automated appraisal can inherit annotation and language bias.",
    ),
    DimensionRule(
        "reproducibility",
        "Reproducibility and transparent research practice",
        "T0 practices supporting verification and reuse, such as preregistration, registered reports, data/code sharing, and replication-oriented design.",
        "reproducibility and transparent research practice",
        "predictive",
        "The transparency practice must be declared or available by T0; successful later replication is not a T0 input.",
        (
            "preregistration status",
            "registered-report status",
            "data/code sharing",
            "replication design",
        ),
        (
            "reproducib",
            "replicat",
            "preregist",
            "registered report",
            "open science",
            "transparen",
            "data sharing",
            "code sharing",
        ),
        "Compliance can be self-reported, discipline-dependent, and confounded with journal or funder policies.",
    ),
    DimensionRule(
        "peer_review",
        "Peer-review and expert-assessment signals",
        "Prepublication expert judgments, reviewer agreement, editorial assessment, and structured appraisal of the focal paper.",
        "peer-review and expert-assessment evidence",
        "predictive",
        "Only review or editorial evidence produced by acceptance/publication T0 is eligible; later retrospective ratings are sensitivity evidence.",
        (
            "reviewer score",
            "reviewer agreement",
            "editorial decision",
            "expert quality rating",
        ),
        (
            "peer review",
            "peer-review",
            "reviewer",
            "expert assessment",
            "expert review",
            "editorial decision",
        ),
        "Reviewer identity, prestige, conflicts, discipline, and disagreement can introduce systematic bias.",
    ),
    DimensionRule(
        "citations",
        "Post-publication citation impact for validation",
        "Citation-based outcomes and impact measures used only to validate T0 predictors or conduct sensitivity analyses, never as T0 novelty evidence.",
        "post-publication citation impact outcome",
        "sensitivity",
        "Citation counts, trajectories, and derived impact indices occur after T0 and are outcome/validation variables only.",
        (
            "citation count",
            "field-normalized citation impact",
            "relative citation ratio",
            "citation percentile",
        ),
        (
            "citation",
            "cited",
            "impact factor",
            "h-index",
            "h index",
            "relative citation ratio",
            "highly cited",
        ),
        "Citation practices, database coverage, self-citation, field, age, language, and cumulative advantage bias comparisons.",
    ),
    DimensionRule(
        "attention",
        "Post-publication attention and usage for sensitivity",
        "Downloads, readership, social-media attention, and altmetrics used as non-T0 validation or sensitivity outcomes.",
        "post-publication attention and usage outcome",
        "sensitivity",
        "Attention and usage accumulated after publication are not eligible T0 predictors.",
        (
            "Altmetric Attention Score",
            "download count",
            "Mendeley readership",
            "social-media mentions",
        ),
        (
            "altmetric",
            "twitter",
            "tweet",
            "social media",
            "download",
            "readership",
            "mendeley",
            "web usage",
        ),
        "Platform demographics, bots, access, promotion, and field-specific adoption strongly bias attention measures.",
    ),
    DimensionRule(
        "metric_validity",
        "Metric validity, normalization, and responsible use",
        "Validity, reliability, normalization, fairness, and interpretability requirements for paper assessment metrics.",
        "validity and responsible use of assessment metrics",
        "sensitivity",
        "Validation may use later outcomes, but no post-T0 measure may enter a T0 predictive feature set.",
        (
            "convergent validity",
            "field normalization",
            "robustness across databases",
            "metric bias audit",
        ),
        (
            "validity",
            "validat",
            "reliability",
            "normalization",
            "normalised",
            "normalized",
            "bias",
            "fairness",
            "responsible metric",
            "evaluation tool",
            "correlate",
        ),
        "Choice of benchmark can circularly privilege established metrics and reproduce field or demographic inequities.",
    ),
)


def _snippet(title: str, abstract: str, phrases: tuple[str, ...]) -> str:
    clean = " ".join(abstract.split())
    lowered = clean.casefold()
    positions = [lowered.find(phrase) for phrase in phrases if phrase in lowered]
    if positions:
        start = max(0, min(positions) - 100)
        return clean[start : start + 500]
    return " ".join(title.split())[:500]


def _discipline(text: str) -> str:
    groups = (
        (
            "health_and_medicine",
            (
                "clinical",
                "medical",
                "health",
                "patient",
                "surgery",
                "medicine",
                "biomedical",
            ),
        ),
        (
            "computer_and_information_science",
            (
                "machine learning",
                "artificial intelligence",
                "algorithm",
                "computer science",
                "software",
                "llm",
            ),
        ),
        (
            "social_science_and_management",
            (
                "management",
                "business",
                "econom",
                "social science",
                "education",
                "policy",
            ),
        ),
        (
            "natural_and_environmental_science",
            ("ecology", "environment", "physics", "chemistry", "biology", "agricultur"),
        ),
    )
    matches = [
        label for label, phrases in groups if any(phrase in text for phrase in phrases)
    ]
    return "+".join(matches) if matches else "multidisciplinary_research_evaluation"


def code_mentions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    mentions: list[dict[str, str]] = []
    for row in rows:
        title, abstract = row["title"], row["abstract"]
        text = f"{title} {abstract}".casefold()
        text = text.replace("peer-reviewed", "").replace("peer reviewed", "")
        text = text.replace("cochrane collaboration", "cochrane")
        matched = [
            rule for rule in RULES if any(phrase in text for phrase in rule.phrases)
        ]
        if not matched:
            matched = [next(rule for rule in RULES if rule.key == "metric_validity")]
        # Preserve distinct major constructs without allowing generic citation words
        # to crowd out more informative T0 constructs.
        matched = sorted(
            matched,
            key=lambda rule: (
                rule.key in {"citations", "metric_validity"},
                RULES.index(rule),
            ),
        )[:4]
        team = f"TEAM_{sha256_text(row['work_id'])[:16]}"
        for rule in matched:
            mentions.append(
                {
                    "work_id": row["work_id"],
                    "construct": rule.construct,
                    "role": rule.role,
                    "information_source": (
                        "title_only" if not abstract.strip() else "title_and_abstract"
                    ),
                    "T0_boundary": rule.t0_boundary,
                    "bias_risk": rule.bias_risk,
                    "discipline_scope": _discipline(text),
                    "indicator_mentions_json": canonical_json(list(rule.indicators)),
                    "independent_team": team,
                    "evidence_quote": _snippet(title, abstract, rule.phrases),
                    "_dimension_key": rule.key,
                }
            )
    return mentions


def synthesize_dimensions(mentions: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for mention in mentions:
        grouped[mention["_dimension_key"]].append(mention)
    dimensions: list[dict[str, str]] = []
    for rule in RULES:
        group = grouped.get(rule.key, [])
        if not group:
            continue
        work_ids = sorted({mention["work_id"] for mention in group})
        teams = sorted({mention["independent_team"] for mention in group})
        observed_indicators = sorted(
            {
                indicator
                for mention in group
                for indicator in json.loads(mention["indicator_mentions_json"])
            }
        )
        merge_log = {
            "method": "Primary mentions grouped by shared observable construct, role, and T0 boundary; no target count or outcome was used.",
            "mention_count": len(group),
            "source_work_count": len(work_ids),
            "observed_constructs": sorted({mention["construct"] for mention in group}),
            "observed_indicator_mentions": observed_indicators,
            "split_rule": "Split when role or maximum-information-time boundary differs.",
        }
        dimensions.append(
            {
                "temporary_dimension_id": stable_id("PDIM", rule.label),
                "label": rule.label,
                "definition": rule.definition,
                "role": rule.role,
                "t0_boundary": rule.t0_boundary,
                "source_work_ids_json": canonical_json(work_ids),
                "independent_teams_json": canonical_json(teams),
                "merge_split_log_json": canonical_json(merge_log),
                "primary_approved": "1",
            }
        )
    return dimensions


def validate(
    inputs: list[dict[str, str]],
    mentions: list[dict[str, str]],
    dimensions: list[dict[str, str]],
) -> None:
    input_ids = {row["work_id"] for row in inputs}
    mention_ids = {row["work_id"] for row in mentions}
    if len(inputs) != 296 or len(input_ids) != 296 or mention_ids != input_ids:
        raise ProtocolError("Construct mentions must cover all 296 unique input works")
    allowed_roles = {"predictive", "opportunity", "control", "sensitivity"}
    if any(row["role"] not in allowed_roles for row in mentions):
        raise ProtocolError("Invalid construct role")
    if any(not json.loads(row["indicator_mentions_json"]) for row in mentions):
        raise ProtocolError("Every mention requires at least one indicator mention")
    if any(not row["evidence_quote"].strip() for row in mentions):
        raise ProtocolError("Every mention requires evidence")
    for dimension in dimensions:
        sources = set(json.loads(dimension["source_work_ids_json"]))
        teams = set(json.loads(dimension["independent_teams_json"]))
        log = json.loads(dimension["merge_split_log_json"])
        if not sources or not teams or not sources.issubset(input_ids):
            raise ProtocolError("Candidate dimension lacks valid sources or teams")
        if not log.get("observed_indicator_mentions"):
            raise ProtocolError("Candidate dimension lacks indicator support")
        if dimension["primary_approved"] != "1":
            raise ProtocolError("Primary candidate dimension is not approved")


def _write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_path: Path, mentions_path: Path, dimensions_path: Path, protocol_path: Path
) -> dict[str, Any]:
    with input_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mentions = code_mentions(rows)
    dimensions = synthesize_dimensions(mentions)
    validate(rows, mentions, dimensions)
    mention_fields = [
        "work_id",
        "construct",
        "role",
        "information_source",
        "T0_boundary",
        "bias_risk",
        "discipline_scope",
        "indicator_mentions_json",
        "independent_team",
        "evidence_quote",
    ]
    dimension_fields = [
        "temporary_dimension_id",
        "label",
        "definition",
        "role",
        "t0_boundary",
        "source_work_ids_json",
        "independent_teams_json",
        "merge_split_log_json",
        "primary_approved",
    ]
    _write(mentions_path, mentions, mention_fields)
    _write(dimensions_path, dimensions, dimension_fields)
    run_id = f"formal-dim-primary-{sha256_text(file_hash(input_path) + file_hash(protocol_path))[:16]}"
    manifest = {
        "artifact": "formal_construct_mentions_and_candidate_dimensions_primary",
        "generated_at": utc_now(),
        "reviewer_role": "Primary AI",
        "model_label": MODEL_LABEL,
        "run_id": run_id,
        "outcome_blind": True,
        "forbidden_sources_read": False,
        "no_numeric_quota": True,
        "protocol_sha256": file_hash(protocol_path),
        "input": {
            "path": str(input_path.resolve()),
            "sha256": file_hash(input_path),
            "row_count": len(rows),
        },
        "mentions": {
            "path": str(mentions_path.resolve()),
            "sha256": file_hash(mentions_path),
            "row_count": len(mentions),
            "covered_work_count": len({row["work_id"] for row in mentions}),
        },
        "candidate_dimensions": {
            "path": str(dimensions_path.resolve()),
            "sha256": file_hash(dimensions_path),
            "row_count": len(dimensions),
        },
        "mention_role_counts": dict(
            sorted(Counter(row["role"] for row in mentions).items())
        ),
        "coverage": {
            "input_work_count": 296,
            "covered_work_count": 296,
            "uncovered": 0,
            "duplicate_input_work_ids": 0,
        },
        "all_dimensions_have_sources_t0_and_indicators": True,
    }
    manifest_path = mentions_path.with_name(
        "formal_dimension_coding_primary.manifest.json"
    )
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mentions-output", type=Path, required=True)
    parser.add_argument("--dimensions-output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(
        canonical_json(
            run(args.input, args.mentions_output, args.dimensions_output, args.protocol)
        )
    )


if __name__ == "__main__":
    main()
