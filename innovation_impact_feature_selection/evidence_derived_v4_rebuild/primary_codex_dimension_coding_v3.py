from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from common import sha256_file, utc_now, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT / "outputs" / "human_tasks" / "formal_terminal_dimension_AI_v3.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "independent_codex_review_v3"
    / "formal_terminal_dimension_AI_REVIEWED_v3.csv"
)
DEFAULT_PROTOCOL = ROOT / "INDEPENDENT_CODEX_DIMENSION_CODING_PROTOCOL_V3.json"
REVIEWER_ID = "primary_codex_dimension_coding_v3"
MODEL = "codex_configured_default"
MODEL_DIGEST = "codex-thread:019fa728-bf6c-7453-9af8-9ade78756aae"
PROVENANCE_FIELDS = (
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_review_note",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
)

ROLE_BY_SCOPE = {
    "direct_innovation": "substantive_innovation",
    "t0_substantive": "t0_potential",
    "t0_opportunity": "opportunity",
    "context_control": "context_control",
}


def _rule(
    rule_id: str,
    pattern: str,
    label: str,
    definition: str,
    information_source: str,
    bias_risk: str,
    role: str = "",
) -> Dict[str, str]:
    return {
        "rule_id": rule_id,
        "pattern": pattern,
        "label": label,
        "definition": definition,
        "information_source": information_source,
        "bias_risk": bias_risk,
        "role": role,
    }


RULES: Sequence[Dict[str, str]] = (
    _rule(
        "C01",
        r"\b(systematic[- ]review|review (?:protocol|search|selection|"
        r"data[- ]extraction|evidence synthesis|heterogeneity|quality|"
        r"method|registration|eligibility)|prisma|amstar|search strategy)\b",
        "Evidence-synthesis rigor and transparency",
        "Design, execution, appraisal, synthesis, and transparent reporting "
        "of review-level evidence production.",
        "review protocol, manuscript methods, and evidence-synthesis metadata",
        "review-type heterogeneity and incomplete reporting",
    ),
    _rule(
        "C02",
        r"\b(novel\w*|original\w*|creativ\w*|recombin\w*|"
        r"atypic\w*|unconvention\w*|conventionality|pivot distance|"
        r"innovation inertia|methodological-innovation|"
        r"topic redundancy|new-concept|research-gap novelty)\b",
        "Knowledge novelty and recombination",
        "Departure from established knowledge through original, atypical, "
        "or newly recombined ideas, topics, or prior work.",
        "manuscript content, topics, concepts, and prior-knowledge links",
        "field-dependent baselines and construct-validity ambiguity",
        "substantive_innovation",
    ),
    _rule(
        "C03",
        r"\b(interdisciplin\w*|multidisciplin\w*|"
        r"disciplinar\w* divers\w*|rao[- ]stirling|"
        r"reference (?:variety|balance|disparity|diversity)|"
        r"topical diversity|citation structural diversity|"
        r"entropy diversity|entropy (?:balance|variety)|"
        r"content diversity)\b",
        "Cognitive breadth and interdisciplinary integration",
        "Variety, balance, disparity, or integration across the knowledge "
        "domains brought together by a paper.",
        "cited-reference fields, topic taxonomies, and cross-domain distances",
        "taxonomy coverage, field granularity, and distance-matrix drift",
        "substantive_innovation",
    ),
    _rule(
        "C04",
        r"\b(backward[- ]citation age|cited[- ]knowledge[- ]recency|"
        r"knowledge age|reference age)\b",
        "Knowledge-base temporal depth",
        "The recency or historical depth of the prior knowledge used by a "
        "paper at publication.",
        "focal publication year and cited-reference years",
        "missing reference years and field-specific citation-age norms",
    ),
    _rule(
        "C05",
        r"\b(bibliographic coupling|citation-network neighborhood|"
        r"citation network|network centrality|citation-context|"
        r"citation-(?:motive|reason|metadata-error)|"
        r"cited-reference prior impact|"
        r"prior-work-combination|reference count|reference-list|"
        r"reference accuracy|cited-reference disciplinary composition)\b",
        "Bibliographic knowledge-base structure",
        "The size, composition, linkage, and prior-network position of the "
        "paper's cited knowledge base.",
        "reference list and strictly prior bibliographic graph",
        "citation-database coverage and discipline-dependent reference norms",
    ),
    _rule(
        "C06",
        r"\b(collaboration-network|coauthorship-network|co-authorship network|"
        r"ego-centric coauthorship|algebraic connectivity|"
        r"research-network involvement|social-network-analysis)\b",
        "Prior collaboration-network position",
        "The pre-publication network position, cohesion, or reach available "
        "to the focal research team.",
        "author identities and strictly prior collaboration graph",
        "author disambiguation, career-length, and network-coverage bias",
        "opportunity",
    ),
    _rule(
        "C07",
        r"\b(author count|authorship characteristics|co-authorship|"
        r"coauthorship|collaborat\w*|multicenter|multi-institutional|"
        r"international collaboration|country count|country scale|"
        r"international-team|country composition|"
        r"team expertise|team diversity|author-team ethnic diversity|"
        r"rise of research teams|industry-academic collaboration)\b",
        "Team composition and collaboration scale",
        "The size, composition, heterogeneity, and collaborative reach of "
        "the team producing the paper.",
        "authorship and publication-time affiliation metadata",
        "author and affiliation coverage plus protected-attribute bias",
    ),
    _rule(
        "C08",
        r"\b(author (?:academic|authority|career|identity|order|"
        r"publication history|qualification|tenure|topic-expertise)|"
        r"first-author|prior-author|prior publication|advisor|"
        r"coauthor career|corresponding-authorship|tenure-system|"
        r"research-advisor|reviewer methodological training|"
        r"surgeon expertise|career experience|career stage|h-index)\b",
        "Prior investigator capability and reputation",
        "Experience, expertise, prior output, and accumulated reputation "
        "available to the investigators before publication.",
        "author histories, qualifications, and career metadata",
        "cumulative advantage, name disambiguation, and prestige confounding",
        "opportunity",
    ),
    _rule(
        "C09",
        r"\b(fund\w*|sponsor\w*|centre-of-excellence|"
        r"research infrastructure|research support|"
        r"experimental platform|research-advisor capacity)\b",
        "Research resources and funding opportunity",
        "Financial, organizational, and infrastructure resources available "
        "to support the research before publication.",
        "funding statements, sponsor, institution, and infrastructure metadata",
        "resource confounding, disclosure gaps, and sponsor-related bias",
        "opportunity",
    ),
    _rule(
        "C10",
        r"\b(journal|venue|conference|meeting|peer-review process|"
        r"blinded peer-review|digital dissemination|article publishing charge|"
        r"doi-presence|orcid|open-access|early-access|public-access mandate|"
        r"access-status|document type)\b",
        "Venue, access, and dissemination opportunity",
        "Publication-channel properties that shape discoverability, access, "
        "editorial selection, and dissemination opportunity.",
        "journal, venue, access, DOI, and editorial-process metadata",
        "prestige, access, selection, and platform confounding",
        "opportunity",
    ),
    _rule(
        "C11",
        r"\b(affiliat\w*|institution\w*|author country|country-field|"
        r"country similarity|geographic origin|research-location|"
        r"trial-enrolment geography|study geographic|study sampling location|"
        r"research site|academic program)\b",
        "Institutional and geographic context",
        "Institutional setting and geographic location associated with the "
        "paper and its production environment.",
        "author affiliations, institutions, countries, and study locations",
        "regional coverage, prestige confounding, and geographic inequity",
    ),
    _rule(
        "C12",
        r"\b(research field|article topic|topic orientation|subject-"
        r"classification|field citation intensity|field-year|"
        r"country-field|disciplinary terminology|hybrid-discipline)\b",
        "Field and topic context",
        "The disciplinary and topical environment in which the paper is "
        "positioned at publication.",
        "topic models, subject taxonomies, and field-period metadata",
        "taxonomy drift, multidisciplinary ambiguity, and normalization bias",
    ),
    _rule(
        "C13",
        r"\b(data (?:accessibility|availability|findability|format|licensing|"
        r"reuse)|data-availability|data-reuse|data-sharing|shared-data|"
        r"research-data sharing|research-code sharing|public full-text|"
        r"open-science|machine-readable data|"
        r"public study-protocol|publication-data linkage|open science)\b",
        "Open research artifacts and reproducibility support",
        "Availability, accessibility, documentation, licensing, and reuse "
        "readiness of data, code, protocols, and other research artifacts.",
        "data/code/protocol statements and linked research artifacts",
        "repository coverage, access inequality, and statement-content gaps",
    ),
    _rule(
        "C14",
        r"\b(abstract|title|writing|prose|readability|rhetorical|"
        r"linguistic|metadiscourse|dependency distance|keyword|"
        r"lay-summary|manuscript length policy|initial article information|"
        r"reader cognitive load|article structure|article length|"
        r"structural-element|article-production-quality|accessible scientific|"
        r"word-limit|language availability)\b",
        "Scientific communication and manuscript presentation",
        "Clarity, accessibility, structure, length, and rhetorical form of "
        "the paper's title, abstract, and manuscript.",
        "title, abstract, manuscript text, and document structure",
        "language, discipline, tokenization, and style-norm bias",
    ),
    _rule(
        "C15",
        r"\b(reporting|consort|cheers|hardy-weinberg|imaging-protocol|"
        r"technique-description|species-interaction|governance and ethics)\b",
        "Reporting completeness and guideline adherence",
        "Completeness, transparency, and compliance of paper reporting with "
        "relevant guidance.",
        "manuscript reporting items and guideline checklists",
        "guideline applicability and missing-full-text bias",
    ),
    _rule(
        "C16",
        r"\b(randomization|blinded outcome|confounding|study design|"
        r"methodological\w*|method quality|question-method alignment|"
        r"experimental control|experimental study|risk of bias|"
        r"selection-bias|internal validity|analytic feature selection|"
        r"prespecified outcome|primary-outcome|patient-selection)\b",
        "Study design rigor and internal validity",
        "Design and execution safeguards supporting unbiased causal or "
        "descriptive inference within the study.",
        "methods, design, sampling, allocation, and analysis plan",
        "design heterogeneity, residual confounding, and reporting bias",
    ),
    _rule(
        "C17",
        r"\b(effect size|statistical|sample size|power|precision|"
        r"significance|heterogeneity|fragility|estimate extremity|"
        r"diagnostic-tool accuracy|scoti score)\b",
        "Statistical evidence strength and result precision",
        "Magnitude, precision, robustness, and statistical informativeness "
        "of the evidence reported at publication.",
        "sample, estimates, uncertainty, tests, and statistical diagnostics",
        "selective reporting, scale heterogeneity, and small-study effects",
    ),
    _rule(
        "C18",
        r"\b(reproducib\w*|replicab\w*|reliab\w*|validation|validity|"
        r"inter-reviewer|omega reliability|gold-standard|"
        r"phantom-study|criterion-measure|construct validity|"
        r"exposure identification|measurement-tool)\b",
        "Measurement validity and reproducibility",
        "Validity, verification, replicability, and reliability of the "
        "paper's measurements, methods, and claims.",
        "measurement methods, validation evidence, and replication artifacts",
        "construct non-equivalence and validation-population shift",
    ),
    _rule(
        "C19",
        r"\b(evidence (?:applicability|consistency|directness|"
        r"generalizability|relevance|reliability|strength)|"
        r"biological plausibility|external validity)\b",
        "Evidence relevance and generalizability",
        "Directness, applicability, consistency, plausibility, and transfer "
        "potential of the paper's evidence.",
        "study population, evidence appraisal, and substantive interpretation",
        "context dependence and subjective appraisal",
    ),
    _rule(
        "C20",
        r"\b(research-question|focused review question|theoretical perspective|"
        r"theoretical grounding|conceptual-definition|"
        r"research-gap feasibility)\b",
        "Research-question and theoretical contribution",
        "The framing, conceptual clarity, theoretical grounding, and "
        "feasibility of the question addressed by the paper.",
        "research question, theory, hypotheses, and conceptual definitions",
        "discipline-specific judgment and construct ambiguity",
    ),
    _rule(
        "C21",
        r"\b(conflict-of-interest|research integrity|plagiarism|"
        r"paper integrity|research risk|researcher reflexivity|"
        r"researcher-participant role|reviewer subjectivity|"
        r"spin bias|citation integrity|publication and funding bias|"
        r"winner's-curse|pleiotropy bias|reporting bias|citation bias)\b",
        "Integrity and bias sensitivity",
        "Indicators of integrity threats, selective processes, or bias that "
        "may qualify interpretation of a paper's potential.",
        "disclosures, manuscript claims, analysis, and provenance metadata",
        "under-reporting and inconsistent bias definitions",
        "sensitivity",
    ),
    _rule(
        "C22",
        r"\b(implementation|scalability|de-implementation|"
        r"value-for-money|socioeconomic-benefit|public involvement|"
        r"intended impact audience|"
        r"clinical content|economic-evaluation)\b",
        "Translational and societal relevance",
        "Feasibility, implementability, clinical or societal relevance, and "
        "prospective value of the research at publication.",
        "manuscript implications, intervention context, and value assessment",
        "speculative claims and context-dependent value judgments",
    ),
    _rule(
        "C23",
        r"\b(data quality|outcome assessment|endpoint-assessment|"
        r"core outcome|study measurement|phenotypic feature|"
        r"outcome measurement granularity|longitudinal imaging|"
        r"study sample characteristics|participant-demographics)\b",
        "Data, measurement, and sample adequacy",
        "Fitness, breadth, and quality of the paper's data, measurements, "
        "outcomes, and sampled population.",
        "data, measurement instruments, outcomes, and sample metadata",
        "missing-data, measurement-error, and representativeness bias",
    ),
    _rule(
        "C27",
        r"\b(article annotation|key-term association|knowledge-claim|"
        r"knowledge-graph|knowledge synergy|prior-knowledge engagement|"
        r"mesh topic|citation category)\b",
        "Semantic content and knowledge claims",
        "Concepts, claims, annotations, and semantic relations expressed or "
        "engaged by the paper.",
        "manuscript terms, subject headings, claims, and semantic graph",
        "vocabulary coverage, ontology drift, and text-extraction bias",
    ),
    _rule(
        "C28",
        r"\b(evaluation practice|paper evaluation|peer reviewer rating|"
        r"multi-reviewer consensus|librarian involvement|"
        r"anecdotal-only evaluation)\b",
        "External appraisal and evaluation process",
        "Structured appraisal, reviewer consensus, or expert evaluation "
        "available for judging a paper's research potential.",
        "reviewer assessments and evaluation-process metadata",
        "rater subjectivity, selection, and inter-reviewer dependence",
        "t0_potential",
    ),
    _rule(
        "C29",
        r"\b(biological justification|biomedical study-subject|"
        r"ischemia-model|preclinical sex|animal age and sex|"
        r"clinical content|experimental platform)\b",
        "Domain-specific substantive adequacy",
        "Discipline-specific evidence that the substantive research setup "
        "and biological or clinical context are adequate.",
        "domain-specific methods, subjects, and substantive justification",
        "limited transportability across research domains",
        "t0_potential",
    ),
    _rule(
        "C24",
        r"\b(publication year|publication recency|tenure timing|"
        r"covid-19 topic|early-access timing)\b",
        "Publication-time context",
        "Calendar-period and timing context needed to compare papers "
        "published under different scientific and publishing conditions.",
        "publication dates and period indicators",
        "secular trends and period confounding",
        "context_control",
    ),
    _rule(
        "C25",
        r"\b(study spatial scale|clinical research status|"
        r"basic/applied|empirical-study|research type|educational setting|"
        r"theoretical perspective|study sponsor|pharmaceutical-funding|"
        r"document type|publication language)\b",
        "Study and publication context controls",
        "Design, domain, language, and study-setting variables used to "
        "control comparability rather than represent innovation.",
        "study, publication, language, and sponsor metadata",
        "coarse coding and residual contextual confounding",
        "context_control",
    ),
    _rule(
        "C26",
        r"\b(research quality|study quality|scientific paper quality|"
        r"teacher-education manuscript quality|quality control|"
        r"evaluation criteria|evaluated study-design attributes)\b",
        "Overall research-quality potential",
        "Holistic or composite assessments of the credibility and quality "
        "potential visible at publication.",
        "manuscript, appraisal framework, and study-level quality evidence",
        "rater subjectivity and construct breadth",
        "t0_potential",
    ),
)


def _read_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _fallback(scope_role: str) -> Dict[str, str]:
    values = {
        "direct_innovation": (
            "Other substantive innovation constructs",
            "Source-linked substantive novelty constructs not stably merged "
            "with a more specific family by the primary coder.",
            "source-specific manuscript or metadata evidence",
            "construct heterogeneity and sparse independent support",
            "substantive_innovation",
        ),
        "t0_substantive": (
            "Other T0 research-potential constructs",
            "Other source-linked paper attributes proposed as substantive "
            "potential at publication.",
            "source-specific paper evidence available or proposed at T0",
            "construct heterogeneity and incomplete formula evidence",
            "t0_potential",
        ),
        "t0_opportunity": (
            "Other research-opportunity constructs",
            "Other source-linked opportunity or exposure conditions available "
            "to the paper at publication.",
            "source-specific publication-time opportunity metadata",
            "confounding and cumulative-advantage bias",
            "opportunity",
        ),
        "context_control": (
            "Other contextual controls",
            "Other source-linked contextual covariates used for comparability "
            "rather than as substantive innovation.",
            "source-specific contextual metadata",
            "coarse adjustment and residual confounding",
            "context_control",
        ),
    }
    label, definition, source, bias, role = values[scope_role]
    return {
        "rule_id": "C99",
        "label": label,
        "definition": definition,
        "information_source": source,
        "bias_risk": bias,
        "role": role,
    }


def _classify(name: str, scope_role: str) -> Dict[str, str]:
    for rule in RULES:
        if re.search(rule["pattern"], name, flags=re.IGNORECASE):
            result = dict(rule)
            if not result["role"]:
                result["role"] = ROLE_BY_SCOPE[scope_role]
            if result["role"] == "opportunity" and scope_role == "context_control":
                result["label"] += " control"
                result["definition"] += (
                    " In this source it is used as a background control."
                )
                result["role"] = "context_control"
            return result
    return _fallback(scope_role)


def _t0_boundary(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized == "t0":
        return "Available no later than publication at T0."
    if "unverified" in normalized:
        return (
            "Candidate construct only: the exact maximum information time "
            "remains unverified without the cited English full text."
        )
    return (
        "The evidence does not establish a publication-time boundary; this "
        "family is excluded from predictor-dimension coding."
    )


def code_rows(
    input_path: Path,
    protocol_path: Path,
    completed_at: str,
    run_id: str,
) -> tuple[List[str], List[Dict[str, str]], Dict[str, Any]]:
    fields, rows = _read_rows(input_path)
    prompt_sha = sha256_file(protocol_path)
    labels: Counter[str] = Counter()
    rules: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    reviewed: List[Dict[str, str]] = []
    for row in rows:
        scope_role = str(row["scope_role_evidence"]).strip().casefold()
        boundary = str(
            row["maximum_information_time_evidence"]
        ).strip()
        if scope_role not in ROLE_BY_SCOPE or (
            boundary.casefold() != "t0"
            and "unverified" not in boundary.casefold()
        ):
            row.update(
                {
                    "dimension_label": "",
                    "dimension_definition": "",
                    "construct_role": "",
                    "information_source": "",
                    "t0_boundary": _t0_boundary(boundary),
                    "bias_risk": "",
                    "decision": "exclude",
                    "reason": (
                        "The extracted family is outcome-only, out of scope, "
                        "or lacks even a provisional publication-time "
                        "predictor boundary; it cannot define a T0 model "
                        "dimension."
                    ),
                }
            )
            rule_id = "EXCLUDE_SCOPE_OR_TIME"
        else:
            classification = _classify(
                str(row["canonical_name_en"]),
                scope_role,
            )
            row.update(
                {
                    "dimension_label": classification["label"],
                    "dimension_definition": classification["definition"],
                    "construct_role": classification["role"],
                    "information_source": classification[
                        "information_source"
                    ],
                    "t0_boundary": _t0_boundary(boundary),
                    "bias_risk": classification["bias_risk"],
                    "decision": "include",
                    "reason": (
                        "Post-extraction semantic coding of the canonical "
                        f"indicator family under rule {classification['rule_id']}; "
                        f"source scope role={scope_role}. The label is a "
                        "candidate mapping for independent H1/H2 review, not "
                        "a retained dimension decision."
                    ),
                }
            )
            rule_id = classification["rule_id"]
            labels[row["dimension_label"]] += 1
        decisions[row["decision"]] += 1
        rules[rule_id] += 1
        row.update(
            {
                "draft_method": (
                    "primary_codex_post_extraction_semantic_dimension_coding"
                ),
                "independent_ai_review_status": "complete",
                "independent_ai_reviewer_id": REVIEWER_ID,
                "independent_ai_reviewed_at": completed_at,
                "independent_ai_review_action": (
                    f"{row['decision']}:{rule_id}"
                ),
                "independent_ai_review_note": row["reason"],
                "independent_ai_run_id": run_id,
                "independent_ai_model": MODEL,
                "independent_ai_prompt_sha256": prompt_sha,
            }
        )
        reviewed.append(row)
    output_fields = fields + [
        field for field in PROVENANCE_FIELDS if field not in fields
    ]
    summary = {
        "decision_counts": dict(sorted(decisions.items())),
        "candidate_label_count": len(labels),
        "candidate_label_counts": dict(sorted(labels.items())),
        "rule_counts": dict(sorted(rules.items())),
        "fallback_count": rules["C99"],
        "model_outcomes_used": False,
        "target_count_influence": False,
    }
    return output_fields, reviewed, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    protocol_path = args.protocol.resolve()
    completed_at = utc_now()
    run_id = (
        "primary_codex_dimension_coding_v3_"
        + completed_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    )
    output_fields, rows, summary = code_rows(
        input_path,
        protocol_path,
        completed_at,
        run_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rows, output_fields)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "run_id": run_id,
        "action": "primary_post_extraction_dimension_coding",
        "artifact_path": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "reviewer_role": "AI",
        "reviewer_id": REVIEWER_ID,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "protocol_path": str(protocol_path),
        "prompt_sha256": sha256_file(protocol_path),
        "parameters": {
            **summary,
            "qwen_or_ollama_used": False,
            "local_or_external_llm_api_used": False,
            "round_13": False,
        },
        "item_count": len(rows),
        "completed_at": completed_at,
        "status": "complete",
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "artifact_path": str(output_path),
                "artifact_sha256": sha256_file(output_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                **summary,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
