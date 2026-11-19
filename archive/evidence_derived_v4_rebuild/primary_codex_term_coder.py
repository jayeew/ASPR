from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


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
TERM_RELATIONS = {
    "canonical",
    "synonym",
    "abbreviation",
    "full_form",
    "historical_name",
    "morphological_variant",
    "parameter_variant",
}
NONAUTHORIZING_SOURCE_TYPES = {
    "development_seed_hint",
    "pilot_v2_indicator",
    "pilot_v2_literature",
}
DOMAINS: Dict[str, str] = {
    "Novelty and transformative potential": (
        "Paper-level novelty, unconventionality, disruption, transformative "
        "potential, creative search, and changes in research direction."
    ),
    "Interdisciplinarity and knowledge integration": (
        "Paper-level integration, diversity, convergence, or distance across "
        "disciplines and knowledge bases."
    ),
    "Topic and field structure": (
        "Paper topics, semantic or field position, topical diversity, and "
        "the distribution or evolution of research areas."
    ),
    "Text and scholarly communication": (
        "Publication-time title, abstract, keyword, linguistic, rhetorical, "
        "readability, and writing characteristics."
    ),
    "Research rigor and reporting": (
        "Paper-level design quality, methodological rigor, bias, reporting, "
        "reproducibility, transparency, and evidence-synthesis quality."
    ),
    "Open science and accessibility": (
        "Publication-time openness, data or material sharing, persistent "
        "identifiers, repository linkage, and accessibility."
    ),
    "Team and collaboration": (
        "Authorship, team composition, collaboration structure, affiliation, "
        "and coauthorship-network characteristics."
    ),
    "Reference and scholarly network structure": (
        "Publication-time references, cited knowledge, bibliographic or "
        "citation-network position, and reference-derived measures."
    ),
    "Publication opportunity and context": (
        "Venue, policy, funding, peer-review, field, crisis, and other "
        "publication-time opportunity or background conditions."
    ),
    "Document structure and metadata": (
        "Paper length, sections, figures, tables, equations, article type, "
        "and other publication-time structural metadata."
    ),
    "Research difficulty and complexity": (
        "Operational measures or constructs describing the difficulty or "
        "complexity of an individual research paper."
    ),
    "Scholarly impact validation outcomes": (
        "Post-publication citation, readership, visibility, diffusion, and "
        "scientific-influence outcomes used only to validate T0 predictors."
    ),
    "Publication and peer-review validation outcomes": (
        "Publication, acceptance, rejection, review-score, and publication-"
        "delay outcomes used only for validation."
    ),
    "Societal and translational validation outcomes": (
        "Clinical, policy, market, public, or translational outcomes used "
        "only as validation targets."
    ),
    "Bias and background controls": (
        "Paper-level bias risks, confounding, and protected or contextual "
        "background variables retained only for control or sensitivity use."
    ),
    "Other paper-level evidence constructs": (
        "Evidence-linked paper-level constructs that remain searchable but "
        "do not fit another stable domain at primary coding."
    ),
}
QUERY_FAMILIES: Dict[str, str] = {
    "Novelty and transformative potential": (
        "novelty and transformation measurement or validation"
    ),
    "Interdisciplinarity and knowledge integration": (
        "interdisciplinarity and knowledge-integration measurement"
    ),
    "Topic and field structure": "topic and field-structure features",
    "Text and scholarly communication": (
        "textual and communication-feature effects"
    ),
    "Research rigor and reporting": (
        "research-quality and reporting assessment"
    ),
    "Open science and accessibility": (
        "open-science and accessibility effects"
    ),
    "Team and collaboration": "team and collaboration effects",
    "Reference and scholarly network structure": (
        "reference and scholarly-network features"
    ),
    "Publication opportunity and context": (
        "publication opportunity and context effects"
    ),
    "Document structure and metadata": (
        "document-structure and metadata features"
    ),
    "Research difficulty and complexity": (
        "research-difficulty measurement"
    ),
    "Scholarly impact validation outcomes": (
        "scholarly-impact validation outcomes"
    ),
    "Publication and peer-review validation outcomes": (
        "publication and peer-review validation outcomes"
    ),
    "Societal and translational validation outcomes": (
        "societal and translational validation outcomes"
    ),
    "Bias and background controls": "bias and background adjustment",
    "Other paper-level evidence constructs": (
        "other paper-level construct measurement"
    ),
}


EXCLUSION_PATTERNS: Sequence[Tuple[str, str]] = (
    (
        r"^(accuracy|appeal|concepts|dynamics|impact|implementation|"
        r"influence|innovations|outcome|quality|sus)$",
        "The isolated phrase is too vague to preserve a paper-level construct.",
    ),
    (
        r"(classical topic modeling|deep embedding clustering|doc2vec|"
        r"embedding and clustering approaches|extreme gradient boosting|"
        r"modified deep clustering|principal component \\(pc\\) analysis|"
        r"probabilistic text model|random forest classifier|"
        r"structural topic model|vector-based representations)",
        "This is a generic analysis algorithm rather than a paper feature.",
    ),
    (
        r"(cluster labels|cluster tagging|clustering performance indicators|"
        r"downstream tasks|importance of individual features|"
        r"most common pattern|prediction accuracies|prediction power|"
        r"prediction probabilities|required qualifications|"
        r"specially developed checklist|single indicator|"
        r"subjective expected utility framework|unstructured reading)",
        "The phrase is an unnamed method, model diagnostic, or generic prose.",
    ),
    (
        r"^(7 topic categories|eight pcs \\(pc1-pc8\\)|four article types|"
        r"four categories|type 11|type 5)$",
        "The label is a study-specific category without a reusable construct.",
    ),
    (
        r"(roland-morris|rmdq|eq-5d|european quality of life|"
        r"visual analogue scale|pain \\(vas\\)|radiological outcome|"
        r"disability and function)",
        "This is a clinical outcome instrument, not a paper-level feature.",
    ),
    (
        r"^(broad value|new terms|outstanding research)$",
        "The phrase is non-operational evaluative prose.",
    ),
)


FAMILY_RULES: Sequence[Tuple[str, str, str]] = (
    (
        r"(relative citation ratio|\\brcrs?\\b|mean rcr|median rcr)",
        "relative citation ratio",
        "Relative citation ratio",
    ),
    (
        r"(journal cumulative citation for 5 years|\\bjcc5\\b)",
        "journal cumulative five-year citations",
        "Journal cumulative five-year citations",
    ),
    (
        r"(cumulative citation for 5 years|\\bcc5\\b)",
        "cumulative five-year citations",
        "Cumulative five-year citations",
    ),
    (
        r"(number of (article )?citations|citation counts?|citations count|"
        r"citation frequency|citation rates?|web of science citations per "
        r"year|isi citations)",
        "citation count or rate",
        "Citation count or rate",
    ),
    (
        r"(citation impact|normalized citations|academic impact|"
        r"scholarly impact|scientific impact)",
        "scholarly citation impact",
        "Scholarly citation impact",
    ),
    (
        r"(journal impact factor|impact factor|thomson reuters impact factor|"
        r"\\bjif\\b)",
        "journal impact factor",
        "Journal impact factor",
    ),
    (
        r"(disruption index|disruptive innovation level|"
        r"disruptive innovation evaluation|disruptive innovation and)",
        "disruption index",
        "Disruption index",
    ),
    (
        r"(scientific novelty|\\bnovelty\\b|types of scientific novelty)",
        "scientific novelty",
        "Scientific novelty",
    ),
    (
        r"(transformativeness|transformative science|paradigm-changing)",
        "transformative potential",
        "Transformative potential",
    ),
    (
        r"(interdisciplinarity|interdisciplinary research)$",
        "interdisciplinarity",
        "Interdisciplinarity",
    ),
    (
        r"(topic interdisciplinarity)",
        "topic interdisciplinarity",
        "Topic interdisciplinarity",
    ),
    (
        r"(knowledge-base interdisciplinarity)",
        "knowledge-base interdisciplinarity",
        "Knowledge-base interdisciplinarity",
    ),
    (
        r"(rao-stirling|\\bdiv\\b)",
        "Rao-Stirling diversity",
        "Rao-Stirling diversity",
    ),
    (
        r"(betweenness centrality)",
        "betweenness centrality",
        "Betweenness centrality",
    ),
    (
        r"(variety)$",
        "interdisciplinary variety",
        "Interdisciplinary variety",
    ),
    (
        r"(balance)$",
        "interdisciplinary balance",
        "Interdisciplinary balance",
    ),
    (
        r"(disparity)$",
        "interdisciplinary disparity",
        "Interdisciplinary disparity",
    ),
    (
        r"(open access|arxiv|early view)",
        "open-access or early-view status",
        "Open-access or early-view status",
    ),
    (
        r"(data shar|shared research data|share their research data)",
        "research data sharing",
        "Research data sharing",
    ),
    (
        r"(linking publications to research data|link to data|"
        r"well-formed links to data)",
        "publication-data linkage",
        "Publication-data linkage",
    ),
    (
        r"(orcid)",
        "ORCID identifier availability",
        "ORCID identifier availability",
    ),
    (
        r"(reporting quality|quality of reporting)",
        "reporting quality",
        "Reporting quality",
    ),
    (
        r"(methodological quality|methodological rigor|\\brigor\\b)",
        "methodological rigor",
        "Methodological rigor",
    ),
    (
        r"(risk of bias in systematic reviews|\\brobis\\b)",
        "ROBIS risk-of-bias assessment",
        "ROBIS risk-of-bias assessment",
    ),
    (
        r"(risk of bias)$",
        "risk of bias",
        "Risk of bias",
    ),
    (
        r"(publication bias)",
        "publication bias",
        "Publication bias",
    ),
    (
        r"(spin bias|forms of spin|type 3 spin|prevalence of spin)",
        "spin bias",
        "Spin bias",
    ),
    (
        r"(prisma-s)",
        "PRISMA-S reporting checklist",
        "PRISMA-S reporting checklist",
    ),
    (
        r"(prisma-p)",
        "PRISMA-P reporting checklist",
        "PRISMA-P reporting checklist",
    ),
    (
        r"(prisma preferred reporting items)",
        "PRISMA reporting guideline",
        "PRISMA reporting guideline",
    ),
    (
        r"(quorom)",
        "QUOROM reporting guideline",
        "QUOROM reporting guideline",
    ),
    (
        r"(amstar 2)",
        "AMSTAR 2 review-quality tool",
        "AMSTAR 2 review-quality tool",
    ),
    (
        r"(title length|shorter titles|optimal title length)",
        "title length",
        "Title length",
    ),
    (
        r"(title attributes|a colon|title optimization)",
        "title characteristics",
        "Title characteristics",
    ),
    (
        r"(abstract stylistic|writing styles|objective writing style)",
        "abstract or writing style",
        "Abstract or writing style",
    ),
    (
        r"(keyword discoverability|redundant keywords|key terms|"
        r"keyword limitations)",
        "keyword characteristics",
        "Keyword characteristics",
    ),
    (
        r"(linguistic properties|language features|textual properties|"
        r"lexical and sentiment|specialised vocabulary|jargon)",
        "linguistic characteristics",
        "Linguistic characteristics",
    ),
    (
        r"(number of authors)",
        "team size",
        "Team size",
    ),
    (
        r"(coauthorship network centrality|coauthorship networks)",
        "coauthorship-network position",
        "Coauthorship-network position",
    ),
    (
        r"(international collaboration|multi-institutional|"
        r"collaborations across institutions)",
        "collaboration structure",
        "Collaboration structure",
    ),
    (
        r"(funding resources|funding instruments)",
        "research funding context",
        "Research funding context",
    ),
    (
        r"(research difficulty)",
        "research difficulty",
        "Research difficulty",
    ),
    (
        r"(citation advantage)",
        "citation advantage",
        "Citation advantage",
    ),
    (
        r"(readership impact)",
        "readership impact",
        "Readership impact",
    ),
    (
        r"(view count|number of views)",
        "view count",
        "View count",
    ),
    (
        r"(visibility|findability|discoverability|retrievability)",
        "scholarly visibility",
        "Scholarly visibility",
    ),
    (
        r"(peer-reviewed publication|publication success|publishability|"
        r"acceptance probability|accepted and rejected)",
        "publication or acceptance outcome",
        "Publication or acceptance outcome",
    ),
    (
        r"(reference count|bibliographical references)",
        "reference-list size",
        "Reference-list size",
    ),
    (
        r"(number of pages|longer papers)",
        "paper length",
        "Paper length",
    ),
    (
        r"(number of figures)",
        "figure count",
        "Figure count",
    ),
    (
        r"(number of tables)",
        "table count",
        "Table count",
    ),
    (
        r"(number of equations)",
        "equation count",
        "Equation count",
    ),
)


def normalize(value: str) -> str:
    """Return a stable lexical comparison form."""
    text = value.casefold().replace("‐", "-").replace("–", "-")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it as one large object."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    """Read one UTF-8 CSV while preserving row order."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, str]],
) -> None:
    """Write one UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def exclusion_reason(term: str) -> str:
    """Return a frozen exclusion reason, or an empty string."""
    value = normalize(term)
    for pattern, reason in EXCLUSION_PATTERNS:
        if re.search(pattern, value):
            return reason
    return ""


def outcome_domain(value: str) -> str:
    """Separate future validation outcomes by role."""
    if re.search(
        r"(accept|reject|peer review|publish|publication|dissertation)",
        value,
    ) and not re.search(r"(visibility|citation|impact factor)", value):
        return "Publication and peer-review validation outcomes"
    if re.search(
        r"(clinical|market|public|translation|translational|transferability|"
        r"practice|policy)",
        value,
    ):
        return "Societal and translational validation outcomes"
    return "Scholarly impact validation outcomes"


def classify_domain(term: str, proposed_role: str) -> str:
    """Apply the frozen bottom-up domain codebook."""
    value = normalize(term)
    if proposed_role == "validation_outcome":
        return outcome_domain(value)
    if proposed_role == "control":
        return "Bias and background controls"
    rules: Sequence[Tuple[str, str]] = (
        (
            r"(research difficulty|complexity)",
            "Research difficulty and complexity",
        ),
        (
            r"(open access|arxiv|data shar|research data|repository|orcid|"
            r"public access policy|findable and reusable|data reuse|"
            r"linking publications|link to data|accessible data)",
            "Open science and accessibility",
        ),
        (
            r"(interdisciplin|multidisciplin|cross-disciplinary|convergence|"
            r"knowledge integration|diversity of references|"
            r"diversity of the publication|rao-stirling|betweenness|"
            r"variety|balance|disparity)",
            "Interdisciplinarity and knowledge integration",
        ),
        (
            r"(novel|innovat|disrupt|transform|radical|paradigm|pivot|"
            r"creative search|exploration|non-mainstream|scientific "
            r"revolution|new concepts|obsolescence|conventionality|risky)",
            "Novelty and transformative potential",
        ),
        (
            r"(title|abstract|keyword|linguistic|lexical|sentiment|writing|"
            r"rhetorical|metadiscourse|jargon|vocabulary|language|word "
            r"limit|scholarly communication|scientific communication)",
            "Text and scholarly communication",
        ),
        (
            r"(quality|report|prisma|quorom|amstar|robis|bias|rigor|"
            r"reproduc|integrity|protocol|review standards|evidence "
            r"synthesis|methodological|transparen|blinding|spin|"
            r"statistically significant|implementation appraisal|"
            r"search strategy|data extraction coding and synthesis)",
            "Research rigor and reporting",
        ),
        (
            r"(author|coauthor|collaborat|team|gender|matilda|role congruity|"
            r"institution|countries|country-specific|female|male)",
            "Team and collaboration",
        ),
        (
            r"(reference|citation-based|bibliometric|scientometric|"
            r"co-occurrence network|network centrality|cited knowledge|"
            r"referencing practices|reasons for citing|mesh weights)",
            "Reference and scholarly network structure",
        ),
        (
            r"(topic|field|semantic|research area|research trend|term "
            r"communit|sampling location|ocean-related|affective research|"
            r"knowledge production|research focus)",
            "Topic and field structure",
        ),
        (
            r"(funding|journal|venue|policy|peer review|quartile|proceedings|"
            r"covid|emerging science|centres of excellence|resource "
            r"allocation|disciplinary evaluation|research topic selection)",
            "Publication opportunity and context",
        ),
        (
            r"(number of pages|number of figures|number of tables|number of "
            r"equations|article types|document|combined lengths|character "
            r"count|reference count|longer papers|metadata inputs)",
            "Document structure and metadata",
        ),
    )
    for pattern, domain in rules:
        if re.search(pattern, value):
            return domain
    return "Other paper-level evidence constructs"


def family_for(term: str) -> tuple[str, str]:
    """Map true lexical or parameter variants to one term family."""
    value = normalize(term)
    for pattern, canonical, family in FAMILY_RULES:
        if re.search(pattern, value):
            return canonical, family
    canonical = re.sub(r"\s+", " ", term.strip().strip("\"“”'"))
    family = canonical[:1].upper() + canonical[1:]
    return canonical, family


def relation_for(term: str, canonical: str) -> str:
    """Assign one controlled lexical relation."""
    source = normalize(term)
    target = normalize(canonical)
    if source == target:
        return "canonical"
    if re.search(r"\([A-Z][A-Z0-9-]{1,12}", term):
        return "full_form"
    if re.fullmatch(r"[A-Z][A-Z0-9-]{1,15}", term.strip()):
        return "abbreviation"
    if re.search(
        r"\d|count|rate|ratio|percentile|length|year|score|index|"
        r"probability|median|mean",
        source,
    ):
        return "parameter_variant"
    return "synonym"


def code_row(row: Mapping[str, str]) -> Dict[str, str]:
    """Code one term using only its blind input fields."""
    output = dict(row)
    term = str(row.get("verbatim_term") or "").strip()
    role = str(row.get("proposed_role") or "").strip().casefold()
    source_type = str(row.get("source_type") or "").strip()
    excluded = exclusion_reason(term)
    if excluded:
        output.update(
            {
                "canonical_term": "",
                "term_family_label": "",
                "term_relation": "",
                "search_domain_label": "",
                "search_domain_definition": "",
                "query_family_label": "",
                "cross_domain": "false",
                "decision": "exclude",
                "reason": excluded,
            }
        )
        return output
    domain = classify_domain(term, role)
    canonical, family = family_for(term)
    relation = relation_for(term, canonical)
    if relation not in TERM_RELATIONS:
        raise ValueError(f"Invalid relation for {row['term_id']}")
    output.update(
        {
            "canonical_term": canonical,
            "term_family_label": family,
            "term_relation": relation,
            "search_domain_label": domain,
            "search_domain_definition": DOMAINS[domain],
            "query_family_label": QUERY_FAMILIES[domain],
            "cross_domain": "false",
            "decision": "include",
            "reason": (
                (
                    "Nonauthorizing development term retained only as "
                    "retrieval enrichment for "
                    if source_type in NONAUTHORIZING_SOURCE_TYPES
                    else f"Direct English {role.replace('_', ' ')} term "
                    "retained under "
                )
                + (
                    f"the evidence-derived {domain} boundary."
                    if source_type not in NONAUTHORIZING_SOURCE_TYPES
                    else f"the directly supported {domain} boundary."
                )
            ),
        }
    )
    return output


def build_codebook(rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    """Summarize the frozen bottom-up assignments for audit."""
    domain_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    excluded = 0
    for row in rows:
        if row["decision"] == "exclude":
            excluded += 1
            continue
        domain = row["search_domain_label"]
        family = row["term_family_label"]
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "protocol": "primary_codex_bottom_up_term_coding_v3",
        "no_target_count": True,
        "domains": [
            {
                "label": label,
                "definition": DOMAINS[label],
                "query_family_label": QUERY_FAMILIES[label],
                "coded_terms": domain_counts.get(label, 0),
            }
            for label in DOMAINS
            if domain_counts.get(label, 0)
        ],
        "term_family_count": len(family_counts),
        "included_rows": sum(domain_counts.values()),
        "excluded_rows": excluded,
    }


def main() -> None:
    """Run the frozen primary-Codex blind coding transform."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--codebook-output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reviewed-at", required=True)
    parser.add_argument("--thread-id", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    codebook_path = Path(args.codebook_output).resolve()
    prompt_path = Path(args.prompt).resolve()
    fields, input_rows = read_csv(input_path)
    if len({row["term_id"] for row in input_rows}) != len(input_rows):
        raise ValueError("Blind AI input contains duplicate term_id values")
    prompt_sha = sha256_file(prompt_path)
    output_fields = list(fields)
    for field in PROVENANCE_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    coded_rows = [code_row(row) for row in input_rows]
    for row in coded_rows:
        row.update(
            {
                "draft_method": "primary_codex_session_coding",
                "independent_ai_review_status": "complete",
                "independent_ai_reviewer_id": (
                    "primary_codex_ai_term_v3"
                ),
                "independent_ai_reviewed_at": args.reviewed_at,
                "independent_ai_review_action": "blind_term_coding",
                "independent_ai_review_note": (
                    "Primary Codex bottom-up codebook decision."
                ),
                "independent_ai_run_id": args.run_id,
                "independent_ai_model": "codex_configured_default",
                "independent_ai_prompt_sha256": prompt_sha,
            }
        )
    write_csv(output_path, output_fields, coded_rows)
    codebook_path.parent.mkdir(parents=True, exist_ok=True)
    codebook = build_codebook(coded_rows)
    codebook_path.write_text(
        json.dumps(codebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if [row["term_id"] for row in input_rows] != [
        row["term_id"] for row in coded_rows
    ]:
        raise RuntimeError("Output term order changed")
    script_sha = sha256_file(Path(__file__).resolve())
    manifest = {
        "run_id": args.run_id,
        "artifact_path": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "reviewer_role": "AI",
        "reviewer_id": "primary_codex_ai_term_v3",
        "model": "codex_configured_default",
        "model_digest": f"codex-thread:{args.thread_id}",
        "prompt_sha256": prompt_sha,
        "parameters": {
            "review_method": "primary_codex_reasoned_bottom_up_codebook",
            "independence_boundary": (
                "Blind AI worksheet only; no H1/H2 output, Ollama, Qwen, "
                "local/external LLM API, or downstream model result."
            ),
            "script_sha256": script_sha,
            "codebook_path": str(codebook_path),
            "codebook_sha256": sha256_file(codebook_path),
            "domain_count_in_ai_codebook": len(codebook["domains"]),
            "term_family_count_in_ai_codebook": codebook[
                "term_family_count"
            ],
        },
        "item_count": len(coded_rows),
        "completed_at": args.reviewed_at,
        "status": "complete",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(codebook, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
