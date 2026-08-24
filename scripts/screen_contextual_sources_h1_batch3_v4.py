from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "innovation_impact_feature_selection/evidence_derived_v4_rebuild"
INPUT_PATH = ROOT / "outputs/contextual_source_screening_input_batch3_v4.csv"
OUTPUT_PATH = ROOT / "outputs/contextual_source_screening_H1_batch3_completed_v4.csv"
MANIFEST_PATH = ROOT / "outputs/contextual_source_screening_H1_batch3_completed_v4.manifest.json"


# Ordered by the input CSV. Each decision is independently based on the title/abstract
# supplied in that CSV; no AI/H2 screening result is consumed here.
SCREENING: list[tuple[str, str]] = [
    ("include_definition_or_review", "The title explicitly concerns using bibliometrics to measure research performance, making it a direct review/definition lead for evaluation indicators."),
    ("include_definition_or_review", "The abstract describes WoS as a source of publication and citation data for bibliometric research, relevant to indicator data and validation."),
    ("include_definition_or_review", "The title identifies a review of author-level bibliometric indicators, directly within the requested indicator-review scope."),
    ("exclude_not_relevant", "This is a domain-specific agriculture-drone literature map; it does not indicate a paper-level innovation or potential-impact indicator definition, application, or validation."),
    ("exclude_not_relevant", "The abstract maps online-learning research topics and patterns rather than defining or validating paper-level innovation/impact indicators."),
    ("exclude_not_relevant", "The taxonomy maps entrepreneurship-education literature, not a paper-level innovation or prospective-impact measure."),
    ("uncertain", "The title signals a bibliometric analysis but provides no abstract to establish whether it defines, applies, or reviews a usable paper-level indicator."),
    ("include_definition_or_review", "The title explicitly studies determinants of citation impact, a direct empirical lead for paper-level potential-impact variables."),
    ("exclude_not_relevant", "This is a historical mapping of intellectual-capital literature rather than indicator methodology for individual scholarly papers."),
    ("include_definition_or_review", "The abstract discusses journal impact factors and their use in academic performance evaluation, directly relevant to indicator review."),
    ("exclude_not_relevant", "The paper maps sustainable-tourism scholarship and field trends, not paper-level indicator definitions or applications."),
    ("exclude_not_relevant", "This summarizes bibliometric reviews across AI/BDA application domains rather than reviewing a paper-level innovation or impact indicator."),
    ("exclude_not_relevant", "Career-proactivity literature mapping concerns a substantive career construct, not indicators of a scholarly paper's innovation or potential impact."),
    ("exclude_not_relevant", "Committee gender composition and professorship outcomes are unrelated to paper-level scholarly innovation or impact measures."),
    ("include_definition_or_review", "The abstract measures scholarly influence through extramural citations, a directly relevant impact-indicator application."),
    ("include_definition_or_review", "The abstract evaluates open-access status while controlling article age, impact factor, coauthors, references, pages, field, and type in relation to citations."),
    ("include_definition_or_review", "The abstract explicitly applies evaluative bibliometrics, normative comparisons, co-authorship, and impact analysis to publications."),
    ("exclude_not_relevant", "This is a construction-technology literature review/network map rather than paper-level indicator research."),
    ("exclude_not_relevant", "The paper maps Islamic-finance/fintech research trends and does not claim a paper-level innovation or potential-impact indicator."),
    ("exclude_not_relevant", "The abstract describes a smart-city field landscape, not methods for evaluating individual scholarly papers."),
    ("exclude_not_relevant", "This maps the CSR/sustainability literature and is outside the indicator-definition/application scope."),
    ("exclude_not_relevant", "This bibliometric review concerns career-success research trends, not characteristics or evaluation of scholarly papers."),
    ("include_definition_or_review", "The abstract assesses scholarly impact using publication content, publication counts, and citation analysis."),
    ("include_definition_or_review", "The abstract analyzes tweet counts linked to articles and compares them with citations as complementary scholarly-impact indicators."),
    ("exclude_not_relevant", "The title only identifies a field-level e-learning bibliometric visualization, without evidence of relevant indicator methodology."),
    ("include_definition_or_review", "The title explicitly concerns the application of bibliometric analysis and is retained as a methods/user-aspects review lead."),
    ("include_definition_or_review", "The abstract explicitly applies field- and time-normalised citation-impact indicators to paper comparison."),
    ("include_definition_or_review", "The abstract defines international collaboration through co-authorship and calculates publication-level collaboration indicators."),
    ("exclude_not_relevant", "This maps sustainable-manufacturing literature and constructs a field framework rather than a paper-level scholarly indicator."),
    ("exclude_not_relevant", "The title is a leadership-development field review, not a source on indicators of paper innovation or impact."),
    ("exclude_not_relevant", "This systematic review concerns universities' third mission, not a paper-level indicator definition or validation."),
    ("exclude_not_relevant", "This maps AI-in-innovation research and its firm-level drivers/outcomes, not scholarly-paper indicators."),
    ("include_definition_or_review", "The title explicitly concerns data, measurement, and empirical methods in science-of-science research."),
    ("include_definition_or_review", "The abstract offers a science-of-science data lake and states that it computes frequently used measures from the literature."),
    ("include_definition_or_review", "The abstract quantifies author-level citation inequality and concentration using papers and citations, relevant to scholarly-impact measurement."),
    ("exclude_not_relevant", "This is guidance on conducting review research, not a review or application of paper-level innovation/potential-impact indicators."),
    ("include_definition_or_review", "The abstract reviews assessment of scientists and research and extracts problems and proposed solutions for research evaluation."),
    ("exclude_not_relevant", "The study predicts retractions/corrections as scientific-integrity outcomes, not a publication-time innovation or potential-impact indicator."),
    ("include_definition_or_review", "The abstract develops and applies measures of gender homophily in scholarly co-authorship, a paper/team-level collaboration characteristic."),
    ("exclude_not_relevant", "This is a field-level mapping of social-media-addiction literature, not a scholarly-paper indicator source."),
    ("exclude_not_relevant", "The paper reviews maritime-transport technology applications and maps a domain literature rather than indicator methodology."),
    ("include_definition_or_review", "The abstract presents the measurement school as concerned with alternative impact measurement in its Open Science framework."),
    ("include_definition_or_review", "The abstract compares bibliometric and Altmetric perspectives regarding publication impact and force."),
    ("exclude_not_relevant", "The title concerns fake-news detection taxonomies, unrelated to scholarly-paper innovation or impact indicators."),
    ("exclude_not_relevant", "The title identifies a restaurant-research main-path analysis, a domain mapping rather than an indicator source."),
    ("include_definition_or_review", "The abstract explicitly applies productivity and citation-impact indicators, including the share of top-10%-cited papers."),
    ("include_definition_or_review", "The abstract maps publication in suspected predatory journals, supplying a publication-channel characteristic usable at publication time."),
    ("include_definition_or_review", "The abstract links author ethnic diversity, novelty, audience diversity, and scientific impact in an empirical publication-level model."),
    ("exclude_not_relevant", "This is a brand-equity literature map rather than a source on scholarly-paper innovation or potential-impact indicators."),
    ("exclude_not_relevant", "This systematic review addresses self-regulation and public policy, not evaluation of scholarly papers."),
    ("exclude_not_relevant", "The title signals a digital-marketing field map, not indicator definitions/applications for publications."),
    ("include_definition_or_review", "The abstract is a science-of-science review covering measurements, ranking, and prediction of scientific research outcomes."),
    ("include_definition_or_review", "The abstract studies interdisciplinarity and innovativeness as characteristics associated with publication in highly ranked journals."),
    ("exclude_not_relevant", "This is a manuscript-writing guide for a substantive population, not a paper-level bibliometric indicator source."),
    ("include_definition_or_review", "The abstract analyzes citation and collaboration networks by assigning papers to author-affiliation locations and relates them to research impact."),
    ("exclude_not_relevant", "This is a sustainable-rural-tourism field bibliometric map, without a relevant paper-level indicator focus."),
    ("exclude_not_relevant", "This maps BIM research themes and citation networks at field level, rather than defining/applying a focal-paper innovation or impact indicator."),
    ("include_definition_or_review", "The abstract relates measured novelty and conventionality in finance articles to readership, publication prospects, and impact."),
    ("exclude_not_relevant", "This is a domain bibliometric/coding analysis of blockchain research, not scholarly-paper indicator methodology."),
    ("exclude_not_relevant", "This bibliometric review maps interorganizational-learning research trends rather than defining paper-level indicators."),
    ("include_definition_or_review", "The title directly identifies geographical and institutional proximity of research collaboration, a publication-team characteristic relevant to impact studies."),
    ("exclude_not_relevant", "This evaluates recommendation-message designs and user engagement, not definitions or validations of paper innovation/potential-impact indicators."),
    ("exclude_not_relevant", "This review concerns measurement of personal charisma, not characteristics or impact of scholarly papers."),
    ("exclude_not_relevant", "This is a sociology field review using bibliometrics as a descriptive method, not a paper-level indicator source."),
    ("exclude_not_relevant", "This meta-analysis concerns gender bias in academic evaluation contexts, not definitions/applications of paper-level indicators."),
    ("exclude_not_relevant", "This author-co-citation analysis maps the knowledge-utilization field rather than a focal-paper innovation or impact measure."),
    ("exclude_not_relevant", "This maps start-up-incubation research themes; it is not a scholarly-paper indicator review or application."),
    ("exclude_not_relevant", "This concerns demographic diversity in the economics profession, not paper-level publication measures."),
    ("include_definition_or_review", "The abstract reviews academic promotion and tenure evaluation, including venue prestige as a publication-assessment shortcut."),
    ("include_definition_or_review", "The abstract is an evidence-based review of open-access publishing practices and bibliometric studies of OA prevalence/patterns."),
    ("include_definition_or_review", "The report explicitly discusses journal impact factors, h-indices, and the role of metrics in research assessment and management."),
    ("include_definition_or_review", "The abstract defines harmonic allocation of authorship credit and a harmonic h-index for bibliometric ranking."),
    ("include_definition_or_review", "The abstract introduces a PageRank-based index for measuring and comparing researchers' publication records."),
    ("include_definition_or_review", "The abstract proposes COIRank, using citation-network relationships to assess scholarly-article impact."),
    ("include_definition_or_review", "The abstract defines leadership through corresponding authorship and examines its relationship with scholarly impact in collaborations."),
    ("include_definition_or_review", "The abstract measures article novelty from keywords and applies collaboration/funding variables at multiple levels."),
    ("exclude_not_relevant", "This policy report concerns research-career precarity and does not define or apply a scholarly-paper innovation/impact measure."),
    ("exclude_not_relevant", "This co-citation review maps online-learning trends and does not present an indicator definition or validation relevant to focal papers."),
    ("include_definition_or_review", "The abstract overviews national lists of scholarly publication channels used in performance-based research funding and evaluation."),
    ("include_definition_or_review", "The abstract reviews bibliometric and altmetric tools for assessing research-output impact and their limitations."),
    ("exclude_not_relevant", "This is a field-level higher-education bibliometric review, not a source on paper innovation or prospective-impact indicators."),
    ("exclude_not_relevant", "This maps AI-in-food-safety literature and research trends, not a scholarly-paper indicator."),
    ("exclude_not_relevant", "This is a bibliographic analysis of additive manufacturing/Industry 4.0, not paper-level indicator methodology."),
    ("exclude_not_relevant", "This maps adaptive-learning research publications, without relevance to scholarly-paper innovation/impact indicators."),
    ("exclude_not_relevant", "This maps millennials-at-work research rather than scholarly-paper evaluation metrics."),
    ("exclude_not_relevant", "This review concerns measurement of corporate greenwashing, a firm-reporting construct rather than scholarly-paper indicators."),
    ("exclude_not_relevant", "This systematic review concerns electric-vehicle adoption, not research-paper innovation or potential-impact measures."),
    ("include_definition_or_review", "The abstract develops and applies methodology for gender homophily in scholarly co-authorship."),
    ("include_definition_or_review", "The abstract calculates publication productivity, citations per paper, and top-quartile/top-10% journal outputs for research institutions."),
    ("include_definition_or_review", "The title is a comprehensive overview of bibliometrics and citation analysis techniques and applications."),
    ("include_definition_or_review", "The abstract studies open-access journal visibility through analyses combining Scopus and open-access databases."),
    ("include_definition_or_review", "The abstract compares publication records and citation counts across predatory and recognized open-access journals, yielding a publication-channel lead."),
    ("include_definition_or_review", "The abstract constructs 15 interpretable citation features and evaluates their relationship with citation decisions."),
    ("include_definition_or_review", "The abstract proposes and evaluates an impact variable combining document usage and citation counts for ranking."),
    ("include_definition_or_review", "The abstract develops and evaluates automated text classification of scientific versus societal-oriented publication aims."),
    ("include_definition_or_review", "The abstract evaluates backward/forward citation coverage of databases, directly relevant to citation-data provenance and validation."),
    ("exclude_not_relevant", "This maps construal-level-theory scholarship and research trends, not a focal-paper innovation or potential-impact indicator."),
    ("include_definition_or_review", "The abstract examines writing style, sentiment, seniority, and citation-measured impact of scholarly contributions."),
    ("exclude_not_relevant", "This systematic review is about relevance estimation in information retrieval generally, not scholarly-paper impact/innovation measures."),
    ("uncertain", "The title suggests scientific-communication infrastructure but supplies no abstract to establish a direct indicator definition, application, validation, or review."),
    ("exclude_not_relevant", "This literature survey concerns recommendation-system methods, not paper-level innovation or potential-impact indicators."),
    ("exclude_not_relevant", "This maps ESG/AI finance research domains and techniques rather than evaluating scholarly-paper indicators."),
    ("include_definition_or_review", "The abstract reviews scholarly-impact evaluation and prediction models, including citation counts and journal impact factors."),
    ("exclude_not_relevant", "This maps corporate-reputation research in supply chains, not innovation/potential-impact measures for scholarly papers."),
    ("include_definition_or_review", "The abstract explicitly discusses Field Weighted Citation Impact and Altmetrics Attention Score in a bibliometric review."),
    ("exclude_not_relevant", "This systematic review concerns circular-economy/accounting frameworks, not scholarly-paper indicator methodology."),
    ("exclude_not_relevant", "This green-procurement literature map does not define or validate paper-level innovation or impact indicators."),
    ("exclude_not_relevant", "This is a domain-specific AI-video-generation bibliometric review rather than a source on scholarly-paper indicators."),
    ("include_definition_or_review", "The abstract reviews bibliometric search strategies including lexical queries and citation analysis for delineating nanotechnology publications."),
    ("exclude_not_relevant", "The abstract applies Gini coefficients to university rankings, an institution-level inequality study rather than a focal-paper indicator."),
    ("include_definition_or_review", "The title explicitly concerns measuring internationality of academic journals, a bibliometric indicator topic."),
    ("include_definition_or_review", "The abstract presents scientometric trends and performance indicators including international co-publications and citation impact."),
    ("include_definition_or_review", "The abstract applies international co-authorship, journal impact factors, tiers, and citations to clinical-medicine publications."),
    ("include_definition_or_review", "The abstract applies international co-authorship and multiple citation-impact indicators, including top-10%-cited shares and FWCI."),
    ("include_definition_or_review", "The title directly links departmental-affiliation and reference-list disciplinary diversity with collaboration."),
    ("include_definition_or_review", "The title explicitly evaluates whether altmetrics work for assessing research quality."),
    ("include_definition_or_review", "The abstract analyzes paper citations and journal h-index across innovation research in Latin America."),
    ("exclude_not_relevant", "This maps destructive-leadership literature and thematic clusters, not scholarly-paper innovation or impact indicators."),
    ("include_definition_or_review", "The abstract tests manuscript length, authors, institutions, international coauthors, references, title words, and open access as citation correlates."),
    ("include_definition_or_review", "The abstract evaluates journal open-access conversion using article volumes and normalized impact-factor/average-relative-citation metrics."),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("Input has no header")
        rows = list(reader)
    if len(rows) != 120 or len(SCREENING) != len(rows):
        raise ValueError(f"Expected 120 rows and decisions, got {len(rows)} and {len(SCREENING)}")

    for row, (decision, rationale) in zip(rows, SCREENING, strict=True):
        row["screen_decision"] = decision
        row["evidence_span"] = row["title"]
        row["rationale"] = rationale

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = dict(sorted(Counter(row["screen_decision"] for row in rows).items()))
    manifest = {
        "schema": "contextual_source_screening_h1_batch3_manifest_v4",
        "run": "contextual-source-screening-h1-independent-batch3-v4-20260819",
        "reviewer": "H1",
        "input_sha256": sha256(INPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_count": len(rows),
        "screen_decision_counts": counts,
        "qwen_or_ollama_used": False,
        "read_ai_or_h2_or_other_batch3_outputs": False,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")


if __name__ == "__main__":
    main()
