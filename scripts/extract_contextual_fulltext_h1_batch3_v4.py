from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1] / "innovation_impact_feature_selection/evidence_derived_v4_rebuild"
INPUT_PATH = ROOT / "outputs/contextual_fulltext_extraction_input_batch3_v4.csv"
BRIEF_PATH = ROOT / "CONTEXTUAL_FULLTEXT_EXTRACTION_BATCH3_BRIEF_V4.md"
REVIEW_PATH = ROOT / "outputs/contextual_fulltext_source_review_H1_batch3_v4.csv"
MENTIONS_PATH = ROOT / "outputs/contextual_fulltext_indicator_mentions_H1_batch3_v4.csv"
MANIFEST_PATH = ROOT / "outputs/contextual_fulltext_extraction_H1_batch3_v4.manifest.json"

MENTION_FIELDS = [
    "record_key", "raw_name_en", "canonical_name_en", "source_role", "formula_location",
    "evidence_span", "formula", "parameters", "required_data", "maximum_information_time",
    "scope_role", "requires_future", "extraction_notes",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mention(
    record_key: str, raw_name_en: str, canonical_name_en: str, source_role: str,
    formula_location: str, evidence_span: str, formula: str, parameters: str,
    required_data: str, maximum_information_time: str, scope_role: str,
    extraction_notes: str,
) -> dict[str, str]:
    return {
        "record_key": record_key, "raw_name_en": raw_name_en,
        "canonical_name_en": canonical_name_en, "source_role": source_role,
        "formula_location": formula_location, "evidence_span": evidence_span,
        "formula": formula, "parameters": parameters, "required_data": required_data,
        "maximum_information_time": maximum_information_time, "scope_role": scope_role,
        "requires_future": "false", "extraction_notes": extraction_notes,
    }


DECISIONS = {
    "doi:10.1002/asi.23265": (
        "formula_or_application",
        "Explicitly applies a journal-list based OA publication-channel grouping to focal articles; author publication/citation histories are later or author-level outcomes and are not extracted.",
    ),
    "doi:10.1007/978-3-319-00026-8_2": (
        "review_discovery_only",
        "Conceptual literature review of Open Science schools of thought; alternative impact measurement is terminology/discovery context, not a source-authorized paper-level T0 formula or application.",
    ),
    "doi:10.1007/s00268-018-4579-9": (
        "no_relevant_indicator",
        "Compares article citations and Altmetric scores after publication; its publication-impact measures require future attention/citation events and are not T0 features.",
    ),
    "doi:10.1007/s10961-009-9149-8": (
        "review_discovery_only",
        "Review of nanotechnology social-science studies and bibliometric search strategies. It is useful for discovery terminology and cited source tracing, but does not itself authorize a focal-paper T0 formula.",
    ),
    "doi:10.1007/s11192-013-1121-6": (
        "formula_or_application",
        "Explicit original application defines an internationally collaborating article using its author-affiliation countries; subsequent citations and journal impact factor are excluded.",
    ),
    "doi:10.1007/s11192-014-1423-3": (
        "review_discovery_only",
        "Review of 108 author-level bibliometric indicators. It cannot serve as sole authority for any formula; it is retained only as a discovery source for original definitions.",
    ),
    "doi:10.1007/s11192-015-1775-3": (
        "no_relevant_indicator",
        "Longitudinal study uses researcher-level productivity and later citation-impact outcomes; it does not define or apply a focal-paper T0 innovation, opportunity, or control indicator.",
    ),
    "doi:10.1007/s11192-016-1905-6": (
        "no_relevant_indicator",
        "Institution-level comparison of international co-authorship rates and five-year citation indicators; no original focal-paper T0 definition/application is supplied.",
    ),
    "doi:10.1007/s11192-018-2988-z": (
        "no_relevant_indicator",
        "Evaluates citations, h-indexes, Mendeley readers, and other altmetrics after their accumulation; no T0-computable candidate is available.",
    ),
    "doi:10.1007/s11192-019-03140-w": (
        "no_relevant_indicator",
        "Explicit formulas normalize citation counts using reference sets, but those citation counts and their reported citation windows are post-publication information.",
    ),
    "doi:10.1007/s11192-019-03155-3": (
        "formula_or_application",
        "Explicitly defines a publication as internationally co-authored from author affiliations in different countries, a focal-paper T0 collaboration indicator.",
    ),
    "doi:10.1007/s11192-021-04071-1": (
        "formula_or_application",
        "Explicitly calculates publication-level ethnic diversity from author-name ethnicity distributions. Its novelty implementation requires later three-year pair reuse and audience diversity/citation outcomes require future citing papers, so those are excluded.",
    ),
    "doi:10.1038/s41597-023-02198-9": (
        "formula_or_application",
        "Data-resource paper explicitly defines and provides paper-level reference count, team size, and institution count fields; future citation/attention fields and precomputed retrospective impact scores are not extracted.",
    ),
    "doi:10.1162/qss_a_00018": (
        "review_discovery_only",
        "Describes Web of Science data coverage and research use cases; useful data-provenance context but no source-authorized paper-level T0 indicator formula/application.",
    ),
    "doi:10.12688/f1000research.16493.1": (
        "review_discovery_only",
        "Review of promotion and tenure evaluation practices; it provides assessment terminology but no original focal-paper T0 formula or application.",
    ),
    "doi:10.13140/rg.2.1.4929.1363": (
        "review_discovery_only",
        "The Metric Tide is a policy review of metrics in research assessment; its metric discussion is discovery context and cannot alone authorize formulas.",
    ),
    "doi:10.1371/journal.pbio.2004089": (
        "review_discovery_only",
        "Expert-panel assessment principles and selective literature review for hiring and tenure; no explicit paper-level T0 computational indicator is defined or applied.",
    ),
    "doi:10.1371/journal.pone.0004021": (
        "formula_or_application",
        "Explicitly defines harmonic authorship credit from author rank and team size, which is computable from focal-paper byline information at T0.",
    ),
    "doi:10.48550/arxiv.1909.01284": (
        "no_relevant_indicator",
        "Defines corpus-level gender assortativity over authorships, not a focal-paper innovation/potential-impact/opportunity/control indicator; its statistical test also depends on a broader corpus.",
    ),
    "doi:10.7189/jogh.08.020411": (
        "no_relevant_indicator",
        "Aggregate country/institution productivity and citation study; it does not define or apply a focal-paper T0 indicator.",
    ),
}

MENTIONS = [
    mention(
        "doi:10.1002/asi.23265",
        "Group 1 / Group 2 / Group 3 OA journal groups",
        "Open-access publication-channel group",
        "original_application",
        "Research Design—Data Collection, pp. 6–7.",
        "“We selected a group of 68 journals from Beall’s ‘predatory’ journal list to represent low-quality publications”.",
        "Group 1 = selected journals on Beall’s predatory-journal list; Group 2 = selected OA journals that rejected Bohannon’s fake paper, are registered with DOAJ, and are not on Beall’s list; Group 3 = selected high-status PLOS journals.",
        "journal ISSN/title; membership in Beall list; DOAJ registration; Bohannon outcome; PLOS venue",
        "focal-paper journal identity and a versioned contemporaneous journal-list/registry snapshot",
        "T0 (journal-channel membership must be resolved using lists/registries available no later than the paper publication date).",
        "t0_control",
        "This is an explicit article grouping application. The study's author publication histories and citation counts are not extracted because they are not focal-paper T0 fields.",
    ),
    mention(
        "doi:10.1007/s11192-013-1121-6",
        "International collaboration",
        "International author-affiliation collaboration",
        "original_application",
        "Materials and methods, p. 1523.",
        "“International collaboration was deemed to exist in an article if any author’s affiliation was located outside Malaysia.”",
        "international_collaboration = 1 if any author affiliation is outside Malaysia; otherwise 0 for this Malaysia-based corpus.",
        "author-affiliation country; corpus focal-country rule",
        "focal-paper author affiliations and country identifiers",
        "T0 (computed from affiliations on the focal paper).",
        "t0_opportunity",
        "The country-specific application supplies an explicit paper-level collaboration definition. Future citation outcomes and journal impact factor are excluded.",
    ),
    mention(
        "doi:10.1007/s11192-019-03155-3",
        "International co-authorship",
        "International author-affiliation collaboration",
        "original_application",
        "Introduction, p. 748.",
        "“internationally co-authored if it has authors affiliated with institutions located in different countries.”",
        "international_coauthorship = 1 if focal-paper authors are affiliated with institutions in different countries; otherwise 0.",
        "distinct affiliation-country count",
        "focal-paper author-affiliation country identifiers",
        "T0 (computed from affiliations recorded with the paper).",
        "t0_opportunity",
        "Explicit source definition for a focal publication. Researcher-level shares calculated later in the study are not extracted.",
    ),
    mention(
        "doi:10.1007/s11192-021-04071-1",
        "Ethnic diversity (E_D)",
        "Author-team ethnic true diversity",
        "original_application",
        "Measurement of variables—Ethnic diversity, Eq. (1), pp. 7767–7769.",
        "“We used the true diversity measure (Zhang et al., 2016) for ethnic diversity, based on the distribution of authors in ethnic categories for a target publication.”",
        "E_D = 1 − Σ_{i,j=1}^{t_n} S_ij P_ti P_tj, where S_ij is ethnicity similarity and P_ti/P_tj are author shares in ethnic categories.",
        "t_n; P_ti; P_tj; S_ij",
        "focal-paper author names; a fixed author-name-to-ethnicity probability model; frozen ethnicity-similarity matrix",
        "T0 (provided the name-classification model and similarity matrix are fixed using information available at publication).",
        "t0_control",
        "Explicit publication-level application. The source's combinatorial novelty score requires subsequent three-year pair reuse, and its audience/citation variables require future citing records; none are extracted.",
    ),
    mention(
        "doi:10.1038/s41597-023-02198-9",
        "Reference_Count, Team_Size, Institution_Count",
        "Static paper bibliographic and team-size fields",
        "original_application",
        "Table 3, SciSciNet_Papers data-type definitions, p. 6.",
        "“Reference_Count Integer Total reference count of the paper.”",
        "[Reference_Count = total focal-paper references; Team_Size = number of researchers in the paper; Institution_Count = number of institutions in the paper].",
        "reference count; researcher count; institution count",
        "focal-paper reference list; author list; author-affiliation institution identifiers",
        "T0 (all listed fields are determined from the focal-paper bibliographic record).",
        "t0_control",
        "The data-resource paper explicitly defines these static fields. Citation, patent, news, tweet, C5/C10, and retrospective impact/novelty fields are intentionally excluded.",
    ),
    mention(
        "doi:10.1371/journal.pone.0004021",
        "Harmonic credit",
        "Harmonic authorship credit",
        "original_definition",
        "Harmonic Counting Corrects Bibliometric Bias, Eq. (1), p. 2.",
        "“The harmonic credit for the ith author of a publication with N coauthors is calculated as follows”.",
        "credit_i = (1/i) / Σ_{k=1}^{N}(1/k).",
        "i (author rank); N (number of coauthors)",
        "focal-paper ordered author byline",
        "T0 (computed exclusively from the focal paper's author order and team size).",
        "t0_control",
        "This is the paper's explicit original formula. Derived citation-credit and h-index applications are not extracted because they require subsequent citation records or author publication histories.",
    ),
]


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        if fields is None:
            raise ValueError("Input CSV has no header")
        rows = list(reader)
    if len(rows) != 20 or {row["record_key"] for row in rows} != set(DECISIONS):
        raise ValueError("Input rows do not exactly match reviewed decisions")
    if any(row["record_key"] not in DECISIONS for row in MENTIONS):
        raise ValueError("Mention record key absent from input")

    for row in rows:
        row["source_disposition"], row["source_notes"] = DECISIONS[row["record_key"]]

    with REVIEW_PATH.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with MENTIONS_PATH.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=MENTION_FIELDS)
        writer.writeheader()
        writer.writerows(MENTIONS)

    source_hashes = {row["record_key"]: sha256(Path(row["text_path"])) for row in rows}
    manifest: dict[str, Any] = {
        "schema": "contextual_fulltext_extraction_h1_batch3_manifest_v4",
        "run": "contextual-fulltext-extraction-h1-independent-batch3-v4-20260819",
        "reviewer": "H1",
        "input_hashes": {
            "brief_sha256": sha256(BRIEF_PATH),
            "input_csv_sha256": sha256(INPUT_PATH),
            "source_text_sha256_by_record_key": source_hashes,
        },
        "output_hashes": {
            "source_review_csv_sha256": sha256(REVIEW_PATH),
            "indicator_mentions_csv_sha256": sha256(MENTIONS_PATH),
        },
        "source_count": len(rows),
        "source_disposition_counts": dict(sorted(Counter(row["source_disposition"] for row in rows).items())),
        "indicator_mention_count": len(MENTIONS),
        "guards": {
            "read_ai_or_h2_or_other_batch3_outputs": False,
            "used_qwen_or_ollama": False,
            "reviews_used_as_formula_authority": False,
            "final_feature_decision_made": False,
        },
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")


if __name__ == "__main__":
    main()
