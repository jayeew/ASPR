"""Create the independent H1 full-text indicator extraction for batch 8."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_fulltext_extraction_input_batch8_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_fulltext_extraction_H1_batch8_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_fulltext_extraction_H1_batch8_completed_v4.manifest.json"
)
MENTION_FIELDS: Final = [
    "raw_name_en",
    "canonical_name_en",
    "source_role",
    "formula_location",
    "evidence_span",
    "formula",
    "parameters",
    "required_data",
    "maximum_information_time",
    "scope_role",
    "requires_future",
    "extraction_notes",
]

REVIEWS: Final[dict[str, tuple[str, str]]] = {
    "doi:10.1016/j.joi.2017.07.003": (
        "no_relevant_indicator",
        "The source profiles DataCite data-record metadata, not scholarly-paper measures; it provides no paper-level T0 indicator.",
    ),
    "doi:10.1016/j.joi.2018.07.005": (
        "review_discovery_only",
        "The landmark-paper interviews critically discuss unusual cited-reference combinations but do not define or apply a paper-level T0 formula.",
    ),
    "doi:10.1016/j.joi.2018.12.005": (
        "formula_or_application",
        "The source applies author-specified Chinese Library Classification codes to individual papers; this is a paper-level categorical control available by submission/publication.",
    ),
    "doi:10.1016/j.joi.2026.101791": (
        "no_relevant_indicator",
        "Citation-intent and incoming-network measures depend on later citing papers, so no focal-paper T0 candidate is retained.",
    ),
    "doi:10.1016/j.leaqua.2013.10.014": (
        "formula_or_application",
        "The study directly applies paper-level metadata and content/design controls while predicting later citations; only controls usable no later than publication are retained.",
    ),
    "doi:10.1016/j.respol.2024.105025": (
        "formula_or_application",
        "The source explicitly defines paper topic/social-connectivity scores and directly applies paper-level journal, authorship, reference, and author-status controls. T0 use requires time-indexed network and model assets.",
    ),
    "doi:10.1017/s0272263124000743": (
        "formula_or_application",
        "Table 1 explicitly defines eleven paper-, author-, and journal-level controls. All retained controls require only information available by publication when historical bibliometric snapshots are time-bounded.",
    ),
}


def add(
    key: str,
    raw: str,
    canonical: str,
    location: str,
    evidence: str,
    formula: str,
    parameters: str,
    data: str,
    maximum_time: str,
    notes: str,
) -> dict[str, str]:
    """Build a direct source-authorized H1 indicator mention."""
    return {
        "record_key": key,
        "raw_name_en": raw,
        "canonical_name_en": canonical,
        "source_role": "original_application",
        "formula_location": location,
        "evidence_span": evidence,
        "formula": formula,
        "parameters": parameters,
        "required_data": data,
        "maximum_information_time": maximum_time,
        "scope_role": "control",
        "requires_future": "false",
        "extraction_notes": notes,
    }


def make_mentions() -> list[dict[str, str]]:
    """Return every direct paper-level application that can be constrained to T0."""
    clc = "doi:10.1016/j.joi.2018.12.005"
    cited = "doi:10.1016/j.leaqua.2013.10.014"
    connect = "doi:10.1016/j.respol.2024.105025"
    linguistic = "doi:10.1017/s0272263124000743"
    return [
        add(
            clc,
            "Chinese Library Classification (CLC) code of a journal article",
            "paper_clc_discipline_code",
            "Section 2.2.2, pp. 5–6",
            "“The CLC code is also applied to each journal article. This is generally done by asking authors to provide the CLC code when submitting their manuscript.”",
            "paper_clc_discipline_code(p) = CLC code supplied for paper p, after any editorial modification",
            "CLC version; author-supplied code; editorial-modification rule",
            "manuscript CLC code and final bibliographic record",
            "submission or final publication",
            "Categorical discipline control. The source's HHI is journal-level, so it is not retained as a paper-level measure.",
        ),
        add(
            cited,
            "Article type",
            "article_type_indicator",
            "Independent variables, p. 18",
            "“We used k - 1 dummy variables to model article type, including quantitative, qualitative, theory, review, commentary/discussion, methodological, and agent-based simulation articles.”",
            "article_type_indicator_k(p) = 1 if p is type k; 0 otherwise",
            "source uses k - 1 dummies and stated mixed-type tie-break rules",
            "full manuscript/article type",
            "final manuscript or publication",
            "Paper-type categorical control.",
        ),
        add(
            cited,
            "Number of cited references",
            "reference_count",
            "Independent variables, p. 19",
            "“We tallied the number of cited references in the article as reported in the reference list (or footnoted in some cases).”",
            "reference_count(p) = count(references listed or footnoted in p)",
            "reference-list/footnote parsing rule",
            "final manuscript reference list and footnotes",
            "final manuscript or publication",
            "Do not substitute citations received by p.",
        ),
        add(
            cited,
            "Senior editor",
            "senior_editor_indicator",
            "Independent variables, p. 19",
            "“we used k-1 dummy variables to capture these effects”",
            "senior_editor_indicator_k(p) = 1 if p was handled under editor k; 0 otherwise",
            "source combines Robert J. House and Henry L. Tosi as one category",
            "editorial assignment/tenure record",
            "editorial decision or publication",
            "Editorial-context control.",
        ),
        add(
            cited,
            "Number of authors",
            "author_count",
            "Independent variables, p. 19",
            "“The total number of coauthors listed in the by-line of the article.”",
            "author_count(p) = count(authors in p byline)",
            "byline counting rule",
            "paper byline",
            "final manuscript or publication",
            "Direct authorship control.",
        ),
        add(
            cited,
            "Lagged author citations",
            "mean_lagged_author_citation_count",
            "Independent variables, pp. 19–20",
            "“We averaged the citations of all coauthors for a particular article to reflect how well the authors were collectively cited.”",
            "mean_lagged_author_citation_count(p) = mean(citations(a, before publication_year(p)) for a in authors(p))",
            "author disambiguation; pre-publication citation cutoff",
            "paper authors and dated prior-citation records",
            "immediately before publication year",
            "The source calls pre-1996 coverage a partial lag; never use a later citation snapshot.",
        ),
        add(
            cited,
            "Lagged author publications",
            "mean_lagged_author_publication_count",
            "Independent variables, p. 20",
            "“we tallied the number of articles authors had published and averaged this number for author teams.”",
            "mean_lagged_author_publication_count(p) = mean(publication_count(a, before publication_year(p)) for a in authors(p))",
            "author disambiguation; pre-publication cutoff",
            "paper authors and dated prior-publication records",
            "immediately before publication year",
            "Use the source's team average.",
        ),
        add(
            cited,
            "Rank of university affiliation",
            "mean_author_affiliation_rank",
            "Independent variables, p. 20",
            "“we then average the scores across coauthors for each article to obtain a collective score.”",
            "mean_author_affiliation_rank(p) = mean(rank(affiliation(a), snapshot_t0) for a in authors(p))",
            "author-affiliation matching; ranking snapshot dated by T0",
            "paper authors, affiliations, contemporaneous ranking snapshot",
            "paper T0",
            "The source's 2008–2012 average leaks for older papers; T0 use needs a contemporaneous snapshot.",
        ),
        add(
            cited,
            "Issue number",
            "journal_issue_indicator",
            "Independent variables, p. 21",
            "“We used k - 1 dummy variables to control for unobserved heterogeneity due to issue number”",
            "journal_issue_indicator_k(p) = 1 if p appears in issue k; 0 otherwise",
            "source uses k - 1 dummies",
            "paper issue metadata",
            "publication",
            "Distinct from calendar-year volume, which is not T0-safe.",
        ),
        add(
            cited,
            "School of leadership",
            "leadership_school_indicator",
            "Independent variables, p. 21",
            "“We used the following nine categories (modeled as k-1 dummy variables): trait, behavioral, contextual, contingency, relational, information processing ...”",
            "leadership_school_indicator_k(p) = 1 if p is school k; 0 otherwise",
            "source school and hybrid/tie-break rules",
            "full manuscript",
            "final manuscript or publication",
            "Content-coded categorical control.",
        ),
        add(
            cited,
            "Statistical analysis method used for hypothesis testing",
            "hypothesis_test_method_indicator",
            "Independent variables, pp. 21–22",
            "“we used the following independent categories, each modeled separately as one dummy variable: correlation analysis, analysis of variance ... regression, structural equation modeling ...”",
            "hypothesis_test_method_indicator_k(p) = 1 if p uses method k; 0 otherwise",
            "source categories/inclusion rules",
            "full manuscript methods",
            "final manuscript or publication",
            "More than one method indicator may equal one.",
        ),
        add(
            cited,
            "Number of studies",
            "study_count_category",
            "Independent variables, p. 22",
            "“We used the following five categories (modeled as k-1 dummy variables): one, two, three, four or more studies, or not applicable.”",
            "study_count_category_k(p) = 1 if p is category k; 0 otherwise",
            "one/two/three/four-or-more/not-applicable categories",
            "full manuscript",
            "final manuscript or publication",
            "Preserve the capped four-or-more category.",
        ),
        add(
            cited,
            "Sample location",
            "sample_location_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy variable: US, Europe, Asia, cross-national, others, or not applicable.”",
            "sample_location_indicator_k(p) = 1 if p has location k; 0 otherwise",
            "source location categories",
            "full manuscript sample description",
            "final manuscript or publication",
            "Several location indicators may equal one.",
        ),
        add(
            cited,
            "Study design",
            "study_design_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy variable: field survey, laboratory experiment, field experiment, quasi-experiment, archival data, meta-analysis, interview or others.”",
            "study_design_indicator_k(p) = 1 if p uses design k; 0 otherwise",
            "source design categories",
            "full manuscript methods",
            "final manuscript or publication",
            "Several design indicators may equal one.",
        ),
        add(
            cited,
            "Data source",
            "data_source_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy variable: one source, two or more subjective sources, one or more objective source, or not applicable.”",
            "data_source_indicator_k(p) = 1 if p is source category k; 0 otherwise",
            "source subjective/objective categories",
            "full manuscript data description",
            "final manuscript or publication",
            "Research-design control.",
        ),
        add(
            cited,
            "Temporal context of study",
            "study_temporal_context_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy variable: cross-sectional, two time periods ... longitudinal ... or not applicable.”",
            "study_temporal_context_indicator_k(p) = 1 if p is temporal context k; 0 otherwise",
            "source temporal-context categories",
            "full manuscript methods",
            "final manuscript or publication",
            "The source defines two-time-period studies as non-repeated measurement at time one and two.",
        ),
        add(
            cited,
            "Type of scale used",
            "measurement_scale_type_indicator",
            "Independent variables, pp. 22–23",
            "“we used the following independent categories, each modeled separately as one dummy variable: new scale, original scale ... modified scale ... or not applicable.”",
            "measurement_scale_type_indicator_k(p) = 1 if p uses scale type k; 0 otherwise",
            "source scale-type categories",
            "full manuscript measures",
            "final manuscript or publication",
            "Several scale-type indicators may equal one.",
        ),
        add(
            cited,
            "Endogeneity bias",
            "endogeneity_threat_indicator",
            "Dependent variables, p. 18",
            "“For quantitative articles, we coded whether an article had endogeneity bias based on any of the 12 threats to estimate consistency identified by Antonakis et al. (2010).”",
            "endogeneity_threat_indicator(p) = 1 if any specified threat is coded for quantitative p; 0 otherwise",
            "12-threat coding framework; trained coder judgment",
            "full manuscript",
            "final manuscript or publication",
            "Source-applied methodology control, not a later citation outcome.",
        ),
        add(
            connect,
            "Topics connectivity",
            "topic_connectivity_betweenness",
            "Section 3.2.1, pp. 16–17, Eq. (1)",
            "“After publication projection, we calculate for each of them the betweenness centrality score”",
            "b_i = (1/2) * sum_{k != i} sum_{j != i,k} p_kj(i) / p_kj",
            "27-topic document-topic matrix; topic representation exceeds publication mean; overlap-count projection; Brandes algorithm",
            "focal paper title, abstract, keywords; T0-frozen topic model/assets; corpus records available by T0",
            "paper T0, using only a corpus and topic model frozen by T0",
            "The source uses a full 1980–2020 corpus; do not use that future-complete network for earlier focal papers.",
        ),
        add(
            connect,
            "Full-authorship social connectivity",
            "author_social_connectivity_betweenness",
            "Section 3.2.2, pp. 19–20; Eq. (1)",
            "“after overlap count publications projection we computed betweenness centrality score using Equation (1) to capture social connection between authors”",
            "author_social_connectivity(p) = b_p from the projected paper-author bipartite network using Eq. (1)",
            "binary paper-author links; overlap-count projection; Brandes algorithm",
            "focal-paper authors and author-paper records available by T0",
            "paper T0, using only author-publication history available by T0",
            "The source's 17,260-paper historical snapshot must be time-truncated for T0 use.",
        ),
        add(
            connect,
            "Co-authorship team social connectivity",
            "team_social_connectivity_betweenness",
            "Section 3.2.2, pp. 19–20; Eq. (1)",
            "“we constructed a two-mode network from a (11, 230 × 12, 393) matrix where papers are connected by shared teams”",
            "team_social_connectivity(p) = b_p from the projected paper-team bipartite network using Eq. (1)",
            "co-authored-paper team identities; overlap-count projection; Brandes algorithm",
            "focal paper team and prior co-authorship records",
            "paper T0, using only team/publication history available by T0",
            "Only co-authored publications are in this source application.",
        ),
        add(
            connect,
            "Journal dummy variables",
            "journal_indicator",
            "Section 3.2.3, p. 22",
            "“Ju = 1 if the publication p is published by the journal u; 0 otherwise.”",
            "journal_indicator_u(p) = 1 if p is published by journal u; 0 otherwise",
            "source's journal universe",
            "paper journal metadata",
            "publication",
            "Journal-context control.",
        ),
        add(
            connect,
            "Number of authors per publication",
            "author_count",
            "Section 3.2.3, p. 22",
            "“We included, as a count variable, the number of authors per publication.”",
            "author_count(p) = count(authors of p)",
            "byline counting rule",
            "paper byline",
            "final manuscript or publication",
            "Direct authorship control.",
        ),
        add(
            connect,
            "Number of cited references",
            "reference_count",
            "Section 3.2.3, p. 22",
            "“We added the number of cited references for each publication to control for this effect.”",
            "reference_count(p) = count(cited references in p)",
            "reference parsing rule",
            "paper reference list",
            "final manuscript or publication",
            "Direct paper-level reference control.",
        ),
        add(
            connect,
            "Top-author indicator",
            "top_author_indicator",
            "Section 3.2.3 and Appendix A, pp. 22, 39",
            "“Ap = 1 if at least one author of the publication p is a top author; 0 otherwise.”",
            "top_author_indicator(p) = 1 if any author of p belongs to the top-author set at T0; 0 otherwise",
            "H-index definition; top-author-set rule; dated author-metric snapshot",
            "paper authors and a T0-frozen author-metric snapshot",
            "paper T0",
            "The source defines top 20 using the full-period H-index; recompute the set without post-T0 records for T0 use.",
        ),
        add(
            linguistic,
            "CiteScore",
            "journal_citescore",
            "Table 1, p. 10",
            "“The CiteScore, measuring the average citations per document for a given journal over the past 3 years, is sourced from Scopus.”",
            "journal_citescore(p) = CiteScore of p's journal from a dated snapshot",
            "three-year CiteScore definition; snapshot date",
            "paper journal and contemporaneous CiteScore snapshot",
            "submission/publication T0",
            "Use a snapshot available by T0, not the source's later retrieval.",
        ),
        add(
            linguistic,
            "Accessibility",
            "open_access_indicator",
            "Table 1, p. 10",
            "“1 = open access, 0 = restricted access”",
            "open_access_indicator(p) = 1 if p has the Scopus open-access tag; 0 otherwise",
            "source includes gold, hybrid gold, green, and bronze OA",
            "paper access status",
            "publication",
            "Access-status control.",
        ),
        add(
            linguistic,
            "No. of authors",
            "author_count",
            "Table 1, p. 10",
            "“The total number of author(s) listed in the article’s byline is counted.”",
            "author_count(p) = count(authors in p byline)",
            "byline counting rule",
            "paper byline",
            "final manuscript or publication",
            "Direct authorship control.",
        ),
        add(
            linguistic,
            "Internationality",
            "international_coauthorship_indicator",
            "Table 1, p. 10",
            "“Each article is classified as either domestic (one country) or international (more than one country) based on the data of author affiliation(s).”",
            "international_coauthorship_indicator(p) = 1 if affiliation countries(p) > 1; 0 otherwise",
            "affiliation-country parsing rule",
            "paper author affiliations",
            "final manuscript or publication",
            "Paper-team context control.",
        ),
        add(
            linguistic,
            "Geographical origin",
            "first_author_continent_indicator",
            "Table 1, p. 10",
            "“Dummy variables are used to model the geographic features of the first author’s affiliation based on the division of continents”",
            "first_author_continent_indicator_k(p) = 1 if first author's affiliation is continent k; 0 otherwise",
            "Asia/Europe/America/Oceania; Oceania reference group",
            "first-author affiliation",
            "final manuscript or publication",
            "Categorical geographic control.",
        ),
        add(
            linguistic,
            "Funding",
            "funding_indicator",
            "Table 1, p. 10",
            "“1 = funded, 0 = nonfunded”",
            "funding_indicator(p) = 1 if p is classified funded; 0 otherwise",
            "funding classification rule",
            "paper funding details",
            "final manuscript or publication",
            "Funding-context control.",
        ),
        add(
            linguistic,
            "Scholar h-index",
            "first_author_h_index",
            "Table 1, p. 10",
            "“The first author’s h-index was collected from the author’s profile in Scopus.”",
            "first_author_h_index(p) = h_index(first_author(p), snapshot_t0)",
            "author disambiguation; dated Scopus snapshot",
            "first author and T0-frozen citation/publication profile",
            "paper T0",
            "The source collected values in March 2023; earlier-paper use must use a T0 snapshot.",
        ),
        add(
            linguistic,
            "Title length",
            "title_character_count_including_punctuation_spaces",
            "Table 1, p. 10",
            "“The total number of characters in each title, including punctuations and spaces, is counted.”",
            "title_character_count(p) = count(characters in title(p), including punctuation and spaces)",
            "character-count convention",
            "paper title",
            "submission or final publication",
            "This is the direct source-authorized treatment of punctuation: it is included in a title-character count, not a separate punctuation indicator.",
        ),
        add(
            linguistic,
            "No. of references",
            "reference_count",
            "Table 1, p. 10",
            "“The total number of cited references at the end of the article is counted.”",
            "reference_count(p) = count(cited references at end of p)",
            "reference-list parsing rule",
            "paper reference list",
            "final manuscript or publication",
            "Direct paper-level reference control.",
        ),
        add(
            linguistic,
            "Subfield",
            "applied_linguistics_subfield_indicator",
            "Table 1, p. 10",
            "“Dummy variables are used to model the subfields of applied linguistics based on Grabe’s (2012) definition”",
            "applied_linguistics_subfield_indicator_k(p) = 1 if p is subfield k; 0 otherwise",
            "seven source categories; others reference group",
            "full manuscript; source coding manual",
            "final manuscript or publication",
            "Content-coded categorical control.",
        ),
        add(
            linguistic,
            "Methodology",
            "methodology_orientation_indicator",
            "Table 1, p. 10",
            "“Dummy variables are used to model articles’ methodological orientations following the guidelines by Amini Farsani et al. (2021)”",
            "methodology_orientation_indicator_k(p) = 1 if p is methodology k; 0 otherwise",
            "nonempirical/quantitative/qualitative/mixed/research-synthesis; synthesis reference group",
            "full manuscript; source coding guidelines",
            "final manuscript or publication",
            "Content-coded categorical control.",
        ),
    ]


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_input() -> tuple[list[str], list[dict[str, str]]]:
    """Read the frozen batch input."""
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Batch 8 input has no header.")
        return reader.fieldnames, list(reader)


def validate_assets(rows: list[dict[str, str]]) -> None:
    """Verify the specified text and PDF hashes."""
    for row in rows:
        for path_field, hash_field in (
            ("text_path", "text_sha256"),
            ("pdf_path", "pdf_sha256"),
        ):
            if sha256(Path(row[path_field])) != row[hash_field]:
                raise ValueError(
                    f"Asset SHA-256 mismatch: {row['record_key']} {path_field}"
                )


def build_output(
    source_fields: list[str],
    source_rows: list[dict[str, str]],
    mentions: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Create source-review rows followed by separate indicator-mention rows."""
    output: list[dict[str, str]] = []
    for source in source_rows:
        disposition, notes = REVIEWS[source["record_key"]]
        output.append(
            {
                "row_type": "source_review",
                **source,
                "source_disposition": disposition,
                "source_notes": notes,
                **{field: "" for field in MENTION_FIELDS},
            }
        )
    for mention in mentions:
        blank_source = {field: "" for field in source_fields}
        blank_source["record_key"] = mention["record_key"]
        output.append({"row_type": "indicator_mention", **blank_source, **mention})
    return output


def validate(
    source_fields: list[str],
    inputs: list[dict[str, str]],
    output: list[dict[str, str]],
    mentions: list[dict[str, str]],
) -> None:
    """Validate source-field preservation and the strict mention contract."""
    reviews = [row for row in output if row["row_type"] == "source_review"]
    allowed = {
        "formula_or_application",
        "review_discovery_only",
        "no_relevant_indicator",
    }
    if len(inputs) != len(reviews) != 7 or set(REVIEWS) != {
        row["record_key"] for row in inputs
    }:
        raise ValueError("Unexpected batch-8 source review set.")
    for before, after in zip(inputs, reviews, strict=True):
        for field in source_fields:
            if (
                field not in {"source_disposition", "source_notes"}
                and before[field] != after[field]
            ):
                raise ValueError(f"Frozen field changed: {field}")
        if after["source_disposition"] not in allowed:
            raise ValueError("Invalid disposition.")
    keys = {row["record_key"] for row in inputs}
    if len(mentions) != 36:
        raise ValueError(f"Expected 36 mentions, found {len(mentions)}.")
    for mention in mentions:
        if (
            set(mention) != {"record_key", *MENTION_FIELDS}
            or mention["record_key"] not in keys
        ):
            raise ValueError("Invalid mention schema or record key.")
        if mention["requires_future"] != "false" or not all(
            mention[field] for field in MENTION_FIELDS
        ):
            raise ValueError("Mention violates the T0 or completeness contract.")


def write_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write a deterministic UTF-8 long-form output."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    inputs: list[dict[str, str]],
    output: list[dict[str, str]],
    mentions: list[dict[str, str]],
) -> None:
    """Write provenance and blind-review metadata."""
    reviews = [row for row in output if row["row_type"] == "source_review"]
    manifest: dict[str, Any] = {
        "artifact": OUTPUT_PATH.name,
        "artifact_sha256": sha256(OUTPUT_PATH),
        "batch": 8,
        "schema": "contextual_fulltext_extraction_h1_batch8_v4",
        "reviewer": "H1",
        "input": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "source_count": len(reviews),
        "indicator_mention_count": len(mentions),
        "output_row_count": len(output),
        "source_disposition_counts": dict(
            sorted(Counter(row["source_disposition"] for row in reviews).items())
        ),
        "text_and_pdf_sha256_verified": {
            row["record_key"]: {
                "text_sha256": row["text_sha256"],
                "pdf_sha256": row["pdf_sha256"],
            }
            for row in inputs
        },
        "blind_review_constraints": [
            "Read only this batch's brief, input, and seven specified local English full texts.",
            "Did not read AI, H2, or prior-batch output; did not use Qwen or Ollama.",
            "Retained only source-authorized paper-level applications that can be time-bounded to T0.",
            "Excluded citation outcomes and any future-complete network or metric snapshot.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Produce the independent H1 batch-8 source reviews and mentions."""
    source_fields, inputs = read_input()
    if len(inputs) != 7:
        raise ValueError(f"Expected 7 input rows, found {len(inputs)}.")
    validate_assets(inputs)
    mentions = make_mentions()
    output = build_output(source_fields, inputs, mentions)
    validate(source_fields, inputs, output, mentions)
    write_csv(["row_type", *source_fields, *MENTION_FIELDS], output)
    write_manifest(inputs, output, mentions)


if __name__ == "__main__":
    main()
