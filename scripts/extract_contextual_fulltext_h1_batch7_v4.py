"""Create the independent H1 full-text indicator extraction for batch 7."""

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
    "contextual_fulltext_extraction_input_batch7_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_fulltext_extraction_H1_batch7_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_fulltext_extraction_H1_batch7_completed_v4.manifest.json"
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
        (
            "The source profiles DataCite data-record metadata and its coverage. Its fields "
            "describe datasets and repository records, not a focal scholarly paper; it does not "
            "supply a paper-level T0 innovation, potential-impact, opportunity, or control "
            "indicator."
        ),
    ),
    "doi:10.1016/j.joi.2018.07.005": (
        "review_discovery_only",
        (
            "The study critically discusses unusual combinations of cited references through "
            "landmark-paper case studies and author interviews. It supplies terminology and "
            "original-study leads, but does not define or apply a paper-level T0 formula itself."
        ),
    ),
    "doi:10.1016/j.joi.2018.12.005": (
        "formula_or_application",
        (
            "The source directly applies author-specified Chinese Library Classification (CLC) "
            "codes to individual papers. This paper-level categorical classification is observable "
            "by submission/publication and is retained as a contextual control."
        ),
    ),
    "doi:10.1016/j.joi.2026.101791": (
        "no_relevant_indicator",
        (
            "The source's explicit citation-network centralities and intent filtering rely on "
            "citation contexts and incoming network edges from citing papers. Those data arise "
            "after a focal paper's T0, so no T0-computable paper-level candidate is retained."
        ),
    ),
    "doi:10.1016/j.leaqua.2013.10.014": (
        "formula_or_application",
        (
            "The source explicitly applies paper-level reference, authorship, editorial, and "
            "content/design controls while modelling later citations. Only the directly applied "
            "controls that can be fixed no later than publication are retained; later citation "
            "outcomes and calendar-complete measures are not."
        ),
    ),
}


def mention(
    record_key: str,
    raw_name: str,
    canonical_name: str,
    location: str,
    evidence: str,
    formula: str,
    parameters: str,
    required_data: str,
    maximum_information_time: str,
    notes: str,
) -> dict[str, str]:
    """Build one source-authorized, T0-safe indicator mention."""
    return {
        "record_key": record_key,
        "raw_name_en": raw_name,
        "canonical_name_en": canonical_name,
        "source_role": "original_application",
        "formula_location": location,
        "evidence_span": evidence,
        "formula": formula,
        "parameters": parameters,
        "required_data": required_data,
        "maximum_information_time": maximum_information_time,
        "scope_role": "control",
        "requires_future": "false",
        "extraction_notes": notes,
    }


def make_mentions() -> list[dict[str, str]]:
    """Return every direct, focal-paper, T0-safe application in the batch texts."""
    classification_key = "doi:10.1016/j.joi.2018.12.005"
    citation_key = "doi:10.1016/j.leaqua.2013.10.014"
    return [
        mention(
            classification_key,
            "Chinese Library Classification (CLC) code of a journal article",
            "paper_clc_discipline_code",
            "Section 2.2.2, pp. 5–6",
            "“The CLC code is also applied to each journal article. This is generally done by "
            "asking authors to provide the CLC code when submitting their manuscript.”",
            "paper_clc_discipline_code(p) = CLC code supplied for paper p, after any editorial "
            "modification",
            "CLC version; author-supplied code; editorial-modification rule",
            "manuscript CLC code and final bibliographic record",
            "submission or final publication",
            "Categorical paper-discipline control. The source does not construct a paper-level "
            "multidisciplinarity score; its HHI is journal-level and is not retained.",
        ),
        mention(
            citation_key,
            "Article type",
            "article_type_indicator",
            "Independent variables, p. 18",
            "“We used k - 1 dummy variables to model article type, including quantitative, "
            "qualitative, theory, review, commentary/discussion, methodological, and "
            "agent-based simulation articles.”",
            "article_type_indicator_k(p) = 1 if paper p is classified as type k; 0 otherwise",
            "k types: quantitative, qualitative, theory, review, commentary/discussion, "
            "methodological, agent-based simulation; source uses k - 1 dummies",
            "full manuscript/article type",
            "final manuscript or publication",
            "The source supplies detailed tie-breaking coding rules for mixed-method and hybrid "
            "articles; retain those rules if reproducing its categories.",
        ),
        mention(
            citation_key,
            "Number of cited references",
            "reference_count",
            "Independent variables, p. 19",
            "“We tallied the number of cited references in the article as reported in the "
            "reference list (or footnoted in some cases).”",
            "reference_count(p) = count(references listed or footnoted in paper p)",
            "reference-list and footnote parsing rule",
            "final manuscript reference list and applicable footnotes",
            "final manuscript or publication",
            "Direct paper-level count; do not substitute citations received by the focal paper.",
        ),
        mention(
            citation_key,
            "Senior editor",
            "senior_editor_indicator",
            "Independent variables, p. 19",
            "“we used k-1 dummy variables to capture these effects”",
            "senior_editor_indicator_k(p) = 1 if paper p was handled under senior editor k; "
            "0 otherwise",
            "editorial assignment/tenure record; source combines Robert J. House and Henry L. "
            "Tosi as one category",
            "paper editorial metadata",
            "editorial decision or publication",
            "Editorial-context control. Use only an editor identity known by the stated cutoff.",
        ),
        mention(
            citation_key,
            "Number of authors",
            "author_count",
            "Independent variables, p. 19",
            "“The total number of coauthors listed in the by-line of the article.”",
            "author_count(p) = count(authors in the byline of paper p)",
            "byline counting rule",
            "paper byline",
            "final manuscript or publication",
            "Direct paper-level authorship control.",
        ),
        mention(
            citation_key,
            "Lagged author citations",
            "mean_lagged_author_citation_count",
            "Independent variables, pp. 19–20",
            "“We averaged the citations of all coauthors for a particular article to reflect how "
            "well the authors were collectively cited.”",
            "mean_lagged_author_citation_count(p) = mean(citations(a, before publication_year(p)) "
            "for a in authors(p))",
            "author identity resolution; prior-citation window ending before publication year",
            "paper authors and their citation records available before the paper's publication year",
            "immediately before paper publication year",
            "The paper labels the pre-1996 Scopus coverage a partial lag; do not use a later "
            "citation snapshot for a focal paper.",
        ),
        mention(
            citation_key,
            "Lagged author publications",
            "mean_lagged_author_publication_count",
            "Independent variables, p. 20",
            "“we tallied the number of articles authors had published and averaged this number "
            "for author teams.”",
            "mean_lagged_author_publication_count(p) = mean(publication_count(a, before "
            "publication_year(p)) for a in authors(p))",
            "author identity resolution; prior-publication window ending before publication year",
            "paper authors and their publication records available before the paper's publication year",
            "immediately before paper publication year",
            "Use a dated publication snapshot; the source specifies the team average, not a single "
            "lead author's count.",
        ),
        mention(
            citation_key,
            "Rank of university affiliation",
            "mean_author_affiliation_rank",
            "Independent variables, p. 20",
            "“for each author, we used the average QS ranking of the author’s university; we then "
            "average the scores across coauthors for each article to obtain a collective score.”",
            "mean_author_affiliation_rank(p) = mean(rank(affiliation(a), snapshot_t0) for a in "
            "authors(p))",
            "author affiliations; a university-ranking snapshot available by T0; author-to-"
            "affiliation matching rule",
            "paper authors, affiliations, and a contemporaneous ranking snapshot",
            "paper T0, using a ranking snapshot available by T0",
            "The source's 2008–2012 average is not T0-safe for its earlier papers. This retained "
            "form records its directly stated team-average operation but requires a contemporaneous "
            "snapshot in any T0 use.",
        ),
        mention(
            citation_key,
            "Issue number",
            "journal_issue_indicator",
            "Independent variables, p. 21",
            "“We used k - 1 dummy variables to control for unobserved heterogeneity due to issue "
            "number”",
            "journal_issue_indicator_k(p) = 1 if paper p appears in issue k; 0 otherwise",
            "journal issue assignment; source uses k - 1 dummy variables",
            "paper publication metadata",
            "publication",
            "Publication-context control. This is distinct from calendar-year article volume, which "
            "the source measures only after the year is complete and is not retained.",
        ),
        mention(
            citation_key,
            "School of leadership",
            "leadership_school_indicator",
            "Independent variables, p. 21",
            "“We used the following nine categories (modeled as k-1 dummy variables): trait, "
            "behavioral, contextual, contingency, relational, information processing ...”",
            "leadership_school_indicator_k(p) = 1 if paper p is coded in school k; 0 otherwise",
            "paper theoretical content; source category and tie-breaking rules; source uses k - 1 "
            "dummy variables",
            "full manuscript",
            "final manuscript or publication",
            "Content-coded control. The source combines skeptics with information processing and "
            "defines its hybrid category explicitly.",
        ),
        mention(
            citation_key,
            "Statistical analysis method used for hypothesis testing",
            "hypothesis_test_method_indicator",
            "Independent variables, pp. 21–22",
            "“we used the following independent categories, each modeled separately as one dummy "
            "variable: correlation analysis, analysis of variance ... regression, structural "
            "equation modeling ...”",
            "hypothesis_test_method_indicator_k(p) = 1 if paper p uses method k; 0 otherwise",
            "full manuscript methods; source categories and inclusion rules",
            "full manuscript",
            "final manuscript or publication",
            "Multiple method indicators may equal one because the source says an article may use "
            "more than one method.",
        ),
        mention(
            citation_key,
            "Number of studies",
            "study_count_category",
            "Independent variables, p. 22",
            "“We used the following five categories (modeled as k-1 dummy variables): one, two, "
            "three, four or more studies, or not applicable.”",
            "study_count_category_k(p) = 1 if paper p belongs to study-count category k; 0 otherwise",
            "full manuscript; category rule for one, two, three, four-or-more, or not applicable",
            "full manuscript",
            "final manuscript or publication",
            "Categorical research-design control; preserve the source's capped four-or-more category.",
        ),
        mention(
            citation_key,
            "Sample location",
            "sample_location_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy "
            "variable: US, Europe, Asia, cross-national, others, or not applicable.”",
            "sample_location_indicator_k(p) = 1 if paper p contains sample location k; 0 otherwise",
            "full manuscript sample description; source location categories",
            "full manuscript",
            "final manuscript or publication",
            "Multiple location indicators may equal one because the source permits data from several locations.",
        ),
        mention(
            citation_key,
            "Study design",
            "study_design_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy "
            "variable: field survey, laboratory experiment, field experiment, quasi-experiment, "
            "archival data, meta-analysis, interview or others.”",
            "study_design_indicator_k(p) = 1 if paper p uses data-collection design k; 0 otherwise",
            "full manuscript methods; source study-design categories",
            "full manuscript",
            "final manuscript or publication",
            "Multiple design indicators may equal one because the source says an article may use "
            "more than one data-collection method.",
        ),
        mention(
            citation_key,
            "Data source",
            "data_source_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy "
            "variable: one source, two or more subjective sources, one or more objective source, "
            "or not applicable.”",
            "data_source_indicator_k(p) = 1 if paper p is coded in source category k; 0 otherwise",
            "full manuscript data description; source data-source categories",
            "full manuscript",
            "final manuscript or publication",
            "Research-design control; keep the source's subjective/objective categorization.",
        ),
        mention(
            citation_key,
            "Temporal context of study",
            "study_temporal_context_indicator",
            "Independent variables, p. 22",
            "“we used the following independent categories, each modeled separately as one dummy "
            "variable: cross-sectional, two time periods ... longitudinal ... or not applicable.”",
            "study_temporal_context_indicator_k(p) = 1 if paper p is coded in temporal context k; "
            "0 otherwise",
            "full manuscript methods; source temporal-context categories",
            "full manuscript",
            "final manuscript or publication",
            "The source defines two-time-period studies as non-repeated measurement at time one and two.",
        ),
        mention(
            citation_key,
            "Type of scale used",
            "measurement_scale_type_indicator",
            "Independent variables, pp. 22–23",
            "“we used the following independent categories, each modeled separately as one dummy "
            "variable: new scale, original scale ... modified scale ... or not applicable.”",
            "measurement_scale_type_indicator_k(p) = 1 if paper p uses scale type k; 0 otherwise",
            "full manuscript measures; source scale-type categories",
            "full manuscript",
            "final manuscript or publication",
            "Multiple scale-type indicators may equal one because the source says an article may use "
            "more than one scale.",
        ),
        mention(
            citation_key,
            "Endogeneity bias",
            "endogeneity_threat_indicator",
            "Dependent variables, p. 18",
            "“For quantitative articles, we coded whether an article had endogeneity bias based on "
            "any of the 12 threats to estimate consistency identified by Antonakis et al. (2010).”",
            "endogeneity_threat_indicator(p) = 1 if any of 12 specified consistency threats is "
            "coded for quantitative paper p; 0 otherwise",
            "full manuscript; the 12-threat coding framework cited by the source; trained coder judgment",
            "full manuscript",
            "final manuscript or publication",
            "Paper-methodology control. It is a source-applied binary coding, not a later citation outcome.",
        ),
    ]


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_input() -> tuple[list[str], list[dict[str, str]]]:
    """Load the frozen batch input."""
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Batch 7 input has no header.")
        return reader.fieldnames, list(reader)


def verify_assets(rows: list[dict[str, str]]) -> None:
    """Verify that all specified English text and PDF assets are frozen."""
    for row in rows:
        record_key = row["record_key"]
        if sha256(Path(row["text_path"])) != row["text_sha256"]:
            raise ValueError(f"Text SHA-256 mismatch for {record_key}")
        if sha256(Path(row["pdf_path"])) != row["pdf_sha256"]:
            raise ValueError(f"PDF SHA-256 mismatch for {record_key}")


def source_review_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Make source-review rows while changing only the authorized review fields."""
    completed: list[dict[str, str]] = []
    for row in rows:
        disposition, notes = REVIEWS[row["record_key"]]
        completed.append(
            {
                **row,
                "row_type": "source_review",
                "source_disposition": disposition,
                "source_notes": notes,
                **{field: "" for field in MENTION_FIELDS},
            }
        )
    return completed


def mention_rows(
    source_fields: list[str], mentions: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Make one long-form output row for each extracted indicator mention."""
    rows: list[dict[str, str]] = []
    for mention_row in mentions:
        row = {field: "" for field in source_fields}
        row["record_key"] = mention_row["record_key"]
        rows.append({"row_type": "indicator_mention", **row, **mention_row})
    return rows


def validate(
    source_fields: list[str],
    input_rows: list[dict[str, str]],
    output_rows: list[dict[str, str]],
    mentions: list[dict[str, str]],
) -> None:
    """Validate source preservation, mention schema, and T0 guardrails."""
    review_rows = [row for row in output_rows if row["row_type"] == "source_review"]
    allowed_dispositions = {
        "formula_or_application",
        "review_discovery_only",
        "no_relevant_indicator",
    }
    if len(input_rows) != 5 or len(review_rows) != len(input_rows):
        raise ValueError("Expected exactly five source-review rows.")
    if {row["record_key"] for row in input_rows} != set(REVIEWS):
        raise ValueError("Input record keys do not match the fixed H1 reviews.")
    for before, after in zip(input_rows, review_rows, strict=True):
        for field in source_fields:
            if (
                field not in {"source_disposition", "source_notes"}
                and before[field] != after[field]
            ):
                raise ValueError(f"Frozen source field changed: {field}")
        if after["source_disposition"] not in allowed_dispositions:
            raise ValueError("Invalid source disposition.")
    source_keys = {row["record_key"] for row in input_rows}
    if not mentions or any(row["record_key"] not in source_keys for row in mentions):
        raise ValueError("Mention record key is absent from the input.")
    for row in mentions:
        if set(row) != {"record_key", *MENTION_FIELDS}:
            raise ValueError("Unexpected indicator-mention schema.")
        if row["requires_future"] != "false":
            raise ValueError("A retained mention requires future information.")
        if not all(row[field] for field in MENTION_FIELDS):
            raise ValueError("A mention field is blank.")


def write_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the long-form completed extraction CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    input_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    mentions: list[dict[str, str]],
) -> None:
    """Write an auditable manifest for the blind batch-7 extraction."""
    manifest: dict[str, Any] = {
        "artifact": OUTPUT_PATH.name,
        "artifact_sha256": sha256(OUTPUT_PATH),
        "batch": 7,
        "blind_review_constraints": [
            "Read only the batch-7 brief, its input, and the five specified local English full texts.",
            "Did not read AI, H2, prior-batch outputs, Qwen, or Ollama results.",
            "Retained only source-authorized paper-level indicators usable no later than T0.",
            "Did not retain later citation outcomes, calendar-complete volume, or future citation-network measures.",
        ],
        "indicator_mention_count": len(mentions),
        "input": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_row_count": len(source_rows) + len(mentions),
        "reviewer": "H1",
        "schema": "contextual_fulltext_extraction_h1_batch7_v4",
        "source_count": len(source_rows),
        "source_disposition_counts": dict(
            sorted(Counter(row["source_disposition"] for row in source_rows).items())
        ),
        "text_and_pdf_sha256_verified": {
            row["record_key"]: {
                "pdf_sha256": row["pdf_sha256"],
                "text_sha256": row["text_sha256"],
            }
            for row in input_rows
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Create the blinded H1 batch-7 source reviews and extraction mentions."""
    source_fields, input_rows = read_input()
    if len(input_rows) != 5:
        raise ValueError(f"Expected 5 input rows, found {len(input_rows)}.")
    verify_assets(input_rows)
    mentions = make_mentions()
    reviews = source_review_rows(input_rows)
    output_rows = [*reviews, *mention_rows(source_fields, mentions)]
    output_fields = ["row_type", *source_fields, *MENTION_FIELDS]
    validate(source_fields, input_rows, output_rows, mentions)
    write_csv(output_fields, output_rows)
    write_manifest(input_rows, reviews, mentions)


if __name__ == "__main__":
    main()
