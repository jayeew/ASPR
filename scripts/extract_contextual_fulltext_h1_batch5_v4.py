"""Write the independent H1 batch-5 contextual full-text extraction."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_fulltext_extraction_input_batch5_v4.csv"
)
SOURCE_REVIEW_PATH: Final = ROOT / (
    "outputs/contextual_fulltext_source_review_H1_batch5_completed_v4.csv"
)
MENTIONS_PATH: Final = ROOT / (
    "outputs/contextual_fulltext_indicator_mentions_H1_batch5_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_fulltext_H1_batch5_completed_v4.manifest.json"
)

SOURCE_FIELDS: Final = ["H1_source_disposition", "H1_source_notes"]
MENTION_FIELDS: Final = [
    "record_key",
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

SOURCE_REVIEWS: Final[dict[str, dict[str, str]]] = {
    "doi:10.1002/asi.22715": {
        "H1_source_disposition": "no_relevant_indicator",
        "H1_source_notes": "The paper creates document-by-MeSH matrices for later mapping, but MeSH is assigned through MEDLINE indexing and the text supplies no source-authorized publication-time availability rule for a paper-level T0 feature.",
    },
    "doi:10.1007/978-3-030-75263-7": {
        "H1_source_disposition": "review_discovery_only",
        "H1_source_notes": "This edited volume provides conceptual and descriptive peer-review context; no directly authorized, paper-level T0 indicator formula or operational application was identified for the linked leads.",
    },
    "doi:10.1007/bf02017063": {
        "H1_source_disposition": "formula_or_application",
        "H1_source_notes": "The study explicitly classifies paper outputs by publication language, including an operational internationality rule; that paper-level categorical control is retained below.",
    },
    "doi:10.1007/s00799-021-00305-y": {
        "H1_source_disposition": "formula_or_application",
        "H1_source_notes": "The source defines an unsupervised, text-input paper-topic classifier with stated thresholds and a paper-topic relevance score; assets must be version-frozen at T0.",
    },
    "doi:10.1007/s10961-016-9550-z": {
        "H1_source_disposition": "no_relevant_indicator",
        "H1_source_notes": "The operational objects are Italian patent applications and inventor-pair collaborations, not paper-level indicators, so no candidate meets this batch's paper-level inclusion rule.",
    },
    "doi:10.1007/s11135-022-01480-z": {
        "H1_source_disposition": "formula_or_application",
        "H1_source_notes": "Table 1 directly applies multiple paper-level static text, reference, funding, authorship, and document-structure controls. Post-publication citation and profile variables are excluded from mentions.",
    },
    "doi:10.1007/s11192-017-2630-5": {
        "H1_source_disposition": "formula_or_application",
        "H1_source_notes": "The article operationalizes paper review-history indicators and static paper controls. Retained timing is no later than editorial acceptance/final publication, not future citation observation.",
    },
    "doi:10.1007/s11192-019-03018-x": {
        "H1_source_disposition": "review_discovery_only",
        "H1_source_notes": "This is a conceptual review of bibliometrics-based heuristics and cites possible measures, but it supplies no source-authorized paper-level T0 formula/application for the linked leads.",
    },
    "doi:10.1007/s11192-019-03263-0": {
        "H1_source_disposition": "no_relevant_indicator",
        "H1_source_notes": "Its originality score is explicitly based on subsequent citing papers and their links to the focal paper's references; it therefore requires future information and is excluded from T0 mentions.",
    },
    "doi:10.1007/s11192-021-04128-1": {
        "H1_source_disposition": "formula_or_application",
        "H1_source_notes": "The paper applies named, paper-level metadata counts as publication predictors. Corpus-trained topic mixtures and researcher-survey variables are not retained as T0 paper-level mentions.",
    },
    "doi:10.1057/s41599-024-02915-8": {
        "H1_source_disposition": "formula_or_application",
        "H1_source_notes": "RS/DIV use a citation matrix built with the entire OpenAlex dataset and are excluded for future-data dependence. The same source directly applies paper team size and reference count, which are retained as T0 controls.",
    },
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
    maximum_information_time: str = "publication",
    scope_role: str = "context_control",
    source_role: str = "original_application",
) -> dict[str, str]:
    """Build one schema-conformant, retained indicator mention."""
    return {
        "record_key": record_key,
        "raw_name_en": raw_name,
        "canonical_name_en": canonical_name,
        "source_role": source_role,
        "formula_location": location,
        "evidence_span": evidence,
        "formula": formula,
        "parameters": parameters,
        "required_data": required_data,
        "maximum_information_time": maximum_information_time,
        "scope_role": scope_role,
        "requires_future": "0",
        "extraction_notes": "Retained because the source directly applies a paper-level operation using information available no later than the stated T0.",
    }


MENTIONS: Final[list[dict[str, str]]] = [
    mention(
        "doi:10.1007/bf02017063",
        "Internationality by language of publication",
        "publication_language_internationality",
        "Assessment of 'scientific' output, p. 425",
        "“all publications in languages other than Dutch are international, as well as papers written in Dutch, but published in Belgium and former Dutch colonies.”",
        "international = 1 under the stated language-and-location rule; otherwise 0.",
        "reference language= Dutch; specified Dutch-language publication locations=Belgium/former Dutch colonies",
        "paper language; publication country/location",
        "publication",
    ),
    mention(
        "doi:10.1007/s00799-021-00305-y",
        "CSO Classifier topic set",
        "cso_classifier_topic_set",
        "Sections 4.1–4.3, pp. 95–99",
        "“takes as input the textual components of a research paper (usually title, abstract, and keywords) and outputs the relevant topics drawn from CSO.”",
        "TopicSet = postprocess(S_syntactic union S_semantic): syntactic topics have Levenshtein similarity >= msm=0.94; semantic candidates use top-ten embedding neighbours with cosine similarity > 0.7, then outlier filtering and super-topic enhancement.",
        "msm=0.94; semantic neighbour count=10; cosine threshold=0.7; generic-word cutoff n=3,000; frozen CSO, embeddings, and classifier version",
        "paper title, abstract, keywords; frozen CSO ontology; frozen word embeddings/classifier assets",
        "paper submission/publication, using only a classifier and ontology version frozen at T0",
        "t0_substantive",
        "original_definition",
    ),
    mention(
        "doi:10.1007/s00799-021-00305-y",
        "CSO semantic topic relevance score",
        "cso_semantic_topic_relevance_score",
        "Section 4.2.4, p. 97",
        "“The relevance score of a topic is computed as the product between the number of times it was identified (frequency) and the number of unique n-grams that led to it (diversity).”",
        "relevance(topic) = frequency(topic) * unique_trigger_ngram_count(topic); a directly mentioned topic receives the maximum score found.",
        "direct-mention override=max observed topic score; frozen classifier version",
        "paper title, abstract, keywords; frozen CSO classifier output and trigger n-grams",
        "paper submission/publication, using only a classifier version frozen at T0",
        "t0_substantive",
        "original_definition",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Abstract share from paper title",
        "abstract_title_word_repetition_share",
        "Table 1, p. 3693",
        "“Frequency of repetition of title words in abstract divided by abstract length (without stop-words)”.",
        "count(title words repeated in abstract) / abstract_length_without_stopwords.",
        "stop-word list; tokenization and repetition convention",
        "paper title and abstract",
        "final manuscript/publication",
        "t0_substantive",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Title share from abstract",
        "title_word_coverage_by_abstract",
        "Table 1, p. 3693",
        "“No. of title words in abstract divided by title length (without stop-words)”.",
        "count(title words occurring in abstract) / title_length_without_stopwords.",
        "stop-word list; tokenization and occurrence convention",
        "paper title and abstract",
        "final manuscript/publication",
        "t0_substantive",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Average age of the references",
        "average_reference_age",
        "Table 1, p. 3694",
        "“Average year of publication of references for each document minus the year of publication of that document”.",
        "mean(publication_year(reference)) - publication_year(focal paper), exactly as stated by the source.",
        "reference list; reference publication years",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "No. of references",
        "reference_count",
        "Table 1, p. 3694",
        "“No. of references”.",
        "count(references in focal paper).",
        "reference-list parsing rule",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Article length",
        "article_page_count",
        "Table 1, p. 3694",
        "“Article length: No. of article pages”.",
        "count(article pages).",
        "start and end page or final document pagination",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Co-authors",
        "author_count",
        "Table 1, p. 3694",
        "“Co-authors: No. of authors”.",
        "count(authors on focal paper).",
        "authorship byline",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Author's self-citations in references",
        "author_self_citation_reference_count",
        "Table 1, p. 3694",
        "“No. of target article’s references which have been written by at least one of the target article’s authors”.",
        "count(references sharing at least one author with the focal paper).",
        "focal-paper authors; reference authors; author-identity matching rule",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Journal self-citations in references",
        "journal_self_citation_reference_count",
        "Table 1, p. 3694",
        "“No. of times the target journal is cited in the references of the article”.",
        "count(references published in the focal paper's journal).",
        "focal journal identity; reference outlet metadata",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "No. of national funding organizations",
        "national_funding_organization_count",
        "Table 1, p. 3694",
        "“No. of national funding organizations”.",
        "count(national funding organizations acknowledged for focal paper).",
        "funding/acknowledgement metadata; focal-country convention",
        "publication",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "No. of international funding organizations",
        "international_funding_organization_count",
        "Table 1, p. 3694",
        "“No. of international funding organizations”.",
        "count(international funding organizations acknowledged for focal paper).",
        "funding/acknowledgement metadata; focal-country convention",
        "publication",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Title length",
        "title_word_count_excluding_punctuation",
        "Table 1, p. 3694",
        "“Title length: Number of title words without punctuation”.",
        "count(title words after removing punctuation).",
        "title text; punctuation and tokenization convention",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Abstract length",
        "abstract_character_count_excluding_spaces_punctuation",
        "Table 1, p. 3694",
        "“Number of abstract characters irrespective of space character and punctuation”.",
        "count(abstract characters after excluding spaces and punctuation).",
        "abstract text; character and punctuation convention",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "No. of figures",
        "figure_count",
        "Table 1, p. 3694",
        "“No. of figures”.",
        "count(figures in focal paper).",
        "final paper structure or document-texture extraction",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "No. of tables",
        "table_count",
        "Table 1, p. 3694",
        "“No. of tables”.",
        "count(tables in focal paper).",
        "final paper structure or document-texture extraction",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Title has punctuations",
        "title_punctuation_presence",
        "Table 1, p. 3695",
        "“Title has punctuations” is recorded as “Logical”.",
        "logical indicator for title punctuation presence (source does not specify the punctuation inventory).",
        "punctuation inventory; title text",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "At least one author from US",
        "has_us_affiliated_author",
        "Table 1, p. 3695",
        "“At least one author from US” is recorded as “Logical”.",
        "logical indicator for whether at least one focal-paper author is from the US.",
        "author affiliation(s); country normalization rule",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Document has formulae",
        "document_formula_presence",
        "Table 1, p. 3695",
        "“Document has formulae” is recorded as “Logical”.",
        "logical indicator for formulae presence in the focal document.",
        "final document text/layout; formula-detection rule",
        "publication",
        "t0_substantive",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Title contains a question",
        "title_question_presence",
        "Table 1, p. 3695",
        "“Title contains a question” is recorded as “Logical”.",
        "logical indicator for a question in the document title.",
        "title text; question-mark/question-classification rule",
        "publication",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Open-access paper",
        "open_access_status_at_publication",
        "Table 1, p. 3695",
        "“Open-access paper: Is the paper open access?” is recorded as “Logical”.",
        "logical indicator for the paper's open-access status.",
        "publisher/indexed open-access status recorded at publication",
        "publication",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11135-022-01480-z",
        "Document type",
        "document_type_category",
        "Table 1, p. 3695",
        "“Document type” has the categories “Review” and “Article”.",
        "categorical document type: Review or Article, as recorded in Web of Science.",
        "Web of Science document-type metadata",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Turnaround time",
        "peer_review_turnaround_time_days",
        "Variables section, p. 1093",
        "“the amount of time the paper spends in review from the point at which it is received for review to the point at which it is accepted.”",
        "acceptance_date - received_for_review_date.",
        "received-for-review date; acceptance date",
        "editorial acceptance/final publication",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Original decision",
        "initial_editorial_decision_category",
        "Variables section, p. 1093",
        "“immediate accept, requiring minor revisions, requiring major revisions or being in a revise and resubmit category.”",
        "initial decision in {Accept, Minor Revision, Major Revision, Reject and Resubmit}.",
        "first editorial decision record",
        "first editorial decision",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Revision Effort",
        "revision_effort_category",
        "Review process: revision effort, p. 1097",
        "“combining the accept and minor revision papers into one set of low effort papers and the major revision and revise and resubmit papers into another set of high effort papers.”",
        "low if initial decision in {Accept, Minor Revision}; high if initial decision in {Major Revision, Reject and Resubmit}.",
        "first editorial decision record",
        "first editorial decision",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Number of submissions",
        "submission_count",
        "Data set fields, p. 1094",
        "“the number of submissions”.",
        "count(submissions recorded by the publisher for the focal manuscript).",
        "publisher manuscript-history record",
        "editorial acceptance/final publication",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Count of non-editor reviews",
        "non_editor_reviewer_count",
        "Data set fields, p. 1094",
        "“number of reviewers in addition to editor involved in the review of the paper.”",
        "count(reviewers other than the editor involved in review).",
        "publisher manuscript-history record",
        "editorial acceptance/final publication",
        "t0_opportunity",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Number of authors",
        "author_count",
        "Variables section, p. 1093; data cross-check, p. 1094",
        "“the count of authors” and “Number of Authors”.",
        "count(authors on focal paper).",
        "authorship byline or Scopus author metadata",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Article length",
        "article_length",
        "Variables section, p. 1093",
        "“The length of the paper was also considered”.",
        "paper length as recorded for the focal article (the source does not fix a unit).",
        "final paper pagination or full text; chosen length unit",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-017-2630-5",
        "Count of keywords",
        "keyword_count",
        "Variables section, p. 1093",
        "“the effect of the number of keywords”.",
        "count(keywords attached to focal paper).",
        "paper keyword metadata; keyword delimiter rule",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-021-04128-1",
        "Paper length (number of pages)",
        "article_page_count",
        "Table 11, p. 8405",
        "“Paper length (number of pages)”.",
        "count(article pages).",
        "start/end page or final document pagination",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-021-04128-1",
        "Number of authors",
        "author_count",
        "Table 11, p. 8405",
        "“Number of authors”.",
        "count(authors on focal paper).",
        "authorship byline or Scopus metadata",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-021-04128-1",
        "Number of figures",
        "figure_count",
        "Table 11, p. 8405",
        "“Number of figures”.",
        "count(figures in focal paper).",
        "final paper structure or full text",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-021-04128-1",
        "Number of references",
        "reference_count",
        "Table 11, p. 8405",
        "“Number of references”.",
        "count(references in focal paper).",
        "reference list",
        "publication",
    ),
    mention(
        "doi:10.1007/s11192-021-04128-1",
        "Colons in title",
        "title_colon_presence",
        "Table 11, p. 8405",
        "“Colons in title”.",
        "indicator for colon occurrence in the focal title.",
        "title text; colon-character convention",
        "publication",
    ),
    mention(
        "doi:10.1057/s41599-024-02915-8",
        "Team size",
        "author_count",
        "Variables and regression models, p. 4",
        "“team_size_i the number of co-authors in publication i”.",
        "team_size_i = count(co-authors in publication i).",
        "paper authorship metadata",
        "publication",
    ),
    mention(
        "doi:10.1057/s41599-024-02915-8",
        "References count",
        "reference_count",
        "Variables and regression models, p. 4",
        "“references_count_i the number of references in publication i”.",
        "references_count_i = count(references in publication i).",
        "paper reference list",
        "publication",
    ),
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input() -> tuple[list[dict[str, str]], list[str]]:
    """Read the frozen batch input rows."""
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader), reader.fieldnames


def validate_input(rows: list[dict[str, str]]) -> None:
    """Check input coverage and full-text immutability before writing results."""
    if len(rows) != len(SOURCE_REVIEWS):
        raise ValueError(
            f"Expected {len(SOURCE_REVIEWS)} input rows, found {len(rows)}"
        )
    actual_keys = {row["record_key"] for row in rows}
    if actual_keys != set(SOURCE_REVIEWS):
        raise ValueError("Input record keys do not match the H1 review registry")
    for row in rows:
        if sha256_file(Path(row["text_path"])) != row["text_sha256"]:
            raise ValueError(f"Text SHA mismatch for {row['record_key']}")
        if sha256_file(Path(row["pdf_path"])) != row["pdf_sha256"]:
            raise ValueError(f"PDF SHA mismatch for {row['record_key']}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> str:
    """Write a CSV and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return sha256_file(path)


def write_manifest(
    source_rows: list[dict[str, str]], source_sha: str, mentions_sha: str
) -> None:
    """Write H1 provenance, counts, and artifact hashes."""
    manifest = {
        "input": str(INPUT_PATH),
        "input_sha256": sha256_file(INPUT_PATH),
        "source_review_artifact": str(SOURCE_REVIEW_PATH),
        "source_review_sha256": source_sha,
        "indicator_mentions_artifact": str(MENTIONS_PATH),
        "indicator_mentions_sha256": mentions_sha,
        "source_row_count": len(source_rows),
        "indicator_mention_count": len(MENTIONS),
        "fulltext_text_and_pdf_sha256_verified": True,
        "qwen_or_ollama_used": False,
        "review_scope": "Independent H1 batch-5 extraction; retains only paper-level, source-authorized, T0-computable indicators without future-data dependence.",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Build independent H1 source review and retained indicator mentions."""
    source_rows, input_fields = read_input()
    validate_input(source_rows)
    h1_source_rows = [
        {**row, **SOURCE_REVIEWS[row["record_key"]]} for row in source_rows
    ]
    source_sha = write_csv(
        SOURCE_REVIEW_PATH, [*input_fields, *SOURCE_FIELDS], h1_source_rows
    )
    mentions_sha = write_csv(MENTIONS_PATH, MENTION_FIELDS, MENTIONS)
    write_manifest(source_rows, source_sha, mentions_sha)


if __name__ == "__main__":
    main()
