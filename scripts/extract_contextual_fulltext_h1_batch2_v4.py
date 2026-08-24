from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1] / "innovation_impact_feature_selection/evidence_derived_v4_rebuild"
INPUT_PATH = ROOT / "outputs/contextual_fulltext_extraction_input_batch2_v4.csv"
BRIEF_PATH = ROOT / "CONTEXTUAL_FULLTEXT_EXTRACTION_BRIEF_V4.md"
REVIEW_PATH = ROOT / "outputs/contextual_fulltext_source_review_H1_batch2_v4.csv"
MENTIONS_PATH = ROOT / "outputs/contextual_fulltext_indicator_mentions_H1_batch2_v4.csv"
MANIFEST_PATH = ROOT / "outputs/contextual_fulltext_extraction_H1_batch2_v4.manifest.json"

MENTION_FIELDS = [
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mention(
    record_key: str,
    raw_name_en: str,
    canonical_name_en: str,
    source_role: str,
    formula_location: str,
    evidence_span: str,
    formula: str,
    parameters: str,
    required_data: str,
    maximum_information_time: str,
    scope_role: str,
    extraction_notes: str,
) -> dict[str, str]:
    return {
        "record_key": record_key,
        "raw_name_en": raw_name_en,
        "canonical_name_en": canonical_name_en,
        "source_role": source_role,
        "formula_location": formula_location,
        "evidence_span": evidence_span,
        "formula": formula,
        "parameters": parameters,
        "required_data": required_data,
        "maximum_information_time": maximum_information_time,
        "scope_role": scope_role,
        "requires_future": "false",
        "extraction_notes": extraction_notes,
    }


DECISIONS = {
    "doi:10.1016/j.ipm.2023.103323": (
        "formula_or_application",
        "Explicitly applies five static paper-level features, including abstract word count, separately from its later citation features; only the static vector is extracted.",
    ),
    "doi:10.1016/j.jeap.2023.101253": (
        "review_discovery_only",
        "Conceptual discussion of academic attention and rhetorical presentation; it supplies terminology but no explicit paper-level T0 indicator formula or applied coding scheme.",
    ),
    "doi:10.1017/cbo9781316161012": (
        "review_discovery_only",
        "Book-length overview of open access and humanities scholarship; discussion of altmetrics and scholarly communication is discovery context, not a source-authorized focal-paper formula/application.",
    ),
    "doi:10.1038/s41562-022-01351-5": (
        "no_relevant_indicator",
        "Defines country-to-country citational lensing using five-year citation flows and aggregate national text networks; it is neither a focal-paper T0 indicator nor usable without future citations.",
    ),
    "doi:10.1214/09-sts285": (
        "review_discovery_only",
        "Review and critique of citation statistics and h-index practice; it points to methodological sources but cannot independently authorize a focal-paper formula.",
    ),
    "doi:10.1371/journal.pbio.1002541": (
        "no_relevant_indicator",
        "Presents an article-level relative citation-ratio implementation, but its measure is derived from citation rates and therefore requires post-publication citation information.",
    ),
    "doi:10.1371/journal.pone.0005910": (
        "no_relevant_indicator",
        "Uses contemporaneous expert ratings and later citation/F1000 outcomes to compare assessment methods; no automatically computable focal-paper T0 indicator is defined or applied.",
    ),
    "doi:10.1371/journal.pone.0006022": (
        "no_relevant_indicator",
        "PCA of journal-level impact measures calculated from citation and usage logs; the measures are not paper-level T0 indicators.",
    ),
    "doi:10.1371/journal.pone.0120495": (
        "formula_or_application",
        "Explicit original application defines paper-level document characteristics and collaboration variables as independent variables; later citation and social-media counts are not extracted.",
    ),
    "doi:10.1371/journal.pone.0135095": (
        "formula_or_application",
        "Explicitly operationalizes reference-list disciplinary diversity and institutional collaboration at the paper level; later normalized citation impact is excluded.",
    ),
    "doi:10.1371/journal.pone.0199031": (
        "no_relevant_indicator",
        "Historical network study of bibliometric-research actors and citation-metric proposals; its aggregate longitudinal network measures are not focal-paper T0 indicators.",
    ),
    "doi:10.1371/journal.pone.0230416": (
        "formula_or_application",
        "Explicitly classifies data-availability statements into four paper-level categories and applies pre-publication author-reputation controls; citation-accrual outcomes are not extracted.",
    ),
    "doi:10.1371/journal.pone.0251493": (
        "formula_or_application",
        "Explicitly applies a text/reference-based article-level subfield classifier. The direct-citation comparator is not extracted because it includes incoming citations.",
    ),
    "doi:10.1371/journal.pone.0274693": (
        "no_relevant_indicator",
        "Measures policy relevance through citations accumulated in policy documents and journal reputation; both are post-publication/aggregate and are ineligible as T0 paper features.",
    ),
    "doi:10.1515/9783110255553": (
        "review_discovery_only",
        "Monograph reviewing journal-evaluation approaches and readability-related terminology; it does not independently apply a focal-paper T0 formula.",
    ),
    "doi:10.1609/icwsm.v6i1.14305": (
        "formula_or_application",
        "Explicit original application calculates Gunning Fog and Flesch readability indices on individual scientific abstracts; later virality outcomes are excluded.",
    ),
    "doi:10.2196/10961": (
        "no_relevant_indicator",
        "Bibliometric review of a conference/lab series based on later citations and task outcomes; no focal-paper T0 innovation, opportunity, or control indicator is defined.",
    ),
    "doi:10.22439/fs.v25i2.5578": (
        "no_relevant_indicator",
        "Historical and conceptual account of citation notation without an explicit paper-level T0 indicator definition or application.",
    ),
    "doi:10.3145/epi.2022.jul.11": (
        "review_discovery_only",
        "Theoretical proposal about publications and citation links; it reviews citation theories but provides no computable focal-paper T0 measure.",
    ),
    "doi:10.3389/fnhum.2013.00291": (
        "review_discovery_only",
        "Critical review of journal rank and its consequences; it offers context but no source-authorized paper-level T0 formula/application.",
    ),
    "doi:10.3389/frma.2016.00001": (
        "review_discovery_only",
        "Conceptual critique of the interpretation of citation indicators; no eligible focal-paper T0 indicator is defined or empirically applied.",
    ),
    "doi:10.48550/arxiv.2106.01083": (
        "formula_or_application",
        "Develops and applies a sentence-level rhetorical-move classification scheme for data-paper abstracts; this is a paper-text T0 coding application.",
    ),
    "doi:10.5465/annals.2017.0099": (
        "review_discovery_only",
        "Review of topic-modeling applications in management research; its discussed novelty and topic measures require the cited original studies for formula authorization.",
    ),
}

MENTIONS = [
    mention(
        "doi:10.1016/j.ipm.2023.103323",
        "Static features (number of authors, keywords, abstract length, paper length, references)",
        "Static bibliographic paper-attribute vector",
        "original_application",
        "Methods §3.1, Table 1, p. 5.",
        "“X3 Abstract length Word count of the abstract.”",
        "[number of authors, number of author-defined keywords, abstract word count, article page count, reference count]",
        "author count; author-defined keyword count; abstract word count; page count; reference count",
        "focal-paper metadata, abstract text, pagination, and reference list",
        "T0 (all five listed static features are available from the published paper and its metadata).",
        "t0_control",
        "The same table lists later citation features; those post-publication features and the citation-based target are intentionally excluded.",
    ),
    mention(
        "doi:10.1371/journal.pone.0120495",
        "Document characteristics and collaboration indicators (PG, NR, TI, AU, IN, CU)",
        "Bibliographic and collaboration control vector",
        "original_application",
        "Materials and Methods—Indicators and variables, pp. 4–5.",
        "“Number of characters in title [TI]: length of a document’s title in the number of characters.”",
        "[page count, reference count, title-character count, author count, distinct institution count, distinct country count]",
        "PG; NR; TI; AU; IN; CU",
        "focal-paper pagination, reference list, title text, author list, and author-affiliation addresses",
        "T0 (the source uses these as independent document/collaboration variables; its social-media and citation dependent variables are later outcomes).",
        "t0_control",
        "Definitions are explicitly given by the source. Citation and social-media coverage/density/intensity are excluded because their events accrue after publication.",
    ),
    mention(
        "doi:10.1371/journal.pone.0135095",
        "Variety, balance, and disparity of reference disciplines",
        "Reference-list disciplinary diversity components",
        "original_application",
        "Table 2, “Operationalisations of the attributes of diversity,” pp. 7–8.",
        "“Variety We use the number of distinctive WoS categories (n) cited in an article.”",
        "Variety = n; Balance = −(Σ_i p_i ln p_i)/ln(n); Disparity = [Σ_ij (1 − s_ij)]/[n(n − 1)], over represented reference disciplines.",
        "n; p_i; s_ij; d_ij = 1 − s_ij",
        "focal-paper reference list; cited-journal-to-WoS-category mapping; frozen category similarity matrix",
        "T0 (with a reference-classification and similarity snapshot fixed no later than publication).",
        "t0_innovation",
        "The source separately applies the three components to a focal paper’s references. Its later normalized-citation dependent variable is not extracted.",
    ),
    mention(
        "doi:10.1371/journal.pone.0135095",
        "Rao-Stirling diversity",
        "Rao–Stirling reference diversity",
        "original_application",
        "Data and Methods—Measures, p. 9.",
        "“The Rao-Stirling diversity indicator can be expressed as follows: Rao-Stirling diversity = Σ_ij p_i p_j d_ij.”",
        "Rao–Stirling = Σ_ij p_i p_j d_ij",
        "p_i; p_j; d_ij",
        "focal-paper reference-list category proportions and a frozen between-category distance matrix",
        "T0 (with the category mapping and distance matrix fixed at publication).",
        "t0_innovation",
        "Explicitly used as a reference-diversity benchmark; citation impact results are excluded as post-publication outcomes.",
    ),
    mention(
        "doi:10.1371/journal.pone.0135095",
        "National_collab, Internat_collab, and No_Collab",
        "Institutional collaboration geographic scope",
        "original_application",
        "Data and Methods—Control variables, p. 10.",
        "“National_collab takes value 1 if there are at least two different institutions from the same country.”",
        "National_collab = 1 if ≥2 institutions and one country; Internat_collab = 1 if ≥2 countries; No_Collab = 1 if one institution participates.",
        "institution count; country count",
        "focal-paper author-affiliation institution and country identifiers",
        "T0 (computed from the focal paper’s author affiliations).",
        "t0_opportunity",
        "Source explicitly constructs the three collaboration dummies. The study’s later citation outcome is not extracted.",
    ),
    mention(
        "doi:10.1371/journal.pone.0230416",
        "Data availability statement (DAS) category",
        "Data-availability statement category",
        "original_application",
        "Materials and methods—Data availability statements: Classification, Table 1, pp. 4–5.",
        "“We identified four categories of DAS, further described in Table 1.”",
        "0 = not available; 1 = available on request; 2 = available with paper/supplementary files; 3 = available in a repository.",
        "DAS category 0–3",
        "focal-paper data-availability statement text and, for category 3, repository-link/identifier evidence",
        "T0 (the statement is part of the published paper).",
        "t0_opportunity",
        "The paper applies this four-category coding to individual publications. Later citation advantage is intentionally excluded.",
    ),
    mention(
        "doi:10.1371/journal.pone.0230416",
        "Mean and median author H-index at publication time",
        "Author-team prior H-index aggregation",
        "original_application",
        "Materials and methods—Citation prediction, Independent variables, p. 8; Table 3, p. 9.",
        "“the mean and median H-index of an article’s authors at the time of publication”",
        "h_index_mean = mean of focal-paper authors’ H-indices at publication; h_index_median = median of those H-indices.",
        "per-author prior H-index; aggregation statistic",
        "author identities and citation records restricted to each author’s pre-publication record",
        "T0 (only when each author H-index is calculated from citations available by the focal paper’s publication date).",
        "t0_control",
        "The source explicitly frames the aggregation as measured at publication time. Its citation-accrual dependent variable is excluded.",
    ),
    mention(
        "doi:10.1371/journal.pone.0251493",
        "Article-level deep-learning subfield classification",
        "Article-level scientific subfield assignment",
        "original_application",
        "Methods—Deep learning, Table 1, pp. 6–7.",
        "“The model performed best when it was given the following features: authors’ affiliations, names of journals referenced in the bibliography, titles of references, publication abstract, publication keywords, publication title, and classification of publication references.”",
        "Character-based convolutional classifier maps the listed text/reference inputs to one of the Science-Metrix scientific subfields (softmax output).",
        "title; keywords; abstract; author affiliations; referenced journal/title text; reference-subfield vector; fixed trained model",
        "focal-paper metadata and reference list; pre-publication reference-subfield index; frozen trained classifier",
        "T0 if the classifier and reference-subfield index are frozen using records available by publication; the source’s retrospective corpus snapshot is not itself an event-time guarantee.",
        "t0_control",
        "Only the text/reference deep-learning application is extracted. The source’s direct-citation comparator includes incoming citations and is excluded.",
    ),
    mention(
        "doi:10.1609/icwsm.v6i1.14305",
        "Gunning Fog and Flesch readability indices",
        "Abstract readability indices",
        "original_application",
        "Readability Index Tests, p. 477.",
        "“We use two indices to compute the difficulty of an abstract: the Gunning Fog (Gunning 1952) and the Flesch indices (Flesch 1946).”",
        "Apply Gunning Fog and Flesch readability indices to the focal abstract; the source names but does not print the equations.",
        "word length; sentence length; syllabic/complex-word information as required by the named indices",
        "focal-paper abstract text",
        "T0 (computed from the published abstract text).",
        "t0_opportunity",
        "This is an explicit individual-abstract application. The source’s downloads, bookmarks, and citations are later outcomes and are excluded.",
    ),
    mention(
        "doi:10.48550/arxiv.2106.01083",
        "Rhetorical moves in data-paper abstracts",
        "Abstract rhetorical-move composition",
        "original_definition",
        "Method §3.2, Table 1, “Classification and definition of rhetorical moves in data paper abstracts.”",
        "“we manually classified all sentences in our paper sample to understand the distribution of rhetorical moves”",
        "Classify each abstract sentence as Introduction, Purpose, Method, Results, Conclusion, Data Description, Data Uses, Data Accessibility, or Related Research Article; summarize the resulting move presence/counts or shares.",
        "sentence move labels; sentence count; optional fractional count for multi-move sentences",
        "focal-paper abstract text and the published move-codebook definitions",
        "T0 (the source text is available at publication; coding must be performed without later outcome information).",
        "t0_opportunity",
        "The source develops the expanded scheme and applies it sentence-by-sentence. It is a coding specification, not a claim that any move predicts later citations.",
    ),
]


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        input_fields = reader.fieldnames
        if input_fields is None:
            raise ValueError("Input CSV has no header")
        rows = list(reader)

    keys = {row["record_key"] for row in rows}
    if keys != set(DECISIONS):
        raise ValueError("Decision keys do not exactly match input record keys")
    if any(row["record_key"] not in keys for row in MENTIONS):
        raise ValueError("Mention refers to a record absent from input")

    for row in rows:
        disposition, notes = DECISIONS[row["record_key"]]
        row["source_disposition"] = disposition
        row["source_notes"] = notes

    with REVIEW_PATH.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=input_fields)
        writer.writeheader()
        writer.writerows(rows)
    with MENTIONS_PATH.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=MENTION_FIELDS)
        writer.writeheader()
        writer.writerows(MENTIONS)

    source_text_hashes = {
        row["record_key"]: sha256(Path(row["text_path"])) for row in rows
    }
    manifest: dict[str, Any] = {
        "schema": "contextual_fulltext_extraction_h1_batch2_manifest_v4",
        "run": "contextual-fulltext-extraction-h1-independent-batch2-v4-20260819",
        "reviewer": "H1",
        "input_hashes": {
            "brief_sha256": sha256(BRIEF_PATH),
            "input_csv_sha256": sha256(INPUT_PATH),
            "source_text_sha256_by_record_key": source_text_hashes,
        },
        "output_hashes": {
            "source_review_csv_sha256": sha256(REVIEW_PATH),
            "indicator_mentions_csv_sha256": sha256(MENTIONS_PATH),
        },
        "source_count": len(rows),
        "source_disposition_counts": dict(sorted(Counter(row["source_disposition"] for row in rows).items())),
        "indicator_mention_count": len(MENTIONS),
        "guards": {
            "read_ai_or_h2_batch2_outputs": False,
            "used_qwen_or_ollama": False,
            "final_feature_decision_made": False,
            "reviews_used_as_formula_authority": False,
        },
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")


if __name__ == "__main__":
    main()
