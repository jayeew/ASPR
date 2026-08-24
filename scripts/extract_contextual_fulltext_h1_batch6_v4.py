"""Write the independent H1 review for contextual full-text batch 6."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/home/jayee/workspace/ASPR")
SOURCE_DIR = (
    ROOT / "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs"
)
OUTPUT_DIR = ROOT / "outputs"
INPUT_PATH = SOURCE_DIR / "contextual_fulltext_extraction_input_batch6_v4.csv"
SOURCE_REVIEW_PATH = (
    OUTPUT_DIR / "contextual_fulltext_source_review_H1_batch6_completed_v4.csv"
)
MENTIONS_PATH = (
    OUTPUT_DIR / "contextual_fulltext_indicator_mentions_H1_batch6_completed_v4.csv"
)
MANIFEST_PATH = OUTPUT_DIR / "contextual_fulltext_H1_batch6_completed_v4.manifest.json"

SOURCE_COLUMNS = [
    "record_key",
    "doi",
    "title",
    "text_path",
    "text_sha256",
    "pdf_path",
    "pdf_sha256",
    "linked_v3_feature_ids_json",
    "linked_v3_labels_json",
    "source_disposition",
    "source_notes",
    "H1_source_disposition",
    "H1_source_notes",
]
MENTION_COLUMNS = [
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
    """Return the SHA-256 digest of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_for(record_key: str) -> tuple[str, str]:
    """Return the source-level H1 disposition and concise reason."""
    reviews = {
        "doi:10.1007/s11192-022-04509-0": (
            "no_relevant_indicator",
            "Explicit variables are scholar-career aggregates (e.g., Early-bloom, career-years 6–20 MNCS) and outcomes use later publications/citations; none is a focal-paper T0 indicator.",
        ),
        "doi:10.1007/s11192-024-04928-1": (
            "formula_or_application",
            "Applies the paper-record author number directly from PubMed to analyze publication-level authorship; author count is computable from the byline at paper T0.",
        ),
        "doi:10.1007/s11192-024-05132-x": (
            "no_relevant_indicator",
            "Defines field/concept-set distances, longitudinal topic drift, and author-level network measures, not a focal-paper T0 indicator; temporal analyses also rely on evolving graph snapshots.",
        ),
        "doi:10.1007/s11192-025-05455-3": (
            "formula_or_application",
            "Explicitly applies paper-level manuscript variables, framed as observable at double-blind submission, for citation prediction. Future citation-count and SJR response variables are excluded from H1 mentions.",
        ),
    }
    return reviews[record_key]


def mention(
    record_key: str,
    raw_name: str,
    canonical_name: str,
    evidence: str,
    formula: str,
    required_data: str,
    notes: str,
    parameters: str = "",
) -> dict[str, str]:
    """Build one schema-complete, paper-level H1 mention row."""
    return {
        "record_key": record_key,
        "raw_name_en": raw_name,
        "canonical_name_en": canonical_name,
        "source_role": "original_application",
        "formula_location": "Table 1, p. 6040",
        "evidence_span": evidence,
        "formula": formula,
        "parameters": parameters,
        "required_data": required_data,
        "maximum_information_time": "paper T0 / submission stage",
        "scope_role": "paper-level potential-impact/control candidate",
        "requires_future": "no",
        "extraction_notes": notes,
    }


def make_mentions() -> list[dict[str, str]]:
    """Return only direct, focal-paper, no-future-leakage operational applications."""
    authorship_key = "doi:10.1007/s11192-024-04928-1"
    prestige_key = "doi:10.1007/s11192-025-05455-3"
    rows = [
        {
            "record_key": authorship_key,
            "raw_name_en": "Author number per record",
            "canonical_name_en": "author_count",
            "source_role": "original_application",
            "formula_location": "Methods—Data retrieval, p. 1303",
            "evidence_span": "This returned a complete list of all distinct PubMed entries ... – Author number per record",
            "formula": "author_count(p) = number of listed authors in PubMed record p",
            "parameters": "complete author list required",
            "required_data": "paper byline / PubMed author list",
            "maximum_information_time": "paper T0 / publication metadata",
            "scope_role": "paper-level control candidate",
            "requires_future": "no",
            "extraction_notes": "The source extracts author number per record and verifies completeness of author lists; it does not establish a causal innovation claim.",
        },
        mention(
            prestige_key,
            "MeSH",
            "mesh_term_count",
            "MeSH | Numeric | Number of MeSH terms",
            "mesh_term_count(p) = number of MeSH terms assigned to p",
            "paper MeSH terms",
            "Direct Table 1 operational definition; use only a contemporaneously available controlled-vocabulary assignment.",
        ),
        mention(
            prestige_key,
            "Scores",
            "mesh_animal_domain_proportion",
            "Scores | Numeric | Proportion of MeSH terms: A, C and H",
            "animal_score(p) = count(MeSH terms in A) / mesh_term_count(p)",
            "paper MeSH terms and domain A membership",
            "The source defines Scores as proportions for A, C and H; this row records its A component.",
            "A = animal",
        ),
        mention(
            prestige_key,
            "Scores",
            "mesh_molecular_domain_proportion",
            "Scores | Numeric | Proportion of MeSH terms: A, C and H",
            "molecular_score(p) = count(MeSH terms in C) / mesh_term_count(p)",
            "paper MeSH terms and domain C membership",
            "The source defines Scores as proportions for A, C and H; this row records its C component.",
            "C = molecular",
        ),
        mention(
            prestige_key,
            "Scores",
            "mesh_human_domain_proportion",
            "Scores | Numeric | Proportion of MeSH terms: A, C and H",
            "human_score(p) = count(MeSH terms in H) / mesh_term_count(p)",
            "paper MeSH terms and domain H membership",
            "The source defines Scores as proportions for A, C and H; this row records its H component.",
            "H = human",
        ),
        mention(
            prestige_key,
            "Title",
            "title_word_count",
            "Title | Numeric | Length of the title in words",
            "title_word_count(p) = number of words in p title",
            "paper title",
            "Direct Table 1 operational definition.",
        ),
        mention(
            prestige_key,
            "References",
            "reference_count",
            "References | Numeric | Number of references",
            "reference_count(p) = number of references in p",
            "paper reference list",
            "Direct Table 1 operational definition; references are available in the submitted manuscript.",
        ),
        mention(
            prestige_key,
            "Age",
            "mean_reference_age_years",
            "Age | Numeric | Mean age of references in years",
            "mean_reference_age_years(p) = mean(publication_year(p) - publication_year(r)) for r in references(p)",
            "paper publication year and referenced-work publication years",
            "Direct Table 1 operational definition; no post-paper information is needed.",
        ),
        mention(
            prestige_key,
            "Length",
            "paper_length_pages",
            "Length | Numeric | Length of the paper in pages",
            "paper_length_pages(p) = page length of p",
            "submitted manuscript page count",
            "Direct Table 1 operational definition.",
        ),
        mention(
            prestige_key,
            "Triangle",
            "biomedical_triangle_position",
            "Triangle | Categorical | Position in the biomedical triangle (see Weber (2013))",
            "biomedical_triangle_position(p) = categorical position in the source's biomedical triangle",
            "paper MeSH-domain scores and the source's stated triangle classification",
            "The source applies this categorical field position but refers to Weber (2013) for its detailed geometric construction; no unstated formula is supplied here.",
        ),
        mention(
            prestige_key,
            "Year",
            "publication_year",
            "Year | Categorical | Year of publication",
            "publication_year(p) = publication year of p",
            "paper publication date",
            "Direct Table 1 operational definition; a field/time control, not novelty evidence.",
        ),
        mention(
            prestige_key,
            "Language",
            "language_rank",
            "English is assigned the rank one, bilingual papers the rank two and every paper in any other language the rank three.",
            "language_rank(p) = 1 if English; 2 if bilingual; 3 otherwise",
            "submitted manuscript language",
            "Explicit p. 6040–6041 application; preserves the source's ordinal coding.",
        ),
        mention(
            prestige_key,
            "Clinical",
            "clinical_paper_indicator",
            "Clinical | Binary | Clinical paper (Yes/No)",
            "clinical_paper_indicator(p) = 1 if p is clinical; 0 otherwise",
            "paper clinical classification",
            "Direct Table 1 operational definition.",
        ),
        mention(
            prestige_key,
            "Research",
            "research_article_indicator",
            "Research | Binary | Research article (Yes/No)",
            "research_article_indicator(p) = 1 if p is a research article; 0 otherwise",
            "paper article-type classification",
            "Direct Table 1 operational definition.",
        ),
        mention(
            prestige_key,
            "Publication",
            "publication_type_indicator",
            "Publication | Binary | Type of publication",
            "publication_type_indicator_k(p) = 1 if publication type k is present; 0 otherwise",
            "paper publication-type classification",
            "The source states that multiple publication types are treated as binary presence/absence variables (p. 6041).",
            "one indicator per publication type k",
        ),
    ]
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write deterministic UTF-8 CSV output."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Validate frozen assets and create the completed H1 artifacts."""
    with INPUT_PATH.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 4:
        raise ValueError(f"Expected 4 input rows, found {len(source_rows)}")

    reviewed_rows: list[dict[str, str]] = []
    for row in source_rows:
        if sha256(Path(row["text_path"])) != row["text_sha256"]:
            raise ValueError(f"Text hash mismatch for {row['record_key']}")
        if sha256(Path(row["pdf_path"])) != row["pdf_sha256"]:
            raise ValueError(f"PDF hash mismatch for {row['record_key']}")
        disposition, notes = review_for(row["record_key"])
        reviewed_rows.append(
            {**row, "H1_source_disposition": disposition, "H1_source_notes": notes}
        )

    mentions = make_mentions()
    if {row["record_key"] for row in mentions} - {
        row["record_key"] for row in source_rows
    }:
        raise ValueError("Mention contains a record key absent from the input")
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_csv(SOURCE_REVIEW_PATH, SOURCE_COLUMNS, reviewed_rows)
    write_csv(MENTIONS_PATH, MENTION_COLUMNS, mentions)

    manifest: dict[str, Any] = {
        "artifact": "contextual_fulltext_h1_batch6_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "source_review_path": str(SOURCE_REVIEW_PATH),
        "source_review_sha256": sha256(SOURCE_REVIEW_PATH),
        "indicator_mentions_path": str(MENTIONS_PATH),
        "indicator_mentions_sha256": sha256(MENTIONS_PATH),
        "source_rows": len(reviewed_rows),
        "indicator_mentions": len(mentions),
        "source_disposition_counts": {
            "formula_or_application": sum(
                row["H1_source_disposition"] == "formula_or_application"
                for row in reviewed_rows
            ),
            "review_discovery_only": sum(
                row["H1_source_disposition"] == "review_discovery_only"
                for row in reviewed_rows
            ),
            "no_relevant_indicator": sum(
                row["H1_source_disposition"] == "no_relevant_indicator"
                for row in reviewed_rows
            ),
        },
        "fulltext_text_and_pdf_sha256_verified": True,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Read only batch-6 input, brief, and the four specified English full texts/PDFs.",
            "Did not read AI, H2, or prior batch task outputs.",
            "Kept only direct source definitions/applications for focal-paper T0 candidates.",
            "Excluded future citation-count and SJR response variables.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
