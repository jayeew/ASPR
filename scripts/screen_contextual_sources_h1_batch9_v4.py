"""Create the independent H1 title-and-abstract screening for batch 9."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_source_screening_input_batch9_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch9_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch9_completed_v4.manifest.json"
)
H1_FIELDS: Final = ["H1_decision", "H1_rationale"]

INCLUDE_IDS: Final = {
    1,
    10,
    15,
    21,
    22,
    24,
    25,
    27,
    29,
    30,
    35,
    36,
    40,
    51,
    52,
    54,
    57,
    58,
    60,
    61,
    62,
    64,
    65,
    66,
    69,
    71,
    75,
    76,
    77,
    79,
    84,
    85,
    90,
    93,
    94,
    96,
    97,
    108,
    119,
}
UNCERTAIN_IDS: Final = {16, 20, 32, 45, 50, 53, 89, 92, 106, 116, 120}

INCLUDE_RATIONALES: Final[dict[int, str]] = {
    1: "Examines digitized research-quality performance measures and their use in university research governance; retain as a conceptual evaluation source.",
    10: "Reviews bibliometric measures of researcher success, including citations, h-index, g-index, impact factor, SNIP, and Eigenfactor.",
    15: "Compares article-level bibliometric and altmetric impact measures by research type, including field-weighted citation impact and percentiles.",
    21: "Scoping review of bibliometric-analysis reporting recommendations; useful methodological/review source for metric definitions and reproducibility.",
    22: "Tests a directly observable title-format feature against subsequent citations with paper, journal, and timing controls.",
    24: "Introduces decision trees for choosing simple versus field-normalized bibliometric indicators in research evaluation.",
    25: "Operationalizes paper-level cross-disciplinary research through author and reference diversity and tests policy-document uptake.",
    27: "Assesses coverage of open bibliographic data for academic evaluation across fields and application roles.",
    29: "Uses post-retraction citations to study citation behavior and the meaning of citation-based evaluation.",
    30: "Applies publication, first-authorship, affiliation, collaboration, journal-tier, and citation metrics to research-integrity concerns.",
    35: "Surveys the perceived evaluative roles of publication count, author count/order, and journal impact factor.",
    36: "Reviews and appraises research-impact indicators used in a REF pilot and discusses their measurement limitations.",
    40: "Systematically reviews definitions of public-health research policy impact, including bibliometric and use-based definitions.",
    51: "Provides a methodological analysis of journal impact factor versus article citations for individual-article assessment.",
    52: "Defines a doctoral student's research team as a co-authorship network and relates network cohesion and publication/citation measures to later outcomes.",
    54: "Develops discipline-weighted citation windows and cited half-life from citation and aging patterns by document format.",
    57: "Proposes a probability-based method to assess the citation merit of a set of scientific papers across field and set size.",
    58: "Examines peer-review percentile scores and bibliometric publication indicators, including limitations of using citations as impact.",
    60: "Defines a composite journal-rank method using five citation indices and estimates rank uncertainty.",
    61: "Operationalizes article-level data items, figure density, authors, pages, and references as components of an average publishable unit.",
    62: "Tests article and journal bibliometric factors, reader counts, social-media attention, and quality against citation influence.",
    64: "Proposes citation-impact characterizations based on knowledge-flow intensity, diffusion capacity, and transfer capacity.",
    65: "Models associations between paper-level popular-media attention, author/journal reputation, and scientific citations.",
    66: "Uses author, institution, and country collaboration-network measures and tests their association with productivity and citations.",
    69: "Defines a multi-component, value-based article-level impact framework combining social, scholarly, and societal metrics.",
    71: "Measures paper-level open-science practices (data, code, preprint) and their association with citations using controls.",
    75: "Constructs paper-level quality indicators and aggregate sustainability indices; retain for its explicit indicator framework.",
    76: "Applies computational readability and writing-style metrics to academic abstracts.",
    77: "Defines paper novelty occurrence, breadth, and distance and relates them to field-level industry publishing contribution.",
    79: "Introduces a paper-level semantic interdisciplinarity indicator and compares it with established diversity measures and citation impact.",
    84: "Bibliometric review explicitly surveys innovation metrics and their indicator families; retain as a review-discovery source.",
    85: "Tests article, author, journal, title, and presentation characteristics as drivers of article citations.",
    90: "Reviews and argues about the document property reflected by citation count; useful conceptual source on citation-based value.",
    93: "Research-publishing strategy report discusses publication-output incentives and their effects on journals, individuals, and institutions.",
    94: "Quantifies an article-text feature (academic vocabulary coverage) across journal ranks; retain as a textual-feature source.",
    96: "Discusses indicator-based academic evaluation and proposes an evaluative-inquiry concept; retain as contextual review evidence.",
    97: "Analyzes bibliometric infrastructures and quantitative indicators used in academic evaluation.",
    108: "Uses topic-modeling and Kullback-Leibler measures to operationalize text-based novelty, resonance, and impact.",
    119: "Title directly indicates a bibliometrics foundations source; retain for later full-text verification despite an absent abstract.",
}

UNCERTAIN_RATIONALES: Final[dict[int, str]] = {
    16: "The title suggests innovation support, but the abstract is absent and does not establish a scholarly-paper indicator or review.",
    20: "The supplied abstract is only acknowledgements, so relevance and any indicator definition cannot be determined from title/abstract evidence.",
    32: "Discusses academics' perceptions of bibliometrics, but the abstract does not establish an operational indicator definition or systematic review scope.",
    45: "Only a proceedings fragment is supplied; title/abstract evidence is insufficient to determine indicator content.",
    50: "The supplied text is acknowledgements only; the title is relevant to scientometrics but does not establish a usable definition or review.",
    53: "Describes ORCID/repository integration and impact assessment, but does not state an indicator definition or a paper-level operationalization.",
    89: "Internal interview report on views of metrics may offer contextual insight, but the abstract does not establish a definition or systematic review.",
    92: "Broad scholarly-publishing report may discuss publication/citation indicators, but the supplied description does not establish relevant operational content.",
    106: "Grant-allocation/gender review may contain evaluation concepts, but the abstract does not establish a focal-paper indicator or review of such indicators.",
    116: "The title suggests research-investment and productivity data, but no abstract is supplied to establish the indicator definitions or unit of analysis.",
    120: "The title suggests citation and authorship-credit mechanisms, but no abstract is supplied to establish a relevant paper-level indicator.",
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_for(index: int) -> tuple[str, str]:
    """Return one independent H1 title-and-abstract screening decision."""
    if index in INCLUDE_IDS:
        return "include_definition_or_review", INCLUDE_RATIONALES[index]
    if index in UNCERTAIN_IDS:
        return "uncertain", UNCERTAIN_RATIONALES[index]
    return (
        "exclude_not_relevant",
        "Title and abstract describe a substantive-domain study, field mapping, or general review without an evident scholarly-paper indicator definition or relevant indicator review.",
    )


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write deterministic UTF-8 CSV output."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Copy frozen input fields and append independent H1 screening fields."""
    with INPUT_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_columns = reader.fieldnames
        input_rows = list(reader)
    if input_columns is None:
        raise ValueError("Input CSV has no header")
    if len(input_rows) != 120:
        raise ValueError(f"Expected 120 input rows, found {len(input_rows)}")
    if INCLUDE_IDS & UNCERTAIN_IDS:
        raise ValueError("Decision sets overlap")
    if max(INCLUDE_IDS | UNCERTAIN_IDS) > len(input_rows):
        raise ValueError("Decision set contains an out-of-range row")

    output_rows = []
    for index, row in enumerate(input_rows, start=1):
        decision, rationale = decision_for(index)
        output_rows.append({**row, "H1_decision": decision, "H1_rationale": rationale})
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    write_csv(OUTPUT_PATH, [*input_columns, *H1_FIELDS], output_rows)

    counts = {
        decision: sum(row["H1_decision"] == decision for row in output_rows)
        for decision in (
            "include_definition_or_review",
            "exclude_not_relevant",
            "uncertain",
        )
    }
    manifest: dict[str, Any] = {
        "artifact": "contextual_source_screening_H1_batch9_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_rows": len(output_rows),
        "h1_field_count": len(H1_FIELDS),
        "decision_counts": counts,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Used only English titles and abstracts in the batch-9 input.",
            "Did not use AI/H2 or prior batch output as evidence.",
            "Used uncertain where title/abstract evidence did not establish relevance or an operational/review role.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
