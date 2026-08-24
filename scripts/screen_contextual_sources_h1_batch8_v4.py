"""Create the independent H1 title-and-abstract screen for batch 8."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_source_screening_input_batch8_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch8_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch8_completed_v4.manifest.json"
)
H1_FIELDS: Final = ["H1_decision", "H1_rationale"]

INCLUDE_RATIONALES: Final[dict[str, str]] = {
    "doi:10.1016/j.ipm.2009.06.003": (
        "Include: the title explicitly concerns validation and improvement of subject-classification schemes, a paper-level topical assignment pathway relevant to subsequent indicator definition."
    ),
    "doi:10.1016/j.joi.2014.09.005": (
        "Include: this overview explicitly evaluates altmetrics as indicators of broader research impact, making it a review lead for paper-level impact measures."
    ),
    "doi:10.1016/j.joi.2018.07.005": (
        "Include: the title explicitly tests whether a paper's cited references reflect its creative potential, directly linking a paper-level reference feature to innovation."
    ),
    "doi:10.1016/j.joi.2018.12.005": (
        "Include: the title explicitly compares journal and paper-level classifications of science, a direct source for defining or validating paper-level field/topic assignments."
    ),
    "doi:10.1016/j.joi.2021.101159": (
        "Include: the title reports convergent validity of disruptiveness indicators against expert milestone assignments to physics papers, a direct validation of a paper-level innovation measure."
    ),
    "doi:10.1016/j.leaqua.2013.10.014": (
        "Include: the title asks what makes articles highly cited, indicating an application of article-level characteristics as potential-impact predictors."
    ),
    "doi:10.1016/j.respol.2026.105451": (
        "Include: the abstract critically reassesses the citation-based CD index for individual publications and its ability to measure disruptiveness/scientific progress."
    ),
    "doi:10.1017/s0272263124000743": (
        "Include: the abstract identifies 11 journal-, author-, and article-level factors predicting time-normalized citations of highly cited articles, explicitly including title length, authors, accessibility, and references."
    ),
    "doi:10.1093/reseval/rvae038": (
        "Include: the abstract explicitly develops a theoretical definition of research quality and discusses its empirical methodological consequences, making it a quality-definition review lead."
    ),
    "doi:10.1101/518605": (
        "Include: the abstract compares diversity components (Rao–Stirling, variety, balance, disparity) and paper-level collaboration/reference characteristics with publication impact."
    ),
    "doi:10.1108/lht-10-2023-0514": (
        "Include: the abstract systematically reviews AI-enhanced scientometrics/bibliometrics, explicitly including research-impact prediction and publication-level analysis methods."
    ),
    "doi:10.1109/access.2019.2906106": (
        "Include: the abstract constructs citation-network and similarity matrices from a given article and its references to recommend a publication venue, a direct paper-level network application."
    ),
    "doi:10.1109/access.2022.3159025": (
        "Include: the abstract analyzes quantitative and qualitative indicators plus paper attributes associated with traditional, social-media, media, and policy impact."
    ),
    "doi:10.1111/clr.14196": (
        "Include: the abstract extracts 25 bibliometric variables and paper characteristics, including risk of bias, to model citation counts of individual randomized trials."
    ),
    "doi:10.1057/s41599-025-05701-2": (
        "Include: the abstract directly analyzes publication-level disruptiveness during researchers' hot streaks, providing an application lead for an innovation indicator."
    ),
}

EXCLUSION_OVERRIDES: Final[dict[str, str]] = {
    "doi:10.1016/j.heliyon.2023.e23781": (
        "Exclude: the abstract forecasts popularity of scientific topics using topic-level time series, reviews, and patents rather than defining or applying an indicator to an individual paper."
    ),
    "doi:10.1016/j.ijinfomgt.2021.102426": (
        "Exclude: this is an editorial about developing quality articles and avoiding desk rejection; the input has no abstract and the title gives no operational paper-level indicator."
    ),
    "doi:10.1016/j.infsof.2022.106896": (
        "Exclude: the abstract applies fuzzy AHP to prioritize software-industry quantum-computing challenges, not a scholarly-paper innovation, quality, or potential-impact indicator."
    ),
    "doi:10.1016/j.ipm.2009.08.002": (
        "Exclude: the title concerns citation-network-supported scientific authoring tools, but the input supplies no abstract evidencing a paper-level innovation, quality, or potential-impact indicator."
    ),
    "doi:10.1016/j.joi.2018.11.007": (
        "Exclude: the abstract defines an index of reviewer contribution using report time, report length, and recommendation alignment; its unit is the reviewer/editorial process, not a paper."
    ),
    "doi:10.1016/j.joi.2019.100990": (
        "Exclude: the title focuses on self-citations in scientific evaluation at leadership/influence/performance level, not a defined paper-level innovation, quality, or potential-impact indicator."
    ),
    "doi:10.1016/j.joi.2020.101098": (
        "Exclude: the title studies whether citation outcomes should be field-normalized; it does not indicate a publication-time paper feature or a non-future potential-impact indicator."
    ),
    "doi:10.1016/j.joi.2021.101196": (
        "Exclude: the abstract surveys faculty preferences for tenure/promotion evaluation criteria, rather than defining or validating an indicator applied to individual papers."
    ),
    "doi:10.1016/j.respol.2014.06.001": (
        "Exclude: the title concerns university-level articulation of three missions, not an individual-paper indicator."
    ),
    "doi:10.1016/j.respol.2018.10.025": (
        "Exclude: the title concerns misconduct and gaming broadly; the input provides no abstract showing a defined paper-level innovation, quality, or potential-impact measure."
    ),
    "doi:10.1016/j.respol.2018.10.027": (
        "Exclude: the title studies alignment of research priorities and societal demand, not an operational paper-level indicator."
    ),
    "doi:10.1016/j.respol.2021.104448": (
        "Exclude: this review concerns individual research productivity in organizations, not paper-level indicators."
    ),
    "doi:10.1016/j.sbspro.2013.05.057": (
        "Exclude: the abstract analyzes gender homophily and embeddedness among principal investigators; the analytical unit is researchers/networks, not papers."
    ),
    "doi:10.1016/s0140-6736(15)00307-4": (
        "Exclude: the title is a commentary on biomedical research value/waste and the input provides no operational paper-level indicator."
    ),
    "doi:10.1017/s0272263122000560": (
        "Exclude: the abstract synthesizes validity and reliability of second-language survey scales, not quality or impact indicators of scholarly papers."
    ),
    "doi:10.1038/s41598-024-72871-5": (
        "Exclude: the abstract models faculty salary gaps with researcher-level rank and performance variables, not paper-level indicators."
    ),
    "doi:10.1057/s41599-020-0438-z": (
        "Exclude: the abstract reports qualitative perceptions of non-academic impact and its evaluation, without defining or applying a paper-level impact indicator."
    ),
    "doi:10.1057/s41599-021-01017-z": (
        "Exclude: the abstract analyzes Nigeria's social-science research system and inputs at country/system level, not paper-level indicators."
    ),
    "doi:10.1057/s41599-023-02050-w": (
        "Exclude: the abstract quantifies citation benefits at journal, discipline, and region levels, using future citation outcomes rather than a paper-level potential-impact feature."
    ),
    "doi:10.1073/pnas.2500322122": (
        "Exclude: the abstract traces tenure-line faculty career trajectories and researcher outputs, so the unit is the researcher/career rather than an individual paper."
    ),
    "doi:10.1093/databa/baaa022": (
        "Exclude: the abstract proposes database citation-summary publications and credit mechanisms, not an operational indicator of an individual scholarly paper's innovation, quality, or potential impact."
    ),
    "doi:10.1093/epolic/eiz009": (
        "Exclude: the abstract models university-department pay and REF performance; it is institution/department-level rather than paper-level."
    ),
    "doi:10.1093/joc/jqab028": (
        "Exclude: this narrative review concerns open-research practices and humanities outputs generally, without a defined paper-level indicator."
    ),
    "doi:10.1093/reseval/rvae011": (
        "Exclude: the abstract analyzes academics' publication strategies through survey data, not metrics defined or applied to individual papers."
    ),
    "doi:10.1093/scipol/scad052": (
        "Exclude: the abstract studies scholars' reactions to journal lists; the relevant unit is journals and researchers, not papers."
    ),
    "doi:10.1101/2023.10.18.23297223": (
        "Exclude: this review concerns hospital-level DEA input/output selection, not indicators of individual scholarly papers."
    ),
    "doi:10.1101/388678": (
        "Exclude: the abstract proposes researcher-level independence indicators based on collaboration networks and thematic independence, not paper-level measures."
    ),
    "doi:10.1108/eor-03-2023-0008": (
        "Exclude: the abstract models global counts of higher-education researchers and publications, which are system-level trends rather than paper-level indicators."
    ),
    "doi:10.1108/ijchm-10-2017-0622": (
        "Exclude: the abstract develops journal-ranking and journal knowledge-domain metrics; its unit is the journal, not the paper."
    ),
    "doi:10.1108/jd-12-2021-0240": (
        "Exclude: the abstract offers a conceptual critique of metrics and epistemic injustice without a defined or applied paper-level indicator."
    ),
    "doi:10.1108/jkm-04-2020-0270": (
        "Exclude: the abstract models cross-fertilization in EU-funded projects and organizations, not individual scholarly papers."
    ),
    "doi:10.1108/jstp-08-2019-0174": (
        "Exclude: the abstract categorizes citation practices around one marketing article but does not identify a paper-level innovation, quality, or potential-impact indicator."
    ),
    "doi:10.1108/lht-09-2021-0305": (
        "Exclude: the abstract measures impact and touristicity for conference series/venues, not for individual papers."
    ),
    "doi:10.1108/s0733-558x201959": (
        "Exclude: the abstract is a reflective volume description on the production of managerial knowledge, with no operational paper-level indicator."
    ),
    "doi:10.1109/access.2023.3309416": (
        "Exclude: the abstract evaluates author-count metrics for ranking researchers and awardees, making the unit researcher-level rather than paper-level."
    ),
    "doi:10.1109/jestpe.2020.2972056": (
        "Exclude: the abstract validates an inverter topology in electrical engineering, not a scholarly-paper innovation, quality, or potential-impact measure."
    ),
    "doi:10.1111/1468-4446.13088": (
        "Exclude: the abstract models gender gaps in academics' career advancement using person-year data, not paper-level indicators."
    ),
    "doi:10.1111/cobi.70001": (
        "Exclude: the abstract analyzes author coauthorship-network structure and actor centrality, not an indicator defined or applied to individual papers."
    ),
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input() -> tuple[list[dict[str, str]], list[str]]:
    """Read frozen title-and-abstract screening input."""
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader), reader.fieldnames


def rationale_for(row: dict[str, str]) -> tuple[str, str]:
    """Return the H1 screen decision and concise title/abstract rationale."""
    record_key = row["record_key"]
    if record_key in INCLUDE_RATIONALES:
        return "include", INCLUDE_RATIONALES[record_key]
    if record_key in EXCLUSION_OVERRIDES:
        return "exclude", EXCLUSION_OVERRIDES[record_key]
    return (
        "exclude",
        "Exclude: the title/abstract describes a domain-level bibliometric, systematic, or narrative literature review and does not evidence a definition, application, validation, or review of an individual-paper innovation, quality, or potential-impact indicator.",
    )


def validate_rows(rows: list[dict[str, str]]) -> None:
    """Confirm the expected batch and mapping coverage."""
    if len(rows) != 120:
        raise ValueError(f"Expected 120 rows, found {len(rows)}")
    keys = {row["record_key"] for row in rows}
    unknown = (set(INCLUDE_RATIONALES) | set(EXCLUSION_OVERRIDES)) - keys
    if unknown:
        raise ValueError(f"Screen mapping has unknown records: {sorted(unknown)}")


def write_output(rows: list[dict[str, str]], fields: list[str]) -> str:
    """Copy all frozen fields and append the two H1 screen fields."""
    output_rows = []
    for row in rows:
        decision, rationale = rationale_for(row)
        output_rows.append({**row, "H1_decision": decision, "H1_rationale": rationale})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*fields, *H1_FIELDS], extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return sha256_file(OUTPUT_PATH)


def write_manifest(rows: list[dict[str, str]], output_sha: str) -> None:
    """Record immutable input/output provenance and blind-review constraints."""
    decisions = [rationale_for(row)[0] for row in rows]
    manifest = {
        "input": str(INPUT_PATH),
        "input_sha256": sha256_file(INPUT_PATH),
        "artifact": str(OUTPUT_PATH),
        "artifact_sha256": output_sha,
        "row_count": len(rows),
        "decision_counts": {
            decision: decisions.count(decision) for decision in sorted(set(decisions))
        },
        "qwen_or_ollama_used": False,
        "review_scope": "Independent H1 screen from English titles and abstracts only; no AI/H2 or earlier-batch task outputs read.",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Create the batch-8 H1 source-screening artifact."""
    rows, fields = read_input()
    validate_rows(rows)
    output_sha = write_output(rows, fields)
    write_manifest(rows, output_sha)


if __name__ == "__main__":
    main()
