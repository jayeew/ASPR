"""Create conservative H2 batch-2 verified-fulltext adjudications."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_INPUT = ROOT / "outputs" / "contextual_fulltext_source_review_H2_batch2_v4.csv"
MENTION_INPUT = ROOT / "outputs" / "contextual_fulltext_indicator_mentions_H2_batch2_v4.csv"
SOURCE_OUTPUT = ROOT / "outputs" / "contextual_fulltext_source_review_H2_batch2_completed_v4.csv"
MENTION_OUTPUT = ROOT / "outputs" / "contextual_fulltext_indicator_mentions_H2_batch2_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_fulltext_h2_batch2_manifest_v4.json"


SOURCE_DECISIONS = {
    "doi:10.1016/j.ipm.2023.103323": ("formula_or_application", "The verified methods table explicitly defines static paper attributes (authors, keywords, abstract length, page count, references) separately from later citation features and the citation target."),
    "doi:10.1016/j.jeap.2023.101253": ("review_discovery_only", "The verified argument paper discusses academic publishing and attention but supplies no explicit eligible paper-level T0 formula/application."),
    "doi:10.1017/cbo9781316161012": ("review_discovery_only", "The verified book provides open-access scholarly-publishing context and terminology, not an explicit original paper-level T0 indicator application."),
    "doi:10.1038/s41562-022-01351-5": ("no_relevant_indicator", "The verified citational-lensing framework compares country-level citation and textual-similarity network edges and depends on citation results, rather than defining an eligible focal-paper T0 indicator."),
    "doi:10.1214/09-sts285": ("review_discovery_only", "The verified report critiques citation statistics and research assessment; it is discovery context, not an original eligible focal-paper T0 formula/application."),
    "doi:10.1371/journal.pbio.1002541": ("no_relevant_indicator", "The verified text ranks papers with citation/co-citation-based RCR variants. Those citation-network measures are not T0 paper features for this recovery task."),
    "doi:10.1371/journal.pone.0005910": ("no_relevant_indicator", "The verified paper compares expert assessment shortly after publication with citation and F1000 impact after three years; it does not define an eligible T0 explanatory feature."),
    "doi:10.1371/journal.pone.0006022": ("no_relevant_indicator", "The verified PCA evaluates scholarly-impact measures calculated from citation and usage data, not a focal-paper T0 innovation/opportunity/control feature."),
    "doi:10.1371/journal.pone.0120495": ("formula_or_application", "The verified methods explicitly define page, reference, title, author, institution, and country document/collaboration variables before analyzing later social-media and citation outcomes."),
    "doi:10.1371/journal.pone.0135095": ("formula_or_application", "The verified text explicitly applies reference-category variety, balance, disparity, Rao-Stirling, and affiliation collaboration definitions to individual papers; later citation impact remains an outcome."),
    "doi:10.1371/journal.pone.0199031": ("no_relevant_indicator", "The verified sociological study examines the legitimacy of bibliometric research at field/practice level and provides no eligible focal-paper T0 operational definition."),
    "doi:10.1371/journal.pone.0230416": ("formula_or_application", "The verified methods explicitly code each publication's data-availability statement and calculate author H-index summaries at publication time; citation advantage is a later outcome."),
    "doi:10.1371/journal.pone.0251493": ("formula_or_application", "The verified text explicitly applies article-level classification approaches, but the event-time status of its trained model/reference index must be independently established before candidate use."),
    "doi:10.1371/journal.pone.0274693": ("no_relevant_indicator", "The verified Overton analysis evaluates research relevance using policy-document attention links and later policy/citation-style outcomes, not an eligible focal-paper T0 feature."),
    "doi:10.1515/9783110255553": ("review_discovery_only", "The verified volume reviews multidimensional journal evaluation and impact-factor alternatives; it is discovery context rather than a focal-paper T0 application."),
    "doi:10.1609/icwsm.v6i1.14305": ("formula_or_application", "The verified text explicitly applies Gunning Fog and Flesch readability indices to individual scientific abstracts; downloads, bookmarks, and citations are later outcomes."),
    "doi:10.2196/10961": ("no_relevant_indicator", "The verified eHealth conference study summarizes program outcomes and bibliometrics at initiative level rather than defining an eligible focal-paper T0 feature."),
    "doi:10.22439/fs.v25i2.5578": ("no_relevant_indicator", "The verified historical/theoretical account of citation notation provides no explicit paper-level T0 indicator definition or application."),
    "doi:10.3145/epi.2022.jul.11": ("review_discovery_only", "The verified social-systems citation theory paper is conceptual discovery material and does not apply an eligible focal-paper T0 indicator."),
    "doi:10.3389/fnhum.2013.00291": ("review_discovery_only", "The verified discussion of journal rank and unintended consequences is contextual review material; it does not supply an eligible original T0 feature formula."),
    "doi:10.3389/frma.2016.00001": ("review_discovery_only", "The verified impact-fallacy review critiques citations as quality indicators and is discovery context, not a focal-paper T0 application."),
    "doi:10.48550/arxiv.2106.01083": ("formula_or_application", "The verified methods define sentence-level rhetorical-move categories and manually classify data-paper abstracts, a paper-text operation requiring no later outcomes."),
    "doi:10.5465/annals.2017.0099": ("review_discovery_only", "The verified topic-modeling review explains a text-analysis method for theory rendering but does not give a specific eligible focal-paper T0 indicator application."),
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one frozen CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write a CSV with its original columns."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def payload(**values: str) -> dict[str, str]:
    """Create a complete retained-candidate payload."""
    return {"h2_decision": "retain_as_candidate", **values}


def retained_candidates() -> dict[tuple[str, str], dict[str, str]]:
    """Return source-supported, paper-level T0 candidate operationalizations."""
    return {
        ("doi:10.1016/j.ipm.2023.103323", "Static bibliographic paper-attribute vector"): payload(
            raw_name_en="Static features (number of authors, keywords, abstract length, paper length, references)",
            canonical_name_en="Static bibliographic paper-attribute vector",
            source_role="original_application",
            formula_location="Methods §3.1, Table 1, p. 5.",
            evidence_span="X3 Abstract length Word count of the abstract.",
            formula="[author count, author-defined keyword count, abstract word count, article page count, reference count]",
            parameters="author count; keyword count; abstract word count; page count; reference count",
            required_data="focal-paper author list; author-defined keywords; abstract text; pagination; reference list",
            maximum_information_time="T0 (all listed static attributes are available from the focal paper and its publication metadata).",
            scope_role="context_control",
            requires_future="false",
            extraction_notes="Retained as a candidate control vector only. The source separately lists citation features and uses a citation-based prediction target; neither is imported as a T0 feature.",
        ),
        ("doi:10.1371/journal.pone.0120495", "Bibliographic and collaboration control vector"): payload(
            raw_name_en="Document characteristics and collaboration indicators (PG, NR, TI, AU, IN, CU)",
            canonical_name_en="Bibliographic and collaboration control vector",
            source_role="original_application",
            formula_location="Materials and Methods—Indicators and variables, pp. 4–5.",
            evidence_span="Number of characters in title [TI]: length of a document's title in the number of characters.",
            formula="[page count, reference count, title-character count, author count, distinct institution count, distinct country count]",
            parameters="PG; NR; TI; AU; IN; CU",
            required_data="focal-paper pagination; reference list; title text; author list; author-affiliation institution and country identifiers",
            maximum_information_time="T0 (source defines these as document/collaboration independent variables before later social-media and citation outcomes).",
            scope_role="context_control",
            requires_future="false",
            extraction_notes="Retained only as a T0 control vector. Social-media coverage/density/intensity and citations accrue after publication and are excluded.",
        ),
        ("doi:10.1371/journal.pone.0135095", "Institutional collaboration geographic scope"): payload(
            raw_name_en="National_collab, Internat_collab, and No_Collab",
            canonical_name_en="Institutional collaboration geographic scope",
            source_role="original_application",
            formula_location="Data and Methods—Control variables, p. 10.",
            evidence_span="National_collab takes value 1 if there are at least two different institutions from the same country.",
            formula="National_collab = 1 if at least two institutions and one country; Internat_collab = 1 if at least two countries; No_Collab = 1 if one institution participates.",
            parameters="institution count; country count",
            required_data="focal-paper author-affiliation institution identifiers; focal-paper author-affiliation country identifiers",
            maximum_information_time="T0 (computed from listed affiliations of the focal paper).",
            scope_role="t0_opportunity",
            requires_future="false",
            extraction_notes="Retained as a paper-level collaboration-context classification. The article's later citation-impact dependent variable is not imported.",
        ),
        ("doi:10.1371/journal.pone.0135095", "Reference-category balance"): payload(
            raw_name_en="Balance",
            canonical_name_en="Reference-category balance",
            source_role="original_application",
            formula_location="Table 2, p. 7.",
            evidence_span="Balance We use Shannon diversity (H) normalised by variety (n), where pi is the proportion of references in WoS category i.",
            formula="Balance = -(sum_i p_i ln(p_i)) / ln(n)",
            parameters="n = number of represented cited categories; p_i = share of focal-paper references in category i",
            required_data="focal-paper reference list; cited-reference subject-category assignments",
            maximum_information_time="T0 (uses only the focal paper's backward references and their category assignments).",
            scope_role="direct_innovation",
            requires_future="false",
            extraction_notes="Retained as a candidate reference-category distribution statistic. Later normalized citation impact in the source is an outcome and is excluded.",
        ),
        ("doi:10.1371/journal.pone.0135095", "Reference-category variety"): payload(
            raw_name_en="Variety",
            canonical_name_en="Reference-category variety",
            source_role="original_application",
            formula_location="Table 2, p. 7.",
            evidence_span="Variety We use the number of distinctive WoS categories (n) cited in an article.",
            formula="n = number of distinctive subject categories represented among focal-paper cited references",
            parameters="n = count of distinctive cited categories",
            required_data="focal-paper reference list; cited-reference subject-category assignments",
            maximum_information_time="T0 (uses only the focal paper's backward references and their category assignments).",
            scope_role="direct_innovation",
            requires_future="false",
            extraction_notes="Retained as a candidate reference-category count. Citation impact in the source is a later outcome and is excluded.",
        ),
        ("doi:10.1371/journal.pone.0230416", "Author-team prior H-index aggregation"): payload(
            raw_name_en="Mean and median author H-index at publication time",
            canonical_name_en="Author-team prior H-index aggregation",
            source_role="original_application",
            formula_location="Materials and methods—Citation prediction, independent variables, p. 8; Table 3, p. 9.",
            evidence_span="the mean and median H-index of an article's authors at the time of publication",
            formula="h_index_mean = mean of focal-paper authors' pre-publication H-indices; h_index_median = median of those H-indices",
            parameters="per-author prior H-index; mean or median aggregation",
            required_data="focal-paper author identities; author publication and citation records restricted to events available by the focal publication date",
            maximum_information_time="T0 (the source explicitly states the aggregation is at publication time).",
            scope_role="context_control",
            requires_future="false",
            extraction_notes="Retained only with a strictly pre-publication citation-history cutoff for every author. The article-level citation-accrual dependent variable is excluded.",
        ),
        ("doi:10.1371/journal.pone.0230416", "Data availability statement category"): payload(
            raw_name_en="Data availability statement category",
            canonical_name_en="Data availability statement category",
            source_role="original_application",
            formula_location="Materials and methods—Data availability statements: Classification, Table 1, pp. 4–5.",
            evidence_span="We identified four categories of DAS, further described in Table 1.",
            formula="0 = not available/access restricted; 1 = available on request; 2 = available with paper or supplementary files; 3 = available in a repository",
            parameters="DAS category 0–3",
            required_data="focal-paper data-availability statement text; stated repository link or identifier when applicable",
            maximum_information_time="T0 (the statement is part of the published focal paper).",
            scope_role="t0_opportunity",
            requires_future="false",
            extraction_notes="Retained as a categorical publication-time data-access context. Later citation advantage and regression outcomes are excluded.",
        ),
        ("doi:10.1371/journal.pone.0230416", "Data-availability statement category"): payload(
            raw_name_en="Data availability statement (DAS) category",
            canonical_name_en="Data availability statement category",
            source_role="original_application",
            formula_location="Materials and methods—Data availability statements: Classification, Table 1, pp. 4–5.",
            evidence_span="We identified four categories of DAS, further described in Table 1.",
            formula="0 = not available/access restricted; 1 = available on request; 2 = available with paper or supplementary files; 3 = available in a repository",
            parameters="DAS category 0–3",
            required_data="focal-paper data-availability statement text; stated repository link or identifier when applicable",
            maximum_information_time="T0 (the statement is part of the published focal paper).",
            scope_role="t0_opportunity",
            requires_future="false",
            extraction_notes="Retained as the same categorical publication-time data-access context supplied by the independent candidate union. Later citation advantage is excluded.",
        ),
        ("doi:10.1609/icwsm.v6i1.14305", "Abstract readability indices"): payload(
            raw_name_en="Gunning Fog and Flesch readability indices",
            canonical_name_en="Abstract readability indices",
            source_role="original_application",
            formula_location="Readability Index Tests, p. 477.",
            evidence_span="We use two indices to compute the difficulty of an abstract: the Gunning Fog (Gunning 1952) and the Flesch indices (Flesch 1946).",
            formula="Apply the named Gunning Fog and Flesch readability definitions to focal-paper abstract text.",
            parameters="sentence length; word length; syllable/complex-word quantities required by the named readability definitions",
            required_data="focal-paper abstract text; sentence segmentation; word and syllable/complex-word counts",
            maximum_information_time="T0 (computed solely from the published focal abstract).",
            scope_role="t0_substantive",
            requires_future="false",
            extraction_notes="Retained as an explicit abstract-text application; the source's downloads, bookmarks, and citations are later outcomes and are excluded.",
        ),
        ("doi:10.48550/arxiv.2106.01083", "Abstract rhetorical-move composition"): payload(
            raw_name_en="Rhetorical moves in data-paper abstracts",
            canonical_name_en="Abstract rhetorical-move composition",
            source_role="original_definition",
            formula_location="Method §3.2, Table 1, Classification and definition of rhetorical moves in data paper abstracts.",
            evidence_span="Two coders independently classified all sentences using the modified classification scheme. Our coding also allows the co-existence of multiple moves in one sentence.",
            formula="Classify each abstract sentence as Introduction, Purpose, Method, Results, Conclusion, Data Description, Data Uses, Data Accessibility, or Related Research Article; summarize move counts or shares, using fractional counts for multi-move sentences.",
            parameters="sentence move labels; sentence count; fractional count for multi-move sentences",
            required_data="focal-paper abstract text; sentence segmentation; published rhetorical-move codebook",
            maximum_information_time="T0 (all inputs are in the focal abstract and coding uses no later outcomes).",
            scope_role="t0_substantive",
            requires_future="false",
            extraction_notes="Retained as a sentence-level text-composition candidate. The source establishes a coding scheme and does not treat later attention/citation as input.",
        ),
        ("doi:10.48550/arxiv.2106.01083", "Abstract rhetorical-move structure"): payload(
            raw_name_en="Rhetorical moves in data-paper abstracts",
            canonical_name_en="Abstract rhetorical-move structure",
            source_role="original_application",
            formula_location="Method §3.2, Table 1; Research Question 3.",
            evidence_span="What is the order in which these moves are structured in the abstract?",
            formula="Apply the published sentence-level rhetorical-move coding scheme and summarize move sequence/position across the focal abstract.",
            parameters="ordered sentence move labels; sentence positions; published move codebook",
            required_data="focal-paper abstract text; sentence segmentation; rhetorical-move codebook",
            maximum_information_time="T0 (all inputs are in the focal abstract and coding uses no later outcomes).",
            scope_role="t0_substantive",
            requires_future="false",
            extraction_notes="Retained as a distinct structural summary candidate; it remains subject to later deduplication with rhetorical-move composition and is not final feature selection.",
        ),
    }


REJECTION_NOTES = {
    ("doi:10.1371/journal.pone.0135095", "Average reference-category disparity"): "Reject: although the disparity algebra is explicit, its s_ij matrix is constructed from citation flows between WoS categories in 2006; the source does not establish a fixed no-later-than-publication snapshot for each focal paper.",
    ("doi:10.1371/journal.pone.0135095", "Rao–Stirling reference diversity"): "Reject: the formula is explicit but relies on the same 2006 citation-flow similarity matrix without an event-time T0 guarantee for each focal paper.",
    ("doi:10.1371/journal.pone.0135095", "Reference-list disciplinary diversity components"): "Reject: this bundled candidate includes disparity, whose source similarity matrix uses 2006 citation flows without a per-paper T0 snapshot; the individually T0-safe variety and balance candidates are adjudicated separately.",
    ("doi:10.1371/journal.pone.0251493", "Article-level scientific subfield assignment"): "Reject: the paper applies a retrospective trained classifier and reference-subfield index, but does not establish that the trained model and all indices were frozen using only information available by each focal paper's publication date.",
    ("doi:10.1371/journal.pone.0251493", "Article-level scientific-publication classification"): "Reject: the generic classification entry includes approaches with retrospective corpus snapshots and a direct-citation comparator; the source does not demonstrate a uniformly T0 event-time implementation.",
}


def main() -> None:
    """Write both H2 outputs and an auditable manifest."""
    source_fields, source_rows = read_csv(SOURCE_INPUT)
    if len(source_rows) != 23 or set(SOURCE_DECISIONS) != {row["record_key"] for row in source_rows}:
        raise ValueError("Unexpected batch-2 source rows.")
    source_h2 = {"h2_final_source_disposition", "h2_final_source_notes"}
    source_protected = [field for field in source_fields if field not in source_h2]
    source_before = [{field: row[field] for field in source_protected} for row in source_rows]
    for row in source_rows:
        row["h2_final_source_disposition"], row["h2_final_source_notes"] = SOURCE_DECISIONS[row["record_key"]]
    if source_before != [{field: row[field] for field in source_protected} for row in source_rows]:
        raise AssertionError("Frozen source fields changed.")
    write_csv(SOURCE_OUTPUT, source_fields, source_rows)

    mention_fields, mention_rows = read_csv(MENTION_INPUT)
    if len(mention_rows) != 16:
        raise ValueError("Expected 16 batch-2 mention candidates.")
    final_fields = {
        "h2_decision", "raw_name_en", "canonical_name_en", "source_role", "formula_location",
        "evidence_span", "formula", "parameters", "required_data", "maximum_information_time",
        "scope_role", "requires_future", "extraction_notes",
    }
    mention_protected = [field for field in mention_fields if field not in final_fields]
    mention_before = [{field: row[field] for field in mention_protected} for row in mention_rows]
    retained = retained_candidates()
    for row in mention_rows:
        key = (row["record_key"], row["ai_canonical_name_en"] or row["h1_canonical_name_en"])
        if key in retained:
            row.update(retained[key])
        else:
            for field in final_fields:
                row[field] = ""
            row["h2_decision"] = "reject"
            try:
                row["extraction_notes"] = REJECTION_NOTES[key]
            except KeyError as error:
                raise ValueError(f"Missing rejection note for {key}") from error
    if mention_before != [{field: row[field] for field in mention_protected} for row in mention_rows]:
        raise AssertionError("Frozen AI/H1 mention fields changed.")
    write_csv(MENTION_OUTPUT, mention_fields, mention_rows)

    source_counts = Counter(row["h2_final_source_disposition"] for row in source_rows)
    mention_counts = Counter(row["h2_decision"] for row in mention_rows)
    manifest = {
        "schema_version": "contextual_fulltext_h2_batch2_manifest_v4",
        "status": "complete",
        "reviewer_role": "H2",
        "reviewer_id": "H2_VERIFIED_FULLTEXT_BATCH2_ADJUDICATOR_V4",
        "model": "codex_configured_default",
        "model_digest": "codex-thread:/root/h2_adjudication",
        "review_completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "method": "Candidate retention requires an explicit paper-level formula or operational definition computable at T0 without future-impact dependence. Source dispositions distinguish formula/application, discovery-only reviews, and no relevant indicators. Citation and other post-publication outcomes are excluded; unproven retrospective event-time assumptions fail closed.",
        "qwen_or_ollama_used": False,
        "input_artifacts": {str(SOURCE_INPUT): sha256(SOURCE_INPUT), str(MENTION_INPUT): sha256(MENTION_INPUT)},
        "output_artifacts": {str(SOURCE_OUTPUT): sha256(SOURCE_OUTPUT), str(MENTION_OUTPUT): sha256(MENTION_OUTPUT)},
        "source_record_count": len(source_rows),
        "source_disposition_counts": dict(sorted(source_counts.items())),
        "mention_candidate_count": len(mention_rows),
        "mention_decision_counts": dict(sorted(mention_counts.items())),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one CSV including every frozen column."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    """Write a complete CSV with original field order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
