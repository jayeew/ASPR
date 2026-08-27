from __future__ import annotations

import hashlib

from reportlab.pdfgen import canvas

from gear.contracts import EvidenceSpan, PaperMetadata, ParseStatus, ReviewRequest
from gear.paper_compiler import PaperCompiler, extract_references


def test_markdown_spans_are_stable_and_hash_addressable(gear_config, paper_request):
    compiler = PaperCompiler(gear_config)
    first = compiler.compile(paper_request)
    second = compiler.compile(paper_request)
    assert first.parse_status == ParseStatus.READY
    assert first.source_format == "markdown"
    assert len(first.pages) == 1
    assert [item.span_id for item in first.spans] == [
        item.span_id for item in second.spans
    ]
    assert first.claims
    assert first.method_result_ledger.model_algorithm
    pages = {page.page: page.text for page in first.pages}
    for span in first.spans:
        expected = "sha256:" + hashlib.sha256(span.text.encode("utf-8")).hexdigest()
        assert span.text_sha256 == expected
        assert span.page == 1
        assert span.char_end >= span.char_start
        assert pages[span.page][span.char_start : span.char_end] == span.text


def test_blank_or_scanned_pdf_fails_closed(tmp_path, gear_config):
    path = tmp_path / "blank.pdf"
    document = canvas.Canvas(str(path))
    document.showPage()
    document.save()
    request = ReviewRequest(
        paper_path=path,
        metadata=PaperMetadata(title="Scan"),
    )
    paper = PaperCompiler(gear_config).compile(request)
    assert paper.parse_status == ParseStatus.UNAVAILABLE
    assert paper.spans == []
    assert paper.claims == []
    assert "paper_text_unavailable" in paper.quality_flags


def test_pdf_is_adapted_to_markdown(gear_config, sample_pdf):
    paper = PaperCompiler(gear_config).compile(ReviewRequest(paper_path=sample_pdf))
    assert paper.source_format == "pdf"
    assert "<!-- GEAR_PAGE: 1 -->" in paper.markdown
    assert len(paper.pages) == 2


def test_heading_with_trailing_space_creates_real_section_path():
    from gear.contracts import PageText
    from gear.paper_compiler import segment_pages

    page = PageText(
        page=1,
        text="## Results \n\nThe experiment improves accuracy against the baseline.",
    )
    spans = segment_pages([page], "source")

    result = next(span for span in spans if span.text.startswith("The experiment"))
    assert result.section_path == ["Results"]
    assert page.text[result.char_start : result.char_end] == result.text


def test_reference_chunks_are_split_and_continuations_are_preserved():
    def span(span_id: str, text: str) -> EvidenceSpan:
        return EvidenceSpan(
            span_id=span_id,
            source_id="source",
            page=1,
            section_path=["References"],
            char_start=0,
            char_end=len(text),
            text=text,
            text_sha256=("sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()),
        )

    references = extract_references(
        [
            span(
                "S1",
                "1. Alpha, A. First method. Journal 10, 1-9 (2019).\n\n"
                "2. Beta, B. A long second method with DOI 10.1234/example",
            ),
            span(
                "S2",
                "continued in the next PDF chunk (2020).\n\n"
                "3. Gamma, C. Third method. Journal 12, 2-8 (2021). "
                "https://doi.org/10.9999/manuscript",
            ),
        ],
        manuscript_doi="10.9999/manuscript",
    )

    assert [item.citation_number for item in references] == [1, 2, 3]
    assert references[1].doi == "10.1234/example"
    assert "continued in the next PDF chunk" in references[1].raw_text
    assert references[1].publication_year == 2020
    assert references[2].doi is None


def test_reference_titles_enable_exact_citation_graph_anchors():
    text = (
        "7. Sangwan, V. K. et al. Multi-terminal memtransistors from "
        "polycrystalline monolayer molybdenum disulfide. Nature 554, "
        "500–504 (2018)."
    )
    span = EvidenceSpan(
        span_id="S-ref",
        source_id="source",
        page=1,
        section_path=["References"],
        char_start=0,
        char_end=len(text),
        text=text,
        text_sha256=("sha256:" + hashlib.sha256(text.encode()).hexdigest()),
    )

    references = extract_references([span])

    assert references[0].title == (
        "Multi-terminal memtransistors from polycrystalline monolayer "
        "molybdenum disulfide"
    )
