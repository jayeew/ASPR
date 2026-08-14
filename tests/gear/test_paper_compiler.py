from __future__ import annotations

import hashlib

from reportlab.pdfgen import canvas

from gear.contracts import PaperMetadata, ParseStatus, ReviewRequest
from gear.paper_compiler import PaperCompiler


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
