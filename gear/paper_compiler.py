"""Compile Markdown, or a PDF adapted to Markdown, into hash-stable PaperIR."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import GearConfig, load_config
from .contracts import (
    ClaimLedger,
    ClaimStrength,
    ClaimType,
    EvidenceReadiness,
    EvidenceSpan,
    MethodResultLedger,
    PageText,
    PaperClaim,
    PaperIR,
    PaperQualityReport,
    ParseStatus,
    ReferenceEntry,
    ReviewRequest,
)

HEADING_PATTERN = re.compile(
    r"^(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\s+)?(?:abstract|introduction|background|related work|"
    r"methods?|materials? and methods?|experiments?|results?|discussion|"
    r"limitations?|conclusions?|references?|supplementary materials?)\s*$",
    flags=re.IGNORECASE,
)
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.IGNORECASE)
REFERENCE_NUMBER_PATTERN = re.compile(r"(?:^|\n)\s*(\d{1,4})[.)]\s+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
PAGE_MARKER = re.compile(r"^<!--\s*GEAR_PAGE:\s*(\d+)\s*-->\s*$", re.MULTILINE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_text(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _normalize_page_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def extract_pdf_pages(path: Path) -> list[PageText]:
    """Extract one normalized text object per PDF page."""
    reader = PdfReader(str(Path(path)))
    return [
        PageText(page=index, text=_normalize_page_text(page.extract_text() or ""))
        for index, page in enumerate(reader.pages, start=1)
    ]


def pdf_to_markdown(path: Path) -> str:
    """Adapt a PDF once at ingress; every downstream stage consumes Markdown."""
    pages = extract_pdf_pages(path)
    return "\n\n".join(
        f"<!-- GEAR_PAGE: {page.page} -->\n\n{page.text}" for page in pages
    ).strip()


def read_markdown(path: Path) -> str:
    """Read and minimally normalize an authored Markdown manuscript."""
    value = Path(path).read_text(encoding="utf-8")
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def markdown_pages(markdown: str) -> list[PageText]:
    """Recover PDF page markers or expose native Markdown as one logical page."""
    matches = list(PAGE_MARKER.finditer(markdown))
    if not matches:
        return [PageText(page=1, text=markdown)] if markdown else []
    pages: list[PageText] = []
    for index, marker in enumerate(matches):
        start = marker.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        pages.append(
            PageText(
                page=int(marker.group(1)),
                text=markdown[start:end].strip(),
            )
        )
    return pages


def _split_long_block(
    text: str, start: int, maximum: int = 1800
) -> Iterable[tuple[int, int, str]]:
    if len(text) <= maximum:
        yield start, start + len(text), text
        return
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        limit = min(cursor + maximum, len(text))
        cut = limit
        if limit < len(text):
            lower = cursor + maximum // 2
            boundaries = list(re.finditer(r"(?<=[.!?。！？])\s+", text[lower:limit]))
            if boundaries:
                cut = lower + boundaries[-1].start()
            else:
                whitespace = max(
                    text.rfind("\n", lower, limit),
                    text.rfind(" ", lower, limit),
                )
                if whitespace > cursor:
                    cut = whitespace
        end = cut
        while end > cursor and text[end - 1].isspace():
            end -= 1
        if end <= cursor:
            end = limit
        yield start + cursor, start + end, text[cursor:end]
        cursor = max(cut, end)


def _blocks_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """Split headings before paragraphs while preserving exact page offsets."""
    blocks: list[tuple[int, int, str]] = []
    heading_ranges: list[tuple[int, int, str]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line_end = cursor + len(raw_line)
        line = raw_line.rstrip("\r\n")
        clean = line.strip()
        if clean and HEADING_PATTERN.fullmatch(clean):
            leading = len(line) - len(line.lstrip())
            heading_ranges.append(
                (cursor + leading, cursor + leading + len(clean), clean)
            )
        cursor = line_end
    if heading_ranges:
        region_start = 0
        for start, end, heading in heading_ranges:
            blocks.extend(_paragraph_blocks(text[region_start:start], region_start))
            blocks.append((start, end, heading))
            region_start = end
        blocks.extend(_paragraph_blocks(text[region_start:], region_start))
        return blocks
    return _paragraph_blocks(text, 0)


def _paragraph_blocks(text: str, base_offset: int) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.DOTALL):
        raw = match.group(0)
        if not raw:
            continue
        blocks.extend(_split_long_block(raw, base_offset + match.start()))
    if len(blocks) <= 1 and "\n" in text:
        line_blocks: list[tuple[int, int, str]] = []
        cursor = 0
        for line in text.splitlines():
            position = text.find(line, cursor)
            cursor = max(position, cursor) + len(line)
            clean = line.strip()
            if len(clean) >= 20 or HEADING_PATTERN.match(clean):
                leading = len(line) - len(line.lstrip())
                line_blocks.extend(
                    _split_long_block(clean, base_offset + max(position, 0) + leading)
                )
        if len(line_blocks) > len(blocks):
            blocks = line_blocks
    return blocks


def segment_pages(pages: Sequence[PageText], source_id: str) -> list[EvidenceSpan]:
    """Create deterministic page-local evidence spans and section paths."""
    spans: list[EvidenceSpan] = []
    current_section = "Document"
    block_index = 0
    for page in pages:
        for start, end, block in _blocks_with_offsets(page.text):
            if HEADING_PATTERN.fullmatch(block.strip()):
                current_section = re.sub(r"^#{1,6}\s*", "", block.strip())
            text_hash = _sha256_text(block)
            identity = f"{source_id}|{page.page}|{block_index}|{text_hash}"
            span_id = "S-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            spans.append(
                EvidenceSpan(
                    span_id=span_id,
                    source_id=source_id,
                    page=page.page,
                    section_path=[current_section],
                    char_start=start,
                    char_end=end,
                    text=block,
                    text_sha256=text_hash,
                )
            )
            block_index += 1
    return spans


def _claim_type(sentence: str) -> ClaimType | None:
    lower = sentence.casefold()
    if re.search(r"\b(first|novel|new|unprecedented|we propose|we introduce)\b", lower):
        return ClaimType.NOVELTY
    if re.search(
        r"\b(causes?|caused|leads? to|results? in|mediates?|drives?)\b", lower
    ):
        return ClaimType.CAUSAL
    if re.search(
        r"\b(generaliz|broadly|across (?:all|multiple)|applicable to|universally)\b",
        lower,
    ):
        return ClaimType.SCOPE
    if re.search(
        r"\b(important|significant impact|potential to|transform|promising)\b", lower
    ):
        return ClaimType.SIGNIFICANCE
    if re.search(
        r"\b(achiev|outperform|improv|increase|decrease|significant|p\s*[<=>])\b", lower
    ):
        return ClaimType.RESULT
    if re.search(
        r"\b(method|model|algorithm|framework|protocol|assay|we develop|we present)\b",
        lower,
    ):
        return ClaimType.METHOD
    return None


def _claim_strength(sentence: str) -> ClaimStrength:
    lower = sentence.casefold()
    if re.search(
        r"\b(first|unprecedented|state[- ]of[- ]the[- ]art|proves?|demonstrates?|causes?)\b",
        lower,
    ):
        return ClaimStrength.STRONG
    if re.search(r"\b(may|might|could|suggests?|potentially|preliminary)\b", lower):
        return ClaimStrength.WEAK
    return ClaimStrength.MODERATE


def _required_evidence(claim_type: ClaimType) -> list[str]:
    mapping: dict[ClaimType, list[str]] = {
        ClaimType.NOVELTY: ["target_span", "prior_art_relation"],
        ClaimType.METHOD: ["target_span", "internal_method_evidence"],
        ClaimType.RESULT: ["target_span", "result_or_table_evidence"],
        ClaimType.SCOPE: ["target_span", "scope_validation"],
        ClaimType.CAUSAL: ["target_span", "causal_design_evidence"],
        ClaimType.SIGNIFICANCE: ["target_span", "calibrated_significance_evidence"],
    }
    return mapping[claim_type]


def extract_claims(
    spans: Sequence[EvidenceSpan], maximum: int = 12
) -> list[PaperClaim]:
    claims: list[PaperClaim] = []
    seen: set[str] = set()
    for span in spans:
        if any("reference" in item.casefold() for item in span.section_path):
            continue
        for sentence in SENTENCE_SPLIT.split(span.text):
            clean = re.sub(r"\s+", " ", sentence).strip()
            if len(clean) < 35:
                continue
            claim_type = _claim_type(clean)
            if claim_type is None:
                continue
            fingerprint = hashlib.sha256(clean.casefold().encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            claims.append(
                PaperClaim(
                    claim_id="C-" + fingerprint[:16],
                    claim_type=claim_type,
                    span_id=span.span_id,
                    text=clean[:1200],
                    strength=_claim_strength(clean),
                    required_evidence=_required_evidence(claim_type),
                )
            )
            if len(claims) >= maximum:
                return claims
    if not claims:
        for span in spans[:1]:
            fingerprint = hashlib.sha256(
                span.text.casefold().encode("utf-8")
            ).hexdigest()
            claims.append(
                PaperClaim(
                    claim_id="C-" + fingerprint[:16],
                    claim_type=ClaimType.METHOD,
                    span_id=span.span_id,
                    text=span.text[:1200],
                    strength=ClaimStrength.WEAK,
                    required_evidence=_required_evidence(ClaimType.METHOD),
                )
            )
    return claims


LEDGER_PATTERNS: dict[str, re.Pattern[str]] = {
    "research_question": re.compile(
        r"\b(aim|question|objective|hypothes)\b", re.IGNORECASE
    ),
    "dataset_sample": re.compile(
        r"\b(dataset|sample|participants?|cohort|corpus|subjects?)\b", re.IGNORECASE
    ),
    "design_comparator": re.compile(
        r"\b(design|control group|comparator|randomi[sz]|intervention)\b", re.IGNORECASE
    ),
    "model_algorithm": re.compile(
        r"\b(model|algorithm|architecture|loss function|assay|protocol)\b",
        re.IGNORECASE,
    ),
    "baselines_metrics_statistics": re.compile(
        r"\b(baseline|metric|accuracy|f1|auc|p[- ]value|confidence interval|statistical)\b",
        re.IGNORECASE,
    ),
    "ablation_robustness": re.compile(
        r"\b(ablation|robust|sensitivity|perturb|subgroup)\b", re.IGNORECASE
    ),
    "main_results": re.compile(
        r"\b(results?|outperform|improv|increase|decrease|significant)\b", re.IGNORECASE
    ),
    "stated_limitations": re.compile(
        r"\b(limitations?|caveat|future work)\b", re.IGNORECASE
    ),
    "figures_tables": re.compile(
        r"\b(fig(?:ure)?\.?\s*\d+|table\s*\d+)\b", re.IGNORECASE
    ),
}


def extract_method_result_ledger(spans: Sequence[EvidenceSpan]) -> MethodResultLedger:
    values: dict[str, list[str]] = {name: [] for name in LEDGER_PATTERNS}
    for span in spans:
        searchable = f"{' '.join(span.section_path)} {span.text}"
        for name, pattern in LEDGER_PATTERNS.items():
            if pattern.search(searchable):
                values[name].append(span.span_id)
    return MethodResultLedger.model_validate(values)


def extract_references(
    spans: Sequence[EvidenceSpan], manuscript_doi: str | None = None
) -> list[ReferenceEntry]:
    """Split numbered bibliography spans into one traceable entry per citation.

    PDF page chunks often contain 10--20 bibliography items.  Treating a whole
    chunk as one reference makes citation seeds ambiguous and can accidentally
    select the manuscript's own DOI.  Continuation text is carried across span
    boundaries while each entry remains anchored to the span where it starts.
    """

    entries: list[tuple[int | None, str, str]] = []
    current_number: int | None = None
    current_source = ""
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_number, current_source, current_parts
        text = " ".join(" ".join(current_parts).split()).strip()
        if current_number is not None and current_source and len(text) >= 20:
            entries.append((current_number, current_source, text))
        current_number = None
        current_source = ""
        current_parts = []

    for span in spans:
        if not any("reference" in item.casefold() for item in span.section_path):
            continue
        matches = list(REFERENCE_NUMBER_PATTERN.finditer(span.text))
        if not matches:
            if current_number is not None:
                current_parts.append(span.text)
            else:
                text = " ".join(span.text.split()).strip()
                if len(text) >= 20:
                    entries.append((None, span.span_id, text))
            continue
        prefix = span.text[: matches[0].start()].strip()
        if prefix and current_number is not None:
            current_parts.append(prefix)
        for index, match in enumerate(matches):
            flush()
            current_number = int(match.group(1))
            current_source = span.span_id
            end = matches[index + 1].start() if index + 1 < len(matches) else None
            current_parts = [span.text[match.end() : end].strip()]
    flush()

    references: list[ReferenceEntry] = []
    normalized_manuscript_doi = _normalize_doi(manuscript_doi)
    for citation_number, source_span_id, raw_text in entries:
        dois = [
            value
            for raw_value in DOI_PATTERN.findall(raw_text)
            if (value := raw_value.rstrip(".,;)"))
            and _normalize_doi(value) != normalized_manuscript_doi
        ]
        identity = f"{source_span_id}|{citation_number or 'unnumbered'}|{raw_text}"
        references.append(
            ReferenceEntry(
                reference_id=(
                    "REF-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                ),
                raw_text=raw_text,
                source_span_id=source_span_id,
                citation_number=citation_number,
                title=_reference_title(raw_text),
                doi=dois[0] if dois else None,
                publication_year=_reference_year(raw_text),
            )
        )
    return references


def _reference_year(text: str) -> int | None:
    years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", text)]
    return years[-1] if years else None


REFERENCE_VENUE_PATTERN = re.compile(
    r"\.\s+(?=(?:Nature|Science|Cell|Nat\.|Mol\.|J\.|Proc\.|Adv\.|Nano|ACS|"
    r"Chem\.|Genes\s+Dev\.|EMBO\s+J\.|FEMS|PLoS|Phys\.|IEEE|Angew\.|"
    r"Soft\s+Matter|Macromolecules|Small))",
    re.IGNORECASE,
)


def _reference_title(text: str) -> str | None:
    """Extract a conservative title for exact bibliography-anchor search.

    A title is useful even when a PDF bibliography omits the DOI, which is the
    common case in the Nature development papers.  Fail closed when the author
    boundary or venue boundary cannot be identified; the raw citation remains
    available for traceability.
    """

    compact = " ".join(text.split()).strip()
    if not compact:
        return None
    author_end = compact.casefold().find(" et al. ")
    if author_end >= 0:
        start = author_end + len(" et al. ")
    else:
        initials = list(re.finditer(r"(?:[A-Z]\.(?:\s*|$)){1,3}", compact))
        if not initials:
            return None
        start = initials[-1].end()
    remainder = compact[start:].strip(" .")
    venue = REFERENCE_VENUE_PATTERN.search(remainder)
    if venue is None:
        return None
    title = remainder[: venue.start()].strip(" .")
    return title if len(title) >= 8 else None


def _normalize_doi(value: str | None) -> str:
    text = str(value or "").strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        text = text.removeprefix(prefix)
    return text.rstrip(".,;)")


def build_claim_ledger(
    claims: Sequence[PaperClaim],
    method_result: MethodResultLedger,
    references: Sequence[ReferenceEntry],
) -> ClaimLedger:
    by_type = {
        claim_type: [
            claim.claim_id for claim in claims if claim.claim_type == claim_type
        ]
        for claim_type in ClaimType
    }
    return ClaimLedger(
        novelty_claim_ids=by_type[ClaimType.NOVELTY],
        method_claim_ids=by_type[ClaimType.METHOD],
        result_claim_ids=by_type[ClaimType.RESULT],
        scope_claim_ids=by_type[ClaimType.SCOPE],
        causal_claim_ids=by_type[ClaimType.CAUSAL],
        significance_claim_ids=by_type[ClaimType.SIGNIFICANCE],
        method_span_ids=list(
            dict.fromkeys(
                [
                    *method_result.dataset_sample,
                    *method_result.design_comparator,
                    *method_result.model_algorithm,
                    *method_result.baselines_metrics_statistics,
                    *method_result.ablation_robustness,
                ]
            )
        ),
        result_span_ids=list(dict.fromkeys(method_result.main_results)),
        table_span_ids=list(dict.fromkeys(method_result.figures_tables)),
        reference_ids=[reference.reference_id for reference in references],
    )


def extract_ledgers(paper_ir: PaperIR) -> ClaimLedger:
    """Rebuild the compact claim ledger from a compiled PaperIR."""
    return build_claim_ledger(
        paper_ir.claims,
        paper_ir.method_result_ledger,
        paper_ir.references,
    )


class PaperCompiler:
    """Build PaperIR from canonical Markdown without external manuscript upload."""

    def __init__(self, config: GearConfig | None = None) -> None:
        self.config = config or load_config()

    def compile(self, request: ReviewRequest) -> PaperIR:
        path = Path(request.paper_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.casefold()
        if suffix not in {".md", ".markdown", ".pdf"}:
            raise ValueError("paper must be Markdown (.md/.markdown) or PDF (.pdf)")
        source_format: Literal["pdf", "markdown"] = (
            "pdf" if suffix == ".pdf" else "markdown"
        )
        parse_failure: str | None = None
        try:
            markdown = (
                pdf_to_markdown(path) if source_format == "pdf" else read_markdown(path)
            )
            pages = markdown_pages(markdown)
        except (OSError, UnicodeDecodeError, PdfReadError, ValueError) as exc:
            markdown = ""
            pages = []
            parse_failure = type(exc).__name__
        paper_hash = _sha256_text(markdown)
        spans = segment_pages(pages, paper_hash) if pages else []
        total_characters = sum(len(page.text) for page in pages)
        nonempty = sum(bool(page.text.strip()) for page in pages)
        ratio = nonempty / len(pages) if pages else 0.0
        combined_text = "".join(page.text for page in pages)
        visible = sum(character.isalnum() for character in combined_text)
        visible_ratio = visible / max(len(combined_text), 1)
        replacement_ratio = combined_text.count("�") / max(len(combined_text), 1)
        severe_garble = bool(combined_text) and (
            visible_ratio < 0.15 or replacement_ratio > 0.02
        )
        flags: list[str] = []
        if parse_failure:
            flags.append(f"paper_parse_error:{parse_failure}")
        if not pages or total_characters == 0:
            status = ParseStatus.UNAVAILABLE
            flags.append("paper_text_unavailable")
        elif severe_garble:
            status = ParseStatus.DEGRADED
            flags.append("paper_text_severely_garbled")
        elif (
            total_characters < self.config.minimum_pdf_characters
            or ratio < self.config.minimum_nonempty_page_ratio
        ):
            status = ParseStatus.DEGRADED
            flags.append("paper_text_coverage_below_gate")
        else:
            status = ParseStatus.READY
        metadata = request.metadata.model_copy(deep=True)
        if not metadata.title and spans:
            metadata.title = spans[0].text.splitlines()[0][:500]
        paper_id = metadata.openalex_id or metadata.doi or paper_hash
        claims = [] if severe_garble else extract_claims(spans, self.config.max_claims)
        method_result = extract_method_result_ledger(spans)
        references = extract_references(spans, metadata.doi)
        sections = {item.section_path[0] for item in spans if item.section_path}
        document_ratio = (
            sum(item.section_path == ["Document"] for item in spans) / len(spans)
            if spans
            else 1.0
        )
        blocking = []
        if status == ParseStatus.UNAVAILABLE:
            readiness = EvidenceReadiness.UNAVAILABLE
            blocking.append("paper_text_unavailable")
        elif status != ParseStatus.READY:
            readiness = EvidenceReadiness.LIMITED
            blocking.append("paper_parse_degraded")
        else:
            readiness = EvidenceReadiness.READY
        quality_report = PaperQualityReport(
            evidence_readiness=readiness,
            section_count=len(sections - {"Document"}),
            document_only_ratio=document_ratio,
            table_figure_anchor_count=len(method_result.figures_tables),
            semantic_extraction_ready=False,
            blocking_reasons=blocking,
            advisories=(["document_only_sections"] if document_ratio > 0.80 else []),
        )
        return PaperIR(
            paper_id=paper_id,
            paper_path=path,
            paper_sha256=paper_hash,
            source_format=source_format,
            markdown=markdown,
            metadata=metadata,
            pages=pages,
            spans=spans,
            claims=claims,
            claim_ledger=build_claim_ledger(claims, method_result, references),
            method_result_ledger=method_result,
            references=references,
            parse_status=status,
            quality_flags=flags,
            quality_report=quality_report,
        )


def compile_paper(
    request: ReviewRequest,
    config: GearConfig | None = None,
) -> PaperIR:
    return PaperCompiler(config).compile(request)


__all__ = [
    "PaperCompiler",
    "build_claim_ledger",
    "compile_paper",
    "extract_claims",
    "extract_ledgers",
    "extract_pdf_pages",
    "markdown_pages",
    "pdf_to_markdown",
    "read_markdown",
    "segment_pages",
]
