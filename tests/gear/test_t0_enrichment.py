from __future__ import annotations

from datetime import date

from gear.t0_enrichment import OpenAlexT0Enricher


def test_review_cutoff_precedes_final_openalex_publication_year(paper_ir):
    year = OpenAlexT0Enricher._publication_year(
        paper_ir,
        {"publication_year": 2023},
        date(2021, 7, 14),
    )
    assert year == 2021
