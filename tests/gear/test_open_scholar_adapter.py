from __future__ import annotations

from datetime import date
from io import BytesIO

from reportlab.pdfgen import canvas

from gear.scholar import OpenScholar


class Args:
    s2_api_key = ""
    openalex_api_key = ""
    and_search = False
    retrieval_provider = "openalex"


class Response:
    status_code = 200
    text = ""

    def __init__(self, payload, *, content=b"", status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {}
        self.content = content
        self.closed = False

    def json(self):
        return self.payload

    def iter_content(self, *, chunk_size):
        yield self.content

    def close(self):
        self.closed = True


def test_openalex_is_default_and_enforces_date_filter_at_api_and_local(monkeypatch):
    calls = []

    def fake_get(url, *, params=None, headers=None, timeout=None):
        calls.append((url, params, headers))
        return Response(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Earlier work",
                        "publication_year": 2019,
                        "publication_date": "2019-01-01",
                    },
                    {
                        "id": "https://openalex.org/W2",
                        "display_name": "Cutoff-day work",
                        "publication_year": 2020,
                        "publication_date": "2020-06-01",
                    },
                    {
                        "id": "https://openalex.org/W3",
                        "display_name": "Unknown date",
                        "publication_year": 2020,
                    },
                ]
            }
        )

    monkeypatch.setattr("gear.scholar.requests.get", fake_get)
    scholar = OpenScholar(Args())
    rows = scholar.search_query(
        "evidence controller",
        date_to=date(2020, 6, 1),
        limit=5,
    )
    assert {row["paperId"] for row in rows} == {
        "https://openalex.org/W1",
        "https://openalex.org/W3",
    }
    assert calls[0][1]["search"] == "evidence controller"
    assert calls[0][1]["filter"] == (
        "has_abstract:true,is_retracted:false,"
        "from_publication_date:1800-01-01,to_publication_date:2020-06-01"
    )
    assert calls[0][1]["sort"] == "relevance_score:desc"
    assert calls[0][1]["per_page"] == 5


def test_openalex_semantic_search_uses_supported_year_filter_and_local_cutoff(
    monkeypatch,
):
    calls = []

    def fake_get(url, *, params=None, headers=None, timeout=None):
        calls.append((url, params))
        return Response(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Earlier semantic match",
                        "publication_year": 2019,
                        "publication_date": "2019-01-01",
                        "abstract_inverted_index": {"same": [0], "mechanism": [1]},
                    },
                    {
                        "id": "https://openalex.org/W2",
                        "display_name": "After the exact cutoff",
                        "publication_year": 2020,
                        "publication_date": "2020-12-01",
                        "abstract_inverted_index": {"same": [0], "mechanism": [1]},
                    },
                ]
            }
        )

    monkeypatch.setattr("gear.scholar.requests.get", fake_get)
    rows = OpenScholar(Args()).search_query(
        "same scientific mechanism",
        date_to=date(2020, 6, 1),
        limit=5,
        search_mode="semantic",
    )
    assert [row["paperId"] for row in rows] == ["https://openalex.org/W1"]
    assert calls[0][1]["search.semantic"] == "same scientific mechanism"
    assert calls[0][1]["filter"] == (
        "has_abstract:true,is_retracted:false,publication_year:<2021"
    )
    assert "sort" not in calls[0][1]


def test_openalex_semantic_server_error_falls_back_to_relevance_text(monkeypatch):
    calls = []

    def fake_get(url, *, params=None, headers=None, timeout=None):
        calls.append(params)
        if "search.semantic" in params:
            return Response({}, status_code=504)
        return Response(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "display_name": "Fallback result",
                        "publication_year": 2019,
                        "publication_date": "2019-01-01",
                        "abstract_inverted_index": {"same": [0], "object": [1]},
                    }
                ]
            }
        )

    monkeypatch.setattr("gear.scholar.requests.get", fake_get)
    rows = OpenScholar(Args()).search_query(
        "same scientific object and mechanism",
        date_to=date(2020, 6, 1),
        limit=5,
        search_mode="semantic",
    )
    assert [row["paperId"] for row in rows] == ["https://openalex.org/W1"]
    assert "search.semantic" in calls[0]
    assert calls[1]["search"] == "same scientific object and mechanism"
    assert calls[1]["sort"] == "relevance_score:desc"


def test_openalex_mapping_keeps_dates_references_and_citation_direction(monkeypatch):
    calls = []

    def fake_get(url, *, params=None, headers=None, timeout=None):
        calls.append((url, params))
        return Response(
            {
                "results": [
                    {
                        "id": "https://openalex.org/W2",
                        "ids": {"openalex": "https://openalex.org/W2"},
                        "display_name": "Citing work",
                        "publication_year": 2001,
                        "publication_date": "2001-01-02",
                        "referenced_works": ["https://openalex.org/W1"],
                    }
                ]
            }
        )

    monkeypatch.setattr("gear.scholar.requests.get", fake_get)
    args = Args()
    args.retrieval_provider = "openalex"
    scholar = OpenScholar(args)
    rows = scholar.fetch_neighbors("W1", "citations", limit=3)
    assert rows[0]["publication_date"] == "2001-01-02"
    assert rows[0]["referenced_works"] == ["https://openalex.org/W1"]
    assert calls[0][1]["filter"] == "cites:W1"


def test_openalex_content_pdf_is_downloaded_and_extracted(monkeypatch):
    buffer = BytesIO()
    document = canvas.Canvas(buffer)
    document.drawString(48, 800, "Recovered OpenAlex full-text evidence.")
    document.save()
    response = Response({}, content=buffer.getvalue())
    calls = []

    def fake_get(url, *, params=None, headers=None, timeout=None, stream=None):
        calls.append((url, stream))
        return response

    monkeypatch.setattr("gear.scholar.requests.get", fake_get)
    text = OpenScholar(Args()).fetch_pdf_text(
        "https://openalex.org/W1",
        max_bytes=1_000_000,
        max_pages=5,
        max_characters=5_000,
    )
    assert "Recovered OpenAlex full-text evidence" in text
    assert calls[0][0].endswith("/W1.pdf")
    assert calls[0][1] is True
    assert response.closed is True
