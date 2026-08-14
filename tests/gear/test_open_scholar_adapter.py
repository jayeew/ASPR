from __future__ import annotations

from datetime import date

from gear.scholar import OpenScholar


class Args:
    s2_api_key = ""
    openalex_api_key = ""
    and_search = False
    retrieval_provider = "semantic_scholar"


class Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload
        self.headers = {}

    def json(self):
        return self.payload


def test_semantic_scholar_date_filter_is_api_and_local(monkeypatch):
    calls = []

    def fake_get(url, *, params=None, headers=None, timeout=None):
        calls.append((url, params, headers))
        return Response(
            {
                "data": [
                    {
                        "paperId": "before",
                        "title": "Earlier work",
                        "year": 2019,
                        "publicationDate": "2019-01-01",
                    },
                    {
                        "paperId": "cutoff",
                        "title": "Cutoff-day work",
                        "year": 2020,
                        "publicationDate": "2020-06-01",
                    },
                    {
                        "paperId": "same-year-unknown",
                        "title": "Unknown date",
                        "year": 2020,
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
    assert {row["paperId"] for row in rows} == {"before", "same-year-unknown"}
    assert calls[0][1]["year"] == "1800-2020"
    assert "publicationDate" in calls[0][1]["fields"]


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
