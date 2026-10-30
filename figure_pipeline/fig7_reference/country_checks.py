"""Independent table reconciliation using exploded paper-country records."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .country_data import ASSETS, BASE, read, write


def verify(out: Path) -> dict[str, Any]:
    papers = read(out / "data/paper_countries.json")
    original = read(out / "data/papers.json")
    assert len(papers) == len({r["paper_id"] for r in papers}) == len(original)
    records = []
    for p in papers:
        if p["cohort"] == "historical" and p["status"] == "complete":
            codes = sorted(set(p["country_codes"]))
            assert codes
            records.extend(
                {"paper_id": p["paper_id"], "country_code": c, "weight": 1 / len(codes)}
                for c in codes
            )
    attribution = pd.DataFrame(records)
    total = attribution.paper_id.nunique()
    expected_country = attribution.groupby("country_code").weight.sum().sort_index()
    c = (
        pd.read_csv(out / "tables/c_country_contributions.csv", keep_default_na=False)
        .set_index("country_code")
        .sort_index()
    )
    np.testing.assert_allclose(c.fractional_papers, expected_country, atol=1e-8)
    assert math.isclose(c.fractional_papers.sum(), total, abs_tol=1e-7)
    phrases = read(BASE / "data/historical_wordcloud.json")
    displayed = sorted(
        [r for r in phrases if r["displayed"]],
        key=lambda r: (-r["frequency"], r["phrase"]),
    )[:16]
    membership = pd.DataFrame(
        [
            {"paper_id": p, "topic": t["display_label"]}
            for t in displayed
            for p in set(t["source_paper_ids"])
        ]
    )
    expanded = membership.merge(attribution, on="paper_id", validate="many_to_many")
    numerator = expanded.groupby(["country_code", "topic"]).weight.sum()
    topic_n = expanded.groupby("topic").paper_id.nunique()
    d = pd.read_csv(out / "tables/d_topic_preferences.csv", keep_default_na=False)
    for row in d.itertuples():
        expected = numerator.get((row.country_code, row.topic), 0)
        assert math.isclose(row.fractional_papers, expected, abs_tol=1e-8)
        assert (
            row.topic_denominator == topic_n[row.topic]
            and row.corpus_denominator == total
        )
        ratio = (
            expected / expected_country[row.country_code] / (topic_n[row.topic] / total)
        )
        assert math.isclose(row.relative_preference, ratio, abs_tol=1e-8)
        if ratio:
            assert math.isclose(
                float(row.log2_preference), math.log2(ratio), abs_tol=1e-8
            )
    nodes = pd.read_parquet(
        ASSETS / "claim_nodes.parquet", columns=["parent_paper_id", "claim_type"]
    )
    per_paper = (
        nodes.groupby(["parent_paper_id", "claim_type"])
        .size()
        .rename("claims")
        .reset_index()
    )
    per_paper["proportion"] = per_paper.claims / per_paper.groupby(
        "parent_paper_id"
    ).claims.transform("sum")
    joined = per_paper.merge(
        attribution, left_on="parent_paper_id", right_on="paper_id"
    )
    joined["mass"] = joined.proportion * joined.weight
    expected_types = joined.groupby(["country_code", "claim_type"]).mass.sum()
    e = pd.read_csv(out / "tables/e_contribution_types.csv", keep_default_na=False)
    for row in e.itertuples():
        expected = (
            expected_types.get((row.country_code, row.claim_type), 0)
            / expected_country[row.country_code]
        )
        assert math.isclose(row.proportion, expected, abs_tol=1e-9)
    np.testing.assert_allclose(e.groupby("country_code").proportion.sum(), 1)
    f = pd.read_csv(out / "tables/f_future_sources.csv", keep_default_na=False)
    links = pd.read_csv(out / "tables/f_source_provenance.csv", keep_default_na=False)
    assert not links.duplicated(["topic", "paper_id", "country_code"]).any()
    np.testing.assert_allclose(links.groupby(["topic", "paper_id"]).weight.sum(), 1)
    np.testing.assert_allclose(f.groupby("topic").proportion.sum(), 1)
    for row in f.itertuples():
        part = links.loc[
            (links.topic == row.topic) & (links.display_group == row.country_code)
        ]
        assert math.isclose(
            row.proportion, part.weight.sum() / row.source_papers, abs_tol=1e-9
        )
    plot = read(out / "data/plot_data.json")
    expected_order = sorted(
        expected_country.index, key=lambda code: (-expected_country[code], code)
    )[:12]
    assert plot["top_countries"] == expected_order
    assert [r["display_label"] for r in plot["historical_topics"]] == [
        r["display_label"] for r in displayed
    ]
    future, seen = [], set()
    for topic in sorted(
        [r for r in read(BASE / "data/forecast_wordcloud.json") if r["displayed"]],
        key=lambda r: (-r["prominence"], r["display_label"]),
    ):
        key = tuple(sorted(topic["direction_ids"]))
        if key not in seen:
            seen.add(key)
            future.append(topic["display_label"])
        if len(future) == 12:
            break
    assert future == [r["display_label"] for r in plot["future_topics"]]
    result = {
        "independent_country_reconciliation": True,
        "independent_preference_reconciliation": True,
        "independent_type_reconciliation": True,
        "future_source_mass_conserved": True,
        "selections_match_fixed_rules": True,
        "eligible_historical_papers": total,
        "countries": len(c),
        "historical_topic_cells": len(d),
        "future_cells": len(f),
    }
    write(out / "qa/independent_reconciliation.json", result)
    return result
