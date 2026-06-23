from __future__ import annotations

import pandas as pd

from scripts import diagnose_fig3_subset_candidates as diag


def test_score_subset_reports_family_balance_and_effects() -> None:
    rows = []
    for domain, family in [("a", "bio"), ("b", "materials"), ("c", "physics")]:
        for idx in range(40):
            score = idx + (10 if domain == "a" else 0)
            rows.append(
                {
                    "paper_id": f"{domain}{idx}",
                    "domain": domain,
                    "year": 2000 + idx % 4,
                    "S_w_oof": float(score),
                    "S_equal": float(idx),
                    "RGPM": float(score + idx % 3),
                }
            )
    table = pd.DataFrame(rows)
    family_map = {"a": "bio", "b": "materials", "c": "physics"}

    row = diag.score_subset(table, ["a", "b", "c"], "S_w_oof", "RGPM", family_map)

    assert row["n_domains"] == 3
    assert row["n_papers"] == 120
    assert row["top_family_share"] == 1 / 3
    assert row["learned_oof_spearman"] > 0.90
    assert row["high_vs_low_tertile_median_rgpm_lift_pp"] > 0.0
    assert "top_vs_bottom_score_decile_rgpm_top20_enrichment" in row


def test_generate_subset_candidates_keeps_requested_domain_sizes() -> None:
    rows = []
    for domain in ["a", "b", "c", "d"]:
        for idx in range(20):
            rows.append(
                {
                    "paper_id": f"{domain}{idx}",
                    "domain": domain,
                    "year": 2000 + idx % 2,
                    "S_w_oof": float(idx),
                    "S_equal": float(idx % 5),
                    "RGPM": float(idx),
                }
            )
    table = pd.DataFrame(rows)
    candidates = diag.generate_subset_candidates(
        table,
        "S_w_oof",
        "RGPM",
        {"a": "x", "b": "y", "c": "z", "d": "w"},
        min_domains=3,
        max_domains=4,
        year_cutoffs=[None],
    )

    assert set(candidates["n_domains"]) == {3, 4}
    assert candidates["passes_family_balance"].eq(1).all()
