from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_quality import write_figure_quality_report, write_json, write_run_manifest

DEFAULT_FIG3_RUN_DIR = PROJECT_ROOT / "outputs" / "redraw_v6a_best_fig3" / "multi_domain"
DEFAULT_WORKS_TABLE = PROJECT_ROOT / "data" / "knowledge_corpus" / "v2_publication_v6a_locked_candidate" / "works.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig7"

OPENALEX_SELECT = (
    "id,doi,display_name,publication_year,publication_date,type,cited_by_count,"
    "referenced_works_count,primary_location,primary_topic,open_access,authorships"
)

NATURE_RED = "#8B1E2D"
SURFACE = "#FCFCFD"
PANEL = "#FFFFFF"
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
AXIS = "#D7DBE7"
NEUTRAL = "#C5CAD3"
NEUTRAL_MID = "#7A828F"
NEUTRAL_DARK = "#464C55"
BLUE = "#A3BEFA"
BLUE_DARK = "#2E4780"
GOLD = "#FFE15B"
GOLD_DARK = "#736422"
ORANGE = "#F0986E"
OLIVE = "#A3D576"
PINK = "#F390CA"
PURPLE_GREY = "#B8AEC8"

FAMILY_COLORS = {
    "Nature Portfolio": NATURE_RED,
    "Science family": GOLD_DARK,
    "Cell Press": BLUE_DARK,
    "PNAS": "#386411",
    "Lancet family": "#BD569B",
    "IEEE / ACM": "#5477C4",
    "Elsevier": "#CC6F47",
    "Springer Nature (other)": "#7A828F",
    "Wiley": "#8A3A6F",
    "ACS": "#736422",
    "Other / low-sample families": "#C5CAD3",
    "Other publishers": "#AEB4BF",
    "Missing venue metadata": "#D7DBE7",
}

MECHANISM_COLUMNS = ["B_z", "RS_z", "DeltaQ0_z", "Uzzi_z", "RTD_z", "BurtIP_z", "PDE_z"]
MECHANISM_LABELS = {
    "B_z": "Boundary\nbridging",
    "RS_z": "Reference\nspread",
    "DeltaQ0_z": "Community\nshift",
    "Uzzi_z": "Atypical\nmix",
    "RTD_z": "Translation\ndistance",
    "BurtIP_z": "Brokerage\npotential",
    "PDE_z": "Diffusion\nentropy",
}


@dataclass(frozen=True)
class VenueFamily:
    family: str
    rule: str


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").lower().replace("&", "and").split())


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def map_venue_family(
    source_name: object,
    host_name: object = "",
    lineage_names: Optional[Sequence[object]] = None,
) -> VenueFamily:
    """Map an OpenAlex source/publisher tuple to an auditable venue family."""
    source = _norm_text(source_name)
    host = _norm_text(host_name)
    lineage = [_norm_text(item) for item in lineage_names or []]
    lineage_text = " | ".join(lineage)
    combined = " | ".join([source, host, lineage_text])

    if not source:
        return VenueFamily("Missing venue metadata", "missing_source_name")

    science_sources = {
        "science",
        "science advances",
        "science immunology",
        "science robotics",
        "science signaling",
        "science translational medicine",
    }
    if source in science_sources or host == "american association for the advancement of science":
        return VenueFamily("Science family", "aaas_science_source_or_host")

    if "cell press" in combined:
        return VenueFamily("Cell Press", "host_lineage_contains_cell_press")

    if source.startswith("cell ") or source in {"cell", "neuron", "immunity", "molecular cell"}:
        return VenueFamily("Cell Press", "source_name_cell_press_journal")

    if source.startswith("the lancet") or source.startswith("lancet "):
        return VenueFamily("Lancet family", "source_name_lancet_family")

    if "proceedings of the national academy of sciences" in source or source.startswith("pnas"):
        return VenueFamily("PNAS", "source_name_pnas")

    nature_exact = {
        "nature",
        "nature biotechnology",
        "nature chemistry",
        "nature communications",
        "nature genetics",
        "nature materials",
        "nature medicine",
        "nature methods",
        "nature nanotechnology",
        "nature neuroscience",
        "nature physics",
        "nature reviews genetics",
        "nature reviews molecular cell biology",
        "scientific reports",
        "scientific data",
    }
    nature_prefixes = ("nature ", "npj ")
    communications_nature = {
        "communications biology",
        "communications chemistry",
        "communications earth and environment",
        "communications engineering",
        "communications materials",
        "communications medicine",
        "communications physics",
    }
    if (
        source in nature_exact
        or source.startswith(nature_prefixes)
        or source in communications_nature
        or (source.startswith("communications ") and "springer nature" in combined)
    ):
        return VenueFamily("Nature Portfolio", "strict_nature_portfolio_source_name")

    if _contains_any(combined, ["institute of electrical and electronics engineers", "ieee", "association for computing machinery"]):
        return VenueFamily("IEEE / ACM", "host_or_source_ieee_acm")

    if "elsevier" in combined:
        return VenueFamily("Elsevier", "host_lineage_contains_elsevier")

    if "springer nature" in combined or "springer" in combined:
        return VenueFamily("Springer Nature (other)", "host_lineage_contains_springer_nature")

    if "wiley" in combined:
        return VenueFamily("Wiley", "host_lineage_contains_wiley")

    if "american chemical society" in combined or source.startswith("acs "):
        return VenueFamily("ACS", "host_or_source_acs")

    if "american physical society" in combined:
        return VenueFamily("APS", "host_lineage_contains_aps")

    if "royal society of chemistry" in combined:
        return VenueFamily("RSC", "host_lineage_contains_rsc")

    if "oxford university press" in combined:
        return VenueFamily("Oxford University Press", "host_lineage_contains_oup")

    if "cambridge university press" in combined:
        return VenueFamily("Cambridge University Press", "host_lineage_contains_cup")

    if "mdpi" in combined:
        return VenueFamily("MDPI", "host_lineage_contains_mdpi")

    if "frontiers media" in combined or source.startswith("frontiers in "):
        return VenueFamily("Frontiers", "host_or_source_frontiers")

    if "public library of science" in combined or source.startswith("plos "):
        return VenueFamily("PLOS", "host_or_source_plos")

    if source == "elife" or "elife sciences" in combined:
        return VenueFamily("eLife", "host_or_source_elife")

    if "iop publishing" in combined:
        return VenueFamily("IOP Publishing", "host_lineage_contains_iop")

    if "american institute of physics" in combined or "aip publishing" in combined:
        return VenueFamily("AIP Publishing", "host_lineage_contains_aip")

    return VenueFamily("Other publishers", "fallback_other_publishers")


def extract_short_work_id(value: object) -> str:
    """Extract W-prefixed OpenAlex ID from a local paper_id or URL."""
    text = str(value or "")
    if "::" in text:
        text = text.rsplit("::", 1)[1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def load_jsonl_cache(path: Path) -> Dict[str, Mapping[str, Any]]:
    records: Dict[str, Mapping[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            work_id = extract_short_work_id(payload.get("id"))
            if work_id:
                records[work_id] = payload
    return records


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _chunks(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def fetch_openalex_metadata(
    work_ids: Sequence[str],
    cache_path: Path,
    *,
    batch_size: int = 50,
    max_fetch: Optional[int] = None,
    sleep_seconds: float = 0.12,
    timeout_seconds: float = 45.0,
) -> Dict[str, Mapping[str, Any]]:
    """Fetch bounded OpenAlex work metadata and keep an append-only JSONL cache."""
    cached = load_jsonl_cache(cache_path)
    unique_ids = list(dict.fromkeys(work_id for work_id in work_ids if work_id.startswith("W")))
    missing = [work_id for work_id in unique_ids if work_id not in cached]
    if max_fetch is not None:
        missing = missing[: max(0, int(max_fetch))]

    session = requests.Session()
    for batch in _chunks(missing, batch_size):
        params = {
            "filter": "openalex:" + "|".join(batch),
            "select": OPENALEX_SELECT,
            "per-page": str(len(batch)),
        }
        last_error: Optional[Exception] = None
        response_payload: Optional[Mapping[str, Any]] = None
        for attempt in range(4):
            try:
                response = session.get("https://api.openalex.org/works", params=params, timeout=timeout_seconds)
                response.raise_for_status()
                response_payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
        if response_payload is None:
            raise RuntimeError(f"OpenAlex metadata fetch failed for batch {batch[:3]}: {last_error}")

        results = list(response_payload.get("results", []))
        _append_jsonl(cache_path, results)
        for result in results:
            cached[extract_short_work_id(result.get("id"))] = result
        time.sleep(sleep_seconds)
    return cached


def source_row_from_work(work_id: str, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Flatten the OpenAlex metadata needed for Fig.7."""
    payload = payload or {}
    primary_location = payload.get("primary_location") if isinstance(payload.get("primary_location"), Mapping) else {}
    source = primary_location.get("source") if isinstance(primary_location.get("source"), Mapping) else {}
    open_access = payload.get("open_access") if isinstance(payload.get("open_access"), Mapping) else {}
    authorships = payload.get("authorships") if isinstance(payload.get("authorships"), list) else []
    lineage_names = source.get("host_organization_lineage_names") or []
    if not isinstance(lineage_names, list):
        lineage_names = []
    family = map_venue_family(source.get("display_name", ""), source.get("host_organization_name", ""), lineage_names)
    return {
        "openalex_work_id": work_id,
        "openalex_id": payload.get("id", ""),
        "openalex_doi": payload.get("doi", ""),
        "openalex_title": payload.get("display_name", ""),
        "openalex_year": payload.get("publication_year", np.nan),
        "publication_date": payload.get("publication_date", ""),
        "openalex_type": payload.get("type", ""),
        "openalex_cited_by_count": payload.get("cited_by_count", np.nan),
        "openalex_reference_count": payload.get("referenced_works_count", np.nan),
        "source_id": source.get("id", ""),
        "source_display_name": source.get("display_name", ""),
        "source_type": source.get("type", ""),
        "issn_l": source.get("issn_l", ""),
        "host_organization": source.get("host_organization", ""),
        "host_organization_name": source.get("host_organization_name", ""),
        "host_organization_lineage_names": " | ".join(str(x) for x in lineage_names),
        "venue_family": family.family,
        "venue_family_rule": family.rule,
        "is_open_access": bool(open_access.get("is_oa", primary_location.get("is_oa", False))),
        "team_size": len(authorships),
        "metadata_found": int(bool(payload.get("id"))),
    }


def finite_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def field_year_zscore(
    df: pd.DataFrame,
    value_col: str,
    *,
    field_col: str = "primary_field",
    year_col: str = "year",
    min_group: int = 8,
) -> pd.Series:
    """Z-score by field-year, with field and global fallback for sparse cells."""
    values = finite_series(df[value_col])
    group_keys = [field_col, year_col]
    group_count = values.groupby([df[field_col], df[year_col]]).transform("count")
    group_mean = values.groupby([df[field_col], df[year_col]]).transform("mean")
    group_std = values.groupby([df[field_col], df[year_col]]).transform("std").replace(0, np.nan)

    field_count = values.groupby(df[field_col]).transform("count")
    field_mean = values.groupby(df[field_col]).transform("mean")
    field_std = values.groupby(df[field_col]).transform("std").replace(0, np.nan)

    global_mean = float(values.mean())
    global_std = float(values.std()) or 1.0
    use_group = group_count.ge(min_group) & group_std.notna()
    use_field = ~use_group & field_count.ge(min_group) & field_std.notna()
    z = pd.Series(np.nan, index=df.index, dtype=float)
    z.loc[use_group] = (values.loc[use_group] - group_mean.loc[use_group]) / group_std.loc[use_group]
    z.loc[use_field] = (values.loc[use_field] - field_mean.loc[use_field]) / field_std.loc[use_field]
    other = z.isna() & values.notna()
    z.loc[other] = (values.loc[other] - global_mean) / global_std
    return z.clip(-4.0, 4.0)


def residualize_controls(
    df: pd.DataFrame,
    value_col: str,
    *,
    article_type_col: str = "article_type",
    reference_col: str = "reference_count",
    team_size_col: str = "team_size",
    oa_col: str = "is_open_access",
) -> pd.Series:
    """Residualize a normalized metric against article type and reference controls."""
    y = finite_series(df[value_col])
    valid = y.notna()
    if valid.sum() < 5:
        return y

    features = [np.ones(int(valid.sum()))]
    log_refs = np.log1p(finite_series(df.loc[valid, reference_col]).fillna(0.0).to_numpy(dtype=float))
    if np.nanstd(log_refs) > 0:
        features.append((log_refs - np.nanmean(log_refs)) / np.nanstd(log_refs))

    if team_size_col in df.columns:
        log_team = np.log1p(finite_series(df.loc[valid, team_size_col]).fillna(0.0).to_numpy(dtype=float))
        if np.nanstd(log_team) > 0:
            features.append((log_team - np.nanmean(log_team)) / np.nanstd(log_team))

    if oa_col in df.columns:
        oa = df.loc[valid, oa_col].fillna(False).astype(int).to_numpy(dtype=float)
        if np.nanstd(oa) > 0:
            features.append(oa - np.nanmean(oa))

    article_types = df.loc[valid, article_type_col].fillna("unknown").astype(str)
    dummies = pd.get_dummies(article_types, prefix="type", drop_first=True, dtype=float)
    for column in dummies.columns:
        values = dummies[column].to_numpy(dtype=float)
        if values.sum() > 0:
            features.append(values)

    x = np.column_stack(features)
    beta, *_ = np.linalg.lstsq(x, y.loc[valid].to_numpy(dtype=float), rcond=None)
    fitted = x @ beta
    residual = pd.Series(np.nan, index=df.index, dtype=float)
    residual.loc[valid] = y.loc[valid].to_numpy(dtype=float) - fitted
    return residual


def bootstrap_mean_interval(values: Sequence[float], seed: int = 7, n_boot: int = 800) -> Tuple[float, float, float]:
    arr = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (mean, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def bootstrap_sum_interval(values: Sequence[float], seed: int = 13, n_boot: int = 800) -> Tuple[float, float, float]:
    """Bootstrap a fixed-size aggregate contribution interval."""
    arr = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    total = float(np.sum(arr))
    if len(arr) == 1:
        return (total, total, total)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).sum(axis=1)
    return (total, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def bootstrap_ratio_interval(
    flags: Sequence[int],
    expected_rate: float,
    seed: int = 11,
    n_boot: int = 800,
) -> Tuple[float, float, float]:
    arr = np.array(flags, dtype=float)
    if len(arr) == 0 or expected_rate <= 0:
        return (np.nan, np.nan, np.nan)
    observed_rate = float(arr.mean())
    ratio = observed_rate / expected_rate
    if len(arr) == 1:
        return (ratio, ratio, ratio)
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1) / expected_rate
    return (ratio, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def prepare_analysis_frame(
    score_table: Path,
    indicator_table: Path,
    future_table: Path,
    works_table: Path,
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    min_family_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge Fig.3 metrics with OpenAlex venue metadata and normalized controls."""
    score = pd.read_csv(score_table)
    indicators = pd.read_csv(indicator_table)
    future = pd.read_csv(future_table)
    works = pd.read_csv(
        works_table,
        usecols=lambda col: col
        in {
            "id",
            "short_id",
            "document_type",
            "reference_count",
            "cited_by_count",
            "domain",
            "primary_field",
            "year",
        },
    )
    indicators = indicators[["paper_id"] + [col for col in MECHANISM_COLUMNS if col in indicators.columns]]
    future_cols = [
        "paper_id",
        "n_future_citers",
        "community_reach",
        "field_entropy",
        "cross_community_adoption",
        "path_shortening",
        "modularity_shock",
        "partition_change",
        "boundary_mixing",
        "hub_formation",
    ]
    future = future[[col for col in future_cols if col in future.columns]]
    df = score.merge(indicators, on="paper_id", how="left", suffixes=("", "_indicator"))
    df = df.merge(future, on="paper_id", how="left")
    df["openalex_work_id"] = df["paper_id"].map(extract_short_work_id)

    source_rows = [source_row_from_work(work_id, metadata.get(work_id)) for work_id in df["openalex_work_id"]]
    meta = pd.DataFrame(source_rows)
    df = df.merge(meta, on="openalex_work_id", how="left")
    works = works.rename(columns={"id": "local_openalex_id", "short_id": "openalex_work_id", "document_type": "local_document_type"})
    df = df.merge(works, on="openalex_work_id", how="left", suffixes=("", "_works"))

    df["article_type"] = df["openalex_type"].fillna("").replace("", np.nan)
    df["article_type"] = df["article_type"].fillna(df.get("local_document_type", "unknown")).fillna("unknown")
    df["reference_count_control"] = finite_series(df["reference_count"]).fillna(finite_series(df.get("openalex_reference_count", pd.Series(np.nan, index=df.index))))
    df["reference_count"] = df["reference_count_control"]
    df["team_size"] = finite_series(df["team_size"]).fillna(0)
    df["is_open_access"] = df["is_open_access"].fillna(False).astype(bool)

    df["publication_day_signal_raw"] = finite_series(df["S_w"])
    df["future_impact_raw"] = finite_series(df["RGPM"])
    df["publication_day_signal_fy_z"] = field_year_zscore(df, "publication_day_signal_raw")
    df["future_impact_fy_z"] = field_year_zscore(df, "future_impact_raw")
    df["publication_day_signal_controlled"] = residualize_controls(df, "publication_day_signal_fy_z")
    df["future_impact_controlled"] = residualize_controls(df, "future_impact_fy_z")

    for col in MECHANISM_COLUMNS:
        if col in df.columns:
            z_col = f"{col}_field_year_z"
            df[z_col] = field_year_zscore(df, col)
            df[f"{col}_controlled"] = residualize_controls(df, z_col)

    family_counts = df["venue_family"].fillna("Missing venue metadata").value_counts()
    df["venue_family_plot"] = df["venue_family"].fillna("Missing venue metadata")
    low_sample = df["venue_family_plot"].map(family_counts).fillna(0).lt(min_family_n)
    df.loc[low_sample, "venue_family_plot"] = "Other / low-sample families"
    return df, meta


def build_venue_family_mapping_audit(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_id",
        "source_display_name",
        "source_type",
        "issn_l",
        "host_organization_name",
        "host_organization_lineage_names",
        "venue_family",
        "venue_family_rule",
    ]
    audit = (
        df[cols]
        .fillna("")
        .groupby(cols, dropna=False)
        .size()
        .reset_index(name="n_papers")
        .sort_values(["venue_family", "n_papers", "source_display_name"], ascending=[True, False, True])
    )
    return audit


def build_portfolio_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family = []
    for name, part in df.groupby("venue_family_plot", dropna=False):
        signal_mean, signal_low, signal_high = bootstrap_mean_interval(part["publication_day_signal_controlled"])
        future_mean, future_low, future_high = bootstrap_mean_interval(part["future_impact_controlled"])
        vci_total, vci_low, vci_high = bootstrap_sum_interval(part["future_impact_controlled"])
        family.append(
            {
                "venue_family": name,
                "n_papers": int(len(part)),
                "n_sources": int(part["source_id"].replace("", np.nan).nunique()),
                "mean_publication_day_signal": signal_mean,
                "publication_day_signal_ci_low": signal_low,
                "publication_day_signal_ci_high": signal_high,
                "vci": vci_total,
                "vci_ci_low": vci_low,
                "vci_ci_high": vci_high,
                "vci_per_paper": future_mean,
                "positive_future_contribution": float(np.maximum(part["future_impact_controlled"], 0).sum()),
                "positive_publication_day_signal": float(np.maximum(part["publication_day_signal_controlled"], 0).sum()),
                "future_impact": future_mean,
                "future_impact_ci_low": future_low,
                "future_impact_ci_high": future_high,
                "mean_reference_count": float(finite_series(part["reference_count"]).mean()),
                "article_type_mix": "; ".join(f"{k}:{v}" for k, v in Counter(part["article_type"].fillna("unknown")).most_common(4)),
            }
        )
    portfolio = pd.DataFrame(family).sort_values("vci", ascending=False)
    portfolio["vci_rank"] = np.arange(1, len(portfolio) + 1)

    rankings = portfolio.copy()
    for k, quantile in [("top_1pct", 0.99), ("top_5pct", 0.95)]:
        threshold = float(df["future_impact_controlled"].quantile(quantile))
        df[k] = (df["future_impact_controlled"] >= threshold).astype(int)

    enrichment_rows = []
    for name, part in df.groupby("venue_family_plot", dropna=False):
        for label in ["top_1pct", "top_5pct"]:
            expected_rate = float(df[label].mean())
            ratio, low, high = bootstrap_ratio_interval(part[label].astype(int), expected_rate)
            enrichment_rows.append(
                {
                    "venue_family": name,
                    "top_k": label,
                    "n_papers": int(len(part)),
                    "observed_top_k": int(part[label].sum()),
                    "expected_top_k": float(len(part) * expected_rate),
                    "enrichment_ratio": ratio,
                    "enrichment_ci_low": low,
                    "enrichment_ci_high": high,
                    "overall_top_k_rate": expected_rate,
                    "top_k_metric": "future_impact_controlled",
                }
            )
    enrichment = pd.DataFrame(enrichment_rows)

    mechanism_rows = []
    for name, part in df.groupby("venue_family_plot", dropna=False):
        row: Dict[str, Any] = {"venue_family": name, "n_papers": int(len(part))}
        for col in MECHANISM_COLUMNS:
            c = f"{col}_controlled"
            if c in part.columns:
                row[col] = float(part[c].mean())
        mechanism_rows.append(row)
    mechanism = pd.DataFrame(mechanism_rows)
    if not portfolio.empty:
        order = list(portfolio["venue_family"])
        mechanism["_order"] = mechanism["venue_family"].map({name: i for i, name in enumerate(order)})
        mechanism = mechanism.sort_values("_order").drop(columns=["_order"])

    binned = df.copy()
    binned["nature_status"] = np.where(binned["venue_family_plot"].eq("Nature Portfolio"), "Nature Portfolio", "Other families")
    binned["signal_bin"] = pd.qcut(
        binned["publication_day_signal_controlled"].rank(method="first"),
        q=min(8, max(2, len(binned) // 50)),
        labels=False,
        duplicates="drop",
    )
    prepost = (
        binned.groupby(["nature_status", "signal_bin"], dropna=False)
        .agg(
            n_papers=("paper_id", "count"),
            mean_publication_day_signal=("publication_day_signal_controlled", "mean"),
            mean_future_impact=("future_impact_controlled", "mean"),
        )
        .reset_index()
    )
    return portfolio, rankings, enrichment, mechanism, prepost


def build_journal_supplement(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_id",
        "journal",
        "venue_family",
        "n_papers",
        "vci",
        "vci_ci_low",
        "vci_ci_high",
        "future_impact",
    ]
    rows = []
    for (source_id, source_name), part in df.groupby(["source_id", "source_display_name"], dropna=False):
        if not str(source_name or "").strip():
            continue
        total, low, high = bootstrap_sum_interval(part["future_impact_controlled"])
        rows.append(
            {
                "source_id": source_id,
                "journal": source_name,
                "venue_family": part["venue_family"].mode().iloc[0] if not part["venue_family"].mode().empty else "",
                "n_papers": int(len(part)),
                "vci": total,
                "vci_ci_low": low,
                "vci_ci_high": high,
                "future_impact": float(part["future_impact_controlled"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["vci", "n_papers"], ascending=[False, False])


def build_metric_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Nature rank across plausible intensity and contribution definitions."""
    rows: List[Dict[str, Any]] = []
    definitions = [
        ("publication_day_signal_controlled", "mean", "publication-day signal intensity"),
        ("publication_day_signal_controlled", "sum", "aggregate publication-day signal"),
        ("publication_day_signal_controlled", "positive_sum", "positive publication-day signal mass"),
        ("publication_day_signal_controlled", "top5_count", "top 5% publication-day signal count"),
        ("future_impact_controlled", "mean", "future graph-impact intensity"),
        ("future_impact_controlled", "sum", "aggregate future graph-impact contribution"),
        ("future_impact_controlled", "positive_sum", "positive future graph-impact mass"),
        ("future_impact_controlled", "top5_count", "top 5% future graph-impact count"),
    ]
    for metric_col, aggregation, label in definitions:
        threshold = float(df[metric_col].quantile(0.95))
        records = []
        for family, part in df.groupby("venue_family_plot", dropna=False):
            values = finite_series(part[metric_col]).dropna()
            if aggregation == "mean":
                value = float(values.mean())
            elif aggregation == "sum":
                value = float(values.sum())
            elif aggregation == "positive_sum":
                value = float(np.maximum(values, 0).sum())
            elif aggregation == "top5_count":
                value = int((values >= threshold).sum())
            else:
                value = np.nan
            records.append({"venue_family": family, "value": value})
        metric_df = pd.DataFrame(records).sort_values("value", ascending=False).reset_index(drop=True)
        metric_df["rank"] = np.arange(1, len(metric_df) + 1)
        nature = metric_df.loc[metric_df["venue_family"].eq("Nature Portfolio")]
        rows.append(
            {
                "metric": metric_col,
                "aggregation": aggregation,
                "label": label,
                "top_family": str(metric_df.iloc[0]["venue_family"]) if not metric_df.empty else "",
                "top_value": float(metric_df.iloc[0]["value"]) if not metric_df.empty else np.nan,
                "nature_rank": int(nature["rank"].iloc[0]) if not nature.empty else 0,
                "nature_value": float(nature["value"].iloc[0]) if not nature.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_confounder_audit(df: pd.DataFrame, portfolio: pd.DataFrame, sensitivity: pd.DataFrame, min_family_n: int) -> pd.DataFrame:
    total = len(df)
    metadata_coverage = float(df["metadata_found"].fillna(0).mean()) if total else 0.0
    field_year_cells = df.groupby(["primary_field", "year"]).size()
    enough_field_year = float((field_year_cells >= 8).mean()) if len(field_year_cells) else 0.0
    nature = portfolio.loc[portfolio["venue_family"].eq("Nature Portfolio")]
    nature_n = int(nature["n_papers"].iloc[0]) if not nature.empty else 0
    nature_rank = int(nature["vci_rank"].iloc[0]) if not nature.empty else 0
    signal_rank = sensitivity.loc[sensitivity["label"].eq("publication-day signal intensity"), "nature_rank"]
    signal_rank_value = int(signal_rank.iloc[0]) if not signal_rank.empty else 0
    top = portfolio.sort_values("vci", ascending=False).head(2)
    interval_separated = False
    if len(top) >= 2 and top.iloc[0]["venue_family"] == "Nature Portfolio":
        interval_separated = bool(top.iloc[0]["vci_ci_low"] > top.iloc[1]["vci_ci_high"])
    controls = [
        {
            "audit_item": "analysis_corpus",
            "status": "measured",
            "value": str(total),
            "note": "Paper-level Fig.3 eligible corpus with publication-day and future graph metrics.",
        },
        {
            "audit_item": "metadata_coverage",
            "status": "pass" if metadata_coverage >= 0.90 else "gap",
            "value": f"{metadata_coverage:.3f}",
            "note": "Share of papers with OpenAlex source metadata in cache/API top-up.",
        },
        {
            "audit_item": "field_year_normalization",
            "status": "measured",
            "value": f"{enough_field_year:.3f}",
            "note": "Share of field-year cells with at least eight papers; sparse cells use documented field/global fallback.",
        },
        {
            "audit_item": "vci_definition",
            "status": "measured",
            "value": "sum(controlled RGPM)",
            "note": "Main VCI is aggregate field-year/control normalized future graph-impact contribution, not per-paper intensity.",
        },
        {
            "audit_item": "article_type_control",
            "status": "measured",
            "value": "; ".join(f"{k}:{v}" for k, v in Counter(df["article_type"].fillna("unknown")).most_common(6)),
            "note": "Article type dummies are included in residualized VCI.",
        },
        {
            "audit_item": "reference_count_control",
            "status": "measured",
            "value": f"median={float(finite_series(df['reference_count']).median()):.1f}",
            "note": "log1p(reference_count) is included in residualized VCI.",
        },
        {
            "audit_item": "team_size_and_oa_control",
            "status": "measured",
            "value": f"team_size_nonzero={float((finite_series(df['team_size']) > 0).mean()):.3f}",
            "note": "log1p(team_size) and open-access flag are included when available.",
        },
        {
            "audit_item": "nature_sample_size",
            "status": "pass" if nature_n >= min_family_n else "gap",
            "value": str(nature_n),
            "note": f"Nature Portfolio must have at least {min_family_n} papers for the main family layer.",
        },
        {
            "audit_item": "nature_rank",
            "status": "pass" if nature_rank == 1 else "gap",
            "value": str(nature_rank),
            "note": "Rank is based on aggregate controlled field-year normalized VCI among main venue families.",
        },
        {
            "audit_item": "per_paper_signal_intensity_rank",
            "status": "measured",
            "value": str(signal_rank_value),
            "note": "Nature is not the top per-paper publication-day signal family; the claim is portfolio-level contribution.",
        },
        {
            "audit_item": "strict_interval_separation",
            "status": "pass" if interval_separated else "gap",
            "value": str(int(interval_separated)),
            "note": "Strict headline support requires Nature lower bootstrap CI to exceed runner-up upper CI.",
        },
    ]
    return pd.DataFrame(controls)


def _hex(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size=size)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    width: int,
    line_spacing: int = 4,
) -> int:
    lines: List[str] = []
    for para in str(text).split("\n"):
        if not para:
            lines.append("")
            continue
        current = ""
        for word in para.split():
            candidate = word if not current else f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=_hex(fill))
        y += font.size + line_spacing
    return y


def _new_panel(title: str, subtitle: str, size: Tuple[int, int] = (1400, 920)) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, _hex(SURFACE))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, size[0] - 18, size[1] - 18), radius=18, fill=_hex(PANEL), outline=_hex(AXIS), width=1)
    draw.text((54, 44), title, font=_font(34, bold=True), fill=_hex(INK))
    _draw_wrapped(draw, (54, 90), subtitle, _font(19), MUTED, size[0] - 108, line_spacing=5)
    return image, draw


def _scale(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if not np.isfinite(value) or vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) * (out_max - out_min) / (vmax - vmin)


def _family_color(name: str) -> str:
    return FAMILY_COLORS.get(name, PURPLE_GREY)


def draw_portfolio_map(portfolio: pd.DataFrame, path: Path) -> None:
    image, draw = _new_panel(
        "a  Venue portfolio map",
        "Family-level portfolio view; x = publication-day signal intensity, y = aggregate future graph-impact contribution, size = papers.",
    )
    plot = portfolio.sort_values("n_papers", ascending=False).head(12).copy()
    left, top, right, bottom = 150, 185, 1280, 790
    draw.rectangle((left, top, right, bottom), fill=_hex("#FFFFFF"), outline=_hex(AXIS))
    for i in range(5):
        y = top + i * (bottom - top) / 4
        draw.line((left, y, right, y), fill=_hex(GRID), width=1)
        x = left + i * (right - left) / 4
        draw.line((x, top, x, bottom), fill=_hex(GRID), width=1)

    x_min, x_max = float(plot["mean_publication_day_signal"].min()), float(plot["mean_publication_day_signal"].max())
    y_min, y_max = float(plot["vci"].min()), float(plot["vci"].max())
    pad_x = max(0.05, (x_max - x_min) * 0.15)
    pad_y = max(0.05, (y_max - y_min) * 0.15)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y
    draw.text((left, bottom + 40), "Publication-day signal intensity (controlled z mean)", font=_font(18), fill=_hex(INK))
    draw.text((42, top + 20), "Aggregate future contribution", font=_font(18), fill=_hex(INK))
    draw.line((left, bottom, right, bottom), fill=_hex(NEUTRAL_DARK), width=2)
    draw.line((left, top, left, bottom), fill=_hex(NEUTRAL_DARK), width=2)
    zero_x = _scale(0, x_min, x_max, left, right)
    zero_y = _scale(0, y_min, y_max, bottom, top)
    draw.line((zero_x, top, zero_x, bottom), fill=_hex("#D0D4DE"), width=1)
    draw.line((left, zero_y, right, zero_y), fill=_hex("#D0D4DE"), width=1)

    max_n = max(1, int(plot["n_papers"].max()))
    for _, row in plot.iterrows():
        x = _scale(float(row["mean_publication_day_signal"]), x_min, x_max, left, right)
        y = _scale(float(row["vci"]), y_min, y_max, bottom, top)
        radius = int(_scale(math.sqrt(float(row["n_papers"])), 1, math.sqrt(max_n), 14, 46))
        color = _family_color(str(row["venue_family"]))
        outline = NATURE_RED if row["venue_family"] == "Nature Portfolio" else NEUTRAL_DARK
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_hex(color), outline=_hex(outline), width=3)
        label = str(row["venue_family"])
        if label == "Other / low-sample families":
            label = "Other low-n"
        draw.text((x + radius + 7, y - 10), label, font=_font(17, bold=row["venue_family"] == "Nature Portfolio"), fill=_hex(INK))
    image.save(path, dpi=(180, 180))


def draw_ranking(rankings: pd.DataFrame, path: Path) -> None:
    image, draw = _new_panel(
        "b  Field-year normalized venue contribution index",
        "Dot-and-interval ranking; VCI is aggregate controlled future graph-impact contribution across papers in each venue family.",
    )
    plot = rankings.sort_values("vci", ascending=True).copy()
    if len(plot) > 18:
        keep = set(plot.tail(17)["venue_family"])
        keep.add("Nature Portfolio")
        plot = plot.loc[plot["venue_family"].isin(keep)].sort_values("vci", ascending=True)
    left, top, right, bottom = 390, 175, 1240, 805
    x_min = float(min(plot["vci_ci_low"].min(), 0))
    x_max = float(max(plot["vci_ci_high"].max(), 0))
    pad = max(0.05, (x_max - x_min) * 0.12)
    x_min, x_max = x_min - pad, x_max + pad
    draw.line((_scale(0, x_min, x_max, left, right), top, _scale(0, x_min, x_max, left, right), bottom), fill=_hex(AXIS), width=2)
    for i, (_, row) in enumerate(plot.iterrows()):
        y = bottom - i * ((bottom - top) / max(1, len(plot) - 1))
        color = _family_color(str(row["venue_family"]))
        draw.text((54, y - 13), str(row["venue_family"])[:30], font=_font(18, bold=row["venue_family"] == "Nature Portfolio"), fill=_hex(INK))
        x1 = _scale(float(row["vci_ci_low"]), x_min, x_max, left, right)
        x2 = _scale(float(row["vci_ci_high"]), x_min, x_max, left, right)
        xm = _scale(float(row["vci"]), x_min, x_max, left, right)
        draw.line((x1, y, x2, y), fill=_hex(color), width=4)
        draw.ellipse((xm - 9, y - 9, xm + 9, y + 9), fill=_hex(color), outline=_hex(NEUTRAL_DARK), width=2)
        draw.text((right + 18, y - 12), f"n={int(row['n_papers'])}", font=_font(16), fill=_hex(MUTED))
    draw.line((left, bottom + 34, right, bottom + 34), fill=_hex(NEUTRAL_DARK), width=2)
    draw.text((left, bottom + 50), f"{x_min:.1f}", font=_font(15), fill=_hex(MUTED))
    draw.text((right - 45, bottom + 50), f"{x_max:.1f}", font=_font(15), fill=_hex(MUTED))
    image.save(path, dpi=(180, 180))


def draw_enrichment(enrichment: pd.DataFrame, path: Path) -> None:
    image, draw = _new_panel(
        "c  Top-K enrichment",
        "Observed / expected share of top future graph-impact papers; squares = top 1%, dots = top 5%.",
    )
    pivot = enrichment.pivot_table(index="venue_family", columns="top_k", values="enrichment_ratio", aggfunc="first")
    pivot["max_ratio"] = pivot.max(axis=1)
    families = list(pivot.sort_values("max_ratio", ascending=False).head(9).index)
    if "Nature Portfolio" not in families:
        families.append("Nature Portfolio")
    plot = enrichment.loc[enrichment["venue_family"].isin(families)].copy()
    family_order = list(
        pivot.loc[families]
        .sort_values("max_ratio", ascending=True)
        .index
    )
    left, top, right, bottom = 420, 175, 1090, 790
    x_min = 0.0
    x_max = float(max(2.0, plot["enrichment_ci_high"].replace([np.inf, -np.inf], np.nan).max()))
    x_ref = _scale(1.0, x_min, x_max, left, right)
    draw.line((x_ref, top, x_ref, bottom), fill=_hex(NEUTRAL_MID), width=2)
    draw.text((x_ref + 5, top - 26), "expected", font=_font(15), fill=_hex(MUTED))
    for i, family in enumerate(family_order):
        y_center = bottom - i * ((bottom - top) / max(1, len(family_order) - 1))
        draw.text((54, y_center - 13), family[:32], font=_font(18, bold=family == "Nature Portfolio"), fill=_hex(INK))
        for top_k, offset, marker in [("top_1pct", -8, "square"), ("top_5pct", 8, "dot")]:
            row = plot.loc[plot["venue_family"].eq(family) & plot["top_k"].eq(top_k)]
            if row.empty:
                continue
            item = row.iloc[0]
            color = _family_color(family)
            y = y_center + offset
            x1 = _scale(float(item["enrichment_ci_low"]), x_min, x_max, left, right)
            x2 = _scale(float(item["enrichment_ci_high"]), x_min, x_max, left, right)
            xm = _scale(float(item["enrichment_ratio"]), x_min, x_max, left, right)
            draw.line((x1, y, x2, y), fill=_hex(color), width=3)
            if marker == "square":
                draw.rectangle((xm - 7, y - 7, xm + 7, y + 7), fill=_hex(color), outline=_hex(NEUTRAL_DARK), width=2)
            else:
                draw.ellipse((xm - 7, y - 7, xm + 7, y + 7), fill=_hex(color), outline=_hex(NEUTRAL_DARK), width=2)
        nature_font = _font(15, bold=family == "Nature Portfolio")
        row1 = plot.loc[plot["venue_family"].eq(family) & plot["top_k"].eq("top_1pct")]
        row5 = plot.loc[plot["venue_family"].eq(family) & plot["top_k"].eq("top_5pct")]
        if not row1.empty and not row5.empty:
            label = f"1% {float(row1.iloc[0]['enrichment_ratio']):.2f}x / 5% {float(row5.iloc[0]['enrichment_ratio']):.2f}x"
            draw.text((right + 12, y_center - 12), label, font=nature_font, fill=_hex(INK))
    draw.rectangle((left, bottom + 33, left + 14, bottom + 47), fill=_hex(NEUTRAL_DARK))
    draw.text((left + 22, bottom + 29), "top 1%", font=_font(15), fill=_hex(MUTED))
    draw.ellipse((left + 105, bottom + 34, left + 119, bottom + 48), fill=_hex(NEUTRAL_DARK))
    draw.text((left + 128, bottom + 29), "top 5%", font=_font(15), fill=_hex(MUTED))
    draw.text((left + 230, bottom + 29), "Future graph-impact enrichment ratio", font=_font(18), fill=_hex(INK))
    image.save(path, dpi=(180, 180))


def _interp_color(value: float, vmin: float, vmax: float) -> Tuple[int, int, int]:
    if not np.isfinite(value):
        return _hex("#F4F5F7")
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin if vmax != vmin else 1.0)))
    low = np.array(_hex("#EAF1FE"))
    mid = np.array(_hex("#FFFFFF"))
    high = np.array(_hex(NATURE_RED))
    if t < 0.5:
        mix = low * (1 - 2 * t) + mid * (2 * t)
    else:
        mix = mid * (1 - 2 * (t - 0.5)) + high * (2 * (t - 0.5))
    return tuple(int(x) for x in mix)


def draw_mechanism_heatmap(mechanism: pd.DataFrame, path: Path) -> None:
    image, draw = _new_panel(
        "d  Mechanism signature heatmap",
        "Venue-family mean of controlled mechanism z-scores; red cells indicate stronger publication-day graph perturbation signatures.",
    )
    nature = mechanism.loc[mechanism["venue_family"].eq("Nature Portfolio")]
    rest = mechanism.loc[~mechanism["venue_family"].eq("Nature Portfolio")].head(8)
    plot = pd.concat([nature, rest], ignore_index=True).head(9).copy()
    left, top = 355, 190
    cell_w, cell_h = 122, 58
    values = plot[[col for col in MECHANISM_COLUMNS if col in plot.columns]].to_numpy(dtype=float).ravel()
    finite = values[np.isfinite(values)]
    lim = max(0.35, float(np.nanquantile(np.abs(finite), 0.90)) if finite.size else 1.0)
    vmin, vmax = -lim, lim
    for j, col in enumerate(MECHANISM_COLUMNS):
        if col not in plot.columns:
            continue
        x = left + j * cell_w
        _draw_wrapped(draw, (x, top - 72), MECHANISM_LABELS[col], _font(14, bold=True), INK, cell_w - 4, line_spacing=1)
    for i, (_, row) in enumerate(plot.iterrows()):
        y = top + i * cell_h
        draw.text((54, y + 17), str(row["venue_family"])[:31], font=_font(17, bold=row["venue_family"] == "Nature Portfolio"), fill=_hex(INK))
        for j, col in enumerate(MECHANISM_COLUMNS):
            if col not in plot.columns:
                continue
            x = left + j * cell_w
            color = _interp_color(float(row[col]), vmin, vmax)
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), fill=color, outline=_hex(PANEL))
            draw.text((x + 32, y + 17), f"{float(row[col]):+.2f}", font=_font(14), fill=_hex(INK))
    draw.text((left, top + len(plot) * cell_h + 26), f"Scale clipped at +/-{lim:.2f} controlled z", font=_font(16), fill=_hex(MUTED))
    image.save(path, dpi=(180, 180))


def draw_prepost(prepost: pd.DataFrame, path: Path) -> None:
    image, draw = _new_panel(
        "e  Publication-day signal vs future impact",
        "Binned paper-level relationship after field-year, article type and reference-count adjustment.",
    )
    left, top, right, bottom = 160, 180, 1250, 790
    draw.rectangle((left, top, right, bottom), fill=_hex("#FFFFFF"), outline=_hex(AXIS))
    for i in range(5):
        y = top + i * (bottom - top) / 4
        draw.line((left, y, right, y), fill=_hex(GRID), width=1)
        x = left + i * (right - left) / 4
        draw.line((x, top, x, bottom), fill=_hex(GRID), width=1)
    x_min, x_max = float(prepost["mean_publication_day_signal"].min()), float(prepost["mean_publication_day_signal"].max())
    y_min, y_max = float(prepost["mean_future_impact"].min()), float(prepost["mean_future_impact"].max())
    pad_x = max(0.05, (x_max - x_min) * 0.1)
    pad_y = max(0.05, (y_max - y_min) * 0.15)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y
    for status, part in prepost.sort_values("signal_bin").groupby("nature_status"):
        color = NATURE_RED if status == "Nature Portfolio" else BLUE_DARK
        points = []
        for _, row in part.iterrows():
            x = _scale(float(row["mean_publication_day_signal"]), x_min, x_max, left, right)
            y = _scale(float(row["mean_future_impact"]), y_min, y_max, bottom, top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=_hex(color), width=4)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=_hex(color), outline=_hex(PANEL), width=2)
        if points:
            draw.text((points[-1][0] + 12, points[-1][1] - 10), status, font=_font(17, bold=status == "Nature Portfolio"), fill=_hex(color))
    draw.text((left, bottom + 40), "Publication-day graph signal (controlled z)", font=_font(18), fill=_hex(INK))
    draw.text((42, top + 25), "Future graph impact", font=_font(18), fill=_hex(INK))
    image.save(path, dpi=(180, 180))


def draw_audit_summary(
    audit: pd.DataFrame,
    portfolio: pd.DataFrame,
    path: Path,
    headline_supported: bool,
    strict_claim: bool,
) -> None:
    image, draw = _new_panel(
        "f  Control and audit summary",
        "Claim status, controls and data gaps for the venue-family contribution analysis.",
    )
    y = 170
    if strict_claim:
        headline = "Strict headline supported"
        color = NATURE_RED
    elif headline_supported:
        headline = "Nature ranks #1 by aggregate VCI; interval caveat remains"
        color = NATURE_RED
    else:
        headline = "Pipeline-ready; Nature headline needs more support"
        color = ORANGE
    draw.rounded_rectangle((54, y, 1288, y + 95), radius=16, fill=_hex("#FFF7F2"), outline=_hex(color), width=3)
    draw.text((82, y + 28), headline, font=_font(28, bold=True), fill=_hex(color))
    y += 130
    top_rows = portfolio.sort_values("vci", ascending=False).head(3)
    draw.text((54, y), "Top controlled VCI families", font=_font(22, bold=True), fill=_hex(INK))
    y += 40
    for _, row in top_rows.iterrows():
        draw.ellipse((74, y + 5, 94, y + 25), fill=_hex(_family_color(str(row["venue_family"]))))
        draw.text((110, y), f"{int(row['vci_rank'])}. {row['venue_family']}: {float(row['vci']):+.3f}  (n={int(row['n_papers'])})", font=_font(20), fill=_hex(INK))
        y += 36
    y += 28
    draw.text((54, y), "Controls and gaps", font=_font(22, bold=True), fill=_hex(INK))
    y += 38
    gap_rows = audit.loc[audit["status"].eq("gap")]
    other_rows = audit.loc[~audit["status"].eq("gap")]
    display_rows = pd.concat([gap_rows, other_rows], ignore_index=True)
    for _, row in display_rows.iterrows():
        status = str(row["status"])
        mark_color = OLIVE if status == "pass" or status == "measured" else ORANGE
        draw.rectangle((70, y + 6, 88, y + 24), fill=_hex(mark_color), outline=_hex(NEUTRAL_DARK))
        y_end = _draw_wrapped(
            draw,
            (105, y),
            f"{row['audit_item']}: {row['value']} - {row['note']}",
            _font(15),
            INK,
            1120,
            line_spacing=2,
        )
        y = y_end + 8
        if y > 815:
            break
    image.save(path, dpi=(180, 180))


def compose_full_figure(panel_paths: Sequence[Path], path: Path, headline_supported: bool, strict_claim: bool) -> None:
    panel_images = [Image.open(p).convert("RGB") for p in panel_paths]
    w, h = panel_images[0].size
    margin = 34
    header_h = 130
    canvas = Image.new("RGB", (3 * w + 4 * margin, 2 * h + 3 * margin + header_h), _hex(SURFACE))
    draw = ImageDraw.Draw(canvas)
    title = "Fig. 7 | Nature Portfolio has the strongest field-year normalized venue contribution in our corpus"
    if not headline_supported:
        title = "Fig. 7 pipeline-ready | Venue-family contribution; Nature headline requires stricter support"
    draw.text((margin, 36), title, font=_font(38, bold=True), fill=_hex(INK))
    if strict_claim:
        caveat = "bootstrap intervals separate Nature from the runner-up"
    elif headline_supported:
        caveat = "point estimate supported; interval separation and per-paper-intensity caveats are audited"
    else:
        caveat = "pipeline-ready claim audit"
    subtitle = (
        "Main layer: venue family. Nature Portfolio is highlighted in deep red; "
        "VCI is portfolio-level aggregate future graph-impact contribution; "
        f"{caveat}."
    )
    _draw_wrapped(draw, (margin, 84), subtitle, _font(19), MUTED, canvas.width - 2 * margin, line_spacing=4)
    positions = [
        (margin, header_h + margin),
        (2 * margin + w, header_h + margin),
        (3 * margin + 2 * w, header_h + margin),
        (margin, header_h + 2 * margin + h),
        (2 * margin + w, header_h + 2 * margin + h),
        (3 * margin + 2 * w, header_h + 2 * margin + h),
    ]
    for img, pos in zip(panel_images, positions):
        canvas.paste(img, pos)
    canvas.save(path, dpi=(180, 180))
    for img in panel_images:
        img.close()


def write_methods_and_gaps(
    out_dir: Path,
    audit: pd.DataFrame,
    portfolio: pd.DataFrame,
    metadata_source: str,
    headline_supported: bool,
    strict_claim: bool,
) -> List[str]:
    gaps = []
    for _, row in audit.loc[audit["status"].eq("gap")].iterrows():
        gaps.append(f"- `{row['audit_item']}`: {row['note']} Current value: {row['value']}.")
    if not gaps:
        gaps.append("- No blocking gaps were detected by the current quality gates.")

    methods = f"""# Fig.7 Methods

Generated at {dt.datetime.now(dt.timezone.utc).isoformat()}.

Analysis corpus: existing Fig.3 eligible paper-level table from `{DEFAULT_FIG3_RUN_DIR}`. This bounded corpus is used because it contains publication-day graph perturbation signals, future graph impact metrics, field, year and reference-count controls.

Venue metadata: {metadata_source}. OpenAlex source fields are flattened into `fig7_venue_family_mapping_audit.csv`; family mapping is rule-based and auditable.

Main metric: `VCI = sum(residual(field-year z(RGPM) ~ article type + log1p(reference count) + log1p(team size) + open-access flag))` by venue family. This is an aggregate portfolio-level contribution index, not a per-paper intensity score.

Publication-day mechanism scores use the same field-year/control basis on the Fig.3 publication-day graph indicators. Top-K enrichment is observed family share of the top 1% and top 5% papers by controlled future graph-impact score divided by the corpus-wide expected share.

Headline gate: Nature Portfolio must rank first by aggregate VCI for point-estimate support. Strict support additionally requires its 95% bootstrap interval to clear the runner-up interval. Current point-estimate support: `{headline_supported}`. Current strict gate: `{strict_claim}`.
"""
    (out_dir / "fig7_methods.md").write_text(methods, encoding="utf-8")

    gap_text = "# Fig.7 Gap List\n\n"
    if strict_claim:
        gap_text += "The current run passes the strict headline gate.\n"
    elif headline_supported:
        gap_text += "The current run supports the headline by aggregate VCI point estimate, but does not strictly separate Nature's bootstrap interval from the runner-up.\n"
    else:
        gap_text += "The current run is pipeline-ready but does not strictly prove the headline under the configured quality gate.\n"
    gap_text += "\n".join(gaps) + "\n\n"
    if not portfolio.empty:
        gap_text += "Top VCI ranking in current run:\n\n"
        for _, row in portfolio.sort_values("vci", ascending=False).head(8).iterrows():
            gap_text += f"- {int(row['vci_rank'])}. {row['venue_family']}: VCI {float(row['vci']):+.3f}, n={int(row['n_papers'])}\n"
    (out_dir / "fig7_gap_list.md").write_text(gap_text, encoding="utf-8")
    return gaps


def build_fig7(
    *,
    fig3_run_dir: Path,
    works_table: Path,
    out_dir: Path,
    max_fetch: Optional[int] = None,
    min_family_n: int = 15,
) -> Mapping[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    score_table = fig3_run_dir / "fig3_score_table.csv"
    indicator_table = fig3_run_dir / "fig3_publication_day_indicators.csv"
    future_table = fig3_run_dir / "fig3_future_graph_deltas.csv"
    paper_ids = pd.read_csv(score_table, usecols=["paper_id"])["paper_id"].map(extract_short_work_id).tolist()
    cache_path = out_dir / "openalex_work_metadata.jsonl"
    metadata = fetch_openalex_metadata(paper_ids, cache_path, max_fetch=max_fetch)
    metadata_source = f"OpenAlex API/cache `{cache_path}` with {len(metadata)} cached records for {len(set(paper_ids))} requested works"

    df, meta = prepare_analysis_frame(score_table, indicator_table, future_table, works_table, metadata, min_family_n=min_family_n)
    portfolio, rankings, enrichment, mechanism, prepost = build_portfolio_tables(df)
    mapping_audit = build_venue_family_mapping_audit(df)
    journal_supplement = build_journal_supplement(df)
    sensitivity = build_metric_sensitivity(df)
    audit = build_confounder_audit(df, portfolio, sensitivity, min_family_n)

    strict_claim = bool(audit.loc[audit["audit_item"].eq("strict_interval_separation"), "status"].eq("pass").any())
    headline_supported = bool(audit.loc[audit["audit_item"].eq("nature_rank"), "status"].eq("pass").any())
    quality_gates = {
        "overall_pass": headline_supported,
        "status_label": (
            "strict headline supported"
            if strict_claim
            else "headline point-estimate supported with strict interval caveat"
            if headline_supported
            else "pipeline-ready with headline gaps"
        ),
        "checks": {row["audit_item"]: int(row["status"] in {"pass", "measured"}) for _, row in audit.iterrows()},
        "headline_claim": "Nature Portfolio has the strongest field-year normalized venue contribution in our corpus",
        "headline_point_estimate_supported": headline_supported,
        "strict_claim_supported": strict_claim,
        "min_family_n": min_family_n,
    }

    df.to_csv(out_dir / "fig7_paper_level_metrics.csv", index=False)
    meta.to_csv(out_dir / "fig7_openalex_metadata_flat.csv", index=False)
    portfolio.to_csv(out_dir / "fig7_venue_portfolio.csv", index=False)
    rankings.to_csv(out_dir / "fig7_vci_rankings.csv", index=False)
    enrichment.to_csv(out_dir / "fig7_topk_enrichment.csv", index=False)
    mechanism.to_csv(out_dir / "fig7_mechanism_signature.csv", index=False)
    prepost.to_csv(out_dir / "fig7_pre_post_publication_signal.csv", index=False)
    audit.to_csv(out_dir / "fig7_confounder_audit.csv", index=False)
    sensitivity.to_csv(out_dir / "fig7_metric_sensitivity.csv", index=False)
    mapping_audit.to_csv(out_dir / "fig7_venue_family_mapping_audit.csv", index=False)
    journal_supplement.to_csv(out_dir / "fig7_journal_supplement.csv", index=False)

    panel_paths = [
        out_dir / "fig7_panel_a.png",
        out_dir / "fig7_panel_b.png",
        out_dir / "fig7_panel_c.png",
        out_dir / "fig7_panel_d.png",
        out_dir / "fig7_panel_e.png",
        out_dir / "fig7_panel_f.png",
    ]
    draw_portfolio_map(portfolio, panel_paths[0])
    draw_ranking(rankings, panel_paths[1])
    draw_enrichment(enrichment, panel_paths[2])
    draw_mechanism_heatmap(mechanism, panel_paths[3])
    draw_prepost(prepost, panel_paths[4])
    draw_audit_summary(audit, portfolio, panel_paths[5], headline_supported, strict_claim)
    compose_full_figure(panel_paths, out_dir / "fig7_full.png", headline_supported, strict_claim)
    gaps = write_methods_and_gaps(out_dir, audit, portfolio, metadata_source, headline_supported, strict_claim)

    write_json(
        out_dir / "fig7_panel_chart_map.json",
        {
            "panel_a": "venue portfolio map: publication-day signal intensity vs aggregate future contribution by venue family",
            "panel_b": "aggregate VCI dot-and-interval ranking by venue family",
            "panel_c": "top 5% future graph-impact enrichment by venue family",
            "panel_d": "controlled mechanism signature heatmap",
            "panel_e": "publication-day signal vs future impact binned relationship",
            "panel_f": "control/audit summary and strict headline status",
        },
    )
    write_run_manifest(
        out_dir,
        figure="fig7",
        argv=sys.argv,
        inputs={
            "fig3_run_dir": str(fig3_run_dir),
            "works_table": str(works_table),
            "openalex_select": OPENALEX_SELECT,
            "max_fetch": max_fetch,
        },
        domains=sorted(df["domain"].dropna().astype(str).unique()),
        quality_gates=quality_gates,
        extra={"gaps": gaps},
    )
    write_figure_quality_report(
        out_dir,
        figure="fig7",
        generated_files=panel_paths + [out_dir / "fig7_full.png"],
        quality_gates=quality_gates,
        visual_checks={
            "palette": "Nature Portfolio deep red; other families low-saturation auxiliary colors",
            "main_layer": "venue_family",
            "supplement_layer": "journal",
        },
        extra={"metadata_source": metadata_source},
    )
    return {
        "out_dir": str(out_dir),
        "headline_point_estimate_supported": headline_supported,
        "strict_claim_supported": strict_claim,
        "n_papers": int(len(df)),
        "metadata_coverage": float(df["metadata_found"].fillna(0).mean()),
        "top_family": str(portfolio.sort_values("vci", ascending=False).iloc[0]["venue_family"]) if not portfolio.empty else "",
        "gaps": gaps,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Fig.7 venue-family contribution panels.")
    parser.add_argument("--fig3-run-dir", type=Path, default=DEFAULT_FIG3_RUN_DIR)
    parser.add_argument("--works-table", type=Path, default=DEFAULT_WORKS_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-fetch", type=int, default=None, help="Optional cap for new OpenAlex fetches; cache is always used.")
    parser.add_argument("--min-family-n", type=int, default=15)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_fig7(
        fig3_run_dir=args.fig3_run_dir,
        works_table=args.works_table,
        out_dir=args.out_dir,
        max_fetch=args.max_fetch,
        min_family_n=args.min_family_n,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
