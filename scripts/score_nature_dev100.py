#!/usr/bin/env python3
"""Materialize the legacy Nature dev100 ASPR development-score release.

The source list is pinned to the final commit that retained the dev100
reconstruction packages.  Scores use the same reconstructed Full-text-16
development replay as the existing ``nature_mini3`` release.  A missing or
unscoreable paper aborts the run; partial score releases are never published.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import sleep
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.calibration import FEATURE_NAMES, _load_official_joblib, sha256_file
from gear.contracts import PaperMetadata, ReviewRequest
from gear.graph_prior import build_graph_search_hints
from gear.nature_multihorizon.runtime_replay_v3 import build_runtime_context_for_year
from gear.nature_multihorizon.t0_runtime_v3 import (
    ContextSnapshot,
    coerce_fulltext16_storage_schema,
    materialize_fulltext16,
)
from gear.paper_compiler import PaperCompiler
from gear.t0_enrichment import OpenAlexT0Enricher

DEV100_SOURCE_REVISION = "6d348b52ac4b31fac133084482733b1e640dc0f6"
DEV100_ROOT = "data/gear_review_reconstruction/nature_dev100"
MODEL_DIR = PROJECT_ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v3/experiments/"
    "oof_feature_set_comparison_v3/outputs/legacy_baseline_hgb_uncapped_v2_20260820"
)
MATRIX_PATH = PROJECT_ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v3/experiments/"
    "oof_feature_set_comparison_v3/outputs/uncapped_v2/indicator_matrix_16.parquet"
)
METADATA_PATH = PROJECT_ROOT / (
    "data/knowledge_corpus/nature_multihorizon_v6_1_uncapped_v2/"
    "papers_primary_articles.parquet"
)
RUNTIME_DIR = PROJECT_ROOT / "data/calibration/runtime_replay/fulltext16_v3"


def _dev100_dois() -> list[str]:
    command = [
        "git",
        "ls-tree",
        "-r",
        "--name-only",
        DEV100_SOURCE_REVISION,
        DEV100_ROOT,
    ]
    result = subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    suffix = "/reconstruction/package.json"
    dois = sorted(
        {
            Path(path).parts[3].replace("_", "/")
            for path in result.stdout.splitlines()
            if path.endswith(suffix)
        }
    )
    if len(dois) != 100:
        raise ValueError(
            f"Pinned dev100 source has {len(dois)} package IDs, expected 100"
        )
    return dois


def _openalex_work(doi: str) -> dict[str, Any]:
    last_error: requests.RequestException | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                f"https://api.openalex.org/works/https://doi.org/{doi}",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=60,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 2:
                raise
            sleep(1 + attempt)
    else:
        raise last_error or RuntimeError("OpenAlex request did not execute")
    payload = response.json()
    if (
        not isinstance(payload, dict)
        or not payload.get("id")
        or not payload.get("publication_year")
    ):
        raise ValueError(f"OpenAlex lacks exact identity/year for {doi}")
    return payload


def _paper_path(doi: str) -> Path:
    filename = doi.replace("10.1038/", "").replace("/", "_")
    path = PROJECT_ROOT / "data/nature_markdown/paper" / f"{filename}.md"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _score_one(
    doi: str,
    compiler: PaperCompiler,
    enricher: OpenAlexT0Enricher,
    context: ContextSnapshot,
    model_bundle: dict[str, Any],
    reference: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    work = _openalex_work(doi)
    publication_year = int(work["publication_year"])
    metadata = PaperMetadata(
        title=str(work.get("display_name") or ""),
        doi=doi,
        openalex_id=str(work["id"]),
        publication_date=date(publication_year, 1, 1),
        venue="Nature Communications",
    )
    paper = compiler.compile(
        ReviewRequest(paper_path=_paper_path(doi), metadata=metadata)
    )
    target = enricher.build_target(paper, evidence_date=metadata.publication_date)
    seed_work_ids, search_terms = build_graph_search_hints(target, context)
    values = materialize_fulltext16(target, context)
    feature_frame = coerce_fulltext16_storage_schema(
        pd.DataFrame([{key: values.get(key) for key in FEATURE_NAMES}])
    )
    uptake_raw, conditional_raw = model_bundle["model"].predict_raw(feature_frame)
    uptake = float(model_bundle["uptake_calibrator"].predict(uptake_raw)[0])
    conditional = float(
        model_bundle["conditional_calibrator"].predict(conditional_raw)[0]
    )
    raw = uptake * conditional
    score = (
        100.0 * float(np.searchsorted(reference, raw, side="right")) / len(reference)
    )
    paper_id = f"sha256:{hashlib.sha256(doi.encode('utf-8')).hexdigest()}"
    flags = [
        "post_frozen_cohort_year",
        "registered_model_asset_missing",
        "full_cohort_score_replay_passed",
    ]
    record = {
        "paper_id": paper_id,
        "source_paper_id": doi.replace("10.1038/", ""),
        "openalex_id": str(work["id"]),
        "doi": doi,
        "publication_year": publication_year,
        "horizon": 5,
        "aspr_score": score,
        "score_performance_percentile": score / 100.0,
        "raw_prediction_score": raw,
        "p_uptake": uptake,
        "conditional_diffusion": conditional,
        "feature_coverage": 1.0,
        "status": "development_replay",
        "model_version": "pgc-v3-d5-fulltext16-prediction-replay",
        "quality_flags": ";".join(flags),
        "claim_scope": "publication-time Full-text-16; historical context through 2022; development replay, non-confirmatory",
    }
    features = {
        "paper_id": paper_id,
        "source_paper_id": record["source_paper_id"],
        "openalex_id": record["openalex_id"],
        **{name: values.get(name) for name in FEATURE_NAMES},
    }
    source = {
        "paper_id": paper_id,
        "source_paper_id": record["source_paper_id"],
        "doi": doi,
        "openalex_id": record["openalex_id"],
        "paper_sha256": paper.paper_sha256,
        "reference_count": len(target.references),
        "metadata_observed": target.metadata_observed,
    }
    return record, {
        "features": features,
        "source": source,
        "seed_work_ids": seed_work_ids,
        "search_terms": search_terms,
    }


def _reference_scores() -> np.ndarray:
    scores = pd.read_parquet(MODEL_DIR / "official_aspr_scores.parquet")
    metadata = pd.read_parquet(METADATA_PATH, columns=["paper_id", "publication_year"])
    merged = scores.merge(metadata, on="paper_id", how="inner", validate="one_to_one")
    values = (
        merged.loc[merged["publication_year"].le(2020), "raw_prediction_score"]
        .dropna()
        .to_numpy(dtype=float)
    )
    if not len(values):
        raise ValueError("Mature D5 percentile reference is empty")
    return np.sort(values)


def _emit_graph_results_from_existing(output: Path, workers: int) -> None:
    """Add V4 search hints to an existing score release without rescoring."""
    score_path = output / "paper_scores.parquet"
    context_path = output / "rebuilt_context_snapshot.joblib"
    if not score_path.is_file() or not context_path.is_file():
        raise FileNotFoundError("existing score table or rebuilt context is missing")
    context = joblib.load(context_path)
    if not isinstance(context, ContextSnapshot) or context.source_max_year != 2022:
        raise ValueError("Unexpected cached runtime context")
    records = pd.read_parquet(score_path).to_dict("records")

    def build(record: dict[str, Any]) -> dict[str, Any]:
        compiler = PaperCompiler()
        enricher = OpenAlexT0Enricher()
        metadata = PaperMetadata(
            doi=str(record["doi"]),
            openalex_id=str(record["openalex_id"]),
            publication_date=date(int(record["publication_year"]), 1, 1),
            venue="Nature Communications",
        )
        paper = compiler.compile(
            ReviewRequest(paper_path=_paper_path(str(record["doi"])), metadata=metadata)
        )
        target = enricher.build_target(paper, evidence_date=metadata.publication_date)
        seed_work_ids, search_terms = build_graph_search_hints(target, context)
        return {
            "contract": "aspr_graph_result_v4",
            "paper_id": record["openalex_id"],
            "score_0_100": record["aspr_score"],
            "p_uptake": record["p_uptake"],
            "conditional_diffusion": record["conditional_diffusion"],
            "feature_coverage": record["feature_coverage"],
            "seed_work_ids": seed_work_ids,
            "search_terms": search_terms,
        }

    target_path = output / "graph_results.jsonl"
    with (
        target_path.open("w", encoding="utf-8") as handle,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for index, (record, result) in enumerate(
            zip(records, pool.map(build, records), strict=True), start=1
        ):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(records)}] {record['doi']} hints", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="nature_dev100")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/gear/aspr_scoring/nature_dev100",
    )
    parser.add_argument("--graph-results-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.graph_results_only:
        _emit_graph_results_from_existing(output, max(1, args.workers))
        print(output / "graph_results.jsonl")
        return 0
    if output.exists() and (output / "paper_scores.parquet").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (RUNTIME_DIR / "runtime_replay_manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(MATRIX_PATH) != manifest["official_matrix_sha256"]:
        raise ValueError("Full-text-16 matrix does not match the replay gate")
    context_path = RUNTIME_DIR / "context_snapshot.joblib"
    context_rebuilt = False
    cached_context = output / "rebuilt_context_snapshot.joblib"
    cached_context_manifest = output / "rebuilt_context_manifest.json"
    if (
        context_path.is_file()
        and sha256_file(context_path) == manifest["context_snapshot_sha256"]
    ):
        context = joblib.load(context_path)
    elif cached_context.is_file() and cached_context_manifest.is_file():
        context_manifest = json.loads(
            cached_context_manifest.read_text(encoding="utf-8")
        )
        if (
            context_manifest["source_hashes"]
            != manifest["input_manifest"]["source_hashes"]
        ):
            raise ValueError(
                "Cached runtime context source hashes differ from replay gate"
            )
        context = joblib.load(cached_context)
    else:
        print("Rebuilding 2022 graph context from pinned full cohort...", flush=True)
        context, context_manifest = build_runtime_context_for_year(
            project_root=PROJECT_ROOT,
            official_matrix_path=MATRIX_PATH,
            target_year=2023,
        )
        if (
            context_manifest["source_hashes"]
            != manifest["input_manifest"]["source_hashes"]
        ):
            raise ValueError(
                "Rebuilt runtime context source hashes differ from replay gate"
            )
        joblib.dump(context, cached_context, compress=3)
        cached_context_manifest.write_text(
            json.dumps(context_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        context_rebuilt = True
    if not isinstance(context, ContextSnapshot) or context.source_max_year != 2022:
        raise ValueError("Unexpected runtime context")
    model_path = MODEL_DIR / "official_hgb_model.joblib"
    score_path = MODEL_DIR / "official_aspr_scores.parquet"
    model_bundle = _load_official_joblib(model_path)
    reference = _reference_scores()
    compiler = PaperCompiler()
    enricher = OpenAlexT0Enricher()
    records: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    graph_hints: dict[str, tuple[list[str], list[str]]] = {}
    for index, doi in enumerate(_dev100_dois(), start=1):
        record, extras = _score_one(
            doi, compiler, enricher, context, model_bundle, reference
        )
        records.append(record)
        features.append(extras["features"])
        sources.append(extras["source"])
        graph_hints[record["openalex_id"]] = (
            extras["seed_work_ids"],
            extras["search_terms"],
        )
        print(f"[{index}/100] {doi} score={record['aspr_score']:.6f}", flush=True)
    pd.DataFrame(records).to_parquet(output / "paper_scores.parquet", index=False)
    pd.DataFrame(features).to_parquet(output / "score_features.parquet", index=False)
    with (output / "graph_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            seed_work_ids, search_terms = graph_hints[record["openalex_id"]]
            graph_result = {
                "contract": "aspr_graph_result_v4",
                "paper_id": record["openalex_id"],
                "score_0_100": record["aspr_score"],
                "p_uptake": record["p_uptake"],
                "conditional_diffusion": record["conditional_diffusion"],
                "feature_coverage": record["feature_coverage"],
                "seed_work_ids": seed_work_ids,
                "search_terms": search_terms,
            }
            handle.write(json.dumps(graph_result, ensure_ascii=False) + "\n")
    (output / "source_manifest.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    release = {
        "contract": "aspr_scoring_release_v1",
        "release": args.release,
        "status": "development_replay",
        "confirmatory": False,
        "record_count": len(records),
        "horizon": 5,
        "source_max_year": context.source_max_year,
        "score_definition": "100 times empirical CDF of raw D5 HGB prediction in mature D5 cohort",
        "model_sha256": sha256_file(model_path),
        "full_cohort_score_sha256": sha256_file(score_path),
        "runtime_replay_manifest_sha256": sha256_file(
            RUNTIME_DIR / "runtime_replay_manifest.json"
        ),
        "quality_flags": [
            "registered_model_asset_missing",
            "reconstructed_model_serialization_hash_differs",
            "full_cohort_score_replay_passed",
            "post_frozen_cohort_year",
            *(
                ["runtime_context_rebuilt_from_pinned_source"]
                if context_rebuilt
                else []
            ),
        ],
        "dev100_source_revision": DEV100_SOURCE_REVISION,
    }
    (output / "release_manifest.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
