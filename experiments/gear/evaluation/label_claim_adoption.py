"""Label claim adoption from real, post-cutoff citation-context evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from gear.config import load_config
from gear.model_client import ModelClientUnavailableError, build_json_model_client

RELATION_WEIGHT = {
    "irrelevant": 0.0,
    "background": 0.0,
    "method_or_result_use": 0.6,
    "extension": 0.8,
    "cross_field_transfer": 1.0,
}

LABEL_PROMPT = """You are a blinded citation-context adjudicator. Match each real
future citation context to at most one focal-paper claim. Label adoption only when
the context says the citing work uses, extends, reproduces, transfers, or directly
depends on that claim. A citation, comparison, or background mention alone is not
adoption. Use only supplied text. Return exactly one judgment per context_id. Never
infer missing context and never use graph scores, citation counts, or paper outcomes."""
MAX_VALIDATION_ATTEMPTS = 3


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextJudgment(_StrictModel):
    context_id: str
    claim_id: str | None = None
    relation: Literal[
        "irrelevant",
        "background",
        "method_or_result_use",
        "extension",
        "cross_field_transfer",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class AdoptionResponse(_StrictModel):
    judgments: list[ContextJudgment]


def label_adoption(
    claims_path: Path,
    contexts_path: Path,
    output_dir: Path,
    config_path: Path,
    *,
    context_papers_path: Path | None = None,
    chunk_size: int = 20,
    workers: int = 1,
) -> dict[str, Any]:
    """Adjudicate contexts independently of HGB and aggregate observable labels."""
    claims = pd.read_parquet(claims_path)
    contexts = pd.DataFrame(_read_jsonl(contexts_path))
    _validate_inputs(claims, contexts)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "claim_adoption_judgments.jsonl"
    completion_checkpoint = output_dir / "claim_adoption_completed_papers.jsonl"
    judgments = _read_jsonl(checkpoint)
    completed_rows = _read_jsonl(completion_checkpoint)
    if not completed_rows and judgments:
        completed_rows = [
            {"paper_id": paper_id, "status": "labeled"}
            for paper_id in sorted({str(row["paper_id"]) for row in judgments})
        ]
    completed = {str(row["paper_id"]) for row in completed_rows}
    failures: list[dict[str, str]] = []
    context_statuses = _context_paper_statuses(context_papers_path, contexts)
    valid_context_papers = {
        paper_id
        for paper_id, status in context_statuses.items()
        if status.startswith("resolved")
    }
    _validate_checkpoint(
        claims,
        contexts,
        judgments,
        completed_rows,
        valid_context_papers,
    )
    shared = sorted(set(claims["paper_id"].astype(str)) & valid_context_papers)
    pending = [paper_id for paper_id in shared if paper_id not in completed]
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _label_one,
                config_path,
                paper_id,
                claims[claims["paper_id"].eq(paper_id)],
                contexts[contexts["paper_id"].eq(paper_id)],
                chunk_size,
            ): paper_id
            for paper_id in pending
        }
        for future in as_completed(futures):
            paper_id = futures[future]
            try:
                paper_rows = future.result()
            except (ModelClientUnavailableError, TypeError, ValueError) as exc:
                failures.append({"paper_id": paper_id, "reason": type(exc).__name__})
                continue
            judgments.extend(paper_rows)
            judgments.sort(
                key=lambda row: (str(row["paper_id"]), str(row["context_id"]))
            )
            _write_jsonl(checkpoint, judgments)
            completed_rows.append(
                {
                    "paper_id": paper_id,
                    "status": "labeled" if paper_rows else "observed_zero_contexts",
                }
            )
            completed_rows.sort(key=lambda row: str(row["paper_id"]))
            _write_jsonl(completion_checkpoint, completed_rows)
    completed = {str(row["paper_id"]) for row in completed_rows}
    labels, papers = _aggregate(
        claims,
        contexts,
        judgments,
        completed,
        context_statuses=context_statuses,
    )
    labels_path = output_dir / "claim_adoption_labels.parquet"
    papers_path = output_dir / "paper_claim_adoption.parquet"
    labels.to_parquet(labels_path, index=False)
    papers.to_parquet(papers_path, index=False)
    summary = {
        "contract": "gear_real_claim_adoption_labels_v1",
        "data_role": "future_outcome_only",
        "papers_resolved": len(shared),
        "papers_with_nonzero_contexts": int(
            contexts[contexts["paper_id"].isin(shared)]["paper_id"].nunique()
        ),
        "papers_with_observed_zero_contexts": sum(
            row["status"] == "observed_zero_contexts" for row in completed_rows
        ),
        "papers_labeled": int(labels["paper_id"].nunique()) if not labels.empty else 0,
        "claim_labels": len(labels),
        "adopted_claims": (
            int(labels["future_adoption"].gt(0).sum()) if not labels.empty else 0
        ),
        "failed_papers": failures,
        "context_observation_status_counts": {
            status: sum(
                paper_id in shared and observed_status == status
                for paper_id, observed_status in context_statuses.items()
            )
            for status in sorted(set(context_statuses.values()))
        },
        "truncated_adoption_interpretation": "observed_lower_bound",
        "labeler_blind_to_hgb": True,
        "workers": workers,
        "claim_labels_sha256": "sha256:"
        + hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "paper_labels_sha256": "sha256:"
        + hashlib.sha256(papers_path.read_bytes()).hexdigest(),
    }
    (output_dir / "claim_adoption_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _label_one(
    config_path: Path,
    paper_id: str,
    paper_claims: pd.DataFrame,
    paper_contexts: pd.DataFrame,
    chunk_size: int,
) -> list[dict[str, Any]]:
    client = build_json_model_client(load_config(config_path))
    return _label_paper(client, paper_id, paper_claims, paper_contexts, chunk_size)


def _label_paper(
    client: Any,
    paper_id: str,
    claims: pd.DataFrame,
    contexts: pd.DataFrame,
    chunk_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    claim_payload = claims[["claim_id", "claim_text"]].to_dict(orient="records")
    for start in range(0, len(contexts), chunk_size):
        chunk = contexts.iloc[start : start + chunk_size]
        context_payload = chunk[["context_id", "context"]].to_dict(orient="records")
        response = _generate_valid_chunk(
            client,
            paper_id=paper_id,
            claim_payload=claim_payload,
            context_payload=context_payload,
            claims=claims,
            contexts=chunk,
        )
        context_index = chunk.set_index("context_id")
        for judgment in response.judgments:
            source = context_index.loc[judgment.context_id]
            output.append(
                {
                    "paper_id": paper_id,
                    **judgment.model_dump(),
                    "citing_paper_id": source["citing_paper_id"],
                    "citing_fields": source["citing_fields"],
                    "data_role": "future_outcome_only",
                }
            )
    return output


def _generate_valid_chunk(
    client: Any,
    *,
    paper_id: str,
    claim_payload: list[dict[str, Any]],
    context_payload: list[dict[str, Any]],
    claims: pd.DataFrame,
    contexts: pd.DataFrame,
) -> AdoptionResponse:
    """Retry only with explicit validation feedback; never repair labels locally."""
    correction: str | None = None
    for attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        payload: dict[str, Any] = {
            "paper_id": paper_id,
            "claims": claim_payload,
            "contexts": context_payload,
        }
        if correction is not None:
            payload["correction_required"] = correction
            payload["validation_attempt"] = attempt
        raw = client.generate_json(
            system=LABEL_PROMPT,
            user=json.dumps(payload, ensure_ascii=False),
            response_schema=AdoptionResponse.model_json_schema(),
        )
        try:
            response = AdoptionResponse.model_validate(raw)
            _validate_judgments(response.judgments, claims, contexts)
        except ValueError as exc:
            correction = (
                f"Previous output failed strict validation: {exc}. Return exactly "
                "one judgment for every supplied context_id. Any adoption relation "
                "must reference one supplied claim_id; otherwise use background or "
                "irrelevant with claim_id null."
            )
            continue
        return response
    raise ValueError(
        f"claim-adoption validation failed after {MAX_VALIDATION_ATTEMPTS} attempts"
    )


def _validate_judgments(
    judgments: list[ContextJudgment], claims: pd.DataFrame, contexts: pd.DataFrame
) -> None:
    expected = set(contexts["context_id"].astype(str))
    observed = [item.context_id for item in judgments]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("context judgments are not complete and one-to-one")
    allowed_claims = set(claims["claim_id"].astype(str))
    for item in judgments:
        if item.claim_id is not None and item.claim_id not in allowed_claims:
            raise ValueError("claim adoption judgment references an unknown claim")
        if item.relation not in {"irrelevant", "background"} and item.claim_id is None:
            raise ValueError("adoption relation requires a claim_id")


def _aggregate(
    claims: pd.DataFrame,
    contexts: pd.DataFrame,
    judgments: list[dict[str, Any]],
    completed_paper_ids: set[str] | None = None,
    context_statuses: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    judged = pd.DataFrame(judgments)
    paper_ids = set(completed_paper_ids or set())
    if not judged.empty:
        paper_ids.update(judged["paper_id"].astype(str))
    if not paper_ids:
        return pd.DataFrame(), pd.DataFrame()
    labels = claims[claims["paper_id"].isin(paper_ids)].copy()
    statuses = context_statuses or {
        paper_id: "resolved_contexts_only" for paper_id in paper_ids
    }
    labels["context_observation_status"] = labels["paper_id"].map(statuses)
    centrality_sum = labels.groupby("paper_id")["claim_centrality"].transform("sum")
    labels["attribution_weight"] = labels["claim_centrality"] / centrality_sum
    adopted = (
        judged[judged["relation"].map(RELATION_WEIGHT).gt(0)].copy()
        if not judged.empty
        else pd.DataFrame()
    )
    metrics = _claim_metrics(adopted)
    labels = labels.merge(metrics, on=["paper_id", "claim_id"], how="left")
    labels["future_adoption"] = labels["future_adoption"].fillna(0.0)
    labels["adopting_context_count"] = (
        labels["adopting_context_count"].fillna(0).astype(int)
    )
    labels["adopting_paper_count"] = (
        labels["adopting_paper_count"].fillna(0).astype(int)
    )
    labels["adoption_evidence_context_ids"] = labels[
        "adoption_evidence_context_ids"
    ].map(lambda value: value if isinstance(value, list) else [])
    papers = _paper_metrics(contexts, adopted, paper_ids)
    papers["context_observation_status"] = papers["paper_id"].map(statuses)
    papers["adoption_is_lower_bound"] = papers["context_observation_status"].eq(
        "resolved_truncated"
    )
    return labels, papers


def _claim_metrics(adopted: pd.DataFrame) -> pd.DataFrame:
    if adopted.empty:
        return pd.DataFrame(
            columns=[
                "paper_id",
                "claim_id",
                "future_adoption",
                "adopting_context_count",
                "adopting_paper_count",
                "adoption_evidence_context_ids",
            ]
        )
    adopted["relation_weight"] = adopted["relation"].map(RELATION_WEIGHT)
    paper_level = (
        adopted.groupby(["paper_id", "claim_id", "citing_paper_id"], observed=True)[
            "relation_weight"
        ]
        .max()
        .reset_index()
    )
    metrics = (
        paper_level.groupby(["paper_id", "claim_id"], observed=True)
        .agg(
            future_adoption=("relation_weight", "sum"),
            adopting_paper_count=("citing_paper_id", "nunique"),
        )
        .reset_index()
    )
    contexts = (
        adopted.groupby(["paper_id", "claim_id"], observed=True)
        .agg(
            adopting_context_count=("context_id", "size"),
            adoption_evidence_context_ids=("context_id", list),
        )
        .reset_index()
    )
    return metrics.merge(
        contexts,
        on=["paper_id", "claim_id"],
        how="inner",
        validate="one_to_one",
    )


def _paper_metrics(
    contexts: pd.DataFrame, adopted: pd.DataFrame, paper_ids: set[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for paper_id in sorted(paper_ids):
        all_fields = _flatten_fields(contexts[contexts["paper_id"].eq(paper_id)])
        adopted_fields = (
            set()
            if adopted.empty
            else _flatten_fields(adopted[adopted["paper_id"].eq(paper_id)])
        )
        breadth = len(adopted_fields) / len(all_fields) if all_fields else 0.0
        rows.append(
            {
                "paper_id": paper_id,
                "claim_adoption_breadth": breadth,
                "adopted_field_count": len(adopted_fields),
                "observed_citing_field_count": len(all_fields),
            }
        )
    return pd.DataFrame(rows)


def _flatten_fields(frame: pd.DataFrame) -> set[str]:
    output: set[str] = set()
    if "citing_fields" not in frame:
        return output
    for values in frame["citing_fields"]:
        output.update(str(value) for value in (values or []))
    return output


def _validate_inputs(claims: pd.DataFrame, contexts: pd.DataFrame) -> None:
    claim_required = {"paper_id", "claim_id", "claim_text", "claim_centrality"}
    context_required = {
        "paper_id",
        "context_id",
        "context",
        "citing_paper_id",
        "citing_fields",
    }
    missing = (claim_required - set(claims)) | (context_required - set(contexts))
    if missing:
        raise ValueError(f"claim adoption inputs missing: {sorted(missing)}")


def _validate_checkpoint(
    claims: pd.DataFrame,
    contexts: pd.DataFrame,
    judgments: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    valid_context_papers: set[str],
) -> None:
    """Fail closed rather than attach stale judgments to a changed inventory."""
    completed_ids = [str(row["paper_id"]) for row in completed_rows]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("claim-adoption completion checkpoint duplicates a paper")
    claim_keys = set(
        zip(claims["paper_id"].astype(str), claims["claim_id"].astype(str))
    )
    context_keys = set(
        zip(contexts["paper_id"].astype(str), contexts["context_id"].astype(str))
    )
    observed_contexts: dict[str, list[str]] = {}
    for row in judgments:
        paper_id = str(row["paper_id"])
        context_id = str(row["context_id"])
        if paper_id not in completed_ids or (paper_id, context_id) not in context_keys:
            raise ValueError("claim-adoption checkpoint references an unknown context")
        claim_id = row.get("claim_id")
        if claim_id is not None and (paper_id, str(claim_id)) not in claim_keys:
            raise ValueError("claim-adoption checkpoint references an unknown claim")
        observed_contexts.setdefault(paper_id, []).append(context_id)
    claim_papers = set(claims["paper_id"].astype(str))
    for paper_id in completed_ids:
        if paper_id not in valid_context_papers or paper_id not in claim_papers:
            raise ValueError(
                "claim-adoption completion checkpoint is outside the cohort"
            )
        expected = {key[1] for key in context_keys if key[0] == paper_id}
        observed = observed_contexts.get(paper_id, [])
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("claim-adoption checkpoint is not complete and one-to-one")


def _context_paper_statuses(
    path: Path | None, contexts: pd.DataFrame
) -> dict[str, str]:
    if path is None:
        return {
            paper_id: "resolved_contexts_only"
            for paper_id in contexts["paper_id"].astype(str).unique()
        }
    return {
        str(row["paper_id"]): str(row.get("fetch_status", "unknown"))
        for row in _read_jsonl(path)
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--context-papers", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = label_adoption(
        args.claims,
        args.contexts,
        args.output_dir,
        args.config,
        context_papers_path=args.context_papers,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["label_adoption"]
