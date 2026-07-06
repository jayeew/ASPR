"""Run a current generic LLM-only baseline for Fig.10."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig10.build_fig10_ablation import METRICS  # noqa: E402

DEFAULT_FIG4_METRICS = PROJECT_ROOT / "outputs" / "kg_perturbation_fig4_full50" / "fig4_metrics_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig10"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
INPUT_SCOPE = "title_abstract_only_no_graph_no_peer_review"
SCORING_PROTOCOL = "fig10_title_abstract_proxy_rubric"
REQUIRED_REVIEW_SECTIONS = ["novelty", "significance", "prior_art", "evidence_rigor", "limitations", "future_work"]


def clean_text(value: Any, limit: int = 2400) -> str:
    """Normalize model prompt text and cap it to a reproducible length."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit]


def build_generic_prompt(row: pd.Series) -> str:
    """Build a generic LLM-only review prompt from manuscript metadata only."""
    title = clean_text(row.get("title"), limit=400)
    abstract = clean_text(row.get("abstract"), limit=2600)
    schema = {
        "scores_1_5": {
            "novelty": "integer 1-5",
            "significance": "integer 1-5",
            "prior_art": "integer 1-5",
            "evidence_rigor": "integer 1-5",
            "limitations": "integer 1-5",
            "future_work": "integer 1-5",
            "overall": "integer 1-5",
            "unsupported_or_overclaiming_risk": "integer 1-5, where 1 is low risk and 5 is high risk",
        },
        "review_points": {section: ["one short point"] for section in REQUIRED_REVIEW_SECTIONS},
        "recommendation": "accept | minor revision | major revision | reject",
        "confidence": "number 0-1",
    }
    return (
        "You are a general-purpose LLM reviewer baseline, not ASPR and not a domain-tuned ASPR-Qwen checkpoint.\n"
        "Use only the title and abstract below. Do not use citation graphs, retrieval, evidence traces, external search, or peer-review text.\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Title: {title}\n\n"
        f"Abstract: {abstract}\n"
    )


def parse_json_response(text: str) -> Dict[str, Any]:
    """Extract a JSON object from a model response."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("model response JSON is not an object")
    return payload


def score_1_5(value: Any, default: float = 3.0) -> float:
    """Convert a score-like value to a bounded 1-5 scale."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    if pd.isna(score):
        score = default
    return max(1.0, min(5.0, score))


def unit_interval(value: float) -> float:
    """Clip a number into the closed unit interval."""
    return max(0.0, min(1.0, float(value)))


def normalized_score_1_5(value: Any, default: float = 3.0) -> float:
    """Convert a 1-5 score to a 0-1 score."""
    return unit_interval((score_1_5(value, default=default) - 1.0) / 4.0)


def peer_closeness(model_score: Any, peer_score: Any) -> Optional[float]:
    """Score agreement between model and peer 1-5 ratings."""
    if peer_score is None or pd.isna(peer_score):
        return None
    return unit_interval(1.0 - abs(score_1_5(model_score) - score_1_5(peer_score)) / 4.0)


def review_points(response: Mapping[str, Any], section: str) -> List[str]:
    """Return review points for one section."""
    points = response.get("review_points", {})
    if not isinstance(points, Mapping):
        return []
    values = points.get(section, [])
    if isinstance(values, str):
        return [values]
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def evidence_anchor_fraction(response: Mapping[str, Any]) -> float:
    """Estimate how many generated points include explicit evidence language."""
    points: List[str] = []
    for section in REQUIRED_REVIEW_SECTIONS:
        points.extend(review_points(response, section))
    if not points:
        return 0.0
    anchor_words = {"abstract", "data", "assay", "experiment", "method", "analysis", "result", "evidence", "control"}
    anchored = 0
    for point in points:
        tokens = {token.lower() for token in re.findall(r"[A-Za-z]{3,}", point)}
        if tokens & anchor_words:
            anchored += 1
    return anchored / len(points)


def structure_coverage(response: Mapping[str, Any]) -> float:
    """Compute coverage of expected review sections."""
    present = sum(1 for section in REQUIRED_REVIEW_SECTIONS if review_points(response, section))
    return present / len(REQUIRED_REVIEW_SECTIONS)


def score_generic_response(
    paper: pd.Series,
    response: Mapping[str, Any],
    *,
    model_name: str,
    prompt_hash: str,
) -> Dict[str, Any]:
    """Score one generic model response using the Fig.10 metric schema."""
    scores = response.get("scores_1_5", {})
    if not isinstance(scores, Mapping):
        scores = {}
    closeness_values = [
        peer_closeness(scores.get("novelty"), paper.get("peer_novelty")),
        peer_closeness(scores.get("significance"), paper.get("peer_significance")),
        peer_closeness(scores.get("evidence_rigor"), paper.get("peer_rigor")),
        peer_closeness(scores.get("limitations"), paper.get("peer_limitations")),
        peer_closeness(scores.get("future_work"), paper.get("peer_future_work")),
    ]
    valid_closeness = [value for value in closeness_values if value is not None]
    semantic_agreement = sum(valid_closeness) / len(valid_closeness) if valid_closeness else 0.5
    unsupported = normalized_score_1_5(scores.get("unsupported_or_overclaiming_risk"), default=3.0)
    trace = evidence_anchor_fraction(response)
    structure = structure_coverage(response)
    novelty = normalized_score_1_5(scores.get("novelty"), default=3.0) * structure
    prior_art = normalized_score_1_5(scores.get("prior_art"), default=2.0) * (1.0 if review_points(response, "prior_art") else 0.5)
    factuality = unit_interval(1.0 - unsupported)
    readability = unit_interval(0.55 + 0.35 * structure + 0.10 * float(response.get("confidence", 0.5) or 0.5))
    return {
        "case_id": str(paper.get("paper_id")),
        "paper_id": str(paper.get("paper_id")),
        "title": clean_text(paper.get("title"), limit=240),
        "model_name": model_name,
        "prompt_hash": prompt_hash,
        "input_scope": INPUT_SCOPE,
        "scoring_protocol": SCORING_PROTOCOL,
        "run_status": "ok",
        "source": "observed_generic_llm_run",
        "semantic_agreement": semantic_agreement,
        "novelty_coverage": unit_interval(novelty),
        "prior_art_accuracy": unit_interval(prior_art),
        "factuality": factuality,
        "readability": readability,
        "unsupported_claim_rate": unsupported,
        "evidence_trace_completeness": unit_interval(trace),
        "review_structure_coverage": unit_interval(structure),
        "recommendation": str(response.get("recommendation", "")),
        "confidence": unit_interval(float(response.get("confidence", 0.5) or 0.5)),
    }


def prompt_hash(prompt: str) -> str:
    """Return a short stable hash for a prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def call_ollama(prompt: str, *, model_name: str, ollama_url: str, timeout: int) -> str:
    """Call an Ollama chat model and return the assistant content."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 900},
    }
    response = requests.post(ollama_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    message = data.get("message", {})
    if not isinstance(message, Mapping):
        raise ValueError("Ollama response does not contain a message object")
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response content is empty")
    return content


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write rows to JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL rows, skipping blank lines."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_generic_baseline_results(
    out_dir: Path,
    *,
    rows: List[Dict[str, Any]],
    expected_case_count: int,
    model_name: str,
) -> Dict[str, Any]:
    """Write generic baseline CSVs and a manifest with coverage status."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "fig10_generic_llm_baseline_results.csv", index=False)
    metric_keys = [metric for metric, _, _ in METRICS]
    summary = (
        results[metric_keys]
        .agg(["mean", "std", "count"])
        .transpose()
        .reset_index()
        .rename(columns={"index": "metric"})
        if len(results)
        else pd.DataFrame(columns=["metric", "mean", "std", "count"])
    )
    summary.insert(1, "model_name", model_name)
    summary.to_csv(out_dir / "fig10_generic_llm_baseline_summary.csv", index=False)
    case_count = int(results["case_id"].nunique()) if "case_id" in results.columns else 0
    ok_count = int(results.get("run_status", pd.Series(dtype=str)).eq("ok").sum())
    complete = bool(case_count >= expected_case_count and ok_count >= expected_case_count and expected_case_count > 0)
    status = "observed_generic_llm_run" if complete else "observed_generic_llm_run_partial"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "input_scope": INPUT_SCOPE,
        "scoring_protocol": SCORING_PROTOCOL,
        "case_count": case_count,
        "expected_case_count": int(expected_case_count),
        "ok_count": ok_count,
        "status": status,
        "results_csv": str(out_dir / "fig10_generic_llm_baseline_results.csv"),
        "summary_csv": str(out_dir / "fig10_generic_llm_baseline_summary.csv"),
    }
    (out_dir / "fig10_generic_llm_baseline_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_generic_baseline(
    *,
    fig4_metrics: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_cases: Optional[int] = None,
    timeout: int = 90,
    resume: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run the generic LLM baseline over Fig.4 cases."""
    fig4 = pd.read_csv(fig4_metrics)
    selected = fig4.head(max_cases) if max_cases else fig4
    selected_ids = [str(paper_id) for paper_id in selected.get("paper_id", pd.Series(dtype=str)).tolist()]
    selected_id_set = set(selected_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_by_paper: Dict[str, Dict[str, Any]] = {}
    scored_by_case: Dict[str, Dict[str, Any]] = {}
    if resume:
        for row in read_jsonl(out_dir / "fig10_generic_llm_baseline_outputs.jsonl"):
            paper_id = str(row.get("paper_id") or row.get("case_id") or "")
            if paper_id and paper_id in selected_id_set:
                raw_by_paper[paper_id] = row
        results_path = out_dir / "fig10_generic_llm_baseline_results.csv"
        if results_path.exists():
            existing = pd.read_csv(results_path)
            for row in existing.to_dict("records"):
                case_id = str(row.get("case_id") or row.get("paper_id") or "")
                if case_id and case_id in selected_id_set:
                    scored_by_case[case_id] = row
    for _, paper in selected.iterrows():
        paper_id = str(paper.get("paper_id"))
        existing_raw = raw_by_paper.get(paper_id)
        if (
            resume
            and skip_existing
            and existing_raw is not None
            and str(existing_raw.get("run_status")) == "ok"
            and paper_id in scored_by_case
        ):
            continue
        prompt = build_generic_prompt(paper)
        p_hash = prompt_hash(prompt)
        raw: Dict[str, Any] = {
            "paper_id": paper_id,
            "model_name": model_name,
            "prompt_hash": p_hash,
            "input_scope": INPUT_SCOPE,
        }
        try:
            content = call_ollama(prompt, model_name=model_name, ollama_url=ollama_url, timeout=timeout)
            parsed = parse_json_response(content)
            raw.update({"run_status": "ok", "raw_response": content, "parsed_response": parsed})
            scored_by_case[paper_id] = score_generic_response(paper, parsed, model_name=model_name, prompt_hash=p_hash)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            raw.update({"run_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
            scored_by_case.pop(paper_id, None)
        raw_by_paper[paper_id] = raw
    raw_rows = [raw_by_paper[paper_id] for paper_id in selected_ids if paper_id in raw_by_paper]
    scored_rows = [scored_by_case[paper_id] for paper_id in selected_ids if paper_id in scored_by_case]
    write_jsonl(out_dir / "fig10_generic_llm_baseline_outputs.jsonl", raw_rows)
    return write_generic_baseline_results(
        out_dir,
        rows=scored_rows,
        expected_case_count=len(fig4),
        model_name=model_name,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig4-metrics", type=Path, default=DEFAULT_FIG4_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--resume", action="store_true", help="Reuse previous baseline output/result files when present.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases with existing ok output and scored result.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_generic_baseline(
        fig4_metrics=args.fig4_metrics,
        out_dir=args.out_dir,
        model_name=args.model_name,
        ollama_url=args.ollama_url,
        max_cases=args.max_cases,
        timeout=args.timeout,
        resume=args.resume,
        skip_existing=args.skip_existing,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
