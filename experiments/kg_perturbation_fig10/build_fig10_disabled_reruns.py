"""Run observed disabled-module reruns for Fig.10."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig10.build_fig10_ablation import (  # noqa: E402
    DISABLED_MODULE_VARIANTS,
    FIG10_COMPLETED_DISABLED_RERUNS_FILE,
    METRICS,
    safe_token,
)
from experiments.kg_perturbation_fig10.build_fig10_generic_baseline import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    REQUIRED_REVIEW_SECTIONS,
    call_ollama,
    clean_text,
    parse_json_response,
    prompt_hash,
    review_points,
    score_1_5,
    unit_interval,
    write_jsonl,
)


DEFAULT_FIG4_METRICS = PROJECT_ROOT / "outputs" / "kg_perturbation_fig4_full50" / "fig4_metrics_summary.csv"
DEFAULT_FIG4_AGENT_OUTPUTS = PROJECT_ROOT / "outputs" / "kg_perturbation_fig4_full50" / "fig4_agent_outputs.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig10"
SCORING_PROTOCOL = "fig10_observed_disabled_module_proxy_rubric"
INPUT_SCOPE = "fig4_frozen_sample_disabled_module_rerun"
DISABLED_VARIANTS = [
    "no graph agent",
    "no ASPR-Qwen",
    "no prior-art retrieval",
    "no evidence trace",
    "no fusion",
    "no verifier",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL rows from a path."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def normalize_space(value: Any, limit: int = 2400) -> str:
    """Normalize text for prompts and artifacts."""
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def format_retrieved_papers(retrieved_papers: Sequence[Mapping[str, Any]], limit: int = 6) -> str:
    """Render retrieved prior-art papers into a compact prompt block."""
    lines: List[str] = []
    for idx, paper in enumerate(retrieved_papers[:limit], start=1):
        title = normalize_space(paper.get("title"), limit=180)
        abstract = normalize_space(paper.get("abstract"), limit=520)
        year = normalize_space(paper.get("year") or paper.get("publicationDate"), limit=40)
        doi = normalize_space(paper.get("doi"), limit=80)
        lines.append(f"[{idx}] {title} ({year}; {doi})\nAbstract: {abstract}")
    return "\n\n".join(lines)


def disabled_module_instruction(variant: str) -> str:
    """Return a reviewer-facing instruction for one disabled ASPR module."""
    instructions = {
        "no graph agent": (
            "Disable the graph-perturbation lane: do not use graph priors, seven-indicator scores, "
            "structural trajectories, or graph-derived novelty mechanisms."
        ),
        "no ASPR-Qwen": (
            "Disable the ASPR-Qwen reviewer lane: write as a generic structured reviewer without "
            "domain-tuned reviewer-style wording or checkpoint-specific review behavior."
        ),
        "no prior-art retrieval": (
            "Disable prior-art retrieval: use only the manuscript title, abstract, and any non-retrieval "
            "graph cue explicitly provided. Do not infer retrieved papers."
        ),
        "no evidence trace": (
            "Disable evidence trace generation: produce review judgements without claim-to-source trace cards."
        ),
        "no fusion": (
            "Disable fusion: do not reconcile graph, retrieval, and reviewer lanes into a fused final stance; "
            "report the single-lane assessment directly."
        ),
        "no verifier": (
            "Disable the self-check verifier: do not run contradiction, overclaiming, or calibration checks."
        ),
    }
    if variant not in instructions:
        raise ValueError(f"unsupported disabled variant: {variant}")
    return instructions[variant]


def build_disabled_variant_prompt(
    paper: pd.Series,
    *,
    variant: str,
    retrieved_papers: Sequence[Mapping[str, Any]],
    graph_evidence: Any,
) -> str:
    """Build one Fig.10 disabled-module rerun prompt."""
    if variant not in DISABLED_MODULE_VARIANTS:
        raise ValueError(f"variant is not a disabled-module condition: {variant}")
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
    blocks = [
        "You are running a Fig.10 ASPR disabled-module rerun for ablation evidence.",
        f"Variant: {variant}",
        disabled_module_instruction(variant),
        "Return only valid JSON matching this schema:",
        json.dumps(schema, ensure_ascii=False),
        f"Title: {clean_text(paper.get('title'), limit=500)}",
        f"Abstract: {clean_text(paper.get('abstract'), limit=2800)}",
    ]
    if variant != "no prior-art retrieval":
        prior_art = format_retrieved_papers(retrieved_papers)
        blocks.append("Retrieved prior art:\n" + (prior_art or "No retrieved prior art available."))
    else:
        blocks.append("Retrieved prior art: DISABLED for this ablation.")
    if variant != "no graph agent":
        graph_text = normalize_space(json.dumps(graph_evidence, ensure_ascii=False) if isinstance(graph_evidence, Mapping) else graph_evidence, limit=1800)
        blocks.append("Graph-perturbation evidence:\n" + (graph_text or "No graph evidence available."))
    else:
        blocks.append("Graph-perturbation evidence: DISABLED for this ablation.")
    return "\n\n".join(blocks) + "\n"


def evidence_anchor_fraction(response: Mapping[str, Any]) -> float:
    """Estimate evidence-language grounding in generated review points."""
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
    return sum(1 for section in REQUIRED_REVIEW_SECTIONS if review_points(response, section)) / len(REQUIRED_REVIEW_SECTIONS)


def normalized_score_1_5(value: Any, default: float = 3.0) -> float:
    """Convert a 1-5 score to [0, 1]."""
    return unit_interval((score_1_5(value, default=default) - 1.0) / 4.0)


def peer_closeness(model_score: Any, peer_score: Any) -> Optional[float]:
    """Compare a generated 1-5 score with a peer label when available."""
    if peer_score is None or pd.isna(peer_score):
        return None
    return unit_interval(1.0 - abs(score_1_5(model_score) - score_1_5(peer_score)) / 4.0)


def render_disabled_review_text(response: Mapping[str, Any], variant: str) -> str:
    """Render response JSON to a readable review text artifact."""
    sections = [f"Variant: {variant}"]
    points = response.get("review_points") if isinstance(response.get("review_points"), Mapping) else {}
    for key, label in [
        ("novelty", "Novelty"),
        ("significance", "Significance"),
        ("prior_art", "Prior-art comparison"),
        ("evidence_rigor", "Evidence and rigor"),
        ("limitations", "Limitations"),
        ("future_work", "Future work"),
    ]:
        values = points.get(key, []) if isinstance(points, Mapping) else []
        if isinstance(values, str):
            values = [values]
        clean_values = [str(item).strip() for item in values if str(item).strip()] if isinstance(values, list) else []
        if clean_values:
            sections.append(f"{label}: " + " ".join(clean_values))
    if response.get("recommendation"):
        sections.append(f"Recommendation: {response.get('recommendation')}")
    if response.get("confidence") not in {None, ""}:
        sections.append(f"Confidence: {response.get('confidence')}")
    return "\n\n".join(sections).strip() + "\n"


def score_disabled_response(
    paper: pd.Series,
    response: Mapping[str, Any],
    *,
    variant: str,
    model_name: str,
    prompt_hash: str,
    runtime_seconds: float,
    review_text_path: str,
    evidence_trace_path: str,
) -> Dict[str, Any]:
    """Score one observed disabled-module rerun with the Fig.10 metric schema."""
    scores = response.get("scores_1_5") if isinstance(response.get("scores_1_5"), Mapping) else {}
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
    structure = structure_coverage(response)
    trace = 0.0 if variant == "no evidence trace" else evidence_anchor_fraction(response)
    prior_art_multiplier = 0.0 if variant == "no prior-art retrieval" else (1.0 if review_points(response, "prior_art") else 0.5)
    return {
        "case_id": str(paper.get("paper_id")),
        "paper_id": str(paper.get("paper_id")),
        "variant": variant,
        "source": "true_disabled_module_rerun",
        "run_status": "ok",
        "review_text_path": review_text_path,
        "evidence_trace_path": evidence_trace_path,
        "runtime_seconds": float(runtime_seconds),
        "failure_reason": "",
        "model_name": model_name,
        "prompt_hash": prompt_hash,
        "input_scope": INPUT_SCOPE,
        "scoring_protocol": SCORING_PROTOCOL,
        "semantic_agreement": unit_interval(semantic_agreement),
        "novelty_coverage": unit_interval(normalized_score_1_5(scores.get("novelty"), default=3.0) * structure),
        "prior_art_accuracy": unit_interval(normalized_score_1_5(scores.get("prior_art"), default=2.0) * prior_art_multiplier),
        "factuality": unit_interval(1.0 - unsupported),
        "readability": unit_interval(0.55 + 0.35 * structure + 0.10 * float(response.get("confidence", 0.5) or 0.5)),
        "unsupported_claim_rate": unsupported,
        "evidence_trace_completeness": unit_interval(trace),
        "review_structure_coverage": unit_interval(structure),
        "recommendation": str(response.get("recommendation", "")),
        "confidence": unit_interval(float(response.get("confidence", 0.5) or 0.5)),
    }


def load_agent_outputs(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load Fig.4 agent-output rows keyed by paper id."""
    return {str(row.get("paper_id") or row.get("case_id") or ""): row for row in read_jsonl(path)}


def load_retrieved_papers(agent_row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Load retrieved papers referenced by a Fig.4 agent row."""
    raw_path = str(agent_row.get("retrieved_papers_cache") or "").strip()
    if not raw_path:
        return []
    path = Path(raw_path)
    if not path.exists() or not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    papers = payload.get("retrieved_papers") if isinstance(payload, Mapping) else []
    return [dict(item) for item in papers if isinstance(item, Mapping)]


def existing_ok_pairs(path: Path, selected_ids: set[str]) -> Dict[tuple[str, str], Dict[str, Any]]:
    """Return existing successful sidecar rows for selected cases."""
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return {}
    pairs: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in frame.to_dict("records"):
        case_id = str(row.get("case_id") or row.get("paper_id") or "")
        variant = str(row.get("variant") or "")
        if case_id in selected_ids and variant in DISABLED_VARIANTS and str(row.get("run_status") or "").lower() == "ok":
            pairs[(case_id, variant)] = row
    return pairs


def write_disabled_manifest(
    out_dir: Path,
    *,
    model_name: str,
    total_case_count: int,
    selected_case_count: int,
    requested_variants: Sequence[str],
    ok_count: int,
) -> Dict[str, Any]:
    """Write and return a disabled-rerun manifest."""
    expected_rows = selected_case_count * len(requested_variants)
    full_expected_rows = total_case_count * len(DISABLED_VARIANTS)
    complete = (
        full_expected_rows > 0
        and selected_case_count >= total_case_count
        and set(requested_variants) == set(DISABLED_VARIANTS)
        and ok_count >= full_expected_rows
    )
    status = "observed_disabled_module_rerun_complete" if complete else "observed_disabled_module_rerun_partial"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "input_scope": INPUT_SCOPE,
        "scoring_protocol": SCORING_PROTOCOL,
        "requested_variants": list(requested_variants),
        "total_case_count": int(total_case_count),
        "selected_case_count": int(selected_case_count),
        "expected_case_variant_rows": int(expected_rows),
        "full_expected_disabled_case_variant_rows": int(full_expected_rows),
        "ok_count": int(ok_count),
        "status": status,
        "sidecar_csv": str(out_dir / FIG10_COMPLETED_DISABLED_RERUNS_FILE),
        "outputs_jsonl": str(out_dir / "fig10_disabled_module_rerun_outputs.jsonl"),
    }
    (out_dir / "fig10_disabled_module_rerun_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_disabled_module_reruns(
    *,
    fig4_metrics: Path,
    out_dir: Path,
    variants: Sequence[str] = tuple(DISABLED_VARIANTS),
    fig4_agent_outputs: Path = DEFAULT_FIG4_AGENT_OUTPUTS,
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_cases: Optional[int] = None,
    timeout: int = 120,
    resume: bool = False,
    skip_existing: bool = False,
) -> Dict[str, Any]:
    """Run Fig.10 disabled-module variants over frozen Fig.4 cases."""
    requested_variants = [str(variant) for variant in variants]
    unsupported = [variant for variant in requested_variants if variant not in DISABLED_MODULE_VARIANTS]
    if unsupported:
        raise ValueError(f"unsupported disabled variants: {unsupported}")
    fig4 = pd.read_csv(fig4_metrics)
    selected = fig4.head(max_cases) if max_cases else fig4
    selected_ids = {str(paper.get("paper_id")) for _, paper in selected.iterrows()}
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = out_dir / FIG10_COMPLETED_DISABLED_RERUNS_FILE
    existing_pairs = existing_ok_pairs(sidecar_path, selected_ids) if resume else {}
    agent_outputs = load_agent_outputs(fig4_agent_outputs)
    sidecar_rows: Dict[tuple[str, str], Dict[str, Any]] = dict(existing_pairs)
    raw_rows: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in read_jsonl(out_dir / "fig10_disabled_module_rerun_outputs.jsonl"):
        case_id = str(row.get("case_id") or row.get("paper_id") or "")
        variant = str(row.get("variant") or "")
        if case_id in selected_ids and variant in requested_variants:
            raw_rows[(case_id, variant)] = row
    for _, paper in selected.iterrows():
        case_id = str(paper.get("paper_id"))
        agent_row = agent_outputs.get(case_id, {})
        retrieved_papers = load_retrieved_papers(agent_row)
        graph_evidence = agent_row.get("graph_metric_evidence") or {
            "top_mechanisms": paper.get("top_mechanisms", ""),
            "graph_confidence": paper.get("graph_confidence", ""),
        }
        for variant in requested_variants:
            pair = (case_id, variant)
            if skip_existing and pair in sidecar_rows:
                continue
            prompt = build_disabled_variant_prompt(
                paper,
                variant=variant,
                retrieved_papers=retrieved_papers,
                graph_evidence=graph_evidence,
            )
            p_hash = prompt_hash(prompt)
            start = time.time()
            raw: Dict[str, Any] = {
                "case_id": case_id,
                "paper_id": case_id,
                "variant": variant,
                "model_name": model_name,
                "prompt_hash": p_hash,
                "input_scope": INPUT_SCOPE,
            }
            try:
                content = call_ollama(prompt, model_name=model_name, ollama_url=ollama_url, timeout=timeout)
                response = parse_json_response(content)
                runtime_seconds = time.time() - start
                safe_case = safe_token(case_id)
                safe_variant = safe_token(variant)
                artifact_dir = out_dir / "disabled_module_rerun_outputs" / safe_case / safe_variant
                artifact_dir.mkdir(parents=True, exist_ok=True)
                review_path = artifact_dir / "review.txt"
                trace_path = artifact_dir / "evidence_trace.json"
                review_path.write_text(render_disabled_review_text(response, variant), encoding="utf-8")
                trace_path.write_text(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "variant": variant,
                            "model_name": model_name,
                            "prompt_hash": p_hash,
                            "input_scope": INPUT_SCOPE,
                            "scoring_protocol": SCORING_PROTOCOL,
                            "disabled_instruction": disabled_module_instruction(variant),
                            "retrieved_papers_count": len(retrieved_papers) if variant != "no prior-art retrieval" else 0,
                            "graph_evidence_enabled": variant != "no graph agent",
                            "raw_response": response,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                sidecar_rows[pair] = score_disabled_response(
                    paper,
                    response,
                    variant=variant,
                    model_name=model_name,
                    prompt_hash=p_hash,
                    runtime_seconds=runtime_seconds,
                    review_text_path=str(review_path.resolve()),
                    evidence_trace_path=str(trace_path.resolve()),
                )
                raw.update({"run_status": "ok", "raw_response": content, "parsed_response": response, "runtime_seconds": runtime_seconds})
            except Exception as exc:  # noqa: BLE001 - failures are recorded and do not count toward the sidecar gate.
                raw.update({"run_status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
            raw_rows[pair] = raw
    ordered_sidecar = [
        sidecar_rows[(str(paper.get("paper_id")), variant)]
        for _, paper in selected.iterrows()
        for variant in requested_variants
        if (str(paper.get("paper_id")), variant) in sidecar_rows
    ]
    pd.DataFrame(ordered_sidecar).to_csv(sidecar_path, index=False)
    ordered_raw = [
        raw_rows[pair]
        for pair in sorted(raw_rows, key=lambda item: (item[0], item[1]))
    ]
    write_jsonl(out_dir / "fig10_disabled_module_rerun_outputs.jsonl", ordered_raw)
    return write_disabled_manifest(
        out_dir,
        model_name=model_name,
        total_case_count=len(fig4),
        selected_case_count=len(selected),
        requested_variants=requested_variants,
        ok_count=len(ordered_sidecar),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig4-metrics", type=Path, default=DEFAULT_FIG4_METRICS)
    parser.add_argument("--fig4-agent-outputs", type=Path, default=DEFAULT_FIG4_AGENT_OUTPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--variants", nargs="+", default=DISABLED_VARIANTS)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    result = run_disabled_module_reruns(
        fig4_metrics=args.fig4_metrics,
        fig4_agent_outputs=args.fig4_agent_outputs,
        out_dir=args.out_dir,
        variants=args.variants,
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
