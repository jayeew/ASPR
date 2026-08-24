"""Independent module publication, resolution, and review-comparison CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from artifact_store import ArtifactReference, ArtifactStore
from artifact_store.catalog import validate_dependency
from experiments.gear.review_reconstruction.evaluation import (
    MatchLabel,
    PointMatchDecision,
    evaluate_corpus,
    evaluate_review_pair,
)

from .module_registry import MODULES, artifact_root, reference_root
from .review_contracts import StructuredReview
from .trace import EvidenceStore


def _load_reference(path: Path) -> ArtifactReference:
    return ArtifactReference.model_validate_json(path.read_text(encoding="utf-8"))


def _write_reference(reference: ArtifactReference) -> Path:
    target = reference_root() / reference.producer / f"{reference.release}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = reference.model_dump_json(indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != payload:
        raise FileExistsError(
            f"reference release is already pinned differently: {target}"
        )
    target.write_text(payload, encoding="utf-8")
    return target


def _publish(args: argparse.Namespace) -> int:
    spec = MODULES[args.module]
    source = args.source.resolve()
    if not (source / spec.primary_file).is_file():
        raise FileNotFoundError(
            f"{args.module} release requires primary file {spec.primary_file}"
        )
    _validate_primary(args.module, source / spec.primary_file)
    dependencies = [_load_reference(path) for path in args.dependency]
    for dependency in dependencies:
        validate_dependency(args.module, dependency.producer)
    reference = ArtifactStore(artifact_root()).publish_directory(
        producer=args.module,
        artifact=spec.artifact,
        release=args.release,
        source=source,
        dependencies=dependencies,
        metadata={"primary_file": spec.primary_file, "module": args.module},
    )
    pinned = _write_reference(reference)
    print(json.dumps({"reference": str(pinned), "artifact": reference.model_dump()}))
    return 0


def _resolve(args: argparse.Namespace) -> int:
    reference = _load_reference(args.reference)
    path = ArtifactStore(artifact_root()).resolve(reference)
    print(json.dumps({"path": str(path), "reference": reference.model_dump()}))
    return 0


def _load_jsonl(path: Path) -> list[StructuredReview]:
    reviews = [
        StructuredReview.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not reviews:
        raise ValueError(f"StructuredReview release is empty: {path}")
    paper_ids = [review.paper_id for review in reviews]
    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError(
            f"StructuredReview release has duplicate paper_id values: {path}"
        )
    return reviews


def _validate_primary(module: str, path: Path) -> None:
    if module in {"review_reconstruction", "gear_agent"}:
        _load_jsonl(path)
    elif module == "indicator_definition":
        json.loads(path.read_text(encoding="utf-8"))


def _export_agent(args: argparse.Namespace) -> int:
    reviews: list[StructuredReview] = []
    sources: list[dict[str, object]] = []
    evidence: dict[str, list[str]] = {}
    traces: list[tuple[Path, str]] = []
    for run_dir in args.run_dir:
        resolved_run = run_dir.resolve()
        review_path = resolved_run / "review.json"
        review = StructuredReview.model_validate_json(
            review_path.read_text(encoding="utf-8")
        )
        reviews.append(review)
        store = EvidenceStore(resolved_run)
        failures = (
            store.validate_manifest()
            if (resolved_run / "run_manifest.json").is_file()
            else ["run_manifest_missing"]
        )
        if failures and failures != ["run_manifest_missing"]:
            raise ValueError(f"invalid Agent run {resolved_run}: {failures}")
        evidence[review.paper_id] = sorted(store.ids())
        trace_path = resolved_run / "evidence_trace.jsonl"
        trace_name = (
            hashlib.sha256(review.paper_id.encode("utf-8")).hexdigest() + ".jsonl"
        )
        if trace_path.is_file():
            traces.append((trace_path, trace_name))
        sources.append(
            {
                "paper_id": review.paper_id,
                "run_dir": str(resolved_run),
                "evidence_status": "verified" if not failures else "limited",
                "validation_failures": failures,
            }
        )
    if len({review.paper_id for review in reviews}) != len(reviews):
        raise ValueError("agent export contains duplicate paper_id values")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "agent_structured_reviews.jsonl").write_text(
        "".join(review.model_dump_json() + "\n" for review in reviews),
        encoding="utf-8",
    )
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "evidence_keys.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if traces:
        trace_dir = args.output_dir / "evidence"
        trace_dir.mkdir()
        for source, name in traces:
            shutil.copy2(source, trace_dir / name)
    return 0


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}|[\u3400-\u9fff]", text.casefold()))


def _decisions(
    reference: StructuredReview, candidate: StructuredReview
) -> list[PointMatchDecision]:
    rows: list[PointMatchDecision] = []
    for left in reference.all_points():
        for right in candidate.all_points():
            overlap = len(_terms(left.text) & _terms(right.text)) / max(
                len(_terms(left.text) | _terms(right.text)), 1
            )
            evidence_overlap = bool(set(left.evidence_keys) & set(right.evidence_keys))
            label = (
                MatchLabel.SAME_POINT
                if overlap >= 0.55
                else (
                    MatchLabel.PARTIAL_POINT
                    if overlap >= 0.25 or evidence_overlap
                    else MatchLabel.NO_MATCH
                )
            )
            rows.append(
                PointMatchDecision(
                    paper_id=reference.paper_id,
                    reference_point_id=left.point_id,
                    candidate_point_id=right.point_id,
                    label=label,
                    confidence=max(overlap, 0.5 if evidence_overlap else 0.0),
                    rationale="deterministic lexical/evidence-overlap baseline",
                )
            )
    return rows


def _compare(args: argparse.Namespace) -> int:
    store = ArtifactStore(artifact_root())
    human_ref = _load_reference(args.human_reference)
    agent_ref = _load_reference(args.agent_reference)
    validate_dependency("review_evaluation", human_ref.producer)
    validate_dependency("review_evaluation", agent_ref.producer)
    human_dir = store.resolve(human_ref)
    agent_dir = store.resolve(agent_ref)
    humans = _load_jsonl(human_dir / MODULES["review_reconstruction"].primary_file)
    agents = _load_jsonl(agent_dir / MODULES["gear_agent"].primary_file)
    evidence_path = agent_dir / "evidence_keys.json"
    evidence_by_paper = (
        json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence_path.is_file()
        else {}
    )
    agent_by_id = {review.paper_id: review for review in agents}
    pairs = [
        (review, agent_by_id[review.paper_id])
        for review in humans
        if review.paper_id in agent_by_id
    ]
    if not pairs:
        raise ValueError("human and agent releases contain no shared paper_id")
    pair_metrics = [
        evaluate_review_pair(
            left,
            right,
            _decisions(left, right),
            valid_evidence_keys=set(evidence_by_paper.get(right.paper_id, [])),
            development_non_confirmatory=True,
        )
        for left, right in pairs
    ]
    corpus = evaluate_corpus(
        pair_metrics,
        [(left.novelty.judgment, right.novelty.judgment) for left, right in pairs],
        development_non_confirmatory=True,
        bootstrap_samples=args.bootstrap_samples,
    )
    with tempfile.TemporaryDirectory(prefix="gear-evaluation-") as directory:
        result_dir = Path(directory)
        (result_dir / "corpus_metrics.json").write_text(
            corpus.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (result_dir / "sample_metrics.jsonl").write_text(
            "".join(row.model_dump_json() + "\n" for row in pair_metrics),
            encoding="utf-8",
        )
        (result_dir / "summary.json").write_text(
            json.dumps(
                {
                    "paper_count": len(pairs),
                    "matcher": "deterministic_baseline",
                    "evidence_validation": (
                        "verified_from_agent_evidence_store"
                        if all(
                            evidence_by_paper.get(right.paper_id) for _, right in pairs
                        )
                        else "limited_missing_evidence_store"
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        reference = store.publish_directory(
            producer="review_evaluation",
            artifact=MODULES["review_evaluation"].artifact,
            release=args.release,
            source=result_dir,
            dependencies=[human_ref, agent_ref],
            metadata={
                "primary_file": "corpus_metrics.json",
                "module": "review_evaluation",
            },
        )
    pinned = _write_reference(reference)
    print(
        json.dumps(
            {"reference": str(pinned), "metrics": corpus.model_dump(mode="json")}
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--module", choices=sorted(MODULES), required=True)
    publish.add_argument("--release", required=True)
    publish.add_argument("--source", type=Path, required=True)
    publish.add_argument("--dependency", type=Path, action="append", default=[])
    publish.set_defaults(handler=_publish)
    resolve = commands.add_parser("resolve")
    resolve.add_argument("--reference", type=Path, required=True)
    resolve.set_defaults(handler=_resolve)
    compare = commands.add_parser("compare")
    compare.add_argument("--human-reference", type=Path, required=True)
    compare.add_argument("--agent-reference", type=Path, required=True)
    compare.add_argument("--release", required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=2000)
    compare.set_defaults(handler=_compare)
    export_agent = commands.add_parser(
        "export-agent", help="Collect validated review.json files into one release"
    )
    export_agent.add_argument("--run-dir", type=Path, action="append", required=True)
    export_agent.add_argument("--output-dir", type=Path, required=True)
    export_agent.set_defaults(handler=_export_agent)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
