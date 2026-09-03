"""Recoverable orchestration for the unified GEAR-only evaluation."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiments.gear.review_reconstruction.evaluation import (
    build_blind_match_package,
    evaluate_review_pair,
)
from gear.config import load_config
from gear.contracts import PaperMetadata, ReviewRequest
from gear.graph_guidance import GRAPH_GUIDANCE_POLICY_VERSION
from gear.graph_prior import cutoff_safe_runtime_packet
from gear.graph_prior_contracts import GraphRuntimePacket
from gear.review_contracts import StructuredReview
from gear.review_pipeline import ServiceRegistry, review_paper
from gear.review_verifier import ReviewVerifier
from gear.trace import EvidenceStore, canonical_json, sha256_file, sha256_value

from .artifacts import (
    build_context_pack,
    load_manifest,
    load_reference_release,
    load_review_bundle,
)
from .client import CachedEvaluatorClient, EvaluationJudgeError, load_evaluator_config
from .efficiency import (
    efficiency_metrics,
    graph_action_metrics,
    run_integrity_metrics,
    supported_major_efficiency,
)
from .graph_ablation import assert_branch_isolation, graph_variants
from .judges import (
    judge_blind_review_preference,
    judge_point_support,
    judge_revision_issues,
    judge_semantic_matches,
    score_review_quality,
)
from .metrics import (
    bootstrap_ci,
    concern_coverage_metrics,
    evidence_support_metrics,
    macro,
    novelty_direction_metrics,
    retrieval_ranking_metrics,
    revision_metrics,
    rubric_metrics,
    semantic_match_metrics,
)

STAGES = (
    "preflight",
    "run-clean",
    "run-faults",
    "run-graph-ablations",
    "prepare-judges",
    "run-judges",
    "score",
    "report",
)


def _logical_retrieval_total(ledger: Any) -> int | None:
    if ledger is None:
        return None
    return sum(
        int(getattr(ledger, field, 0) or 0)
        for field in (
            "logical_provider_searches",
            "logical_direct_fetches",
            "logical_neighbor_expansions",
        )
    )


def _aggregate_graph_deltas(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["comparison"]].append(row)
    output: dict[str, dict[str, Any]] = {}
    for comparison, comparison_rows in grouped.items():
        metric_names = sorted(
            {
                key
                for row in comparison_rows
                for key, value in row.items()
                if key.endswith(("_delta", "_rate")) and isinstance(value, (int, float))
            }
        )
        output[comparison] = {}
        for index, name in enumerate(metric_names):
            values = [
                float(row[name])
                for row in comparison_rows
                if isinstance(row.get(name), (int, float))
            ]
            if not values:
                continue
            output[comparison][name] = {
                "n": len(values),
                "mean": sum(values) / len(values),
                "bootstrap_95_ci": bootstrap_ci(
                    values, samples=samples, seed=seed + index
                ),
                "positive_rate": sum(value > 0 for value in values) / len(values),
                "negative_rate": sum(value < 0 for value in values) / len(values),
                "zero_rate": sum(value == 0 for value in values) / len(values),
            }
    return output


class StaticGraphScorer:
    def __init__(self, result: GraphRuntimePacket) -> None:
        self.result = result

    def score(self, paper_ir: Any, cutoff: Any) -> GraphRuntimePacket:
        packet = self.result.model_copy(update={"paper_id": paper_ir.paper_id})
        return cutoff_safe_runtime_packet(packet, cutoff)


class EvaluationRunner:
    def __init__(
        self,
        *,
        manifest_path: Path,
        judge_config_path: Path,
        output_dir: Path,
        resume: bool,
        runtime_config_path: Path | None = None,
        workers: int = 1,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.judge_config_path = judge_config_path.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(self.manifest_path)
        self.judge_config = load_evaluator_config(self.judge_config_path)
        self.runtime_config_path = (
            runtime_config_path.resolve() if runtime_config_path is not None else None
        )
        self.resume = resume
        self.workers = max(1, workers)
        self.fingerprint = self._fingerprint()

    def run(self, requested: str) -> None:
        stages = STAGES if requested == "all" else (requested,)
        if requested not in {*STAGES, "all"}:
            raise ValueError(f"unknown evaluation stage: {requested}")
        for stage in stages:
            handler = getattr(self, stage.replace("-", "_"))
            self._run_stage(stage, handler)
        self._write_run_manifest()

    def preflight(self) -> None:
        reviews, _ = load_reference_release(self.manifest.reference_release_dir)
        for case in self.manifest.cases:
            for path in (case.manuscript_path, case.metadata_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if case.paper_id not in reviews:
                raise ValueError(f"reference review missing for {case.paper_id}")
            if case.clean_run_dir is not None:
                bundle = load_review_bundle(case.clean_run_dir)
                failures = EvidenceStore(case.clean_run_dir).validate_manifest()
                if failures or not bundle.verification.passed:
                    raise ValueError(
                        f"clean run is not reusable: {case.case_id}: {failures}"
                    )

    def run_clean(self) -> None:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            records = list(pool.map(self._run_clean_case, self.manifest.cases))
        self._write_json("clean_runs.json", records)

    def _run_clean_case(self, case: Any) -> dict[str, Any]:
        target = self.output_dir / "clean_runs" / case.case_id
        if case.clean_run_dir is not None:
            bundle = load_review_bundle(case.clean_run_dir)
            run_dir = case.clean_run_dir
            reused = True
        else:
            existed = (target / "review_bundle.json").is_file()
            bundle = self._run_case(case, case.graph_result, target)
            run_dir = target
            reused = self.resume and existed
        return {
            "case_id": case.case_id,
            "paper_id": case.paper_id,
            "run_dir": str(run_dir),
            "status": bundle.status.value,
            "reused": reused,
        }

    def run_faults(self) -> None:
        from .faults import default_fault_scenarios, run_contract_fault_matrix

        rows = run_contract_fault_matrix(self.manifest)
        executable = {
            "graph_exception",
            "graph_id_mismatch",
            "graph_invalid",
            "semantic_exception",
            "semantic_invalid",
            "retrieval_exception",
            "agent_invalid",
            "qwen_required_missing",
        }
        clean = self._clean_run_map()
        scenarios = {
            scenario.kind: scenario
            for scenario in default_fault_scenarios()
            if scenario.kind in executable
        }
        observed = []
        for case in self.manifest.cases:
            clean_bundle = load_review_bundle(clean[case.case_id])
            for kind, scenario in scenarios.items():
                target = self.output_dir / "fault_runs" / kind / case.case_id
                bundle = self._run_fault_case(case, clean_bundle, kind, target)
                reasons = list(
                    (bundle.process_diagnostic or {}).get("blocking_reasons", [])
                )
                passed = bundle.status == scenario.expected_status and set(
                    scenario.required_reason_codes
                ).issubset(reasons)
                observed.append(
                    {
                        "case_id": case.case_id,
                        "paper_id": case.paper_id,
                        "scenario": scenario.model_dump(mode="json"),
                        "execution_level": "deterministic_service_registry",
                        "observed_status": bundle.status.value,
                        "observed_reason_codes": reasons,
                        "passed": passed,
                        "run_dir": str(target),
                    }
                )
            trace_scenario = next(
                item
                for item in default_fault_scenarios()
                if item.kind == "trace_corruption"
            )
            trace_target = (
                self.output_dir / "fault_runs" / "trace_corruption" / case.case_id
            )
            if not trace_target.exists():
                shutil.copytree(clean[case.case_id], trace_target)
                action_path = trace_target / "action_trace.jsonl"
                action_path.write_text(action_path.read_text() + "\n")
            manifest_failures = EvidenceStore(trace_target).validate_manifest()
            detected = any(
                "trace_file_hash_mismatch" in item for item in manifest_failures
            )
            observed.append(
                {
                    "case_id": case.case_id,
                    "paper_id": case.paper_id,
                    "scenario": trace_scenario.model_dump(mode="json"),
                    "execution_level": "deterministic_manifest_corruption",
                    "observed_status": "limited" if detected else "complete",
                    "observed_reason_codes": (
                        ["evidence_integrity_failed"] if detected else []
                    ),
                    "manifest_corruption_detected": detected,
                    "passed": detected,
                    "run_dir": str(trace_target),
                }
            )
        fixture_keys = {(row["case_id"], row["scenario"]["kind"]) for row in observed}
        rows = [
            row
            for row in rows
            if (row["case_id"], row["scenario"]["kind"]) not in fixture_keys
        ]
        rows.extend(observed)
        self._write_jsonl("fault_results.jsonl", rows)

    def run_graph_ablations(self) -> None:
        rows: list[dict[str, Any]] = []
        isolation: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = pool.map(
                self._run_ablation_case,
                self.manifest.cases,
            )
            for case_rows, payloads in results:
                rows.extend(case_rows)
                isolation[case_rows[0]["case_id"]] = payloads
        for payloads in isolation.values():
            assert_branch_isolation(payloads)
        self._write_jsonl("graph_ablation_results.jsonl", rows)

    def _run_ablation_case(
        self, case: Any
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, object]]]:
        variants = graph_variants(
            case.graph_result,
            placebo=case.placebo_graph_result,
        )
        rows: list[dict[str, Any]] = []
        payloads: dict[str, dict[str, object]] = {}
        ledgers: dict[str, Any] = {}
        clean_dir = self._clean_run_map()[case.case_id]
        clean_bundle = load_review_bundle(clean_dir)
        for variant in variants:
            target = self.output_dir / "graph_ablations" / variant.name / case.case_id
            bundle = self._run_ablation_variant(
                case,
                variant.name,
                variant.result,
                target,
                clean_bundle,
                clean_dir,
            )
            branch, state = bundle.agent_review, bundle.state
            if branch is None or state is None:
                raise ValueError("ablation run lacks Agent branch or ReviewState")
            ledgers[variant.name] = state.resource_ledger
            candidates = sorted(
                (point.point_id, point.text) for point in branch.all_points()
            )
            payloads[variant.name] = {
                "draft_sha256": sha256_value(branch),
                "local_candidate_pool_sha256": _routing_pool_sha256(state, "local"),
                "remote_candidate_pool_sha256": _routing_pool_sha256(state, "remote"),
                "resource_caps_sha256": sha256_value(state.resource_ledger.caps),
                "agent_input_sha256": branch.input_sha256,
                "candidate_set_sha256": sha256_value(candidates),
            }
            rows.append(
                {
                    "case_id": case.case_id,
                    "paper_id": case.paper_id,
                    "variant": variant.name,
                    "status": bundle.status.value,
                    "run_dir": str(target),
                    "graph_result": variant.result.model_dump(mode="json"),
                    **payloads[variant.name],
                }
            )
        cap_payloads = {
            sha256_value(ledger.caps)
            for ledger in ledgers.values()
            if ledger is not None
        }
        if len(cap_payloads) != 1:
            raise ValueError("Graph ablation resource caps are not identical")
        if {"score_topology", "placebo_graph"} <= ledgers.keys():
            topology = ledgers["score_topology"]
            wrong = ledgers["placebo_graph"]
            topology_total = _logical_retrieval_total(topology)
            wrong_total = _logical_retrieval_total(wrong)
            matched = bool(
                topology is not None
                and wrong is not None
                and topology_total == wrong_total
            )
            for row in rows:
                if row["variant"] == "placebo_graph":
                    row["topology_control_resource_matched"] = matched
                    row["topology_direct_fetches"] = (
                        topology.logical_direct_fetches
                        if topology is not None
                        else None
                    )
                    row["control_direct_fetches"] = (
                        wrong.logical_direct_fetches if wrong is not None else None
                    )
                    row["topology_neighbor_expansions"] = (
                        topology.logical_neighbor_expansions
                        if topology is not None
                        else None
                    )
                    row["control_neighbor_expansions"] = (
                        wrong.logical_neighbor_expansions if wrong is not None else None
                    )
                    row["topology_logical_retrieval_requests"] = topology_total
                    row["control_logical_retrieval_requests"] = wrong_total
        return rows, payloads

    def _run_ablation_variant(
        self,
        case: Any,
        variant_name: str,
        graph: GraphRuntimePacket,
        target: Path,
        clean_bundle: Any,
        clean_dir: Path,
    ) -> Any:
        from .faults import (
            FixedAgentReviewer,
            FixedPaperCompiler,
            FixedPaperExtractor,
            FixedQwenReviewer,
        )

        if self.resume and (target / "review_bundle.json").is_file():
            return self._run_case(case, graph, target)
        branch = clean_bundle.agent_review
        if branch is None:
            raise ValueError("clean bundle has no Agent branch for Graph ablation")
        record = EvidenceStore(clean_dir).get("A:BRANCH")
        payload = dict(record.payload.get("input_payload") or {}) if record else {}
        metadata = PaperMetadata.model_validate_json(case.metadata_path.read_text())
        request = ReviewRequest(
            paper_path=case.manuscript_path,
            metadata=metadata,
            evaluation_date=case.cutoff_date,
        )
        config = load_config(self.runtime_config_path)
        config = config.model_copy(
            update={
                "graph_guidance": config.graph_guidance.model_copy(
                    update={
                        "score_routing_enabled": variant_name != "neutral",
                        "topology_enabled": variant_name
                        in {"score_topology", "placebo_graph"},
                    }
                )
            }
        )
        services = ServiceRegistry(
            evidence_store=EvidenceStore(target),
            paper_compiler=FixedPaperCompiler(clean_bundle.paper_ir),
            paper_extractor=FixedPaperExtractor(),
            graph_scorer=StaticGraphScorer(graph),
            agent_reviewer=FixedAgentReviewer(branch, payload),
            qwen_reviewer=FixedQwenReviewer(),
        )
        return review_paper(
            request,
            output_dir=target,
            config=config,
            services=services,
            full_artifacts=False,
        )

    def prepare_judges(self) -> None:
        human, _ = load_reference_release(self.manifest.reference_release_dir)
        clean = self._clean_run_map()
        targets = self.output_dir / "judge_inputs"
        targets.mkdir(parents=True, exist_ok=True)
        for case in self.manifest.cases:
            bundle = load_review_bundle(clean[case.case_id])
            context = build_context_pack(
                bundle.paper_ir,
                human[case.paper_id],
                bundle.structured_review,
            )
            package = build_blind_match_package(
                human[case.paper_id], bundle.structured_review
            )
            (targets / f"{case.case_id}.context.json").write_text(
                context.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            (targets / f"{case.case_id}.match.json").write_text(
                package.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )

    def run_judges(self) -> None:
        human, revision_labels = load_reference_release(
            self.manifest.reference_release_dir
        )
        clean = self._clean_run_map()
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        indexed_cases = list(enumerate(self.manifest.cases))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = pool.map(
                lambda item: self._judge_clean_case(
                    item[0], item[1], human, revision_labels, clean
                ),
                indexed_cases,
            )
            for case_rows, case_failures in results:
                rows.extend(case_rows)
                failures.extend(case_failures)
        ablation_path = self.output_dir / "graph_ablation_results.jsonl"
        ablation_rows = list(_jsonl(ablation_path))
        grouped = [
            [row for row in ablation_rows if row["case_id"] == case.case_id]
            for case in self.manifest.cases
        ]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = pool.map(
                lambda item: self._judge_ablation_case(item[0], item[1], human),
                zip(self.manifest.cases, grouped, strict=True),
            )
            for case_rows, case_failures in results:
                rows.extend(case_rows)
                failures.extend(case_failures)
        self._write_jsonl("judge_decisions.jsonl", rows)
        self._write_json("judge_failures.json", failures)

    def _judge_clean_case(
        self,
        case_index: int,
        case: Any,
        human: dict[str, StructuredReview],
        revision_labels: dict[str, list[Any]],
        clean: dict[str, Path],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        client = CachedEvaluatorClient(self.judge_config)
        bundle = load_review_bundle(clean[case.case_id])
        context = build_context_pack(
            bundle.paper_ir, human[case.paper_id], bundle.structured_review
        )
        package = build_blind_match_package(
            human[case.paper_id], bundle.structured_review
        )
        try:
            values: dict[str, Any] = {
                "rubric": score_review_quality(
                    client, context, human[case.paper_id], bundle.structured_review
                ),
                "support": judge_point_support(
                    client,
                    bundle.structured_review,
                    _evidence_payloads(clean[case.case_id]),
                ),
                "matches": judge_semantic_matches(client, package),
            }
            labels = revision_labels.get(case.paper_id)
            if labels:
                values["revision"] = judge_revision_issues(
                    client, case.paper_id, labels, bundle.structured_review
                )
            other = self.manifest.cases[(case_index + 1) % len(self.manifest.cases)]
            values["wrong_paper_support"] = judge_point_support(
                client,
                bundle.structured_review,
                _evidence_payloads(clean[other.case_id]),
            )
            return _decision_rows(case, values), []
        except (EvaluationJudgeError, ValueError) as exc:
            return [], [{"case_id": case.case_id, "reason": f"judge_failed:{exc}"}]

    def _judge_ablation_case(
        self,
        case: Any,
        ablations: list[dict[str, Any]],
        human: dict[str, StructuredReview],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        client = CachedEvaluatorClient(self.judge_config)
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        bundles: dict[str, Any] = {}
        for item in ablations:
            run_dir = Path(item["run_dir"])
            bundle = load_review_bundle(run_dir)
            bundles[item["variant"]] = bundle
            context = build_context_pack(
                bundle.paper_ir, human[case.paper_id], bundle.structured_review
            )
            package = build_blind_match_package(
                human[case.paper_id], bundle.structured_review
            )
            try:
                values = {
                    "rubric": score_review_quality(
                        client,
                        context,
                        human[case.paper_id],
                        bundle.structured_review,
                    ),
                    "support": judge_point_support(
                        client, bundle.structured_review, _evidence_payloads(run_dir)
                    ),
                    "matches": judge_semantic_matches(client, package),
                }
                rows.extend(
                    _decision_rows(case, values, prefix=f"ablation/{item['variant']}/")
                )
            except (EvaluationJudgeError, ValueError) as exc:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "variant": item["variant"],
                        "reason": f"judge_failed:{exc}",
                    }
                )
        if {"neutral", "topology"} <= bundles.keys():
            topology = bundles["topology"]
            neutral = bundles["neutral"]
            context = build_context_pack(
                topology.paper_ir,
                human[case.paper_id],
                topology.structured_review,
            )
            review_a, review_b = (
                (topology.structured_review, neutral.structured_review)
                if _topology_is_a(case.case_id)
                else (neutral.structured_review, topology.structured_review)
            )
            try:
                preference = judge_blind_review_preference(
                    client,
                    context,
                    review_a,
                    review_b,
                )
                rows.extend(
                    _decision_rows(
                        case,
                        {"preference": preference},
                        prefix="ablation/",
                    )
                )
            except (EvaluationJudgeError, ValueError) as exc:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "variant": "topology-neutral-preference",
                        "reason": f"judge_failed:{exc}",
                    }
                )
        return rows, failures

    def score(self) -> None:
        human, labels = load_reference_release(self.manifest.reference_release_dir)
        clean = self._clean_run_map()
        decisions = self._judge_map()
        samples = []
        not_measured: list[dict[str, Any]] = []
        directions = []
        for case in self.manifest.cases:
            bundle = load_review_bundle(clean[case.case_id])
            review = bundle.structured_review
            kinds = decisions.get(case.case_id, {})
            row: dict[str, Any] = {
                "case_id": case.case_id,
                "paper_id": case.paper_id,
                "status": bundle.status.value,
                **efficiency_metrics(clean[case.case_id], review),
                **graph_action_metrics(clean[case.case_id]),
                **run_integrity_metrics(clean[case.case_id], review, bundle.paper_ir),
            }
            if bundle.state is not None:
                used = bundle.state.action_budget.actions_used
                maximum = bundle.state.action_budget.total_actions_max
                row.update(
                    {
                        "evidence_actions_used": used,
                        "evidence_action_budget": maximum,
                        "evidence_action_budget_utilization": used / maximum,
                    }
                )
            if {"rubric", "support", "matches", "wrong_paper_support"} <= kinds.keys():
                rubric = kinds["rubric"]
                support = kinds["support"]
                matches = kinds["matches"]
                row.update(rubric_metrics(rubric.scores))
                row.update(evidence_support_metrics(review, support.decisions))
                row.update(
                    supported_major_efficiency(
                        clean[case.case_id],
                        review,
                        {
                            item.point_id
                            for item in support.decisions
                            if item.label == "SUPPORTED"
                        },
                    )
                )
                wrong_support_metrics = evidence_support_metrics(
                    review, kinds["wrong_paper_support"].decisions
                )
                row["wrong_paper_strict_support_precision"] = wrong_support_metrics[
                    "strict_support_precision"
                ]
                correct_support = row["strict_support_precision"]
                wrong_support = row["wrong_paper_strict_support_precision"]
                row["judge_discrimination_failure"] = (
                    correct_support is not None
                    and wrong_support is not None
                    and wrong_support >= correct_support
                )
                pair = evaluate_review_pair(
                    human[case.paper_id],
                    review,
                    matches.decisions,
                    valid_evidence_keys=set(EvidenceStore(clean[case.case_id]).ids()),
                    semantically_supported_point_ids={
                        item.point_id
                        for item in support.decisions
                        if item.label == "SUPPORTED"
                    },
                    development_non_confirmatory=self.manifest.development_non_confirmatory,
                )
                row.update(
                    {
                        "atomic_strict_precision": pair.atomic_precision,
                        "atomic_strict_recall": pair.atomic_recall,
                        "atomic_strict_f1": pair.atomic_f1,
                        "novelty_reasoning_strict_precision": pair.novelty_point_precision,
                        "novelty_reasoning_strict_recall": pair.novelty_point_recall,
                        "novelty_reasoning_strict_f1": pair.novelty_point_f1,
                    }
                )
                match_metrics = semantic_match_metrics(
                    matches.decisions,
                    reference_count=len(human[case.paper_id].all_points()),
                    candidate_count=len(review.all_points()),
                )
                row.update(
                    {f"atomic_{key}": value for key, value in match_metrics.items()}
                )
                row.update(
                    concern_coverage_metrics(
                        human[case.paper_id], review, matches.decisions
                    )
                )
                novelty_reference_ids = {
                    point.point_id
                    for point in [
                        *human[case.paper_id].novelty.supporting_points,
                        *human[case.paper_id].novelty.limiting_points,
                        *human[case.paper_id].novelty.uncertain_points,
                    ]
                }
                novelty_candidate_ids = {
                    point.point_id
                    for point in [
                        *review.novelty.supporting_points,
                        *review.novelty.limiting_points,
                        *review.novelty.uncertain_points,
                    ]
                }
                novelty_decisions = [
                    item
                    for item in matches.decisions
                    if item.reference_point_id in novelty_reference_ids
                    and item.candidate_point_id in novelty_candidate_ids
                ]
                novelty_matches = semantic_match_metrics(
                    novelty_decisions,
                    reference_count=len(novelty_reference_ids),
                    candidate_count=len(novelty_candidate_ids),
                )
                row.update(
                    {
                        f"novelty_reasoning_{key}": value
                        for key, value in novelty_matches.items()
                    }
                )
            else:
                not_measured.append(
                    {
                        "paper_id": case.paper_id,
                        "metric": "semantic_judge_metrics",
                        "reason": "judge_failed",
                    }
                )
            if labels.get(case.paper_id) and "revision" in kinds:
                revision_decisions = kinds["revision"].decisions
                matched_issue_ids = {
                    item.issue_id
                    for item in revision_decisions
                    if item.label in {"SAME_POINT", "PARTIAL_POINT"}
                }
                row.update(revision_metrics(labels[case.paper_id], matched_issue_ids))
            elif not labels.get(case.paper_id):
                not_measured.append(
                    {
                        "paper_id": case.paper_id,
                        "metric": "revision",
                        "reason": "reference release has no revision sidecar",
                    }
                )
            if case.prior_art_gold_path is None:
                not_measured.append(
                    {
                        "paper_id": case.paper_id,
                        "metric": "prior_art_ranking",
                        "reason": "prior_art_gold not supplied",
                    }
                )
            else:
                gold_rows = _jsonl(case.prior_art_gold_path)
                gold_ids = {
                    str(item.get("work_id") or item.get("prior_work_id"))
                    for item in gold_rows
                    if (item.get("work_id") or item.get("prior_work_id"))
                    and item.get("paper_id", case.paper_id) == case.paper_id
                }
                store = EvidenceStore(clean[case.case_id])
                ranked_ids = []
                for key in store.ids():
                    if not key.startswith("R:"):
                        continue
                    record = store.get(key)
                    if record is not None and record.payload.get("prior_work_id"):
                        ranked_ids.append(str(record.payload["prior_work_id"]))
                if gold_ids:
                    row.update(retrieval_ranking_metrics(ranked_ids, gold_ids))
                else:
                    not_measured.append(
                        {
                            "paper_id": case.paper_id,
                            "metric": "prior_art_ranking",
                            "reason": "prior_art_gold has no usable work IDs",
                        }
                    )
            directions.append(
                (human[case.paper_id].novelty.judgment, review.novelty.judgment)
            )
            samples.append(row)
        self._write_jsonl("sample_metrics.jsonl", samples)
        aggregate = self._aggregate(samples)
        aggregate["completed_coverage"] = (
            sum(row.get("status") == "complete" for row in samples) / len(samples)
            if samples
            else None
        )
        aggregate["novelty_judgment"] = novelty_direction_metrics(
            [left for left, _ in directions], [right for _, right in directions]
        )
        from .faults import reliability_metrics

        fault_rows = _jsonl(self.output_dir / "fault_results.jsonl")
        aggregate["reliability"] = reliability_metrics(fault_rows)
        not_measured.extend(
            {
                "paper_id": None,
                "metric": metric,
                "reason": "no applicable end-to-end stress observation",
            }
            for metric, value in aggregate["reliability"].items()
            if value is None
        )
        self._write_json("metrics.json", aggregate)
        self._write_json("not_measured.json", not_measured)
        self._score_graph_ablations(not_measured)

    def report(self) -> None:
        metrics = json.loads((self.output_dir / "metrics.json").read_text())
        text = [
            "# GEAR Evaluation Results",
            "",
            "> Development-only, non-confirmatory Nature pilot. These results do not support generalization, capability, acceptance-rate, or reviewer-level claims.",
            "",
            "## Primary KPIs",
            "",
        ]
        for key in (
            "completed_coverage",
            "reference_concern_coverage",
            "weighted_alignment_f1",
            "analytical_quality",
            "major_support_precision",
        ):
            value = (
                metrics.get(key)
                if key == "completed_coverage"
                else metrics.get("paper_macro", {}).get(key)
            )
            text.append(f"- `{key}`: {_render_metric(value)}")
        text.extend(["", "## Drivers", ""])
        for key in (
            "issue_family_coverage",
            "major_reference_concern_coverage",
            "persistent_concern_recall",
            "independent_prior_count",
            "relation_count",
            "graph_trigger_compliance",
        ):
            text.append(
                f"- `{key}`: {_render_metric(metrics.get('paper_macro', {}).get(key))}"
            )
        text.extend(["", "## Guardrails", ""])
        for key in (
            "unsupported_major_rate",
            "post_cutoff_leakage_rate",
            "wrong_paper_relation_contamination_rate",
        ):
            text.append(
                f"- `{key}`: {_render_metric(metrics.get('paper_macro', {}).get(key))}"
            )
        text.extend(["", "## Exact reproduction diagnostics", ""])
        for label, key in (
            ("Exact Point Reproduction F1", "atomic_strict_f1"),
            (
                "Exact Novelty Argument Reproduction F1",
                "novelty_reasoning_strict_f1",
            ),
        ):
            text.append(
                f"- {label} (`{key}`): "
                f"{_render_metric(metrics.get('paper_macro', {}).get(key))}"
            )
        graph_path = self.output_dir / "graph_ablation_metrics.json"
        if graph_path.is_file():
            graph_metrics = json.loads(graph_path.read_text())
            text.extend(["", "## Graph ablation", ""])
            for comparison in (
                "topology-neutral",
                "topology-wrong_paper_topology",
            ):
                aggregate = graph_metrics.get("aggregate", {}).get(comparison, {})
                text.append(f"- `{comparison}`:")
                for key in (
                    "relation_count_delta",
                    "claim_relevant_verified_relation_count_delta",
                    "material_correction_count_delta",
                    "claim_relevant_verified_relation_yield_delta",
                    "relation_to_material_correction_rate_delta",
                    "independent_prior_count_delta",
                    "analytical_quality_delta",
                    "major_support_precision_delta",
                    "novelty_reasoning_soft_f1_delta",
                    "unsupported_major_count_delta",
                ):
                    if key in aggregate:
                        text.append(
                            f"  - `{key}` mean: "
                            f"{_render_metric(aggregate[key].get('mean'))} "
                            f"(n={aggregate[key].get('n')})"
                        )
            text.append(
                "- `wrong_paper_topology_harm_rate`: "
                f"{_render_metric(graph_metrics.get('wrong_paper_topology_harm_rate'))}"
            )
            text.append(
                "- `blind_review_preference_score`: "
                f"{_render_metric(graph_metrics.get('blind_review_preference_score'))}"
            )
        text.extend(["", "## Reliability", ""])
        for key, value in metrics.get("reliability", {}).items():
            text.append(f"- `{key}`: {_render_metric(value)}")
        text.extend(["", "No cross-category composite score is produced.", ""])
        (self.output_dir / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
        self._write_paper_results()

    def _write_paper_results(self) -> None:
        rows = _jsonl(self.output_dir / "sample_metrics.jsonl")
        columns = (
            ("case_id", "Case"),
            ("status", "Status"),
            ("reference_concern_coverage", "Concern cov."),
            ("weighted_alignment_f1", "Weighted F1"),
            ("analytical_quality", "Analytical"),
            ("major_support_precision", "Major support"),
            ("novelty_reasoning_soft_f1", "Novelty soft F1"),
            ("persistent_concern_recall", "Persistent recall"),
            ("resolved_issue_resurrection_rate", "Resolved resurrection"),
            ("relation_count", "Relations"),
            ("wall_time_seconds", "Wall seconds"),
        )
        text = [
            "# Per-paper GEAR evaluation results",
            "",
            "> Development-only, non-confirmatory. Null means the metric was not applicable for that paper.",
            "",
            "| " + " | ".join(label for _, label in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for row in rows:
            values = []
            for key, _ in columns:
                value = row.get(key)
                values.append(
                    str(value)
                    if key in {"case_id", "status"}
                    else _render_metric(value)
                )
            text.append("| " + " | ".join(values) + " |")
        text.append("")
        (self.output_dir / "PAPER_RESULTS.md").write_text(
            "\n".join(text), encoding="utf-8"
        )

    def _score_graph_ablations(self, not_measured: list[dict[str, Any]]) -> None:
        path = self.output_dir / "graph_ablation_results.jsonl"
        if not path.is_file():
            return
        rows = _jsonl(path)
        human, _ = load_reference_release(self.manifest.reference_release_dir)
        decisions = self._judge_map()
        for row in rows:
            run_dir = Path(row["run_dir"])
            bundle = load_review_bundle(run_dir)
            review = bundle.structured_review
            row["metrics"] = {
                **efficiency_metrics(run_dir, review),
                **graph_action_metrics(run_dir),
                **run_integrity_metrics(run_dir, review, bundle.paper_ir),
                "unsupported_major_count": bundle.verification.unsupported_major_count,
                "graph_semantic_violation": bundle.verification.graph_semantic_violation_count,
            }
            reference_direction = human[row["paper_id"]].novelty.judgment
            final_direction = review.novelty.judgment
            direction_metrics = novelty_direction_metrics(
                [reference_direction], [final_direction]
            )
            row["metrics"].update(
                {
                    "novelty_direction_exact_match": direction_metrics[
                        "judgment_accuracy"
                    ],
                    "novelty_direction_agreement": direction_metrics[
                        "novelty_direction_agreement"
                    ],
                }
            )
            row["reference_novelty_direction"] = reference_direction.value
            row["initial_novelty_direction"] = (
                bundle.agent_review.novelty.judgment.value
                if bundle.agent_review is not None
                else None
            )
            row["final_novelty_direction"] = final_direction.value
            graph_payload = row["graph_result"]
            score_fraction = float(graph_payload["score_0_100"]) / 100.0
            row["score_fraction"] = score_fraction
            row["score_band"] = (
                "low"
                if score_fraction < 0.25
                else "medium" if score_fraction < 0.75 else "high"
            )
            prefix = f"ablation/{row['variant']}"
            kinds = decisions.get(row["case_id"], {})
            required = {
                f"{prefix}/rubric",
                f"{prefix}/support",
                f"{prefix}/matches",
            }
            if required <= kinds.keys():
                rubric = kinds[f"{prefix}/rubric"]
                support = kinds[f"{prefix}/support"]
                matches = kinds[f"{prefix}/matches"]
                row["metrics"].update(rubric_metrics(rubric.scores))
                row["metrics"].update(
                    evidence_support_metrics(review, support.decisions)
                )
                reference = human[row["paper_id"]]
                novelty_reference_ids = {
                    point.point_id
                    for point in [
                        *reference.novelty.supporting_points,
                        *reference.novelty.limiting_points,
                        *reference.novelty.uncertain_points,
                    ]
                }
                novelty_candidate_ids = {
                    point.point_id
                    for point in [
                        *review.novelty.supporting_points,
                        *review.novelty.limiting_points,
                        *review.novelty.uncertain_points,
                    ]
                }
                novelty_decisions = [
                    item
                    for item in matches.decisions
                    if item.reference_point_id in novelty_reference_ids
                    and item.candidate_point_id in novelty_candidate_ids
                ]
                matched = semantic_match_metrics(
                    novelty_decisions,
                    reference_count=len(novelty_reference_ids),
                    candidate_count=len(novelty_candidate_ids),
                )
                row["metrics"].update(
                    {
                        f"novelty_reasoning_{key}": value
                        for key, value in matched.items()
                    }
                )
        by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_case[row["case_id"]][row["variant"]] = row
        deltas = []
        keys = (
            "relation_count",
            "claim_relevant_verified_relation_count",
            "material_correction_count",
            "claim_relevant_verified_relation_yield",
            "relation_to_material_correction_rate",
            "logical_retrieval_requests",
            "network_retrieval_attempts",
            "independent_prior_count",
            "evidence_action_count",
            "unsupported_major_count",
            "post_cutoff_leakage_rate",
            "graph_semantic_violation",
            "cumulative_stage_time_seconds",
            "analytical_quality",
            "major_support_precision",
            "novelty_reasoning_strict_f1",
            "novelty_reasoning_soft_f1",
            "novelty_direction_exact_match",
            "novelty_direction_agreement",
        )
        for case_id, variants in by_case.items():
            for left_name, right_name in (
                ("topology", "neutral"),
                ("topology", "wrong_paper_topology"),
            ):
                if not {left_name, right_name} <= variants.keys():
                    continue
                if (
                    right_name == "wrong_paper_topology"
                    and variants[right_name].get("topology_control_resource_matched")
                    is not True
                ):
                    not_measured.append(
                        {
                            "case_id": case_id,
                            "track": "graph_ablation",
                            "metric": "topology-wrong_paper_topology",
                            "reason": "total logical retrieval resources were not matched",
                        }
                    )
                    continue
                delta: dict[str, Any] = {
                    "case_id": case_id,
                    "comparison": f"{left_name}-{right_name}",
                }
                for key in keys:
                    left = variants[left_name]["metrics"].get(key)
                    right = variants[right_name]["metrics"].get(key)
                    delta[f"{key}_delta"] = (
                        float(left) - float(right)
                        if left is not None and right is not None
                        else None
                    )
                delta["graph_guided_decision_change_rate"] = (
                    1.0
                    if variants[left_name].get("final_novelty_direction")
                    != variants[right_name].get("final_novelty_direction")
                    else 0.0
                )
                deltas.append(delta)
        self._write_jsonl("graph_ablation_results.jsonl", rows)
        harm_rows = [
            row
            for row in deltas
            if row["comparison"] == "topology-wrong_paper_topology"
        ]
        harm_flags = [
            (row.get("unsupported_major_count_delta") or 0) > 0
            or any(
                row.get(key) is not None and row[key] < 0
                for key in (
                    "analytical_quality_delta",
                    "major_support_precision_delta",
                    "novelty_reasoning_soft_f1_delta",
                )
            )
            for row in harm_rows
        ]
        topology_deltas = [
            row for row in deltas if row["comparison"] == "topology-neutral"
        ]
        usefulness = None
        if topology_deltas:
            usefulness = all(
                (
                    (row.get("relation_count_delta") or 0) > 0
                    or (row.get("novelty_reasoning_soft_f1_delta") or 0) > 0
                    or (row.get("major_support_precision_delta") or 0) > 0
                )
                and (row.get("unsupported_major_count_delta") or 0) <= 0
                and (row.get("post_cutoff_leakage_rate_delta") or 0) <= 0
                and (row.get("graph_semantic_violation_delta") or 0) <= 0
                for row in topology_deltas
            )
        preference_rows: list[dict[str, Any]] = []
        for case in self.manifest.cases:
            preference = decisions.get(case.case_id, {}).get("ablation/preference")
            if preference is None:
                continue
            topology_label = "A" if _topology_is_a(case.case_id) else "B"
            score = (
                0.5
                if preference.preferred == "TIE"
                else 1.0 if preference.preferred == topology_label else 0.0
            )
            preference_rows.append(
                {
                    "case_id": case.case_id,
                    "topology_position": topology_label,
                    "preferred": preference.preferred,
                    "topology_score": score,
                    "confidence": preference.confidence,
                }
            )
        self._write_json(
            "graph_ablation_metrics.json",
            {
                "comparisons": deltas,
                "aggregate": _aggregate_graph_deltas(
                    deltas,
                    samples=self.manifest.bootstrap_samples,
                    seed=self.manifest.seed,
                ),
                "latency_delta_cache_order_confounded": True,
                "wrong_paper_topology_harm_rate": (
                    sum(harm_flags) / len(harm_flags) if harm_flags else None
                ),
                "blind_review_preference_score": (
                    sum(row["topology_score"] for row in preference_rows)
                    if preference_rows
                    else None
                ),
                "blind_review_preference_macro": (
                    sum(row["topology_score"] for row in preference_rows)
                    / len(preference_rows)
                    if preference_rows
                    else None
                ),
                "blind_review_preferences": preference_rows,
                "graph_usefulness_criterion_met": usefulness,
                "graph_guidance_acceptance_met": any(
                    row["variant"] == "topology"
                    and (row["metrics"].get("graph_seed_fetch_rate") or 0) > 0
                    and (row["metrics"].get("graph_seed_verified_relation_yield") or 0)
                    > 0
                    for row in rows
                ),
            },
        )
        judged_cases = {
            row["case_id"]
            for row in rows
            if row["metrics"].get("analytical_quality") is not None
        }
        not_measured.extend(
            {
                "paper_id": case.paper_id,
                "metric": "graph_ablation_judge_quality_delta",
                "reason": "ablation judge decisions not generated",
            }
            for case in self.manifest.cases
            if case.case_id not in judged_cases
        )
        preference_cases = {row["case_id"] for row in preference_rows}
        not_measured.extend(
            {
                "paper_id": case.paper_id,
                "metric": "blind_review_preference",
                "reason": "pairwise preference judge not generated",
            }
            for case in self.manifest.cases
            if case.case_id not in preference_cases
        )
        self._write_json("not_measured.json", not_measured)

    def _run_case(self, case: Any, graph: GraphRuntimePacket, target: Path) -> Any:
        if self.resume and (target / "review_bundle.json").is_file():
            bundle = load_review_bundle(target)
            if not _current_guidance_bundle(bundle):
                raise ValueError(
                    f"cached run predates GraphRuntimePacket/GuidancePlan: {target}"
                )
            failures = EvidenceStore(target).validate_manifest()
            if failures or not bundle.verification.passed:
                raise ValueError(f"cached run failed validation: {target}: {failures}")
            return bundle
        metadata = PaperMetadata.model_validate_json(case.metadata_path.read_text())
        request = ReviewRequest(
            paper_path=case.manuscript_path,
            metadata=metadata,
            evaluation_date=case.cutoff_date,
        )
        store = EvidenceStore(target)
        services = ServiceRegistry(
            evidence_store=store, graph_scorer=StaticGraphScorer(graph)
        )
        return review_paper(
            request,
            output_dir=target,
            config=load_config(self.runtime_config_path),
            services=services,
        )

    def _run_fault_case(
        self, case: Any, clean_bundle: Any, kind: str, target: Path
    ) -> Any:
        from .faults import (
            EmptyPriorArt,
            FaultGraphScorer,
            FixedAgentReviewer,
            FixedPaperCompiler,
            FixedPaperExtractor,
            FixedQwenReviewer,
            UnusedRelationClassifier,
            semantic_checker_for_fault,
        )

        if self.resume and (target / "review_bundle.json").is_file():
            return load_review_bundle(target)
        metadata = PaperMetadata.model_validate_json(case.metadata_path.read_text())
        request = ReviewRequest(
            paper_path=case.manuscript_path,
            metadata=metadata,
            evaluation_date=case.cutoff_date,
        )
        branch = clean_bundle.agent_review
        if branch is None:
            raise ValueError("clean bundle has no Agent branch for fault injection")
        if kind == "agent_invalid":
            branch = branch.model_copy(
                update={"failures": ["agent_reviewer_unavailable_or_invalid"]}
            )
        clean_store = EvidenceStore(Path(self._clean_run_map()[case.case_id]))
        branch_record = clean_store.get("A:BRANCH")
        payload = (
            dict(branch_record.payload.get("input_payload") or {})
            if branch_record is not None
            else {}
        )
        config = load_config(
            overrides={
                "aspr_qwen": {
                    "enabled": kind == "qwen_required_missing",
                    "required": kind == "qwen_required_missing",
                }
            }
        )
        prior = EmptyPriorArt(failed=kind == "retrieval_exception")
        services = ServiceRegistry(
            evidence_store=EvidenceStore(target),
            paper_compiler=FixedPaperCompiler(clean_bundle.paper_ir),
            paper_extractor=FixedPaperExtractor(),
            graph_scorer=FaultGraphScorer(case.graph_result, kind),
            agent_reviewer=FixedAgentReviewer(branch, payload),
            qwen_reviewer=FixedQwenReviewer(),
            prior_art=prior,
            relation_classifier=UnusedRelationClassifier(),
            verifier=ReviewVerifier(
                config, semantic_checker=semantic_checker_for_fault(kind)
            ),
        )
        return review_paper(
            request, output_dir=target, config=config, services=services
        )

    def _clean_run_map(self) -> dict[str, Path]:
        path = self.output_dir / "clean_runs.json"
        if not path.is_file():
            raise FileNotFoundError("run-clean must complete first")
        return {
            row["case_id"]: Path(row["run_dir"]) for row in json.loads(path.read_text())
        }

    def _judge_map(self) -> dict[str, dict[str, Any]]:
        from experiments.gear.review_reconstruction.evaluation import MatchJudgeResponse

        from .contracts import (
            BlindReviewPreferenceV1,
            PointSupportResponseV1,
            RevisionIssueMatchResponseV1,
            RubricScoreResponseV1,
        )

        models: dict[str, Any] = {
            "rubric": RubricScoreResponseV1,
            "support": PointSupportResponseV1,
            "matches": MatchJudgeResponse,
            "revision": RevisionIssueMatchResponseV1,
            "wrong_paper_support": PointSupportResponseV1,
            "preference": BlindReviewPreferenceV1,
        }
        output: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in _jsonl(self.output_dir / "judge_decisions.jsonl"):
            kind = row["kind"]
            model_key = kind.rsplit("/", 1)[-1]
            output[row["case_id"]][kind] = models[model_key].model_validate(
                row["decision"]
            )
        return output

    def _aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        numeric = sorted(
            {
                key
                for row in rows
                for key, value in row.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        macros = {key: macro([row.get(key) for row in rows]) for key in numeric}
        cis = {
            key: bootstrap_ci(
                [row.get(key) for row in rows],
                samples=self.manifest.bootstrap_samples,
                seed=self.manifest.seed,
            )
            for key in numeric
        }
        return {
            "contract": "gear_evaluation_metrics_v1",
            "dataset_id": self.manifest.dataset_id,
            "paper_count": len(rows),
            "development_non_confirmatory": self.manifest.development_non_confirmatory,
            "paper_macro": macros,
            "paper_cluster_bootstrap_95_ci": cis,
            "composite_score": None,
        }

    def _run_stage(self, stage: str, handler: Callable[[], None]) -> None:
        checkpoint = self.output_dir / "checkpoints" / f"{stage}.json"
        stage_fingerprint = self._stage_fingerprint(stage)
        if self.resume and checkpoint.is_file():
            payload = json.loads(checkpoint.read_text())
            if payload.get("fingerprint") == stage_fingerprint:
                return
        started = time.time()
        handler()
        stage_fingerprint = self._stage_fingerprint(stage)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "fingerprint": stage_fingerprint,
                    "completed_at_epoch": time.time(),
                    "wall_time_seconds": time.time() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def _stage_fingerprint(self, stage: str) -> str:
        dependency_names = {
            "prepare-judges": ("clean_runs.json",),
            "run-judges": ("clean_runs.json", "graph_ablation_results.jsonl"),
            "score": (
                "clean_runs.json",
                "judge_decisions.jsonl",
                "fault_results.jsonl",
                "graph_ablation_results.jsonl",
            ),
            "report": (
                "metrics.json",
                "not_measured.json",
                "graph_ablation_metrics.json",
            ),
        }
        dependencies = {}
        for name in dependency_names.get(stage, ()):
            path = self.output_dir / name
            dependencies[name] = sha256_file(path) if path.is_file() else None
        return sha256_value(
            {"run_fingerprint": self.fingerprint, "dependencies": dependencies}
        )

    def _fingerprint(self) -> str:
        code = _git_fingerprint()
        return sha256_value(
            {
                "contract": "gear_evaluation_checkpoint_v1",
                "manifest": sha256_file(self.manifest_path),
                "judge_config": sha256_file(self.judge_config_path),
                "runtime_config": (
                    sha256_file(self.runtime_config_path)
                    if self.runtime_config_path is not None
                    else None
                ),
                "code": code,
            }
        )

    def _write_run_manifest(self) -> None:
        self._write_json(
            "run_manifest.json",
            {
                "contract": "gear_evaluation_run_manifest_v1",
                "fingerprint": self.fingerprint,
                "dataset_id": self.manifest.dataset_id,
                "development_non_confirmatory": self.manifest.development_non_confirmatory,
                "manifest_path": str(self.manifest_path),
                "judge_config_path": str(self.judge_config_path),
                "outputs": sorted(path.name for path in self.output_dir.iterdir()),
            },
        )

    def _write_json(self, name: str, value: Any) -> None:
        path = self.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(path)

    def _write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        path = self.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(canonical_json(row) + "\n" for row in rows))
        temporary.replace(path)


def _git_fingerprint() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], capture_output=True, text=True, check=False
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    untracked_hashes = {
        path: sha256_file(Path(path)) for path in untracked if Path(path).is_file()
    }
    return {
        "head": head,
        "diff_sha256": sha256_value(diff),
        "untracked": untracked_hashes,
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _evidence_payloads(run_dir: Path) -> dict[str, dict[str, Any]]:
    store = EvidenceStore(run_dir)
    return {
        key: dict(record.payload)
        for key in store.ids()
        if (record := store.get(key)) is not None
    }


def _routing_pool_sha256(state: Any, pool: str) -> str:
    rows = sorted(
        (claim_id, candidate.candidate_id, candidate.pool_rank)
        for claim_id, plan in state.retrieval_routing_plans.items()
        for candidate in plan.candidates
        if candidate.pool == pool
    )
    return sha256_value(rows)


def _decision_rows(
    case: Any, values: dict[str, Any], *, prefix: str = ""
) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "paper_id": case.paper_id,
            "kind": prefix + kind,
            "decision": value.model_dump(mode="json"),
        }
        for kind, value in values.items()
        if value is not None
    ]


def _render_metric(value: Any) -> str:
    return "not measured" if value is None else f"{float(value):.4f}"


def _current_guidance_bundle(bundle: Any) -> bool:
    state = getattr(bundle, "state", None)
    graph = getattr(bundle, "graph_result", None)
    return bool(
        state is not None
        and state.graph_guidance_plan is not None
        and state.graph_guidance_plan.policy_version == GRAPH_GUIDANCE_POLICY_VERSION
        and graph is not None
        and graph.contract == "aspr_graph_runtime_packet_v1"
    )


def _topology_is_a(case_id: str) -> bool:
    return int(sha256_value(case_id).removeprefix("sha256:")[-1], 16) % 2 == 0


__all__ = ["STAGES", "EvaluationRunner", "StaticGraphScorer"]
