"""运行一轮 Fig.1-Fig.10 Nature 投稿级审计。"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "outputs" / "kg_perturbation_final_assembly" / "fig1_fig10_nature_submission_iterative_audit_workflow.md"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "outputs" / "nature_submission_audit"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    kind: str
    evidence: str
    support: str
    required_action: str


@dataclass(frozen=True)
class Risk:
    issue_id: str
    severity: str
    category: str
    issue: str
    evidence: str
    action: str
    disposition: str


@dataclass(frozen=True)
class FigureSpec:
    fig_id: str
    label: str
    title: str
    image_path: str
    figure_class: str
    role: str
    caption_draft: str
    inputs: Sequence[str]
    claims: Sequence[Claim]
    risks: Sequence[Risk]
    reviewer_objections: Mapping[str, Sequence[str]]
    readiness: Mapping[str, float]
    caption_edit: str


def claim(claim_id: str, text: str, kind: str, evidence: str, support: str, action: str) -> Claim:
    return Claim(claim_id, text, kind, evidence, support, action)


def risk(
    issue_id: str,
    severity: str,
    category: str,
    issue: str,
    evidence: str,
    action: str,
    disposition: str,
) -> Risk:
    return Risk(issue_id, severity, category, issue, evidence, action, disposition)


FIGURES: Sequence[FigureSpec] = [
    FigureSpec(
        fig_id="fig1",
        label="Fig.1",
        title="Knowledge-graph perturbation measurement",
        image_path="outputs/redraw_v6a_best_fig1/fig1_multi_domain_real.png",
        figure_class="data-first evidence figure",
        role="定义 graph-perturbation 的可观察结构信号，为 Fig.2-Fig.3 指标化做铺垫。",
        caption_draft="Multi-domain graph-perturbation maps define structural signals used throughout the study.",
        inputs=[
            "outputs/redraw_v6a_best_fig1/figure_quality_report.json",
            "outputs/redraw_v6a_best_fig1/crispr/works_selected.csv",
            "outputs/redraw_v6a_best_fig1/crispr/topic_nodes.csv",
            "outputs/redraw_v6a_best_fig1/crispr/topic_edges.csv",
            "outputs/redraw_v6a_best_fig1/crispr/paper_edges.csv",
            "outputs/redraw_v6a_best_fig1/graphene_2d_materials/works_selected.csv",
            "outputs/redraw_v6a_best_fig1/ipsc_reprogramming/works_selected.csv",
        ],
        claims=[
            claim("fig1_c1", "citation/topic snapshots expose measurable graph perturbations", "methodological", "works/topic/paper edge tables plus quality report", "directly_supported", "keep"),
            claim("fig1_c2", "selected domains illustrate recurring bridge and diffusion structures", "descriptive", "domain panel files in redraw_v6a_best_fig1", "supported_with_assumptions", "caption should say illustrate, not prove universality"),
            claim("fig1_c3", "Fig.1 motivates the later seven-indicator score", "methodological", "Fig.2/Fig.3 downstream metric files", "supported_with_assumptions", "ensure transition text names indicators"),
        ],
        risks=[
            risk("fig1_r1", "P2_moderate", "data", "domain examples may be read as cherry-picked universal proof", "limited visible domains in main figure", "soften to representative examples and keep domain selection in supplement", "retained_with_explicit_caveat"),
            risk("fig1_r2", "P2_moderate", "visual", "dense graph snapshots can look qualitative if metrics are not cross-referenced", "visual maps dominate first figure", "caption must point to Fig.2/Fig.3 metric validation", "retained_with_explicit_caveat"),
        ],
        reviewer_objections={
            "quant_methods": ["How were domains and landmarks selected, and could future recognition leak into the graph snapshot?"],
            "domain_science": ["Do the shown graph structures correspond to real scientific mechanism changes or only citation topology?"],
            "ai_systems": ["How does this measurement layer later become usable evidence for ASPR rather than a decorative graph?"],
        },
        readiness={"data_provenance": 4, "reproducibility": 4, "statistical_credibility": 3, "visual_clarity": 4, "narrative_necessity": 5, "claim_alignment": 4, "reviewer_readiness": 3, "supplement_readiness": 3},
        caption_edit="把 Fig.1 claim 限定为 observable graph-perturbation examples，并显式指向 Fig.2/Fig.3 的定量验证。",
    ),
    FigureSpec(
        fig_id="fig2",
        label="Fig.2",
        title="Empirical indicator validation",
        image_path="outputs/redraw_v6a_best_fig2/fig2_empirical_full.png",
        figure_class="rigorous data figure",
        role="验证 publication-day perturbation indicators 是否具有经验信号。",
        caption_draft="Empirical panels test whether graph indicators align with future scientific signal.",
        inputs=[
            "outputs/redraw_v6a_best_fig2/fig2_input_audit.csv",
            "outputs/redraw_v6a_best_fig2/fig2_candidate_metrics.csv",
            "outputs/redraw_v6a_best_fig2/fig2_indicator_future_corr.csv",
            "outputs/redraw_v6a_best_fig2/fig2_indicator_future_corr_bootstrap.csv",
            "outputs/redraw_v6a_best_fig2/fig2_reference_closure_report.csv",
            "outputs/redraw_v6a_best_fig2/fig2_matched_controls.csv",
            "outputs/redraw_v6a_best_fig2/figure_quality_report.json",
        ],
        claims=[
            claim("fig2_c1", "publication-day indicators contain future-outcome signal", "inferential", "indicator_future_corr and bootstrap tables", "directly_supported", "keep with effect sizes"),
            claim("fig2_c2", "matched controls reduce field/year artifact risk", "methodological", "matched_controls table", "supported_with_assumptions", "report control-tier limitations"),
            claim("fig2_c3", "reference closure is adequate for main interpretation", "methodological", "reference_closure_report and quality gates", "supported_with_assumptions", "flag unmeasured closure cells"),
        ],
        risks=[
            risk("fig2_r1", "P1_major", "statistics", "future outcome could remain citation-biased", "future correlation files rely on realized downstream outcomes", "add or cite non-citation robustness and keep language as association", "retained_with_explicit_caveat"),
            risk("fig2_r2", "P2_moderate", "data", "reference closure may be incomplete in sparse domains", "reference_closure_report required for audit", "surface closure status in supplement", "retained_with_explicit_caveat"),
        ],
        reviewer_objections={
            "quant_methods": ["Are correlations robust after matched controls, multiple comparisons, and field/year stratification?"],
            "domain_science": ["Do future outcomes represent scientific importance or later popularity?"],
            "ai_systems": ["Which of these indicators later enter the ASPR graph agent, and are definitions unchanged?"],
        },
        readiness={"data_provenance": 4, "reproducibility": 4, "statistical_credibility": 3, "visual_clarity": 4, "narrative_necessity": 5, "claim_alignment": 4, "reviewer_readiness": 3, "supplement_readiness": 4},
        caption_edit="把 prediction 相关表述改为 controlled association / future signal，避免写成 deterministic forecasting。",
    ),
    FigureSpec(
        fig_id="fig3",
        label="Fig.3",
        title="Learned perturbation score",
        image_path="outputs/redraw_v6a_best_fig3/fig3_selected_weight_learning_full.png",
        figure_class="rigorous model-validation figure",
        role="学习并验证多指标 perturbation score，作为后续 Fig.4-Fig.7/Fig.10 的量化主线。",
        caption_draft="A learned multi-indicator perturbation score combines seven graph indicators and is cross-validated.",
        inputs=[
            "outputs/redraw_v6a_best_fig3/figure_quality_report.json",
            "outputs/redraw_v6a_best_fig3/multi_domain/fig3_score_table.csv",
            "outputs/redraw_v6a_best_fig3/multi_domain/fig3_best_weights.csv",
            "outputs/redraw_v6a_best_fig3/multi_domain/fig3_cv_summary.csv",
            "outputs/redraw_v6a_best_fig3/multi_domain/fig3_baseline_comparison.csv",
            "outputs/redraw_v6a_best_fig3/multi_domain/fig3_fold_weights.csv",
        ],
        claims=[
            claim("fig3_c1", "learned multi-indicator score outperforms simple baselines", "comparative", "cv_summary and baseline_comparison", "directly_supported", "keep with baseline deltas"),
            claim("fig3_c2", "seven indicators jointly contribute to perturbation ranking", "inferential", "best_weights and fold_weights", "supported_with_assumptions", "avoid overinterpreting exact coefficients"),
            claim("fig3_c3", "score is suitable as downstream graph prior", "methodological", "score_table and Fig.4 graph prior usage", "supported_with_assumptions", "keep as prior, not ground truth"),
        ],
        risks=[
            risk("fig3_r1", "P1_major", "statistics", "weight learning may overfit domain composition", "fold weights and CV summaries must be inspected", "require leave-one-domain or fold stability discussion", "retained_with_explicit_caveat"),
            risk("fig3_r2", "P2_moderate", "logic", "exact learned weights may be overinterpreted mechanistically", "weights are optimized for target score", "caption should frame as predictive ensemble", "retained_with_explicit_caveat"),
        ],
        reviewer_objections={
            "quant_methods": ["Are folds grouped so that one field or year does not leak target structure into validation?"],
            "domain_science": ["Can the learned score be interpreted scientifically, or is it only a ranking proxy?"],
            "ai_systems": ["Does ASPR use this score as evidence with uncertainty, or as an unquestioned oracle?"],
        },
        readiness={"data_provenance": 4, "reproducibility": 4, "statistical_credibility": 3, "visual_clarity": 4, "narrative_necessity": 5, "claim_alignment": 4, "reviewer_readiness": 3, "supplement_readiness": 4},
        caption_edit="强调 score 是 learned graph prior，不是 innovation ground truth；保留 baseline 和 fold-stability 证据。",
    ),
    FigureSpec(
        fig_id="fig4",
        label="Fig.4",
        title="Peer-review validation sample",
        image_path="outputs/kg_perturbation_fig4_full50/fig4_full.png",
        figure_class="peer-review validation figure",
        role="把 graph evidence 与透明 peer review 中的人类判断建立联系。",
        caption_draft="Peer-review validation links graph-derived innovation signals to human reviewer judgments.",
        inputs=[
            "outputs/kg_perturbation_fig4_full50/fig4_metrics_summary.csv",
            "outputs/kg_perturbation_fig4_full50/fig4_input_audit.csv",
            "outputs/kg_perturbation_fig4_full50/fig4_peer_review_screen.csv",
            "outputs/kg_perturbation_fig4_full50/fig4_retrieval_diagnostics.csv",
            "outputs/kg_perturbation_fig4_full50/fig4_aspect_relation_summary.csv",
        ],
        claims=[
            claim("fig4_c1", "ASPR-style graph evidence aligns with some peer-review novelty and rigor judgments", "inferential", "metrics_summary and aspect_relation_summary", "directly_supported", "keep but quantify strict/soft recall"),
            claim("fig4_c2", "no-leakage setup separates manuscript dossier from peer-review target", "methodological", "input_audit and peer_review_screen", "supported_with_assumptions", "keep leakage guard explicit"),
            claim("fig4_c3", "overclaiming and missing peer points are measurable failure modes", "descriptive", "metrics_summary and claim examples", "directly_supported", "use as limitation bridge to Fig.6/Fig.10"),
        ],
        risks=[
            risk("fig4_r1", "P1_major", "credibility", "soft semantic matches may overstate human-review alignment", "semantic claim matching includes strict and soft recall", "report strict recall and no-match examples prominently", "retained_with_explicit_caveat"),
            risk("fig4_r2", "P2_moderate", "visual", "current PNG resolution is lower than most other figures", "fig4 image is smaller than Fig.1-Fig.3", "redraw or export at higher DPI before submission", "fixed_in_future_redraw"),
        ],
        reviewer_objections={
            "quant_methods": ["How reliable are semantic match labels, and what is the false-positive rate?"],
            "domain_science": ["Do matched points reflect substantive reviewer concerns or surface similarity?"],
            "ai_systems": ["Was any peer-review text used by the agent before evaluation?"],
        },
        readiness={"data_provenance": 4, "reproducibility": 4, "statistical_credibility": 3, "visual_clarity": 3, "narrative_necessity": 5, "claim_alignment": 3, "reviewer_readiness": 3, "supplement_readiness": 4},
        caption_edit="将 alignment 写成 partial/structured alignment，并列出 strict recall、soft recall、overclaiming 和 missing-point caveat。",
    ),
    FigureSpec(
        fig_id="fig5",
        label="Fig.5",
        title="Forecast and mechanism handoff",
        image_path="outputs/kg_perturbation_fig5/strict_ai_filtered_image2_handoff/fig5_strict_ai_filtered_image2_generated_preview.png",
        figure_class="visual synthesis figure",
        role="把 graph evidence 转成 forecast/mechanism interpretation，但不能替代定量验证。",
        caption_draft="Forecast and mechanism handoff translates graph-perturbation evidence into forward-looking interpretation.",
        inputs=[
            "outputs/kg_perturbation_fig5/strict_ai_filtered_image2_handoff/fig5_panel_text.json",
            "outputs/kg_perturbation_fig5/strict_ai_filtered_image2_handoff/fig5_image2_prompt.md",
            "outputs/kg_perturbation_fig5/plot_data/derived/fig5_panel_b_top_focus.csv",
            "outputs/kg_perturbation_fig5/plot_data/derived/forecast_focus.csv",
            "outputs/kg_perturbation_fig5/plot_data/base/topic_nodes.csv",
        ],
        claims=[
            claim("fig5_c1", "graph signals can be organized into forecastable research focus areas", "predictive", "forecast_focus and panel_text", "proxy_supported", "label as forecast/handoff"),
            claim("fig5_c2", "mechanism cards summarize auditable graph-derived opportunities", "descriptive", "derived cards and source CSVs", "supported_with_assumptions", "show source trace"),
            claim("fig5_c3", "AI visual synthesis can communicate the forecast layer", "visual", "image2 prompt and generated preview", "illustrative_only", "do not use as numeric evidence"),
        ],
        risks=[
            risk("fig5_r1", "P1_major", "visual", "image-generated synthesis may look more certain than the underlying forecast evidence", "generated preview is primary visual", "add source-note and downgrade to forecast handoff", "retained_with_explicit_caveat"),
            risk("fig5_r2", "P2_moderate", "reproducibility", "image2 rendering is not fully deterministic", "prompt and panel text exist but generated image may not be exactly reproducible", "freeze prompt, seed if possible, and keep source CSVs", "retained_with_explicit_caveat"),
        ],
        reviewer_objections={
            "quant_methods": ["Which forecast claims are quantitatively scored and which are visual synthesis?"],
            "domain_science": ["Are the highlighted mechanisms scientifically plausible or visually curated?"],
            "ai_systems": ["Could image generation introduce unsupported content or alter emphasis?"],
        },
        readiness={"data_provenance": 3, "reproducibility": 2, "statistical_credibility": 2, "visual_clarity": 4, "narrative_necessity": 3, "claim_alignment": 3, "reviewer_readiness": 2, "supplement_readiness": 3},
        caption_edit="明确 Fig.5 是 forecast/mechanism handoff，不是已验证发现；把所有强预测语气改为 candidate/opportunity。",
    ),
    FigureSpec(
        fig_id="fig6",
        label="Fig.6",
        title="Robustness and boundary conditions",
        image_path="outputs/kg_perturbation_fig6/fig6_full.png",
        figure_class="rigorous data figure",
        role="承接 Fig.1-Fig.5，展示方法在噪声、领域、规模、时间窗口、建模选择下的边界。",
        caption_draft="Robustness and boundary-condition analysis tests credibility of graph-perturbation analysis.",
        inputs=[
            "outputs/kg_perturbation_fig6/fig6_cross_domain_reproducibility.csv",
            "outputs/kg_perturbation_fig6/fig6_data_quality_perturbation.csv",
            "outputs/kg_perturbation_fig6/fig6_volume_sensitivity.csv",
            "outputs/kg_perturbation_fig6/fig6_temporal_window_sensitivity.csv",
            "outputs/kg_perturbation_fig6/fig6_modeling_choice_reproducibility.csv",
            "outputs/kg_perturbation_fig6/fig6_failure_modes.csv",
            "outputs/kg_perturbation_fig6/fig6_cache_graph_perturbation.csv",
            "outputs/kg_perturbation_fig6/fig6_source_audit.csv",
            "outputs/kg_perturbation_fig6/fig6_caption.md",
            "outputs/kg_perturbation_fig6/figure_quality_report.json",
        ],
        claims=[
            claim("fig6_c1", "graph-perturbation analysis is robust across several cached stress tests", "inferential", "Fig.6 robustness CSVs plus cache-level indicator rerun audit", "proxy_supported", "explicitly label proxy panels and cache-level partial upgrade"),
            claim("fig6_c2", "failure modes can be categorized and used as safeguards", "methodological", "failure_modes and failure_mode_cases", "supported_with_assumptions", "keep taxonomy as heuristic if not human-labeled"),
            claim("fig6_c3", "method boundaries are visible rather than hidden", "methodological", "caption and panel review", "directly_supported", "keep as credibility bridge"),
        ],
        risks=[
            risk("fig6_r1", "P1_major", "credibility", "Panels B-D are proxy probes from cached score tables; Panel G adds cache-level indicator reruns but not online graph extraction", "fig6_caption.md, fig6_cache_graph_perturbation.csv, and figure_quality_report.json", "retain proxy labels, state cache-level partial upgrade, and keep full-rerun gate blocked", "pipeline_ready_gap"),
            risk("fig6_r2", "P2_moderate", "statistics", "failure taxonomy may be heuristic rather than independently validated", "failure_modes source status", "mark as failure taxonomy, not estimated population rate", "retained_with_explicit_caveat"),
        ],
        reviewer_objections={
            "quant_methods": ["Do proxy perturbation probes reproduce full graph-extraction perturbation results?"],
            "domain_science": ["Are failure modes scientifically meaningful or just metric artifacts?"],
            "ai_systems": ["Which robustness failures trigger ASPR verifier or human-in-loop safeguards?"],
        },
        readiness={"data_provenance": 3, "reproducibility": 4, "statistical_credibility": 3, "visual_clarity": 4, "narrative_necessity": 5, "claim_alignment": 3, "reviewer_readiness": 3, "supplement_readiness": 4},
        caption_edit="把 robustness 写成 cached/proxy stress tests，并保留 full rerun 作为 pipeline-ready gap。",
    ),
    FigureSpec(
        fig_id="fig7",
        label="Fig.7",
        title="Venue-level innovation contribution",
        image_path="outputs/kg_perturbation_fig7/fig7_full.png",
        figure_class="rigorous data figure",
        role="从方法可信性转向 venue-level scientific interpretation。",
        caption_draft="Venue-level contribution analysis compares field-year normalized venue families.",
        inputs=[
            "outputs/kg_perturbation_fig7/fig7_vci_rankings.csv",
            "outputs/kg_perturbation_fig7/fig7_topk_enrichment.csv",
            "outputs/kg_perturbation_fig7/fig7_venue_portfolio.csv",
            "outputs/kg_perturbation_fig7/fig7_mechanism_signature.csv",
            "outputs/kg_perturbation_fig7/fig7_confounder_audit.csv",
            "outputs/kg_perturbation_fig7/fig7_metric_sensitivity.csv",
            "outputs/kg_perturbation_fig7/fig7_pairwise_contribution_tests.csv",
            "outputs/kg_perturbation_fig7/fig7_venue_family_mapping_audit.csv",
            "outputs/kg_perturbation_fig7/figure_quality_report.json",
        ],
        claims=[
            claim("fig7_c1", "Nature Portfolio has the strongest point-estimate venue contribution in the current corpus", "comparative", "vci_rankings and quality_report", "supported_with_assumptions", "retain interval caveat"),
            claim("fig7_c2", "venue-level patterns remain after audited controls", "inferential", "confounder_audit", "supported_with_assumptions", "avoid causal wording"),
            claim("fig7_c3", "mechanism signatures differ by venue family", "descriptive", "mechanism_signature heatmap data", "directly_supported", "keep as association"),
        ],
        risks=[
            risk("fig7_r1", "P1_major", "statistics", "strict interval separation for Nature dominance remains caveated", "figure_quality_report strict_interval_separation gate", "write point-estimate supported with interval caveat", "retained_with_explicit_caveat"),
            risk("fig7_r2", "P1_major", "logic", "venue-level contribution may be misread as causal venue superiority", "caption/headline risk", "replace causal language with association/contribution", "caption_caveat_only"),
            risk("fig7_r3", "P1_major", "statistics", "pairwise aggregate VCI difference versus runner-up remains uncertain", "fig7_pairwise_contribution_tests aggregate_diff_ci_low", "state winner-probability audit and avoid strict dominance language", "retained_with_explicit_caveat"),
        ],
        reviewer_objections={
            "quant_methods": ["Does Nature Portfolio remain top after field/year, article type, team size, references, and OA controls?"],
            "domain_science": ["Is venue contribution scientifically meaningful or a prestige/citation artifact?"],
            "ai_systems": ["Does Fig.7 feed ASPR, or is it a separate sociological result?"],
        },
        readiness={"data_provenance": 4, "reproducibility": 4, "statistical_credibility": 3, "visual_clarity": 4, "narrative_necessity": 3, "claim_alignment": 3, "reviewer_readiness": 3, "supplement_readiness": 4},
        caption_edit="保留 Nature Portfolio point-estimate 结论，但在 caption 中显式写 strict interval separation 和 pairwise difference caveat，禁止因果或严格支配措辞。",
    ),
    FigureSpec(
        fig_id="fig8",
        label="Fig.8",
        title="ASPR algorithm framework",
        image_path="outputs/kg_perturbation_fig8/fig8_full.png",
        figure_class="algorithm framework figure",
        role="提出最终方法 ASPR：graph agent + ASPR-Qwen + fusion/verifier。",
        caption_draft="ASPR is introduced as a dual-path reviewer combining graph evidence and ASPR-Qwen.",
        inputs=[
            "outputs/kg_perturbation_fig8/panel_text.json",
            "outputs/kg_perturbation_fig8/flow_spec.json",
            "outputs/kg_perturbation_fig8/image2_prompt.md",
            "outputs/kg_perturbation_fig8/render_fig8.py",
        ],
        claims=[
            claim("fig8_c1", "ASPR is a dual-path system, not a single LLM", "algorithmic", "flow_spec and rendered framework", "directly_supported", "keep"),
            claim("fig8_c2", "ASPR-Qwen is the reviewer-style model lane trained from paper-review pairs", "algorithmic", "panel_text and flow_spec", "supported_with_assumptions", "label checkpoint status if not final"),
            claim("fig8_c3", "fusion/verifier grounds final review output", "algorithmic", "flow_spec final module", "directly_supported", "align with Fig.9/Fig.10"),
        ],
        risks=[
            risk("fig8_r1", "P2_moderate", "credibility", "architecture diagram may imply ASPR-Qwen checkpoint is fully validated", "Fig.9/Fig.10 mark assumed or LLM-judge layers", "caption should separate architecture from empirical validation", "retained_with_explicit_caveat"),
            risk("fig8_r2", "P3_minor", "visual", "algorithm framework may be too small in manuscript print size", "contact sheet readability", "check final layout at journal column width", "fixed_in_future_redraw"),
        ],
        reviewer_objections={
            "quant_methods": ["Which components are evaluated in data figures versus merely defined architecturally?"],
            "domain_science": ["Does the system preserve scientific nuance or only procedural review structure?"],
            "ai_systems": ["Where are hallucination checks, evidence grounding, and human-in-loop safeguards enforced?"],
        },
        readiness={"data_provenance": 3, "reproducibility": 3, "statistical_credibility": 2, "visual_clarity": 4, "narrative_necessity": 4, "claim_alignment": 4, "reviewer_readiness": 3, "supplement_readiness": 3},
        caption_edit="在 caption 中写清这是 architecture figure，不是 performance evidence；ASPR-Qwen 状态由 Fig.9/Fig.10 caveat 管理。",
    ),
    FigureSpec(
        fig_id="fig9",
        label="Fig.9",
        title="End-to-end ASPR case run",
        image_path="outputs/kg_perturbation_fig9/fig9_full.png",
        figure_class="case storyboard figure",
        role="展示一个真实 Nature Communications case 的 ASPR 流程和 evidence trace。",
        caption_draft="An end-to-end ASPR case run shows a real manuscript flowing through graph evidence and ASPR-Qwen.",
        inputs=[
            "outputs/kg_perturbation_fig9/fig9_case_manifest.csv",
            "outputs/kg_perturbation_fig9/fig9_claim_evidence_trace.csv",
            "outputs/kg_perturbation_fig9/fig9_agent_output.json",
            "outputs/kg_perturbation_fig9/fig9_aspr_qwen_output.json",
            "outputs/kg_perturbation_fig9/fig9_assumed_aspr_qwen_output.json",
            "outputs/kg_perturbation_fig9/fig9_fusion_output.json",
            "outputs/kg_perturbation_fig9/fig9_quality_report.json",
        ],
        claims=[
            claim("fig9_c1", "case uses a real Nature Communications manuscript and peer-review file", "descriptive", "case_manifest and quality_report", "directly_supported", "keep"),
            claim("fig9_c2", "claim-evidence trace anchors final review to manuscript and peer review lines", "methodological", "claim_evidence_trace", "directly_supported", "keep"),
            claim("fig9_c3", "ASPR-Qwen lane demonstrates reviewer-style drafting", "algorithmic", "assumed_aspr_qwen_output", "pipeline_ready", "label assumed pipeline-ready"),
        ],
        risks=[
            risk("fig9_r1", "P1_major", "credibility", "ASPR-Qwen output is assumed rather than checkpoint-generated", "fig9_quality_report notes", "keep explicit label and avoid using case as proof of trained checkpoint", "pipeline_ready_gap"),
            risk("fig9_r2", "P2_moderate", "data", "single case may be cherry-picked", "case storyboard only", "state as auditable example, not representative validation", "caption_caveat_only"),
        ],
        reviewer_objections={
            "quant_methods": ["Why this case, and how representative is it?"],
            "domain_science": ["Does the final review correctly reflect the real peer-review concerns and biological mechanism caveats?"],
            "ai_systems": ["Is the ASPR-Qwen output from an actual checkpoint or an assumed placeholder?"],
        },
        readiness={"data_provenance": 3, "reproducibility": 3, "statistical_credibility": 2, "visual_clarity": 4, "narrative_necessity": 4, "claim_alignment": 3, "reviewer_readiness": 2, "supplement_readiness": 3},
        caption_edit="将 Fig.9 定位为 auditable case run；ASPR-Qwen lane 必须写 assumed pipeline-ready，不能作为 checkpoint performance proof。",
    ),
    FigureSpec(
        fig_id="fig10",
        label="Fig.10",
        title="ASPR module ablation and reinforcement",
        image_path="outputs/kg_perturbation_fig10/fig10_full.png",
        figure_class="data and ablation figure",
        role="证明 ASPR 质量来自 graph agent、ASPR-Qwen、retrieval、trace、fusion、verifier 等模块组合。",
        caption_draft="Ablation and reinforcement analysis tests whether ASPR quality comes from module composition.",
        inputs=[
            "outputs/kg_perturbation_fig10/fig10_ablation_results.csv",
            "outputs/kg_perturbation_fig10/fig10_ablation_forest.csv",
            "outputs/kg_perturbation_fig10/fig10_error_taxonomy.csv",
            "outputs/kg_perturbation_fig10/fig10_human_preference_llm_judge_results.csv",
            "outputs/kg_perturbation_fig10/fig10_reinforcement_results.csv",
            "outputs/kg_perturbation_fig10/fig10_generic_llm_baseline_results.csv",
            "outputs/kg_perturbation_fig10/fig10_generic_llm_baseline_manifest.json",
            "outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_results.csv",
            "outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_claim_matches.jsonl",
            "outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_exclusions.csv",
            "outputs/kg_perturbation_fig10/fig10_generic_llm_same_rubric_manifest.json",
            "outputs/kg_perturbation_fig10/fig10_evidence_provenance.csv",
            "outputs/kg_perturbation_fig10/fig10_replacement_gates.csv",
            "outputs/kg_perturbation_fig10/fig10_panel_text.json",
            "outputs/kg_perturbation_fig10/figure_quality_report.json",
        ],
        claims=[
            claim("fig10_c1", "full ASPR real Fig.4 metrics outperform ablated variants in the current table", "comparative", "ablation_results and ablation_forest", "proxy_supported", "label ablations as LLM-judge estimates"),
            claim("fig10_c2", "removing modules degrades distinct quality dimensions", "inferential", "metric degradation matrix and error_taxonomy", "proxy_supported", "keep source labels visible"),
            claim("fig10_c3", "human preference is currently LLM-as-judge when human ratings are absent", "methodological", "human_preference_llm_judge_results", "pipeline_ready", "replace with human study for final strong claim"),
            claim("fig10_c4", "Nature strong module-causality claims remain blocked until replacement gates pass", "methodological", "fig10_evidence_provenance and fig10_replacement_gates", "pipeline_ready", "use replacement gates as the manuscript claim boundary"),
            claim("fig10_c5", "same-rubric Fig.4 matcher audit shows the observed qwen3 proxy score is not peer-review semantic evidence", "methodological", "fig10_generic_llm_same_rubric_results, exclusions, and manifest", "directly_supported", "report as completed 48/50 evaluable-case bridge with documented zero-peer-point exclusions, not final superiority evidence"),
        ],
        risks=[
            risk("fig10_r1", "P1_major", "credibility", "ablation rows are pipeline estimates, not true module reruns", "fig10_evidence_provenance.csv and fig10_replacement_gates.csv", "record as pipeline-ready and avoid definitive causal module proof", "pipeline_ready_gap"),
            risk("fig10_r2", "P1_major", "credibility", "preference bars use LLM-as-judge rather than human preference", "fig10_evidence_provenance.csv and fig10_replacement_gates.csv", "label and replace with blinded human ratings when available", "pipeline_ready_gap"),
            risk("fig10_r3", "P1_major", "credibility", "qwen3 proxy baseline exceeds full ASPR under proxy scoring, but the completed same-rubric Fig.4 matcher audit finds near-zero peer-review semantic coverage on 48/50 evaluable cases with two zero-peer-point exclusions documented", "fig10_generic_llm_baseline_results.csv, fig10_generic_llm_same_rubric_results.csv, fig10_generic_llm_same_rubric_exclusions.csv, fig10_evidence_provenance.csv, and fig10_replacement_gates.csv", "report proxy-versus-same-rubric discrepancy with documented exclusions and avoid ASPR-superiority wording", "retained_with_explicit_caveat"),
        ],
        reviewer_objections={
            "quant_methods": ["Were ablations actually rerun, or are deltas assumed?"],
            "domain_science": ["Do module failures correspond to meaningful review quality failures?"],
            "ai_systems": ["Why does the observed qwen3 proxy baseline exceed full ASPR under proxy scoring while the same-rubric matcher shows near-zero peer-review semantic coverage, and did ASPR-Qwen checkpoint outputs run?"],
        },
        readiness={"data_provenance": 3, "reproducibility": 4, "statistical_credibility": 2, "visual_clarity": 4, "narrative_necessity": 4, "claim_alignment": 3, "reviewer_readiness": 2, "supplement_readiness": 3},
        caption_edit="把 Fig.10 写成中文的 pipeline-ready ablation evidence：full ASPR 来自真实 Fig.4；qwen3 generic baseline 已真实运行，proxy composite 一度高于 full ASPR，但已完成的 48/50 可评估样本 same-rubric Fig.4 matcher（另有 2 个 zero-peer-point exclusion）显示其几乎不覆盖真实 peer-review 语义点；因此 Fig.10 只能支持模块诊断，不能写成 ASPR superiority；强 claim 仍等待真实模块重跑、人类盲评和 checkpoint 输出。",
    ),
]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def file_status(path: Path) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "path": str(path),
        "exists": int(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "modified_time": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else "",
        "width_px": "",
        "height_px": "",
        "kind": path.suffix.lower().lstrip("."),
    }
    if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        try:
            with Image.open(path) as image:
                status["width_px"] = int(image.width)
                status["height_px"] = int(image.height)
        except OSError:
            pass
    return status


def summarize_table(path: Path) -> Dict[str, Any]:
    base = file_status(path)
    base.update({"rows": "", "columns": "", "column_sample": "", "read_status": "not_read"})
    if not path.exists():
        base["read_status"] = "missing"
        return base
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            base["rows"] = int(len(df))
            base["columns"] = int(len(df.columns))
            base["column_sample"] = "; ".join(map(str, list(df.columns[:12])))
            base["read_status"] = "ok"
        elif path.suffix.lower() == ".json":
            payload = read_json(path)
            base["rows"] = 1
            base["columns"] = len(payload) if isinstance(payload, Mapping) else ""
            base["column_sample"] = "; ".join(list(payload.keys())[:12]) if isinstance(payload, Mapping) else type(payload).__name__
            base["read_status"] = "ok"
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip()]
            base["rows"] = len(lines)
            base["read_status"] = "ok"
        elif path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            base["rows"] = len(text.splitlines())
            base["columns"] = ""
            base["read_status"] = "ok"
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        base["read_status"] = f"read_error:{exc}"
    return base


def safe_mean(values: Iterable[float]) -> float:
    nums = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(nums) / len(nums) if nums else float("nan")


def severity_weight(severity: str) -> int:
    if severity.startswith("P0"):
        return 4
    if severity.startswith("P1"):
        return 3
    if severity.startswith("P2"):
        return 2
    if severity.startswith("P3"):
        return 1
    return 0


def readiness_status(score: float) -> str:
    if score >= 4.5:
        return "Nature 主图候选"
    if score >= 4.0:
        return "强主图候选"
    if score >= 3.0:
        return "pipeline-ready 但必须保留 caveat"
    if score >= 2.0:
        return "探索性或内部迭代"
    return "暂不适合投稿"


ZH_TEXT: Dict[str, str] = {
    "add or cite non-citation robustness and keep language as association": "补充或引用非引用网络稳健性检验，并把结论限定为相关性。",
    "add source-note and downgrade to forecast handoff": "增加来源说明，并把定位降级为 forecast handoff。",
    "caption must point to Fig.2/Fig.3 metric validation": "caption 必须指向 Fig.2/Fig.3 的指标验证。",
    "caption should frame as predictive ensemble": "caption 应写成预测性集合指标，而不是单一真值。",
    "caption should separate architecture from empirical validation": "caption 需明确区分系统架构和经验证据。",
    "check final layout at journal column width": "按期刊栏宽检查最终排版可读性。",
    "freeze prompt, seed if possible, and keep source CSVs": "冻结 prompt，尽可能固定 seed，并保留来源 CSV。",
    "keep explicit label and avoid using case as proof of trained checkpoint": "保留明确标签，避免把单个案例当作已训练 checkpoint 的性能证明。",
    "label and replace with blinded human ratings when available": "明确标注 LLM-as-judge；有盲评人工评分后替换。",
    "mark as failure taxonomy, not estimated population rate": "标注为失败类型学，而不是总体发生率估计。",
    "record as pipeline-ready and avoid definitive causal module proof": "记录为 pipeline-ready gap，避免写成确定性的模块因果证明。",
    "redraw or export at higher DPI before submission": "投稿前重绘或以更高 DPI 导出。",
    "replace causal language with association/contribution": "把因果语言替换为 association/contribution。",
    "report strict recall and no-match examples prominently": "突出报告 strict recall 和 no-match 例子。",
    "require leave-one-domain or fold stability discussion": "需要 leave-one-domain 或 fold stability 讨论。",
    "retain proxy labels and state cache-level partial upgrade": "保留 proxy 标签，并写明 cache-level 指标重跑只是部分升级。",
    "retain proxy labels, state cache-level partial upgrade, and keep full-rerun gate blocked": "保留 proxy 标签，写明 cache-level 指标重跑只是部分升级，并保持 full-rerun gate 阻断。",
    "retain proxy labels and avoid definitive robustness wording": "保留 proxy 标签，避免确定性的稳健性措辞。",
    "run current generic LLM baseline before comparative performance claim": "在写比较性能 claim 前，先运行当前 generic LLM baseline。",
    "treat observed generic baseline as proxy-scored contradiction and rerun same-rubric comparison": "把已观测的 generic baseline 视为 proxy-scored 反证，并重跑同 rubric 比较。",
    "report proxy-versus-same-rubric discrepancy with documented exclusions and avoid ASPR-superiority wording": "报告 proxy 与 same-rubric 的差异，明确记录排除样本，并避免 ASPR-superiority 措辞。",
    "report as partial 48/50 evaluable-case bridge, not final superiority evidence": "报告为 48/50 可评估样本的部分桥接审计，而不是最终 superiority 证据。",
    "report as completed 48/50 evaluable-case bridge with documented zero-peer-point exclusions, not final superiority evidence": "报告为已完成的 48/50 可评估样本桥接审计，并记录 zero-peer-point exclusions；不能当作最终 superiority 证据。",
    "soften to representative examples and keep domain selection in supplement": "弱化为代表性示例，并把领域选择细节放入 supplement。",
    "state as auditable example, not representative validation": "表述为可审计案例，而不是代表性验证。",
    "state winner-probability audit and avoid strict dominance language": "写明 winner-probability 审计结果，并避免严格支配措辞。",
    "surface closure status in supplement": "在 supplement 中公开 reference closure 状态。",
    "write point-estimate supported with interval caveat": "写成点估计支持，并保留区间 caveat。",
    "Are correlations robust after matched controls, multiple comparisons, and field/year stratification?": "在匹配控制、多重比较和 field/year 分层后，相关性是否仍然稳健？",
    "Are failure modes scientifically meaningful or just metric artifacts?": "这些失败模式有科学含义，还是只是指标伪影？",
    "Are folds grouped so that one field or year does not leak target structure into validation?": "交叉验证 fold 是否按领域或年份分组，避免目标结构泄漏？",
    "Are the highlighted mechanisms scientifically plausible or visually curated?": "高亮机制是否具有科学合理性，还是主要由视觉选择驱动？",
    "Can the learned score be interpreted scientifically, or is it only a ranking proxy?": "学习得到的分数能否作科学解释，还是只是排序代理？",
    "Could image generation introduce unsupported content or alter emphasis?": "图像生成是否引入了无证据内容或改变了强调重点？",
    "Do future outcomes represent scientific importance or later popularity?": "未来结果代表科学重要性，还是后来的流行度？",
    "Do matched points reflect substantive reviewer concerns or surface similarity?": "匹配点反映了实质性审稿意见，还是表层相似？",
    "Do module failures correspond to meaningful review quality failures?": "模块失败是否对应真实有意义的审稿质量失败？",
    "Do proxy perturbation probes reproduce full graph-extraction perturbation results?": "代理扰动探针是否复现完整 graph-extraction 扰动结果？",
    "Do the shown graph structures correspond to real scientific mechanism changes or only citation topology?": "这些图结构对应真实科学机制变化，还是仅为引用拓扑？",
    "Does ASPR use this score as evidence with uncertainty, or as an unquestioned oracle?": "ASPR 是把该分数作为带不确定性的证据，还是当作不受质疑的 oracle？",
    "Does Fig.7 feed ASPR, or is it a separate sociological result?": "Fig.7 是否服务于 ASPR，还是独立的科学社会学结果？",
    "Does Nature Portfolio remain top after field/year, article type, team size, references, and OA controls?": "控制 field/year、文章类型、团队规模、参考文献数量和 OA 后，Nature Portfolio 是否仍然最高？",
    "Does the final review correctly reflect the real peer-review concerns and biological mechanism caveats?": "最终 review 是否正确反映真实审稿关注点和生物机制 caveat？",
    "Does the system preserve scientific nuance or only procedural review structure?": "系统是否保留科学细节，还是只复刻审稿流程结构？",
    "How does this measurement layer later become usable evidence for ASPR rather than a decorative graph?": "这一测量层后续如何成为 ASPR 可用证据，而不是装饰性图谱？",
    "How reliable are semantic match labels, and what is the false-positive rate?": "语义匹配标签有多可靠，false-positive rate 是多少？",
    "How were domains and landmarks selected, and could future recognition leak into the graph snapshot?": "领域和 landmark 如何选择，未来认可是否泄漏进 graph snapshot？",
    "Is generic LLM-only baseline fair and current, are human preferences real, and did ASPR-Qwen checkpoint outputs run?": "generic LLM-only baseline 是否公平且足够新，偏好结果是否来自真实人工评分，ASPR-Qwen 输出是否来自 checkpoint？",
    "Why does the observed qwen3 proxy baseline exceed full ASPR, and did ASPR-Qwen checkpoint outputs run?": "为什么已观测的 qwen3 proxy baseline 高于 full ASPR，ASPR-Qwen 输出是否来自 checkpoint？",
    "Why does the observed qwen3 proxy baseline exceed full ASPR under proxy scoring while the same-rubric matcher shows near-zero peer-review semantic coverage, and did ASPR-Qwen checkpoint outputs run?": "为什么 qwen3 proxy baseline 在 proxy scoring 下高于 full ASPR，但 same-rubric matcher 显示其几乎没有覆盖 peer-review 语义点？ASPR-Qwen 输出是否来自 checkpoint？",
    "Is generic LLM-only baseline fair and current, and are human preferences real?": "generic LLM-only baseline 是否公平且足够新，偏好结果是否来自真实人工评分？",
    "Is the ASPR-Qwen output from an actual checkpoint or an assumed placeholder?": "ASPR-Qwen 输出来自真实 checkpoint，还是 assumed placeholder？",
    "Is venue contribution scientifically meaningful or a prestige/citation artifact?": "venue contribution 是否有科学含义，还是 prestige/citation artifact？",
    "Was any peer-review text used by the agent before evaluation?": "评估前 agent 是否接触过任何 peer-review 文本？",
    "Were ablations actually rerun, or are deltas assumed?": "消融实验是否真实重跑，还是假设 delta？",
    "Where are hallucination checks, evidence grounding, and human-in-loop safeguards enforced?": "hallucination 检查、证据 grounding 和 human-in-loop safeguard 在哪里强制执行？",
    "Which components are evaluated in data figures versus merely defined architecturally?": "哪些组件在数据图中被评估，哪些只是架构定义？",
    "Which forecast claims are quantitatively scored and which are visual synthesis?": "哪些 forecast claim 有定量评分，哪些只是视觉综合？",
    "Which of these indicators later enter the ASPR graph agent, and are definitions unchanged?": "这些指标中哪些进入 ASPR graph agent，定义是否保持不变？",
    "Which robustness failures trigger ASPR verifier or human-in-loop safeguards?": "哪些稳健性失败会触发 ASPR verifier 或 human-in-loop safeguard？",
    "Why this case, and how representative is it?": "为什么选择这个案例，它有多大代表性？",
}


def zh_text(text: str) -> str:
    return ZH_TEXT.get(text, text)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    fields: List[str] = list(fieldnames or [])
    if not fields:
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: Sequence[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": " ".join(cmd), "exit_code": proc.returncode, "output": proc.stdout.strip()}


def build_contact_sheet(out_dir: Path, figure_rows: Sequence[Mapping[str, Any]]) -> Path:
    thumb_w, thumb_h = 460, 300
    pad, label_h = 28, 58
    cols = 2
    rows = math.ceil(len(figure_rows) / cols)
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        label_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    for idx, row in enumerate(figure_rows):
        col = idx % cols
        row_idx = idx // cols
        x = pad + col * (thumb_w + pad)
        y = pad + row_idx * (thumb_h + label_h + pad)
        path = PROJECT_ROOT / str(row["image_path"])
        title = f"{row['label']} | {row['title']}"
        title_lines = textwrap.wrap(title, width=42)
        draw.text((x, y), title_lines[0], fill="#111827", font=title_font)
        if len(title_lines) > 1:
            draw.text((x, y + 20), title_lines[1], fill="#111827", font=label_font)
        draw.text((x, y + 40), str(row["figure_class"]), fill="#475569", font=label_font)
        if path.exists():
            with Image.open(path).convert("RGB") as image:
                image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                image_x = x + (thumb_w - image.width) // 2
                image_y = y + label_h + (thumb_h - image.height) // 2
                sheet.paste(image, (image_x, image_y))
        draw.rectangle((x, y + label_h, x + thumb_w, y + label_h + thumb_h), outline="#CBD5E1", width=1)
    contact_path = out_dir / "contact_sheet.png"
    sheet.save(contact_path)
    return contact_path


def build_snapshot(out_dir: Path, iteration_id: str) -> Dict[str, Any]:
    git_status = run_command(["git", "status", "--short"])
    (out_dir / "git_status.txt").write_text(git_status["output"] + "\n", encoding="utf-8")
    figure_rows: List[Dict[str, Any]] = []
    for fig in FIGURES:
        path = PROJECT_ROOT / fig.image_path
        row = {
            "fig_id": fig.fig_id,
            "label": fig.label,
            "title": fig.title,
            "image_path": fig.image_path,
            "figure_class": fig.figure_class,
            "role": fig.role,
            **file_status(path),
        }
        figure_rows.append(row)
    write_csv(out_dir / "figure_file_index.csv", figure_rows)
    contact_path = build_contact_sheet(out_dir, figure_rows)
    manifest = {
        "iteration_id": iteration_id,
        "created_at": dt.datetime.now().isoformat(),
        "protocol": str(PROTOCOL_PATH),
        "out_dir": str(out_dir),
        "figure_count": len(FIGURES),
        "contact_sheet": str(contact_path),
        "git_status_exit_code": git_status["exit_code"],
    }
    write_json(out_dir / "snapshot_manifest.json", manifest)
    return manifest


def build_intermediate_index(out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fig in FIGURES:
        for input_path in fig.inputs:
            path = PROJECT_ROOT / input_path
            row = summarize_table(path)
            row.update({"fig_id": fig.fig_id, "label": fig.label, "input_path": input_path})
            rows.append(row)
    write_csv(out_dir / "intermediate_data_index.csv", rows)
    return rows


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_per_figure_outputs(out_dir: Path, intermediate_index: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    aggregate: Dict[str, List[Dict[str, Any]]] = {
        "claims": [],
        "visual": [],
        "readiness": [],
        "risks": [],
        "reviewer_matrix": [],
        "caption_edits": [],
        "decision_board": [],
    }
    by_fig: Dict[str, List[Mapping[str, Any]]] = {}
    for row in intermediate_index:
        by_fig.setdefault(str(row["fig_id"]), []).append(row)

    for fig in FIGURES:
        fig_dir = out_dir / fig.fig_id
        ensure_dir(fig_dir)
        image_status = file_status(PROJECT_ROOT / fig.image_path)
        input_rows = by_fig.get(fig.fig_id, [])
        present_inputs = sum(1 for row in input_rows if int(row.get("exists", 0)) == 1)
        missing_inputs = len(input_rows) - present_inputs

        for item in fig.claims:
            aggregate["claims"].append({"fig_id": fig.fig_id, "label": fig.label, **item.__dict__})
        write_csv(fig_dir / f"{fig.fig_id}_claim_to_evidence_map.csv", [{"fig_id": fig.fig_id, "label": fig.label, **item.__dict__} for item in fig.claims])

        data_forensics = [
            f"# {fig.label} 数据取证",
            "",
            f"## 图的角色",
            "",
            fig.role,
            "",
            "## 当前输入状态",
            "",
            f"- 输入文件数：{len(input_rows)}",
            f"- 已存在：{present_inputs}",
            f"- 缺失：{missing_inputs}",
            f"- 主图：`{fig.image_path}`",
            f"- 主图大小：{image_status.get('width_px')} x {image_status.get('height_px')} px，{image_status.get('size_bytes')} bytes",
            "",
            "## 数据源摘要",
            "",
            markdown_table(input_rows, ["input_path", "exists", "rows", "columns", "read_status", "column_sample"]),
            "",
            "## 核心数据风险",
            "",
        ]
        for item in fig.risks:
            if item.category in {"data", "statistics", "credibility", "logic", "reproducibility"}:
                data_forensics.append(f"- **{item.severity} / {item.category}** `{item.issue_id}`：{item.issue}。证据：{item.evidence}。处理：{zh_text(item.action)}")
        if missing_inputs:
            data_forensics.append(f"- **P1_major / reproducibility**: {missing_inputs} 个预期输入缺失，必须确认是否路径过期或不属于当前 canonical run。")
        (fig_dir / f"{fig.fig_id}_data_forensics.md").write_text("\n".join(data_forensics) + "\n", encoding="utf-8")

        visual_rows = [
            {
                "fig_id": fig.fig_id,
                "label": fig.label,
                "image_path": fig.image_path,
                "figure_class": fig.figure_class,
                "width_px": image_status.get("width_px", ""),
                "height_px": image_status.get("height_px", ""),
                "size_bytes": image_status.get("size_bytes", ""),
                "visual_check": "image_exists_and_dimensions_recorded" if image_status.get("exists") else "missing_image",
                "risk": "print_resolution_or_layout_review_required" if fig.fig_id in {"fig4", "fig8"} else "manual_visual_review_required",
                "style_note": "统一使用 Nature 红、graph 蓝、ASPR-Qwen 紫、verifier 橙。",
            }
        ]
        for item in fig.risks:
            if item.category == "visual":
                visual_rows.append(
                    {
                        "fig_id": fig.fig_id,
                        "label": fig.label,
                        "image_path": fig.image_path,
                        "figure_class": fig.figure_class,
                        "width_px": image_status.get("width_px", ""),
                        "height_px": image_status.get("height_px", ""),
                        "size_bytes": image_status.get("size_bytes", ""),
                        "visual_check": item.issue_id,
                        "risk": item.issue,
                        "style_note": zh_text(item.action),
                    }
                )
        write_csv(fig_dir / f"{fig.fig_id}_visual_audit.csv", visual_rows)
        aggregate["visual"].extend(visual_rows)

        readiness_rows = []
        avg_score = safe_mean(fig.readiness.values())
        for dim, score in fig.readiness.items():
            readiness_rows.append(
                {
                    "fig_id": fig.fig_id,
                    "label": fig.label,
                    "dimension": dim,
                    "score_0_5": score,
                    "status": readiness_status(float(score)),
                    "notes": "单项 Nature 投稿准备度评分",
                }
            )
        readiness_rows.append(
            {
                "fig_id": fig.fig_id,
                "label": fig.label,
                "dimension": "overall_mean",
                "score_0_5": round(avg_score, 2),
                "status": readiness_status(avg_score),
                "notes": "八个准备度维度的平均分",
            }
        )
        write_csv(fig_dir / f"{fig.fig_id}_readiness_score.csv", readiness_rows)
        aggregate["readiness"].extend(readiness_rows)

        persona_names = {
            "quant_methods": "定量方法审稿人",
            "domain_science": "领域科学审稿人",
            "ai_systems": "AI/LLM 系统审稿人",
        }
        reviewer_lines = [f"# {fig.label} 审稿人质疑记录", "", f"## 图的作用", "", fig.role, ""]
        for persona, objections in fig.reviewer_objections.items():
            reviewer_lines.extend([f"## {persona_names.get(persona, persona)}", ""])
            for idx, objection in enumerate(objections, start=1):
                matched_risk = max(fig.risks, key=lambda item: severity_weight(item.severity))
                reviewer_lines.append(f"{idx}. **质疑**：{zh_text(objection)}")
                reviewer_lines.append(f"   - 预备回应：{zh_text(matched_risk.action)}")
                reviewer_lines.append(f"   - 当前处理状态：{matched_risk.disposition}")
                aggregate["reviewer_matrix"].append(
                    {
                        "fig_id": fig.fig_id,
                        "label": fig.label,
                        "reviewer_persona": persona,
                        "objection": objection,
                        "severity": matched_risk.severity,
                        "evidence": matched_risk.evidence,
                        "response": matched_risk.action,
                        "disposition": matched_risk.disposition,
                    }
                )
            reviewer_lines.append("")
        (fig_dir / f"{fig.fig_id}_reviewer_objection_notes.md").write_text("\n".join(reviewer_lines), encoding="utf-8")

        risk_rows = [{"fig_id": fig.fig_id, "label": fig.label, **item.__dict__} for item in fig.risks]
        write_csv(fig_dir / f"{fig.fig_id}_data_risk_table.csv", risk_rows)
        aggregate["risks"].extend(risk_rows)

        claim_text = "\n".join([f"- `{item.claim_id}` ({item.support}): {item.text} -> {item.required_action}" for item in fig.claims])
        (fig_dir / f"{fig.fig_id}_claim_inventory.md").write_text(
            f"# {fig.label} claim 清单\n\n{claim_text}\n",
            encoding="utf-8",
        )

        aggregate["caption_edits"].append(
            {
                "fig_id": fig.fig_id,
                "label": fig.label,
                "current_caption_draft": fig.caption_draft,
                "recommended_edit": fig.caption_edit,
                "claim_scope_action": "soften_or_caveat" if any(item.disposition != "fixed_in_main" for item in fig.risks) else "keep",
            }
        )
        for item in fig.risks:
            aggregate["decision_board"].append(
                {
                    "fig_id": fig.fig_id,
                    "label": fig.label,
                    "issue_id": item.issue_id,
                    "severity": item.severity,
                    "lane": decision_lane(item),
                    "issue": item.issue,
                    "required_input": item.evidence,
                    "action": item.action,
                    "verification_command": "python3 experiments/nature_submission_audit/build_nature_submission_audit.py --iteration-id latest",
                    "stop_condition": item.disposition,
                }
            )
    return aggregate


def decision_lane(item: Risk) -> str:
    if item.disposition == "pipeline_ready_gap":
        return "记录为 pipeline-ready gap"
    if item.disposition == "caption_caveat_only" or "caveat" in item.disposition:
        return "仅在 caption 中加 caveat"
    if item.disposition == "moved_to_supplement":
        return "移至 supplement"
    return "当前主图修复"


def write_aggregate_outputs(out_dir: Path, aggregate: Mapping[str, List[Dict[str, Any]]]) -> None:
    write_csv(out_dir / "claim_to_evidence_map.csv", aggregate["claims"])
    write_csv(out_dir / "claim_inventory.csv", aggregate["claims"])
    unsupported = [row for row in aggregate["claims"] if row["support"] in {"unsupported", "overclaimed", "pipeline_ready", "proxy_supported"}]
    write_csv(out_dir / "unsupported_or_overclaimed_claims.csv", unsupported)
    write_csv(out_dir / "visual_layout_audit.csv", aggregate["visual"])
    write_csv(out_dir / "nature_readiness_scorecard.csv", aggregate["readiness"])
    write_csv(out_dir / "methodology_failure_modes.csv", aggregate["risks"])
    write_csv(out_dir / "reviewer_objection_response_matrix.csv", aggregate["reviewer_matrix"])
    write_csv(out_dir / "caption_edits.csv", aggregate["caption_edits"])
    write_csv(out_dir / "iteration_decision_board.csv", aggregate["decision_board"])
    gaps = [row for row in aggregate["decision_board"] if row["lane"] == "记录为 pipeline-ready gap"]
    write_csv(out_dir / "updated_gaps.csv", gaps)
    write_csv(out_dir / "remaining_gaps.csv", gaps)
    resolved = [row for row in aggregate["decision_board"] if row["lane"] != "记录为 pipeline-ready gap"]
    write_csv(out_dir / "resolved_or_caveated_issues.csv", resolved)

    caption_lines = ["# 图注修改建议", ""]
    for row in aggregate["caption_edits"]:
        caption_lines.extend(
            [
                f"## {row['label']}",
                "",
                f"- 当前草稿：{row['current_caption_draft']}",
                f"- 建议修改：{row['recommended_edit']}",
                f"- Claim 范围处理：`{row['claim_scope_action']}`",
                "",
            ]
        )
    (out_dir / "caption_edits.md").write_text("\n".join(caption_lines), encoding="utf-8")

    report_lines = [
        "# Fig.1-Fig.10 Nature 投稿级一轮审计报告",
        "",
        "本目录按照 `fig1_fig10_nature_submission_iterative_audit_workflow.md` 生成。审计重点包括图片结果、中间数据、claim-to-evidence、可信性、复现性、视觉排版、caption 对齐和 Nature 审稿风险。",
        "",
        "## 关键结论",
        "",
        "- Fig.1-Fig.4 构成 graph-perturbation measurement 与 peer-review validation 主证据链，但 Fig.2/Fig.3/Fig.4 需要持续保留 association、fold stability、semantic matching caveats。",
        "- Fig.5 是 visual synthesis/forecast handoff，不应承载强预测结论。",
        "- Fig.6 是重要稳健性桥接图，但部分 panel 是缓存评分表代理，不是完整 graph rerun。",
        "- Fig.7 当前只能写 Nature Portfolio 点估计贡献最高；严格区间分离优势仍需 caveat。",
        "- Fig.8 是系统框架图，不是性能证据。",
        "- Fig.9 是真实 case trace 加 assumed ASPR-Qwen lane，不能当作 checkpoint performance proof。",
        "- Fig.10 支持模块组合叙事和指标敏感性诊断：qwen3 proxy composite 曾高于 full ASPR，但已完成的 48/50 可评估样本 same-rubric Fig.4 matcher（另有 2 个 zero-peer-point exclusion）显示 generic LLM 几乎不覆盖真实 peer-review 语义点；当前仍包含 pipeline-ready 估计层，不能写成 ASPR superiority，因为模块重跑、人类盲评和 checkpoint 输出仍是 pipeline-ready gap。",
        "",
        "## 本轮输出",
        "",
        "- `claim_to_evidence_map.csv`",
        "- `visual_layout_audit.csv`",
        "- `nature_readiness_scorecard.csv`",
        "- `reviewer_objection_response_matrix.csv`",
        "- `iteration_decision_board.csv`",
        "- `updated_gaps.csv`",
        "- 每张图子目录中的数据取证、视觉审计、claim-to-evidence map、审稿人质疑记录和准备度评分",
    ]
    (out_dir / "audit_iteration_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    next_goal = """运行下一轮 Fig.1-Fig.10 Nature 投稿审计，优先处理本轮 `iteration_decision_board.csv` 中的 P1_major 问题：Fig.2 future-outcome/citation-bias caveat、Fig.3 fold stability、Fig.4 semantic matching strictness、Fig.5 visual synthesis overclaim、Fig.6 proxy robustness、Fig.7 interval separation caveat、Fig.9 assumed ASPR-Qwen、Fig.10 LLM-as-judge ablations、真实模块重跑与 qwen3 proxy-vs-same-rubric discrepancy。停止条件：所有 P1 已修复、降级、转 supplement 或记录为 pipeline-ready gap，并重新生成 caption edits 和 reviewer response matrix。整个长期循环最多约 10 轮，不能无限重做同一类审计。
"""
    (out_dir / "next_iteration_goal.md").write_text(next_goal, encoding="utf-8")


def write_reproducibility_outputs(out_dir: Path) -> None:
    rows = []
    for fig in FIGURES:
        has_quality = any("quality_report" in item for item in fig.inputs)
        has_script = fig.fig_id in {"fig6", "fig7", "fig8", "fig9", "fig10"}
        if fig.fig_id in {"fig5", "fig8"}:
            level = "partially_reproducible_with_external_model_or_render"
        elif fig.fig_id in {"fig9", "fig10"}:
            level = "reproducible_from_cache_with_pipeline_ready_assumptions"
        elif has_quality or has_script:
            level = "reproducible_from_cache"
        else:
            level = "partially_reproducible"
        rows.append(
            {
                "fig_id": fig.fig_id,
                "label": fig.label,
                "reproducibility_level": level,
                "script_or_generation_hint": generation_hint(fig.fig_id),
                "manual_intervention_risk": "image2/pipeline assumption" if fig.fig_id in {"fig5", "fig8", "fig9", "fig10"} else "low",
            }
        )
    write_csv(out_dir / "reproducibility_matrix.csv", rows)
    write_csv(
        out_dir / "external_dependency_audit.csv",
        [
            {"dependency": "OpenAlex cache/API", "used_by": "Fig.7 and graph/venue metadata", "risk": "cache should be frozen and cited in manifest"},
            {"dependency": "image2/generated visual layer", "used_by": "Fig.5 and possibly Fig.8", "risk": "not numeric evidence; prompt/source files must remain frozen"},
            {"dependency": "ASPR-Qwen checkpoint", "used_by": "Fig.8-Fig.10", "risk": "Fig.9/Fig.10 currently include assumed or LLM-judge layers"},
            {"dependency": "Nature markdown corpus", "used_by": "Fig.4/Fig.9 and ASPR-Qwen training story", "risk": "local path and licensing/provenance must be documented"},
        ],
    )
    write_csv(
        out_dir / "manual_intervention_log.csv",
        [
            {"figure": "Fig.5", "intervention": "image2 visual synthesis", "status": "retain with source-note"},
            {"figure": "Fig.8", "intervention": "algorithm framework rendering", "status": "architecture only"},
            {"figure": "Fig.9", "intervention": "assumed ASPR-Qwen output", "status": "pipeline-ready gap"},
            {"figure": "Fig.10", "intervention": "LLM-as-judge ablation/preference estimates", "status": "pipeline-ready gap"},
        ],
    )


def generation_hint(fig_id: str) -> str:
    hints = {
        "fig1": "experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py",
        "fig2": "experiments/kg_perturbation_fig2/fig2_empirical_panels.py",
        "fig3": "experiments/kg_perturbation_fig3/fig3_empirical_weight_learning.py",
        "fig4": "experiments/kg_perturbation_fig4/main_fig4.py",
        "fig5": "experiments/kg_perturbation_fig5/build_fig5_image2_handoff.py",
        "fig6": "experiments/kg_perturbation_fig6/build_fig6_robustness.py",
        "fig7": "experiments/kg_perturbation_fig7/build_fig7_venue_contribution.py",
        "fig8": "outputs/kg_perturbation_fig8/render_fig8.py",
        "fig9": "experiments/kg_perturbation_fig9/build_fig9_case.py",
        "fig10": "experiments/kg_perturbation_fig10/build_fig10_ablation.py",
    }
    return hints.get(fig_id, "")


def write_reviewer_reports(out_dir: Path, aggregate: Mapping[str, List[Dict[str, Any]]]) -> None:
    persona_titles = {
        "quant_methods": "定量方法审稿人",
        "domain_science": "领域科学审稿人",
        "ai_systems": "AI/LLM 系统审稿人",
    }
    for persona, title in persona_titles.items():
        rows = [row for row in aggregate["reviewer_matrix"] if row["reviewer_persona"] == persona]
        lines = [f"# {title}模拟报告", ""]
        for idx, row in enumerate(rows, start=1):
            lines.extend(
                [
                    f"## {idx}. {row['label']}",
                    "",
                    f"- 质疑：{zh_text(str(row['objection']))}",
                    f"- 严重度：`{row['severity']}`",
                    f"- 证据：{row['evidence']}",
                    f"- 预备回应：{zh_text(str(row['response']))}",
                    f"- 处理状态：`{row['disposition']}`",
                    "",
                ]
            )
        (out_dir / f"reviewer_{persona}_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_verification_log(out_dir: Path, manifest: Mapping[str, Any], aggregate: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    checks = {
        "protocol_exists": PROTOCOL_PATH.exists(),
        "ten_figures_configured": len(FIGURES) == 10,
        "all_primary_images_exist": all((PROJECT_ROOT / fig.image_path).exists() for fig in FIGURES),
        "per_figure_data_forensics": all((out_dir / fig.fig_id / f"{fig.fig_id}_data_forensics.md").exists() for fig in FIGURES),
        "per_figure_visual_audit": all((out_dir / fig.fig_id / f"{fig.fig_id}_visual_audit.csv").exists() for fig in FIGURES),
        "per_figure_claim_map": all((out_dir / fig.fig_id / f"{fig.fig_id}_claim_to_evidence_map.csv").exists() for fig in FIGURES),
        "per_figure_reviewer_notes": all((out_dir / fig.fig_id / f"{fig.fig_id}_reviewer_objection_notes.md").exists() for fig in FIGURES),
        "per_figure_readiness": all((out_dir / fig.fig_id / f"{fig.fig_id}_readiness_score.csv").exists() for fig in FIGURES),
        "decision_board_exists": (out_dir / "iteration_decision_board.csv").exists(),
        "updated_gaps_exists": (out_dir / "updated_gaps.csv").exists(),
        "caption_edits_exists": (out_dir / "caption_edits.md").exists() and (out_dir / "caption_edits.csv").exists(),
        "reviewer_response_matrix_exists": (out_dir / "reviewer_objection_response_matrix.csv").exists(),
    }
    quality = {
        "checks": {key: int(value) for key, value in checks.items()},
        "overall_pass": bool(all(checks.values())),
        "status_label": "complete_iteration_with_pipeline_ready_gaps",
        "pipeline_gap_count": len([row for row in aggregate["decision_board"] if row["lane"] == "记录为 pipeline-ready gap"]),
    }
    lines = [
        "# 验证日志",
        "",
        f"- 迭代编号：`{manifest['iteration_id']}`",
        f"- 输出目录：`{out_dir}`",
        "",
        "## 门槛检查",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- `{key}`: {int(value)}")
    lines.extend(
        [
            "",
            f"总体通过：`{quality['overall_pass']}`",
            f"状态：`{quality['status_label']}`",
            f"Pipeline-ready gap 数：`{quality['pipeline_gap_count']}`",
            "",
            "## 复现命令",
            "",
            "```bash",
            "python3 experiments/nature_submission_audit/build_nature_submission_audit.py --iteration-id latest",
            "python3 -m unittest tests.test_nature_submission_audit -v",
            "git diff --check",
            "```",
        ]
    )
    (out_dir / "verification_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "figure_quality_report.json", quality)
    return quality


def build_audit(out_root: Path = DEFAULT_OUT_ROOT, iteration_id: Optional[str] = None) -> Dict[str, Any]:
    if iteration_id is None:
        iteration_id = dt.datetime.now().strftime("iteration_%Y%m%d_%H%M")
    elif not iteration_id.startswith("iteration_") and iteration_id != "latest":
        iteration_id = f"iteration_{iteration_id}"
    if iteration_id == "latest":
        out_dir = out_root / "iteration_latest"
    else:
        out_dir = out_root / iteration_id
    ensure_dir(out_dir)
    manifest = build_snapshot(out_dir, iteration_id)
    intermediate = build_intermediate_index(out_dir)
    aggregate = write_per_figure_outputs(out_dir, intermediate)
    write_aggregate_outputs(out_dir, aggregate)
    write_reproducibility_outputs(out_dir)
    write_reviewer_reports(out_dir, aggregate)
    quality = write_verification_log(out_dir, manifest, aggregate)
    write_json(
        out_dir / "run_manifest.json",
        {
            "iteration_id": iteration_id,
            "created_at": dt.datetime.now().isoformat(),
            "protocol": str(PROTOCOL_PATH),
            "out_dir": str(out_dir),
            "quality": quality,
            "figures": [fig.label for fig in FIGURES],
        },
    )
    return {"out_dir": str(out_dir), "iteration_id": iteration_id, "quality": quality}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--iteration-id", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = build_audit(args.out_root, args.iteration_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
