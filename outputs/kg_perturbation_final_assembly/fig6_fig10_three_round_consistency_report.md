# Fig.6-Fig.10 Consistency and Final Assembly

## Assembly Thesis

Fig.1-Fig.5 establish graph-perturbation evidence and validation. Fig.6 tests the robustness of that method. Fig.7 moves the evidence into venue-level scientific interpretation. Fig.8 introduces ASPR as the final graph-agent plus ASPR-Qwen system. Fig.9 shows one auditable ASPR run. Fig.10 tests whether ASPR quality comes from module composition rather than a single generic LLM.

## Round 1: Cross-Figure Audit

- **complete**: Fig.6 correctly follows Fig.1-Fig.5 by testing robustness and boundary conditions rather than adding another success case. Action: Keep Fig.6 as a data-heavy reliability bridge; retain proxy labels in caption.
- **complete_with_gap**: Fig.7 moves from method validation to venue-level interpretation; the Nature point estimate is supported, but strict interval-separated dominance remains caveated. Action: Use a careful venue contribution caption and record strict interval separation as a pipeline-ready gap.
- **complete**: Fig.8 clearly introduces ASPR as graph agent plus ASPR-Qwen plus fusion/verifier. Action: Keep as algorithm framework; ensure captions do not describe it as a statistical result.
- **complete_with_gap**: Fig.9 shows one real case run and labels the ASPR-Qwen lane as assumed pipeline-ready. Action: Keep as case storyboard; make assumption visible in final caption.
- **complete_with_gap**: Fig.10 proves module contribution narratively with full ASPR real metrics and LLM-as-judge ablation estimates. Action: Keep as ablation figure; make LLM-as-judge status visible in caption and data table.

## Round 2: Terminology, Color, And Panel Order Fixes

- **complete**: The figure set needs one vocabulary ladder: graph-perturbation analysis before Fig.8, ASPR graph agent after Fig.8, ASPR-Qwen only for the SFT reviewer lane. Action: Apply terminology crosswalk in caption drafts and final checklist.
- **complete**: The shared color ledger is already mostly respected: Nature red, graph blue, ASPR-Qwen purple, verifier orange, fusion black/slate. Action: Record canonical palette and use it as the assembly standard for future redraws.
- **complete**: Fig.6, Fig.7, and Fig.10 are data figures; Fig.8 and Fig.9 are algorithm/case figures. This satisfies the requested visual division. Action: Keep panel order: validation, interpretation, method, case, ablation.

## Round 3: Final Checklist And Caption Package

- **pass** `Fig.6承接方法可信度`: Robustness and boundary-condition panels directly test Fig.1-Fig.5 method reliability.
- **pass_with_gap** `Fig.7转向venue-level interpretation`: Venue-level outputs exist; strict interval-separated Nature dominance remains pipeline-ready.
- **pass** `Fig.8提出最终方法ASPR`: ASPR is defined as graph agent plus ASPR-Qwen plus fusion/verifier.
- **pass_with_gap** `Fig.9展示ASPR真实运行`: Real manuscript and peer-review trace are present; ASPR-Qwen lane is assumed and labeled.
- **pass_with_gap** `Fig.10证明模块贡献`: Full ASPR real Fig.4 metrics plus LLM-as-judge ablation estimates.
- **pass** `颜色字体panel label统一`: Final assembly records canonical palette and label/caption conventions.
- **pass** `避免表格堆砌`: Figures use heatmaps, forest plots, scatter, storyboard, module diagrams, matrix cards, and Pareto bars.
- **pass** `完成三轮一致性检查`: Round 1 audit, Round 2 terminology/visual fix, Round 3 checklist/captions are written.

## Pipeline-Ready Gaps

- **Fig.6** (pipeline-ready): Some robustness panels are score-table proxy probes, not full graph extraction reruns. Next replacement: Rerun perturbation experiments with fresh graph extraction and retrieval if strict main-figure claims require it.
- **Fig.7** (pipeline-ready): Strict Nature Portfolio interval separation is not supported by the current bootstrap gate. Next replacement: Increase controlled sample coverage or refine uncertainty estimates until Nature lower CI exceeds the runner-up upper CI.
- **Fig.9** (pipeline-ready): ASPR-Qwen output is assumed for the case storyboard. Next replacement: Replace assumed Qwen JSON with the real ASPR-Qwen checkpoint output.
- **Fig.10** (pipeline-ready): Ablation and preference rows use LLM-as-judge pipeline estimates where true human preference or rerun ablation data are absent. Next replacement: Run real module ablations and collect blinded human preference ratings.

## Output Path Index

- **Fig.1**: `outputs/redraw_v6a_best_fig1/fig1_multi_domain_real.png` - keep
- **Fig.2**: `outputs/redraw_v6a_best_fig2/fig2_empirical_full.png` - keep
- **Fig.3**: `outputs/redraw_v6a_best_fig3/fig3_selected_weight_learning_full.png` - keep
- **Fig.4**: `outputs/kg_perturbation_fig4_full50/fig4_full.png` - keep
- **Fig.5**: `outputs/kg_perturbation_fig5/strict_ai_filtered_image2_handoff/fig5_strict_ai_filtered_image2_generated_preview.png` - keep
- **Fig.6**: `outputs/kg_perturbation_fig6/fig6_full.png` - keep
- **Fig.7**: `outputs/kg_perturbation_fig7/fig7_full.png` - soften headline and retain as venue-level interpretation
- **Fig.8**: `outputs/kg_perturbation_fig8/fig8_full.png` - keep
- **Fig.9**: `outputs/kg_perturbation_fig9/fig9_full.png` - keep
- **Fig.10**: `outputs/kg_perturbation_fig10/fig10_full.png` - keep
