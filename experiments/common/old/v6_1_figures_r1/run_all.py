#!/usr/bin/env python3
"""Run all ten redesigned ASPR v6.1 experiments and render Fig.1--Fig.10."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.source_audit_v6 import sha256_file  # noqa: E402
from experiments.common.old.v6_1_figures_r1.analysis import (  # noqa: E402
    ANGLE_ORDER,
    ANGLE_SHORT,
    experiment01_tables,
    experiment02_tables,
    experiment03_tables,
    experiment04_tables,
    experiment05_tables,
    experiment06_tables,
    experiment07_tables,
    experiment08_tables,
    experiment09_tables,
    experiment10_tables,
    hash_payload,
    load_inputs,
    run_angle_ablation,
    source_hash_table,
)
from experiments.common.old.v6_1_figures_r1.render import render_all  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "aspr_v6_1_figures.json"

EXPERIMENT_MAP = (
    {
        "experiment": 1,
        "question": "What corpus and future-diffusion target are evaluated?",
        "panels": "cohort flow; domain composition; yearly coverage; D3/D5/D8 label summary",
        "claim": "The analysis uses 118,059 Nature papers and a zero-inclusive, fold-local future-diffusion target.",
    },
    {
        "experiment": 2,
        "question": "How were candidate innovation indicators selected?",
        "panels": "candidate flow; role counts; gate matrix; 8-to-5 evidence map",
        "claim": "Fifty source-discovered candidates were screened without future outcomes, yielding eight primary metrics.",
    },
    {
        "experiment": 3,
        "question": "Are the admitted metrics covered, stable and non-identical?",
        "panels": "coverage; resampling stability; feature correlations; within/across-angle overlap",
        "claim": "Primary metrics meet the frozen coverage/stability gates and are not strict duplicates.",
    },
    {
        "experiment": 4,
        "question": "Do innovation metrics improve D5 ranking beyond controls?",
        "panels": "model comparison; paired gain CI; high-impact deciles; mean target separation",
        "claim": "The final indicators increase D5 temporal-OOF Spearman beyond K1 and B0.",
    },
    {
        "experiment": 5,
        "question": "Does the incremental signal persist at D3 and D8?",
        "panels": "horizon comparison; gains; horizon-by-fold matrix; prediction agreement",
        "claim": "The innovation gain is positive at D3, D5 and D8.",
    },
    {
        "experiment": 6,
        "question": "Does the model generalize to strictly later publication periods?",
        "panels": "fold timeline; fold performance; paired fold gains; yearly drift",
        "claim": "Every test block is future to training, with temporal heterogeneity shown explicitly.",
    },
    {
        "experiment": 7,
        "question": "Does the result generalize across all twelve domains?",
        "panels": "domain dumbbells; paired gain CIs; pure/full comparison; sample-size relation",
        "claim": "All twelve domains remain in the analysis; domain heterogeneity is not filtered.",
    },
    {
        "experiment": 8,
        "question": "How much signal remains without any controls?",
        "panels": "global comparison; fold comparison; decile separation; prediction overlap",
        "claim": "The eight innovation metrics alone rank future diffusion, while combination with controls performs best.",
    },
    {
        "experiment": 9,
        "question": "Which observation angles add non-redundant predictive information?",
        "panels": "K1-plus-angle; gain over K1; leave-angle-out loss; fold deletion matrix",
        "claim": "Post-hoc OOF ablations quantify predictive contribution without changing metric admission.",
    },
    {
        "experiment": 10,
        "question": "Do sensitivity, acceptance and reproducibility checks support the claim?",
        "panels": "control sensitivity; gate margins; stress-unit gains; exact replay checks",
        "claim": "All registered acceptance gates pass and reported outputs are reproducible from frozen inputs.",
    },
)


def _write_tables(
    output_root: Path,
    experiment_tables: Mapping[int, Mapping[str, pd.DataFrame]],
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for experiment_index, tables in experiment_tables.items():
        data_dir = output_root / f"experiment_{experiment_index:02d}" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for table_name, frame in tables.items():
            path = data_dir / f"{table_name}.csv"
            frame.to_csv(path, index=False)
            outputs[f"experiment_{experiment_index:02d}/{table_name}"] = path
    return outputs


def _create_contact_sheet(
    png_paths: Mapping[int, Path],
    output_path: Path,
) -> Path:
    images = {index: Image.open(path).convert("RGB") for index, path in png_paths.items()}
    target_width = 1450
    border = 35
    row_gap = 45
    thumbnails: Dict[int, Image.Image] = {}
    row_heights = []
    for index, image in images.items():
        height = int(image.height * target_width / image.width)
        thumbnails[index] = image.resize((target_width, height), Image.Resampling.LANCZOS)
    for row in range(5):
        row_heights.append(
            max(thumbnails[row * 2 + 1].height, thumbnails[row * 2 + 2].height)
        )
    canvas_width = border * 3 + target_width * 2
    canvas_height = border * 2 + sum(row_heights) + row_gap * 4
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    y_position = border
    for row, row_height in enumerate(row_heights):
        for column in range(2):
            index = row * 2 + column + 1
            image = thumbnails[index]
            x_position = border + column * (target_width + border)
            canvas.paste(image, (x_position, y_position))
            draw.rectangle(
                (
                    x_position,
                    y_position,
                    x_position + image.width,
                    y_position + image.height,
                ),
                outline="#D0D5DD",
                width=2,
            )
        y_position += row_height + row_gap
    canvas.save(output_path, quality=95)
    for image in images.values():
        image.close()
    return output_path


def _result_summary(
    experiment_tables: Mapping[int, Mapping[str, pd.DataFrame]],
) -> Dict[str, Any]:
    model_points = experiment_tables[4]["model_points"].set_index("model_id")
    gains = experiment_tables[4]["paired_gains"].set_index("baseline_model_id")
    horizons = experiment_tables[5]["horizon_gains"].set_index("horizon")
    fold_gains = experiment_tables[6]["fold_gain_intervals"]
    domain_gains = experiment_tables[7]["domain_gain_intervals"]
    angles = experiment_tables[9]["angle_summary"].copy()
    gates = experiment_tables[10]["acceptance_gates"]
    return {
        "n_corpus": int(experiment_tables[1]["corpus_flow"].iloc[0]["n"]),
        "n_oof": int(model_points.loc["final_innovation_plus_k1", "n_oof"]),
        "n_candidates": int(experiment_tables[2]["selection_flow"].iloc[0]["n"]),
        "n_primary_indicators": int(
            experiment_tables[2]["selection_flow"].iloc[-1]["n"]
        ),
        "n_angles": 5,
        "d5_spearman_final": float(
            model_points.loc["final_innovation_plus_k1", "spearman_expected"]
        ),
        "d5_spearman_k1": float(
            model_points.loc["k1_controls", "spearman_expected"]
        ),
        "d5_spearman_innovation_only": float(
            model_points.loc["innovation_only", "spearman_expected"]
        ),
        "d5_gain_vs_k1": float(gains.loc["k1_controls", "spearman_gain"]),
        "d5_gain_vs_k1_ci": [
            float(gains.loc["k1_controls", "gain_ci_low"]),
            float(gains.loc["k1_controls", "gain_ci_high"]),
        ],
        "horizon_gains": {
            f"D{int(index)}": float(row["spearman_gain"])
            for index, row in horizons.iterrows()
        },
        "fold_gain_range": [
            float(fold_gains["spearman_gain"].min()),
            float(fold_gains["spearman_gain"].max()),
        ],
        "fold_gain_ci_positive": int(fold_gains["gain_ci_low"].gt(0).sum()),
        "folds_total": len(fold_gains),
        "domain_gain_range": [
            float(domain_gains["spearman_gain"].min()),
            float(domain_gains["spearman_gain"].max()),
        ],
        "domain_gain_ci_positive": int(domain_gains["gain_ci_low"].gt(0).sum()),
        "domains_total": len(domain_gains),
        "single_angle_gains": {
            str(row.angle_id): float(row.increment_over_k1)
            for row in angles.itertuples()
        },
        "leave_angle_out_losses": {
            str(row.angle_id): float(row.drop_from_full)
            for row in angles.itertuples()
        },
        "acceptance_gates_passed": int(gates["margin"].ge(0).sum()),
        "acceptance_gates_total": len(gates),
    }


def _results_markdown(
    summary: Mapping[str, Any],
    output_path: Path,
) -> Path:
    angle_lines = []
    for angle in ANGLE_ORDER:
        angle_lines.append(
            f"- {ANGLE_SHORT[angle]}：单角度相对 K1 增量 "
            f"{summary['single_angle_gains'][angle]:+.4f}；删除该角度后的全模型损失 "
            f"{summary['leave_angle_out_losses'][angle]:+.4f}。"
        )
    text = f"""# ASPR v6.1 实验1–10重跑结果

本轮使用冻结的 Nature v6.1 数据、相同标签和相同六个时间折，重新组织并运行了十个互不重复的实验。旧版 Fig.1–Fig.10 和 v6.1 正式结果均未被覆盖。

## 核心结果

- 数据：{summary['n_corpus']:,} 篇 Nature 论文，12 个自然科学大类；{summary['n_oof']:,} 篇论文获得 1986–2017 时间 OOF 预测。
- 指标：系统候选池 {summary['n_candidates']} 个，经不使用结局的筛选后，保留 5 个观察角度下 8 个主指标。
- D5 主模型：Spearman = {summary['d5_spearman_final']:.4f}；K1 控制模型 = {summary['d5_spearman_k1']:.4f}。
- D5 增量：相对 K1 为 {summary['d5_gain_vs_k1']:+.4f}，95% 配对 bootstrap CI [{summary['d5_gain_vs_k1_ci'][0]:.4f}, {summary['d5_gain_vs_k1_ci'][1]:.4f}]。
- 纯创新指标：不含任何控制特征时 Spearman = {summary['d5_spearman_innovation_only']:.4f}。
- 多窗口：D3/D5/D8 的增量分别为 {summary['horizon_gains']['D3']:+.4f}、{summary['horizon_gains']['D5']:+.4f}、{summary['horizon_gains']['D8']:+.4f}。
- 时间折：6 个时间折中 {summary['fold_gain_ci_positive']}/{summary['folds_total']} 个增量区间下界高于 0；点估计范围为 [{summary['fold_gain_range'][0]:+.4f}, {summary['fold_gain_range'][1]:+.4f}]。
- 学科：12 个领域中 {summary['domain_gain_ci_positive']}/{summary['domains_total']} 个增量区间下界高于 0；点估计范围为 [{summary['domain_gain_range'][0]:+.4f}, {summary['domain_gain_range'][1]:+.4f}]。
- 接受门：{summary['acceptance_gates_passed']}/{summary['acceptance_gates_total']} 个预设结果门通过。

## 五角度消融

{chr(10).join(angle_lines)}

这些消融是结果已知背景下的解释性分析，只用于说明预测信息的互补性；它们不重新筛选指标，也不构成因果效应。

## 图组分工

Fig.1 定义数据和标签；Fig.2 说明指标为何被选择；Fig.3 检查测量质量；Fig.4 给出 D5 主结果；Fig.5 检查 D3/D5/D8；Fig.6 检查时间外推；Fig.7 检查学科外推；Fig.8 隔离纯创新信号；Fig.9 解释五角度互补性；Fig.10 汇总控制敏感性、接受门和精确复跑证据。
"""
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run redesigned ASPR v6.1 experiments 1--10"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(suite_config_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Execute all analyses, ablations, figures, checks and manifests."""
    start = time.monotonic()
    output_dir = Path(output_dir).resolve()
    if (output_dir / "_SUCCESS").is_file():
        raise FileExistsError(
            f"Completed output already exists; choose a new directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(PROJECT_ROOT, suite_config_path)
    stage_log = []

    def stage(name: str, started: float) -> None:
        stage_log.append(
            {"stage": name, "elapsed_seconds": round(time.monotonic() - started, 3)}
        )

    started = time.monotonic()
    experiment_tables: Dict[int, Mapping[str, pd.DataFrame]] = {
        1: experiment01_tables(inputs),
        2: experiment02_tables(inputs),
        3: experiment03_tables(inputs),
        4: experiment04_tables(inputs),
        5: experiment05_tables(inputs),
    }
    stage("experiments_01_05", started)

    started = time.monotonic()
    experiment_tables[6] = experiment06_tables(inputs)
    experiment_tables[7] = experiment07_tables(inputs)
    experiment_tables[8] = experiment08_tables(inputs)
    stage("experiments_06_08", started)

    started = time.monotonic()
    ablation = run_angle_ablation(inputs, output_dir / "experiment_09")
    experiment_tables[9] = experiment09_tables(inputs, ablation)
    stage("experiment_09_fixed_oof_ablation", started)

    started = time.monotonic()
    experiment_tables[10] = experiment10_tables(inputs)
    table_paths = _write_tables(output_dir, experiment_tables)
    lineage_path = output_dir / "source_lineage.csv"
    source_hash_table(inputs).to_csv(lineage_path, index=False)
    chart_map_path = output_dir / "experiment_chart_map.csv"
    pd.DataFrame(EXPERIMENT_MAP).to_csv(chart_map_path, index=False)
    stage("experiment_10_and_tables", started)

    started = time.monotonic()
    render_config = inputs.suite_config["render"]
    figure_paths = render_all(
        experiment_tables,
        output_dir / "figures",
        formats=list(render_config["formats"]),
        dpi=int(render_config["dpi"]),
    )
    contact_sheet = _create_contact_sheet(
        {index: paths["png"] for index, paths in figure_paths.items()},
        output_dir / "fig1_fig10_contact_sheet.png",
    )
    stage("render_figures", started)

    summary = _result_summary(experiment_tables)
    summary_path = output_dir / "results_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = _results_markdown(summary, output_dir / "experiment_results.md")
    all_outputs = {
        **table_paths,
        "source_lineage": lineage_path,
        "chart_map": chart_map_path,
        "results_summary": summary_path,
        "results_report": report_path,
        "contact_sheet": contact_sheet,
    }
    for index, paths in figure_paths.items():
        for file_format, path in paths.items():
            all_outputs[f"figure_{index:02d}_{file_format}"] = path

    manifest: Dict[str, Any] = {
        "artifact_kind": "aspr_v6_1_redesigned_fig1_fig10_suite",
        "suite_id": inputs.suite_config["suite_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_oof_artifact_id": inputs.oof_manifest["artifact_id"],
        "source_innovation_only_artifact_id": inputs.pure_manifest["artifact_id"],
        "source_screening_artifact_id": inputs.screening_manifest["artifact_id"],
        "claim_boundary": inputs.suite_config["claim_boundary"],
        "experiments": list(EXPERIMENT_MAP),
        "summary": summary,
        "stage_log": stage_log,
        "total_elapsed_seconds": round(time.monotonic() - start, 3),
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(all_outputs.items())
        },
    }
    manifest["artifact_id"] = hash_payload(manifest)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "_SUCCESS").write_text(
        f"{manifest['artifact_id']}\n", encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "contact_sheet": str(contact_sheet),
        "results_report": str(report_path),
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run(args.config.resolve(), args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
