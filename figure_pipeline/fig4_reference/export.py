"""Reference-style gallery, source companion and delivery archive."""
from __future__ import annotations

import html
import json
import shutil
import zipfile
from pathlib import Path

from .data import OUT, ROOT
from .render import BOXES, PANELS, TITLES


def handoff() -> Path:
    design = OUT / "design"
    design.mkdir(parents=True, exist_ok=True)
    downloads = Path("/mnt/c/Users/jayee/Downloads")
    for name in ["Fig4_正式优化设计方案.md", "Fig4_实际数据核对与绘图口径.md",
                 "复杂科学信息图：受控信息效应与证据报告机制.png"]:
        shutil.copy2(downloads / name, design / name)
    text = (downloads / "Fig4_正式优化设计方案.md").read_text(encoding="utf-8")
    caption = text.split("# 11. 图注建议稿", 1)[1].split("---", 1)[0].split("**Fig. 4", 1)[1]
    (OUT / "caption_en.md").write_text("**Fig. 4" + caption.rstrip() + "\n", encoding="utf-8")
    (OUT / "README_zh.md").write_text(readme(), encoding="utf-8")
    (ROOT / "figure_pipeline/fig4_reference/README.md").write_text(
        "# Fig4 reference\n\n依据用户2026-10-02正式方案，只读取现有实验结果，不调用模型。\n\n"
        "```bash\n/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_reference all\n"
        "# 可分开运行：prepare / render / package\n```\n\n"
        "输出：`outputs/fig4_reference/`。整图324×237 mm；6 panels、13组件；CSV源表和图注随图交付。\n"
        "数据逻辑核对与图像目视检查随绘图完成，不增加回归测试或独立验证框架。\n", encoding="utf-8")
    for name in ["e2_total", "e2_paper"]:
        for suffix in [".svg", ".pdf", ".png"]:
            (OUT / "components" / (name + suffix)).unlink(missing_ok=True)
    gallery()
    manifest = [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size}
                for p in sorted(OUT.rglob("*")) if p.is_file() and p.suffix != ".zip" and p.name != "file_manifest.json"]
    (OUT / "file_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    archive = OUT / "Fig4_reference_delivery.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix != ".zip": z.write(path, "outputs/fig4_reference/" + str(path.relative_to(OUT)))
        for path in sorted((ROOT / "figure_pipeline/fig4_reference").glob("*")):
            if path.is_file(): z.write(path, str(path.relative_to(ROOT)))
    print(f"Delivery: {archive} ({archive.stat().st_size / 1e6:.1f} MB)", flush=True)
    return archive


def readme() -> str:
    return """# Fig.4｜受控信息效应与证据进入报告的机制

本图依据随包保存的2026-10-02正式方案和绘图口径，数据以正式方案与口径为准；视觉按用户最新要求重绘，参照示例图与Fig1–3 reference的浅蓝圆角卡片、衬线标题、彩色组件及三行排版。
源实验结果和Fig1–3没有改动；没有模型调用、新报告或新科学评价。

## 图件

- `final/Fig4.svg`：324×237 mm整图，可编辑文字及矢量对象。
- `final/Fig4_outlined.svg`：字体转路径的便携副本。
- `final/Fig4.pdf`：矢量PDF；`Fig4.png`：4×预览；`Fig4_6x.png`与`Fig4_10x.png`：高分辨率导出。
- `panels/`：a–f独立SVG/PDF/PNG。
- `components/`：13个独立子组件SVG/PDF/PNG；e2总量带与逐篇组成共同交付。
- `data/`：确定性绘图源表、来源路径、完整动作关联原句和逐关系报告对应。
- `layouts/style.json`：毫米坐标；`caption_en.md`：英文图注；`design/`：用户方案和示例。
- `index.html`：本地浏览整图、panel和子组件。

## 配置名称和终点

T＝纯文本分析；E＝GEAR＋文本分析；G＝Graph＋文本分析；F＝完整系统；
F−J＝完整系统移除联合图分析；F−M＝完整系统屏蔽非路径结构数值；F−P＝完整系统屏蔽论文引用路径。
无分支或无Joint时用相应普通文本任务替代。G含单贡献Graph与Joint。

H＝历史回应覆盖率，V＝有效信息簇/报告，R＝已确认不当断言率。
H/R源值已为百分数；未知支持比例U原本为0–1。U与R不互斥，1−R不代表准确率。
七配置无长度目标，任务一致但实际篇幅/token不同；a–d是既有分析信息的受控条件效应，非整个系统端到端重跑。

## 每个panel的数据

|Panel|问题|绘图数据|
|---|---|---|
|a|信息配置与绝对终点|a_information_exposure、a_condition_endpoints；700报告|
|b|五组件的配对分布及均值区间|b_paper_effects_improvement、b_effect_summary_improvement、b_H_discrete_frequencies；1500配对值，15效应|
|c|四条件加性交互|c_interaction_paper、c_interaction_counts、c_interaction_summary；100论文|
|d|联合结构连接与严格有效跨贡献产出|d_structure_and_report_tracks、d_nonzero_papers；100论文；audit_cross_relation_vs_report_validity列出21个宽口径关系|
|e|旧融合动作转移和信息来源|e1_action_pairs、e1_action_transition_cells；9860动作记录；e2_information_sources_paper/totals；1800信息簇|
|f|旧关系候选核验与双报告处理|f1_all_candidates/counts；76候选；f2_relation_trace_tiles/paired_treatment_counts；44关系×2方法|

## 数值和图形口径

- b所有正值代表改善。R逐论文差值、均值、区间及四分位均反号；原始列保留。均值95%区间直接读取既有bootstrap结果。
- H只取五个离散差值；不做连续KDE或横向抖动。V保留整数点，R保留所有真实连续差值。
- 分布框表示论文四分位范围与中位数，数字注释表示均值及95%区间；所有15区间包含0，不证明等效。
- c面仅插值四个观测均值，z为0–8；灰网格是纯加性参照。交互0.74，[−0.88,2.37]；53/4/43篇正/零/负。
- d按J_topo及paper_id稳定排序；严格跨贡献产出为Full 1、移除Joint 3，沿同一100列显示。关系valid=true的5/16不能替代严格资格1/3。
- e1以paper_id、claim_id、finding_key关联。颜色为行比例，数字为记录数；动作修正不等于已核验科学错误纠正。
- e2按paper_id排列，以原始簇数堆叠，不按增益排序也不改成逐论文百分比；1800=368+551+881，659=368+291。
- f网格来自实际逐关系关联，不按汇总量随机生成；未决保留独立状态，不算遗漏。
- e–f是旧Fig3报告描述性机制分析，与a–d明确分区；不恢复已停止的风险工作。

## 复现与编辑

在ASPR项目根目录：

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_reference all
```

`prepare`只读取原实验，`render`只读取本包data源表并生成图件，`package`整理交付包。
运行依赖NumPy、Pandas、Matplotlib、SciPy、CairoSVG；复用Fig1字体配置函数和Windows Arial与Times New Roman字体，不附字体文件。
SVG独立组件是从矢量层提取的对象，不是PNG切片。修改绘图代码后运行render，随后package更新交付文件。
长原句CSV采用反斜杠escapechar；用Pandas读取时设置`escapechar='\\\\'`，以保留原始控制字符和引句。

校对采用数据读取/关联逻辑阅读及导出图像目视检查，没有编写或运行回归测试，没有另建验证系统；
绘图检查不等于新的科学核验，科学标签均来自已有模型、来源范围内评价。
"""


def gallery() -> None:
    content = ["""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fig.4 reference</title><style>body{font:16px Arial,sans-serif;color:#273039;max-width:1250px;margin:30px auto;padding:0 20px}
.hero{display:block;width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article img{width:100%;height:250px;object-fit:contain}a{color:#3D6FA6}p{line-height:1.5}h2{margin-top:34px}</style>
<h1>Fig.4 · 受控信息效应与证据进入报告的机制</h1><p>324×237 mm · 6 panels · 13矢量子组件 · 新实验与旧报告机制数据分区。</p>
<p><a href="final/Fig4.svg">可编辑SVG</a> · <a href="final/Fig4.pdf">矢量PDF</a> · <a href="final/Fig4_10x.png">10× PNG</a> · <a href="README_zh.md">说明</a> · <a href="caption_en.md">图注</a></p>
<img class="hero" src="final/Fig4.png" alt="Fig4整图"><h2>独立panel</h2><div class="grid">"""]
    for key in PANELS:
        content.append(f'<article><h3>{key} · {html.escape(TITLES[key])}</h3><a href="panels/{key}.svg"><img src="panels/{key}.png" alt="panel {key}"></a><p><a href="panels/{key}.pdf">PDF</a> · <a href="panels/{key}.svg">SVG</a></p></article>')
    content.append('</div><h2>独立子组件</h2><div class="grid">')
    for key in BOXES:
        content.append(f'<article><h3>{key}</h3><a href="components/{key}.svg"><img src="components/{key}.png" alt="{key}"></a></article>')
    content.append('</div></html>')
    (OUT / "index.html").write_text("\n".join(content), encoding="utf-8")
