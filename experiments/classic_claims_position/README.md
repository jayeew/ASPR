# 第一阶段：经典成果数据与图谱准备

范围仅为18条claim的科学证据、6组中性数据契约及3个历史快照。不运行GEAR/Graph创新分析、不插入目标、不编辑Fig10。

从仓库根目录执行。输入固定清单见`catalog.py`，支持范围与条件级缺口见`scope_notes.py`。材料保存在`outputs/classic_claims_position_2025_2026/preparation/`（下称OUT）。归档的web工具返回是来源文件，不是脚本生成的论文全文；重建交付包需要保留这些输入。

## 环境

```bash
python3 -m venv --system-site-packages outputs/classic_claims_position_2025_2026/preparation/.venv
outputs/classic_claims_position_2025_2026/preparation/.venv/bin/python -m pip install beautifulsoup4 lxml pymupdf faiss-cpu==1.15.1 python-igraph==1.0.0 leidenalg==0.12.0
```

复用本机已有torch/CUDA、numpy、pyarrow及项目依赖。确切版本见交付目录`environment.json`。本次实际构图曾借用现有Fig10环境的只读解释器，版本与准备环境一致；当前脚本默认使用独立准备环境，不写入旧案例环境。GPU阶段应串行，避免显存竞争。三个独立快照的非GPU步骤可由调用者安排；本次中断重试均保留日志。

## 可重复执行命令

```bash
# 网络获取：缓存存在则复用；失败记录保留。不会调用LLM。
outputs/classic_claims_position_2025_2026/preparation/.venv/bin/python -m experiments.classic_claims_position.acquire
outputs/classic_claims_position_2025_2026/preparation/.venv/bin/python -m experiments.classic_claims_position.supplement

# 从随包归档的web提取中选择明确定位的科学片段，生成中性JSONL/CSV。
python3 -m experiments.classic_claims_position.curate
python3 -m experiments.classic_claims_position.package

# 独立构图，原生语义/社区/统计流程；已完成分块可resume。
python3 -m experiments.classic_claims_position.snapshot --cutoff 2025-10-06
python3 -m experiments.classic_claims_position.snapshot --cutoff 2025-10-07
python3 -m experiments.classic_claims_position.snapshot --cutoff 2025-10-08

# 生成数据时读取各资产的实际最大日期。
python3 -m experiments.classic_claims_position.package
python3 -m experiments.classic_claims_position.audit_reused_references
python3 -m experiments.classic_claims_position.audit_motifs
python3 -m experiments.classic_claims_position.validate
python3 -m pytest -q experiments/classic_claims_position/test_preparation.py
python3 -m ruff check experiments/classic_claims_position
python3 -m experiments.classic_claims_position.deliver
```

`validate.py`调用原生`validate_assets`并扩展日期、孤立边、向量行、社区、映射、来源哈希、显式提示及原始文件保护检查。也可直接执行`python3 -m gear validate-assets --claim-graph-root <snapshot>`。图谱资产完整哈希在`graph_assets.json`，科学材料哈希在`source_files.json`；它们用于复核此次交付，不声称把此前全研究冻结为发布版。

`build_graph.py`仅替换离线论文路径查询为同等关系的集合交运算，保持原生定义与其他构建阶段。入口对3组语义端点与原生SQL作精确对照；`audit_motifs.py`另从三个快照各取7个非空标注，与未修改的原生SQL比对，整数和布尔量完全一致，浮点量容许Parquet float32存储误差。社区、质心和统计各自重建。

原生统计脚本会输出历史百分位表；本阶段未使用，清单明确禁用，后续不得套用于阈值化目标邻居指标。2026组复用原始2023—2025图谱，不扩建2026数据。

`preflight.json`是早期预检查的审计记录，不是最终状态；最终以`validation.json`、`review_report_zh.md`及`graph_assets.json`为准。EvidenceStore保留早期整理版本，当前评估有效集合仅由`evaluation/evidence.jsonl`定义。

## 数据边界

`evaluation/`是中性科学数据，`external_provenance/`是用户背景、外部奖项依据及原ID映射，两者分开。原始科学HTML可能含网站导航，不应直接作为模型提示；使用经筛选的evidence片段。仅消除显式提示，不消除模型已有知识。

这不是gear论文全文InputContract。需要在下一阶段单独适配固定claim、多来源证据绑定、知识位置口径和联合分析输出。本阶段不实现适配，也不把已有新颖性类别改名。

原始引用索引有1条晚于2026截止日的元数据记录及4202条无日期记录，见`reused_reference_anomalies.json`。未来记录不被任何Nature目标引用，故不会进入其原生路径标注；无日期元数据仍不能作合格科学支持。claim覆盖与引用索引覆盖分开报告。
