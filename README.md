# ASPR

ASPR（Academic Scientific Paper Review）是一个面向学术论文自动评审的 Python 项目。当前仓库包含论文检索、相关论文召回与重排、LLM 创新性评价、GraphRAG 实验、训练数据构建、模型微调脚本，以及若干知识图谱创新评价实验。

本仓库已经按功能重新整理。核心原则是：可复用代码放在 `aspr/`，一次性任务脚本放在 `scripts/`，缓存和数据放在 `data/`，生成结果放在 `outputs/`，独立实验放在 `experiments/`。

## 环境变量与密钥

项目启动时会自动读取仓库根目录的 `.env` 和 `.env.local`。推荐先复制示例文件，再在本地填写密钥：

```bash
cp .env.example .env
```

常用变量包括 `OPENALEX_EMAIL`、`OPENALEX_API_KEY`、`OPENALEX_API_KEYS`、`S2_API_KEY`、`NCBI_EMAIL`、`NCBI_API_KEY`、`HF_TOKEN`。其中 `OPENALEX_API_KEYS` 支持逗号或空格分隔的多个 key，corpus/Fig1/Fig2/Fig3 抓取路径会轮换使用。命令行参数仍可覆盖 `.env`。

## 项目结构

```text
ASPR/
├── aspr/                         # ASPR 核心代码包
│   ├── __init__.py               # 包初始化文件
│   ├── open_scholar.py           # 主评审流程：关键词抽取、Semantic Scholar 检索、召回重排、创新性评价
│   ├── lats.py                   # LATS 树搜索与反思式创新性评价
│   ├── graph_rag.py              # 基于 nano-graphrag + Ollama 的 GraphRAG 辅助模块
│   ├── pdf_downloader.py         # ACL Anthology PDF 下载工具
│   └── prompts.py                # 关键词抽取、综述生成、创新性评价等提示词模板
│
├── scripts/                      # 可直接运行的任务脚本
│   ├── download_nature.py        # Nature 系列期刊论文 PDF/同行评审文件下载器
│   ├── download_six_journals_2023_2025.sh
│   │                              # 批量下载 6 本 Nature 期刊 2023-2025 年论文
│   ├── hfdata_builder.py         # 构建 paper reconstruction SFT 数据集
│   ├── train_sft_qwen.sh         # Qwen 0.6B 全量 SFT 训练脚本
│   └── train_sft_lora_qwen.sh    # Qwen 8B LoRA SFT 训练脚本
│
├── data/                         # 数据与缓存
│   ├── papers.json               # 论文数据缓存/样例数据
│   ├── total_related_papers.json # Semantic Scholar 检索得到的候选相关论文缓存
│   ├── most_related_paper.json   # 召回与重排后的高相关论文缓存
│   └── paper_reconstruction_sft/ # 本地 HuggingFace Dataset 格式训练数据
│
├── outputs/                      # 程序生成物和实验结果
│   ├── demo/                     # kg_validator 演示模式历史输出图表
│   ├── kg_validator/             # 知识图谱验证实验输出
│   ├── kg_perturbation_fig1/     # Fig. 1 知识扰动实验输出
│   ├── logs/                     # 运行日志
│   ├── downloads/                # PDF 下载结果，默认不纳入版本管理
│   ├── checkpoints/              # 训练 checkpoint，默认不纳入版本管理
│   └── graphrag/                 # GraphRAG 索引与缓存，默认不纳入版本管理
│
├── experiments/                  # 独立研究实验
│   ├── __init__.py
│   ├── kg_validator/             # 科学创新评价/知识图谱验证实验
│   │   ├── __init__.py
│   │   ├── main.py               # 实验入口：demo/full/field_contrast/paper_contrast 等模式
│   │   ├── fetcher.py            # OpenAlex 数据拉取
│   │   ├── graph_builder.py      # NetworkX 引用图构建、切片、GraphML 导出
│   │   ├── metrics.py            # 七维创新评价指标计算
│   │   ├── comparator.py         # 前后图谱对比、图表绘制、结果导出
│   │   └── README.md             # kg_validator 详细说明
│   │
│   └── kg_perturbation_fig1/     # Fig. 1 风格知识图谱扰动实验
│       ├── fig1_knowledge_perturbation_v3.py
│       │                          # OpenAlex 元数据下载、图构建、扰动指标计算和绘图
│       ├── configs/              # CRISPR、graphene、iPSC、transformer 等领域配置
│       ├── run_crispr_example.sh # CRISPR 样例运行脚本
│       ├── README.md             # 实验说明
│       └── FIG1_KNOWLEDGE_PERTURBATION_EXPLAINED_ZH.md
│                                  # 中文代码讲解文档
│
├── tests/                        # 轻量手工检查脚本
│   └── test_openai_client.py     # 调用本地 OpenAI 兼容服务的简单测试
│
├── archives/                     # 原始归档文件
│   └── kg_perturbation_fig1_code.zip
│
├── AGENTS.md                     # 面向编码代理的项目规则和命令说明
├── README.md                     # 当前文档
└── .gitignore                    # 忽略缓存、日志、下载、checkpoint 等生成物
```

## 核心流程说明

主流程入口是 `aspr.open_scholar`。它大致做下面几步：

1. 从待评审论文标题、摘要中抽取关键词。
2. 调用 Semantic Scholar API 检索候选相关论文。
3. 使用 BGE-M3 做初步召回。
4. 使用 OpenScholar reranker 做重排。
5. 将最相关论文缓存到 `data/most_related_paper.json`。
6. 调用 `aspr.lats.evaluate_paper_innovation()` 生成创新性评价。

相关缓存路径已经改到 `data/` 下，不再散落在仓库根目录。

## 常用命令

### 运行主评审流程

```bash
python -m aspr.open_scholar
```

### 单独运行 LATS 创新性评价测试

```bash
python -m aspr.lats
```

### GraphRAG 插入和查询

```bash
python -m aspr.graph_rag insert "paper/reference text"
python -m aspr.graph_rag query "What is novel?"
```

GraphRAG 工作目录默认为：

```text
outputs/graphrag/
```

### PDF 下载工具测试

```bash
python -m aspr.pdf_downloader
```

默认下载测试结果会放到：

```text
outputs/downloads/acl_tests/
```

### Nature 期刊批量下载

```bash
bash scripts/download_six_journals_2023_2025.sh
```

或单独运行：

```bash
python scripts/download_nature.py \
  --journal-name "Nature Microbiology" \
  --journal-id 41564 \
  --year 2024 \
  --email your@email.com
```

默认输出路径：

```text
outputs/downloads/nature_pdfs_<journal_id>_<year>/
outputs/logs/
```

### 构建 SFT 数据集

```bash
python scripts/hfdata_builder.py
```

默认读取：

```text
../dataset/paper/
../dataset/reconstruction/
```

默认写入：

```text
data/paper_reconstruction_sft/
```

如需上传 HuggingFace Hub：

```bash
python scripts/hfdata_builder.py --push-to-hub
```

### 模型训练

```bash
bash scripts/train_sft_qwen.sh
bash scripts/train_sft_lora_qwen.sh
```

checkpoint 默认写入：

```text
outputs/checkpoints/
```

## 实验模块

### kg_validator：知识图谱创新评价实验

演示模式：

```bash
python -m experiments.kg_validator.main --mode demo
```

基于 DOI 的论文邻域前后对比：

```bash
python -m experiments.kg_validator.main --mode field_contrast \
  --paper-dois "10.1038/s41586-021-03819-2" \
  --event-label "AlphaFold neighborhood" \
  --email your@email.com
```

默认输出路径：

```text
outputs/kg_validator/
outputs/logs/kg_validator.log
```

更多参数说明见：

```text
experiments/kg_validator/README.md
```

### kg_perturbation_fig1：知识图谱扰动 Fig. 1 实验

运行 CRISPR 示例：

```bash
bash experiments/kg_perturbation_fig1/run_crispr_example.sh
```

默认输出路径：

```text
outputs/kg_perturbation_fig1/
```

配置文件位于：

```text
experiments/kg_perturbation_fig1/configs/
```

## 数据与生成物约定

- `data/`：相对稳定的数据、缓存和本地数据集，可以作为后续流程输入。
- `outputs/`：运行时生成的图表、日志、下载文件、checkpoint、GraphRAG 索引。
- `outputs/downloads/`、`outputs/checkpoints/`、`outputs/graphrag/`、`outputs/logs/` 默认在 `.gitignore` 中忽略。
- 已经存在的演示图表和实验历史结果保留在 `outputs/demo/`、`outputs/kg_validator/`、`outputs/kg_perturbation_fig1/`。

### 统一论文图谱数据层

推荐把可复用论文图谱语料集中放在：

```text
data/knowledge_corpus/v1_large/
```

该目录由 `aspr.corpus` 管理，包含 canonical 表 `works.csv`、`citations.csv`、`topics.csv`、`topic_edges.csv`、`domains.csv`、`landmarks.csv`，以及各实验可直接读取的 `views/fig1|fig2|fig3|fig5/`。

快速复用现有本地 Fig1/Fig3 数据：

```bash
python -m aspr.corpus build --offline
```

按大规模混合领域配置从本地数据起步，并用 OpenAlex 补齐缺口领域：

```bash
python -m aspr.corpus build \
  --profile v1_large \
  --max-domains 48 \
  --papers-per-domain 4000 \
  --start-year 1980 \
  --end-year 2025
```

质量审计与重新生成实验视图：

```bash
python -m aspr.corpus audit
python -m aspr.corpus make-views
```

Fig2/Fig3/Fig5 的默认输入会优先使用该 corpus view；没有生成时仍回退到旧的 `outputs/kg_perturbation_fig1/` 或 Fig3 输出目录。Fig1 可通过 `--corpus-dir data/knowledge_corpus/v1_large` 先 materialize 兼容缓存再绘图。

## 依赖说明

核心依赖包括：

```text
openai
langchain
langchain-openai
langgraph
pydantic
requests
pypdf
FlagEmbedding
nano-graphrag
ollama
datasets
backoff
typing_extensions
deepspeed
networkx
numpy
pandas
matplotlib
scikit-learn
tqdm
pyyaml
```

不同子模块依赖不同。只运行主评审流程时，不一定需要安装所有实验依赖；运行 `experiments/kg_validator` 或 `experiments/kg_perturbation_fig1` 时需要额外安装图计算和绘图相关包。

## 远程资源

- Dataset: https://huggingface.co/datasets/jayeew/paper-reconstruction-sft
- Qwen 0.6B review model: https://huggingface.co/jayeew/qwen-0.6b-review
- Qwen 8B review QLoRA model: https://huggingface.co/jayeew/qwen-8b-review-qlora
