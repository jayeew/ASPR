# 全文创新分析分阶段实验（原 200 篇，支持扩展至 1,000 篇）

当前研究主线是 Claim Graph + GEAR 创新性分析。目录名和默认运行目录沿用
`innovation_200`，实际样本范围以运行目录的 `papers.jsonl` 为准。下面的抽样命令
最初用于建立 200 篇样本；已扩展的研究不要用初始抽样流程重置名单。

## 1,000 篇扩展的范围

```bash
python3 experiments/innovation_200/run_claims_reviews_1000.py \
  --study outputs/innovation_200_20260907 \
  --manifest data/nature_2026_testset/manifest.jsonl \
  --workers 64 --cli-limit 48
```

此入口要求源 manifest 有 1,000 个唯一论文 ID，原研究名单有 200 或 1,000 篇，
且对应 `metadata/{paper_id}.json` 已存在。它更新 `papers.jsonl`，保留原名单顺序，
追加其余论文，生成 `full_corpus_distribution.json`，并并行续跑贡献抽取与审稿参考
重建。它不执行 GEAR、Graph、报告或最终测评；不要把名单扩展或两个准备阶段完成
写成 1,000 篇完整实验完成。`--dry-run` 仍会更新名单和分布文件，只跳过模型阶段，
不是只读预览。运行时不要让其他名单更新器同时修改同一研究。

## 分阶段执行

该目录实现九个可独立运行的阶段。阶段之间只交换运行目录中的普通 JSON、JSONL、Markdown 和 CSV；已有目标文件默认续跑跳过，只有显式 `--overwrite` 才覆盖对应阶段。

```bash
STUDY=outputs/innovation_200_20260907

python3 experiments/innovation_200/sample_papers.py --output "$STUDY"
python3 experiments/innovation_200/reconstruct_reviews.py --study "$STUDY"
python3 experiments/innovation_200/extract_claims.py --study "$STUDY"
python3 experiments/innovation_200/run_gear.py --study "$STUDY"
python3 experiments/innovation_200/run_graph.py --study "$STUDY"
python3 experiments/innovation_200/generate_reports.py --study "$STUDY"
python3 experiments/innovation_200/evaluate_human.py --study "$STUDY"
python3 experiments/innovation_200/compare_reports.py --study "$STUDY"
python3 experiments/innovation_200/summarize_results.py --study "$STUDY"
```

每个脚本都会把带时间戳的日志同时打印到终端，并追加写入
`$STUDY/logs/{脚本名}.log`。日志包含启动参数、阶段步骤、逐任务进度、跳过与失败数、吞吐、等待上游文件的状态和总耗时；`--verbose` 会在终端额外显示逐任务 JSON 详情。机器可读的断点状态仍独立保存在 `$STUDY/status/*.json`，日志不参与续跑判断。

也可用 `run_all.py` 编排这些阶段。claims 完成后，GEAR、Graph、报告、人工测评和匿名比较作为独立进程流式衔接；一篇论文的上游文件齐全后即可进入下游，不等待整个当前名单完成。CLI 总并发默认 16，可显式设为 32、48 或 64。文献网络并发默认 16，可通过 `GEAR_NETWORK_MAX_PROCESSES` 提高到 32。Graph 先集中批量准备图事实，再使用统一模型任务队列；GEAR 默认两个论文工作进程；所有 Codex 请求仍受跨进程总并发锁控制。正式运行时全部请求使用 `gpt-5.6-luna`，角色决定 low/high。实验配置关闭模型回答缓存、关系稳定性复判和输入指纹核对，失败请求最多重试两次。

主要中间文件：

- `papers.jsonl`：当前研究名单；`field_distribution.json`：初始抽样的历史目标和领域组成；扩展名单的实际组成见 `full_corpus_distribution.json`。
- `human_refs/{paper_id}.json`：全部轮次的 A/B 类具体人工贡献；`use_in_main` 标明主测评采用的最后一次明确意见。
- `papers/{paper_id}/shared/`：正文 IR 和最多八条共享 claims。
- `papers/{paper_id}/gear/`、`graph/`：GEAR、单条 Graph 和整篇联合 Graph 结果。
- `reports/{system}/`：八组整篇报告及按实际引用直接拼接的附录。
- `human_evaluation/`、`reviewer_consistency/`、`pairwise/`：人工参考对比、审稿人分歧和匿名成对比较。
- `summary.md`、`summary.json`、`tables/`：有效分母、缺失结果、组间和消融结果。

代码交接测试只使用合成论文和模拟模型，不会把合成论文加入正式研究名单。

Graph 批处理执行（需要先完成 `extract_claims.py`）：

```bash
STUDY=outputs/innovation_200_20260907
python3 experiments/innovation_200/run_graph.py --study "$STUDY" \
  --stage all --embedding-batch-size 32 --workers 32 --cli-limit 32
```

`prepare` 阶段跨论文收集未完成 claims，复用一个 Qwen3-Embedding-4B encoder（默认 `data/models/Qwen3-Embedding-4B`，与当前 2560 维图向量一致），以 `--embedding-batch-size` 为上限批量编码，沿用原有 cutoff、历史邻居、相似度阈值与父论文路径规则生成事实。编码全部完成后释放 GPU，再准备整篇联合结构事实；这个阶段不调用语言模型。显存不足时减小 batch size，例如 8。

`analyze` 阶段把所有单 claim 分析和所有整篇联合分析放在同一个线程池中，`--workers` 控制全局任务并发，`--cli-limit` 控制跨进程模型请求上限。联合分析只依赖图事实，无需等待单 claim 的模型结果。这里的语言模型批处理是集中并发独立请求，仍逐项校验输出身份与证据键。最终保持原有 `graph/analysis.json`、`graph/joint/analysis.json` 和报告路径，`run_all.py` 也使用这一入口。

可拆开运行或仅重试模型阶段：

```bash
python3 experiments/innovation_200/run_graph.py --study "$STUDY" --stage prepare --embedding-batch-size 32
python3 experiments/innovation_200/run_graph.py --study "$STUDY" --stage analyze --workers 32 --cli-limit 32
```

默认复用已有事实与成功的模型结果。准备失败时保存 `status/prepare_graph.json` 并以非零退出码停止；修复后重跑 `prepare` 或 `all`。模型阶段失败记录在 `status/run_graph_models.json`，分支汇总明确为 limited，可重跑 `analyze`；无合格历史邻居不调用模型，也不会生成肯定判断。`--overwrite` 仅适用于 `prepare/all`，将旧 graph 目录移入 `graph_attempts/` 后重建，保留原始证据。日志位于 `logs/run_graph.log`，逐任务模型用量位于 `status/usage/run_graph/`。

GEAR 运行期间，可单独并发运行不依赖它的两组报告及测评：

```bash
python3 experiments/innovation_200/generate_reports.py --study "$STUDY" --systems direct_llm graph --workers 4 --cli-limit 32 --wait-for-inputs
python3 experiments/innovation_200/evaluate_human.py --study "$STUDY" --systems direct_llm graph --workers 2 --cli-limit 32 --wait-for-inputs
```

两个阶段现在按“论文 × 实验组”分配任务；测评只等待对应组的报告。直接 LLM 和 Graph 报告不会读取正在更新的 GEAR 证据。省略 `--systems` 仍运行八组。新增后处理任务默认在 Linux `MemAvailable` 低于 5 GiB 时等待，可通过 `GEAR_POSTPROCESS_MIN_AVAILABLE_GIB` 调整；这属于启动前检查，不是进程硬内存限制，正在执行的模型调用不会被中断。所有阶段需使用相同 `--cli-limit` 才能共享一致的 CLI 并发上限。

历史文献 PDF 下载默认关闭，在项目 `.env` 配置：

```dotenv
GEAR_HISTORICAL_PDF_ENABLED=false
```

设为 `true` 时，对符合预算的历史文献下载 PDF 并在本地提取正文；设为 `false` 时，跳过全文补强及缺摘要时的 PDF 兜底，继续以可获取的摘要作历史对比，缺少证据时保留限制。投稿论文本身的正文分析不受影响。启动日志打印有效开关值；环境变量优先于 `.env`，`.env.local` 优先于 `.env`。修改后需要结束旧进程并重新启动，运行中的客户端不会热加载开关。

切换模式后可以在原目录不带 `--overwrite` 续跑，已完成论文、claim assessment 和 GEAR card 会复用。没有形成 card 的中断 claim 会归档原始证据后重试。新生成 card 的 claim 目录记录 `retrieval_policy.json`。已有全文证据和已生成结果不会删除或降级，因此续跑属于保留历史结果的混合模式；若要得到全体统一“仅摘要”的对照实验，应在独立输出目录重新运行 GEAR 及依赖它的报告/测评，Graph、共享 claims、人工参考与 direct_llm/graph 报告无需因此重新生成。

终端手动续跑仍使用原有入口：

```bash
cd /home/jayee/workspace/ASPR
CODEX_HOME=/mnt/c/Users/jayee/.codex \
GEAR_CODEX_SERVICE_TIER=fast \
GEAR_NETWORK_MAX_PROCESSES=2 \
GEAR_NETWORK_RETRIES=5 \
python3 -u experiments/innovation_200/run_gear.py \
  --study outputs/innovation_200_20260907 \
  --workers 16 --cli-limit 32 --verbose
```

先结束旧进程再执行，不加 `--overwrite`。启动日志应显示 `PDF下载=False`；无需额外启动脚本。

## 报告、测评与消融的解释边界

八组报告为 `direct_llm`、`gear`、`graph`、`fusion`、`fusion_no_joint`、
`fusion_no_metrics`、`fusion_no_citation_paths` 和 `fusion_text_only`。
直接 LLM 独立读取论文，其余组使用共享 claims 和对应信息条件。这里的整篇
`fusion` 报告使用 GEAR、单条 Graph 及联合 Graph；它不同于标准运行入口生成的
逐 claim `fusion/analysis.json`，后者没有自动纳入联合分析。

模型响应缓存关闭不等于已有阶段文件不复用。实验还关闭了输入指纹检查，不能将
所有续跑结果描述为统一冻结、逐文件哈希验证的发布版本。标准 `validate-run`
要求逐 claim fusion 和指纹，因此不适合作为这套报告实验的整体校验入口。

人工参考来自已发表审稿意见，A 类含明确创新判断，B 类仅支持贡献识别；模型执行
参考抽取、匹配和匿名偏好判断，不能将其标为新采集的人工偏好。审稿人没有提及
某项贡献不表示该贡献错误。最终论文与较早轮次审稿意见的比较还需解释版本差异。

现有消融是已实现的信息条件。一些变体以删减后的原始事实替换完整组的分支解释，
同时改变了信息与中间处理；正式归因前需要控制生成路径和相应资源预算。普通
RAG 与共享 claim 对照另见 `gear/innovation/experiments.py`，不属于当前八组报告。

新增 Fig.1–Fig.10 的分工、子实验及图形尚在讨论。这里的现成实验组不能直接等同于
已确认的新图方案，运行完成或结构检查通过也不等于已证明融合优于基线。
