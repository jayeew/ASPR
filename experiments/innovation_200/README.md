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

此入口的新模型请求固定使用 `gpt-5.6-luna`、`low` 推理强度和 `fast` 服务层，
同时覆盖 claim 抽取、合并、核验及审稿参考抽取、分组角色。启动日志和逐次模型
调用日志都会打印实际模型参数；并发和实时进度打印保持不变。它只设置两个子阶段
的进程环境，不修改其他独立阶段的默认 Luna 配置。已完成文件仍跳过，因此保留
旧模型或不同推理强度结果的续跑属于混合配置数据，不能称为全体统一 Luna/low 生成。

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

也可用 `run_all.py` 编排这些阶段。claims 完成后，GEAR、Graph、报告、人工测评和匿名比较作为独立进程流式衔接；一篇论文的上游文件齐全后即可进入下游，不等待整个当前名单完成。CLI 总并发默认 16，可显式设为 32、48 或 64。文献网络并发默认 16，可通过 `GEAR_NETWORK_MAX_PROCESSES` 提高到 32。Graph 先集中批量准备图事实，再使用统一模型任务队列；GEAR 默认两个论文工作进程；所有 Codex 请求仍受跨进程总并发锁控制。独立阶段默认使用 `gpt-5.6-luna`，角色决定 low/high；上面的 1,000 篇入口对其子阶段统一覆盖为 Luna/low。实验配置关闭模型回答缓存、关系稳定性复判和输入指纹核对，失败请求最多重试两次。

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

`GEAR_HISTORICAL_PDF_ENABLED` 控制原 OpenAlex Content 下载通道，默认关闭。新增独立开关 `GEAR_EXTERNAL_FULLTEXT_ENABLED` 控制开放来源全文；一键入口及本研究 `.env` 已开启，运行库默认仍关闭。外部模式启用时优先使用外部来源，失败也不会回退 OpenAlex Content。两个开关都关闭才是纯摘要模式。投稿论文本身的全文分析不受影响。

外部模式按候选排序顺序，每条 Claim 最多补强 `GEAR_EXTERNAL_FULLTEXT_MAX_WORKS=2` 篇；缓存按 DOI/Work ID 跨 Claim 复用，先尝试开放位置及 PMC XML，必要时用 DOI 查 Europe PMC。下载与解析前后有文献身份、字节、页数和时间检查；全文采用独立队列，跨进程最多并发 2 个下载任务。请求失败保留摘要和获取失败记录。OpenAlex 成功响应缓存默认保留 7 天，全文成功缓存 30 天、失败缓存 1 小时；两类缓存都不保存 API Key。全文原文和实际来源进入追加式证据记录，模型使用抽取片段，后续报告也可引用这些全文来源。检索仍消耗 OpenAlex 搜索额度，开放全文不经过 Content 计费。

本地排序模型跨论文共享；候选判断按每批 8 篇分组，默认并发 2 批（`GEAR_CANDIDATE_GATE_WORKERS`），仍受全局 CLI 上限控制。网络等待、模型调用和本地排序可跨 Claim/论文交错推进。修改配置后需重启进程；环境变量优先于 `.env`，`.env.local` 优先于 `.env`。

切换模式后可以在原目录不带 `--overwrite` 续跑，已完成论文、claim assessment 和 GEAR card 会复用。没有形成 card 的中断 claim 会归档原始证据后重试。新生成 card 的 claim 目录记录 `retrieval_policy.json`。已有全文证据和已生成结果不会删除或降级，因此续跑属于保留历史结果的混合模式；若要得到全体统一“仅摘要”的对照实验，应在独立输出目录重新运行 GEAR 及依赖它的报告/测评，Graph、共享 claims、人工参考与 direct_llm/graph 报告无需因此重新生成。

终端续跑仍由 `experiments/innovation_200/run_gear.py` 执行；可用一键入口：

```bash
cd /home/jayee/workspace/ASPR
CODEX_HOME=/mnt/c/Users/jayee/.codex bash scripts/run_gear_resumable.sh
```

该入口默认仅从开放来源补历史全文，不请求 OpenAlex Content PDF；每条 Claim 最多补强 2 篇，保留摘要回退。默认 2 篇论文并发、2 条 Claim 并发、CLI 上限 8，论文间共享一个排序模型，开始新论文前要求可用内存至少 5 GiB。内存检查不是硬上限。可用 `--workers 1 --cli-limit 4` 降低并发，或通过 `GEAR_EXTERNAL_FULLTEXT_ENABLED=false` 暂时仅用摘要。所有参数传入原 Python 入口；脚本不自动清理结果。

普通续跑会复用已有 `analysis.json`（包括科学证据受限的 `limited`），不会循环重做。`limited` 不等于网络或模型故障；状态详情中 `branch_status` 表示科学结果范围，`execution_status` 表示技术执行情况；旧产物缺少执行记录时为 `legacy_unknown`，不能推断全部请求成功。

检索或模型故障恢复应优先使用：

```bash
CODEX_HOME=/mnt/c/Users/jayee/.codex bash scripts/run_gear_resumable.sh --retry-failed
```

`--retry-failed` 只归档有明确执行错误、缺失或损坏证据的 Claim；科学上 `limited` 而没有执行错误的分析保持复用。证据卡和 trace 完整、仅最终 assessment 缺失或损坏时，保留证据卡，只重做总结，避免重复检索。只有故障影响的论文分析和下游融合/报告/测评失效。也可加 `--cleanup-only` 只完成恢复整理、暂不调用模型。

只有明确要重做受限结果时，先停止旧进程，再单独执行一次清理：

```bash
python3 experiments/innovation_200/run_gear.py \
  --study outputs/innovation_200_20260907 --clean-limited --cleanup-only
```

清理将当前受限/非法/未完成 Claim 的完整目录移入 `status/gear_recovery/<时间>/artifacts/`，保留无局限、身份与证据引用校验通过的 Claim，以及已完成且非 `limited` 的整篇结果。归档清单记录每篇保留和待重试的 Claim，不删除或改写原始证据。相应旧融合、GEAR 依赖报告/测评及汇总也归档，Graph、direct_llm、共享 claims 与人工参考保留。旧 `--retry-limited` 参数现会报错，避免每次启动反复清空进度。`--overwrite` 仍会完整归档 GEAR 后重跑，应谨慎使用。

按已保留结果续跑即再次执行一键入口。调试一篇论文可执行：

```bash
CODEX_HOME=/mnt/c/Users/jayee/.codex bash scripts/run_gear_resumable.sh \
  --paper-id s42003-026-09574-2 --workers 1
```

`--paper-id` 可重复，所选论文必须存在于 `papers.jsonl`。选样运行使用独立的 `status/run_gear_selected_<时间>.json` 和日志，避免覆盖全量状态。启动及清理共用 study 写锁，重复启动会立即报错。检索或模型发生实际失败时退出码非零，科学局限本身不作为进程执行失败。重跑后仍可能科学上受限；不要为清除 `limited` 标签反复清理。保留旧结果并使用新全文流程属于有意的混合模式恢复。

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

### 2026-09-08 对照检索覆盖修复与定向恢复

普通检索曾占满每 Claim 的 5 个候选名额，导致后续 `legacy_contrastive`
在发出网络请求前被候选上限拦截。当前 study 为普通检索和对照检索分别保留
最多 5 个候选，总计最多 10 个关系判断；每 Claim 的外部全文获取仍最多 2 篇，
不随候选预算增加。重复的对照查询会尝试去掉新方法机制，使用对象、问题和传统
术语生成不同的检索意图；无法构造有效查询时仍明确保留覆盖缺口。

旧运行使用 5 个候选，新修复使用最多 10 个，因此属于有记录的混合条件恢复，
不应作为完全一致的预算实验。严格的条件比较应另建输出目录。

已有研究的定向恢复（只需准备一次）：

```bash
bash scripts/run_gear_resumable.sh --repair-coverage --cleanup-only
bash scripts/run_gear_resumable.sh
```

恢复仅限 `gear_papers.jsonl` 中的原 200 篇。旧 Claim 目录和受影响的论文汇总、
下游结果会归档到 `status/gear_recovery/<timestamp>/artifacts/`，不会删除。
健康但缺少对照检索的 Claim 留下 `recovery_source.json`；运行时复用归档中的
稿件证据、文献、已获取全文和关系判断，补做对照检索并重新生成该 Claim 总结。
技术失败或不完整的证据阶段需要重试，成功的检索和全文缓存继续复用。
未缺少对照检索的科学 `limited` 结果不因该标签被清理。

补检索继承旧候选数时，由于旧追踪没有保存全部被丢弃的候选 ID，合并后的
`unique_eligible_count` 使用可证明的下界 `max(旧候选数, 当前已知唯一ID数)`，
并在 advisory 中标注；不把两轮计数直接相加。旧追踪保持原样，新证据写入新尝试。

`run_gear` 状态记录分别提供：

- `execution_status`: `completed` / `completed_with_errors` / `legacy_unknown`。
- `coverage_status` 与 `coverage_by_claim`: 是否达到覆盖条件及缺失检索角色。
- `scientific_limited`: 是否存在科学适用边界。

原 `analysis.json.status=limited` 保留既有语义，不再把它当成运行失败计数。
执行完成不代表检索穷尽、没有科学限制或已证明首次性。

2026-09-09：一键入口默认采用 4 篇论文并发、每篇 2 条 Claim、最多 12 个
Codex 调用；检索并发仍为 2。新论文启动前要求至少 8 GiB 可用内存。
这是启动检查，不会中止已在执行的调用，也不是操作系统硬内存上限。
所有论文仍共享同一个带锁的 GPU 排序器。已有进程需要正常退出后重启
才能采用新配置；完成结果自动复用，勿使用清理或覆盖参数。可通过
`GEAR_PAPER_WORKERS=2 GEAR_CLI_LIMIT=8` 临时恢复较低并发。
