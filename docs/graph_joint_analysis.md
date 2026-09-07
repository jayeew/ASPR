# Graph 单条及整篇联合分析

当前实现说明（2026-09-07）。本页描述原生 Claim Graph 的运行机制；实验是否完成及其有效分母须读取对应研究输出，不能由代码实现情况推断。

## 邻居与父论文引用

默认配置 `graph_top_k=10`、`graph_min_similarity=0.5`。在排除目标论文自身与截止日期之后节点的历史语义候选中，最多保留10条且严格要求 cosine > 0.5。不足10条不补齐；没有合格邻居时明确返回受限结果。

通过 `data/claim_graph/canonical_target_works.parquet` 查询历史近邻的 Nature article ID → OpenAlex Work ID 映射（当前24919条）。匹配成功后，用目标论文输入中的 reference_work_ids 与本地 paper_graph_index.sqlite 查询直接引用、两跳路径、共同参考文献。缓存未命中时保留语义边，记录 parent_id_cache_hit=false。命中ID本身并不证明引用存在；只有实际查到父论文路径才标记 semantic_and_paper_path。

此规则沿用 scripts/claim_graph/07_build_claim_graph.py 的 paper_path_edges：父论文路径增强既有语义边，不凭引用单独建立所有跨论文claim组合，也不把论文引用当作claim层面的支持/推导证据。

## 整篇联合插入

保留原有逐条Graph分析。在graph或all阶段，单条分支执行后自动增加graph_joint：

1. 合并所有单条插入事实卡的历史近邻，按claim ID去重。
2. 从只读历史图查询整个合并邻域的语义骨架边，包括不同单条邻域之间原有的边。
3. 在临时联合视图中同时接入全部有事实卡的本文claims，不修改历史图。
4. 计算历史邻居接入前后的连通部分、新增可达历史节点对、不同claims共享的邻居及经历史图的连接情况。
5. 不简单累加单条指标，也不凭同属一篇论文就在新claims之间造边（历史建图排除了同论文语义边）。缺失事实卡的claims单独列出，联合结果标为limited。
6. graph_analysis模型角色读取所有原始Graph事实卡及联合结构，独立解释知识基础、贡献的互补/重复、共同解释能力与可能的发展关系。观察和解释分字段存储；模型输出必须绑定已有claim IDs及证据键。不给模型GEAR分析，也不要求每种概念联系必须有引用。

输出位于 `graph/joint/`：facts.json、analysis.json、report.md、status.json及append-only证据存储。联合结果是论文层级输出，未强行写入每条claim的融合标签。

## 缓存与指标可比性

新图分支保存insertion_policy.json，旧无阈值结果不能直接作为新规则结果复用；使用新运行目录。联合分析按共享claims、事实卡、配置及版本绑定指纹，输入改变时拒绝覆盖原结果。

14项单条原始指标保留。默认阈值规则下不使用旧选择规则生成的历史百分位；重新校准前不能把新原始值与旧百分位混用。社区及历史图保持原状，没有重新聚类或训练。

## 与默认运行和整篇报告的关系

`python3 -m gear review` 默认使用 `knowledge` 模式。单论文 `all` 顺序执行
GEAR、Graph（含联合 Graph）、逐 claim fusion；两条分析分支在信息上独立。
标准 `fusion/analysis.json` 和 `fusion/innovation_report.md` 按 claim 融合，
不会自动读取联合 Graph。`experiments/innovation_200/reporting.py` 另行生成整篇
报告，其完整 `fusion` 组会读取 GEAR、单条 Graph 和联合 Graph 解释。

运行配置默认启用续跑指纹检查；分阶段研究配置明确关闭了该检查。因此，本页所述
指纹保护适用于检查启用的运行，不应写成所有研究续跑都强制校验。无论指纹设置
如何，仍须保留证据与缺失状态，且不应将混合输入或混合策略结果称为统一对照。

当前系统不调用旧 paper-level HGB 分数来给 claims 分配创新性分数，也没有旧的
ASPR-Qwen 审稿分支。Qwen3-Embedding-4B 是图向量/召回组件；它与旧审稿模型不同。
