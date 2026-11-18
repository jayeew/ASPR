# 原5篇：以具体科学问题解释七配置的消融

目标是说明GEAR的独立原文核对、Graph的知识关系以及Joint/结构数值/引用路径各自提供什么。
仅使用原5篇。先用原文和图材料审查图的科学用途；无实际图用途的论文排除并公开原因，
不参考七条件输赢，不引入替代论文。没有正向路径记录单独记该方面不适用。

每篇先建立10个具体参考问题（每方面2个），每问题两个有依据的答案要点。
参考由GPT-6-Astra/xhigh依据原文和原生图事实生成，主助手阅读核对；不是人类专家标注。
参考不提供给GEAR、Graph解释器或共同writer。通用写作任务对七配置相同。

现有Luna报告按新问题评价，另生成一轮处理路径统一的报告：

- GEAR解释器：论文、已有共享贡献及GEAR已有检索原文，重新做证据核对，独立于Graph。
- Graph解释器：论文、实际配置的原生图和已有Graph原文。完整图和三个删除视图均重解释。
- 完整Graph解释供仅Graph和完整系统复用；GEAR解释供含GEAR条件复用。
- 纯论文报告只收论文。仅GEAR不补Graph，仅Graph不补GEAR；无组件不添替代分析。
- GPT-6.1-Sol/xhigh承担新解释和共同写作。字段删除继承已确认口径，路径标签直接删除。
- 保留的节点/边若能推导移除的数值，正确推导仍计正确，不能为了制造下降禁止它。

评价使用GPT-6-Astra/xhigh，按每篇每方面同时盲评7份报告，逐答案要点判定正确、错误、
未写出或未决，并分开记录实际使用的原文/图/稿件证据。报告引文以段落ID本地恢复。
评价器知道各匿名报告实际可见的材料类型，避免用完整评价参考反驳报告的真实材料限制。
不使用整体0–3评分，不合成总分，也不要求完整系统获胜。

原任务为5份参考、25份分支解释、35份报告，以及两组各25个方面评价；全不适用方面本地
输出NA，不请求模型。另有参考修订、失败恢复及THz三次定点诊断，实际计数以账本为准。
实际失败、用量、已完成响应均保留。没有95篇任务入口。

```bash
python -m figure_pipeline.fig4_explanation_study reference
python -m figure_pipeline.fig4_explanation_study evaluate-existing
python -m figure_pipeline.fig4_explanation_study analysis
python -m figure_pipeline.fig4_explanation_study reports
python -m figure_pipeline.fig4_explanation_study evaluate-new
python -m figure_pipeline.fig4_explanation_study aggregate
python -m figure_pipeline.fig4_explanation_study status
```

不同阶段按依赖顺序执行；reference完成后先检查实际参考和纳入理由。
客户端、HTTP、JSON记录复用现有工具；不新增全量检索、下载、建图、信息聚类或风险追踪。
工程核对依据代码逻辑与实际实验产物，不编写或运行回归测试。

各阶段支持`--paper-id`指定原名单中的对象。已有完整响应本地恢复；失败请求只有查阅
记录后显式传入`--retry-failed`才再次提交，原失败记录保留，不因失败跳过论文。
`aggregate`同时给出预先选定问题的内容覆盖、外部证据落实、逐要点原句与完整配对差异，
以及局部图卡与联合图的真实信息重叠。这些是探索性问题覆盖，不能当成整篇报告总质量。

本轮完成状态：128次请求，119次完成、9次失败尝试留档；24份分支解释、33份新报告。
TDP-43完整Graph请求受到服务端限制，其新仅Graph／完整系统报告缺失；不自动重试此对象。
原35份报告全部完成评价；新报告按四篇完整七配置评价，TDP-43五份部分报告保留。
THz的`joint_probe.py`仅包含同题两次作答与一次评价，结果单列，不合入主指标。
参考开发修订和最终89个适用答案要点的说明见输出目录的《实验口径.md》。
