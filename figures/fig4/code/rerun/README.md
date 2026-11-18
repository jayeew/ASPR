# Fig4 最小成本重跑

本目录实现用户确认的七种直接删减配置。旧 Fig3/Fig4 结果只读，新结果位于
`outputs/fig4_rerun/`。不调用替代分析、材料摘要、逐贡献整理、聚类或融合追踪。

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_rerun prepare
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_rerun run --pilot
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_rerun status
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_rerun aggregate
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_rerun render
# 剩余95篇必须等用户审阅试跑、明确授权后，再运行 run --remaining。
```

报告一次调用使用 gpt-5.6-luna/high；两条核心贡献合并评价一次，使用
gpt-5.6-luna/xhigh。试跑最多70次，总队列最多1400次。
call_ledger.jsonl 记录所有任务启动，失败同样占预算。不会自动重试、修复模型格式
或调用模型压缩材料；已启动失败对象续跑只恢复已有有效响应，不重复请求。
HTTP provider 的 request_max_retries 与 stream_max_retries 均为0，沿用已有登录。
配置依据：https://learn.chatgpt.com/docs/config-file/config-reference。

纯论文输入仅含 manuscript 字段。Graph 输入统一是结构化事实，不携带旧自由解释；
移除条件通过字段删除实现。历史原文按现有模块引用或图邻居来源选择，保留位置与
摘要/全文类型；本地选择片段不等于完整检索。所有论文正文保持完整，无长度目标。

评价只使用固定核心与参考和原文，不使用条件自身的分析作答案。C为核心回应覆盖率，
H为有回应且历史比较和必要范围均正确的覆盖率，S为范围正确回应率。
C固定200条核心，H适用200条，S适用199条；逐论文计算后等权宏平均。
未回应不计分，科学未决不计确认正确，技术失败保留缺失，不填零。
不再输出旧V信息簇统计或旧R全报告不当断言率。

纸面配置：T纯论文，E仅GEAR，G仅Graph，F完整输入，F_noJ移除联合图，
F_noM屏蔽非路径结构数值，F_noP屏蔽引用路径。图件沿用Fig4 reference样式，
a–d仅展示本次完成队列；e–f复用旧Fig3描述性数据并明确分区。

正文、评价、调用原始响应和失败记录保留；pilot_review.md及full_lower_examples.json
供用户审阅完整系统落后的原句与依据。范围改变或补跑必须另行商议。
报告引句允许本地恢复字体、空白、Markdown和列表编号；科学句子逐字匹配、但
段尾引用标记被移动时，恢复为同一原段的完整连续片段。保留原始响应以及
quote_restorations记录；科学文字的实词增删或改写仍记为技术缺失，不进行模型定位。
condition_summary.csv保留各配置实际有效数；condition_summary_common.csv只用
七配置共同完整论文，供panel a比较。panel b–d按各自完整配对计算，不把缺失当零。
pilot_issues_for_review.md列出评价尺度疑点与待确认处理，unresolved_evaluations.json
保留未对齐的原始响应及仅供人工审阅的邻近段落。
工程校对只阅读代码逻辑和实际试跑结果，不编写或运行回归测试。

用户授权统一覆盖标准后，5篇的35份报告保持不变，七配置各重评一次，共新增35次：
`run --pilot --reevaluate-pilot`。旧评价/汇总/图件保存在previous_evaluation/v1；
新评价为v2_scientific_content，按明确科学对象、结果及必要条件识别同义回应。
遗漏非决定性的数字不自动判未覆盖或范围错误；科学内容缺失仍不覆盖，历史正确性
独立判断，具体性能优势依赖的数字仍是必要内容。引用采用短的连续原句。
用户随后明确：1400次约束最终采纳版本的100篇任务量，而非全部历史请求。
采纳的5篇为35次原报告生成+35次v2评价=70次；旧版35次评价单独保留历史记录。
剩余95篇获授权新增1330次。adopted_entries只计采纳版本，历史账本不删除，
status同时公开采纳次数、全部历史次数及被替代次数。失败请求仍计入当前版本预算，
不会自动额外请求。pilot_acceptance.json记录用户采纳32份有效评价及3份缺失的试跑。
