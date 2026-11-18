# Fig.5 鲁棒性数据准备

100篇沿用最新Fig.4完整条件评价；排除5篇开发样本后固定抽20篇，进行E50、K5、Luna配对实验；
其中5篇完整重复。85个报告对象，65份新增报告。仅数据，不绘图。

```bash
python3 -m figure_pipeline.fig5_robustness prepare
python3 -m figure_pipeline.fig5_robustness run --wait-for-fig4
python3 -m figure_pipeline.fig5_robustness status
python3 -m figure_pipeline.fig5_robustness aggregate
python3 -m figure_pipeline.fig5_robustness package
```

`run --stage reference|generate|evaluate|all`可分阶段；`--paper-id`、`--condition`可重复。
输出固定为`outputs/fig5_robustness`；`--output`用于定向验证。写入命令使用进程锁，status只读。
默认不与Fig.4运行重叠，`--wait-for-fig4`等待其进程结束。每次恢复只复用成功产物，不自动换样本。

领域×稀疏分层配额为8/7/3/2，稀疏/饱和10/10；种子20261002。重复为生命2篇，其余各1篇，
稀疏3篇、饱和2篇。E50以DOI及本地身份去重，保留floor(N/2)，跨分支删除原文别名；
K5取原合格前5邻居，调用原生统计并重建联合图，保持来源池。历史图只读，不重新嵌入。

生成Sol 6.1/medium，LUNA条件Luna 5.6/medium；参考和评价Sol 6.1/high。
参考先于新报告；公共任务、适用性及要点ID固定，K5参考依据更新。私有参考不进writer。
评价匿名按论文×五方面进行，保留正确性、范围、证据及明确保留/合理性/无依据确定断言。
技术NA与科学未决分开；无依据断言率为已确认比例，并同时报告未决数。

最多295次常规请求＋30次额外请求＝325次；失败计数。临时网络错误至多重试一次，
不作格式修复或换模型。超容量评价按候选分批，新增批次占额外额度；不截断原文。
2026-10-03按用户要求提高为32槽位（`--cli-limit`可指定），保留4GiB内存余量，无新增自适应调度。

原始模型响应与事件在logs/calls，复用Fig.4输入与调用记录在baseline。
call_ledger只记录Fig.5新增尝试，历史调用用于成本归属而不计入新增支出。
包内保留实际模型调用record及所有分析源表，模型事件流留在本地。

100篇分层结果与20篇统一新评价分开；不混用Fig.3旧评分。不把抽样波动区间解释为评委准确率。
成本仅为固定材料下分析写作，非建库、检索、嵌入或理想无失败端到端成本。

定向检查：`python3 -m pytest -q tests/innovation_v2/test_fig5_robustness.py`。
