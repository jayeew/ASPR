# Fig4 百篇扩展

旧结果只读；新事实输入、模型结果与账本均在 `outputs/fig4_explanation_100/`。

```bash
python -m figure_pipeline.fig4_explanation_100 prepare
python -m figure_pipeline.fig4_explanation_100 run --pilot --stage reference
python -m figure_pipeline.fig4_explanation_100 run --pilot
python -m figure_pipeline.fig4_explanation_100 aggregate
python -m figure_pipeline.fig4_explanation_100 render
python -m figure_pipeline.fig4_explanation_100 package
python -m figure_pipeline.fig4_explanation_100 status
```

参考完成后先阅读公开问题是否泄露答案、原文与参考是否对应，再继续生成。用户审阅通过后才记录 `pilot_acceptance.json` 的 accepted=true 及审阅说明，使用 `run --remaining`。不得根据完整系统排名自动通过。

`--paper-id`、`--condition` 可重复；`--stage reference/reports/evaluate/all` 指定阶段。已完成文件直接复用；完整原始响应可本地恢复。口径变更必须明确归档受影响产物后重跑，不自动失效、不清理旧目录。

模型统一 Sol 6.1，生成 medium、参考/评价 high。常规1800、额外180、总1980次上限；失败请求也计入账本。只对明确临时网络错误或429做一次原请求重试，无格式修复、隐式重试或压缩模型调用。64线程，初始32模型槽位、4GiB内存预留、8成功增加8槽位、429减半；依赖等待不占槽位。

技术缺失NA；不适用不进入分母；科学未决不算确认正确。内容正确覆盖率与外部证据落实率分别输出。七配置绝对比较取七配置共同完整论文；配对比较取对应完整配对；论文等权、10000次bootstrap。另给出排除五篇开发样本的源表。

没有运行或编写回归测试；工程核对阅读代码与真实产物。科学评价是本实验数据。

## 当前审阅边界

五篇公开问题与私有参考先于候选报告形成。THz、TDP-43没有正向引文接触，两个路径任务不适用。TDP-43先前完整图解释存在服务端内容限制：保留该技术状态，其他已授权独立条件仍执行；该篇不进入要求七配置共同完整的绝对比较，但可用的其他配置不丢弃。

已提供的 `render` 是原五篇审阅预览；百篇正式版须在用户通过审阅并完成其余队列后，加入真实案例、联合独有边与可重建性注释，不把预览当作百篇完成图。

## 联合图与结构数值的五篇修订

`python -m figure_pipeline.fig4_explanation_100.refinement prepare/reference/run/aggregate/render/package/status`

该入口只操作原五篇，输出到 `outputs/fig4_explanation_100/pilot_refinement_v2/`。H/K/P参考不改，J/S从同一组原文与图事实重新定义；共同writer加入证据条件保留检查，全部配置共用，仍为一次生成。每次新增调用同时写入本轮账本及百篇主账本，并占用180次补充请求预算。旧结果保留，既有服务限制不通过改写任务绕过。

J题区分共同邻居、间接历史边与整篇联合组织；S题区分加权分布、社区间语义差异、局部连通和历史组合频度。科学等价答案必须保持同一待解释对象与量的含义，仍允许正确重建。没有更改删除规则或增加参考答案到writer。该版是开发校准，不把与上一版评分差解释成同一量表上的性能改善。

## 已批准的95篇扩展：仅准备数据

用户已批准复用修订五篇并追加95篇。使用 `python -m figure_pipeline.fig4_explanation_100.expansion prepare/run/aggregate/package/status`；不调用render。

沿用 `pilot_refinement_v2/` 中五篇实际结果，追加其余95篇材料和结果。新增论文用一次参考调用建立全部十个问题，采用修订J/S构念；联合图统一评分尺度直接纳入首次评价，不安排固定复评。H/K/P一般判据不变。支持 `--paper-id` 和 `--stage`。常规请求计入1800额度，临时网络恢复计入180补充额度，共享1980上限。

`package` 仅交付数据与运行记录到 `outputs/fig4_explanation_100/Fig4_100_data_v2.zip`，排除旧五篇图件、旧审阅汇总及模型事件流。旧五篇审阅压缩包保留；正式统计包含全体及排除五篇开发样本两套表。
