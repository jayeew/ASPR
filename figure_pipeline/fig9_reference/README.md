# Fig.9 — Evidence Supervisor 工作机制

根据当前 `gear/evidence_supervisor.py`、用户提供的 V3.0 绘图说明与参考图绘制。
采用 Fig.1 的 `Scene` SVG 图元及 `fonts`、`outline`、`convert` 导出链，
使用 Times New Roman，画布 420 × 240 mm。所有图元及文字可编辑，未嵌入参考位图。

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig9_reference
```

输出：`outputs/fig9_reference/final/`，含普通 PNG、600 dpi PNG、PDF、可编辑 SVG、
文字轮廓化 SVG；`qa.json` 记录源码哈希及画布边界检查。

这是当前运行机制的示意图，不是历史 Fig.9 checkpoint 实验结果，不调用模型或检索。
微型进度条、候选排序条、facet 标记及覆盖检查符号均为示意，不表示观测数据。

相对参考图的代码事实修正：

- Finalize 走独立终止路径，不返回下一轮。
- 无先例且存在可扩展文献时，合法集合为 `citation_expand`、`finalize`。
- 六行合法动作按优先级匹配；未分类文献先进入关系核验。
- 稳定性检查必须有先例、配置开启且尚未执行；与直接先例独立复核不同。
- 覆盖审计在循环后执行；覆盖不足不能解释为无创新。
- bounded_no_antecedent 还要求存在关系证据，所有关系时间有效且无未决关系。
- 图中表述 schema-constrained selection；实际动作限制由传入规划器的 JSON schema
  enum 描述。图不额外声称存在独立的运行时成员校验。
- 分支 Assessment 后的 Fusion 还需要独立 Graph 分支的输入。

图内箭头表示控制或证据流；五列并非每个 claim 无条件执行一次的固定流水线。
内部不支持的 claim 跳过外部检索。Evidence ledger 是来源追溯示意，
不是所有中间对象都直接作为 EvidenceStore 中的同名 record 写入。

本次外观修订：对照原示意图补齐矢量图标与彩色状态标签，缩小执行卡片以增大箭头间距，分离下行派发与上行反馈通道；删除标题下灰色脚注。文字检查覆盖画布边界、文字间相交和 Supervisor/执行列卡片边界，并目视检查箭头。旧版预览保存在 outputs/fig9_reference/previous_v1/。

连接复核：检索派发只连接 Step02（normal）和 Step04（contrastive/expand）；
核验派发只连接 Step03（verify/stability）；finalize 独立连接 Step05，不回流。
Step02/03/04 的结果由青色虚线汇入 S1 下边框；所有派发/反馈端点接到卡片真实边框。
交叉处采用带底色遮罩的跨线弧，表示不连接。移除了 S4→S1 的装饰捷径以及
Step04→Step03 的直接短箭头；新候选由 Supervisor 在下一轮重新派发核验。
qa.json 记录连接映射与边界坐标。
