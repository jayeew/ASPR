# 统一入口

逐图代码与数据已迁移到 [`../figures/`](../figures/README.md)。本目录保留共享工具和原 Python 模块名的符号引用；没有第二份实现。下面的历史命令继续有效。

# Fig1–Fig11 绘图入口

当前正式图、逐图工作目录、直接导出方法和数据依赖统一维护在 [输出与重绘指南](../outputs/README.md)。Fig5 使用机制研究四面板，Fig7 使用包含国家比较 c、d 的四面板，Fig10 使用 MOF 案例 C03–C06。此前的清理依据见 [2026-10-08 依赖审计](../docs/fig1_8_cleanup_audit_20261008.json)。

## Historical Figure Pipeline v2

以下为历史 FROM_WEB_v2 工作流，其结果按历史版本解读。正式 Fig1–Fig11 请使用上方逐图入口。

本次修订保留 `outputs/FROM_WEB` 和旧 ZIP，使用冻结 200 篇快照；输出为 `outputs/FROM_WEB_v2`。这是可审计重绘与方法修复，不是新增完整 GEAR/Full、人工标注或前沿预测实验。

运行：

```bash
MPLCONFIGDIR=/tmp/aspr-mpl python3 -m figure_pipeline.build --mode draft
python3 -m figure_pipeline.build --mode publication
python3 -m pytest -q tests/figure_pipeline
```

第二条命令在缺核心证据时应失败，不能把草稿当成已完成的论文主结果。Fig5/6/7 的效果图不由准备度面板替代；运行覆盖集中在 Supplement S1。

模块：`benchmark_revision.py` 负责配对统计及状态分离，`graph_revision.py` 负责图结构清理与诊断，`case_revision.py` 负责机制、基线、期刊与真实案例；`specs/contract.py` 定义证据门槛，`qa/checks.py` 分开检查格式、物理字体和研究证据状态。输入冻结文件通过 SHA-256 检查未被修改。新独立核验/受控实验协议见 `specs/research_protocol.md`。

科学边界：Graph + joint 不等于 Full fusion；已保存 AI 审稿匹配不等于独立人工正确性；n,m 零模型不是创新真值；标题短语热度不等于知识前沿。
