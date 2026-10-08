# Fig4 final reference

当前正式图与数据统一位于 `outputs/fig4_reference/`。入口转发至已采纳的 `fig4_explanation_reference` 绘图实现，不再调用本目录旧版 data/render/export。

```bash
/home/jayee/miniconda3/envs/openrlhf/bin/python -m figure_pipeline.fig4_reference all
# prepare / render / package
```

`experiment/`保留本轮百篇正式数据，旧版实验输出已清理。仓库中的历史模块因被当前实验和Fig5导入而保留；不要直接调用旧版模块重新生成旧图。
