# GEAR 收缩重构基线（2026-08-27）

本文件冻结重构前可复核事实；不把旧评测继续当成科学结论。

## 旧状态

- `make gear-test`：173 通过、15 失败；其中 7 个失败来自缺失的
  `prepublication_graph_v3:d5_fulltext16` release，8 个来自旧的全局
  positive/mixed 方向修正规则。
- Nature dev100 的旧 HGB 分数中位数约 90.10；整数查询槽位使 96/100
  论文得到相同的 1 local + 2 remote 行为。
- 旧 Nature dev100 没有 prior-art gold；旧重建共有 483 条问题：resolved
  264、persists 124、partially_resolved 74、unverifiable 21，其中仅 20 条
  novelty_prior_art 和 4 条 contribution。
- 旧重建不具备逐问题 reviewer quote、reviewer/round 和最终论文跨度的
  完整绑定，因此只保留为历史 silver 基线。
- 旧 2022 图上下文不能用于 cutoff 不晚于 2022 年的历史案例。

## 冻结哈希

- 旧 dev100 指标：
  `sha256:acc196166a33c64e68b99cbda7b9e8d8f0ba4afed8eca5b8323e6f41dd33971c`
- 旧 dev100 问题标签：
  `sha256:ad1f892895a2a57b055c40c3358fae2a06a5f8a1d5f068a7fd2e296c7c667ab2`
- Fig.3 v6.1 OOF：
  `sha256:1e5f9024c0fe348ab435eb7952fa9984ba2444d6fdc177d5efc311a5648827f3`
- 新 D5 release manifest：
  `sha256:c56e04e3118f0752ba611b26f3ee0b2f54d98bb10c08287e435de1d096d8be76`

## 目录规模

- `gear/` 约 5.0 MB；`tests/gear/` 约 1.2 MB；`experiments/gear/` 约 1.1 MB。
- 重构前 `outputs/gear/` 约 1.8 GB。
- 新唯一 D5 release 约 80 MB。

## 删除原则

- 删除原因是接口已失效、资产缺失、重复兼容或结果可重建，而不是旧结果
  “不好看”。
- 在新 release、核心测试和资产回放通过前不删除旧运行目录。
- 人工 3 篇闸门没有真实人工标签前，不宣称 Graph 已通过；也不扩展到 10 篇。
