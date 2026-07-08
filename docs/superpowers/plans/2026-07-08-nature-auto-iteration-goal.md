# Nature-Level Fig.1-Fig.10 Auto-Iteration Goal

## Objective

把 Fig.1-Fig.10 推进到 Nature 论文级别：数据证据可信、claim 边界清楚、视觉统一、信息密度高、主图读者能在短时间内读出科学贡献。执行时不在中途停下来让用户选择；所有可默认判断的取舍按本协议自动执行。

## Automatic Execution Policy

- 最多 6 轮主迭代 + 1 轮最终小修，防止无限循环。
- 每轮都按 `build -> audit -> reflect -> fix-list -> auto-continue` 执行。
- 只有硬 blocker 才停止：缺关键数据、外部接口不可用且无本地替代、生成结果无法复现、或强 claim 被数据否定。
- 软性选择一律默认处理：Nature 可信度优先，其次美观，其次 panel 数量。
- 输出目录使用 `outputs/nature_iter/r0/` 到 `outputs/nature_iter/r6/`；新一轮通过后删除上一轮旧 PNG/SVG/PDF，只保留 CSV、JSON、manifest、quality report、review notes 和 fix-list。
- 不生成“请选择路线”的中间报告；每轮 fix-list 直接决定下一轮动作，默认一路执行到完成、降级、转 supplement、pipeline-ready gap 或硬 blocker。

## Iteration Rounds

- Round 0 baseline audit：冻结当前 Fig.1-Fig.10 问题清单、数据来源、claim scope、style ledger。
- Round 1 data/claim repair：修 Fig.1 landmark 时间窗、Fig.2 真实 Fig.1 裁图接线、Fig.5 AI 热点数据门、Fig.4/Fig.10 claim 降级门。
- Round 2 layout redesign：压缩低密度 panel，修重叠越界，统一 Fig.1-Fig.10 色调、字号和 caption 语气。
- Round 3 evidence strengthening：核查 landmark、AI 热点、peer-review claims；无法核查的强表述自动降级。
- Round 4 density pass：以 5-10 秒可读性、panel 负载、信息密度和跨图叙事顺序重排主图。
- Round 5 targeted repair：只修 Round 4 未通过项，不重动已通过图。
- Round 6 final assembly and convergence：生成最终图集、caption draft、claim ledger、strict evidence report、submission readiness checklist；未完成项必须完成、降级、转 supplement 或列为 pipeline-ready gap。
- Final patch：只修 typo、轻微 label overlap、导出路径和 manifest 不一致，不再重构数据或设计。

## Round 1 Defaults

- Fig.1：每个领域必须有 landmark 前、当年附近、landmark 后、late/current 四段；首个子图不得含 landmark。
- Fig.2：panel a 必须来自 Fig.1 真实输出；panel c 合并或迁出；panel d 右侧标签不得越界。
- Fig.3：panel e 改成七指标联合贡献 fingerprint，不再用近零柱状图制造低密度视觉。
- Fig.4：peer-review alignment 证据不过门时自动降级强 validation claim。
- Fig.5：默认改为 2024-2026 AI/AI-enabled frontier；若本地数据不能支持 AI 热点，则输出 blocker/gap，不用非 AI 旧词条冒充。
- Fig.6/Fig.7：减少折线图，优先 atlas、matrix、badge、small-multiple evidence map。
- Fig.8/Fig.9：文字减量，图像承担结构说明；GPT-image 视觉层必须绑定 manifest。
- Fig.10：统一色调，减少 panel，突出 ablation evidence 与 replacement gates。

## Stop Criteria

- 所有主文候选图通过 layout、style、provenance、claim-scope gates。
- Fig.5 top AI terms 通过 AI relevance 和 2024-2026 evidence audit。
- 连续两轮只剩轻微视觉问题时，进入 final patch 并结束。
- 第 6 轮后仍失败的项目输出 blocker/gap 清单，不继续循环美化。
