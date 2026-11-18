# Fig8 统一数据

这是 Fig8 实验唯一的当前数据入口。后续直接追加、覆盖、合并现有表；不再创建数据版本目录或要求保留父版本。

| 用途 | 当前表/报告 |
|---|---|
| Nature 主刊论文、机构与国家/高校分配 | `nature/papers.jsonl` |
| Nature 含采集字段的工作表 | `nature/research_roster_release.parquet` |
| Nature 完整性与统计口径 | `nature/完整性报告.md`、`nature/protocol.json` |
| 三个领域论文元数据主表 | `fields/papers/*.parquet` |
| 类型、在线年份、去重及纳入判定 | `fields/qualification/decisions/*.parquet` |
| 当前已核实研究论文清单 | `fields/qualification/research_roster.csv` |
| 资格待核与元数据补齐队列 | `fields/qualification/unresolved.csv.gz`、`research_metadata_gaps.csv.gz` |
| Fig8b 当前完整性报告 | `fields/qualification/完整性报告.md` |

论文主表与资格表按相同分片内的 `work_id` 关联。全候选覆盖与研究子集覆盖分开；
正式纳入以资格表的 `eligibility_status` 为准，不使用采集时的宽泛 Article 标签代替。
Nature 的署名槽位与全期刊的观测作者粒度不同，两者是同一实验的关联表，不能直接相加作者分母。

## 候选池与逐刊比较的分母

三个领域数据池最初按 OpenAlex 的 2023–2025 年发表日期、英文、期刊来源、
Article/Letter 类型及主要主题领域筛选：生物化学/遗传学/分子生物学（13）、
材料科学（25）、物理与天文学（31），并排除当时已标记撤稿或旁文本的记录。
这些数据库字段仅用于发现候选；最终资格仍以匹配的出版物证据、明确研究类型、
在线年份和去重判定为准。后续追加仅限既定范围内经核实的漏收记录。

“覆盖全期刊”表示不按期刊声望筛选上述领域内的候选论文，不表示已收齐每本期刊的全部发文。
Fig8b 每个点代表该期刊在相应领域及年份内的合格论文集合；Nature、Science、Cell、PNAS
在本表中的数量不能直接解释为各刊整体发文量。已核实数量少也不能证明该刊研究论文少，
必须结合剩余资格缺口和来源覆盖评估比较可用性。已纳入不等于全文、机构映射或创新分析已完成。

病例报告按用户明确选择排除主比较、保留扩展口径，状态为
`excluded_case_report_main_scope`；仅标为 Case Series 的记录不自动套用此规则。
MRA 的 Genome Sequences 同样保留扩展口径，状态为
`excluded_resource_announcement_main_scope`。两类记录仍在原主表中；扩展分析须另检查
在线年份、数据完整性和证据冲突。它们被排除主比较不等于被判定为非原创。

## 日常更新

增量采集器已负责按新证据更新资格与元数据。批次完成后，在仓库根目录统一刷新当前清单、覆盖率、缺口账本、逐刊准入表与完整性报告：

```bash
python3 -m experiments.fig8_extension.derived_refresh --force
```

该入口取得 `supplement/checkpoint.lock` 后串行执行复算，网络采集可继续。不要在后台写入期间直接运行不持锁的全库重建命令。仅新增摘要、机构或日期记录不需要重新抓取或重建全部来源。

新增论文、身份更正和资格证据应通过对应增量写入器进入主表、资格表及变更账本；完成后使用上述入口复算。类型规则改变时，还需定向重算受影响资格，并保持 `fields/qualification/decide.py` 与当前规则同步。完整重建是维护操作，不是每批补齐的默认步骤。

离线检查入口（在写入结束或取得同一检查点锁后运行）：

```bash
python3 outputs/fig8_data/fields/qualification/recompute.py outputs/fig8_data/fields/qualification
```

报告更新时间见 `supplement/derived_refresh_status.json`。存在 `derived_refresh_pending.json` 表示主表可能已比派生报告更新；不能把旧报告覆盖率描述为实时结果。

## 当前日期批次的续跑入口

| 固定任务范围 | 获取与资格更新 | 元数据应用状态 |
|---|---|---|
| 启动时所有仅缺在线日期的候选 | `supplement/current_online_date_queue/status.json` | 同目录 `metadata_status.json` |
| 现有 MDPI 类型采集名录中仍缺日期的候选 | `supplement/mdpi_oai/remaining_date_deposits/status.json` | 同目录 `metadata_status.json` |

各目录的 `scope.json` 只保存任务身份及线索，数据仍更新到同一主表，不是数据版本。采集回执统一复用 `fields/crossref_raw/`；查询成功但未返回字段的 DOI 不重复请求。日期不能替代原创研究类型证据。

对应入口为 `experiments.fig8_extension.current_online_date_queue` 和 `experiments.fig8_extension.mdpi_remaining_date_deposits`。恢复前确认进程已结束；锁冲突或观察超时不代表任务失败。后一个入口等待前一个固定批次完成，遇到来源权限或限流终止则保留未请求项。`current_online_date_follow.run(folder, expected_status)` 在对应获取任务完成后只应用已合格论文的元数据；默认参数对应第一批。

第一批的 `current_online_date_refresh` 已排队等待元数据应用，再执行一次统一复算。批次终态只表示该来源路径已处理，不代表全部 Fig8 数据收尾或正式比较就绪。

更新按 `work_id` 定位、按 DOI 检查跨 Work 重复；有来源的新值可替换旧值，空值不覆盖已确认值。
保留来源、修复原因与未解决状态。若是新证据与现值冲突，进入待核，而非按文件时间盲目覆盖。
覆盖率、清单、缺口队列为可重建视图。证据文件、查询回执和审核记录不是独立数据版本。

`metadata/` 与各 `nature_*_review/` 是修复链条仍使用的输入及证据，不作为最终统计入口。
原路径映射见 `path_aliases.json`；相同来源文件通过相对链接共享一份内容，见 `source_aliases.json`。
这些链接都在本数据目录内解析；搬移时应保留相对目录结构。

清理记录见 `cleanup_log.json`；迁移后的复算结果见 `validation.json`。
统一存储不意味着数据缺口已解决，正式绘图资格仍以完整性报告为准。
