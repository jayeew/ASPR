# 论文创新性与潜在影响力特征选择（独立包）

本目录不修改、导入或调用 ASPR 现有实验代码。它只完成一件事：在未来结果不可见的条件下，以可追溯文献、发表时信息边界、本地数据可用性和固定规则，决定应保留哪些维度与特征。

## 核心边界

- `innovation_evidence`：论文在发表时体现出的知识重组新颖性证据。
- `substantive_potential`：可能影响后续使用的科学定位，包括组合结构、知识多样性、参考知识时间结构和主题动量。
- `opportunity_visibility`：团队、知识网络和发表载体带来的传播或注意机会。
- `context_control`：年份和学科背景。
- 未来引用、扩散和颠覆只能在特征冻结后作为训练标签，不能参与指标定义、入选或去重。
- 不把引用潜力解释为论文质量、正确性、社会价值或创新真值。

## 固定工作流

```bash
python3 innovation_impact_feature_selection/search_literature.py --limit 25
python3 innovation_impact_feature_selection/select_features.py
python3 innovation_impact_feature_selection/audit_local_data.py
python3 innovation_impact_feature_selection/tests.py
```

第一步冻结 Crossref 和 OpenAlex 的 20 条查询、两家数据库各前 25 条结果。第二步对每个候选特征执行 12 道硬门槛并在同族内确定性去重。第三步核验本地源文件哈希。第四步检查构念边界、关键排除、检索完整性和两次运行字节一致性。

## 输入

- `protocol.json`：构念、T0 信息、结果隔离、公平性与复现协议。
- `search_queries.json`：冻结的全维度检索式。
- `literature_evidence.json`：支持、混合、零结果和反证并存的来源级证据表。
- `dimensions.json`：候选维度及解释边界。
- `feature_registry.json`：完整候选特征家族库。
- `screening_rules.json`：机器可执行的硬门槛、同族去重和角色规则。
- `data_capabilities.json`：本地列覆盖、稳定性、可推导输入和文件哈希。

## 主要输出

- `outputs/literature_evidence_table.csv`：便于人工审阅的文献证据表。
- `outputs/evidence_discovery_crosswalk.csv`：每条证据在冻结数据库排序中的命中、查询式与引文追溯路径。
- `outputs/feature_decisions.csv`：每个候选特征的 12 道门槛、全部失败原因和同族胜者。
- `outputs/final_dimensions.json`：维度是否入选、独立证据组和所含特征。
- `outputs/final_features.json`：最终特征注册表，含公式、时间边界、缺失规则、来源和角色。
- `outputs/training_feature_sets.json`：创新核心、实质潜力、机会、默认、扩展和敏感性集合。
- `outputs/selection_report.md`：中文结果报告。
- `outputs/audit_manifest.json`：输入、脚本和输出哈希。
- `outputs/local_data_audit.json`：本地数据源哈希核验结果。

## 证据声明

这是“可复现的范围证据普查＋关键原始研究核验”，不是事后声称的 PRISMA 系统综述或元分析。公开数据库排序会变化，因此当前检索快照被保留；以后更新应创建新版本，不覆盖本次证据冻结。
