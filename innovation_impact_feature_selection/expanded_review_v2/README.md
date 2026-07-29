# 扩展版文献检索与指标普查（v2）

本目录是一个独立流程，不修改上一级已有的 1000 条检索快照、64 个候选指标家族或既有筛选结果。

## 核心原则

1. **不再截取“每库前 25 条”**：OpenAlex 对冻结的精确检索式使用 cursor 逐页取完。
2. **不把宽泛检索的全部噪声当作系统检索**：检索式必须有明确概念块和研究语境块；命中过大时改写检索式，而不是静默截断。
3. **Crossref 与 OpenAlex 分工**：
   - OpenAlex 用于题名、摘要和全文索引的系统发现；
   - Crossref REST 用于 DOI 元数据核验；
   - 若要求“Crossref 全库穷尽”，使用 Crossref 年度公共快照在本地逐条扫描，而不是穷举 REST 的模糊相关性结果。
4. **候选数量不预设**：先建立候选指标概念普查，再核验原始论文和公式，最后按固定硬门槛筛选。
5. **实时数据库结果不等于可重复数据**：每次运行记录检索式、截止日期、游标、页数、总数、哈希和完成状态；正式分析冻结 SQLite 文件及其 SHA-256。
6. **自动化不能替代双人判断**：题录纳排和指标归并需要两名研究者独立编码、分歧仲裁并报告一致性。程序负责保存过程，但不会伪造人工一致性。

## 文件

- `protocol_v2.json`：预注册式方法协议和停止规则。
- `search_queries_v2.json`：12 个概念块、72 条冻结检索式的生成矩阵。
- `expanded_search.py`：OpenAlex 全分页、库存统计、断点续跑和 Crossref 快照扫描。
- `audit_retrieval_v2.py`：完整性、去重、元数据覆盖和已知种子召回审计。
- `additional_literature_evidence.json`：扩展检索直接依据的综述和代表性原始研究。
- `build_indicator_registry_v2.py`：把原 64 个已核验家族与扩展候选概念合并。
- `screen_registry_v2.py`：沿用上一级固定硬门槛，生成候选、排除原因和当前可用指标。
- `tests_v2.py`：离线可重复性测试。

## 推荐运行顺序

```bash
python3 expanded_review_v2/build_indicator_registry_v2.py
python3 expanded_review_v2/screen_registry_v2.py
python3 expanded_review_v2/tests_v2.py
python3 expanded_review_v2/expanded_search.py compile

# 先查看每条检索式的命中量，不下载记录：
OPENALEX_API_KEY=... python3 expanded_review_v2/expanded_search.py openalex \
  --inventory-only

# 对全部 72 条检索式逐页取完；可中断、可继续：
OPENALEX_API_KEY=... python3 expanded_review_v2/expanded_search.py openalex \
  --workers 3
python3 expanded_review_v2/audit_retrieval_v2.py

# 若已下载 Crossref 年度快照，则本地全库扫描：
python3 expanded_review_v2/expanded_search.py crossref-snapshot \
  --snapshot /absolute/path/to/crossref-snapshot
```

`--max-records-per-query` 只用于试跑。只要设置上限，清单就会把该检索式标记为 `complete=false`，不能作为正式系统检索结果。

## “全部结果”的操作性定义

“全部”是指：在指定数据库版本/访问日期、截止日期、文献类型、字段范围和冻结布尔检索式下，数据库返回的每一条记录均被读取并登记。它不意味着把数据库中所有与主题无关的记录下载下来，也不意味着不同日期再次调用实时 API 会得到完全相同的集合。
