# Fig8：单一持续更新的数据集

唯一数据入口是 `outputs/fig8_data/`。不再创建 v1/v2、父子版本或冻结发布目录。
可追加论文、覆盖已确认字段、按身份键合并修复。原始来源和每条未解决状态继续保留；
更新后重算派生表，不能用旧覆盖率描述新数据。

## 当前数据口径

- `nature/`：Nature 主刊 2023–2025 年 3,683 篇论文的署名、机构、国家和高校映射。
  `papers.jsonl` 是国家/高校分配的标准化表；`research_roster_release.parquet`
  是其带完整采集字段的工作表，不是另一个版本。二者通过 DOI 关联。
- `fields/papers/`：三个领域唯一的论文元数据主表，包含摘要和观测作者。
- `fields/qualification/decisions/`：同一批论文的类型、在线年份、去重和资格判定表。
  按分片名及 `work_id` 关联主表；`eligibility_status` 是当前纳入口径。
- `fields/qualification/research_roster.csv`：从两表生成的研究论文清单。
  不再另存一套完整的 `research_papers/`。
- `fields/qualification/完整性报告.md`：当前 Fig8b 分母、覆盖率及未解决项。
- `metadata/` 和 `nature_*_review/` 等目录保留采集、修复所需输入及证据。
  它们不是可替代上述当前表的实验数据版本。

Nature 主刊的精细署名槽位与三个领域的 OpenAlex 观测作者粒度不同；
一份数据集不意味着可以把两种作者分母直接相加。
Fig8b 复用 Nature 信息时须保持 DOI、作者身份与证据的精确匹配。

## 更新与复算

仅摘要、机构或国家字段更新后：

```bash
python3 -m experiments.fig8_extension.fig8b_qualification_recompute outputs/fig8_data/fields/qualification --refresh
python3 -m experiments.fig8_extension.fig8b_qualification_report
```

新增/删除候选、修改身份字段或取得新类型/日期证据后：

```bash
python3 -m experiments.fig8_extension.fig8b_qualification_evidence index
python3 -m experiments.fig8_extension.fig8b_qualification_build
python3 -m experiments.fig8_extension.fig8b_qualification_recompute outputs/fig8_data/fields/qualification
python3 -m experiments.fig8_extension.fig8b_qualification_report
```

检查现有数据而不接受覆盖率改变时，省略 `--refresh`。复算直接读当前主表，
不会依赖旧版本、冻结哈希或重复的研究子集。新论文没有资格表时会明确报错，
须重建判定，不能静默漏计。

Nature 机构映射更新后，由 `nature_release_export` 刷新标准化表，
再用 `nature_release_report` 刷新其比较结果和报告；模块名中的 release
仅为现有代码名称，不再创建版本目录或 ZIP 备份。独立核对：

```bash
python3 outputs/fig8_data/nature/recompute.py outputs/fig8_data/nature
python3 -m pytest -q tests/fig8_extension
```

## 获取追加证据

```bash
python3 -m experiments.fig8_extension.fig8b_qualification_catalogue
python3 -m experiments.fig8_extension.fig8b_qualification_aps_all
python3 -m experiments.fig8_extension.fig8b_qualification_aps_missing
python3 -m experiments.fig8_extension.fig8b_qualification_evidence fetch --workers 8
python3 -m experiments.fig8_extension.fig8b_qualification_secondary
```

`fig8b_collect`、`fig8b_build`、`fig8b_apply_full_authors` 是从采集证据重建或补充
元数据的工具；不要在只刷新资格/覆盖率时重跑元数据重建，以免覆盖后续人工修复。
修改主表时按 `work_id` 更新，按规范化 DOI 检查跨 Work 重复，空值不覆盖已确认值，
冲突保留来源与状态。来源记录和修复日志用于追踪，不用于要求数据永久冻结。

## 科学边界

当前主源覆盖偏向 Nature Portfolio、APS 和 PMC 可开放读取内容，仍有大量资格待核。
统一存储不改变纳入数，不补出缺失证据，也不使当前子集成为无偏的全期刊样本。
只有明确研究类型和 2023–2025 在线年份的匹配记录才能纳入；新闻、评论等排除。
Crossref/OpenAlex 的宽泛 Article 标签不能单独确认原创研究。
年度 Nature Portfolio imprint 与 Springer Nature 集团标签分开。
机构有值不等于全部署名分支核实，也不直接确认高校归属。

## 统一补充任务与来源能力

`outputs/fig8_data/supplement/` 保存当前任务账本、分层试测、逐字段修改日志、预算及 panel 就绪评估。

```bash
python3 -m experiments.fig8_extension.supplement_run pilot
python3 -m experiments.fig8_extension.supplement_run continue-pmc --workers 16
```

`pilot` 会发现当前任务，按 field×year 轮转、再按期刊轮转选择最多4,000个唯一DOI（seed=20261021），
获取公开来源，重建资格，补充摘要及唯一匹配作者的署名信息，并离线复算。
`continue-pmc` 发现尚未读取的当前PMC线索，执行后完成同一套更新；该全量步骤可能需要数十小时。
`all` 顺序执行两者。抓取遵循每主机节奏；HTTP限制、浏览器验证和重试耗尽都显式保留。

仅将已经取得的证据写回当前表并生成报告：

```bash
python3 -m experiments.fig8_extension.supplement_run checkpoint
```

`execution_status.json` 与 `logs/`给出真实阶段进度；`tasks.sqlite` 的任务、尝试和字段变更可查询。
字段有值与署名结构已验证分开。机构注册类型只供候选分类，不直接推断高校所有权。
`panel_readiness.json` 使用Nature兼容的8种分配口径，并报告未知份额的保守稳定性界限。
`boundary_candidates.csv` 是缓存官方目录中不在当前名录的候选线索，未证明primary field之前不追加。

新工具不购买数据、不调用付费模型，也不将访问失败解释为非研究或缺摘要。全量PMC执行完毕
仍可能有其他出版社资格与官方机构归属未解决；任务已入账、来源路径已尝试和正式比较就绪分别报告。


宿主机发生中断后，先运行 `python3 -m experiments.fig8_extension.supplement_recover`，
核对恢复回执，再运行 `supplement_queue discover` 和 `supplement_run continue-pmc`。
检查实际进程命令、启动时间与 `/proc/sys/kernel/random/boot_id`；旧 PID 和旧进度文件不能证明任务仍在执行。
每 10,000 个 PMC 处理项执行当前证据检查点；最终使用 PMC 官方 OAI front-matter 回退，
遵循单并发、每秒不超过3次及官方批量获取时段，随后再次独立复算。


## 缓存优先采集

`continue-pmc` 先执行证据检查点：解析已存主源、批量查询尚未处理的 Crossref
类型/日期和合格论文摘要/作者/机构/国家缺口、从缓存索引读取摘要并更新资格。
随后只给仍缺字段或存在冲突的论文安排逐篇请求。已成功或已终止的来源记录直接复用；
失败仍是未解决，不能改成纳入或排除。每个 DOI 每轮最多安排一个新 PMC 标识，
其他标识留待下一轮重建队列；不会丢弃。每次检查点后重新判断缺口。

`bulk_cache.sqlite` 只保存既有批量响应的位置索引和查询状态，不复制另一套论文数据。
索引按文件修改状态增量维护；查询成功但没有匹配记录与查询失败分开留痕。
`batch_benchmark.json` 保存同一120 DOI的批量试测，原逐篇请求使用已有耗时记录，
不会为了对照再下载一遍。批量返回的泛化文章类型不构成原创研究确认。

```bash
python3 -m experiments.fig8_extension.supplement_acquisition index
python3 -m experiments.fig8_extension.supplement_acquisition plan
```

采集器最多保持 `--workers` 个在途请求。需要安全停止新版逐篇采集器时，创建
`outputs/fig8_data/supplement/stop_acquisition`，它会保存已在途结果后停止派发；
明确恢复时删除该标记再启动。进度以 `execution_status_full_pmc.json` 的实际阶段为准，
缓存/批量阶段不能用上一轮的 `pmc_retrieval_status.json` 推算当前速度。

已审查机构规则的增量复用（离线，无网络请求）：

```bash
python3 -m experiments.fig8_extension.official_rule_reuse
```

此入口读取 `supplement/official_entities/current_rule_reuse_scope.json` 中的
`next_rule_paths`，只检查上次成功检查点后作者署名发生变化的合格论文。
它直接按论文 ID 定位，不依赖缺口索引已刷新；映射仍要求完整署名精确匹配，
保留冲突、国家字段和 Nature 已核查记录。新规则首次覆盖已有论文仍须执行其
完整适用范围应用，不能以增量检查替代。更新后仅标记派生结果待合并刷新。

ACS 短篇研究类型的缓存处理：Nano Letters 与 The Journal of Physical Chemistry
Letters 仅在归档 JATS 同时给出 `rapid-communication`、明确 `Letter` 分类和
匹配期刊 ISSN 时使用已核查的期刊类型规则；日期、身份和冲突检查继续适用。
`journal_format_apply` 应用资格规则，`journal_format_authors.run(selected_work_ids=...)`
按新增纳入清单复用作者署名。随后可运行：

```bash
python3 -m experiments.fig8_extension.acs_cached_imprint
```

该命令只处理 `supplement/journal_type_policies/acs_letter_application.json` 的固定范围，
从原始权限声明中的明确“Published [year] by American Chemical Society”提取出版商。
单独的版权所有者不能证明出版商；此步骤不把逐篇证据扩大为全年期刊归属规则。

共享作者组署名的离线补齐：

```bash
python3 -m experiments.fig8_extension.group_affiliation_cache_scope
python3 -m experiments.fig8_extension.group_affiliation_apply
python3 -m experiments.fig8_extension.official_rule_reuse
```

首个入口固定已有合格论文的缓存范围，逐条留痕；原生 APS JSON 不使用 JATS 规则。
第二个入口核对原始缓存哈希、DOI、题名和期刊，仅在作者组只有一个直属单位、
且组内不存在作者级单位链接时使用共享地址。仅补唯一精确作者匹配的缺失连接，
保留已有署名、Nature 核查和作者列表完整性状态。结果位于
`supplement/group_affiliation_cache/`，未匹配和身份冲突继续保留。

`springer_author_cache.run()` 默认使用 4 路只读缓存准备、单路锁内写入，
最多预取 8 条记录。可传 `workers=1` 使用串行准备；不会改变来源限速或增加下载。
实际缓存并行试验记录为 `supplement/concurrency_benchmark.json`。

Cell Press 已缓存年度名录的跨刊复用：

```bash
python3 -m experiments.fig8_extension.cell_press_annual_cache
```

该入口仅使用已有的 2024 年合同附件及 2025 年期刊名录，要求 ISSN 与刊名同时匹配，
按对应年份补年度规则、候选年注释与已核定在线年份标签。已有确认值不覆盖，
年度冲突与刊名不一致单列；不把 2024/2025 的证据外推至 2023 年。
证据、年度前后值和处理统计位于 `supplement/cell_press_annual_cache/`。

APS 原生批量摘要缓存补齐：

```bash
python3 -m experiments.fig8_extension.aps_cached_abstracts
```

首轮固定当前合格 APS 论文的摘要缺口；后续可重复执行相同范围，不重复下载。
使用原生摘要字段，要求 DOI、题名和期刊身份匹配，复用已验证的 APS 期刊 ID—ISSN
规则；多个缓存摘要不一致时保留冲突。只填空摘要，逐篇保存来源定位及修改日志，
已有摘要保持不变。检查范围与统计保存在 `supplement/aps_cached_abstracts/`。

EDP Sciences 官方 OAI 批量缓存：

```bash
python3 -m experiments.fig8_extension.edp_oai_harvest
```

入口使用 `supplement/edp_oai/scope.json` 的既有候选范围与官方 ListSets 的期刊名称，
复用已保存的首批响应并按 resumptionToken 续取；重复 token、来源失败或集合耗尽时
保存明确状态。仅保留范围内 DOI 的匹配记录，原始批次作为证据保留；不会修改资格表。
Dublin Core `Text` 不构成原创研究类型证据，OAI 修改时间不当作在线发表时间；
未匹配集合、无法提供的丰富格式和受限文章页在能力表中保留。

EDP lossless cache review: `python3 -m experiments.fig8_extension.edp_oai_review` reuses raw batches, preserves inline XML text, and checks exact DOI/title/journal-set identity. Results remain in `supplement/edp_oai/cache_review.json`; generic DC Text and untyped DC dates do not establish research eligibility or online year. The active collector may retain the older convenience parser until restart; the review always reparses raw XML.

EDP explicit errata: `python3 -m experiments.fig8_extension.edp_errata_apply` consumes only exact-identity records with native errata subjects, adds source-bound nonresearch evidence under the shared writer lock, and preserves conflicting primary evidence. It does not infer online dates from DC dates. Evidence and replay validation are in `supplement/edp_oai/errata_application/`.

EDP automatic cache consumption: `python3 -m experiments.fig8_extension.edp_cache_follow` processes the initial cache, then each 1,000 new matched records and the terminal tail. It uses a single-consumer lock, never issues HTTP requests, records applied evidence fingerprints, and exits when the collector stops. This does not close unresolved qualification tasks.

eLife official list batches: `python3 -m experiments.fig8_extension.elife_batch` uses the fixed DOI scope in `supplement/elife_official/scope.json`, four in-flight requests with host pacing, cached receipts, and a fixed initial API page bound. It caches article types/publication status/dates for later identity and qualification review; it does not include Reviewed Preprints merely because a DOI exists. Live pagination duplicates/missing targets remain explicit.

eLife qualification: `python3 -m experiments.fig8_extension.elife_evidence` applies source-linked native article type and first-publication date through the shared matcher and lock. Publisher schema copies and commit provenance are in `supplement/elife_official/schema/`. Reviewed-preprint RP paths hold dates for review; revision dates never substitute for first publication. This step creates metadata follow-up tasks and requests coalesced coverage refresh.

eLife version identities: `python3 -m experiments.fig8_extension.elife_version_details` requests native v8 only for existing suffix-DOI/RP cases whose article IDs occur in cached official journal batches. Default API responses omit doiVersion. Distinct IDs are fetched once, with explicit versioned Accept provenance, four paced workers, cached receipts, and source-restriction stops. Missing journal-list IDs remain unresolved.

eLife v8 application: `python3 -m experiments.fig8_extension.elife_version_apply` verifies native DOI/doiVersion, title, journal, published status, representation and source hash. It retains an existing canonical candidate when available, applies evidence only to that candidate, and records version-alias proposals for explicit duplicate adjudication. Only the provisional RP-path hold is released by matching v8 publication evidence; unrelated type/date/content conflicts remain.

eLife v8 continuous processing: `python3 -m experiments.fig8_extension.elife_version_follow` consumes 250-record increments plus the terminal tail, applying qualification, explicit canonical/version duplicate rules, and cached metadata. All writers use the shared checkpoint lock; no HTTP requests occur in this consumer. Source conflicts and unresolved entities remain explicit. `elife_duplicate_apply` writes reusable native alias rules without changing candidate DOI values; `elife_version_metadata` reuses native abstracts/bylines and preserves existing values.

Pleiades distribution routing: existing pending candidates without PMC identifiers are queued in the shared Springer adapter after an official distributor statement and exact-identity parser check. Publisher and Nature Portfolio fields are unchanged. `python3 -m experiments.fig8_extension.springer_queue_extension` waits for the current collector to finish successfully before dispatching the fixed extension, and exits without requests if that source stopped or failed. Scope and evidence: `supplement/pleiades_route/`.

Elsevier locator cache: `python3 -m experiments.fig8_extension.elsevier_redirect_cache` extracts only explicit DOI-bound HTTPS ScienceDirect targets from existing Linking Hub pages. The initial direct target check returned 403 and was not retried or expanded. `supplement/elsevier_redirects/capability_assessment.json` records this distinction; unvisited targets and qualifications are not marked resolved.

eLife retained-source comparison: `python3 -m experiments.fig8_extension.elife_byline_comparison` compares cached native v8 bylines with retained primary signatures, fills only missing unique exact author links, and preserves existing primary/Nature values. Each processed source pair has an append-only review event; unresolved entity/ownership assignments remain open. Identical processed source pairs are not compared again without new evidence.

Additional NLM annual imprint reuse: `python3 -m experiments.fig8_extension.annual_nlm_additional` reads existing hash-checked official NLM responses for exact Lippincott Williams & Wilkins, Thieme, De Gruyter and Cell Press imprints. It requires a valid matching ISSN and an explicit date interval, excludes interval boundary years, preserves established labels, and retains previous rules on replay. It uses the shared annual applier and checkpoint lock; no network requests occur. Evidence, before/after updates and scoped label validation are in `supplement/annual_nlm/additional_imprints/`. Open-ended catalog intervals remain working annual evidence and must reopen on contradictory transfer evidence.

MDPI official batch metadata: `python3 -m experiments.fig8_extension.mdpi_oai_harvest` resumes a finite OAI sequence and matches only the existing MDPI DOI roster saved in `supplement/mdpi_oai/scope.json`. It reuses immutable request receipts, keeps full raw XML, and writes matched-page evidence without updating qualification or the paper master. Native OAI type sets are kept separately from generic DC `Text`; OAI datestamps and DC dates remain distinct. The observed response contains publication years outside the requested window, so downstream qualification must independently verify identity and date semantics. Author names alone do not establish affiliation completeness. Restriction/failure, repeated/exhausted token, initial provider-count boundary, and zero target matches after 1,000 pages terminate this source with unresolved work retained.

MDPI cached type processing: `python3 -m experiments.fig8_extension.mdpi_cache_follow` processes 500 matched-record increments and the terminal tail. It independently reparses the raw OAI XML and requires DOI, exact normalized title, journal and identifier ISSN agreement. Official header article-type sets are evidence; generic DC Text and untyped DC dates are not promoted into research or online-date assertions. Existing evidence conflicts remain active through the shared qualification engine. The consumer issues no network requests and does not claim author-affiliation completion.

`derived_refresh --force` synchronizes the offline `fields/qualification/decide.py` from the current incremental rule implementation under the shared checkpoint lock before replay. Changed rule-code hashes and reasons are appended to `supplement/offline_rule_sync_events.jsonl`; replay must still pass before statistics are considered current.

Native subtype conflicts: `python3 -m experiments.fig8_extension.cached_native_subtype_holds` searches existing qualified-paper JATS metadata for explicit image/commentary formats or anchored editorial-letter/comment titles, then rereads the raw source and verifies DOI/title/journal identity. It appends a subtype-review hold rather than declaring such articles nonresearch; research-letter types and generic scientific uses of “image” or “perspective” are not blanket rejected. Existing study-design adjudications and Nature main-journal records are preserved. No HTTP requests occur. Remaining unmatched identities retain an explicit unresolved record.

APS annual publisher rules: `python3 -m experiments.fig8_extension.aps_annual_cache` reuses native APS batch caches with publication year equal to an explicit APS copyright year, named APS publisher and American Physical Society rights holder. Candidate DOI/title and native journal name (or the already-validated APS journal-ID/ISSN mapping) must agree. It retains earliest/latest dated article proofs per journal/year, preserves prior rules on replay, and uses the shared annual applier. No labels are extrapolated to other years; contradictory transfer evidence must reopen the rule. Raw receipts, hashes, rights statements and before/after changes are retained in `supplement/aps_annual_cache/`.

Dated JATS publication statements: `python3 -m experiments.fig8_extension.jats_annual_copyright` reuses archived front matter only when an explicit electronic publication year agrees with a copyright year and a named MDPI licensee or Wiley published-by statement. Generic author copyright alone does not establish the publisher. DOI/title/journal matching is checked again against raw XML. Earliest/latest matched proofs and rule provenance are retained; existing known labels and transfer conflicts use the shared annual updater.

Validate these annual cache updates with `python3 -m experiments.fig8_extension.validate_annual_cache aps_annual_cache` and `python3 -m experiments.fig8_extension.validate_annual_cache jats_annual_copyright`. The validator checks raw proof hashes and statements, annual labels, and all affected verified-year decision labels. It does not substitute for the global qualification, coverage or sensitivity recomputation.

PLOS annual publisher reuse: `python3 -m experiments.fig8_extension.plos_annual_cache` matches cached official PLOS DOI/title/eISSN/publisher records to the current qualified electronic publication day. It never turns the API's undifferentiated publication date into an online date by itself. The resulting dated annual rules retain first/latest matching article proofs and the corroborating qualification shard. Run `python3 -m experiments.fig8_extension.validate_annual_cache plos_annual_cache` to verify raw hashes, native fields, an additional matched non-PLOS-API electronic-date evidence record, and every affected decision label.

Reuse accepted dated imprints across a journal/year with `python3 -m experiments.fig8_extension.accepted_imprint_annual`. This applies the existing per-paper `portfolio_label` policy with no inherited annual label, retains matched native JATS publisher evidence and any independent date corroboration, and excludes journal/year pairs containing an existing positive Nature Portfolio label. Earliest/latest usable article proofs and their raw hashes remain in the rule. Unknowns are not inferred from publisher aliases or OpenAlex candidate years. Validate with `python3 -m experiments.fig8_extension.validate_annual_cache accepted_imprint_annual`.

Frontiers native annual reuse: `python3 -m experiments.fig8_extension.frontiers_annual_cache` reparses cached publisher HTML with exact DOI/title/journal/ISSN matching, named Frontiers imprint and explicit Published history. Accepted dates and current publisher profiles do not establish the year. Same-journal/year rules preserve raw proof hashes and do not extrapolate across years. Validate with `python3 -m experiments.fig8_extension.validate_annual_cache frontiers_annual_cache`.

Annual JATS imprint reuse also accepts an explicit ACS published-by statement with matching electronic/copyright year. Generic ACS copyright alone is insufficient. Annual candidate annotations read only seven columns first; complete Arrow tables are loaded only for changed shards and unrelated columns remain intact. The local benchmark is in `supplement/annual_column_projection_benchmark.json`; it measures reading/materialization, not total pipeline speed.

Additional dated ASM/PeerJ imprints: `python3 -m experiments.fig8_extension.additional_dated_imprints` reuses matched native publisher names and verified electronic dates. Undifferentiated full publication dates require exact matched electronic-date corroboration; the dated official publisher context does not itself establish every journal/year. PeerJ imprint labels do not backdate Taylor & Francis group ownership. Run `python3 -m experiments.fig8_extension.validate_annual_cache additional_dated_imprints` to check raw sources, date corroboration, identities and affected annual/decision labels.

EDP fixed remainder: `python3 -m experiments.fig8_extension.edp_unmapped_records` fetches exact native identifiers only for the existing plan's unmatched-journal candidates, reuses receipts, and retains unreturned identifiers without inventing journal aliases. `python3 -m experiments.fig8_extension.edp_unmapped_review` waits for the collector and checks returned native DOI/title against the current candidate; generic DC types/dates and unconfirmed journal aliases remain unresolved. `python3 -m experiments.fig8_extension.edp_source_ledger` records each source outcome without closing qualification tasks or claiming alternative sources are exhausted. Mapped-set completion and unmatched-candidate completion are reported separately in `supplement/edp_oai/completeness.json`.

Wiley dated statements can establish the publisher when the native publisher field is empty, provided the explicit electronic year matches the copyright year and the statement explicitly says published by Wiley. A different nonempty native publisher remains a conflict. The existing JATS annual validator covers these rules.

`official_entities/kit_imperial_self_rules.json` adds exact university-self signatures to the shared rules. KIT uses a cached official PDF; Imperial's evidence explicitly identifies official search-index excerpts and retained direct-access 403 responses. These rules do not map clinical subsidiaries or joint centres, and preserve signature countries and previously reviewed Nature records.

Springer directory capability review is retained in `supplement/springer_directory_review/assessment.json`. Its control contains explicit article types but unlabelled article dates. It does not establish electronic-date semantics or justify replacing already cached primary article metadata.

The dated native-imprint adapter also covers exact `Impact Journals, LLC`, `Dove Press`, and `Life Science Alliance LLC` names, with separate context in `supplement/independent_imprint_context.json`. Earlier ASM/PeerJ context receipts remain unchanged. Every new rule still requires DOI/title/journal agreement and explicit electronic date or an exact full publication date corroborated by matched primary evidence. Validate the accumulated rules with `python3 -m experiments.fig8_extension.validate_annual_cache additional_dated_imprints`.

Country alias repair: `Turkey` and `Türkiye` both resolve to `TR` only as explicit terminal country tokens (or explicit country fields), following the UN name-change record retained in `supplement/turkey_country_alias/scope.json`. No location is inferred from an institution name. `springer_author_cache.run(selected_jobs=..., reprocess_selected=True)` permits scoped same-source byline re-enrichment after a parser correction, including `aps_only=True` with explicitly selected APS receipts; a global forced replay is rejected. Previously reviewed institution fields and Nature records remain protected. The selected publisher jobs, APS jobs, unavailable adapters, initial no-change shortcut result and corrected results are retained under `supplement/turkey_country_alias/`.

`python3 -m experiments.fig8_extension.turkey_cached_jats` handles the same alias repair for original-scope records outside the publisher-job index. It only replays the exact existing primary JATS receipt behind the unresolved country slot, independently checks DOI/title/journal, and retains unsupported source types or identity conflicts. It performs no HTTP requests. These scoped replays write source-specific results; do not treat their processed count as papers completely repaired.


Incremental consumers now finish the metadata chain without manual per-batch dispatch. `python3 -m experiments.fig8_extension.mdpi_cache_follow` applies each native-type increment, extends the fixed candidate archive follow-up only for unique unreviewed PMC leads, then uses `mdpi_pubmed_increment` for newly identified PMID date gaps. Its immutable request-batch scopes are acquisition receipts, not separate dataset releases. Existing ready PMID/DOI scopes are excluded; incomplete batches resume from their cached responses. Exact PubMed DOI/title/journal and explicit electronic-date checks precede qualification updates. Generic types, ambiguous identifiers and primary-source conflicts remain unresolved. Author metadata follows qualification, and reviewed entity rules are reused on changed author records.

`publisher_delta_apply --follow` uses eight cache workers after each qualification increment and likewise reuses reviewed entity rules. `official_rule_reuse` serializes its incremental change cursor with `official_entities/rule_reuse.lock`; main-table mutations retain `checkpoint.lock`. Do not start duplicate consumers. Existing downloaded receipts and completed source outcomes are reused on resume. Full reports are coalesced via `derived_refresh_pending.json`; report timestamps must not be represented as covering later metadata changes.

Dated publisher-imprint reuse additionally uses the separate `specialist_imprint_context.json`, `society_imprint_context.json`, and `medical_imprint_context.json` evidence files. Current publisher identity is only context; native journal/year proofs remain mandatory. The accumulated raw-source and decision-label validator is unchanged. Exact physics-department and independent CIEMAT/MPP institution rules have their own source, application and validation receipts under `supplement/official_entities/`; campus ambiguity, joint signatures and countries are not inferred from those rules.

Exact signed-location country repair: `supplement/korea_exact_locations/rules.json` documents only the complete Kyungpook National University / Daegu / Korea and Seoul National University / Seoul / Korea signatures. Official university addresses disambiguate these as KR; a bare Korea token, a university name alone, branches and other signatures do not use this exception. `nature_backfill_resolve.address_country` implements exact normalized matching. Scoped same-source publisher, JATS and APS cache replays and before/after validation are retained in that directory; one Crossref linkage adapter remains separately pending. Collectors were not restarted; idle metadata consumers were reloaded to apply the rule to future increments.

Cached PMC primary-byline comparison: `pensoft_byline_comparison.run(ROOT / "supplement/pmc_journal_batches", "supplement/pmc_journal_batches/%")` reuses the common exact-author missing-link comparator on held, qualified records. The PMC journal adapter unwraps the unique OAI JATS article via `supplement_oai.parse_oai`; raw response hashes remain checked before application. Duplicate incoming receipts per work are rejected. Existing primary slots and Nature reviews are preserved; sequence/address disagreements remain explicit and do not imply truncation. `python3 -m experiments.fig8_extension.primary_comparison_ledger outputs/fig8_data/supplement/pmc_journal_batches/primary_comparison` appends idempotent per-task outcomes without closing entity or country gaps.

APS held-byline comparisons use `pensoft_byline_comparison.run(ROOT / "supplement/aps_cached_byline", "fields/qualification/aps_all_raw/%")`. The shared comparator loads one exact DOI from an HTTP-200 cached APS batch through the existing native adapter; missing/duplicate DOI records fail closed, the complete raw batch hash is checked again before updates, and DOI/title/journal matching remains mandatory. Original primary slots are retained. Prior journal-policy comparisons are reconciled from existing audit/application receipts rather than rerun; see `supplement/primary_byline_comparison/status_reconciliation.json`.

Annual publisher reuse can now target unresolved journal/year pairs with `apply_proposals(..., apply_pairs=...)`; the complete rules file is retained while table updates are scoped. `validate_annual_cache.run(name, selected_pairs)` validates only that increment and writes `incremental_validation.json`. APS Creative Commons records qualify as explicit publisher evidence when the native publisher is APS, the rights statement is exactly “Published by the American Physical Society”, and publication/rights years agree; copyright ownership is not inferred. Scope and source receipts are under `supplement/aps_annual_cache/open_access_scope.json`.

Copernicus annual imprint reuse accepts its existing publisher-native family in addition to archived JATS. The adapter selects a unique DOI from retained OAI or standalone front matter, normalizes only English calendar month notation, and still requires explicit online-date evidence or an exact independent matched full-date corroboration. It does not change research eligibility or infer online dates from untyped dates alone. See `supplement/additional_dated_imprints/copernicus_applied_scope.json`, `copernicus_parser_review.json`, and `incremental_validation.json`.

### EDP 富文本元数据补查

`edp_rich_set_review` 在已固定的 32 个 EDP 期刊集合内检查官方 OAI `pmc`
格式，复用已有 Astronomy & Astrophysics 失败字段回执，不把一个集合的无结果
推广到其他期刊。`edp_rich_candidates` 等待检查结束，复用批量页，再对有富文本
结果的集合中的既定 DOI 使用精确 GetRecord；不追加期刊或年份。
`edp_rich_apply` 等待采集结束，使用统一资格、摘要和作者映射入口写回。

```bash
python3 -m experiments.fig8_extension.edp_rich_set_review
python3 -m experiments.fig8_extension.edp_rich_candidates
python3 -m experiments.fig8_extension.edp_rich_apply
```

来源和状态保存在 `supplement/edp_oai/rich_set_review/`。这里的 `pmc` 是 EDP
声明的 XML 格式名称，不表示来自 PMC 数据库。原始 OAI 修改时间仅限定获取范围；
资格年份必须来自文章内明确的电子发表日期。解析成功不等于资格通过。

PMC cloud acquisition now uses conditional 64 KiB S3 ranges (`pmc_cloud_front`). Existing full XML caches take precedence. A complete captured front is retained byte-for-byte through `</front>` in `supplement/pmc_cloud/xml_front_raw/`; the parser adds the closing article element only in memory. Prefix integrity uses the conditional ETag and retained SHA256, and must not be reported as a computed full-object MD5. Valid ranges lacking complete front matter fall back to the ordinary full-object cache; access and identity failures do not trigger that fallback. Offline full/front equality checks and live transfer measurements are under `supplement/pmc_cloud/front_*`. The consumer defaults to 2,000 versions per write batch with 64 acquisition workers, preserving the consumed cursor and shared writer lock. Batch events record acquisition, qualification, and author/ledger elapsed times; reduced request bytes alone do not establish wall-clock speedup.

Dated JATS publisher rules also inspect qualified cloud captures from `supplied_publisher_processed`, because those direct applications are absent from the legacy `source_metadata` index. Identity, electronic/copyright year, and explicit named-publisher requirements remain unchanged. Validate a newly applied pair set with `validate_annual_cache.run('jats_annual_copyright', selected_pairs)`.

### Incremental change-ledger source check

Run `python3 -m experiments.fig8_extension.change_source_audit` to check the
current change-ledger prefix without network requests or paper-data changes.
The persistent cursor and source-path cache are under
`outputs/fig8_data/supplement/change_source_audit/`; `audit.sqlite` contains
per-change unresolved issues. Each invocation fixes its ending change ID.
Subsequent invocations inspect new rows and newly supported receipt schemas.
This checks explicit receipt paths, change reasons and before/after presence;
it does not certify source content, scientific applicability, or qualification
logs. The coalesced phase refresh runs it before the overall goal audit.

`python3 -m experiments.fig8_extension.qualification_source_audit` separately
checks source paths in a consistent read snapshot of the primary/secondary
qualification evidence index. It shares receipt-resolution rules and deduplicates
source checks. Results and any per-evidence unresolved paths are in
`supplement/qualification_source_audit/`. This is a snapshot audit, not a
per-batch refresh. The coalesced phase refresh runs it after the change-source
audit and before the overall goal report, once acquisition settles.
It does not replace evidence-content or decision-ID linkage validation.
