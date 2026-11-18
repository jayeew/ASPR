# 科学创新五维分析：复现、恢复与交付

新克隆可运行 `python3 -m figure_pipeline.innovation_five_dimensions --help` 查看入口。下面的研究材料、模型结果、统计、图件和稿件路径均需对应的本地文件。

## 当前 Full 覆盖重跑

本轮已完成：实际CLI调用200次，包括100次生成与100次核验调用；后者含用户明确授权的1次超时重试。获得99份新Full报告及99份有效核验记录。
`s41467-026-68493-2`追加核验成功；`s42003-026-09574-2`生成被服务端内容安全检查拒绝，其核验未启动，保留缺失。被拒绝论文的陈旧报告和评分已移除，原详细分析与原始材料保留。
统计与完整Fig3已按实际分母原位覆盖。99份报告的原详细分析附件逐字保持，181项代码测试通过；5,573个受保护文件未改变。没有新增结果版本或备份集。

本轮固定为主队列100篇Full生成＋100篇Full核验；已有五篇开发样本包含在这200项主体任务中。
直接覆盖本目录对应结果，不创建版本或备用数据集。其他五种方法的500份报告和评分、全部共用清单及DAMAGE保持原样。
Full使用完整稿件、完整原详细分析，以及已有GEAR/Graph/Full原文选段去重并集；正文不设目标字数或硬性上限。
核验指令、0–3分规则及清单适用性保持不变。模型及推理强度沿用原配置。程序不自动增加模型重试或修复调用，失败先商议。

限定入口依次为 `python3 -m figure_pipeline.innovation_five_dimensions rerun-full --phase prepare`、
`--phase pilot`、`--phase run`；`--phase check`检查受保护文件。先检查五篇材料及输出完整性，再进入全量。
当前执行情况见 `status/full_rerun.json` 和 `status/full_rerun_calls.json`。成功任务可恢复复用；发生过失败调用的任务不能擅自再次调用。
汇总与Fig3现已对应本轮Full结果及复用的其他方法结果；此前论文草稿仍对应此前统计，不能当作本轮更新稿。
论文草稿、论文图件副本和论文PDF本轮均不更新。下文原72,000字符选段及自动修复说明仍适用于此前流程，不适用于本次Full覆盖重跑。

五个维度统一为**历史原创性、认识与能力推进、知识组织与整合、证据基础、科学价值**。本目录保存独立后处理实验；旧分析、旧评分、历史原文和图计算结果只读复用。

当前主实验有599份有效核验记录：533份无参考定位技术缺失、66份含逐维技术缺失，另有1份Full缺失；“无技术缺失”不等于五维均有数值评分。Full五维实际可评分数依次为55、97、95、94、98，其他方法依次为56、98、96、95、99。未决、技术缺失和缺失不补零。DAMAGE独立保留，历史维技术缺失，其余四维为2、2、3、3。

本轮Full相对GEAR和Graph的五维配对平均分差均为正，95%区间均不含0。比较对应更新后的Full材料组织与生成要求，其他方法保留原条件，核验规则未变；评分衡量报告分析表现。准确数值见`summary.md`及`tables/paired_effects.csv`。旧详细分析的历史差异识别等结果属于不同评价对象，仍单独保留。

本轮只更新输出目录中的统计和Fig3。此前45页中文主稿、2页独立Limitation及论文图件副本未修改；下文有关论文交付的内容记录此前工作，不代表本轮新交付。

## 三个对象必须分开

| 对象 | 内容 | 对应产物/统计 |
|---|---|---|
| 目标论文的创新分析 | 贡献概述→五维具体判断→综合结论；各维保存理由、引句、范围和状态 | `reports/{method}/{paper_id}.json`及`.body.md` |
| 五维分析表现 | 核验新主体是否准确、完整、有据；0–3级分数评价报告怎样解释贡献 | 新`evaluations/`、逐维评分、配对差异与支持状态 |
| 保留的原详细分析 | 原有GEAR、Graph、综合分析或基线报告正文，逐字附在新主体之后 | 完整`.md`后半部；旧历史差异识别、贡献覆盖、断言、信息聚类、审稿匹配与偏好诊断 |

五维不相加为论文创新总分。准确指出新增有限可得高分。原附件不能补足新主体的遗漏；旧评分不能改名为新五维评分。“证据基础”评论文怎样支持贡献，独立核验评报告对这些证据及其他问题的解释是否准确。

主实验使用Fig3的100篇论文、六种方法，最多600个新报告槽位；DAMAGE为独立案例，单列分母。原600个报告文件与旧594组有效质量评估含义不同，二者均不能直接作为新评分的有效数量。真人框架依据统一采用994篇有实质性评审材料的论文。

## 材料、模型和运行边界

- `inputs/`保存完整来源及旧报告附件，私密本地路径只在审计中保留。除本轮Full外，原流程模型请求从原文中按72,000字符来源包预算选取片段，优先保证稿件覆盖，再兼顾历史来源、图事实及原文多样性；逐请求保存选中范围和coverage。原分析正文完整保留，不以生成摘要充当原始证据。
- 生成只读本方法已有材料。Direct不增加共享主张；GEAR与Graph材料条件独立；Full复用自身综合分析与来源。共用清单只读中立贡献与原始来源，不接收候选报告、旧评分或真人态度。
- 核验接收匿名新报告、共用清单、原始材料及生成阶段的可见材料范围。报告真实缺源说明与核验器新增可见材料分开；重要比较的实质遗漏仍按相同完整性规则评价。
- 统一模型路由为`gpt-5.6-luna`，生成`high`，清单和核验`xhigh`。使用现有Codex CLI登录和provider，不新增API密钥。每次`codex exec`使用独立临时目录、新会话、`--ephemeral`、只读沙箱、schema和JSON事件记录。违规工具使用的请求不纳入结果。
- 每次仅一个`Study/Scheduler`，生成、清单、核验、定位修复共享额度。试跑最多8；正式配置初始16、最大64。实际准入另受可用内存减4 GiB、每活动任务0.5 GiB预算约束。32次成功窗口升8，限流减半，超时比例达10%降8。限流、超时等暂时性失败最多一次同条件重试；低分、科学未决和明确拒绝不触发改写重跑。
- 主运行与恢复共用`.run.lock`。已有成功checkpoint按协议、模型、推理强度、提示、实际输入及schema指纹复用。科学字段不允许在格式修复中改变；确定性JSON语法恢复与受核验的无损格式修复分别记录。

## 原流程运行与中断恢复（不用于本轮Full）

命令均从项目根目录`/home/jayee/workspace/ASPR`执行。当前有批处理时只查看状态，不启动第二套runner。

```bash
python3 -m figure_pipeline.innovation_five_dimensions status
```

首次复现使用以下顺序。已有本次产物时保留目录，按同一入口续跑；`prepare`不调用模型、检索或图构建，但运行中不要改写其输入。

```bash
python3 -m figure_pipeline.innovation_five_dimensions prepare
python3 -m figure_pipeline.innovation_five_dimensions pilot --workers 8 --initial 8 --timeout 900
python3 -m figure_pipeline.innovation_five_dimensions run --workers 64 --initial 16 --timeout 900
```

`pilot`选择固定五篇开发论文，复用到全量；`run`默认包括百篇及DAMAGE。可用`--paper-id DOI后缀`限定任务，或`--cohort damage`仅处理案例。更换`--output`会使用另一独立输出根，需先准备其材料。中断后再次运行同一科学配置，已完成且指纹匹配的任务复用；不要调用旧研究中会重建或清理实验的入口。

`status/progress.json`记录当前并发和阶段状态；`status/all_completion.json`保存主批结束时的任务结果。runner主动退出1表示任务失败；宿主进程退出状态与科学任务记录分别保留。本次恢复的宿主会话返回143，但完成记录为103项全部结束、0项失败，随后601份当前产物全部通过聚合门禁；详见`validation/recovery_completion_host_status.json`。已完成任务没有因此重复执行。

## 失败参考清单的确定性恢复

主批结束、释放锁且`status/all_completion.json`存在后执行：

```bash
python3 -m figure_pipeline.innovation_five_dimensions.recovery --maximum 64 --initial 16 --timeout 900
```

该入口只选原主批中被参考清单失败阻断的评价；原报告与成功参考不重生成。DAMAGE优先排队，所有恢复评分仍共用一个Scheduler。可用`--paper DOI后缀`限定待恢复论文。重复运行复用有效评价，不挑选较高分答案。

恢复读取原模型checkpoint和已经保存的引文定位结果，只从**原请求已可见的原文片段及核心贡献引句**核对位置。原稿全文仅用于证实原位置、重叠和纯空白间隔，不补入未见科学段落。只容许确定性排版恢复或纯图号括号恢复，criteria、scope、reason、适用性、来源身份及解释保持不变。

`partial_references/{paper_id}.json`保存原清单、仍失败引句、原request/checkpoint指纹、可核验维度及技术失败审计。只有可核验维度进入评分模型；其余维度由程序写为`technical_failure/null`、空checks/quotes并附origin。全维技术不可用时不发空的模型请求。恢复审计写入`recovery_runs/{时间戳}/`，原all_completion保留；`status/recovery_completion.json`记录本次恢复。出现失败时CLI退出1。

| 状态 | 含义 | 数值处理 |
|---|---|---|
| assessed，含0分 | 问题可评；0可能表示漏答或实质错误 | 计入该维分布和合格配对 |
| unresolved | 共用科学参考不足 | 空值，不补0 |
| not_applicable | 清单确定确实不适用 | 空值，不补0 |
| technical_failure | 参考引文定位技术失败，由程序补齐 | 空值，单独技术失败表 |
| missing | 缺失/无效/过期产物或尚未完成 | 空值，记录原因 |

正式核验使用v3选择器：报告引句选择真实正文栏位R01–R17，来源引句选择可见片段和schema限定的合法子句下标。程序复制原句，多条来源分别保存；不把分离文本拼成虚假的连续引文。位置可查与科学支持分开判断，技术恢复不能把失败维称为通过。

## 汇总与数值复算

主批及必要恢复结束后：

```bash
python3 -m figure_pipeline.innovation_five_dimensions aggregate
```

入口重新构造各结果的实际依赖，经`aggregate.load_result`验证原输入、模型/提示协议、匿名映射、选择索引、来源定位和partial审计。存在文件不自动代表结果有效。读取成功参考优先；失败参考使用经严格重建的partial路径。

- `summary.json`、`summary.md`：完整、部分、全维技术失败及缺失情况，逐维结果，不预设有效分母。
- `tables/dimension_scores.csv`：所有预期论文×方法×维度槽位及状态。
- `tables/summary.csv`：逐维平均与区间、适用/未决/技术缺失等数量。
- `tables/paired_effects.csv`：Full分别与GEAR、Graph的主比较，其余方法为补充；单位是0–3级评分的**分值差**。
- `tables/support_by_paper.csv`、`support_summary.csv`：原句支持状态及范围，先论文内计算，再论文间等权。
- `tables/technical_failures.csv`、`missing.csv`：技术维缺失与其他无效/缺失记录分开。
- `tables/model_calls.csv`、`model_usage.json/md`：实际CLI调用、阶段时间与已报告token；缺失用量不当作0，不估算无法获得的交互子代理用量或价格。

配对只使用同一论文同一维度双方均`assessed`的结果，报告实际配对分母及10,000次论文层面bootstrap区间。DAMAGE不进入百篇配对。不同维度可以核查相同命题，其断言数不能相加解释为独立证据数。

## 绘图与旧结果对应

统计审核后再生成Fig3完整图：

```bash
python3 -m figure_pipeline.innovation_five_dimensions render
```

图面c1为新五维分布，c2为新配对差异，布局内部键分别是c1和c3。保留原整体复合图，同时更新授权流程文字。输出`figures/Fig3.svg/pdf/png`及`fig3_replacement_audit.json`。缺少尚未结束的任务时默认拒绝正式绘图；只有已审阅并完整记录真实缺失后，才可显式加`--allow-incomplete`，不能以此提前发布试跑。

| 结果/图件 | 当前对象 | 是否重算 |
|---|---|---|
| Fig3新五维分布、配对差异 | 新报告主体分析表现 | 新生成与新核验 |
| Fig3历史差异识别、贡献覆盖、断言、信息簇、审稿匹配及偏好 | 保留的原详细分析 | 原统计保留 |
| Fig4、Fig5 | 原分析环节的信息条件与证据响应 | 不重跑，不改名为新五维实验 |
| Fig2 | 五维输出名称与保留详细分析示意 | 仅授权文字更新 |
| 原Fig6 / ED1 | DAMAGE实际五维报告的英文概述 | 仅报告面板及投稿图号更新 |
| 新ED6 | 994篇真人关注点与五维来源映射 | 复用已有编码与原文，Python统计绘图 |

框架与Fig2可在不调用模型的情况下复现：

```bash
python3 -m figure_pipeline.innovation_five_dimensions.framework
python3 -m figure_pipeline.innovation_five_dimensions.fig2_update
```

DAMAGE先核对真实`reports/full/s41467-026-69179-5.json`与`figures/DAMAGE_editorial_en.json`的报告SHA256、逐维原句及英文忠实性。独立评价有效后才调用以下函数；不要在尚未完成时绕过门槛：

```bash
python3 -c 'from figure_pipeline.innovation_five_dimensions.damage_figure import render; render()'
```

完整或部分核验由`load_result`统一判断，至少一个维度实际受评才可展示；部分核验审计明确记录未评分/技术失败维，不能称五维全部通过。输出保持完整图，标题为Extended Data Fig.1，a–d及报告面板外对象除授权标题文字均不变。

## 此前中文稿与独立Limitation交付（本轮未更新）

已审阅工作稿保留在`draft/manuscript_zh_v9_pending.tex`这一交接路径，全部待填标记已清零；完成记录见`draft/manuscript_zh_v9_pending_items.md`。正式源码与PDF位于`manuscript/nature_innovation/manuscript_zh_v9.tex/.pdf`。正文和图注使用实际新统计，旧诊断单独保留。DAMAGE候选中文在`draft/DAMAGE_five_dimension_candidate_zh.md`；实际模型报告从不被编辑性文本覆盖。

```bash
python3 -m figure_pipeline.innovation_five_dimensions.finalize stage
python3 -m figure_pipeline.innovation_five_dimensions.finalize check --source outputs/innovation_five_dimensions/draft/manuscript_zh_v9_pending.tex
python3 -m figure_pipeline.innovation_five_dimensions.finalize publish --source outputs/innovation_five_dimensions/draft/manuscript_zh_v9_pending.tex
```

装配到`manuscript/nature_innovation/figures_v9/`，旧`figures/`与旧稿保留。`publish`拒绝PENDING、缺结果或缺整图的草稿，默认要求主队列完整评价；如果最终保留部分评价/真实缺失，主代理先核对分母与说明，再显式使用`--allow-incomplete`。完整图号映射、命令和资源路径见`delivery/ASSEMBLY.md`，逐文件审计见`delivery/figure_manifest.json`。

正文顺序保持：讨论结束→六张主图各跟图注→Methods→六张扩展图各跟图注，无标题/作者信息。新版Limitation依据为`draft/limitation_zh_v9.tex`，单独审阅、复制与编译，不能用旧Limitation覆盖。

在真实结果填入、整图齐备、稿件发布到`manuscript/nature_innovation/`后，从该目录分别编译：

```bash
/home/jayee/workspace/ASPR/outputs/innovation_five_dimensions/tools/tectonic --keep-logs manuscript_zh_v9.tex
/home/jayee/workspace/ASPR/outputs/innovation_five_dimensions/tools/tectonic --keep-logs limitation_zh_v9.tex
```

上述编译已成功执行。新稿引用`figures_v9/`，字体复用`fonts/`；45页主稿与2页独立Limitation均已检查。12张整图的尺寸、顺序和后续图注匹配，无缺失字形、未定义引用或明显裁切。整图页面保留原生尺寸；监督器原图中的工作流动作说明仍沿用原图，主稿没有Limitation章节。PDF检查记录见`validation/pdf_v9/visual_check.json`。

## 文件追溯入口

| 路径 | 用途 |
|---|---|
| `framework/` | 994篇主题统计、原文位置、八项主题边界审计、五维映射、ED6 |
| `inputs/` | 完整材料、原报告附件、方法隔离审计、名册 |
| `requests/` | 实际传入材料、匿名映射、选择器索引 |
| `reports/` | 新主体JSON/Markdown与附原详细分析的完整报告 |
| `references/`、`partial_references/` | 成功共用清单或原失败清单的确定性恢复记录 |
| `evaluations/`、`provenance/`、`audits/` | 逐维来源核验、评分、技术补齐和依赖指纹 |
| `checkpoints/`、`logs/calls/` | 原模型响应缓存、请求、事件、用量、失败记录 |
| `status/`、`recovery_runs/` | 当前状态、主批与恢复阶段记录 |
| `tables/`、`figures/` | 新统计、整图与不变区域审计 |
| `draft/`、`delivery/` | 已完成工作稿、独立Limitation、装配与交付记录 |
| `validation/` | 独立代码/统计/图文核查与针对性复现记录 |

追溯顺序为：真人关注点与Graph扩展→五维定义→实际报告主体→来源核验→逐维方法比较→图件与正文结论。任何阶段的已运行、已定位或已编译状态，均不自动等于科学判断正确或方法优于对照。

## 最终文件

- 中文阅读 PDF（45 页）位于 `manuscript/nature_innovation/manuscript_zh_v9.pdf`。
- 独立 Limitation（2 页）位于 `manuscript/nature_innovation/limitation_zh_v9.pdf`。
- LaTeX 与整图源码包位于 `manuscript/nature_innovation/nature_innovation_zh_v9_source.zip`。
- 五维主比较位于 `figures/fig3/data/five_dimensions/summary.md`。

英文稿未重新翻译。原论文、原报告、原图及旧统计均保留。
