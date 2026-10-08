# Fig.4 详细设计方案：逐 panel 排版、数据需求与实验接口

**英文图题：Fig. 4 | Complementarity, controlled component effects and fusion risks**

**设计依据：用户最后确认的五-panel 示意图。** 上排 a／b／c，下排 d／e；a 为实验对照关系地图，b 为三列雨云分布，c 为四条件三维交互展示及论文级交互分布，d 为热力散点与边际分布，e1 为错误去向马赛克，e2 为逐论文风险矩阵。

本文件是绘图与实验设计，不是新增实验结果。示意图中的均值、区间、相关系数、错误计数、风险比例、18 个示例论文列和密集点云均不直接进入正式图。初版设计时尚未执行七条件模型生成与科学核验；当前数据准备已启动，真实覆盖以运行状态和源表为准，本次不绘图。

---

## 当前实施口径（2026-10-01）

**最新范围调整（用户确认方案一）：主交付保留 panel a–d，撤下主图 panel e。**
七条件报告、H/V/R、信息聚类、跨贡献关系与组件／交互／联合图效应仍按100篇完整队列汇总。
剩余分支来源核验、错误归并、融合去向、纠正依赖性、分支存在性扫描及细分类风险任务停止，
已完成结果与原始记录保留为探索性补充；覆盖非随机，不外推100篇总体，不把未完成填零。
R保留为最终报告已确认不当断言比例，不据此声称错误由融合新增或纠正依赖另一分支。
运行范围见 `outputs/fig4_mechanisms/analysis_scope.json`；主数据完成门槛不再要求融合追踪产物。
下文五-panel排版与第9节panel e要求保留为原设计记录，不再作为本次主交付要求；
本次不绘图，a–d后续排版可调整，不用探索性e结果填充原主图。

本次数据准备采用固定既有分析的信息组件比较：复用 Fig3 的 GEAR、完整 Graph，以及 F/F−J 的逐贡献融合判断；M/P 条件从屏蔽后的事实重算解释与融合。结论不扩展为统一来源下重新运行整个系统的端到端消融效果。

七条件统一取消最终报告长度目标，保留有依据的信息与必要范围说明、去掉重复，实际篇幅和用量如实记录。共同历史原文合并既有可读来源，不新增全量检索或建图。所有新增生成／评价调用均为 gpt-5.6-luna，生成和聚类 high，科学支持和历史比较核验 xhigh。本次完成主评价与实际问题定点复核，暂不进行固定20篇重复评价；不编写或运行回归测试，仅核对代码逻辑。

2026-10-01 修复：先完成按固定领域顺序选择的5篇×7条件全流程，再追加其余95篇；已有结果复用，不按效果筛选。聚类使用候选簇ID合并并由代码展开成员，缺项定点修复；融合小批任务共享完整原始报告及必要证据，取消逐错误重复整理整套评价记录。生成、基础评价、信息聚类和融合事件设独立阶段，完整五篇仍为扩展前提。

实现入口为 `python3 -m figure_pipeline.fig4_mechanisms`，独立输出 `outputs/fig4_mechanisms/`，不覆盖 Fig3。700份报告和各评价槽位是目标覆盖量，不是已完成结果；实际进度见该目录 summary.json／run_log.jsonl。本轮不绘图。

## 1. 全图任务与阅读顺序

Fig.4 只回答“同一批输入和可比生成条件下，哪些信息与分析环节改变了输出，以及融合带来哪些纠错和风险”。不重复 Fig.3 的完整系统排名，不把参数、模型、领域与成本曲线搬入本图，也不展开 Fig.6 的长篇案例。

|Panel|英文标题|读者要获得的结论类型|主图型|
|---|---|---|---|
|a|Controlled contrasts|明确哪些条件互相比，移除了什么|四角对照地图＋三消融分叉|
|b|Component effects|五类组件分别改变历史判断、有效信息与错误多少|三列×五行雨云分布|
|c|Branch interaction|两分支是否存在加性尺度上的交互，论文间是否一致|四条件三维插值面＋交互量直方图|
|d|Joint-dependent gain|联合图专属结构与有效跨贡献解释增量的关系|论文级热力散点＋上、右边际分布|
|e|Error resolution and fusion risks|分支错误的去向，以及风险在论文间的分布|来源×去向马赛克＋8行风险矩阵|

a 是实验定义，b／c 是主要受控效应，d 是联合图配对效应与结构关联，e 是机制追踪与风险描述。d 的关联和 e 的构成不能单独替代受控对比。

## 2. 母版、位置与统一视觉

### 2.1 尺寸与大排版

按已确认示意图的约 4∶3 比例，设计母版为 **183 × 137 mm**，图注另排。保留一行紧凑总标题；正文排版不需要图内总标题时，可去掉约 5–6 mm 标题区并整体上移，不整体压缩字号。相较早期 183 × 120 mm 方案，本版增加的高度用于雨云、边际分布和逐论文风险矩阵，不增加 panel 数量。

坐标单位为 mm，原点在页面左上角。下表为初始施工坐标，真实文字装入后仅作小幅局部调整。

|Panel|x|y|宽|高|内部安排|
|---|---:|---:|---:|---:|---|
|a|3|8|49|67|标题；四条件地图；三消融卡；公共条件条|
|b|54|8|77|67|标题；三列标题；五行雨云；轴与图例|
|c|133|8|47|67|上部三维面；下部交互量分布|
|d|3|77|80|56|上边际；中央散点；右边际；轴与图例|
|e|85|77|95|56|上部 e1 马赛克；下部 e2 风险矩阵|

保留浅蓝细边框和白色图内区域，去掉厚阴影。内部组件不再各套一层大圆角框。不得将已生成整张示意图贴进最终图；它只作为构图参考。

### 2.2 内部施工区域

下表坐标相对所属 panel 左上角，包含该组件自己的标题／轴／必要标签，并非都等于坐标轴绘图区。

|组件|相对 x|相对 y|宽|高|内容|
|---|---:|---:|---:|---:|---|
|a1|2|6|45|34|T／E／G／F 四角对照地图|
|a2|2|42|45|17|从 F 分出的 J／M／P 消融|
|a3|2|61|45|4|共同输入条件窄条|
|b1|17.5|6|18|55|历史比较差值雨云|
|b2|37|6|18|55|有效信息数差值雨云|
|b3|56.5|6|18|55|已确认不当断言率差值雨云|
|b-labels|2|17|14.5|41|五个组件行名|
|b-legend|2|62|72.5|3.5|点、箱体、统计文字含义|
|c1|1.5|6|44|34|四条件曲面与加性参照|
|c2|2|43|43|21|交互量直方图、均值、区间|
|d-top|12|6|52|6|横轴边际分布|
|d-main|12|14|52|32|热力散点|
|d-right|66|14|8|32|纵轴边际分布|
|e1|2|5|91|20|来源错误马赛克及图例|
|e2|2|27|91|27|论文风险矩阵、领域带及 n/N|

d 的轴标题使用左侧和底部留白。b 的每列下方共享该列横轴；五行不分别重复刻度。e2 的类别标签占约 27 mm、矩阵约 53 mm、右侧统计约 9 mm，其余用于间隙。

### 2.3 字体、颜色与线宽

沿用资料包中 Times New Roman／Arial 的字体体系：图题和 panel 标题可用 Times New Roman，坐标、数字、复杂字段标签用 Arial；缺少字体时明确使用可用替代字体，不静默混入多套字体。

最终尺寸字号：总标题 9–10 pt，panel 字母 10–11 pt，panel 标题 8–8.5 pt，正文／行名／轴标签 6.5–7 pt，紧凑数字注释约 6 pt。标签太长优先改为两行短标题、扩大局部标签区或放到单 panel 放大版，不不断缩字。

|用途|颜色|说明|
|---|---|---|
|GEAR|`#E16C35`|沿用 Fig.2 角色色|
|Graph|`#1269DF`|沿用 Fig.2 角色色|
|Full|`#713BCB`|沿用 Fig.2 角色色|
|Joint 局部对照|`#9A73CC`|本图使用紫色系，不表示另一种模型|
|结构摘要／路径摘要|`#778391`／`#A3ADB7`|中性辅助色，靠标签区分|
|主文字／浅框线|`#172033`／`#D4E3F0`|不使用浓重蓝底|
|已纠正／未传播／仍传播／未决|`#69B991`／`#75A9E5`／`#E97781`／`#ADA7BC`|仅用于 e1 的结果状态|
|新错误／已支持内容改坏|`#E97781`／`#9A73CC`|e2 两个风险家族|
|仅遗漏／结构性不适用|`#9AA2AC`／浅灰|仅遗漏不是错误断言|

密度／曲面使用单调可读的连续色阶，建议蓝—青—浅金，不使用方法角色色冒充数值色阶。c 色条表示信息产出高度，d 色条表示局部密度，两者分别标单位，不能合并。

线宽：坐标轴 0.55–0.7 pt，参考线 0.45–0.6 pt，框线 0.35–0.5 pt，强调连接 0.75–0.9 pt。网格仅淡化辅助定位，不盖过真实数据。

## 3. 当前数据、实验条件与复用范围

### 3.1 当前对象

依据当前数据包 `outputs/fig3_reference/study/derived/`：100 篇论文、717 条合法共享贡献、200 条核心贡献；本轮直接检查 `core.csv`，100 篇均有 2 条核心贡献。现有方法—论文统计对象为 594 个；这是旧原生比较的可用对象数，不是 Fig.4 新实验完成数。

717 条贡献是生成与报告对齐对象；200 条核心贡献是历史判断评价对象。两者不能互换分母。旧 660／719 条贡献、600 份报告或 200 篇配置不恢复。

旧 Full 比其他方法使用更宽松的报告长度条件。因此旧 Full−GEAR／Graph 差异不直接充当本图的受控差异；原文、贡献、独立参考和未变化的原生图事实可复用，新的条件报告及其评价要重新生成。

### 3.2 七条件

信息列固定为 E／S／J／M／P：显式 GEAR 比较、单贡献 Graph、显式 Joint Graph、显式非路径结构数值、显式父论文路径注释。

|condition_id|图中标签|E|S|J|M|P|含义|
|---|---|---:|---:|---:|---:|---:|---|
|T|T|0|0|0|0|0|共同历史文本＋共同贡献任务|
|E|E|1|0|0|0|0|共同文本＋GEAR 分析|
|G|G|0|1|1|1|1|共同文本＋完整 Graph 分析|
|F|F|1|1|1|1|1|共同文本＋两个分支|
|F_noJ|F−J|1|1|0|1|1|联合图阶段替换为文本跨贡献综合|
|F_noM|F−M|1|1|1|0|1|屏蔽显式非路径结构数值及依赖解释|
|F_noP|F−P|1|1|1|1|0|屏蔽显式父论文引用路径注释及依赖解释|

计划 100 × 7＝700 份最终报告，并非 700 次模型调用。T／E／G／F 均是任务匹配条件；T 不等于旧 Direct，G 不是 single-only。

### 3.3 共同固定与屏蔽规则

所有条件固定论文、共享贡献 ID、可用历史原文并集、来源定位、截止规则、模型与实际推理设置、报告栏目；七条件均不设正文长度目标。统一下游整理与writer任务，实际字数、token和调用量如实记录，不声称等额token。

移除某阶段时提供等预算的普通原文阅读／整理机会；F−J 用同机会的文本级跨贡献综合替代，不禁止模型自行思考多条贡献。各条件的最终writer、逐贡献整理机会保持一致；既有分支判断作为固定输入复用，共用替代文本任务只生成一次。

共同来源只包括可定位的历史原文，不包括其他条件的 GEAR 判断、Graph 解释或评价答案。GEAR 与 Graph 在融合前保持独立，不将一个分支的结果反向灌入另一个分支。

M 对应 D3–D11 与相应联合结构数值摘要；保留 D1–D2、节点／边／社区身份和 P。P 对应 D12–D14、逐邻居路径字段、路径见证及泄露这些关系的摘要；来源 ID、DOI、引句、原文位置仍保留。按字段名称屏蔽，不按数组位置猜测。

屏蔽发生在解释生成前；依赖被移除输入的解释、融合与 writer 必须重算。旧解释中可能已经写出“合并多少分量”“直接引用”等信息，不能只删原始数字而复用这些解释。

历史时间边界沿用当前协议；记录材料和图资产实际可用状态。不能仅凭局部节点日期过滤就宣称所有全局社区和路径资产都无未来信息。无法证实的资产边界明确说明，不伪造已完成验证，也不为本图擅自增加全量下载或建图任务。

## 4. 三项共同终点与统计单位

令 i 为论文、m 为条件。

### 4.1 H：历史比较正确覆盖率

先依据独立参考确定该论文“可判定且适用历史比较”的核心贡献集合 K_i，与方法输出无关。

`H_im = 100 × 正确回应的适用核心贡献数 / |K_i|`。

“正确回应”同时要求没有遗漏、历史对象与范围正确。正常生成但未回应计 0；原始参考不足导致的不适用事先固定；技术漏评使该论文—条件—指标为 NA，不删掉漏评对象后重新算一个更漂亮的比率。

`|K_i|=0` 为 NA。主图差值单位是百分点 pp，不是相对提升百分数。

### 4.2 V：有效非重复信息簇数

`V_im = 该报告中经独立来源支持、范围正确、非作者同义复述的不同信息簇数`。

所有条件的候选内容匿名并集后，在论文内按命题、对象、关系与范围对齐，再分别核验每个条件原本的表述。跨贡献内容在论文级只计一次。主类别保留历史增量、跨文献关系、范围修正、跨贡献关系；可以另外记录次级标签，但总体计数去重。

支持范围必须是报告实际写出的范围。评价者自行缩窄／修补后才成立的命题，不能反向算作原报告已经正确。未知支持状态不计入 V，另报未知比例；技术漏评时不把未完成产出当成 0。

### 4.3 R：已确认不当断言率

采用与现有断言评价相容的原子单位：`needs_verification or substantive`。在全部这些单位均有评价记录的前提下：

`R_im = 100 × 至少含一种已核实不当判断的单位数 / 全部需核验或实质单位数`。

错误包括无依据确定性、虚假先例／首次性、语义跳到因果、必要范围遗漏等。一个单位包含多个错误也只进入总体分子一次；总体错误率不得把各类型直接相加。

有记录但科学证据无法判定的单位仍保留为 unknown，并单独报告 `unknown_support_share`。R 是已确认不当判断的观察比例，不是全部科学错误的穷尽率；`1−R` 不能叫正确率。缺任何必要评价记录时 R 为 NA；零断言分母为 NA，不能写成“100% 正确”。

### 4.4 共用配对规则

b 先对每篇同指标算 F−对照；c 用 T／E／G／F 四条件共同完整论文；d 用 F／F−J 完整配对且有合法结构分母的论文。各图均保留实际有效 n。

总体效应以论文等权宏平均为主。建议沿用 10,000 次论文 bootstrap，论文内全部贡献、断言与配对条件一起重采样。IQR 是论文效应分布，不是均值的 95% CI；重复裁判和 bootstrap draws 不增加论文 n。该区间不自动覆盖单次生成的随机性。 主图默认展示点位区间而非显著性星号；若报告显著性，b的15项组件×终点比较作为预先声明的检验家族进行Holm等多重比较处理，c的交互检验另行声明。全零样本导致的退化区间不证明总体风险绝对为零。

## 5. Panel a｜Controlled contrasts

### 5.1 画面内容

a1 为四角对照地图：T 左下、E 左上、G 右下、F 右上。横向 T→G、E→F 标 `+Graph`；纵向 T→E、G→F 标 `+GEAR`。箭头表示实验条件之间增加的信息，不是执行顺序。

节点采用平面圆角矩形，大字条件代号＋一行短说明：`Text`／`+GEAR`／`+Graph`／`Both`。不需要文档、齿轮等装饰图标；图注给出完整含义。

a2 在四角图下放三张等宽小卡：`F−J / Joint removed`、`F−M / Metrics hidden`、`F−P / Paths hidden`。三条连接必须从 **F** 出发，沿右侧预留走廊下降后分叉，不从 G 发出，也不穿过 G 的节点。

每卡底部是五槽 E/S/J/M/P，小实心槽表示保留、空心表示隐藏。F−J＝11011，F−M＝11101，F−P＝11110。卡上英文尽量两行，不挤进解释段。

a3 只保留 `Shared claims · sources · model · writer`。

### 5.2 数据与接口

输入 `conditions.json` 和 `contrasts.csv`。最少字段：`condition_id, label, gear_on, single_graph_on, joint_on, metrics_on, paths_on, replacement_stage`；对照表含 `from_condition, to_condition, operation, edge_label`。

不需要新科学评价；需要真实运行配置。图中的信息槽应由同一配置生成，不手绘一套与运行不一致的开关。

### 5.3 Python 实现与允许结论

使用固定坐标、`FancyBboxPatch`、`FancyArrowPatch` 和普通文字，不使用随机 network 布局。a 只证明条件定义清楚，不证明组件有益。

## 6. Panel b｜Component effects

### 6.1 主体排版

五个行标签共享于左侧，顺序不按结果排序：GEAR、Graph branch、Joint、Structural summaries、Paper paths。三列从左到右为：

`Historical comparison (pp)`；`Valid insights / report`；`Improper assertions (pp)`。

列标题下写 `Higher is better`／`Higher is better`／`Lower is better`。总副标题或图注统一写 `Full minus ablated`。Graph branch 指整个 Graph，不再标 single-claim。

|行|对照|对应差值|
|---|---|---|
|GEAR|F vs G|H/V/R(F) − H/V/R(G)|
|Graph branch|F vs E|H/V/R(F) − H/V/R(E)|
|Joint|F vs F−J|H/V/R(F) − H/V/R(F−J)|
|Structural summaries|F vs F−M|H/V/R(F) − H/V/R(F−M)|
|Paper paths|F vs F−P|H/V/R(F) − H/V/R(F−P)|

### 6.2 每一格的四层信息

上层：半小提琴／密度轮廓；中间：Q1–Q3 小矩形与中位数短竖线；下层：真实论文差值点，必要时仅在纵向轻微错开；格子下缘或独立数值栏：论文平均差值与 95% CI 的文字。

不连接两个圆点，不使用哑铃，不把 IQR 的边端画成大球。可取消箱须，因为真实点已经显示全范围。五行共享该列横轴范围与零线，三列因单位不同独立定范围。

颜色表示组件身份，不再保留示意图底部统一的 `Effect magnitude (pp)` 渐变条：中间列不是 pp，而且雨云本身不使用这个色条，保留会误导。

### 6.3 离散数据必须原样显示

每篇当前只有两条核心贡献；适用集合为 1 或 2 条时，H 常是 0、50、100 等离散值，差值也离散。b1 因而以真实点列与箱体为主要数据，半密度仅作视觉辅助，不能为画出宽厚连续“云”而生成中间分数。

b2 信息数也是整数差值；b3 通常更连续。密度参数只按统一的绘图规则设置，不按谁显著来调。少量样本、全同值或极低取值多样性时，保留堆叠点／箱体，退化为窄轮廓，不强行拟合奇异 KDE。

每格最多 100 个论文点。10,000 次 bootstrap 只用于区间，不当作论文散点。缺失论文没有数据点，但在有效 n 中体现。

### 6.4 数据表

`component_effects_paper.csv` 每行一个论文×组件×终点：

`paper_id, component, metric, full_condition, comparator_condition, full_value, comparator_value, delta, eligible, missing_reason, full_denominator, comparator_denominator`。

满覆盖计划最多 100×5×3＝1,500 行；保留不适用状态时文件可含这些空值行，不产生假的数值。均值区间写入 `component_effects_summary.csv`：`component, metric, n_papers, mean_delta, ci_low, ci_high, q1, median, q3`。

只有 summary.csv 的 15 个均值，画不出真实雨云；必须读取逐论文差值。

### 6.5 解释边界

五个消融效应存在嵌套与重叠，不能加成 100% 或拼成提升瀑布。R 的正值是新增风险，不能暗中反转符号。跨零、负效应与大量零值照常呈现。

## 7. Panel c｜Branch interaction

### 7.1 c1：保留三维视觉，但只有四个实际条件

横轴 `GEAR (off/on)`，纵轴 `Graph (off/on)`，高度 `Valid insights / report`。四角固定为 T=(0,0)、E=(1,0)、G=(0,1)、F=(1,1)。G 不得误标为 O。

四点取同一组四条件完整论文的宏平均。面只做双线性视觉插值：

`z(x,y)=(1−x)(1−y)·mean_T + x(1−y)·mean_E + (1−x)y·mean_G + xy·mean_F`。

必须写 `4 conditions; bilinear interpolation`，不能标成连续参数扫描或许多中间组合的实际测量。网格细度只影响渲染；不能随意平滑拟合额外峰值。

可叠加一个稀疏、浅灰的“纯加性参照面”：

`z_add(x,y)=mean_T+x(mean_E−mean_T)+y(mean_G−mean_T)`。

在 F 角从 `mean_E+mean_G−mean_T` 到 `mean_F` 标一段细竖线／括注，名称为 `Interaction residual I`。不预先写 `Synergistic gain`；正、零、负结果都可能。曲面颜色表示 z，而不是 I。参照面若超出当前高度范围，扩展坐标，不静默截掉。

初始视角可用 elev≈25°、azim≈−135°，随后仅为避免标签遮挡微调。使用较弱透视或正交投影，四角标签在外缘；曲面浅色，参照面最多几条网格线，避免两层密网互相干扰。

### 7.2 c2：论文交互量分布

逐论文计算：`I_i=V_iF−V_iE−V_iG+V_iT`。

使用灰色直方图；必要时加很浅的密度轮廓。整数取值跨度小时以整数对齐分箱，跨度大时采用少量有明确边界的箱。横轴保留 0；均值用紫色竖线，均值 95% bootstrap CI 用浅紫竖向带，不画两个圆点的区间图。

图内给 `mean I [95% CI], n`；区间带表达均值不确定性，不是覆盖95%论文的范围。正效应论文比例可放源数据，避免追加第三个小图。

### 7.3 数据

`interaction_paper.csv`：`paper_id, V_T, V_E, V_G, V_F, I, complete, missing_reason`。

`interaction_summary.json`：共同有效 n、四条件均值和区间、I 的均值和区间、渲染用双线性系数。c1 与 c2 必须使用完全相同的论文集合，不按每条件分别剔除后再连接四个均值。

### 7.4 允许的结论

I 描述本信息数尺度上的交互，不是生物／科学机理的因果证明。Full 超过单分支不自动意味着 I>0；I≈0 不代表 Full 无用；I<0 可能来自冗余、压缩或干扰。错误方向需结合 b/e，不从信息数单独判断质量。

## 8. Panel d｜Joint-dependent gain

### 8.1 主散点：每点一篇论文

横轴：`Joint-exclusive connectivity, J_topo`，合法范围 0–1。可仅显示实际占用范围，例如 0–0.1，但保持原比例单位，不默默转 z-score。示意图的 −2 到 2 不能照搬。

纵轴：`Added valid cross-claim insights`，即 `V_cross(F)−V_cross(F−J)`，单位 clusters/report，允许正、零、负值。

跨贡献信息必须涉及至少两条目标贡献，并由报告原句、目标原文及必要历史来源支持。只有共享邻居、连通增益或 Joint 解读本身不够。仅评价者修补后成立的不计作原报告有效内容。

### 8.2 联合专属结构量

对论文 i 构建同一历史邻域并集 H_i。联合插入和每条贡献单独插入均基于这个 H_i；不拿不同大小单贡献小图直接相减。

R_all 为联合插入新增连通的历史节点对；R_c 为单条贡献插入新增连通的历史节点对：

`J_topo = |R_all \ union_c R_c| / choose(|V_H|, 2)`。

只计无序历史节点对；不加入 target-target 边，不计新增贡献节点自身。|V_H|<2 时 NA，可计算而没有专属连接时为真实 0。

这是建议新增的派生结构量，不修改 Fig.1 原始指标，也不把它叫创新分数。原始节点／边、每条插入的历史邻居身份是必要材料；只有均值指标不能复算该集合差。

### 8.3 热力层与边际层

中央先画稀疏热密度底图或少量等密度层，再画所有真实论文点，点不被底图淹没。建议 4–6 级密度，带宽固定规则记录；不为了填满画面生成数千个点。取值重叠可用绘图错位，但统计始终使用原坐标，并在图注说明。

上边际显示 J_topo 的分布，右边际显示跨贡献增益的分布；两者严格共享中央坐标范围。J_topo 的 0 可能是大点质量，应明确标 `J_topo=0: n/N`，不能用平滑把 0 扩散成大量正连接。

SciPy KDE 对退化协方差或全同值数据不适用时，使用二维分箱计数底图／真实点，保留热力散点版式而不伪造连续关系。密度估计不得延伸到负 J_topo 区域并当作真实样本。

### 8.4 趋势线、参考线与标注

保留 y=0。竖线最多表示明确标注的 x 中位数；若中位数为0或无解释必要就不画，不能固定0.5暗示科学阈值。

趋势线若绘制，作为描述性线性拟合，带为论文 bootstrap 的点位均值区间；数据退化时不拟合。主注释可写 Spearman rho、区间和实际 n，不预填 r=0.62 或显著 p 值。

只标2–3个真实论文短ID或中性区域说明。负纵值表示“有效跨贡献内容减少”，不自动叫“Joint 引入错误”；需要相应错误核验才能使用风险措辞。高J、低增益与零J、正增益都保留。

### 8.5 数据

`joint_effects_paper.csv`：`paper_id, history_node_count, history_pair_denominator, new_pairs_all, new_pairs_single_union, joint_exclusive_pairs, J_topo, V_cross_F, V_cross_F_noJ, delta_cross, improper_cross_F, improper_cross_F_noJ, complete, missing_reason`。

`joint_graph_records.jsonl` 保存图节点／边与插入邻居，以支持复算；`joint_relations.jsonl` 保存贡献对、报告原句、原文支持和规范化命题。将全部44条既有支持／缩窄支持关系只作为规则校准材料，不能替代新队列的全量结果。

## 9. Panel e｜Error resolution and fusion risks

### 9.1 e1：来源×去向马赛克

上半部三个来源大区依次为 GEAR-only、Graph-only、Shared。来源按实际进入F融合的发现去重确定，不由E/G最终报告倒推。

令 N_s 为来源s的去重错误数，N_so 为其去向o的计数。在可用总宽W内：

`width_s=(W−gaps)×N_s/sum_s N_s`；`height_so=H×N_so/N_s`。

四种去向的上下顺序固定：Corrected/narrowed、Not propagated、Still propagated、Unresolved。色块面积对应该来源×去向的错误数量；比例是在来源内部计算，不是全部论文宏平均。

顶端放短来源名与 n；足够大的格内放 `n (within-source %)`。小格保留面积，不硬塞文字；确切计数在源数据。来源为0时不强行画宽块，写“no observed eligible issues”。所有来源总数为0时保留空位说明，不造彩块。

**状态判定：**F明确写出且原文支持的修正／缩窄为 corrected；不再提到但没有明确修正为 not propagated；同一错误仍存在为 propagated；原文／对齐不足为 unresolved。若F一边纠正一边仍传播同一错误，单独保留两个flags，互斥主去向保守归为 propagated。沉默不算纠错。

可在 corrected 色块的内部用较深细分区／斜线表示“本次运行中另一分支依赖的纠错子集”：F有已核实修正、去掉另一分支后没有该修正，并且实际另一分支输入存在相关反证。这是 corrected 的子集，面积不能重复计数。Shared 错误不强行做单一分支归因。此标记只追踪本次运行，不宣称已证明稳定因果机制。

**e1 必需字段：**`paper_id, canonical_issue_id, proposition, scope, origin_branches, fusion_input_keys, exposed_to_F, source_group, full_status, full_unit_keys, source_ids, original_locations, correction_present, error_propagated, without_other_branch_status, other_branch_counterevidence, branch_dependent_correction, audit_status`。

汇总 `error_transition_counts.csv` 最多 3×4＝12 种来源×去向组合，另存受控纠错子集数量。旧4250条错误对齐记录并不直接等于新实验中独立的去重错误数。

### 9.2 e2：8行×100论文的风险矩阵

下半部所有论文一列，缺失也保留该论文列。图中18列只是草图占位，不是把正式队列改成18篇。按现有论文领域再按paper_id固定排序，不按“让Full更好看”的结果排序，不凭空加入无样本的领域。

上部四行 New Full-only errors：

|risk_type|图中短标签|判定|
|---|---|---|
|scope_inflation|Scope inflation|F新增、实际分支输入中没有的范围夸大|
|false_antecedence|Wrong antecedent|F新增且经核实的错误先例判断|
|semantic_causal|Causal leap|F新增的无依据因果／技术依赖跳跃|
|other_new_error|Other error|其余有明确核验依据的新错误，保留具体类别|

下部四行 Changes to supported content：

|risk_type|图中短标签|判定|
|---|---|---|
|unsupported_downgrade|Unsupported downgrade|对本有支持的命题无依据地降级，不含合理保留|
|wrong_modification|Wrong modification|改写后成为错误命题|
|context_dropped|Context dropped|必要条件／范围被删除并使表述失真|
|omission_only|Omission only|只是不再输出有效内容，没有把它改写为错误|

新错误要求实际分支输入中不存在相同命题／范围错误，并且F断言确实被独立核实为不当；“分支没说过”本身不是错误。来自支持内容的错误改写归入第二家族，不重复称为全新独立错误。一个论文可以在多行出现；图中行比例不能相加为100%。

**单元格状态：**

- present：至少有一个已核实事件，实色；即使还有其他待核对象，也已经可以确认该论文存在此类风险。
- absent：完成该类相关对象的评价，并确认没有事件，浅色。
- unresolved／incomplete：尚不能判定有没有此类事件，灰色斜线；不能用浅色0冒充。
- not applicable：没有该类风险的适用对象，例如没有任何已支持分支命题可供“改坏”；空白加细点纹，与unknown区分。

右侧 `n/N` 中 n 为present论文数，N为present＋absent的可判定适用论文数；同时在源数据保留unknown与不适用数。此比例是“可判定论文中的风险出现比例”，不是全部100篇断言错误率，也不是要求N每行都等于100。图中只写n/N，百分比与精确分母放表，避免窄列挤满。

顶部领域条采用实际元数据。约53mm绘图区容纳100列时，每列约0.53mm；不显示P1到P100全部标签，最多按领域边界或每20篇显示定位刻度，完整列序写入 `paper_order.csv`。每行约1.7–2mm。单元格之间只留极细白缝，不画粗网格。热图必须nearest或逐矩形绘制，不插值出不存在的中间类别。

上下两家族分别用珊瑚红和紫色；omission_only用中性灰。未知与不适用使用纹理／独立灰色，不靠相近浅紫区别。

**e2 必需字段：**`paper_id, domain, paper_order, risk_family, risk_type, event_count_confirmed, eligible_unit_count, assessment_complete, unresolved_unit_count, applicability, cell_state, event_ids`。

需要底层 `fusion_risk_events.jsonl` 保存每一风险事件、原分支命题、F原句、来源和类型，再聚合 `paper_risk_matrix.csv`，不是根据总体百分比分配随机色块。满队列为100×8＝800行状态记录，未判定可留空状态，不能填实验数值0。

### 9.3 e 的主图外摘要

完整统计另保留F新增已确认错误/全部已评价断言，以及已支持命题被改坏的比例；它们分母不同。仅遗漏单独报告。不能把e1纠错数减去e2受影响论文数或新增断言数来造“净收益”。

## 10. 一套共享数据支撑全图

### 10.1 可复用与必须新增

|材料|复用方式|本轮还缺什么|
|---|---|---|
|100篇、717共享贡献、200核心贡献|保持ID和原文位置|补全可读取的原始材料|
|独立历史参考|保持适用性和判断规则|新报告回应与正确性标签|
|原生单贡献／联合图|输入未变化时复用事实|M/P/J条件的依赖解释，以及J_topo|
|现有断言、簇与错误表|复用字段与评价程序|七条件新报告的全部标签和去重对齐|
|模型调用和writer|复用环境与接口|共同writer、替代阶段和屏蔽条件|
|现有44条已支持关系|校准细则、定位来源|不充当全队列新标签|

资料包未含全部全文、EvidenceStore、大型图、完整报告与调用记录。完整新生成需要从原工作目录读取这些既有材料，不宣称仅凭统计ZIP已经能端到端重跑；不为填空编造引句。

### 10.2 建议共享表

|文件|粒度／关键字段|用途|
|---|---|---|
|conditions.json|7条件开关、共同任务、模型与writer设置|a与实验运行|
|papers.jsonl / claims.jsonl / cores.jsonl|固定身份、领域、原文位置|全图|
|history_sources.jsonl|paper_id、source_id、date、passage、locator、时间状态|共同来源|
|report_units.jsonl|paper、condition、unit_id、quote、claim_ids、支持／范围／错误|b、d、e|
|information_clusters.jsonl|paper、canonical_cluster、成员、类别、每条件支持|b、c、d|
|fusion_inputs.jsonl|F实际看到的分支发现与来源键|e|
|fusion_transitions.jsonl|去重错误、来源、F及对照去向|e1|
|fusion_risk_events.jsonl|风险类型、前后原句、来源、判定|e2|
|paper_metrics.csv|paper、condition、metric、value、分子分母、缺失原因|b、c、d|
|run_log.jsonl|实际阶段、输入文件名、模型、用量、状态、原因|运行记录与Fig.5复用|

### 10.3 直接喂给绘图的表

|文件|预计最大／固定规模|内容|
|---|---|---|
|contrasts.csv|明确列举的条件边|a 对照关系|
|component_effects_paper.csv|1,500个论文×组件×终点槽位|b真实点|
|component_effects_summary.csv|15行|b均值、CI、IQR、n|
|interaction_paper.csv|100行|c四条件值与I|
|interaction_summary.json|1套统计|c1四角与c2区间|
|joint_effects_paper.csv|100行|d结构与配对增益|
|error_transition_counts.csv|12组合|e1面积|
|paper_risk_matrix.csv|800状态槽位|e2真实类别|
|paper_order.csv|100行|e2列序及领域|

这些行数是计划槽位数，不代表已经完成评价。缺失状态有记录不等于有实测数值。

## 11. 实验与代码实施顺序

### 11.1 普通文件流程

新增建议目录 `figure_pipeline/fig4_mechanisms/`、`outputs/fig4_mechanisms/`。使用普通JSON／JSONL／CSV和日志；不增加哈希、指纹、版本控制、自动失效或全目录清理机制。不调用当前Fig.3会清除结果的通用all入口。

先准备固定输入和来源并集，再实现七条件暴露与共同writer，再生成报告，再做一次共享的断言／信息簇／关系评价，最后聚合并绘图。e在同一套报告上补输入追踪和风险事件，不另生成一套报告。

### 11.2 试运行与全量

从100篇中按输入领域、贡献数、历史邻居／路径可用状态选10篇，不按Full胜负筛选。试运行10×7＝70份报告；科学条件不变则它们进入主队列，再补90×7＝630份。条件变化时显式重跑受影响对象，不拼接不兼容结果。

本次暂不安排固定20篇重复评价，只对实际出现的引句定位、范围争议和事件对齐问题进行定点复核。独立评价指不读取系统自身判断作为答案；同模型重复不自动等于跨模型或人工证据。人工实际抽核了多少，就报告多少，不把未做部分写成专家确认。

### 11.3 复用代码的定位

`fig3_revision/scientific_tasks.py::write_scientific_report`：共同writer；`fuse_claims`：实际分支输入与融合记录；`native_graph.py`：原生邻域；`gear/innovation/joint_graph.py`：联合历史并集和插入；`models.py`／`aggregate.py`：原子断言、支持状态、分母和论文聚合。

新增建议模块：`prepare_inputs.py`、`run_conditions.py`、`evaluate_outputs.py`、`aggregate_fig4.py`、`render_panels.py`、`assemble.py`。当前数据准备入口已实现于figure_pipeline/fig4_mechanisms；绘图模块不在本次实施范围。

## 12. Python实现与交付

全部图形可用NumPy／pandas、Matplotlib及按需SciPy完成；e1可选择statsmodels.mosaic，但为了精确控制面积、文字与配色，手工矩形也足够。

|Panel|主要绘图对象|
|---|---|
|a|FancyBboxPatch、FancyArrowPatch、text|
|b|fill_between半密度、scatter真实点、Rectangle四分位框、median短线|
|c|mplot3d.plot_surface、plot_wireframe、hist、axvline、axvspan|
|d|scatter、contourf／pcolormesh、hist、fill_between|
|e1|Rectangle按真实计数比例拼接|
|e2|imshow(interpolation='nearest')／Rectangle、ListedColormap、hatch|

毫米坐标转Matplotlib归一化轴框：`[x/W, 1−(y+h)/H, w/W, h/H]`。高密度层可栅格化，文字、线条、矩形与坐标尽量矢量。

正式交付目标：`Fig4.svg`／`Fig4.pdf`、预览PNG、a–e单独的可编辑矢量输出、全部绘图源表、共同配置和普通日志。单panel可输出更宽的阅读版显示完整CI和paper_id，但不得以放大版替代主图字号检查。初版交付为设计说明和JSON配置；当前实施交付七条件数据准备代码、实验数据和绘图源表，不生成图形文件。

装入真实数据后检查：零线、单位、n与图中点数；G条件完整性；F消融连接的起点；四角插值标记；b1离散值；d比例范围；e1宽高分母；e2三类缺失状态、真实领域和100列；未决不能变为0。所有方法和不利结果按同一规则保留。

## 13. 正式图注模板（待填真实结果）

Controlled analyses used the same paper cohort, shared contributions, time-eligible source passages, model configuration and report-generation task. (a) Task-matched branch contrasts and component-specific ablations of Full. (b) Paper-level Full-minus-ablated changes in correct historical-comparison coverage, validated non-redundant information yield and confirmed inappropriate assertions. Rainclouds show observed paired paper values; boxes show the interquartile range and medians, while numerical intervals refer to the mean paired effect. (c) Four observed branch conditions displayed with a bilinear visual interpolation and an additive reference, together with the distribution of paper-level interaction estimates. Intermediate surface positions are not additional experimental conditions. (d) Paired changes in validated cross-contribution insights versus joint-exclusive historical connectivity, with marginal distributions. Connectivity is a structural descriptor, not proof of a scientific or causal relation. (e) Source-checked outcomes of unique branch errors and paper-level fusion risks. Mosaic areas represent issue counts; matrix cells represent the presence, assessed absence, unresolved status or non-applicability of each risk type. Reported intervals use paper-level resampling. Actual sample sizes, applicable denominators and independent-assessment coverage are reported with the source data.

正式图注只保留实际完成的控制和评价，填入真实n及复核覆盖；不要保留占位效果或写出尚未实施的人类验证。

## 14. 依据与方法实现参考

项目依据：本轮上传的 `Fig1_Fig2_Fig3_code_statistics_20260930.zip`；此前 `Fig4_数据代码核对记录.md`；已确认的最后一版Fig.4示意图。具体字段以本包 `ASPR/outputs/fig3_reference/study/derived/` 和对应代码为准。

本轮额外核对了core.csv的每篇核心数、aggregate.py中历史比较与断言分母、Fig.1/2样式文件。以上是初版设计时的核对范围；当前实施另在工作区启动七条件生成和科学评价，完成情况以 Fig4 运行日志为准。

实现参考（官方文档；用于绘图能力和参数说明，不提供本研究效果的证据）：

- Matplotlib, Violin plot basics：`https://matplotlib.org/stable/gallery/statistics/violinplot.html`
- SciPy, scipy.stats.gaussian_kde：`https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html`
- Matplotlib, 3D surface (colormap)：`https://matplotlib.org/stable/gallery/mplot3d/surface3d.html`
- Matplotlib, Annotated heatmap：`https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html`
- statsmodels, graphics.mosaicplot.mosaic：`https://www.statsmodels.org/stable/generated/statsmodels.graphics.mosaicplot.mosaic.html`
- Matplotlib, FancyArrowPatch：`https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html`
