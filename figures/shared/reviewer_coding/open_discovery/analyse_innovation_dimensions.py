"""Source-bound synthesis of innovation concerns; no model calls or recoding."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(path for path in ROOT.parents if (path / "AGENTS.md").is_file())
SPEC = PROJECT_ROOT / "manuscript/nature_innovation/five_perspective_analysis_spec_zh.md"
THEMES = ("T01", "T02", "T03", "T04", "T05", "T06", "T62", "T64", "T74", "T75")

# Excerpts are selected for conceptual clarity, not as a representative sample.
EXAMPLES = [
    ("E01", "P002", "T01", "历史原创性：确定创新对象", [
        "The paper seemingly overclaims novelty: SAT was not invented by the authors, but rather was previously reported in references [42] and [50].",
    ], "审稿者1指出SAT技术本身已有前作。其后文建议把贡献表述为将既有SAT用于物理神经网络训练并提高鲁棒性。评价必须定位具体新增对象，不能把一种既有方法的名称整体当作创新。", "core"),
    ("E02", "P002", "T06", "知识整合：联系产生了什么认识", [
        "I appreciate the insight that the authors establish a connection between sharp minima in the loss landscape and the poor robustness observed in physical systems.",
    ], "审稿者2肯定的是损失函数尖锐极小值与物理系统鲁棒性不足之间的联系。这里有明确的科学内容；它支持把建立有解释作用的联系作为贡献形式。同一论文两名审稿者的着眼点和原创性判断并不相同，不能合成为审稿人一致认可创新。", "core"),
    ("E03", "P005", "T02", "认识推进：澄清冲突", [
        "These important and very novel data go some way to clarify previous and conflicting evidence",
    ], "审稿者1认为数据有助于澄清此前相互冲突的证据。所选原文是句中连续片段，后文具体讨论镰状细胞病中的ADP依赖血小板聚集。其评价对象是认识得到澄清，而不只是多获得一批数据。", "core"),
    ("E04", "P190", "T03", "能力拓展：新功能组合", [
        "Supplementary Table 4 compares reconfigurable devices in the literature and shows that only this work offers all the functions in a single device.",
    ], "审稿者1依据文献比较指出，该工作把所讨论的全部功能集成到单个器件中。这是对新增能力组合的评价，不等于只看某个性能数值是否最高，也不是本分析独立核实了该器件的首次性。", "core"),
    ("E05", "P781", "T04", "资源贡献：概念与覆盖分开判断", [
        "Its significance lies not in conceptual novelty but in the generation of a high-quality, multi-tissue dataset for a non-model species.",
    ], "审稿者2将贡献放在非模式物种的多组织数据资源，而非概念新颖性。这里应识别资源填补了什么覆盖空白、使什么研究成为可能；仅仅公开数据或提供代码属于另一层面的透明性要求。", "core"),
    ("E06", "P827", "T75", "批判性综合：从汇集到比较", [
        "first, to systematically consolidate the fragmented literature on using solid catalysts for alcohol deuteration; and second, to critically contrast the performance (activity, chemoselectivity, and site selectivity) of heterogeneous systems with conventional homogeneous methods.",
    ], "审稿者1将综述的目标表述为整合分散文献并进行批判性比较。它支持将综合作为可能的贡献形式；不能仅凭这一目标描述就认定综述已产生原创理论，仍要说明比较后得到的认识。", "core"),
    ("E07", "P005", "T05", "科学价值：重要问题与未解决之处", [
        "Pulmonary microvascular thrombosis is a well-known complication of sickle cell disease (SCD) and is associated with significant morbidity and mortality but it is not clear why only a sub-set of patients develop pulmonary thrombosis evidenced by autopsy and CT studies.",
    ], "审稿者3同时交代问题的重要性和未解决之处。这是判断科学价值的参照；还需联系论文实际推进了什么，不能由疾病严重直接推出该论文贡献重大。", "core"),
    ("E08", "P085", "T62", "确认价值：预期内也可以有贡献", [
        "even showing expected relationships between gene and protein interaction networks has been quite uncommon in past studies.",
    ], "审稿者2先提出结果可能较为显然的疑问，随后指出此前研究较少展示这些预期内联系。这说明确认预期可以有贡献，意外性不是价值的必要条件；不是说所有重复确认都有相同价值。", "core"),
    ("E09", "P591", "T62", "负结果的认识价值", [
        "Overlooking a key (negative) finding: One of the most striking observations is that the majority of C-terminal extensions do not significantly affect protein stability.",
        "This negative result is not discussed.",
    ], "审稿者3提醒作者讨论被忽略的负结果。它支持评价研究是否改变了对问题的认识，而不把阳性或显著结果设为贡献前提。T62共91篇是整个确认、互补和渐进价值主题的篇数，并非91篇都讨论负结果。", "core"),
    ("E10", "P071", "T74", "基础价值与直接用途分开判断", [
        "For these two reasons, I conclude that, although I consider this work a remarkable piece of physics and a source of inspiration for future developments in the field, the technique still lacks a clear path toward practical applications in precision sensing.",
    ], "同一位审稿者1同时肯定物理研究本身并质疑直接传感应用。科学价值不能仅由近期可用性决定；原文中的未来启发也不等于已测得后续影响。", "core"),
    ("E11", "P932", "T64", "意外性触发核验，不自动构成创新认可", [
        "From a quantum efficiency standpoint, such an enhancement is unexpectedly large.",
        "Therefore, before recommending the publication of this article in the current form, I would like to suggest the authors confirm their observations and provide a more robust explanation for such a significant efficiency enhancement as below:",
    ], "审稿者2对异常大的效率提升首先要求复现与解释。因此，意外性适合作为识别可能改变既有预期的线索；必须先判明它是可靠的新认识还是待排除的问题。222篇涉及该主题不能解释为222篇获认可的意外创新。", "boundary"),
    ("E12", "P010", "T06", "整合不等于并置", [
        "without showing how these processes relate, how scenarios invoking these processes or settings conflict, or how products could be transferred between settings.",
    ], "审稿者2批评不同环境下的化学过程被放在一起却缺少联系、冲突或物质转移的说明。该例界定科学整合需要什么内容，同时包含论证有效性的批评；不能用出现多个领域或图上新增连边代替这些解释。", "boundary"),
    ("E13", "P002", "T03", "能力主题中的可行性问题", [
        "Is it truly feasible to achieve such precise control over angle and pixel-level shifts in a physical experimental setup?",
    ], "该T03条目直接询问物理实验能否实现精确控制，主要是能力声称的可行性和证据问题，不能单独计作认可新增能力的判断。此次保留原编码，明确它对主题频数解释的影响。", "boundary"),
    ("E14", "P007", "T06", "整合主题中的多证据验证", [
        "The combination of repeat multibeam surveys, geologic sampling, water column imagery and morphological analyses provides compelling evidence for the erosion and redistribution of sediment and hard substrate, in areas with heavy vessel traffic.",
    ], "前文称研究提供了新视角，但这句本身赞许多种方法提供有力证据。它未独立说明整合产生了什么新的认识，不能仅因多种技术并用就作为知识整合创新的明确依据。", "boundary"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalise(text: str) -> str:
    return " ".join(text.split())


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def collect_examples() -> list[dict]:
    reports = {x["pilot_id"]: x for x in map(json.loads, (ROOT / "raw_reports.jsonl").read_text().splitlines())}
    evidence = {}
    for line in (ROOT / "theme_evidence.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item.get("included") and item["kind"] == "match":
            key = (item["pilot_id"], item["theme_id"])
            assert key not in evidence, key
            evidence[key] = item
    result = []
    for identifier, paper, theme, title, quotes, interpretation, role in EXAMPLES:
        item = evidence[(paper, theme)]
        report = reports[paper]
        block = next(b for b in report["blocks"] if b["block_id"] == item["source_block_id"])
        source = Path(item["source_path"]).read_text()
        source_span = source[item["source_start_char"]:item["source_end_char"]]
        assert source_span == item["quote"], identifier
        for quote in quotes:
            assert normalise(quote) in normalise(item["quote"]), identifier
            assert normalise(quote) in normalise(block["text"]), identifier
            assert normalise(quote) in normalise(source_span), identifier
        result.append({
            "example_id": identifier, "title_zh": title, "role": role,
            "pilot_id": paper, "paper_id": report["paper_id"],
            "journal": report["journal"], "field": report["field"],
            "displayed_quotes": quotes, "interpretation_zh": interpretation,
            "normalisation": "Whitespace only; excerpts are not silently joined.",
            "source_span_exact_match": True, "excerpts_verified_in_source_and_review_block": True,
            "evidence": item,
        })
    return result


FRAMEWORK = {
    "population": {"eligible_papers": 994, "unit": "每篇论文是否涉及该关注点", "pooled_only": True},
    "status": "基于主题含义和典型原文提出的概念框架；未经因子结构、独立性或评分效度验证。",
    "selection": "典型原文按概念清晰度目的性选取；不代表另一次随机样本或维度支持率估计。",
    "axes": [
        {"id": "D1", "name_zh": "历史原创性", "question": "相对已有知识，究竟什么是新的？", "themes": ["T01"]},
        {"id": "D2", "name_zh": "实质推进", "question": "这项贡献改变了什么认识或研究能力？", "themes": ["T02", "T03", "T04", "T06", "T64", "T75"]},
        {"id": "D3", "name_zh": "科学价值", "question": "这种改变为什么值得重视？", "themes": ["T05", "T62", "T74"]},
    ],
    "contribution_forms": [
        {"name_zh": "认识推进与预期重估", "themes": ["T02", "T64"]},
        {"name_zh": "知识整合与批判性综合", "themes": ["T06", "T75"]},
        {"name_zh": "研究能力拓展", "themes": ["T03"]},
        {"name_zh": "资源与覆盖空白开拓", "themes": ["T04"]},
    ],
    "original_five_mapping": [
        {"angle": "科学内容", "support": "有内容依据，属于贡献识别与描述层", "proposal": "保留，用四种贡献形式具体化；不当作创新程度轴。"},
        {"angle": "历史新颖性", "support": "有直接依据", "proposal": "保留为历史原创性；明确比较对象、时间及具体新增内容。"},
        {"angle": "结构形式与范围", "support": "科学整合、能力与覆盖改变有部分依据；具体图指标未获直接支持", "proposal": "在理论层讨论改变方式及作用范围；图谱扰动作为实质推进的表征方法。"},
        {"angle": "证据基础", "support": "有核验必要性依据，不是创新程度的同层维度", "proposal": "移至支撑三个核心判断的共同核验层。"},
        {"angle": "科学意义", "support": "有直接依据", "proposal": "保留为科学价值，区分问题重要性、实际贡献及未来潜力。"},
    ],
    "frequency_interpretation": "引用已有主题出现率，不估算新维度支持率；褒扬、批评、质疑或补充要求均可能触发编码。",
    "coding_changes": "无；列明边界案例，不修改既有主题矩阵。",
}


INTRO = """# 从真人审稿关注点提炼创新评价维度

## 结论与建议

建议将创新评价组织为三个核心判断：**历史原创性、实质推进、科学价值**，分别回答“新在哪里”“改变了什么”“为什么重要”。在“实质推进”之下，区分认识推进、知识整合、能力拓展、资源开拓四种贡献形式。它们可以共存，不要求每篇论文全部具备。

这比把原来的五个解释角度直接并列为五维评价更清楚：**科学内容是要识别的对象，历史新颖性和科学意义是核心判断，结构形式是解释改变的方式，证据基础是支撑判断的条件。** 原来的五角度规范仍可作为分析工作流程；本报告提出如何为其建立更清楚的理论层次，不修改该规范或论文正文。

三个核心判断是对现有材料的概念提炼，不是审稿人共同使用的显式量表，也不是已由统计证明相互独立的三个因子。下面分别呈现主题频数、直接原文依据和本次推导，避免把三者混成同一种证据。

## 1. 数据口径与相关关注点

统一使用994篇具有实质性真人审稿材料的论文，聚焦“原创性与知识贡献”和“科学价值与贡献意义”两类中的10个具体主题。每篇论文对每个主题最多计一次；主题可以共现，百分比不能相加。材料来自Nature Portfolio六本期刊，不含Nature主刊；这些频率描述本语料，不能直接作为Nature主刊的评价权重。

频数表示审稿文本涉及相关问题，包括肯定、批评、质疑和补充要求；不表示论文获得创新认可，也不表示这么多审稿人认同本报告提出的维度。引用片段按概念清晰度选择，均保留论文、评审者、原文块及来源位置。不同评审者的意见不合并为一致意见。

{statistics}

T74和T75的1.0%是对全部994篇论文的出现率；不能解释为“基础价值不重要”或“只有1.0%的综述需要批判性综合”。这里没有按适用文章类型另行估计分母。频率反映被明确谈及的程度，不等于理论重要性或评价权重。

## 2. 三个核心判断及其理由

### 2.1 历史原创性：相对已有知识，究竟什么是新的？

T01在801篇论文的审稿中出现（80.6%），是最直接的历史比较依据。判断对象应是具体贡献，而不是整篇文章笼统的“新”或“不新”：既有方法可以被用于新的对象，已知现象可以获得新的解释，已有组成部分也可能构成新的功能组合。

应输出：**已有工作已经完成A；本研究新增B；B与A的差异发生在什么对象、解释、用途或组合上。** 判断首次性、独立性和新增程度需要各自对应的历史依据，不能只凭语言新奇或方法名称。

E01与E02来自同一论文、不同评审者：一人纠正技术原创性的过度声称，另一人肯定其建立的科学联系。两者提示我们必须把“哪一部分新”和“哪一部分有价值”具体化；也展示了评审判断并非天然一致。

### 2.2 实质推进：这项贡献改变了什么认识或研究能力？

历史上未见同样表述，并不足以说明贡献的内容。第二项判断要说明研究前后究竟改变了什么：解释了原先不清楚的现象，联结了原先分散的发现，实现了此前不能实现的操作，还是填补了重要的资源空白。依据来自T02、T03、T04、T06、T64和T75，不能把这些频数简单相加。

| 贡献形式 | 核心问题 | 相关主题及篇数 | 需要说明的实质改变 |
|---|---|---|---|
| 认识推进与预期重估 | 多理解了什么，原有解释或预期受到什么修正？ | T02：510；T64：222 | 澄清冲突、解释现象、修正概念或预期；单纯令人惊讶并不充分 |
| 知识整合与批判性综合 | 原先分散的发现如何构成新的整体认识？ | T06：372；T75：10 | 联结后增加了何种解释或比较结论；并置文献、联合验证不自动构成整合创新 |
| 研究能力拓展 | 以前难以观察、操控或实现的事情，现在能做什么？ | T03：396 | 新的功能、可操作性或能力组合；不是只报告较高性能 |
| 资源与覆盖空白开拓 | 哪些对象、数据或研究问题现在可以被系统研究？ | T04：81 | 新的覆盖及研究条件；不只是资源可下载或数据量更大 |

这四项是贡献形式，不能要求一篇资源论文同时满足概念突破、方法突破和机制突破。它们也并非完全互斥：一个新资源可能带来新认识，一次整合可能同时形成新能力。

“实质推进”中的程度，需要通过具体的研究前后差异论证；本次分析没有从关注点出现率构造高低分，也没有证明某一贡献形式天然优于另一种形式。

### 2.3 科学价值：这种改变为什么值得重视？

T05在559篇（56.2%）、T62在91篇（9.2%）、T74在10篇（1.0%）中出现，分别提供问题重要性、确认或渐进成果价值、基础理解与直接应用价值之区分的依据。

应建立的论证链是：**有一个重要或未解决的问题 → 本研究使这个问题得到何种推进 → 这种推进为什么值得重视。** 前两步不能省略：问题重要不意味着本篇贡献重大；结果可用于某处也不等于它已产生广泛影响。

确认预期、负结果、资源建设和基础认识都可能有科学价值。由此不宜把“颠覆程度”“出乎意料程度”或“即时应用前景”设为所有创新的共同门槛。E08—E10分别展示这三种容易被单一标准漏掉的价值。

三个核心判断可分别形成文字结论：**原创性在哪里，实际改变是什么，改变为什么重要。** 它们不是三个数字相加得出一个创新总分。

## 3. 典型审稿原文及解释

下列英文引文只统一空白与换行；未改写措辞。短于原句的引文在解释中明确说明。完整来源片段和位置见配套JSON。

{examples}

## 4. 与原来五个解释角度逐项对照

| 原角度 | 原文支持什么 | 需要调整什么 | 建议位置 |
|---|---|---|---|
| 科学内容 | 评审确实区分解释、能力、资源和综合等贡献内容（T02、T03、T04、T06、T75） | “科学内容”本身是在描述对象，不能与新颖程度或意义大小同层打分；应明确贡献单位及其形式 | 保留为贡献识别层，用四种贡献形式具体化 |
| 历史新颖性 | 前作比较、归属和实质新增有直接依据（T01，801篇） | 从笼统的首次性扩展为具体贡献与前作的区别；分开既有组成部分和新增组合 | 保留为核心判断“历史原创性” |
| 结构形式与范围 | 新联系、综合见解、功能组合及知识覆盖的改变有内容依据（T06、T75、T03、T04） | 这些原文没有直接提出社区数、中心性、连边数或扰动幅度作为创新标准；科学内容上的联系不等于图上语义邻近 | 高层写“改变方式及作用范围”，放入实质推进；知识图谱扰动作为本文的表征方法 |
| 证据基础 | P932等意见明确要求核验意外结果；能力与整合意见也可能追问贡献声称是否成立 | 这是判断成立的条件，不是与原创性并列的创新程度。研究证据充分性也不等于已测得评价者可靠性 | 保留为贯穿三个核心判断的核验层 |
| 科学意义 | 重要问题、确认价值及基础价值有直接依据（T05、T62、T74） | 分开问题本身的重要性、当前贡献的实际价值和未来可能影响；不把应用性设为唯一价值 | 保留为核心判断“科学价值” |

因此，五个角度不需要全部推翻，但需要重新分层。特别是“结构形式与范围”不能直接宣布蒸馏自真人评审：**真人材料支持关注知识如何被联结、理解如何被推进；用知识图谱扰动刻画这些改变，是本文提出的方法性主张。** 这是两种不同来源的依据。

原规范中的“结构形式与范围”特指知识图谱中的局部邻域、社区、角色及联合插入。本报告没有把它偷换成实验外部有效性或应用范围；建议调整的是理论叙述层次，原有图谱事实仍按其本来含义解释。

## 5. 对知识图谱扰动主线的启发

可以将概念上的工作定义组织为：**科学创新是相对于既有知识形成的、使科学理解或研究能力发生实质改变的贡献；其评价需要分别说明这种改变的原创性、推进内容和科学价值。** 这是根据材料提出的综合定义，不是所有审稿人的共同原话。

知识图谱扰动由此有明确位置：它帮助定位和呈现候选贡献对既有知识组织的改变。例如，新联系可以提示整合，新覆盖可以提示资源空白的填补；再通过原文与历史比较解释这些变化意味着什么。

但图谱表示与科学贡献并非一一对应。语义连边本身不证明真实的科学联系，社区改变不自动等于学科突破，局部扰动幅度也不是已测得的长期影响。P781的资源贡献、P085的预期内确认及P591的负结果进一步说明，单一“扰动越大、创新越高”的规则容易漏掉有价值的贡献。更合适的研究问题是：**不同扰动形式何时对应不同类型的实质推进，它们的价值由什么科学内容决定？**

据此，论文中的逻辑可以更清楚地分成：识别贡献对象 → 比较历史原创性 → 解释改变了什么 → 判断为什么重要。图谱分析辅助识别和解释；证据核验支撑这些判断。针对评价稳定性、一致性或证据响应的实验，则检验评价是否可靠，不直接增加被评价论文的创新程度。

## 6. 本次回查发现的编码边界

此次回查发现具体边界案例，而不只是抽象地担心分类误差：

- E13（P002/T03）直接询问精确控制能否实现，主要属于能力声称的可行性核验。
- E14（P007/T06）强调多种方法提供有力证据，该句主要体现证据汇合，不能仅凭多方法并用就确认整合创新。
- E11（P932/T64）因结果过于意外而要求核验，不是直接赞许意外创新。

所以，本报告保留已有10主题频数，并将其明确标为“相关关注点出现率”；不重新发布三个核心判断的认可率、四种创新形式的发生率或五角度的支持率。理论推导主要依赖含义清楚、来源可查的实例及主题内容，不依赖把每个主题命中都当成纯粹的创新评价。

这也提示，若后续要将该框架写成经过验证的评价维度，需要在每条候选意见中另行区分：描述贡献、比较原创性、评价推进或价值、核验贡献证据。当前不新增编码、不重跑实验，也不修改已有统计矩阵。

## 7. 可以据此怎样表述框架来源

适合现阶段证据的表述是：

> 对994篇论文的真人审稿材料进行主题归纳后，我们从历史比较、认识与能力改变以及贡献价值等关注点中提炼出三个核心评价问题，并区分贡献形式、判断依据与评价可靠性。

不适合现阶段证据的表述是：

> 五个独立评价维度已由994篇真人评审验证，且图谱扰动强度能够直接衡量创新程度。

**对原框架的具体建议：保留五角度作为分析与核验流程，将其理论依据改写为“贡献识别—三个核心判断—共同核验条件”；在实质推进之下说明四种贡献形式。** 这样既保留已有分析的用途，也让每个部分为什么存在、评价什么、依据是什么更清楚。

## 数据与核验文件

- [10个相关关注点的统一统计](innovation_dimension_theme_statistics.csv)
- [概念框架与原五角度映射](innovation_dimension_framework.json)
- [典型原文、完整来源片段和解释](innovation_dimension_examples.json)
- [计数与引文核验](innovation_dimension_analysis_validation.json)
- [可重复生成脚本](analyse_innovation_dimensions.py)
- [原五个解释角度规范]({spec_path})

本次仅生成独立分析材料；论文正文、原五角度规范、图件、实验结果及原始编码均未修改。
"""


def main() -> None:
    spec_hash = hashlib.sha256(SPEC.read_bytes()).hexdigest()
    matrix = read_csv(ROOT / "paper_theme_matrix.csv")
    stats = {r["theme_id"]: r for r in read_csv(ROOT / "theme_statistics.csv")}
    assert len(matrix) == len({r["pilot_id"] for r in matrix}) == 994
    categories = json.loads((ROOT / "theoretical_codebook.json").read_text())["categories"]
    core_ids = {t for c in categories if c["category_id"] in {"C1", "C2"} for t in c["theme_ids"]}
    assert core_ids == set(THEMES)
    axis_themes = [t for axis in FRAMEWORK["axes"] for t in axis["themes"]]
    assert len(axis_themes) == len(set(axis_themes)) == 10 and set(axis_themes) == core_ids
    rows = []
    for theme in THEMES:
        count = sum(int(row[theme]) for row in matrix)
        assert count == int(stats[theme]["present_papers"])
        percentage = 100 * count / len(matrix)
        assert abs(percentage - float(stats[theme]["percentage"])) < 1e-9
        rows.append({"theme_id": theme, "name_zh": stats[theme]["name_zh"], "eligible_papers": 994,
                     "present_papers": count, "percentage": percentage,
                     "interpretation": "相关关注点出现率；不是维度认可率或创新得分"})
    examples = collect_examples()
    write_json(ROOT / "innovation_dimension_examples.json", examples)
    write_json(ROOT / "innovation_dimension_framework.json", FRAMEWORK)
    with (ROOT / "innovation_dimension_theme_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "| 编码 | 相关关注点 | 篇数 / 994 | 出现率 |\n|---|---|---:|---:|\n"
    table += "\n".join(f"| {r['theme_id']} | {r['name_zh']} | {r['present_papers']} | {r['percentage']:.1f}% |" for r in rows)
    sections = []
    for item in examples:
        evidence = item["evidence"]
        header = f"### {item['example_id']} · {item['title_zh']}\n\n"
        identity = f"{item['pilot_id']} · {item['paper_id']} · 评审者{evidence['reviewer_id']} · {evidence['theme_id']}\n\n"
        quotes = "\n\n".join("> " + q for q in item["displayed_quotes"])
        source = f"[审稿原文]({evidence['source_path']})；原文块 `{evidence['source_block_id']}`。"
        sections.append(header + identity + quotes + "\n\n" + item["interpretation_zh"] + "\n\n" + source)
    report = INTRO.replace("{statistics}", table).replace("{examples}", "\n\n".join(sections)).replace("{spec_path}", str(SPEC))
    (ROOT / "innovation_dimensions_zh.md").write_text(report)
    assert hashlib.sha256(SPEC.read_bytes()).hexdigest() == spec_hash
    validation = {
        "status": "passed", "eligible_papers": len(matrix), "pooled_only": True,
        "focused_themes": len(rows), "theme_counts_match_paper_matrix": True,
        "c1_c2_themes_fully_mapped": True, "axis_mapping_has_no_omissions_or_duplicate_assignments": True,
        "examples": len(examples), "source_spans_exactly_verified": len(examples),
        "displayed_excerpt_count": sum(len(e["displayed_quotes"]) for e in examples),
        "displayed_excerpts_verified_in_source_and_review_block": True,
        "quote_normalisation": "whitespace_only", "spec_sha256_unchanged": spec_hash,
        "original_coding_changed": False, "new_model_calls": 0,
        "interpretive_limits": "Topic counts are not estimates of dimension endorsement. Source checking does not validate the proposed dimensional structure.",
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                         ["paper_theme_matrix.csv", "theme_statistics.csv", "theme_evidence.jsonl", "raw_reports.jsonl", "theoretical_codebook.json"]},
    }
    write_json(ROOT / "innovation_dimension_analysis_validation.json", validation)
    print(json.dumps({"status": "passed", "themes": len(rows), "examples": len(examples), "report": str(ROOT / "innovation_dimensions_zh.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
