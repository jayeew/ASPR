# ASPR × 2026 Agent 前沿论文专项资料包

更新时间：2026-08-10

这个资料包不是泛泛收集“用了 Agent 的论文”，而是围绕 ASPR 的最终任务筛选：

> 输入一篇待审论文，自动生成一份完整、专业、可追溯到论文内部内容、先行研究和图谱结构证据的同行评审报告。

## 包内内容

- `papers/`：19 篇 2026 年正式会议论文/录用论文的公开全文。
- `01_19篇论文逐篇直白解析.md`：每篇论文都用直白中文解释，并解释关键英文词。
- `02_ASPR_2026重构方案.md`：允许推翻现有多 Agent 结构后重新设计的 ASPR 方案。
- `03_论文索引.csv`：快速筛选论文、会议、机制、阅读优先级和 ASPR 对应模块。
- `04_来源与版本说明.md`：每篇 PDF 的公开来源和版本说明。
- `05_实现蓝图_状态与动作.md`：更接近代码实现的 Review State、Controller Action、Evidence schema 和执行流程。

## 最建议的阅读顺序

### 第一组：必须先看——直接决定 ASPR 应该变成什么

1. ReviewGrounder — ACL 2026
2. Beyond “Not Novel Enough” — EACL 2026
3. Eigen-Agent — ICLR 2026
4. IterResearch — ICLR 2026
5. SciNet — ICML 2026
6. Counterfactual RAG — ICLR 2026
7. Stop Wasting Your Tokens / SupervisorAgent — ICLR 2026
8. Agentic Confidence Calibration — ICML 2026
9. When Planning Fails Despite Correct Execution — ICML 2026

### 第二组：值得借机制，不建议整套照搬

10. FlowSearcher
11. MEM1
12. Graph-of-Agents
13. Agent Primitives
14. ResearchRubrics
15. Failure is Feedback

### 第三组：用于扩展/离线优化，不建议作为第一版主路径

16. CARD
17. GraphPlanner
18. MASS
19. HieraMAS

## 我对 ASPR 的最终方向判断

不要继续做“Claim Decomposer + Evidence Mapper + Graph Analyst + Skeptic + Meta Reviewer 全部固定上线”的委员会。

更建议把 ASPR 改造成：

**一个维护统一 Review State（评审状态）的 Evidence-State Review System。**

核心由：

`Paper Compiler → Review State → Evidence Controller → Evidence Tools/Primitives → Review Compiler → Verifier`

组成。

真正“Agentic”的地方不在角色数量，而在 Controller 能根据当前证据状态自主决定：

- 是否继续检索；
- 应该查什么；
- 是否需要反事实先行研究搜索；
- 是否需要调用统计/方法/引用专家；
- 当前创新判断是否稳定；
- 当前证据是否已经足够停止。

详细设计见 `02_ASPR_2026重构方案.md`。
