prompts_keywords_extraction = """
Extract up to 5 most relevant and core keywords from the following academic abstract. 

Instructions:
- Identify the key concepts, methods, or topics that are central to the research
- Prioritize domain-specific terminology and significant research themes
- Select keywords that best represent the abstract's main focus
- List between 1-5 keywords, depending on the abstract's content
- Present keywords in order of relevance (most important first)
- Use single words or short phrases (2-3 words maximum per keyword)
- Avoid generic words like "challenges," "research," or "analysis"

Abstract:{abstract}

Output format:
keyword1,keyword2,keyword3,keyword4,keyword5
"""

example_passages_summarization = """
[0] Title: CoQA: A Conversational Question Answering Challenge Abstract: Humans gather information by engaging in conversations involving a series of interconnected questions and answers. For machines to assist in information gathering, it is therefore essential to enable them to answer conversational questions. We introduce CoQA, a novel dataset for building Conversational Question Answering systems. Our dataset contains 127k questions with answers, obtained from 8k conversations about text passages from seven diverse domains. The questions are conversational, and the answers are free-form text with their corresponding evidence highlighted in the passage. We analyze CoQA in depth and show that conversational questions have challenging phenomena not present in existing reading comprehension datasets, e.g., coreference and pragmatic reasoning. We evaluate strong conversational and reading comprehension models on CoQA. The best system obtains an F1 score of 65.4%, which is 23.4 points behind human performance (88.8%), indicating there is ample room for improvement. \n
[1] Title: SQuAD: 100,000+ Questions for Machine Comprehension of Text Abstract: We present the Stanford Question Answering Dataset (SQuAD), a new reading comprehension dataset consisting of 100,000+ questions posed by crowdworkers on a set of Wikipedia articles, where the answer to each question is a segment of text from the corresponding reading passage. We analyze the dataset to understand the types of reasoning required to answer the questions, leaning heavily on dependency and constituency trees. We build a strong logistic regression model, which achieves an F1 score of 51.0%, a significant improvement over a simple baseline (20%). However, human performance (86.8%) is much higher, indicating that the dataset presents a good challenge problem for future research.\n
[2] Title: Interpretation of natural language rules in conversational machine reading Abstract: Most work in machine reading focuses on question answering problems where the answer is directly expressed in the text to read. However, many real-world question answering problems require the reading of text not because it contains the literal answer, but because it contains a recipe to derive an answer together with the reader's background knowledge. One example is the task of interpreting regulations to answer "Can I...?" or "Do I have to...?" questions such as "I am working in Canada. Do I have to carry on paying UK National Insurance?" after reading a UK government website about this topic. This task requires both the interpretation of rules and the application of background knowledge. It is further complicated due to the fact that, in practice, most questions are underspecified, and a human assistant will regularly have to ask clarification questions such as How long have you been working abroad? when the answer cannot be directly derived from the question and text. In this paper, we formalise this task and develop a crowd-sourcing strategy to collect 32k task instances based on real-world rules and crowd-generated questions and scenarios. We analyse the challenges of this task and assess its difficulty by evaluating the performance of rule-based and machine-learning baselines. We observe promising results when no background knowledge is necessary, and substantial room for improvement whenever background knowledge is needed.\n
[3] Title: Passage Re-ranking with BERT Abstract: Recently, neural models pretrained on a language modeling task, such as ELMo (Peters et al., 2017), OpenAI GPT (Radford et al., 2018), and BERT (Devlin et al., 2018), have achieved impressive results on various natural language processing tasks such as question-answering and natural language inference. In this paper, we describe a simple re-implementation of BERT for query-based passage re-ranking. Our system is the state of the art on the TREC-CAR dataset and the top entry in the leaderboard of the MS MARCO passage retrieval task, outperforming the previous state of the art by 27% (relative) in MRR@10. \n
[4] Title: Bidirectional Attention Flow for Machine Comprehension Abstract: Machine comprehension (MC), answering a query about a given context paragraph, requires modeling complex interactions between the context and the query. Recently, attention mechanisms have been successfully extended to MC. Typically these methods use attention to focus on a small portion of the context and summarize it with a fixed-size vector, couple attentions temporally, and/or often form a uni-directional attention. In this paper we introduce the Bi-Directional Attention Flow (BIDAF) network, a multi-stage hierarchical process that represents the context at different levels of granularity and uses bi-directional attention flow mechanism to obtain a query-aware context representation without early summarization. Our experimental evaluations show that our model achieves the state-of-the-art results in Stanford Question Answering Dataset (SQuAD) and CNN/DailyMail cloze test.\n
"""

example_question_summarization = "We present QuAC, a dataset for Question Answering in Context that contains 14K information-seeking QA dialogs (100K questions in total). The dialogs involve two crowd workers: (1) a student who poses a sequence of freeform questions to learn as much as possible about a hidden Wikipedia text, and (2) a teacher who answers the questions by providing short excerpts from the text. QuAC introduces challenges not found in existing machine comprehension datasets: its questions are often more open-ended, unanswerable, or only meaningful within the dialog context, as we show in a detailed qualitative evaluation. We also report results for a number of reference models, including a recently state-of-the-art reading comprehension architecture extended to model dialog context. Our best model underperforms humans by 20 F1, suggesting that there is significant room for future work on this data. "

example_answer_summarization = """
This work builds on span based reading comprehension [1] while also incorporating innovations such as curating questions independently of supporting text to reduce trivial lexical overlap. 
Concurrent to this work, [2] proposed a task of generating and answering yes/no questions for rule focused text (such as traffic laws) by interacting with a user through dialog. 
Also concurrently, [0] propose conversational question answering (CoQA) from text but allow both students and questioners to see the evidence. 
"""

promts_w_references_summarization = ("Given an abstract of an academic paper, the innovation evaluation report of this paper, and a set of passsages from relevant papers, generate a related work section summarizing relevant related work."
                                    "Not all of the passages are relevant, so please carefully read the passages and only use passages that are related."  
                                    "All of citation-worthy statements need to be supported by one of the references we provide as 'References' and appropriate citation numbers should be added at the last of the sentences."
                       "References should be formatted as [0], [1], [2], ..., [n]."
                       "Your answer should be marked as [Response_Start] and [Response_End]."
                       "Here's an example:\n##\n"
                       "References: \n{example_passages}"
                       "\nAbstract: {example_question}"
                       "\nInnovation: {example_innovation}"
                       "\n[Response_Start]{example_answer}[Response_End]\nNow, please generate another related work given the following abstract.\n##\n")
generation_demonstration_summarization = promts_w_references_summarization.format_map({"example_passages": example_passages_summarization, "example_innovation":"", "example_question": example_question_summarization, "example_answer": example_answer_summarization})
generation_instance_prompts_summarization = generation_demonstration_summarization + "References:\n {reference}\n Abstract: {abstract}\n Innovation: {innovation}\n"

# ============================================================================
# LATS 提示词 - 学术创新性评价树搜索
# ============================================================================

# 创新性评价生成提示词
INNOVATION_GENERATION_PROMPT = """你是一位专业的学术论文评审专家。你的任务是对给定的学术论文进行创新性评价。

【待评价论文信息】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究工作】
{related_papers}

{graph_metric_evidence}

{committee_evidence}

【评价要求】
1. **创新性识别**：识别论文的主要创新点（新方法、新发现、新理论、新应用等）
2. **对比分析**：将论文的创新点与相关研究工作进行详细对比
   - 指出现有研究的局限性
   - 说明本论文如何克服这些局限
   - 评估改进的程度和意义
3. **引用规范**：在评价中正确引用相关研究工作，使用 [序号] 格式
4. **评价维度**：
   - 理论创新性（是否提出新理论或新概念）
   - 方法创新性（是否提出新方法或改进现有方法）
   - 应用创新性（是否有新的应用场景或实际价值）
   - 与现有工作的差异性（与最相关工作的区别）
   - 图谱结构证据（七维指标是否支持“跨社区桥接、边界扰动、非典型组合、潜在扩散”等创新声明）

【图谱证据使用原则】
- 图谱指标是约束创新性表述的证据，不是自动判定论文好坏的标签。
- 如果 DeltaQ0 / Uzzi 较高，可以讨论边界扰动、非典型组合或范式重组，但必须结合摘要和相关文献。
- 如果 RS / PDE 较高，可以讨论跨学科广度或潜在扩散范围。
- 如果 B / RTD / BurtIP 较高，可以讨论结构洞、跨社区桥接或低冗余知识连接。
- 如果图谱证据置信度低，必须使用保守措辞，并说明不确定性。

请生成一份详细的创新性评价报告，包含：
1. 主要创新点总结（2-3条）
2. 与相关研究的对比分析
3. 相关文献引用
4. 图谱结构证据如何支持或限制上述创新性判断
5. 按审稿委员会 claim cards 逐条说明：文本证据、相关工作差异、图谱支持、反方质疑和不确定性

评价应当客观、专业、有依据。"""

# 反思提示词 - 针对创新性评价质量
INNOVATION_REFLECTION_PROMPT = """你是一位严格的学术论文评审专家。请对以下创新性评价进行批判性反思。

【待评价论文】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究文献】
{related_papers}

{graph_metric_evidence}

{committee_evidence}

【当前创新性评价】
{current_evaluation}

【反思要求】
请从以下维度评价当前创新性评价的质量：

1. **创新性识别准确性**（0-10分）【权重最高】
   - 识别的创新点是否真实存在？
   - 是否有遗漏的重要创新？
   - 是否存在夸大或错误的创新声明？

2. **对比充分性**（0-10分）
   - 是否充分对比了相关研究工作？
   - 是否正确识别了与现有工作的差异？

3. **引用规范性**（0-10分）
   - 引用格式是否正确 [序号]？
   - 引用是否与论述匹配？
   - 是否有遗漏的重要引用？
   - 是否存在不必要的引用？

4. **图谱结构证据对齐**（0-10分）
   - 当前评价是否正确使用七维图谱指标支持创新性判断？
   - 是否在 DeltaQ0/Uzzi 高时讨论边界扰动或非典型组合？
   - 是否在 RS/PDE 高时讨论知识广度或潜在扩散？
   - 是否在 B/RTD/BurtIP 高时讨论跨社区桥接或结构洞？
   - 是否避免在图谱证据弱时夸大“颠覆性”“开创性”等表述？

5. **不确定性校准**（0-10分）
   - 当图谱证据置信度低、相关论文少或领域标签不足时，是否明确降调？
   - 是否把结构指标表述为证据而不是结论？

6. **表达清晰度**（0-10分）
   - 评价是否结构清晰、术语准确、便于学术审稿使用？

7. **审稿委员会一致性检查**
   - 是否逐条回应了 committee claim cards？
   - 是否呈现了反方质疑和不确定性？
   - 是否遵守 recommended tone？

请输出以下格式的反思结果：
<reflections>
你的详细反思内容，指出评价的优点和不足，提出改进建议...
</reflections>
创新性识别准确性: 0-10
对比充分性: 0-10
引用规范性: 0-10
图谱结构证据对齐: 0-10
不确定性校准: 0-10
表达清晰度: 0-10
<score>综合得分(0-10的整数)</score>
<found_solution>是否达到高质量创新性评价标准(true/false)</found_solution>

注意：只有当创新性评价正确识别了创新点、充分对比了相关研究、引用规范完整、图谱证据使用恰当、并且不确定性表达校准时，found_solution 才为 true。综合得分 = 创新性识别准确性×0.30 + 对比充分性×0.20 + 引用规范性×0.15 + 图谱结构证据对齐×0.20 + 不确定性校准×0.10 + 表达清晰度×0.05"""

# 优化改进提示词
INNOVATION_IMPROVEMENT_PROMPT = """你是一位专业的学术论文评审专家。基于以下反思反馈，改进创新性评价。

【待评价论文】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究工作】
{related_papers}

{graph_metric_evidence}

{committee_evidence}

【当前评价】
{current_evaluation}

【反思反馈】
{reflection_feedback}

【改进要求】
请根据反思反馈，生成改进后的创新性评价。改进应当：
1. 补充遗漏的对比分析
2. 修正不准确的创新声明
3. 完善引用和论证
4. 提升评价的全面性和深度
5. 去除不必要的内容
6. 让创新性声明与七维图谱证据一致：证据强时指出对应机制，证据弱时保守表达并说明不确定性
7. 按审稿委员会 claim cards 修正每条 claim 的证据、图谱支持、反方质疑和不确定性

保持评价的专业性和客观性，确保引用格式正确 [序号]。"""

# 最终评价生成提示词（带完整引用格式）
FINAL_INNOVATION_REPORT_PROMPT = """你是一位资深的学术论文评审专家。请基于以下信息生成最终的创新性评价报告。

【待评价论文】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究工作（带完整信息）】
{related_papers_with_metadata}

{graph_metric_evidence}

{committee_evidence}

【初步创新性评价】
{draft_evaluation}

【最终报告要求】
请生成一份完整、专业的创新性评价报告，包含以下部分：

## 1. 创新点概述 (Innovation Summary)
简明扼要地总结论文的主要创新贡献（2-4条）。

## 2. Claim-level Committee Review
按审稿委员会 claim cards 逐条评价每个创新声明。每条必须包含：
- claim 内容
- 相关工作证据和差异
- 图谱结构支持或限制
- 反方质疑 / counterarguments
- 不确定性等级

## 3. 与现有研究的综合对比 (Comparison with Related Work)
综合对比论文与相关研究工作的差异：
- 现有方法/理论的局限性
- 本论文的改进之处
- 改进的意义和价值

## 4. 图谱结构证据 (Graph-based Evidence)
说明七维图谱指标如何支持或限制上述创新性判断。若证据置信度低，必须明确说明原因，并避免过度声称颠覆性。

## 5. 参考文献 (References)
使用标准学术引用格式列出所有引用的文献：
[1] 作者. 标题. 期刊/会议名, 年份.
[2] 作者. 标题. 期刊/会议名, 年份.
...

注意：
- 引用格式必须规范完整
- 只列出实际引用到的文献
- 确保引用序号与正文中的 [序号] 对应"""
