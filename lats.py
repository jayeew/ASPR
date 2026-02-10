from __future__ import annotations
import getpass
import os
import json
import io
import math
import re
from collections import deque
from typing import Optional, Literal, List, Dict, Any
import backoff
from contextlib import redirect_stdout
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from collections import defaultdict


# ============================================================================
# 提示词定义 - 专门针对学术创新性评价
# ============================================================================

# 创新性评价生成提示词
INNOVATION_GENERATION_PROMPT = """你是一位专业的学术论文评审专家。你的任务是对给定的学术论文进行创新性评价。

【待评价论文信息】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究工作】
{related_papers}

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

请生成一份详细的创新性评价报告，包含：
1. 主要创新点总结（2-3条）
2. 与相关研究的对比分析
3. 创新性的综合评估
4. 相关文献引用

评价应当客观、专业、有依据。"""

# 反思提示词 - 针对创新性评价质量
INNOVATION_REFLECTION_PROMPT = """你是一位严格的学术论文评审专家。请对以下创新性评价进行批判性反思。

【待评价论文】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究文献】
{related_papers}

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

请输出以下格式的反思结果：
<reflections>
你的详细反思内容，指出评价的优点和不足，提出改进建议...
</reflections>
<score>综合得分(0-10的整数)</score>
<found_solution>是否达到高质量创新性评价标准(true/false)</found_solution>

注意：只有当创新性评价正确识别了创新点、充分对比了相关研究、引用规范完整时，found_solution 才为 true。综合得分 = (创新性识别准确性×0.5 + 对比充分性×0.3 + 引用规范性×0.2)"""

# 优化改进提示词
INNOVATION_IMPROVEMENT_PROMPT = """你是一位专业的学术论文评审专家。基于以下反思反馈，改进创新性评价。

【待评价论文】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究工作】
{related_papers}

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

保持评价的专业性和客观性，确保引用格式正确 [序号]。"""

# 最终评价生成提示词（带完整引用格式）
FINAL_INNOVATION_REPORT_PROMPT = """你是一位资深的学术论文评审专家。请基于以下信息生成最终的创新性评价报告。

【待评价论文】
标题: {paper_title}
摘要: {paper_abstract}

【相关研究工作（带完整信息）】
{related_papers_with_metadata}

【初步创新性评价】
{draft_evaluation}

【最终报告要求】
请生成一份完整、专业的创新性评价报告，包含以下部分：

## 1. 创新点概述 (Innovation Summary)
简明扼要地总结论文的主要创新贡献（2-4条）。

## 2. 与现有研究的对比 (Comparison with Related Work)
详细对比论文与相关研究工作的差异：
- 现有方法/理论的局限性
- 本论文的改进之处
- 改进的意义和价值

## 3. 创新性评估 (Innovation Assessment)
从以下维度进行评估（每项使用 ★ 评分，最高5★）：
- 理论创新性：★★★★★
- 方法创新性：★★★★★
- 应用创新性：★★★★★
- 整体创新水平：★★★★★

## 4. 参考文献 (References)
使用标准学术引用格式列出所有引用的文献：
[1] 作者. 标题. 期刊/会议名, 年份.
[2] 作者. 标题. 期刊/会议名, 年份.
...

注意：
- 引用格式必须规范完整
- 只列出实际引用到的文献
- 确保引用序号与正文中的 [序号] 对应"""


# ============================================================================
# LLM 配置
# ============================================================================

# 配置 ChatOpenAI 连接本地 Ollama
llm = ChatOpenAI(
    model="qwen3-coder:30b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.3,
    # max_tokens=9000,
)


# ============================================================================
# 数据模型
# ============================================================================

class PaperInfo(BaseModel):
    """相关论文信息"""
    index: int = Field(description="文献序号")
    title: str = Field(description="论文标题")
    authors: str = Field(description="作者")
    venue: str = Field(description="发表期刊/会议")
    year: int = Field(description="发表年份")
    abstract: str = Field(description="论文摘要")
    citation_count: Optional[int] = Field(default=None, description="被引次数")
    
    def to_citation_string(self) -> str:
        """生成引用字符串"""
        return f"[{self.index}] {self.authors}. {self.title}. {self.venue}, {self.year}."
    
    def to_context_string(self) -> str:
        """生成上下文字符串（用于输入模型）"""
        return f"[{self.index}] Title: {self.title}\nAuthors: {self.authors}\nVenue: {self.venue}, {self.year}\nAbstract: {self.abstract}\n"


class InnovationEvaluation(BaseModel):
    """创新性评价结果"""
    innovation_points: List[str] = Field(description="识别的创新点列表")
    comparison_analysis: str = Field(description="与相关研究的对比分析")
    assessment_summary: str = Field(description="综合评估摘要")
    cited_papers: List[int] = Field(description="引用的文献序号列表")
    full_evaluation: str = Field(description="完整的评价文本")


class Reflection(BaseModel):
    """反思结果"""
    reflections: str = Field(
        default="",
        description="对创新性评价的批判性反思，包括优点、不足和改进建议"
    )
    score: int = Field(
        default=0,
        description="创新性评价质量得分 (1-10分)",
        ge=0,
        le=10,
    )
    found_solution: bool = Field(
        default=False,
        description="是否达到高质量创新性评价标准 (True/False)。只有当评价充分对比了相关研究、正确识别创新点、论证逻辑清晰、引用规范时为 True"
    )
    comparison_adequacy: int = Field(default=0, description="对比充分性评分 (0-10)", ge=0, le=10)
    innovation_accuracy: int = Field(default=0, description="创新性识别准确性评分 (0-10)", ge=0, le=10)
    citation_normative: int = Field(default=0, description="引用规范性评分 (0-10)", ge=0, le=10)

    @field_validator('reflections', mode='before')
    @classmethod
    def extract_str_(cls, v):
        return str(v)

    @field_validator('score', mode='before')
    @classmethod
    def extract_int_(cls, v):
        return int(v)

    @field_validator('found_solution', mode='before')
    @classmethod
    def extract_bool_from_dict(cls, v):
        if isinstance(v, bool):
            return v
        elif isinstance(v, dict):
            potential_bool = v.get('value')
            if isinstance(potential_bool, bool):
                return potential_bool
            if 'type' in v:
                type_value = v['type']
                if type_value == 1:
                    return True
                elif type_value == 0:
                    return False
            result_str = v.get('result', '').lower()
            if result_str == 'true':
                return True
            elif result_str == 'false':
                return False
        return False

    def as_message(self):
        return HumanMessage(
            content=f"反思: {self.reflections}\n得分: {self.score}/10"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


# ============================================================================
# 树节点类
# ============================================================================

class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        evaluation: Optional[str] = None,
        parent: Optional[Node] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.evaluation = evaluation  # 存储创新性评价内容
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)
        lats_logging(f"创建节点: {self}")

    def __repr__(self) -> str:
        return f"<{self.value:.2f},{self.visits},{self._is_solved}>"

    @property
    def is_solved(self):
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child(self):
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self):
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return float('inf')
        average_reward = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections and self.reflection:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        return messages[::-1]

    def _get_all_children(self):
        all_nodes = []
        nodes = deque([self])
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            nodes.extend(node.children)
        return all_nodes

    def get_best_solution(self):
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


# ============================================================================
# 状态定义
# ============================================================================

class TreeState(TypedDict):
    """树搜索状态"""
    root: Node
    paper_title: str
    paper_abstract: str
    related_papers: List[PaperInfo]  # 相关论文列表
    best_evaluation: Optional[str]   # 最佳创新性评价


# ============================================================================
# 工具函数
# ============================================================================

def safe_extract_content(content_value: Any, fallback: str = "") -> str:
    """安全地从各种类型中提取文本内容
    
    Args:
        content_value: 可能是字符串、列表或其他类型的内容
        fallback: 如果无法提取内容时返回的默认值
        
    Returns:
        提取的字符串内容
    """
    if content_value is None:
        return fallback
    
    if isinstance(content_value, str):
        result = content_value
    elif isinstance(content_value, list):
        result = " ".join(str(item) for item in content_value)
    elif hasattr(content_value, 'text'):
        result = str(content_value.text)
    else:
        result = str(content_value)
    
    # 检查是否为无效内容
    if not result or result.strip() == "" or result == "None":
        return fallback
    
    return result.strip()


def extract_with_regex(text: str) -> Dict[str, Any]:
    """从文本中提取反思结果"""
    # 提取 reflections
    reflections_match = re.search(r'<reflections>(.*?)</reflections>', text, re.DOTALL)
    reflections = reflections_match.group(1).strip() if reflections_match else ""
    
    # 提取 score
    score_match = re.search(r'<score>(\d+)</score>', text)
    score = int(score_match.group(1)) if score_match else 0
    
    # 提取 found_solution
    fs_match = re.search(r'<found_solution>(true|false)</found_solution>', text, re.IGNORECASE)
    found_solution = fs_match.group(1).lower() == 'true' if fs_match else False
    
    # 提取各维度评分
    acc_match = re.search(r'创新性识别准确性[:：]\s*(\d+)', text)
    innovation_accuracy = int(acc_match.group(1)) if acc_match else 5
    
    comp_match = re.search(r'对比充分性[:：]\s*(\d+)', text)
    comparison_adequacy = int(comp_match.group(1)) if comp_match else 5
    
    cite_match = re.search(r'引用规范性[:：]\s*(\d+)', text)
    citation_normative = int(cite_match.group(1)) if cite_match else 5
    
    # 计算加权综合得分
    calculated_score = int(
        innovation_accuracy * 0.5 + 
        comparison_adequacy * 0.3 + 
        citation_normative * 0.2
    )
    # 如果LLM没有提供score，使用计算值
    if score == 0 and (innovation_accuracy > 0 or comparison_adequacy > 0 or citation_normative > 0):
        score = calculated_score
    
    return {
        "reflections": reflections,
        "score": score if score > 0 else calculated_score,
        "found_solution": found_solution if score > 0 else (calculated_score >= 7),
        "comparison_adequacy": comparison_adequacy,
        "innovation_accuracy": innovation_accuracy,
        "citation_normative": citation_normative,
    }


def format_related_papers(papers: List[PaperInfo]) -> str:
    """格式化相关论文列表"""
    if not papers:
        return "无相关论文信息。"
    return "\n\n".join([paper.to_context_string() for paper in papers])


def format_related_papers_with_metadata(papers: List[PaperInfo]) -> str:
    """格式化相关论文列表（带完整元数据）"""
    if not papers:
        return "无相关论文信息。"
    lines = []
    for paper in papers:
        lines.append(paper.to_citation_string())
        lines.append(f"    摘要: {paper.abstract[:300]}...")
        lines.append("")
    return "\n".join(lines)


def generate_citations_section(evaluation_text: str, papers: List[PaperInfo]) -> str:
    """根据评价文本中引用的文献生成参考文献列表"""
    # 提取所有引用序号
    cited_indices = set()
    for match in re.finditer(r'\[(\d+)\]', evaluation_text):
        cited_indices.add(int(match.group(1)))
    
    # 生成参考文献列表
    references = []
    for idx in sorted(cited_indices):
        for paper in papers:
            if paper.index == idx:
                references.append(paper.to_citation_string())
                break
    
    return "\n".join(references)


# ============================================================================
# 链定义
# ============================================================================

# 反思链
reflection_prompt_template = PromptTemplate.from_template(INNOVATION_REFLECTION_PROMPT)

reflection_llm_chain_without_parser = (
    reflection_prompt_template
    | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(run_name="Reflection")
)

reflection_llm_chain = (
    reflection_prompt_template
    | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(run_name="Reflection")
    | StrOutputParser()
)

@as_runnable
def reflection_chain(inputs) -> Reflection:
    """执行反思链"""
    try:
        lats_logging(f"当前候选评价: {inputs.get('current_evaluation', '')[:200]}...\n")
        # 使用bind_tools时返回的是AIMessage，包含tool_calls
        response = reflection_llm_chain_without_parser.invoke(inputs)
        
        # 从tool_calls中提取工具调用参数
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            refdict = tool_call.get('args', {})
        else:
            # 回退到文本解析
            raw_text = response.content if hasattr(response, 'content') else str(response)
            refdict = extract_with_regex(raw_text)
        
        lats_logging(f"生成反思: {str(refdict)[:200]}...\n")
        reflection = Reflection(**refdict)
        
        # 只有包含 AIMessage 的候选才可能是有效解
        if not isinstance(inputs.get("candidate", [None])[-1], AIMessage):
            reflection.found_solution = False
            
        return reflection
        
    except ValidationError as e:
        lats_logging(f"数据验证失败: {e.errors()}")
        return Reflection(
            reflections=f"数据验证错误: {str(e)}",
            score=0,
            found_solution=False
        )
    except Exception as e:
        lats_logging(f"反思链执行失败: {e}")
        return Reflection(
            reflections="反思过程发生错误",
            score=0,
            found_solution=False
        )


# 初始创新性评价生成链
initial_evaluation_template = PromptTemplate.from_template(INNOVATION_GENERATION_PROMPT)

initial_answer_chain = initial_evaluation_template | llm.with_config(run_name="GenerateInitialEvaluation")


# 改进创新性评价链
improvement_template = PromptTemplate.from_template(INNOVATION_IMPROVEMENT_PROMPT)

improvement_chain = improvement_template | llm.with_config(run_name="ImproveEvaluation")


# 最终报告生成链
final_report_template =  PromptTemplate.from_template(FINAL_INNOVATION_REPORT_PROMPT)

final_report_chain = final_report_template | llm.with_config(run_name="GenerateFinalReport")


# ============================================================================
# 节点函数
# ============================================================================

@backoff.on_exception(backoff.expo, IndexError, max_tries=5)
def generate_initial_response(state: TreeState) -> dict:
    """生成初始创新性评价"""
    lats_logging("生成初始创新性评价...")
    
    related_papers_str = format_related_papers(state["related_papers"])
    
    try:
        res = initial_answer_chain.invoke({
            "paper_title": state["paper_title"],
            "paper_abstract": state["paper_abstract"],
            "related_papers": related_papers_str
        })
        # 检查res和res.content是否有效
        if res is None:
            raise ValueError("LLM返回None")
        if not hasattr(res, 'content'):
            raise ValueError(f"LLM返回对象没有content属性: {type(res)}")
        
        # 使用辅助函数安全提取内容
        evaluation_text = safe_extract_content(res.content, "")
        if not evaluation_text:
            raise ValueError(f"LLM返回空内容或无效内容: '{res.content}'")
            
        output_messages = [res]
        lats_logging(f"LLM返回内容长度: {len(evaluation_text)}")
    except Exception as e:
        lats_logging(f"初始评价生成失败: {e}")
        # 使用默认评价作为备用 - 确保相关论文字符串存在
        try:
            papers_summary = related_papers_str[:200] if related_papers_str else "无相关论文信息"
        except:
            papers_summary = "无相关论文信息"
        
        evaluation_text = f"""【论文创新性评价】

论文标题：{state["paper_title"]}

创新点概述：
基于论文摘要，该研究在{state["paper_abstract"][:50] if state["paper_abstract"] else "未知领域"}...方面进行了探索。

相关研究对比：
{papers_summary}...

综合评价：
该论文提出了具有一定创新性的方法，在相关领域有潜在贡献。建议进一步完善对比分析和实验验证。

注意：此评价为默认生成，因LLM调用出现问题，建议重新运行以获得更准确的评价。"""
        
        # 创建一个模拟的AIMessage对象
        res = AIMessage(content=evaluation_text)
        output_messages = [res]
        lats_logging(f"使用默认备用评价，长度: {len(evaluation_text)}")
    lats_logging(f"初始评价生成完成，长度: {len(evaluation_text)}")
    
    # 对初始评价进行反思
    reflection = reflection_chain.invoke({
        "paper_title": state["paper_title"],
        "paper_abstract": state["paper_abstract"],
        "related_papers": related_papers_str,
        "current_evaluation": evaluation_text,
        "candidate": output_messages
    })
    
    # 初始评价不应直接标记为已解决
    if reflection.found_solution:
        reflection.score = min(reflection.score, 5)
        reflection.found_solution = False
        
    lats_logging(f"初始反思: 得分={reflection.score}, 是否解决={reflection.found_solution}")
    
    root = Node(output_messages, reflection=reflection, evaluation=evaluation_text)
    lats_logging(f"根节点创建完成: {root}")
    
    return {
        **state,
        "root": root,
        "best_evaluation": evaluation_text,
    }


@backoff.on_exception(backoff.expo, IndexError, max_tries=5)
def expand(state: TreeState, config: RunnableConfig) -> dict:
    """扩展树节点"""
    lats_logging("扩展树节点...")
    root = state["root"]
    print_tree(root)
    
    # 找到当前路径上的最深层叶子节点（沿最佳路径展开）
    def get_deepest_leaf(node: Node) -> Node:
        if not node.children:
            return node
        # 沿着分数最高的子节点递归
        best_child = max(node.children, key=lambda c: c.value)
        return get_deepest_leaf(best_child)
    
    expand_node = get_deepest_leaf(root)
    lats_logging(f"展开节点: {expand_node}, 深度: {expand_node.depth}")
    
    # 获取从根到当前节点的完整轨迹
    messages = expand_node.get_trajectory()
    related_papers_str = format_related_papers(state["related_papers"])
    
    # 生成改进后的评价
    n_candidates = config["configurable"].get("N", 5)
    new_candidates = []
    
    current_evaluation = expand_node.evaluation or state.get("best_evaluation", "")
    if not current_evaluation:
        lats_logging("警告: evaluation 为空，使用默认初始评价")
        current_evaluation = "这是初始创新性评价，需要改进。"
    
    for i in range(n_candidates):
        reflection_feedback = expand_node.reflection.reflections if expand_node.reflection else ""
        improved = improvement_chain.invoke({
            "paper_title": state["paper_title"],
            "paper_abstract": state["paper_abstract"],
            "related_papers": related_papers_str,
            "current_evaluation": current_evaluation,
            "reflection_feedback": reflection_feedback
        })
        new_candidates.append(improved)
    
    lats_logging(f"生成 {len(new_candidates)} 个新候选评价")
    
    # 对每个新候选进行反思
    output_messages = [[candidate] for candidate in new_candidates]
    reflections = reflection_chain.batch(
        [{
            "paper_title": state["paper_title"],
            "paper_abstract": state["paper_abstract"],
            "related_papers": related_papers_str,
            "current_evaluation": safe_extract_content(candidate.content, fallback=current_evaluation),
            "candidate": msges
        } for candidate, msges in zip(new_candidates, output_messages)],
        config,
    )
    
    # 选择分数最高的候选作为唯一子节点（抛弃其他候选）
    child_nodes = []
    for idx, (candidate, reflection) in enumerate(zip(new_candidates, reflections)):
        content_str = safe_extract_content(candidate.content, fallback=current_evaluation)
        child_node = Node([candidate], parent=expand_node, reflection=reflection, evaluation=content_str)
        child_nodes.append(child_node)
    
    # 只选择分数最高的子节点保留，抛弃其他
    best_child = max(child_nodes, key=lambda c: c.value)
    expand_node.children = [best_child]
    
    lats_logging(f"选择最佳候选: {best_child}, 分数: {best_child.value:.4f}")
    
    # 更新最佳评价
    if best_child.evaluation:
        state["best_evaluation"] = best_child.evaluation
    
    return state


def should_loop(state: TreeState) -> Literal["expand", "__end__"]:
    """判断是否继续扩展"""
    root = state["root"]
    lats_logging(f"检查是否继续搜索。树高度: {root.height}, 已解决: {root.is_solved}")
    
    if root.is_solved:
        lats_logging("找到解决方案，结束搜索")
        return "__end__"
    if root.height >= 5:
        lats_logging("达到最大高度，结束搜索")
        return "__end__"
    
    lats_logging("继续扩展树")
    return "expand"


# ============================================================================
# 图构建
# ============================================================================

def build_graph():
    """构建状态图"""
    builder = StateGraph(TreeState)
    builder.add_node("start", generate_initial_response)
    builder.add_node("expand", expand)
    builder.add_edge(START, "start")

    builder.add_conditional_edges(
        "start",
        should_loop,
        {"expand": "expand", "__end__": END}
    )
    builder.add_conditional_edges(
        "expand",
        should_loop,
        {"expand": "expand", "__end__": END}
    )

    graph = builder.compile()
    return graph


# ============================================================================
# 可视化工具
# ============================================================================

def print_tree(node, level=0):
    """打印树结构"""
    pass


# ============================================================================
# 日志
# ============================================================================

lats_log = []


def lats_logging(event: str):
    """记录日志"""
    global lats_log
    lats_log.append(event)
    print(event)


# ============================================================================
# 主运行函数
# ============================================================================

def generate_final_report(paper_title: str, paper_abstract: str, 
                         related_papers: List[PaperInfo], 
                         best_evaluation: str) -> str:
    """生成最终的创新性评价报告"""
    lats_logging("生成最终创新性评价报告...")
    
    related_papers_str = format_related_papers_with_metadata(related_papers)
    
    final_report = final_report_chain.invoke({
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "related_papers_with_metadata": related_papers_str,
        "draft_evaluation": best_evaluation
    })
    
    report_content = final_report.content
    
    # 添加参考文献列表
    references = generate_citations_section(report_content, related_papers)
    if references and "## 4. 参考文献" not in report_content:
        report_content += f"\n\n## 4. 参考文献\n{references}"
    
    return report_content


def run_innovation_evaluation(
    paper_title: str,
    paper_abstract: str,
    related_papers_data: List[Dict[str, Any]],
    max_iterations: int = 5
) -> tuple[str, list]:
    """
    运行创新性评价树搜索
    
    Args:
        paper_title: 论文标题
        paper_abstract: 论文摘要
        related_papers_data: 相关论文数据列表，每项包含 title, authors, venue, year, abstract 等
        max_iterations: 最大迭代次数
    
    Returns:
        (最终评价报告, 日志列表)
    """
    global lats_log
    lats_log = []
    
    # 转换相关论文数据
    related_papers = []
    for i, paper_data in enumerate(related_papers_data[:10]):  # 最多使用10篇相关论文
        paper = PaperInfo(
            index=i,
            title=paper_data.get("title", ""),
            authors=paper_data.get("authors", ""),
            venue=paper_data.get("venue", ""),
            year=paper_data.get("year", 2024),
            abstract=paper_data.get("abstract", ""),
            citation_count=paper_data.get("citationCount")
        )
        related_papers.append(paper)
    
    lats_logging(f"开始创新性评价树搜索")
    lats_logging(f"论文标题: {paper_title}")
    lats_logging(f"相关论文数量: {len(related_papers)}")
    
    last_step = None
    graph = build_graph()
    
    config = {"configurable": {"N": 5, "max_iterations": max_iterations}}
    
    initial_state = {
        "root": None,
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "related_papers": related_papers,
        "best_evaluation": None
    }
    
    for step in graph.stream(initial_state, config=config):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        lats_logging(f"步骤: {step_name}")
        lats_logging(f"树高度: {step_state['root'].height}")
        lats_logging("-" * 50)
    
    # 获取最佳解决方案
    if last_step:
        root = last_step.get("expand", last_step.get("start", {})).get("root")
        if root:
            solution_node = root.get_best_solution()
            best_evaluation = solution_node.evaluation if solution_node else ""
            
            # 生成最终报告
            final_report = generate_final_report(
                paper_title,
                paper_abstract,
                related_papers,
                best_evaluation
            )
            
            lats_logging("最终树结构:")
            print_tree(root)
            
            return final_report, lats_log
    
    return "评价生成失败", lats_log


# ============================================================================
# 与 OpenScholar 集成的接口
# ============================================================================

def evaluate_paper_innovation(
    paper_title: str,
    paper_abstract: str,
    retrieved_papers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    评价论文学术创新性（用于与 open_scholar 集成）
    
    Args:
        paper_title: 待评价论文标题
        paper_abstract: 待评价论文摘要
        retrieved_papers: 从 open_scholar 检索到的相关论文列表
    
    Returns:
        包含创新性评价结果的字典
    """
    report, log = run_innovation_evaluation(
        paper_title=paper_title,
        paper_abstract=paper_abstract,
        related_papers_data=retrieved_papers
    )
    
    return {
        "innovation_evaluation": report,
        "evaluation_log": log,
        "success": True
    }


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    # 测试数据
    test_title = "A Novel Deep Learning Approach for Image Classification"
    test_abstract = """
    We propose a new deep learning architecture that combines transformer and CNN 
    for improved image classification accuracy. Our method achieves state-of-the-art 
    results on ImageNet and CIFAR-10 datasets.
    """
    
    test_related_papers = [
        {
            "title": "Attention Is All You Need",
            "authors": "Vaswani et al.",
            "venue": "NeurIPS",
            "year": 2017,
            "abstract": "We propose a new simple network architecture, the Transformer..."
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "authors": "He et al.",
            "venue": "CVPR",
            "year": 2016,
            "abstract": "We present a residual learning framework to ease the training..."
        },
        {
            "title": "ImageNet Classification with Deep Convolutional Neural Networks",
            "authors": "Krizhevsky et al.",
            "venue": "NeurIPS",
            "year": 2012,
            "abstract": "We trained a large, deep convolutional neural network..."
        }
    ]
    
    print("=" * 60)
    print("测试创新性评价系统")
    print("=" * 60)
    
    result = evaluate_paper_innovation(test_title, test_abstract, test_related_papers)
    
    print("\n" + "=" * 60)
    print("最终创新性评价报告")
    print("=" * 60)
    print(result["innovation_evaluation"])
