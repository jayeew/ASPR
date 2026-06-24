from __future__ import annotations
import getpass
import os
import json
import io
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any
try:
    import backoff
except ImportError:
    class _BackoffShim:
        @staticmethod
        def expo(*args, **kwargs):
            return None

        @staticmethod
        def on_exception(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

    backoff = _BackoffShim()
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

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from aspr.prompts import (
        FINAL_INNOVATION_REPORT_PROMPT,
        INNOVATION_GENERATION_PROMPT,
        INNOVATION_IMPROVEMENT_PROMPT,
        INNOVATION_REFLECTION_PROMPT,
    )
    from aspr.graph_innovation_scorer import GraphInnovationScorer
    from aspr.review_committee import run_reviewer_committee
else:
    from .prompts import (
        FINAL_INNOVATION_REPORT_PROMPT,
        INNOVATION_GENERATION_PROMPT,
        INNOVATION_IMPROVEMENT_PROMPT,
        INNOVATION_REFLECTION_PROMPT,
    )
    from .graph_innovation_scorer import GraphInnovationScorer
    from .review_committee import run_reviewer_committee


# ============================================================================
# LLM 配置
# ============================================================================

# 配置 ChatOpenAI 连接本地 Ollama
llm = ChatOpenAI(
    model=os.getenv("ASPR_LATS_LLM_MODEL", "qwen3-coder:30b"),
    base_url=os.getenv("ASPR_LATS_LLM_BASE_URL")
    or os.getenv("ASPR_LLM_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("ASPR_LATS_LLM_API_KEY")
    or os.getenv("ASPR_LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or "ollama",
    temperature=0.2,
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
    doi: Optional[str] = Field(default=None, description="DOI")
    fields_of_study: List[str] = Field(default_factory=list, description="领域标签")
    
    def to_citation_string(self) -> str:
        """生成引用字符串"""
        return f"[{self.index}] {self.authors}. {self.title}. {self.venue}, {self.year}."
    
    def to_context_string(self) -> str:
        """生成上下文字符串（用于输入模型）"""
        fields = f"\nFields: {', '.join(self.fields_of_study)}" if self.fields_of_study else ""
        return f"[{self.index}] Title: {self.title}\nAuthors: {self.authors}\nVenue: {self.venue}, {self.year}{fields}\nAbstract: {self.abstract}\n"


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
        description="是否达到高质量创新性评价标准 (True/False)。只有当评价充分对比了相关研究、正确识别创新点、论证逻辑清晰、引用规范、且图谱证据对齐时为 True"
    )
    comparison_adequacy: int = Field(default=0, description="对比充分性评分 (0-10)", ge=0, le=10)
    innovation_accuracy: int = Field(default=0, description="创新性识别准确性评分 (0-10)", ge=0, le=10)
    citation_normative: int = Field(default=0, description="引用规范性评分 (0-10)", ge=0, le=10)
    graph_metric_alignment: int = Field(default=0, description="图谱结构证据对齐评分 (0-10)", ge=0, le=10)
    uncertainty_calibration: int = Field(default=0, description="不确定性校准评分 (0-10)", ge=0, le=10)
    readability: int = Field(default=0, description="表达清晰度评分 (0-10)", ge=0, le=10)

    @field_validator('reflections', mode='before')
    @classmethod
    def extract_str_(cls, v):
        return str(v)

    @field_validator(
        'score',
        'comparison_adequacy',
        'innovation_accuracy',
        'citation_normative',
        'graph_metric_alignment',
        'uncertainty_calibration',
        'readability',
        mode='before',
    )
    @classmethod
    def extract_int_(cls, v):
        try:
            return max(0, min(int(float(v)), 10))
        except (TypeError, ValueError):
            return 0

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
            content=(
                f"反思: {self.reflections}\n"
                f"综合得分: {self.score}/10\n"
                f"图谱对齐: {self.graph_metric_alignment}/10\n"
                f"不确定性校准: {self.uncertainty_calibration}/10"
            )
        )

    @property
    def normalized_score(self) -> float:
        component_scores = [
            self.innovation_accuracy,
            self.comparison_adequacy,
            self.citation_normative,
            self.graph_metric_alignment,
            self.uncertainty_calibration,
            self.readability,
        ]
        if any(score > 0 for score in component_scores):
            weighted = (
                self.innovation_accuracy * 0.30
                + self.comparison_adequacy * 0.20
                + self.citation_normative * 0.15
                + self.graph_metric_alignment * 0.20
                + self.uncertainty_calibration * 0.10
                + self.readability * 0.05
            )
            return weighted / 10.0
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
        average_reward = self.value
        exploration_term = math.sqrt(math.log(max(self.parent.visits, 1)) / self.visits)
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
            key=lambda node: (
                int(node.is_solved),
                node.value,
                node.reflection.normalized_score if node.reflection else 0.0,
            ),
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
    graph_metric_evidence: str       # 七指标图谱证据块
    committee_evidence: str          # 审稿委员会结构化证据块
    committee_disagreement_score: float
    recommended_tone: str
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
    def extract_dimension(pattern: str, default: int = 5) -> int:
        match = re.search(pattern, text)
        if not match:
            return default
        try:
            return max(0, min(int(match.group(1)), 10))
        except ValueError:
            return default

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
    innovation_accuracy = extract_dimension(r'创新性识别准确性[:：]\s*(\d+)')
    comparison_adequacy = extract_dimension(r'对比充分性[:：]\s*(\d+)')
    citation_normative = extract_dimension(r'引用规范性[:：]\s*(\d+)')
    graph_metric_alignment = extract_dimension(r'图谱结构证据对齐[:：]\s*(\d+)')
    uncertainty_calibration = extract_dimension(r'不确定性校准[:：]\s*(\d+)')
    readability = extract_dimension(r'表达清晰度[:：]\s*(\d+)')
    
    # 计算加权综合得分
    calculated_score = round(
        innovation_accuracy * 0.30
        + comparison_adequacy * 0.20
        + citation_normative * 0.15
        + graph_metric_alignment * 0.20
        + uncertainty_calibration * 0.10
        + readability * 0.05
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
        "graph_metric_alignment": graph_metric_alignment,
        "uncertainty_calibration": uncertainty_calibration,
        "readability": readability,
    }


def calibrate_reflection_with_committee(
    reflection: Reflection,
    disagreement_score: float = 0.0,
    recommended_tone: str = "balanced",
) -> Reflection:
    """Adjust reflection scores using committee disagreement and tone."""
    tone = (recommended_tone or "balanced").lower()
    disagreement_score = max(0.0, min(float(disagreement_score or 0.0), 1.0))
    if tone == "conservative":
        if reflection.uncertainty_calibration < 7:
            reflection.uncertainty_calibration = max(0, reflection.uncertainty_calibration - 2)
        if reflection.graph_metric_alignment < 7:
            reflection.graph_metric_alignment = max(0, reflection.graph_metric_alignment - 1)
        reflection.found_solution = False
    elif tone == "assertive" and disagreement_score < 0.30:
        if reflection.graph_metric_alignment >= 7:
            reflection.graph_metric_alignment = min(10, reflection.graph_metric_alignment + 1)
        if reflection.uncertainty_calibration >= 6:
            reflection.uncertainty_calibration = min(10, reflection.uncertainty_calibration + 1)

    if disagreement_score >= 0.45:
        reflection.uncertainty_calibration = max(0, reflection.uncertainty_calibration - 1)
        reflection.found_solution = False
    if disagreement_score >= 0.65:
        reflection.graph_metric_alignment = max(0, reflection.graph_metric_alignment - 1)
        reflection.found_solution = False

    reflection.score = max(0, min(round(reflection.normalized_score * 10), 10))
    return reflection


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
        lats_logging(f"当前候选评价: \n{inputs.get('current_evaluation', '')}\n")
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
        
        lats_logging(f"生成反思: \n{str(refdict)}\n")
        reflection = Reflection(**refdict)
        if any(
            score > 0
            for score in (
                reflection.innovation_accuracy,
                reflection.comparison_adequacy,
                reflection.citation_normative,
                reflection.graph_metric_alignment,
                reflection.uncertainty_calibration,
                reflection.readability,
            )
        ):
            reflection.score = max(0, min(round(reflection.normalized_score * 10), 10))
        reflection = calibrate_reflection_with_committee(
            reflection,
            disagreement_score=float(inputs.get("committee_disagreement_score", 0.0) or 0.0),
            recommended_tone=str(inputs.get("recommended_tone", "balanced") or "balanced"),
        )
        
        # 只有包含 AIMessage 的候选才可能是有效解
        if not isinstance(inputs.get("candidate", [None])[-1], AIMessage):
            reflection.found_solution = False
        if reflection.graph_metric_alignment < 6 or reflection.uncertainty_calibration < 5:
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
    graph_metric_evidence = state.get("graph_metric_evidence", "图谱结构证据未计算。")
    committee_evidence = state.get("committee_evidence", "审稿委员会证据未计算。")
    committee_disagreement_score = float(state.get("committee_disagreement_score", 0.0) or 0.0)
    recommended_tone = state.get("recommended_tone", "balanced")
    
    try:
        res = initial_answer_chain.invoke({
            "paper_title": state["paper_title"],
            "paper_abstract": state["paper_abstract"],
            "related_papers": related_papers_str,
            "graph_metric_evidence": graph_metric_evidence,
            "committee_evidence": committee_evidence,
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
        "graph_metric_evidence": graph_metric_evidence,
        "committee_evidence": committee_evidence,
        "committee_disagreement_score": committee_disagreement_score,
        "recommended_tone": recommended_tone,
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
    
    # 沿 UCB 路径选择待扩展叶子节点
    def get_deepest_leaf(node: Node) -> Node:
        if not node.children:
            return node
        best_child = max(node.children, key=lambda c: c.upper_confidence_bound())
        return get_deepest_leaf(best_child)
    
    expand_node = get_deepest_leaf(root)
    lats_logging(f"展开节点: {expand_node}, 深度: {expand_node.depth}")
    
    related_papers_str = format_related_papers(state["related_papers"])
    graph_metric_evidence = state.get("graph_metric_evidence", "图谱结构证据未计算。")
    committee_evidence = state.get("committee_evidence", "审稿委员会证据未计算。")
    committee_disagreement_score = float(state.get("committee_disagreement_score", 0.0) or 0.0)
    recommended_tone = state.get("recommended_tone", "balanced")
    
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
            "graph_metric_evidence": graph_metric_evidence,
            "committee_evidence": committee_evidence,
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
            "graph_metric_evidence": graph_metric_evidence,
            "committee_evidence": committee_evidence,
            "committee_disagreement_score": committee_disagreement_score,
            "recommended_tone": recommended_tone,
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
    
    # 保留多个高分候选，让后续轮次可以继续探索不同修正路径
    beam_width = max(1, int(config["configurable"].get("beam_width", 3)))
    child_nodes = sorted(child_nodes, key=lambda c: c.value, reverse=True)
    expand_node.children = child_nodes[:beam_width]
    best_child = expand_node.children[0]
    
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

def print_tree(node, level=0, is_last=True, prefix=""):
    """打印树结构
    
    以树形格式打印搜索树，显示每个节点的关键信息:
    - 节点深度和ID
    - 价值分数和访问次数
    - 反思得分和解决状态
    """
    if node is None:
        return
    
    # 构建当前行的前缀
    if level == 0:
        branch = ""
    else:
        branch = "└── " if is_last else "├── "
    
    # 获取节点信息
    score = node.reflection.score if node.reflection else 0
    solved_mark = "✓" if node.is_solved else "○"
    evaluation_snippet = ""
    if node.evaluation:
        # 提取评价的第一行作为摘要
        first_line = node.evaluation.strip().split('\n')[0][:40]
        evaluation_snippet = f" | {first_line}..."
    
    # 打印当前节点
    print(f"{prefix}{branch}[{solved_mark}] L{node.depth}: v={node.value:.2f}, visits={node.visits}, score={score}/10{evaluation_snippet}")
    
    # 递归打印子节点
    if node.children:
        for i, child in enumerate(node.children):
            child_is_last = (i == len(node.children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(child, level + 1, child_is_last, child_prefix)


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
                         best_evaluation: str,
                         graph_metric_evidence: str = "图谱结构证据未计算。",
                         committee_evidence: str = "审稿委员会证据未计算。") -> str:
    """生成最终的创新性评价报告"""
    lats_logging("生成最终创新性评价报告...")
    
    related_papers_str = format_related_papers_with_metadata(related_papers)
    
    final_report = final_report_chain.invoke({
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "related_papers_with_metadata": related_papers_str,
        "graph_metric_evidence": graph_metric_evidence,
        "committee_evidence": committee_evidence,
        "draft_evaluation": best_evaluation
    })
    
    report_content = final_report.content
    
    return report_content


def build_graph_metric_evidence(
    paper_title: str,
    paper_abstract: str,
    related_papers_data: List[Dict[str, Any]],
    max_related_papers: int = 10,
) -> tuple[str, Dict[str, Any]]:
    """Compute seven-indicator graph evidence for prompt grounding."""
    evidence = GraphInnovationScorer().score(
        paper_title=paper_title,
        paper_abstract=paper_abstract,
        retrieved_papers=related_papers_data[:max_related_papers],
    )
    return evidence.to_prompt_block(), evidence.to_dict()


def build_committee_evidence(
    paper_title: str,
    paper_abstract: str,
    related_papers_data: List[Dict[str, Any]],
    graph_metric_result: Dict[str, Any],
    use_committee: bool = True,
) -> tuple[str, Dict[str, Any], float, str]:
    """Run the reviewer committee and render its prompt block."""
    if not use_committee:
        empty_report = {
            "claim_cards": [],
            "agent_reviews": [],
            "disagreement_score": 0.0,
            "meta_review_summary": "Reviewer committee disabled for ablation.",
            "recommended_tone": "balanced",
        }
        return "【审稿委员会证据 / Reviewer Committee Evidence】\nReviewer committee disabled for ablation.", empty_report, 0.0, "balanced"

    committee_report = run_reviewer_committee(
        paper_title=paper_title,
        paper_abstract=paper_abstract,
        related_papers=related_papers_data,
        graph_metric_result=graph_metric_result,
    )
    return (
        committee_report.to_prompt_block(),
        committee_report.to_dict(),
        committee_report.disagreement_score,
        committee_report.recommended_tone,
    )


def run_innovation_evaluation(
    paper_title: str,
    paper_abstract: str,
    related_papers_data: List[Dict[str, Any]],
    max_iterations: int = 3,
    max_related_papers: int = 10,
    graph_metric_evidence: Optional[str] = None,
    graph_metric_result: Optional[Dict[str, Any]] = None,
    committee_evidence: Optional[str] = None,
    committee_report_result: Optional[Dict[str, Any]] = None,
    committee_disagreement_score: float = 0.0,
    recommended_tone: str = "balanced",
    use_committee: bool = True,
) -> tuple[str, list, Dict[str, Any], float, str]:
    """
    运行创新性评价树搜索
    
    Args:
        paper_title: 论文标题
        paper_abstract: 论文摘要
        related_papers_data: 相关论文数据列表，每项包含 title, authors, venue, year, abstract 等
        max_iterations: 最大迭代次数
        max_related_papers: 最大相关论文数量
    
    Returns:
        (最终评价报告, 日志列表, committee 报告, committee 分歧分数, 建议语气)
    """
    global lats_log
    lats_log = []
    if graph_metric_evidence is None or graph_metric_result is None:
        graph_metric_evidence, graph_metric_result = build_graph_metric_evidence(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            related_papers_data=related_papers_data,
            max_related_papers=max_related_papers,
        )
    if committee_evidence is None or committee_report_result is None:
        (
            committee_evidence,
            committee_report_result,
            committee_disagreement_score,
            recommended_tone,
        ) = build_committee_evidence(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            related_papers_data=related_papers_data[:max_related_papers],
            graph_metric_result=graph_metric_result,
            use_committee=use_committee,
        )
    
    # 转换相关论文数据
    related_papers = []
    for i, paper_data in enumerate(related_papers_data[:max_related_papers]):
        paper = PaperInfo(
            index=i,
            title=paper_data.get("title", "").replace("<sub>", "").replace("</sub>", ""),
            authors=paper_data.get("authors", "").replace("<sub>", "").replace("</sub>", ""),
            venue=paper_data.get("venue", "").replace("<sub>", "").replace("</sub>", ""),
            year=paper_data.get("year", 2024),
            abstract=paper_data.get("abstract", "").replace("<sub>", "").replace("</sub>", ""),
            citation_count=paper_data.get("citationCount"),
            doi=paper_data.get("doi"),
            fields_of_study=paper_data.get("fieldsOfStudy") or [
                item.get("category", "")
                for item in (paper_data.get("s2FieldsOfStudy") or [])
                if isinstance(item, dict) and item.get("category")
            ],
        )
        related_papers.append(paper)
    
    lats_logging(f"开始创新性评价树搜索")
    lats_logging(f"论文标题: {paper_title}")
    lats_logging(f"相关论文数量: {len(related_papers)}")
    lats_logging("已载入七指标图谱证据")
    lats_logging(f"已载入审稿委员会证据: tone={recommended_tone}, disagreement={committee_disagreement_score:.3f}")
    
    last_step = None
    graph = build_graph()
    
    config = {"configurable": {"N": 5, "max_iterations": max_iterations}}
    
    initial_state = {
        "root": None,
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "related_papers": related_papers,
        "graph_metric_evidence": graph_metric_evidence,
        "committee_evidence": committee_evidence,
        "committee_disagreement_score": committee_disagreement_score,
        "recommended_tone": recommended_tone,
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
                best_evaluation,
                graph_metric_evidence=graph_metric_evidence,
                committee_evidence=committee_evidence,
            )
            
            lats_logging("最终树结构:")
            print_tree(root)
            
            return final_report, lats_log, committee_report_result, committee_disagreement_score, recommended_tone
    
    return "评价生成失败", lats_log, committee_report_result or {}, committee_disagreement_score, recommended_tone


# ============================================================================
# 与 OpenScholar 集成的接口
# ============================================================================

def evaluate_paper_innovation(
    paper_title: str,
    paper_abstract: str,
    retrieved_papers: List[Dict[str, Any]],
    paper_context: Optional[str] = None,
    max_iterations: int = 3,
    graph_metric_evidence: Optional[str] = None,
    graph_metric_result: Optional[Any] = None,
    use_committee: bool = True,
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
    evaluation_abstract = paper_abstract
    if paper_context:
        evaluation_abstract = f"{paper_abstract}\n\nFull paper dossier for evaluation:\n{paper_context}"
    if graph_metric_evidence is None or graph_metric_result is None:
        graph_metric_evidence, graph_metric_result = build_graph_metric_evidence(
            paper_title=paper_title,
            paper_abstract=evaluation_abstract,
            related_papers_data=retrieved_papers,
        )
    (
        committee_evidence,
        committee_report_result,
        committee_disagreement_score,
        recommended_tone,
    ) = build_committee_evidence(
        paper_title=paper_title,
        paper_abstract=evaluation_abstract,
        related_papers_data=retrieved_papers,
        graph_metric_result=graph_metric_result,
        use_committee=use_committee,
    )
    report, log, committee_report_result, committee_disagreement_score, recommended_tone = run_innovation_evaluation(
        paper_title=paper_title,
        paper_abstract=evaluation_abstract,
        related_papers_data=retrieved_papers,
        graph_metric_evidence=graph_metric_evidence,
        graph_metric_result=graph_metric_result,
        committee_evidence=committee_evidence,
        committee_report_result=committee_report_result,
        committee_disagreement_score=committee_disagreement_score,
        recommended_tone=recommended_tone,
        max_iterations=max_iterations,
        use_committee=use_committee,
    )
    
    return {
        "innovation_evaluation": report,
        "graph_metric_evidence": graph_metric_result,
        "committee_report": committee_report_result,
        "committee_disagreement_score": committee_disagreement_score,
        "recommended_tone": recommended_tone,
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
