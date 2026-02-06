from __future__ import annotations
import getpass
import os
import json
import io
import math
import re
from collections import deque
from typing import Optional, Literal
import backoff
from contextlib import redirect_stdout
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from collections import defaultdict
from PrettyPrint import PrettyPrintTree
from llm.new_prompt import (primary_prompt, 
                            summary_prompt, 
                            structured_output_section, 
                            plan_prompt, 
                            code_prompt, 
                            analysis_prompt, 
                            web_prompt,
                            plan_prompt_v2,
                            reflection_prompt,)
from util.search import web_search
from langchain_openai import ChatOpenAI


# 配置 ChatOpenAI 连接本地 Ollama
llm = ChatOpenAI(
    model="qwen3:30b",  # 使用你本地安装的模型名称
    base_url="http://localhost:11434/v1",  # Ollama 的 OpenAI 兼容端点
    api_key="ollama",  # Ollama 不需要真正的 API key，但需要设置一个非空值
    temperature=0.3,
    max_tokens=9000,
    # 其他参数...
)

# # 测试连接
# try:
#     response = llm.invoke("你好")
#     lats_logging("连接成功:", response.content)
# except Exception as e:
#     lats_logging("连接失败:", e)

class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional[Node] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)
        lats_logging(f"Created node : {self}")

    def __repr__(self) -> str:
        return (
            f"<{self.value:.2f},{self.visits},{self._is_solved}>"
            # f"<Node value={self.value:.2f}, visits={self.visits},"
            # f" Response={self.messages[-1].content[:50] if self.messages else 'No messages'}...,"
            # f" Reflection={self.reflection.reflections[:50] if self.reflection else 'No reflection'}...,"
            # f" is_solved={self._is_solved}, depth={self.depth}/>"
        )

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
        if include_reflections:
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

class Reflection(BaseModel):
    reflections: str = Field(
        default="",
        description="Critiques and reflections on the quality of innovation evaluation in academic papers focus on whether such evaluation is appropriate, comprehensive, and legitimate."
    )
    score: int = Field(
        default=0,
        description="Scoring results for candidate responses (1–10 points)",
        ge=0,
        le=10,
    )
    found_solution: bool = Field(
        default=False,
        description="Whether an appropriate, comprehensive, and legitimate evaluation of the paper's innovation has been identified (True/False). This field is True only when a solution is found; otherwise, it is False."
    )

    @field_validator('reflections', mode='before')
    @classmethod
    def extract_str_(cls, v):
        # lats_logging(f'reflections checking : {isinstance(v, str)}\n')
        return str(v)

    @field_validator('score', mode='before')
    @classmethod
    def extract_int_(cls, v):
        return int(v)

    @field_validator('found_solution', mode='before')
    @classmethod
    def extract_bool_from_dict(cls, v):
        """
        在验证前处理输入。如果输入是字典，尝试从中提取布尔值。
        支持多种可能的字典键名，如 'value', 'result', 'type'。
        """
        # 如果已经是布尔值，直接返回
        if isinstance(v, bool):
            return v
        
        # 如果输入是字典，尝试提取
        elif isinstance(v, dict):
            # 方案1: 检查常见的键名，如 'value'
            potential_bool = v.get('value')
            if isinstance(potential_bool, bool):
                return potential_bool
            
            # 方案2: 您的错误信息中字典键是 'type'，值为 1
            # 可以假设 1 代表 True, 0 代表 False（需根据数据源逻辑确认）
            if 'type' in v:
                type_value = v['type']
                if type_value == 1:
                    return True
                elif type_value == 0:
                    return False
            
            # 方案3: 如果字典有 'result' 键，且其值为字符串 'true'/'false'
            result_str = v.get('result', '').lower()
            if result_str == 'true':
                return True
            elif result_str == 'false':
                return False
        
        return False

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0

class TreeState(TypedDict):
    root: Node
    user_query: str
    web_info: str

def extract_with_regex(text):
    # 提取 reflections (字符串)
    reflections_match = re.search(r'<reflections>(.*?)</reflections>', text, re.DOTALL)
    reflections = reflections_match.group(1).strip() if reflections_match else ""
    
    # 提取 score (整数)
    score_match = re.search(r'<score>(\d+)</score>', text)
    score = int(score_match.group(1)) if score_match else 0
    
    # 提取 found_solution (布尔值)
    fs_match = re.search(r'<found_solution>(true|false)</found_solution>', text, re.IGNORECASE)
    found_solution = fs_match.group(1).lower() == 'true' if fs_match else False
    
    return {
        "reflections": reflections,
        "score": score,
        "found_solution": found_solution
    }

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            reflection_prompt
        ),
        ("user", "{user_query}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

# reflection_llm_chain = (
#     prompt
#     | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
#         run_name="Reflection"
#     )
#     | PydanticToolsParser(tools=[Reflection])
# )

reflection_llm_chain = (
    PromptTemplate.from_template(reflection_prompt)
    | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
        run_name="Reflection"
    )
    | StrOutputParser()
)

@as_runnable
def reflection_chain(inputs) -> Reflection:
    global code_rules
    try:
        inputs['code_rules'] = code_rules 
        raw_text = reflection_llm_chain.invoke(inputs)
        refdict = extract_with_regex(raw_text)
        lats_logging(f"Generated reflection: {raw_text} \n")
        reflection = Reflection(**refdict)
        if not isinstance(inputs["candidate"][-1], AIMessage):
            reflection.found_solution = False
        return reflection
        
    except ValidationError as e:
        lats_logging(f"数据验证失败: {e.errors()}")
        return Reflection(
            reflections=f"数据验证错误: {str(e)}",
            score=0,
            found_solution=False
        )
    except (KeyError, IndexError) as e:
        lats_logging(f"输入数据格式错误: {e}")
        return Reflection(
            reflections="输入数据不完整或格式不正确",
            score=0,
            found_solution=False
        )
    except Exception as e:
        lats_logging(f"反思链执行意外失败: {e}")
        return Reflection(
            reflections="反思过程发生意外错误",
            score=0,
            found_solution=False
        )

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            plan_prompt_v2,
        ),
        ("user", "用户评测需求为{user_query}"),
        ("user", "设计实验方案时,可以参考网络检索信息{web_info}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

initial_answer_chain = prompt_template | llm.with_config(run_name="GenerateInitialCandidate")

parser = JsonOutputToolsParser(return_id=True)

@backoff.on_exception(backoff.expo, 
                    IndexError,
                    max_tries=5)
def generate_initial_response(state: TreeState) -> dict:
    lats_logging("Generating initial response")
    res = initial_answer_chain.invoke({"user_query": state["user_query"], "web_info": state["web_info"]})
    output_messages = [res]
    lats_logging(f"Initial response: {res.content}")
    reflection = reflection_chain.invoke(
        {"user_query": state["user_query"], "candidate": output_messages}
    )
    if reflection.found_solution:
        reflection.score = min(reflection.score, 5)
        reflection.found_solution = False
    lats_logging(f"\nInitial reflection: {reflection} \n ")
    root = Node(output_messages, reflection=reflection)
    lats_logging(f"Initial root node created: {root}")
    return {
        **state,
        "root": root,
    }


def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
    n = config["configurable"].get("N", 5)
    lats_logging(f"Generating {n} candidates")
    chat_results = []
    for candnum in range(n):
        chat_result = llm.generate(
            [messages.to_messages()],
            n=n,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates"
        )
        chat_results.append(chat_result)
    # lats_logging(f'generate candidates : {chat_results}')
    lats_logging(f"Chat result length: {len(chat_results)}\n")
    # lats_logging(chat_results[0].generations[0][0].message)
    return [gen.generations[0][0].message for gen in chat_results]

expansion_chain = prompt_template | generate_candidates

@backoff.on_exception(backoff.expo, 
                    IndexError,
                    max_tries=5)
def expand(state: TreeState, config: RunnableConfig) -> dict:
    lats_logging("Expanding tree \n")
    root = state["root"]
    print_tree(root)
    best_candidate: Node = root.best_child if root.children else root
    lats_logging(f"Best candidate for expansion : {best_candidate} \n")
    messages = best_candidate.get_trajectory()

    new_candidates = expansion_chain.invoke(
        {"user_query": state["user_query"], "web_info": state["web_info"], "messages": messages}, config
    )
    lats_logging(f"Generated {len(new_candidates)} new candidates \n")

    output_messages = [[candidate] for candidate in new_candidates]

    reflections = reflection_chain.batch(
        [{"user_query": state["user_query"], "candidate": msges} for msges in output_messages],
        config,
    )

    child_nodes = [
        Node(cand, parent=best_candidate, reflection=reflection)
        for cand, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    lats_logging(f"\n Added {len(child_nodes)} child nodes to the tree \n")

    return state


def should_loop(state: TreeState) -> Literal["expand", "__end__"]:
    root = state["root"]
    lats_logging(f"Checking if should loop again. Root height: {root.height}, Solution Found: {root.is_solved} \n")
    if root.is_solved:
        lats_logging("Root is solved. Ending search. \n")
        return "__end__"
    if root.height > 5:
        lats_logging("Max height reached. Ending search. \n ")
        return "__end__"
    lats_logging("Continuing to expand. \n")
    return "expand"

def build_graph():
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

def color_by_condition(node):
    """根据条件返回带颜色的节点显示值"""
    if node._is_solved:
        return f"✅{str(node)}"  # 绿色
    else:
        return f"❌{str(node)}"  # 红色

def print_tree(node, level=0):

    pt = PrettyPrintTree(
        lambda node: node.children,  
        color_by_condition,
        return_instead_of_print=False,   
    )
    tree_str = pt(node)
    print(tree_str)
    # lats_logging(tree_str)

def lats_logging(event:str):
    global lats_log
    lats_log.append(event)
    print(event)

def run_tree_search(user_query, web_info, code_rules_):
    global code_rules
    global lats_log
    code_rules = code_rules_
    lats_log = []
    lats_logging(f"Starting tree search for question")
    last_step = None
    graph = build_graph()
    for step in graph.stream({"user_query": user_query, "web_info": web_info}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        lats_logging(f"Step: {step_name}")
        lats_logging(f"Tree height: {step_state['root'].height}")
        lats_logging("--------------------------------------------------------")

    if "expand" in last_step:
        solution_node = last_step["expand"]["root"].get_best_solution()
        best_trajectory = solution_node.get_trajectory(include_reflections=False)
        lats_logging("Best solution found:")
        lats_logging(best_trajectory[-1].content)
        # content = best_trajectory[-1].content
    else:
        lats_logging("Tree expansion ended \n")

    lats_logging("Final tree structure:")
    print_tree(last_step["start"]["root"] if "start" in last_step else last_step["expand"]["root"])

    return best_trajectory[-1].content, lats_log


if __name__ == '__main__':

    user_query = ""
    web_info = web_search(user_query+web_prompt, top_k=3, mode='summary', use_jina=False)
    # _, log = run_tree_search(user_query, web_info, '')
    # print(log)
    print(web_info)
