# LATS (Language Agent Tree Search) 创新性评价系统

## 概述

本系统使用树搜索（LATS）方法对学术论文进行深度创新性评价。系统通过多轮反思和迭代优化，生成高质量的创新性评价报告，并自动包含规范的参考文献引用。

## 主要特性

1. **深度创新性分析**
   - 识别论文的理论、方法和应用创新点
   - 与相关研究工作进行详细对比
   - 多维度评估创新水平

2. **智能反思机制**
   - 基于蒙特卡洛树搜索（MCTS）的迭代优化
   - 多维度反思：对比充分性、创新性准确性、论证逻辑性、引用规范性
   - 自动识别并改进低质量评价

3. **规范引用格式**
   - 自动提取引用文献
   - 生成标准学术引用格式
   - 引用与论述自动匹配

4. **与 OpenScholar 集成**
   - 自动使用检索到的相关论文
   - 无缝集成到论文评审流程

## 使用方法

### 方式一：直接使用 LATS 模块

```python
from lats import run_innovation_evaluation

# 准备相关论文数据
related_papers = [
    {
        "title": "论文标题",
        "authors": "作者1, 作者2",
        "venue": "会议/期刊名",
        "year": 2023,
        "abstract": "论文摘要...",
        "citationCount": 100
    },
    # ... 更多论文
]

# 运行创新性评价
report, log = run_innovation_evaluation(
    paper_title="待评价论文标题",
    paper_abstract="待评价论文摘要...",
    related_papers_data=related_papers,
    max_iterations=5  # 最大迭代次数
)

print(report)  # 输出创新性评价报告
```

### 方式二：通过 OpenScholar 使用

```python
from open_scholar import Reviewer
import argparse

# 创建参数
parser = argparse.ArgumentParser()
parser.add_argument('--s2_api_key', type=str, default='your-api-key')
parser.add_argument('--large_model', type=str, default='your-model')
parser.add_argument('--large_model_port', type=int, default=38011)
args = parser.parse_args([])

# 初始化 Reviewer
reviewer = Reviewer(args)

# 准备关键词和论文摘要
keywords = ["machine learning", "deep learning"]
abstract = """
Your paper abstract here...
"""

# 运行评审（自动包含创新性评价）
review = reviewer(keywords, abstract)
print(review)
```

## 创新性评价报告结构

生成的创新性评价报告包含以下部分：

### 1. 创新点概述 (Innovation Summary)
- 简明总结论文的主要创新贡献
- 2-4 条核心创新点

### 2. 与现有研究的对比 (Comparison with Related Work)
- 现有方法/理论的局限性分析
- 本论文的改进之处
- 改进的意义和价值评估

### 3. 创新性评估 (Innovation Assessment)
使用星级评分（最高5★）：
- **理论创新性**：是否提出新理论或新概念
- **方法创新性**：是否提出新方法或改进现有方法
- **应用创新性**：是否有新的应用场景或实际价值
- **整体创新水平**：综合创新程度

### 4. 参考文献 (References)
标准学术引用格式：
```
[1] 作者. 标题. 期刊/会议名, 年份.
[2] 作者. 标题. 期刊/会议名, 年份.
...
```

## 系统架构

### 核心组件

1. **Node 类**：树搜索节点
   - 存储创新性评价内容
   - 管理父子节点关系
   - 维护访问次数和价值评估

2. **Reflection 类**：反思结果
   - 存储反思内容
   - 多维度评分
   - 判断是否达到高质量标准

3. **PaperInfo 类**：论文信息
   - 存储论文元数据
   - 生成引用字符串
   - 格式化上下文信息

### 工作流程

```
1. 输入：论文标题、摘要、相关论文列表
   ↓
2. 生成初始创新性评价
   ↓
3. 反思评价质量
   ↓
4. 检查是否满足质量标准
   ├─ 是 → 生成最终报告
   └─ 否 → 改进评价 → 返回步骤3
   ↓
5. 输出：完整的创新性评价报告
```

## 反思维度

系统从以下四个维度对创新性评价进行反思：

### 1. 对比充分性 (Comparison Adequacy)
- 是否充分对比了相关研究工作
- 是否正确识别了与现有工作的差异
- 引用是否恰当、完整

### 2. 创新性识别准确性 (Innovation Accuracy)
- 识别的创新点是否真实存在
- 是否有遗漏的重要创新
- 是否存在夸大或错误的创新声明

### 3. 论证逻辑性 (Argumentation Logic)
- 评价逻辑是否清晰
- 论证是否有充分依据
- 结论是否合理

### 4. 引用规范性 (Citation Normative)
- 引用格式是否正确 [序号]
- 引用是否与论述匹配
- 是否有遗漏的重要引用

## 配置参数

### LLM 配置（lats.py）

```python
llm = ChatOpenAI(
    model="qwen3:30b",  # 使用的模型
    base_url="http://localhost:11434/v1",  # Ollama 服务端点
    api_key="ollama",
    temperature=0.3,  # 温度参数
    max_tokens=9000,  # 最大输出token数
)
```

### 树搜索配置

```python
config = {
    "configurable": {
        "N": 3,  # 每轮生成的候选数量
        "max_iterations": 5  # 最大迭代次数
    }
}
```

## 依赖项

```bash
pip install langchain langchain-openai langgraph pydantic backoff PrettyPrint
```

## 注意事项

1. **模型要求**：需要使用支持工具调用（tool calling）的模型，如 Qwen、GPT-4 等
2. **相关论文数量**：建议使用 5-10 篇最相关的论文，过多可能影响效果
3. **迭代次数**：默认最大迭代 5 次，可根据需要调整
4. **Ollama 服务**：确保 Ollama 服务在 `localhost:11434` 运行

## 示例输出

```markdown
# 创新性评价报告

## 1. 创新点概述
本论文提出了以下主要创新贡献：
1. 提出了新的 Transformer-CNN 混合架构
2. 设计了自适应注意力机制
3. 在多个基准数据集上取得 SOTA 性能

## 2. 与现有研究的对比
与现有工作 [1][2] 相比，本论文的主要改进包括：
- 克服了 Transformer 计算复杂度高的局限
- 解决了 CNN 长距离依赖建模能力不足的问题
...

## 3. 创新性评估
- 理论创新性：★★★★☆
- 方法创新性：★★★★★
- 应用创新性：★★★☆☆
- 整体创新水平：★★★★☆

## 4. 参考文献
[1] Vaswani et al. Attention Is All You Need. NeurIPS, 2017.
[2] He et al. Deep Residual Learning for Image Recognition. CVPR, 2016.
...
```

## 故障排除

### 问题：无法连接到 Ollama
- 检查 Ollama 服务是否运行：`curl http://localhost:11434/api/tags`
- 确认模型已安装：`ollama list`

### 问题：创新性评价质量不高
- 增加相关论文数量
- 提高最大迭代次数
- 检查 LLM 模型能力

### 问题：引用格式不正确
- 检查相关论文数据是否完整
- 确认论文信息包含 title, authors, venue, year 字段

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本系统。

## 许可证

MIT License
