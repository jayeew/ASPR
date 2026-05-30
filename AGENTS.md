# AGENTS.md - Agentic Coding Guidelines

## Project Overview

ASPR (Academic Scientific Paper Review) - Python system for automated academic paper review generation using LLMs, RAG, and fine-tuning.

## Build/Lint/Test Commands

```bash
# Run main modules
python -m aspr.open_scholar --s2_api_key <key> --large_model_port 38011
python -m aspr.lats
python -m aspr.graph_rag insert "paper/reference text"
python -m aspr.graph_rag query "What is novel?"
python -m aspr.pdf_downloader

# Inline tests
python -c "from aspr.pdf_downloader import test_acl_download; test_acl_download()"
python tests/test_openai_client.py

# Model training
bash scripts/train_sft_qwen.sh        # Full fine-tuning
bash scripts/train_sft_lora_qwen.sh   # LoRA fine-tuning

# Linting
pip install black ruff mypy
black aspr scripts tests experiments
ruff check aspr scripts tests experiments
mypy aspr scripts tests experiments --ignore-missing-imports
```

## Code Style Guidelines

### Import Order
1. `__future__` imports
2. Standard library
3. Third-party
4. Local modules

```python
from __future__ import annotations
import os
import json
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from aspr.prompts import generation_instance_prompts
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `Reviewer`, `OpenScholar`)
- **Functions/Variables**: `snake_case` (e.g., `keywords_extract`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SYSTEM_PROMPT`)
- **Private Methods**: `_leading_underscore`

### Type Hints
Required for all function signatures. Use `Optional[X]`, `List[Type]`, `Dict[K, V]`.

### Error Handling
- Specific exceptions, no bare `except:`
- Log with `print()` or `lats_logging()`
- Use `@backoff.on_exception` for retries

### Documentation
Docstrings for classes/public methods with Args/Returns. Support English/Chinese.

### Code Structure
- Functions under 50 lines preferred
- Section comments: `# ============================================================================`
- `if __name__ == '__main__':` for CLI entry points

### String Formatting
- f-strings for simple cases
- `.format_map()` for complex templates

### Async/Await
Use for I/O-bound operations with `async def` and `asyncio`.

## Dependencies

Key packages: `openai`, `langchain`, `pydantic`, `requests`, `pypdf`, `FlagEmbedding`, `nano-graphrag`, `ollama`, `datasets`, `backoff`, `typing_extensions`, `deepspeed`.

## Project Structure

```
ASPR/
├── aspr/
│   ├── open_scholar.py     # Main reviewer, Semantic Scholar search
│   ├── lats.py            # Tree search with reflection (LATS)
│   ├── graph_rag.py       # GraphRAG with Ollama
│   ├── pdf_downloader.py  # ACL Anthology PDF downloader
│   └── prompts.py         # LLM prompt templates
├── scripts/               # Download, dataset build, and training scripts
├── tests/                 # Lightweight manual checks
├── data/                  # Cached JSON and local datasets
├── outputs/               # Generated artifacts
└── experiments/           # KG validation and perturbation experiments
```

## Development Notes

- No formal test framework - inline tests only
- No CI/CD - manual execution
- Default LLM: Ollama at `localhost:11434`
- Store API keys in args/env vars, never hardcoded
- Use `pathlib.Path` for file I/O

## Common Patterns

### LLM Client (LangChain)
```python
llm = ChatOpenAI(
    model="qwen3:30b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.3,
    max_tokens=9000,
)
```

### Pydantic Model
```python
class PaperInfo(BaseModel):
    index: int = Field(description="文献序号")
    title: str = Field(description="论文标题")
```

### GraphRAG
```python
rag = GraphRAG(working_dir=WORKING_DIR, best_model_func=ollama_model_if_cache)
rag.insert(text)
response = rag.query(query, param=QueryParam(mode="global"))
```

### Retry with Backoff
```python
@backoff.on_exception(backoff.expo, IndexError, max_tries=5)
def fragile_operation(): ...
```
