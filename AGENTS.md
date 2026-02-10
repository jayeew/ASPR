# AGENTS.md - Agentic Coding Guidelines

## Project Overview

ASPR (Academic Scientific Paper Review) - A Python-based system for automated academic paper review generation using LLMs, retrieval-augmented generation (RAG), and fine-tuning.

## Build/Lint/Test Commands

### Running Python Files
```bash
# Run main modules directly
python open_scholar.py --s2_api_key <key> --large_model_port 38011
python lats.py  # Runs innovation evaluation test
python graph_rag.py  # Runs GraphRAG insert and query
python pdf_downloader.py  # Runs test_acl_download()
```

### Running Single Tests
No formal test framework - tests are inline:
```bash
# Test specific functionality
python -c "from pdf_downloader import ACLPDFDownloader; d = ACLPDFDownloader(); d.test_acl_download()"
python test.py  # Test OpenAI client connectivity
```

### Model Training
```bash
# Full fine-tuning with DeepSpeed
bash train_sft_qwen.sh

# LoRA fine-tuning with QLoRA
bash train_sft_lora_qwen.sh
```

### Linting/Formatting
```bash
pip install black ruff mypy
black *.py        # Format code
ruff check *.py   # Lint check
mypy *.py --ignore-missing-imports  # Type check
```

## Code Style Guidelines

### Import Order
1. `__future__` imports (if needed)
2. Standard library imports
3. Third-party imports
4. Local module imports

```python
from __future__ import annotations
import os
import json
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from prompts import generation_instance_prompts
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `Reviewer`, `OpenScholar`, `ACLPDFDownloader`)
- **Functions/Variables**: `snake_case` (e.g., `keywords_extract`, `paper_recalled`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SYSTEM_PROMPT`, `EMBEDDING_MODEL`)
- **Private Methods**: `_leading_underscore` (e.g., `_paper_download`, `_generate_review`)

### Type Hints
- Use type hints for all function signatures
- Use `Optional[X]` for nullable types
- Use `Literal["a", "b"]` for string enums
- Use `List[Type]`, `Dict[KeyType, ValueType]` from typing module

```python
def retrieval_recall(query: str, reference: list[str]) -> tuple[list[str], list[float]]: ...
def download_acl_pdf(self, pdf_url: str, save_dir: str = ".") -> Optional[str]: ...
```

### Error Handling
- Use specific exceptions, avoid bare `except:`
- Log errors with context using `print()` or `lats_logging()`
- Provide fallback values for non-critical failures
- Use `@backoff.on_exception` for retry logic

```python
try:
    keywords_list = [kw.strip() for kw in keywords.split(",")]
except Exception as e:
    print(f"Error parsing keywords: {e}")
    return []

@backoff.on_exception(backoff.expo, IndexError, max_tries=5)
def fragile_operation(): ...
```

### Documentation
- Use docstrings for classes and public methods
- Include Args/Returns sections
- Support both English and Chinese in docstrings
- Use triple-quoted strings for multi-line prompts

```python
def download_acl_pdf(self, pdf_url: str, save_dir: str = ".") -> str:
    """Download PDF from ACL Anthology
    
    Args:
        pdf_url: URL of the PDF file
        save_dir: Directory to save file (default: current dir)
    
    Returns:
        Path to saved file
    """
```

### Code Structure
- Keep functions under 50 lines when possible
- Use classes for related functionality
- Group related functions together
- Use `if __name__ == '__main__':` blocks for CLI entry points
- Organize code with section comments:

```python
# ============================================================================
# Configuration
# ============================================================================

# ============================================================================
# Data Models
# ============================================================================
```

### String Formatting
- Prefer f-strings for simple interpolation
- Use `.format_map()` for complex templates with dicts
- Use triple-quoted strings for multi-line prompts

```python
prompt = f"Hello, {name}!"
complex_prompt = template.format_map({"key": value})
```

### Async/Await
- Use `async def` for I/O-bound operations
- Use `asyncio.run()` or `await` in async contexts

```python
async def ollama_model_if_cache(prompt, **kwargs) -> str:
    ollama_client = ollama.AsyncClient()
    response = await ollama_client.chat(...)
    return response["message"]["content"]
```

## Dependencies

Key dependencies (no requirements.txt - add manually):
- `openai` - OpenAI API client
- `langchain`, `langchain-openai`, `langgraph` - LLM orchestration
- `pydantic` - Data validation with `BaseModel`, `Field`, `field_validator`
- `requests` - HTTP requests
- `pypdf` - PDF text extraction with `PdfReader`
- `FlagEmbedding` - BGE embeddings (`BGEM3FlagModel`) and reranking (`FlagReranker`)
- `nano-graphrag` - Graph-based RAG with `GraphRAG`, `QueryParam`
- `ollama` - Local LLM integration
- `datasets`, `huggingface-hub` - Dataset handling
- `openrlhf`, `deepspeed` - Model training
- `backoff` - Retry logic with `@backoff.on_exception`
- `typing_extensions` - Extended typing utilities

## Project Structure

```
ASPR/
├── open_scholar.py      # Main reviewer class, Semantic Scholar search, paper retrieval
├── lats.py             # Tree search with reflection (LATS) for innovation evaluation
├── graph_rag.py        # GraphRAG integration with Ollama
├── pdf_downloader.py   # ACL Anthology PDF downloader with anti-bot handling
├── prompts.py          # LLM prompt templates for summarization, keyword extraction
├── hfdata_builder.py   # HuggingFace dataset builder
├── test.py             # OpenAI client test script
├── train_sft_qwen.sh   # Full fine-tuning script
└── train_sft_lora_qwen.sh  # LoRA fine-tuning script
```

## Development Notes

1. **No formal testing framework** - Test code is inline
2. **No CI/CD** - Run scripts manually
3. **Local LLM first** - Default to Ollama at `localhost:11434`
4. **API keys** - Store in args/env vars, not hardcoded
5. **File I/O** - Use `pathlib.Path` for cross-platform compatibility
6. **Logging** - Use simple `print()` or project-specific `lats_logging()`
7. **No requirements.txt** - Install dependencies manually

## Common Patterns

### LLM Client Setup (LangChain)
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen3:30b",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.3,
    max_tokens=9000,
)
```

### OpenAI Client Setup (Direct)
```python
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url=f'http://localhost:{port}/v1',
)
```

### GraphRAG Pattern
```python
from nano_graphrag import GraphRAG, QueryParam

rag = GraphRAG(
    working_dir=WORKING_DIR,
    best_model_func=ollama_model_if_cache,
    embedding_func=ollama_embedding,
)
rag.insert(text)
response = rag.query(query, param=QueryParam(mode="global"))
```

### Pydantic Data Model
```python
from pydantic import BaseModel, Field, field_validator

class PaperInfo(BaseModel):
    index: int = Field(description="文献序号")
    title: str = Field(description="论文标题")
    
    @field_validator('field_name', mode='before')
    @classmethod
    def extract_(cls, v):
        return str(v)
```

### Retry with Backoff
```python
import backoff

@backoff.on_exception(backoff.expo, IndexError, max_tries=5)
def fragile_operation(): ...
```
