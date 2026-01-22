
from datasets import Dataset
from pathlib import Path

def serialize_chat(messages):
    """
    messages: List[{"role": "...", "content": "..."}]
    """
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "system":
            text += f"<|system|>\n{content}\n"
        elif role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"
    return text

SYSTEM_PROMPT = """You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. 
Given a paper, you should quickly provide the review results.
"""

paper_dir = Path("../dataset/paper")
recon_dir = Path("../dataset/reconstruction")

data = []

for recon_file in recon_dir.glob("*.md"):
    file_id = recon_file.stem
    paper_file = paper_dir / f"{file_id[:-2]}.md"
    # print(paper_file)
    if not paper_file.exists():
        continue

    input_text = recon_file.read_text(encoding="utf-8")
    output_text = recon_file.read_text(encoding="utf-8")

    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]

    output_messages = [
        {"role": "assistant", "content": output_text}
    ]

    data.append({
        "inputs": serialize_chat(input_messages),
        "outputs": serialize_chat(output_messages),
    })

dataset = Dataset.from_list(data)
dataset.save_to_disk("paper_reconstruction_sft")

repo_id = "jayeew/paper-reconstruction-sft"  # 替换为您的用户名

import os
from huggingface_hub import HfApi, create_repo, login
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
login()
try:
    # 如果仓库不存在，先创建
    create_repo(repo_id, repo_type="dataset")
except:
    pass  # 仓库已存在

# 使用 push_to_hub 方法上传
dataset.push_to_hub(
    repo_id,
    token=""
)

