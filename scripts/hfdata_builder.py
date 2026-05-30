import argparse
import os
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from huggingface_hub import create_repo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "paper_reconstruction_sft"
SYSTEM_PROMPT = """You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers.
Given a paper, you should quickly provide the review results.
"""


def serialize_chat(messages: List[Dict[str, str]]) -> str:
    """
    Serialize chat messages into the token format used by the SFT dataset.

    Args:
        messages: List of {"role": "...", "content": "..."} messages.

    Returns:
        Serialized chat string.
    """
    text = ""
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            text += f"<|system|>\n{content}\n"
        elif role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"
    return text


def build_dataset(paper_dir: Path, recon_dir: Path) -> Dataset:
    """
    Build the paper reconstruction SFT dataset from markdown files.

    Args:
        paper_dir: Directory containing original paper markdown files.
        recon_dir: Directory containing reconstructed markdown files.

    Returns:
        HuggingFace Dataset ready for saving or upload.
    """
    data = []
    for recon_file in recon_dir.glob("*.md"):
        file_id = recon_file.stem
        paper_file = paper_dir / f"{file_id[:-2]}.md"
        if not paper_file.exists():
            continue

        input_text = paper_file.read_text(encoding="utf-8")
        output_text = recon_file.read_text(encoding="utf-8")
        data.append(
            {
                "inputs": serialize_chat(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": input_text},
                    ]
                ),
                "outputs": serialize_chat([{"role": "assistant", "content": output_text}]),
            }
        )
    return Dataset.from_list(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the paper reconstruction SFT dataset.")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_SOURCE_ROOT / "paper")
    parser.add_argument("--recon-dir", type=Path, default=DEFAULT_SOURCE_ROOT / "reconstruction")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default="jayeew/paper-reconstruction-sft")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(args.paper_dir, args.recon_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))
    print(f"Saved {len(dataset)} examples to {args.output_dir}")

    if args.push_to_hub:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("Set HF_TOKEN before using --push-to-hub.")
        create_repo(args.repo_id, repo_type="dataset", token=token, exist_ok=True)
        dataset.push_to_hub(args.repo_id, token=token)


if __name__ == "__main__":
    main()
