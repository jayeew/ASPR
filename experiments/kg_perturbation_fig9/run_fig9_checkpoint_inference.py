"""Run a saved ASPR-Qwen checkpoint for the Fig.9 case output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig9.build_fig9_case import (  # noqa: E402
    ARTICLE_LABEL,
    CASE_ID,
    DEFAULT_MARKDOWN_ROOT,
    DEFAULT_OUTPUT_DIR,
    DOI,
    TITLE,
    YEAR,
    atomic_write_text,
    read_json_if_present,
    write_json,
)


DEFAULT_CHECKPOINT_PATH = Path("/home/jayee/workspace/checkpoint/qwen-0.6b-review")


def parse_checkpoint_json(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object from a checkpoint response."""
    text = str(raw_text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _list_from_payload(payload: Mapping[str, Any], key: str, fallback: Sequence[str]) -> list[str]:
    value = payload.get(key)
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if items:
            return items
    if isinstance(value, str) and value.strip():
        pieces = [part.strip(" -\t") for part in re.split(r"\n|;", value) if part.strip(" -\t")]
        if pieces:
            return pieces
    return list(fallback)


def normalize_checkpoint_review_payload(payload: Mapping[str, Any], *, raw_text: str) -> dict[str, Any]:
    """Normalize a checkpoint response to the Fig.9 checkpoint-output contract."""
    summary = str(payload.get("summary_judgement") or payload.get("summary") or "").strip()
    if not summary:
        summary = "Checkpoint-generated ASPR-Qwen review output; see raw_checkpoint_text for the full model response."
    strengths = _list_from_payload(
        payload,
        "major_strengths",
        ["Checkpoint response identifies manuscript strengths but did not emit a structured strengths list."],
    )
    concerns = _list_from_payload(
        payload,
        "major_concerns",
        ["Checkpoint response identifies manuscript concerns but did not emit a structured concerns list."],
    )
    return {
        "case_id": str(payload.get("case_id") or CASE_ID),
        "doi": DOI,
        "output_origin": "checkpoint_generated_aspr_qwen_output",
        "pipeline_ready": True,
        "checkpoint_invoked": True,
        "summary_judgement": summary,
        "major_strengths": strengths,
        "major_concerns": concerns,
        "reviewer_style_recommendation": str(
            payload.get("reviewer_style_recommendation")
            or payload.get("recommendation")
            or "Checkpoint output did not provide a separate recommendation."
        ),
        "raw_checkpoint_text": raw_text,
    }


def checkpoint_hash(checkpoint_path: Path) -> str:
    """Return a stable sha256 hash over key checkpoint files."""
    digest = hashlib.sha256()
    files = [
        path
        for path in sorted(checkpoint_path.rglob("*"))
        if path.is_file()
        and path.name
        in {
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "tokenizer_config.json",
        }
        or (path.is_file() and path.name.startswith("model-") and path.suffix == ".safetensors")
    ]
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under {checkpoint_path}")
    for path in files:
        digest.update(str(path.relative_to(checkpoint_path)).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _data_version() -> str:
    dataset_info = PROJECT_ROOT / "data" / "paper_reconstruction_sft" / "dataset_info.json"
    state = PROJECT_ROOT / "data" / "paper_reconstruction_sft" / "state.json"
    digest = hashlib.sha256()
    found = False
    for path in [dataset_info, state]:
        if path.exists():
            found = True
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    if not found:
        return "paper_reconstruction_sft:unversioned"
    return f"paper_reconstruction_sft:{digest.hexdigest()[:16]}"


def build_checkpoint_metadata(
    *,
    checkpoint_path: Path,
    prompt: str,
    decoding_config: Mapping[str, Any],
    seed: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build the metadata sidecar required by the Fig.9 checkpoint gate."""
    config = read_json_if_present(checkpoint_path / "config.json")
    return {
        "model_hash": checkpoint_hash(checkpoint_path),
        "training_config": {
            "base_model": str(config.get("_name_or_path") or config.get("model_type") or "unknown"),
            "adapter_or_checkpoint_path": str(checkpoint_path),
            "training_script": "scripts/train_sft_lora_qwen.sh or scripts/train_sft_qwen.sh",
            "hyperparameters": {
                "checkpoint_architecture": config.get("architectures", []),
                "dtype": config.get("dtype", config.get("torch_dtype", "")),
            },
        },
        "data_version": _data_version(),
        "prompt": prompt,
        "decoding_config": dict(decoding_config),
        "seed": int(seed),
        "runtime_seconds": float(runtime_seconds),
    }


def _clip(text: str, limit: int) -> str:
    clean = "\n".join(line.rstrip() for line in str(text or "").splitlines())
    return clean[:limit].strip()


def build_prompt(markdown_root: Path, *, max_chars: int = 8000) -> str:
    """Build a no-peer-review-leakage prompt for the Fig.9 checkpoint run."""
    paper_path = markdown_root / "paper" / f"{CASE_ID}.md"
    paper_text = paper_path.read_text(encoding="utf-8")
    paper_excerpt = _clip(paper_text, max_chars)
    return (
        "You are ASPR-Qwen, a scientific peer-review assistant. "
        "Use only the manuscript excerpt below; do not assume access to human peer review. "
        "Return one JSON object with keys: case_id, summary_judgement, "
        "major_strengths, major_concerns, reviewer_style_recommendation. "
        "major_strengths and major_concerns must be arrays of concise strings.\n\n"
        f"case_id: {CASE_ID}\n"
        f"title: {TITLE}\n"
        f"venue: {ARTICLE_LABEL}\n"
        f"year: {YEAR}\n"
        f"doi: {DOI}\n\n"
        "Manuscript excerpt:\n"
        f"{paper_excerpt}\n"
    )


def run_checkpoint_inference(
    *,
    checkpoint_path: Path,
    markdown_root: Path,
    output_dir: Path,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_input_chars: int,
) -> dict[str, Any]:
    """Run the checkpoint and save Fig.9 checkpoint output plus metadata."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prompt = build_prompt(markdown_root, max_chars=max_input_chars)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = prompt
    inputs = tokenizer(rendered, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    runtime = time.perf_counter() - start
    decoding_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "max_input_chars": max_input_chars,
    }
    metadata = build_checkpoint_metadata(
        checkpoint_path=checkpoint_path,
        prompt=prompt,
        decoding_config=decoding_config,
        seed=seed,
        runtime_seconds=runtime,
    )
    payload = normalize_checkpoint_review_payload(parse_checkpoint_json(raw_text), raw_text=raw_text)
    payload["checkpoint_metadata"] = metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "fig9_checkpoint_metadata.json", metadata)
    write_json(output_dir / "fig9_aspr_qwen_output.json", payload)
    atomic_write_text(output_dir / "fig9_checkpoint_raw_output.txt", raw_text.rstrip() + "\n")
    return {
        "output_path": str(output_dir / "fig9_aspr_qwen_output.json"),
        "metadata_path": str(output_dir / "fig9_checkpoint_metadata.json"),
        "raw_output_path": str(output_dir / "fig9_checkpoint_raw_output.txt"),
        "runtime_seconds": runtime,
        "model_hash": metadata["model_hash"],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--max-input-chars", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result = run_checkpoint_inference(
        checkpoint_path=args.checkpoint_path,
        markdown_root=args.markdown_root,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_input_chars=args.max_input_chars,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
