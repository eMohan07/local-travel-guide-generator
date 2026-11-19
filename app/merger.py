"""Adapter merging module for efficient inference.

Applies the lfllm-lflora technique to permanently merge trained LoRA adapter weights
(W = W0 + (alpha / r) * B * A) with the base model weights, eliminating the LoRA wrapper
and enabling zero-latency-overhead inference with standard PreTrainedModel architectures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import PeftModel

from app.dataset import CITY_NAME, format_instruction_prompt
from app.lora_trainer import DEFAULT_ADAPTER_DIR, DEFAULT_BASE_MODEL, get_model_and_tokenizer

DEFAULT_MERGED_DIR = "models/merged_model"


def merge_lora_adapter(
    adapter_path: Union[str, Path] = DEFAULT_ADAPTER_DIR,
    output_dir: Union[str, Path] = DEFAULT_MERGED_DIR,
    base_model_name: str = DEFAULT_BASE_MODEL,
    safe_merge: bool = True,
    device: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Dict[str, Any]]:
    """Merge trained LoRA adapter weights directly into the base language model.
    
    Fuses delta weights (W0 + (alpha/r)*BA) and unloads PEFT layers, returning
    a standard PreTrainedModel suitable for production serving without PEFT dependencies.
    """
    adapter_path = Path(adapter_path)
    output_path = Path(output_dir)

    # 1. Load base model and tokenizer
    base_model, tokenizer = get_model_and_tokenizer(base_model_name, local_files_only=True)

    # 2. Attach LoRA adapter
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No LoRA adapter found at '{adapter_path}'. Train an adapter first (Milestone 2)."
        )

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    # 3. Fuse adapter weights into base weights and unload the PEFT wrapper
    merged_model = peft_model.merge_and_unload(safe_merge=safe_merge)

    if device is not None:
        merged_model.to(device)

    # 4. Save merged model, tokenizer, and metadata
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    report = {
        "status": "success",
        "technique": "lfllm-lflora adapter merging",
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "output_dir": str(output_path),
        "merged_model_type": type(merged_model).__name__,
        "is_peft_model": isinstance(merged_model, PeftModel),
        "total_parameters": sum(p.numel() for p in merged_model.parameters()),
    }

    with open(output_path / "merge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return merged_model, tokenizer, report


def generate_with_merged_model(
    destination: str,
    interests: list[str] | str,
    traveler_profile: str = "",
    merged_model_dir: Union[str, Path] = DEFAULT_MERGED_DIR,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[str] = None,
) -> str:
    """Run inference using the merged base model without PEFT runtime overhead."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    merged_path = str(merged_model_dir)

    # Load standalone merged model using standard transformers (no PEFT required)
    tokenizer = AutoTokenizer.from_pretrained(merged_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(merged_path, local_files_only=True)
    model.to(device)
    model.eval()

    prompt = format_instruction_prompt(
        destination=destination,
        interests=interests,
        traveler_profile=traveler_profile,
        city=CITY_NAME,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return generated_text
