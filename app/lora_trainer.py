"""LoRA (Low-Rank Adaptation) adapter injection and training module.

Applies the lfllm-lflora technique to inject trainable low-rank decomposition
matrices into pre-trained causal language models (e.g. GPT-2) while freezing base weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)

DEFAULT_BASE_MODEL = "gpt2"
DEFAULT_ADAPTER_DIR = "models/lora_adapter"


class TravelGuideDataset(Dataset):
    """PyTorch Dataset for formatting and tokenizing travel guide training pairs."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings: List[Dict[str, torch.Tensor]] = []

        # Ensure padding token is defined (GPT-2 doesn't have a default pad token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for item in samples:
            # Use the pre-formatted text if available, or concatenate input and output
            text = item.get("text") or f"{item.get('input', '')}{item.get('output', '')}"
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)

            # Mask padding tokens in labels with -100 so they don't contribute to loss
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.encodings.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encodings[idx]


def get_model_and_tokenizer(
    model_name_or_path: str = DEFAULT_BASE_MODEL,
    local_files_only: Optional[bool] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model and tokenizer, prioritizing local cache to avoid network delays."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True if local_files_only is None else local_files_only,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            local_files_only=True if local_files_only is None else local_files_only,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    return model, tokenizer


def inject_lora_adapter(
    base_model: PreTrainedModel,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    fan_in_fan_out: Optional[bool] = None,
) -> Tuple[PeftModel, Dict[str, Any]]:
    """Inject a LoRA adapter into the base language model.
    
    Freezes all base model weights and inserts low-rank decomposition matrices (A and B)
    into target attention/projection layers.
    """
    if target_modules is None:
        # Default target modules for GPT-2 Conv1D attention layers
        target_modules = ["c_attn"]

    # Detect if target module uses Conv1D (common in GPT-2)
    if fan_in_fan_out is None:
        is_gpt2 = "gpt2" in getattr(base_model.config, "model_type", "").lower()
        fan_in_fan_out = is_gpt2

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        fan_in_fan_out=fan_in_fan_out,
    )

    peft_model = get_peft_model(base_model, lora_config)

    # Compute trainable vs frozen parameter statistics
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    param_stats = {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": 100.0 * trainable_params / all_params,
        "lora_rank": r,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
    }

    return peft_model, param_stats


def train_lora_adapter(
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_samples: List[Dict[str, Any]],
    output_dir: Union[str, Path] = DEFAULT_ADAPTER_DIR,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 5e-4,
    max_length: int = 256,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Train the injected LoRA adapter on travel guide input-output pairs.
    
    Only the low-rank adapter parameters are updated; the base model parameters remain frozen.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    dataset = TravelGuideDataset(
        samples=dataset_samples,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    loss_history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        loss_history.append(avg_loss)

    # Save trained adapter and tokenizer configs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Save training report
    report = {
        "status": "success",
        "epochs": epochs,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "adapter_dir": str(output_path),
    }
    with open(output_path / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
