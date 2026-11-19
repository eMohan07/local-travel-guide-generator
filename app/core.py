"""Your code. One stub per milestone; build them in order and keep each one small."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.dataset import (
    CITY_NAME,
    TRAVEL_RECOMMENDATIONS_DATA,
    format_instruction_prompt,
    format_recommendation_response,
    format_dataset_item,
    prepare_travel_dataset,
)

# --- Milestone 1: Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.
# Hint: Prepare your dataset by formatting travel interests and recommendations into suitable input-output pairs for language model training.
def milestone_1(save_path: Optional[str | Path] = "data/travel_dataset.json") -> List[Dict[str, Any]]:
    """Milestone 1: Prepare and format a dataset of travel interests and recommendations for Zephyria.
    
    Returns:
        List of formatted training examples with 'input', 'output', and 'text' suitable
        for fine-tuning a language model with LoRA.
    """
    return prepare_travel_dataset(output_path=save_path)


from app.lora_trainer import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL,
    get_model_and_tokenizer,
    inject_lora_adapter,
    train_lora_adapter,
)

# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
# Hint: Apply the lfllm-lflora technique to inject and train a LoRA adapter on a pre-trained LLM.
def milestone_2(
    model_name: str = DEFAULT_BASE_MODEL,
    output_dir: str = DEFAULT_ADAPTER_DIR,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 5e-4,
    r: int = 8,
    lora_alpha: int = 16,
    dataset: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Milestone 2: Inject and train a LoRA adapter on a pre-trained LLM.
    
    Uses the lfllm-lflora technique to freeze base LLM weights and insert low-rank
    trainable matrices into attention projection layers.
    """
    # 1. Ensure dataset from Milestone 1 is available
    if dataset is None:
        dataset = milestone_1()

    # 2. Load pre-trained base model and tokenizer
    base_model, tokenizer = get_model_and_tokenizer(model_name)

    # 3. Inject LoRA adapter (freezes base model parameters)
    peft_model, param_stats = inject_lora_adapter(
        base_model=base_model,
        r=r,
        lora_alpha=lora_alpha,
    )

    # 4. Train LoRA adapter on travel guide dataset
    training_report = train_lora_adapter(
        model=peft_model,
        tokenizer=tokenizer,
        dataset_samples=dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    return {
        "status": "success",
        "model": peft_model,
        "tokenizer": tokenizer,
        "param_stats": param_stats,
        "training_report": training_report,
        "adapter_dir": str(output_dir),
    }


from app.generator import generate_mini_guide, load_inference_model

# --- Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
# Hint: Use Streamlit to build an interactive interface for users to input details and view generated guides.
def milestone_3(
    destination: str = "Lantern Quarter",
    interests: Union[List[str], str] = "Street Food, Night Markets, Live Acoustic Music",
    traveler_profile: str = "Solo foodie",
    adapter_path: Optional[str] = "models/lora_adapter",
) -> str:
    """Milestone 3: Generate a personalized mini-guide for tourist interests and destination.
    
    Generates tailored local recommendations using the fine-tuned LoRA model.
    """
    return generate_mini_guide(
        destination=destination,
        interests=interests,
        traveler_profile=traveler_profile,
        adapter_path=adapter_path,
    )


# --- Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
# Hint: Qualitatively assess the generated recommendations for relevance, appeal, and personalization.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Utilize the lfllm-lflora technique to merge the trained LoRA adapter with the base model for efficient inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
