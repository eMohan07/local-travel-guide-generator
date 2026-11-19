"""Your code. One stub per milestone; build them in order and keep each one small."""

from pathlib import Path
from typing import Any, Dict, List, Optional

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


# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
# Hint: Apply the lfllm-lflora technique to inject and train a LoRA adapter on a pre-trained LLM.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
# Hint: Use Streamlit to build an interactive interface for users to input details and view generated guides.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
# Hint: Qualitatively assess the generated recommendations for relevance, appeal, and personalization.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Utilize the lfllm-lflora technique to merge the trained LoRA adapter with the base model for efficient inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
