"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

import json
from pathlib import Path
import pytest
from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5
from app.dataset import CITY_NAME, format_instruction_prompt, format_recommendation_response

# Milestone 1: Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.
def test_milestone_1(tmp_path):
    test_json_path = tmp_path / "travel_dataset.json"
    dataset = milestone_1(save_path=test_json_path)

    # 1. Dataset should be non-empty and reasonably sized (at least 15 diverse pairs)
    assert isinstance(dataset, list)
    assert len(dataset) >= 15

    # 2. Check that the dataset was saved properly to disk
    assert test_json_path.exists()
    saved_data = json.loads(test_json_path.read_text(encoding="utf-8"))
    assert len(saved_data) == len(dataset)

    # 3. Inspect every training example structure and content
    required_keys = {"id", "city", "district", "interests", "traveler_profile", "input", "output", "text"}
    districts = set()

    for item in dataset:
        assert required_keys.issubset(item.keys())
        assert item["city"] == CITY_NAME
        assert isinstance(item["interests"], list) and len(item["interests"]) >= 2
        assert len(item["district"]) > 0
        districts.add(item["district"])

        # Check prompt formatting
        assert "### Instruction:" in item["input"]
        assert "### Input:" in item["input"]
        assert "### Response:" in item["input"]
        assert item["district"] in item["input"]
        for interest in item["interests"]:
            assert interest in item["input"]

        # Check response formatting
        assert "🌟 **Top Highlights" in item["output"]
        assert "🤫 **Local Secret" in item["output"]
        assert "💡 **Insider Tip" in item["output"]

        # Check full training sequence for causal LM
        assert item["text"].startswith(item["input"])
        assert item["text"].endswith(item["output"])

    # 4. Verify diversity of districts
    assert len(districts) >= 10


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def test_milestone_2(tmp_path):
    adapter_dir = tmp_path / "test_lora_adapter"
    sample_data = milestone_1()[:2]  # Use 2 samples for fast test execution

    result = milestone_2(
        output_dir=str(adapter_dir),
        epochs=1,
        batch_size=2,
        dataset=sample_data,
        r=4,
        lora_alpha=8,
    )

    # 1. Verify successful output structure
    assert result["status"] == "success"
    assert "model" in result
    assert "tokenizer" in result
    assert "param_stats" in result

    # 2. Check parameter efficiency (LoRA trains < 1% of total parameters)
    stats = result["param_stats"]
    assert stats["trainable_params"] > 0
    assert stats["all_params"] > 100_000_000  # GPT-2 has ~124M parameters
    assert stats["trainable_percentage"] < 1.0  # Typically ~0.1% to 0.5%
    assert stats["lora_rank"] == 4

    # 3. Check base model parameters are frozen while LoRA parameters are trainable
    model = result["model"]
    lora_params_found = False
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_params_found = True
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False
    assert lora_params_found

    # 4. Check saved adapter files
    assert adapter_dir.exists()
    assert (adapter_dir / "adapter_config.json").exists()
    has_weights = (
        (adapter_dir / "adapter_model.safetensors").exists()
        or (adapter_dir / "adapter_model.bin").exists()
    )
    assert has_weights

    # 5. Check training report & loss history
    report = result["training_report"]
    assert report["epochs"] == 1
    assert len(report["loss_history"]) == 1
    assert isinstance(report["loss_history"][0], float)
    assert report["loss_history"][0] > 0.0


# Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
def test_milestone_3():
    # 1. Test the core generation function
    guide = milestone_3(
        destination="Lantern Quarter",
        interests=["Street Food", "Night Markets"],
        traveler_profile="Solo foodie looking for evening street snacks",
    )
    assert isinstance(guide, str)
    assert len(guide.strip()) > 20

    # 2. Test Streamlit app rendering using AppTest
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app/main.py", default_timeout=60)
    at.run(timeout=60)
    assert not at.exception

    # Check input widgets exist
    assert len(at.selectbox) >= 1
    assert len(at.multiselect) >= 1
    assert len(at.button) >= 1


from app.evaluation import BENCHMARK_EVALUATION_CASES, evaluate_single_recommendation

# Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
def test_milestone_4(tmp_path):
    report_file = tmp_path / "evaluation_report.json"
    
    # 1. Evaluate single recommendation directly
    sample_guide = (
        "Welcome to Zephyria's Lantern Quarter!\n\n"
        "🌟 **Top Highlights & Curated Experiences:**\n"
        "- Canal-side Night Bazaar with floating amber silk lanterns\n"
        "- Dragonfly Dumpling Den for spiced plum street dumplings\n"
        "- Acoustic Alleyway with live buskers\n\n"
        "🤫 **Local Secret / Hidden Gem:**\nThe Gilded Kettle basement tea parlor.\n\n"
        "⏰ **Best Time to Visit:**\n8:00 PM to midnight.\n\n"
        "💡 **Insider Tip:**\nBring exact copper change for quick discounts."
    )
    single_eval = evaluate_single_recommendation(
        guide_text=sample_guide,
        destination="Lantern Quarter",
        interests=["Street Food", "Night Markets"],
        traveler_profile="Solo foodie seeking vibrant evening street eats",
        expected_themes=["lantern", "dumpling", "night", "food"],
    )
    assert single_eval["scores"]["relevance"] >= 3.0
    assert single_eval["scores"]["appeal"] >= 3.0
    assert single_eval["scores"]["personalization"] >= 3.0
    assert single_eval["scores"]["overall"] >= 3.0
    assert single_eval["passed"] is True

    # 2. Run full evaluation across benchmark test cases
    test_cases = BENCHMARK_EVALUATION_CASES[:2]
    report = milestone_4(test_cases=test_cases, save_report_path=report_file)

    assert report["status"] == "success"
    assert report["total_cases_evaluated"] >= 2
    assert report["pass_rate_percentage"] >= 60.0

    avg_scores = report["average_scores"]
    for metric in ["relevance", "appeal", "personalization", "overall"]:
        assert metric in avg_scores
        assert 1.0 <= avg_scores[metric] <= 5.0
        assert avg_scores[metric] >= 3.0

    # 3. Check report saved to disk
    assert report_file.exists()
    saved = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved["total_cases_evaluated"] == report["total_cases_evaluated"]


from peft import PeftModel
from transformers import AutoModelForCausalLM

# Milestone 5: Implement adapter merging for inference.
def test_milestone_5(tmp_path):
    test_adapter_dir = tmp_path / "test_lora_adapter"
    test_merged_dir = tmp_path / "test_merged_model"

    # 1. Prepare a small LoRA adapter for fast merging test
    sample_data = milestone_1()[:2]
    milestone_2(
        output_dir=str(test_adapter_dir),
        epochs=1,
        batch_size=2,
        dataset=sample_data,
        r=4,
        lora_alpha=8,
    )

    # 2. Run milestone 5 adapter merging
    result = milestone_5(
        adapter_path=str(test_adapter_dir),
        output_dir=str(test_merged_dir),
    )

    # 3. Verify successful merge and PEFT wrapper unloading
    assert result["status"] == "success"
    assert result["is_peft_model"] is False
    assert not isinstance(result["model"], PeftModel)

    # 4. Check exported model weights and configs
    assert test_merged_dir.exists()
    assert (test_merged_dir / "config.json").exists()
    assert (test_merged_dir / "merge_report.json").exists()
    has_weights = (
        (test_merged_dir / "model.safetensors").exists()
        or (test_merged_dir / "pytorch_model.bin").exists()
    )
    assert has_weights

    # 5. Check standalone reload using standard HuggingFace AutoModel (no PEFT)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        str(test_merged_dir),
        local_files_only=True,
    )
    assert not isinstance(loaded_model, PeftModel)
    assert sum(p.numel() for p in loaded_model.parameters()) > 100_000_000

