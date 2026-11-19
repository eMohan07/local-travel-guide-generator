"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

import json
from pathlib import Path
import pytest
from app.core import milestone_1, milestone_2, milestone_3
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

    at = AppTest.from_file("app/main.py", default_timeout=30)
    at.run(timeout=30)
    assert not at.exception

    # Check input widgets exist
    assert len(at.selectbox) >= 1
    assert len(at.multiselect) >= 1
    assert len(at.button) >= 1

    # Simulate user clicking Generate button
    at.button[0].click().run(timeout=30)
    assert not at.exception


# Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

