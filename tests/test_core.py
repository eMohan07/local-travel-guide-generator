"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

import json
from pathlib import Path
import pytest
from app.core import milestone_1
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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

