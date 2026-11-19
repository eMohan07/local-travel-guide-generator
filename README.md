# Local Travel Guide Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/xyyqwrjzjf/badge.svg)](https://ask.thili.ai/b/xyyqwrjzjf)

This project, "Local Travel Guide Generator," provides a solution for the Travel & Hospitality domain. It enables travel agencies to fine-tune a language model using LoRA, creating concise and personalized local travel guides and recommendations for tourists based on their interests and chosen destination.

## What you are building

**Goal.** You will build a tool for travel agencies to generate personalized local travel guides by fine-tuning a language model with LoRA, adapting it to specific tourist interests and destinations.

**Done when.** You will have a trained LoRA adapter on a pre-trained LLM, capable of generating personalized local travel recommendations. The system will produce relevant and appealing mini-guides for tourists based on their input interests and destination, demonstrating the successful application of the LoRA fine-tuning technique.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.**  
      Hint: Prepare your dataset by formatting travel interests and recommendations into suitable input-output pairs for language model training.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the lfllm-lflora technique to inject and train a LoRA adapter on a pre-trained LLM.
- [ ] **3. Develop a Streamlit UI to input interests and destination, generating a mini-guide.**  
      Hint: Use Streamlit to build an interactive interface for users to input details and view generated guides.
- [ ] **4. Evaluate the relevance and appeal of the generated recommendations.**  
      Hint: Qualitatively assess the generated recommendations for relevance, appeal, and personalization.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Utilize the lfllm-lflora technique to merge the trained LoRA adapter with the base model for efficient inference.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the generation of guides for multiple destinations or interests simultaneously.
- Add a simple feedback mechanism in the UI for users to rate the generated recommendations.
- Experiment with different LoRA hyperparameters (e.g., rank, alpha) to analyze their effect on the quality of generated guides.
