# 🧭 Local Travel Guide Generator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/xyyqwrjzjf/badge.svg)](https://ask.thili.ai/b/xyyqwrjzjf)
[![CI Tests](https://github.com/eMohan07/local-travel-guide-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/eMohan07/local-travel-guide-generator/actions)
![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-PEFT%20LoRA-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B.svg)

An AI-powered concierge system designed for travel agencies to generate hyper-personalized, culturally rich local travel mini-guides. Using **Low-Rank Adaptation (LoRA)** on pre-trained causal language models (GPT-2) and production **Adapter Weight Merging**, this tool adapts destination knowledge to specific tourist personas while maintaining ultra-low inference latency.

---

## 🌆 The World of Zephyria

All guides and recommendations are tailored to the fictional coastal metropolis of **Zephyria** — the *City of Whispering Winds and Lanterns*. Zephyria features 20 distinct districts spanning diverse vibes:
* **🏮 Lantern Quarter**: Nocturnal street food bazaars, dumpling dens, floating silk lanterns, and acoustic alleyways.
* **⚓ Old Quay & Seafront**: Maritime history, smoked seafood taverns, and coastal breakwater sea caves.
* **⚙️ Clockwork Ridge**: Antique horology workshops, neo-Victorian towers, and panoramic cable cars.
* **🌿 Mistwood Gardens**: Bioluminescent flora trails, nocturnal mosses, and glass-petal orchid conservatories.
* **💎 Crystal Cove**: Sea kayaking through quartz sea caves and tidal marine pools.
* **📚 The Scholar's Atoll**: Floating book barges, espresso bars, and quiet writing solariums.
* **🍷 Elysian Terraces**: Cliffside vineyards, classical twilight quartets, and fine dining.
* ...plus *Shadowfall Caverns*, *Breeze Heights*, *Starlight Pier*, *The Alchemist's Quarter*, and more!

---

## 🚀 Key Architectural Features

1. **Parameter-Efficient LoRA Fine-Tuning (`lfllm-lflora`)**:
   * Injects low-rank decomposition matrices ($r = 8, \alpha = 16$) into attention projection layers (`c_attn`).
   * **Parameter Efficiency**: Freezes **99.76%** of base model weights, training only **294,912** parameters out of 124.7M total.
2. **Instruction-Tuned Formatting**:
   * Structures travel recommendations using standard instruction-tuning delimiters (`### Instruction:`, `### Input:`, `### Response:`).
3. **Interactive Streamlit Concierge UI**:
   * Rich dashboard with district selectors, interest tags, quick-select traveler presets, parameter sliders (temperature, top-p, max tokens), guide markdown rendering, `.md` download, and itinerary history.
4. **Qualitative Evaluation Framework**:
   * Evaluates generated guides across **Relevance (40%)**, **Appeal (35%)**, and **Personalization (25%)**.
   * Achieved **100% benchmark pass rate** with an average score of **4.33 / 5.0 (Rating: Excellent)**.
5. **Zero-Overhead Adapter Merging**:
   * Permanently fuses delta weights ($W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$) via `merge_and_unload()`.
   * Unloads PEFT wrappers to export a native, standalone model (`models/merged_model/`) for high-throughput serving without runtime PEFT overhead.

---

## 📋 Completed Milestones

All project milestones are fully implemented, tested, and integrated:

- [x] **1. Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.**  
      *Delivered 20 curated district recommendations for Zephyria formatted as instruction-tuning input-output pairs (`data/travel_dataset.json`).*
- [x] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      *Implemented `inject_lora_adapter` and `train_lora_adapter` using the `lfllm-lflora` technique; saved trained adapter to `models/lora_adapter/`.*
- [x] **3. Develop a Streamlit UI to input interests and destination, generating a mini-guide.**  
      *Built full-featured interactive web app in `app/main.py` with presets, custom inputs, cached inference, and one-click export.*
- [x] **4. Evaluate the relevance and appeal of the generated recommendations.**  
      *Implemented comprehensive qualitative scoring rubric in `app/evaluation.py` producing `data/evaluation_report.json` with average score of 4.33 / 5.0.*
- [x] **5. Implement adapter merging for inference.**  
      *Implemented mathematical weight merging in `app/merger.py`, saving standalone model in `models/merged_model/` for zero-overhead inference.*

---

## 📊 Evaluation Results

Benchmark assessment of 5 diverse traveler personas across Zephyria:

| Evaluation Dimension | Weight | Benchmark Target | Achieved Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Relevance** | 40% | $\ge 3.0$ | **4.03 / 5.0** | ✅ Exceeded |
| **Appeal** | 35% | $\ge 3.0$ | **4.50 / 5.0** | ✅ Exceeded |
| **Personalization** | 25% | $\ge 3.0$ | **4.60 / 5.0** | ✅ Exceeded |
| **Composite Score** | 100% | $\ge 3.0$ | **4.33 / 5.0** | ✅ Exceeded |
| **Pass Rate** | — | $\ge 60\%$ | **100.0%** | ✅ Perfect |

Full evaluation report available at [`data/evaluation_report.json`](data/evaluation_report.json).

---

## 🗂️ Project Structure

```text
local-travel-guide-generator/
├── .github/workflows/
│   └── ci.yml                     # Automated GitHub Actions test workflow
├── app/
│   ├── core.py                    # Milestone entry points (milestone_1 through milestone_5)
│   ├── dataset.py                 # Zephyria dataset & instruction prompt formatting
│   ├── evaluation.py              # Qualitative evaluation rubrics & benchmark runner
│   ├── generator.py               # Autoregressive text generation & inference logic
│   ├── llm.py                     # Configurable fallback chat provider
│   ├── lora_trainer.py            # LoRA adapter injection & PyTorch training loop
│   ├── main.py                    # Streamlit web application
│   ├── merger.py                  # LoRA adapter weight merging for inference
│   └── storage.py                 # JSON key-value persistence
├── data/
│   ├── travel_dataset.json        # 20 formatted training pairs
│   └── evaluation_report.json     # Milestone 4 qualitative assessment results
├── models/
│   ├── lora_adapter/              # Trained LoRA adapter weights & config
│   └── merged_model/              # Standalone fused model weights for production
├── tests/
│   ├── test_core.py               # Unit & integration tests for Milestones 1–5
│   └── test_smoke.py              # Plumbing, environment, and syntax tests
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## ⚡ Quickstart

### 1. Prerequisites
* Python 3.12+ (or 3.13)
* Git

### 2. Installation
Clone the repository and set up your virtual environment:

```bash
git clone https://github.com/eMohan07/local-travel-guide-generator.git
cd local-travel-guide-generator

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Interactive Streamlit App
Run the concierge web UI:

```bash
streamlit run app/main.py
```
*Or via python module:*
```bash
python -m streamlit run app/main.py
```

### 4. Run the Full Test Suite
Execute all smoke tests and milestone verification suites:

```bash
pytest -v
```

---

## 💻 Programmatic Usage

You can invoke any milestone directly in Python:

```python
from app.core import milestone_1, milestone_2, milestone_3, milestone_4, milestone_5

# 1. Dataset Generation
dataset = milestone_1()
print(f"Loaded {len(dataset)} training examples.")

# 2. LoRA Fine-Tuning
train_result = milestone_2(epochs=3, batch_size=4)
print(f"LoRA trained. Trainable parameters: {train_result['param_stats']['trainable_percentage']:.3f}%")

# 3. Generate a Personalized Guide
guide = milestone_3(
    destination="Lantern Quarter",
    interests=["Street Food", "Night Markets"],
    traveler_profile="Solo foodie seeking midnight street snacks",
)
print(guide)

# 4. Run Qualitative Evaluation
report = milestone_4()
print(f"Pass Rate: {report['pass_rate_percentage']}%, Average Score: {report['average_scores']['overall']}/5.0")

# 5. Merge Adapter for Zero-Overhead Inference
merge_result = milestone_5()
print(f"Fused weights saved to: {merge_result['output_dir']}")
```

---

## 💡 Stretch Ideas & Future Enhancements
* **Multi-Destination Route Synthesis**: Generate connected day-by-day itineraries linking multiple Zephyria districts.
* **Interactive Client Feedback Loop**: Integrated star ratings in Streamlit stored via `app.storage.Store`.
* **Hyperparameter Sweeps**: Comparative analysis of varying LoRA rank ($r \in \{4, 8, 16, 32\}$) and alpha scalings.

---

## 📄 License
This project is part of the askThili Travel & Hospitality AI Curriculum.
