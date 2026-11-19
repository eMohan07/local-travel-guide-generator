"""Streamlit UI for the Local Travel Guide Generator.

Allows travel agencies to input tourist interests and destinations to generate
concise, personalized local travel guides powered by a fine-tuned LoRA model.
"""
from __future__ import annotations

from pathlib import Path
import streamlit as st

from app.dataset import CITY_NAME, TRAVEL_RECOMMENDATIONS_DATA
from app.generator import generate_mini_guide, load_inference_model
from app.lora_trainer import DEFAULT_ADAPTER_DIR
from app.merger import DEFAULT_MERGED_DIR, generate_with_merged_model

# Page configuration
st.set_page_config(
    page_title=f"{CITY_NAME} Local Travel Guide Generator",
    page_icon="🧭",
    layout="wide",
)

# Initialize session state for guide history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar: System configuration and Model status
with st.sidebar:
    st.header("⚙️ Model Configuration")
    adapter_path = DEFAULT_ADAPTER_DIR
    adapter_exists = (Path(adapter_path) / "adapter_config.json").exists()
    merged_exists = (Path(DEFAULT_MERGED_DIR) / "config.json").exists()

    model_mode = st.radio(
        "Inference Engine",
        options=["Merged Model (Zero Overhead)", "LoRA Adapter (PEFT)"],
        index=0 if merged_exists else 1,
        help="Choose between fully merged standalone weights or dynamically attached LoRA adapter",
    )

    if model_mode.startswith("Merged"):
        if merged_exists:
            st.success(f"✅ Merged Model Active (`{DEFAULT_MERGED_DIR}`)")
        else:
            st.info("Merged model not found yet. Run Milestone 5 to fuse weights.")
    else:
        if adapter_exists:
            st.success(f"✅ LoRA Adapter Active (`{adapter_path}`)")
        else:
            st.warning(f"⚠️ Adapter not found at `{adapter_path}`.")

    st.markdown("**Base LLM:** `GPT-2 (124M)`")
    st.markdown("**Fine-Tuning:** LoRA (PEFT, Rank=8, Alpha=16)")

    st.divider()
    st.subheader("🎛️ Generation Settings")
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.05)
    max_tokens = st.slider("Max New Tokens", min_value=50, max_value=300, value=160, step=10)
    top_p = st.slider("Top-p (Nucleus Sampling)", min_value=0.5, max_value=1.0, value=0.9, step=0.05)

    st.divider()
    st.subheader("💡 Quick Presets")
    presets = {
        "Custom": None,
        "Night Owl Foodie": {
            "district": "Lantern Quarter",
            "interests": ["Street Food", "Night Markets", "Live Acoustic Music"],
            "profile": "Night owl foodie seeking lively evening spots and indie tunes",
        },
        "Coastal Nature Explorer": {
            "district": "Crystal Cove",
            "interests": ["Sea Kayaking", "Coastal Caves", "Tidal Marine Life"],
            "profile": "Active outdoor adventurer wanting water recreation and scenic discovery",
        },
        "Steampunk & Architecture": {
            "district": "Clockwork Ridge",
            "interests": ["Vintage Horology", "Neo-Victorian Architecture", "Panoramic Views"],
            "profile": "Architectural admirer interested in mechanical history",
        },
        "Botanical & Zen Hiker": {
            "district": "Mistwood Gardens",
            "interests": ["Bioluminescent Flora", "Nature Hiking", "Botanical Photography"],
            "profile": "Nature photographer eager to explore serene, otherworldly flora",
        },
    }
    selected_preset = st.selectbox("Load Sample Traveler Profile", list(presets.keys()))

# Main Header
st.title(f"🧭 {CITY_NAME} Local Travel Guide Generator")
st.markdown(
    f"Personalized local travel concierge for **{CITY_NAME}** — the city of whispering winds, "
    "clockwork spires, and lantern canals. Tailor mini-guides for your clients with our fine-tuned LoRA model."
)

# Extract default values from selected preset
preset_data = presets.get(selected_preset)
default_district_idx = 0
default_interests = []
default_profile = ""

district_options = [item["district"] for item in TRAVEL_RECOMMENDATIONS_DATA]

all_interests = sorted(
    list({interest for item in TRAVEL_RECOMMENDATIONS_DATA for interest in item["interests"]})
)

if preset_data:
    if preset_data["district"] in district_options:
        default_district_idx = district_options.index(preset_data["district"])
    default_interests = [i for i in preset_data["interests"] if i in all_interests]
    default_profile = preset_data["profile"]

# Form inputs
col1, col2 = st.columns([1, 1])

with col1:
    selected_district = st.selectbox(
        "Destination District / Neighborhood",
        options=district_options,
        index=default_district_idx,
        help="Select a specific district in the fictional city of Zephyria",
    )

    traveler_profile_input = st.text_input(
        "Traveler Persona / Profile",
        value=default_profile,
        placeholder="e.g. Solo foodie, family with young children, romantic couple...",
        help="Contextual details about who the guide is for",
    )

with col2:
    selected_interests = st.multiselect(
        "Traveler Interests & Themes",
        options=all_interests,
        default=default_interests if default_interests else all_interests[:2],
        help="Select the activities, passions, and culinary tastes of the tourist",
    )

# Cache model and tokenizer to prevent re-loading on each interaction
@st.cache_resource
def get_model(adapter_dir: str | None):
    return load_inference_model(adapter_path=adapter_dir)

generate_btn = st.button("✨ Generate Personalized Mini-Guide", type="primary", use_container_width=True)

if generate_btn:
    if not selected_interests:
        st.error("Please select at least one travel interest.")
    else:
        with st.spinner("Generating personalized mini-guide with LoRA adapter..."):
            model, tokenizer = get_model(DEFAULT_ADAPTER_DIR if adapter_exists else None)
            guide_result = generate_mini_guide(
                destination=selected_district,
                interests=selected_interests,
                traveler_profile=traveler_profile_input,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        st.success(f"Guide generated for **{selected_district}**!")

        # Display result
        st.markdown("### 🗺️ Tailored Local Travel Guide")
        st.markdown(guide_result)

        # Download button
        st.download_button(
            label="📥 Download Mini-Guide (.md)",
            data=f"# {CITY_NAME} Travel Guide: {selected_district}\n\n**Interests:** {', '.join(selected_interests)}\n\n{guide_result}",
            file_name=f"{CITY_NAME.lower()}_{selected_district.lower().replace(' ', '_')}_guide.md",
            mime="text/markdown",
        )

        # Record in session history
        st.session_state.history.append(
            {
                "district": selected_district,
                "interests": selected_interests,
                "profile": traveler_profile_input,
                "guide": guide_result,
            }
        )

# Display history of generated guides
if st.session_state.history:
    st.divider()
    st.subheader("📚 Previously Generated Guides in this Session")
    for idx, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"{item['district']} ({', '.join(item['interests'])})"):
            st.markdown(item["guide"])
