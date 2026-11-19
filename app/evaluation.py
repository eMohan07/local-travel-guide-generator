"""Qualitative evaluation module for assessing generated local travel guides.

Assesses generated recommendations across three core dimensions:
1. Relevance: Alignment with requested destination and tourist interests.
2. Appeal: Tone, vividness, enticing travel prose, and structured presentation.
3. Personalization: Tailoring to traveler persona, specific secret spots, and actionable tips.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.dataset import CITY_NAME, TRAVEL_RECOMMENDATIONS_DATA
from app.generator import generate_mini_guide

DEFAULT_EVALUATION_REPORT_PATH = "data/evaluation_report.json"

# Representative qualitative evaluation benchmark cases
BENCHMARK_EVALUATION_CASES: List[Dict[str, Any]] = [
    {
        "id": "eval-001",
        "destination": "Lantern Quarter",
        "interests": ["Street Food", "Night Markets", "Live Acoustic Music"],
        "traveler_profile": "Solo foodie seeking vibrant evening street eats and indie melodies",
        "expected_themes": ["lantern", "dumpling", "night", "food", "market", "music", "busker"],
    },
    {
        "id": "eval-002",
        "destination": "Clockwork Ridge",
        "interests": ["Vintage Horology", "Neo-Victorian Architecture", "Panoramic Views"],
        "traveler_profile": "Steampunk architecture admirer interested in clock mechanics and views",
        "expected_themes": ["clock", "gear", "orrery", "cable car", "ridge", "view", "architecture"],
    },
    {
        "id": "eval-003",
        "destination": "Crystal Cove",
        "interests": ["Sea Kayaking", "Coastal Caves", "Tidal Marine Life"],
        "traveler_profile": "Active outdoor adventurer wanting coastal waters and tide pool marine life",
        "expected_themes": ["kayak", "cave", "tide", "cove", "sea", "marine", "reef", "water"],
    },
    {
        "id": "eval-004",
        "destination": "Mistwood Gardens",
        "interests": ["Bioluminescent Flora", "Nature Hiking", "Botanical Photography"],
        "traveler_profile": "Nature photographer and hiker seeking tranquil botanical wonders",
        "expected_themes": ["glow", "bioluminescent", "trail", "garden", "orchid", "moss", "nature"],
    },
    {
        "id": "eval-005",
        "destination": "Breeze Heights",
        "interests": ["Family-Friendly Fun", "Kite Flying", "Whimsical Puppet Shows"],
        "traveler_profile": "Family with young children seeking playful outdoor activities",
        "expected_themes": ["windmill", "kite", "puppet", "family", "children", "bakery", "meadow"],
    },
]


def assess_relevance(
    guide_text: str,
    destination: str,
    interests: List[str] | str,
    expected_themes: Optional[List[str]] = None,
) -> Tuple[float, List[str]]:
    """Assess how relevant the recommendation is to the destination and requested interests.
    
    Score: 1.0 to 5.0
    """
    guide_lower = guide_text.lower()
    dest_lower = destination.lower()
    feedback: List[str] = []

    score = 1.0

    # 1. Destination recognition
    dest_tokens = [t for t in re.split(r"[\s,&/-]+", dest_lower) if len(t) > 2]
    matched_dest = any(t in guide_lower for t in dest_tokens) or CITY_NAME.lower() in guide_lower
    if matched_dest:
        score += 1.5
        feedback.append(f"Destination context for '{destination}' accurately reflected.")
    else:
        feedback.append(f"Missing explicit reference to destination '{destination}'.")

    # 2. Interests alignment
    if isinstance(interests, str):
        interest_tokens = [i.strip().lower() for i in interests.split(",") if i.strip()]
    else:
        interest_tokens = [i.strip().lower() for i in interests]

    matched_interests = sum(
        1 for item in interest_tokens if any(w in guide_lower for w in item.split())
    )
    if interest_tokens:
        interest_ratio = matched_interests / len(interest_tokens)
        score += interest_ratio * 1.5
        if interest_ratio >= 0.6:
            feedback.append(f"Strong alignment with requested interests ({matched_interests}/{len(interest_tokens)} matched).")
        else:
            feedback.append(f"Moderate interest coverage ({matched_interests}/{len(interest_tokens)} matched).")

    # 3. Expected thematic vocabulary
    if expected_themes:
        matched_themes = sum(1 for t in expected_themes if t.lower() in guide_lower)
        theme_ratio = matched_themes / max(len(expected_themes), 1)
        score += min(theme_ratio * 1.0, 1.0)

    score = max(1.0, min(5.0, round(score, 2)))
    return score, feedback


def assess_appeal(guide_text: str) -> Tuple[float, List[str]]:
    """Assess tone, descriptive richness, structured formatting, and engaging appeal.
    
    Score: 1.0 to 5.0
    """
    feedback: List[str] = []
    score = 1.0

    # 1. Structural appeal (uses formatting markers, emojis, bullet points, headers)
    structure_markers = ["highlight", "secret", "tip", "best time", "welcome", "🌟", "🤫", "⏰", "💡", "###", "**"]
    structure_count = sum(1 for m in structure_markers if m in guide_text.lower())
    if structure_count >= 5:
        score += 1.5
        feedback.append("Excellent structured formatting with clear highlights, secrets, and tips.")
    elif structure_count >= 3:
        score += 1.0
        feedback.append("Good formatting with basic section indicators.")
    else:
        feedback.append("Prose lacks clear bulleted sections or callout formatting.")

    # 2. Evocative & appealing vocabulary
    enticing_words = [
        "vibrant", "breathtaking", "curated", "panoramic", "authentic", "hidden",
        "stunning", "delight", "historic", "enchanting", "explore", "flavor",
        "quaint", "scenic", "serene", "artisan", "glowing", "spectacular"
    ]
    evocative_count = sum(1 for w in enticing_words if w in guide_text.lower())
    if evocative_count >= 4:
        score += 1.5
        feedback.append("Enticing, vivid, and appealing travel prose suitable for a premium agency.")
    elif evocative_count >= 2:
        score += 1.0
        feedback.append("Welcoming tone with moderate descriptive flair.")
    else:
        feedback.append("Tone is straightforward but could be more evocative.")

    # 3. Text length & completeness
    word_count = len(guide_text.split())
    if word_count >= 80:
        score += 1.0
        feedback.append(f"Comprehensive guide depth ({word_count} words).")
    elif word_count >= 40:
        score += 0.5
        feedback.append(f"Concise mini-guide length ({word_count} words).")
    else:
        feedback.append(f"Output is brief ({word_count} words).")

    score = max(1.0, min(5.0, round(score, 2)))
    return score, feedback


def assess_personalization(
    guide_text: str,
    traveler_profile: str,
    destination: str,
) -> Tuple[float, List[str]]:
    """Assess how tailored the recommendation is to the traveler's persona and context.
    
    Score: 1.0 to 5.0
    """
    feedback: List[str] = []
    score = 1.0
    guide_lower = guide_text.lower()

    # 1. Specific local elements (named spots, secret locations)
    has_secret = any(term in guide_lower for term in ["secret", "hidden", "grotto", "terrace", "parlor", "perch", "alley"])
    has_tip = any(term in guide_lower for term in ["tip", "insider", "ask", "bring", "reserve", "order", "carry"])
    has_timing = any(term in guide_lower for term in ["best time", "morning", "afternoon", "dusk", "night", "twilight", "pm", "am"])

    if has_secret and has_tip and has_timing:
        score += 2.0
        feedback.append("Highly personalized with specific hidden gems, actionable tips, and optimal visit timings.")
    elif has_secret or has_tip:
        score += 1.0
        feedback.append("Includes local tips or hidden recommendations.")
    else:
        feedback.append("Missing distinctive insider secrets or timing recommendations.")

    # 2. Persona context matching
    if traveler_profile:
        persona_tokens = [w.lower() for w in re.split(r"\W+", traveler_profile) if len(w) > 3]
        matched_persona_tokens = sum(1 for t in persona_tokens if t in guide_lower)
        if matched_persona_tokens >= 2:
            score += 1.5
            feedback.append(f"Clear contextual resonance with traveler persona ('{traveler_profile}').")
        elif matched_persona_tokens >= 1:
            score += 1.0
            feedback.append("Partial alignment with traveler persona.")
        else:
            score += 0.5
            feedback.append("General recommendation; subtle persona-specific tailoring.")
    else:
        score += 1.0
        feedback.append("Standard traveler profile used.")

    # 3. Grounded local flavor
    has_city = CITY_NAME.lower() in guide_lower
    if has_city:
        score += 0.5

    score = max(1.0, min(5.0, round(score, 2)))
    return score, feedback


def evaluate_single_recommendation(
    guide_text: str,
    destination: str,
    interests: List[str] | str,
    traveler_profile: str = "",
    expected_themes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform a holistic qualitative assessment of a single generated travel guide."""
    relevance_score, relevance_fb = assess_relevance(
        guide_text, destination, interests, expected_themes
    )
    appeal_score, appeal_fb = assess_appeal(guide_text)
    personalization_score, personalization_fb = assess_personalization(
        guide_text, traveler_profile, destination
    )

    # Weighted overall score (out of 5.0)
    overall_score = round(
        (0.40 * relevance_score) + (0.35 * appeal_score) + (0.25 * personalization_score),
        2,
    )

    return {
        "destination": destination,
        "interests": interests if isinstance(interests, list) else [i.strip() for i in interests.split(",")],
        "traveler_profile": traveler_profile,
        "scores": {
            "relevance": relevance_score,
            "appeal": appeal_score,
            "personalization": personalization_score,
            "overall": overall_score,
        },
        "feedback": {
            "relevance": relevance_fb,
            "appeal": appeal_fb,
            "personalization": personalization_fb,
        },
        "passed": overall_score >= 3.0,
        "guide_preview": guide_text[:200] + "..." if len(guide_text) > 200 else guide_text,
    }


def run_qualitative_evaluation(
    test_cases: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[Path | str] = DEFAULT_EVALUATION_REPORT_PATH,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run qualitative assessment across multiple benchmark cases and produce an evaluation report."""
    cases = test_cases or BENCHMARK_EVALUATION_CASES
    evaluated_cases: List[Dict[str, Any]] = []

    for case in cases:
        guide = generate_mini_guide(
            destination=case["destination"],
            interests=case["interests"],
            traveler_profile=case.get("traveler_profile", ""),
            adapter_path=adapter_path,
        )
        assessment = evaluate_single_recommendation(
            guide_text=guide,
            destination=case["destination"],
            interests=case["interests"],
            traveler_profile=case.get("traveler_profile", ""),
            expected_themes=case.get("expected_themes"),
        )
        assessment["full_guide"] = guide
        evaluated_cases.append(assessment)

    # Aggregate metric averages
    avg_relevance = round(sum(c["scores"]["relevance"] for c in evaluated_cases) / len(evaluated_cases), 2)
    avg_appeal = round(sum(c["scores"]["appeal"] for c in evaluated_cases) / len(evaluated_cases), 2)
    avg_personalization = round(sum(c["scores"]["personalization"] for c in evaluated_cases) / len(evaluated_cases), 2)
    avg_overall = round(sum(c["scores"]["overall"] for c in evaluated_cases) / len(evaluated_cases), 2)

    pass_rate = 100.0 * sum(1 for c in evaluated_cases if c["passed"]) / len(evaluated_cases)

    report = {
        "status": "success",
        "city": CITY_NAME,
        "total_cases_evaluated": len(evaluated_cases),
        "pass_rate_percentage": pass_rate,
        "average_scores": {
            "relevance": avg_relevance,
            "appeal": avg_appeal,
            "personalization": avg_personalization,
            "overall": avg_overall,
        },
        "qualitative_verdict": (
            "Excellent" if avg_overall >= 4.0 else "Satisfactory" if avg_overall >= 3.0 else "Needs Improvement"
        ),
        "case_evaluations": evaluated_cases,
    }

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report
