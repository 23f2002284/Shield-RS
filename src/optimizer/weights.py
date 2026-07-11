"""
Module 3b: Weight Configuration

Default weights for the agent-centric scoring function.
These can be adjusted based on user meta-preferences.

The key insight: positive weights REWARD agent-centric signals,
negative weights (applied as subtraction) PENALIZE human-centric
manipulation tactics. This is the "algorithmic inversion".
"""

from typing import Dict

DEFAULT_WEIGHTS = {
    "info_density": 0.20,       # How much useful info per minute?
    "credibility": 0.20,        # Is the source an expert (not just popular)?
    "goal_alignment": 0.30,     # Does this serve the user's stated goal?
    "bias_balance": 0.10,       # Is it balanced, multi-perspective?
    "clickbait_penalty": 0.25,  # PENALIZE clickbait tactics
    "emotional_penalty": 0.15,  # PENALIZE emotional manipulation
}


def get_weights_for_preference(preference: str) -> Dict[str, float]:
    """
    Adjust weights based on user meta-preferences.

    Supported preferences:
    - "scientific": Boost credibility + bias balance
    - "diverse": Boost bias balance, reduce goal alignment strictness
    - "efficient": Boost info density, tighten time constraints
    - "default": Standard balanced weights
    """
    weights = DEFAULT_WEIGHTS.copy()

    if preference == "scientific":
        weights["credibility"] = 0.35
        weights["bias_balance"] = 0.15
        weights["goal_alignment"] = 0.25
    elif preference == "diverse":
        weights["bias_balance"] = 0.20
        weights["goal_alignment"] = 0.20
    elif preference == "efficient":
        weights["info_density"] = 0.35
        weights["goal_alignment"] = 0.25
        weights["credibility"] = 0.15
    # "default" uses DEFAULT_WEIGHTS as-is

    return weights
