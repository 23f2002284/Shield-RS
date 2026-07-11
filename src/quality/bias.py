"""
Module 2d: Bias / Balance Scorer

Detects whether content presents a one-sided or balanced view.
Prefers multi-perspective, balanced content over opinion-heavy pieces.

Signals:
- Opinion vs. fact ratio (subjective language detection)
- Loaded/emotionally charged language (NRC Emotion Lexicon subset)
- Balanced framing indicators
"""

import re
from typing import Dict

# ── Subjective opinion indicators ──
OPINION_MARKERS = {
    "i think", "i believe", "in my opinion", "personally", "i feel",
    "obviously", "clearly", "of course", "everyone knows", "undeniably",
    "without a doubt", "hands down", "no question", "period",
    "the best", "the worst", "the only", "never", "always",
    "absolutely", "definitely", "certainly", "guarantee"
}

# ── Loaded / emotionally charged language ──
LOADED_WORDS = {
    "radical", "extreme", "fanatic", "propaganda", "brainwash",
    "sheep", "hoax", "agenda", "regime", "tyranny", "leftist",
    "rightist", "woke", "snowflake", "elitist", "conspiracy",
    "mainstream media", "fake news", "deep state", "sheeple",
    "nazi", "communist", "fascist", "socialist",
}

# ── Balanced framing indicators (positive signal) ──
BALANCE_MARKERS = {
    "however", "on the other hand", "conversely", "alternatively",
    "some argue", "critics point out", "proponents say", "both sides",
    "nuanced", "complex", "it depends", "pros and cons", "trade-off",
    "advantages and disadvantages", "strengths and weaknesses",
    "multiple perspectives", "different viewpoints", "debatable"
}


def compute_bias_score(title: str, description: str = "") -> float:
    """
    Compute bias/balance score ∈ [0, 1].
    1.0 = perfectly balanced, multi-perspective content.
    0.0 = completely one-sided, opinion-heavy, loaded language.
    """
    text = f"{title} {description}".lower()

    # Count opinion markers
    opinion_count = sum(1 for m in OPINION_MARKERS if m in text)

    # Count loaded language
    loaded_count = sum(1 for w in LOADED_WORDS if w in text)

    # Count balance indicators
    balance_count = sum(1 for b in BALANCE_MARKERS if b in text)

    # First-person pronoun density (higher = more opinionated)
    first_person = len(re.findall(r'\b(i|my|me|we|our|mine)\b', text))
    words_total = max(len(text.split()), 1)
    first_person_ratio = first_person / words_total

    # Compute the score: start at 0.5 (neutral baseline)
    score = 0.5

    # Penalize opinion and loaded language
    score -= min(opinion_count * 0.05, 0.25)
    score -= min(loaded_count * 0.08, 0.25)
    score -= min(first_person_ratio * 2.0, 0.15)

    # Reward balanced framing
    score += min(balance_count * 0.06, 0.25)

    # Clamp to [0, 1]
    return max(0.0, min(score, 1.0))
