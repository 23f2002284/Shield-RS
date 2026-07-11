"""
Module 1a: Clickbait Classifier

Uses a pre-trained DistilRoBERTa model fine-tuned on clickbait datasets.
Model: valurank/distilroberta-clickbait (HuggingFace)

Outputs:
  - clickbait_score ∈ [0, 1]
  - tactic_vector (heuristic attribution for explainability)

The model handles the heavy lifting of classification.
The tactic vector provides explainability on WHY it's clickbait.
"""

import re
from typing import Dict

# ── Lazy-loaded model ──
_classifier = None


def _get_classifier():
    """Lazy-load the pre-trained clickbait classifier."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification",
            model="valurank/distilroberta-clickbait",
            top_k=None,  # Return scores for all labels
        )
    return _classifier


# ── Tactic patterns for explainability ──
CURIOSITY_GAP_PATTERNS = [
    r"you won'?t believe", r"what happens next", r"the truth about",
    r"no one tells you", r"they don'?t want you to know", r"this is why",
    r"here'?s what", r"what they'?re not telling", r"the real reason",
]

FALSE_URGENCY_PATTERNS = [
    r"before it'?s too late", r"you need to know", r"right now",
    r"stop everything", r"don'?t miss", r"last chance", r"act now",
    r"watch this before", r"do this immediately",
]

EMOTIONAL_BAIT_PATTERNS = [
    r"shocking", r"insane", r"jaw[- ]?dropping", r"mind[- ]?blowing",
    r"unbelievable", r"incredible", r"terrifying", r"heartbreaking",
    r"devastating", r"disgusting",
]

EXAGGERATION_PATTERNS = [
    r"best ever", r"worst ever", r"the only", r"never seen before",
    r"changed my life", r"life[- ]?changing", r"game[- ]?changer",
    r"most \w+ ever", r"100%", r"literally",
]


def _count_pattern_matches(text: str, patterns: list) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


def _caps_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _punctuation_score(text: str) -> float:
    return min((text.count("!") + text.count("?")) * 0.1, 1.0)


def compute_clickbait_score(title: str) -> float:
    """
    Compute clickbait score ∈ [0, 1] using the pre-trained model.
    Falls back to heuristics if the model can't be loaded.
    """
    try:
        classifier = _get_classifier()
        results = classifier(title[:512])  # Truncate to model max length

        # The model returns labels like "clickbait" and "not_clickbait"
        # Find the clickbait probability
        for result_list in (results if isinstance(results[0], list) else [results]):
            for item in result_list:
                label = item["label"].lower()
                if "clickbait" in label and "not" not in label:
                    return round(item["score"], 4)
                elif label in ("1", "label_1", "positive"):
                    return round(item["score"], 4)

        # If label structure is unexpected, return first score
        return round(results[0][0]["score"], 4)

    except Exception as e:
        # Fallback to heuristic scoring if model fails to load
        print(f"[WARN] Clickbait model unavailable, using heuristic fallback: {e}")
        return _heuristic_clickbait_score(title)


def _heuristic_clickbait_score(title: str) -> float:
    """Heuristic fallback when the pre-trained model is unavailable."""
    score = 0.0
    score += min(_count_pattern_matches(title, CURIOSITY_GAP_PATTERNS) * 0.15, 0.3)
    score += min(_count_pattern_matches(title, FALSE_URGENCY_PATTERNS) * 0.12, 0.25)
    score += min(_count_pattern_matches(title, EMOTIONAL_BAIT_PATTERNS) * 0.12, 0.25)
    score += min(_count_pattern_matches(title, EXAGGERATION_PATTERNS) * 0.08, 0.2)
    score += min(_caps_ratio(title) * 0.3, 0.15)
    score += min(_punctuation_score(title), 0.1)
    return min(score, 1.0)


def compute_tactic_vector(title: str) -> Dict[str, float]:
    """
    Return a tactic attribution vector showing WHICH manipulation
    tactics are present in the title. Used by the explainability layer.
    (Always heuristic — the model gives us the score, this gives us the WHY.)
    """
    return {
        "curiosity_gap": min(_count_pattern_matches(title, CURIOSITY_GAP_PATTERNS) * 0.25, 1.0),
        "false_urgency": min(_count_pattern_matches(title, FALSE_URGENCY_PATTERNS) * 0.25, 1.0),
        "emotional_bait": min(_count_pattern_matches(title, EMOTIONAL_BAIT_PATTERNS) * 0.25, 1.0),
        "exaggeration": min(_count_pattern_matches(title, EXAGGERATION_PATTERNS) * 0.2, 1.0),
        "caps_shouting": min(_caps_ratio(title), 1.0),
        "punctuation_abuse": _punctuation_score(title),
    }
