"""
Module 1b: Emotional Manipulation Scorer

Uses the pre-trained GoEmotions model to detect emotional manipulation.
Model: SamLowe/roberta-base-go_emotions (HuggingFace)

This model classifies text into 28 emotion categories. We focus on
the "extreme" emotions that signal manipulation:
  - anger, disgust, fear, surprise, grief → manipulation signals
  - neutral, approval, curiosity → non-manipulative

The manipulation score is computed from the intensity of extreme
emotions in the title (weighted 70%) and description preview (30%).
"""

from typing import Dict

# ── Lazy-loaded model ──
_classifier = None

# Emotions that signal manipulation when present in titles
MANIPULATION_EMOTIONS = {
    "anger", "annoyance", "disgust", "fear", "grief",
    "nervousness", "embarrassment", "disappointment",
}

# Emotions that signal shock/clickbait tactics
SHOCK_EMOTIONS = {
    "surprise", "excitement", "desire",
}

# Neutral/non-manipulative emotions
SAFE_EMOTIONS = {
    "neutral", "approval", "realization", "curiosity",
    "admiration", "gratitude", "optimism", "caring",
}


def _get_classifier():
    """Lazy-load the pre-trained GoEmotions model."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,  # Return all 28 emotion scores
            truncation=True,
            max_length=512,
        )
    return _classifier


def compute_emotion_scores(text: str) -> Dict[str, float]:
    """
    Compute detailed emotion scores for a text using the GoEmotions model.
    Returns a dict mapping emotion labels to their probabilities.
    """
    try:
        classifier = _get_classifier()
        results = classifier(text)
        # results is a list of lists of dicts: [[{"label": ..., "score": ...}, ...]]
        if isinstance(results[0], list):
            scores = {item["label"]: round(item["score"], 4) for item in results[0]}
        else:
            scores = {item["label"]: round(item["score"], 4) for item in results}
        return scores
    except Exception as e:
        print(f"[WARN] GoEmotions model unavailable, using fallback: {e}")
        return _heuristic_emotion_scores(text)


def compute_emotional_manipulation_score(title: str, description: str = "") -> float:
    """
    Compute a single emotional manipulation score ∈ [0, 1].

    High scores indicate the content uses extreme emotional framing
    to drive engagement (anger, fear, shock, disgust).

    We weight the title 70% because titles are the primary click surface.
    """
    try:
        # Score the title
        title_scores = compute_emotion_scores(title)

        # Score the description preview (first 200 words)
        desc_preview = " ".join(description.split()[:200]) if description else ""
        desc_scores = compute_emotion_scores(desc_preview) if desc_preview else {}

        # Compute manipulation signal from title
        title_manip = sum(title_scores.get(e, 0) for e in MANIPULATION_EMOTIONS)
        title_shock = sum(title_scores.get(e, 0) for e in SHOCK_EMOTIONS)
        title_safe = sum(title_scores.get(e, 0) for e in SAFE_EMOTIONS)

        # Compute manipulation signal from description
        desc_manip = sum(desc_scores.get(e, 0) for e in MANIPULATION_EMOTIONS)

        # Combined score: manipulation + shock signals, dampened by safe signals
        title_signal = min(title_manip + title_shock * 0.5, 1.0) * max(1.0 - title_safe * 0.3, 0.3)
        desc_signal = min(desc_manip, 1.0)

        score = 0.7 * title_signal + 0.3 * desc_signal

        return round(min(score, 1.0), 4)

    except Exception:
        return _heuristic_emotional_manipulation_score(title, description)


def get_top_emotions(text: str, top_k: int = 5) -> list:
    """Return the top-k emotions for a piece of text. Useful for explainability."""
    scores = compute_emotion_scores(text)
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_emotions[:top_k]


# ── Heuristic fallbacks ──

FEAR_WORDS = {
    "danger", "dangerous", "warning", "threat", "risk", "scary", "terrifying",
    "alarming", "crisis", "catastrophe", "disaster", "collapse", "panic",
    "die", "death", "deadly", "fatal", "toxic", "nightmare", "doomed",
}

ANGER_WORDS = {
    "outrage", "outrageous", "furious", "angry", "disgusting", "exposed",
    "scam", "fraud", "corrupt", "betrayal", "lie", "liar", "destroy",
    "shame", "pathetic", "ridiculous", "unacceptable",
}

SHOCK_WORDS = {
    "shocking", "stunned", "unbelievable", "jaw-dropping", "mind-blowing",
    "insane", "crazy", "incredible", "unreal", "wtf", "omg", "bombshell",
    "epic", "wild", "legendary", "unprecedented",
}


def _heuristic_emotion_scores(text: str) -> Dict[str, float]:
    words = set(text.lower().split())
    return {
        "fear": min(len(words & FEAR_WORDS) * 0.2, 1.0),
        "anger": min(len(words & ANGER_WORDS) * 0.2, 1.0),
        "surprise": min(len(words & SHOCK_WORDS) * 0.2, 1.0),
    }


def _heuristic_emotional_manipulation_score(title: str, description: str = "") -> float:
    desc_preview = " ".join(description.split()[:200])
    title_scores = _heuristic_emotion_scores(title)
    desc_scores = _heuristic_emotion_scores(desc_preview)
    title_max = max(title_scores.values())
    title_sum = sum(title_scores.values())
    desc_max = max(desc_scores.values()) if desc_scores else 0
    score = 0.7 * min(title_max + title_sum * 0.3, 1.0) + 0.3 * desc_max
    return round(min(score, 1.0), 4)
