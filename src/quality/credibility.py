"""
Module 2b: Source Credibility Scorer

Combines two approaches:
1. Pre-trained fake news classifier (hamzab/roberta-fake-news-classification)
   to detect misinformation-style writing patterns.
2. Heuristic signals: citations, academic language, balanced framing.

Explicitly ignores: subscriber_count, view_count, verified badges.
"""

import re
from typing import Dict, Any

# ── Lazy-loaded model ──
_classifier = None


def _get_classifier():
    """Lazy-load the pre-trained fake news classifier."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification",
            model="hamzab/roberta-fake-news-classification",
            top_k=None,
            truncation=True,
            max_length=512,
        )
    return _classifier


# ── Academic/formal language indicators ──
FORMAL_TERMS = {
    "research", "study", "studies", "evidence", "data", "analysis", "findings",
    "published", "journal", "peer-reviewed", "methodology", "hypothesis",
    "experiment", "systematic", "review", "meta-analysis", "statistical",
    "significant", "correlation", "causation", "theory", "framework",
    "reference", "citation", "according to", "peer reviewed",
}

# ── Hedging/balanced framing indicators ──
HEDGING_TERMS = {
    "however", "although", "nevertheless", "on the other hand", "conversely",
    "some argue", "critics say", "proponents suggest", "it depends",
    "nuanced", "complex", "debatable", "both sides", "perspective",
    "according to", "suggests", "indicates", "may", "might", "could",
    "potentially", "arguably",
}


def _compute_model_credibility(title: str, description: str) -> float:
    """Use the fake news classifier to assess credibility of writing style."""
    try:
        classifier = _get_classifier()
        # The model expects: "Title: ... Content: ..."
        text = f"{title}. {description[:1000]}"
        results = classifier(text)

        # Find the "Real" / "reliable" label score
        if isinstance(results[0], list):
            result_list = results[0]
        else:
            result_list = results

        for item in result_list:
            label = item["label"].lower()
            if any(pos in label for pos in ["real", "true", "reliable", "not fake"]):
                return item["score"]
            elif label in ("0", "label_0"):
                # Some models use 0 = real, 1 = fake
                return item["score"]

        # If we can't determine the label mapping, return moderate credibility
        return 0.5

    except Exception as e:
        print(f"[WARN] Fake news model unavailable: {e}")
        return 0.5


def _compute_heuristic_credibility(title: str, description: str, tags: list) -> float:
    """Heuristic credibility scoring based on content signals."""
    text = f"{title} {description}".lower()
    words = set(text.split())
    score = 0.0

    # 1. Citation presence (max 0.3)
    url_count = len(re.findall(r'https?://', description))
    source_mentions = len(re.findall(r'source[s]?:', text, re.IGNORECASE))
    paper_refs = len(re.findall(r'\(\d{4}\)', text))
    score += min((url_count * 0.03 + source_mentions * 0.1 + paper_refs * 0.08), 0.3)

    # 2. Academic language (max 0.25)
    formal_hits = len(words & FORMAL_TERMS)
    score += min(formal_hits * 0.05, 0.25)

    # 3. Balanced framing (max 0.25)
    hedging_hits = sum(1 for h in HEDGING_TERMS if h in text)
    score += min(hedging_hits * 0.05, 0.25)

    # 4. Description depth (max 0.2)
    desc_words = len(description.split()) if description else 0
    score += min(desc_words / 300, 0.2)

    return min(score, 1.0)


def compute_credibility_score(
    title: str,
    description: str,
    channel_name: str = "",
    tags: list = None,
) -> float:
    """
    Compute source credibility score ∈ [0, 1].

    Combines:
    - Model-based writing style analysis (60% weight)
    - Heuristic signals like citations, academic language (40% weight)

    Explicitly NOT using: subscriber_count, view_count, verified badge.
    """
    if tags is None:
        tags = []

    # Model-based score
    model_score = _compute_model_credibility(title, description)

    # Heuristic-based score
    heuristic_score = _compute_heuristic_credibility(title, description, tags)

    # Weighted combination
    combined = 0.6 * model_score + 0.4 * heuristic_score

    return round(min(combined, 1.0), 4)
