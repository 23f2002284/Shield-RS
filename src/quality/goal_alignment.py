"""
Module 2c: Goal Alignment / Topic Coverage Scorer

Measures how well a video's content aligns with the user's stated goal.
Uses sentence-transformers to compute semantic similarity between
the user query and the video's title + description.

This is the core "does this content serve what you ASKED for?" signal.
"""

import numpy as np
from typing import List, Optional

# Lazy-loaded model (loaded once on first use)
_model = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def compute_goal_alignment(user_goal: str, title: str, description: str = "") -> float:
    """
    Compute goal alignment score ∈ [0, 1].
    Uses cosine similarity between the user's goal embedding and
    the video's content embedding (title + first 200 words of description).
    """
    model = _get_model()

    # Build the content representation
    desc_preview = " ".join(description.split()[:200]) if description else ""
    content_text = f"{title}. {desc_preview}"

    # Encode both
    embeddings = model.encode([user_goal, content_text], normalize_embeddings=True)

    # Cosine similarity (already normalized, so just dot product)
    similarity = float(np.dot(embeddings[0], embeddings[1]))

    # Clamp to [0, 1] — cosine similarity of normalized vectors is already in [-1, 1]
    return max(0.0, min(similarity, 1.0))


def compute_coverage_gain(
    candidate_title: str,
    candidate_description: str,
    already_selected_texts: List[str],
) -> float:
    """
    Compute marginal coverage gain — how much NEW information does this
    candidate add beyond what's already been selected?

    High gain = this video covers a subtopic not yet represented.
    Low gain = this video is redundant with already-selected content.
    """
    if not already_selected_texts:
        return 1.0  # First item always has maximum coverage gain

    model = _get_model()

    candidate_text = f"{candidate_title}. {' '.join(candidate_description.split()[:200])}"

    # Encode candidate and all already-selected items
    all_texts = [candidate_text] + already_selected_texts
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    candidate_emb = embeddings[0]
    selected_embs = embeddings[1:]

    # Max similarity to any already-selected item
    max_similarity = max(float(np.dot(candidate_emb, sel)) for sel in selected_embs)

    # Coverage gain is inverse of max similarity (more different = more gain)
    gain = max(0.0, 1.0 - max_similarity)

    return gain
