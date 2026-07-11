"""
Trained Ranker Inference — Use the trained neural ranker for scoring.

Loads the trained agent-centric model and uses it to score new videos,
replacing the hand-tuned weighted scoring function.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, List

from src.pipeline.ranker import NeuralRanker, FEATURE_COLS
from src.manipulation.clickbait import compute_clickbait_score
from src.manipulation.emotion import compute_emotional_manipulation_score
from src.quality.info_density import compute_info_density_score
from src.quality.credibility import compute_credibility_score
from src.quality.bias import compute_bias_score

# ── Lazy-loaded model ──
_model = None
_norm_stats = None


def _load_model():
    """Load the trained agent-centric ranker model."""
    global _model, _norm_stats

    model_path = os.path.join("models", "ranker_agent.pt")
    stats_path = os.path.join("models", "ranker_agent_norm_stats.json")

    if not os.path.exists(model_path):
        return None, None

    model = NeuralRanker(input_dim=len(FEATURE_COLS))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    with open(stats_path) as f:
        norm_stats = json.load(f)

    _model = model
    _norm_stats = norm_stats
    return model, norm_stats


def score_video_with_trained_model(video: Dict[str, Any]) -> float:
    """
    Score a video using the trained neural ranker.
    Falls back to heuristic scoring if the model isn't available.
    """
    global _model, _norm_stats

    if _model is None:
        _model, _norm_stats = _load_model()

    if _model is None:
        return None  # Model not trained yet

    title = video.get("title", "")
    description = video.get("description", "")
    duration = video.get("duration_seconds", 0)
    tags = video.get("tags", [])
    transcript = video.get("transcript", None)

    # Compute raw features
    features = {
        "info_density": compute_info_density_score(title, description, duration, transcript=transcript),
        "credibility": compute_credibility_score(title, description, tags=tags),
        "bias_balance": compute_bias_score(title, description),
        "clickbait_score": compute_clickbait_score(title),
        "emotional_score": compute_emotional_manipulation_score(title, description),
        "duration_minutes": duration / 60 if duration > 0 else 0,
        "title_length": len(title),
        "desc_length": len(description) if description else 0,
        "tag_count": len(tags) if isinstance(tags, list) else 0,
        "has_transcript": 1.0 if transcript else 0.0,
    }

    # Normalize using training statistics
    normalized = []
    for col in FEATURE_COLS:
        mean = _norm_stats["means"][col]
        std = _norm_stats["stds"][col]
        std = std if std > 0 else 1.0
        normalized.append((features[col] - mean) / std)

    # Run inference
    x = torch.tensor([normalized], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        score = _model(x).item()

    return score
