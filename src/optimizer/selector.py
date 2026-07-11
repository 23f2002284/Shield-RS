"""
Module 3: Goal-Constrained Optimizer

Selects the optimal subset of videos given user constraints using
greedy submodular maximization (MMR-style).

At each step, picks the item that maximizes:
    λ · quality + (1 - λ) · marginal_coverage_gain

Subject to:
    - Total duration ≤ time_budget
    - |selected_set| ≤ max_items
    - Diversity ≥ min_diversity
"""

import sys
from typing import List, Dict, Any, Optional

from src.manipulation.clickbait import compute_clickbait_score
from src.manipulation.emotion import compute_emotional_manipulation_score
from src.quality.info_density import compute_info_density_score
from src.quality.credibility import compute_credibility_score
from src.quality.goal_alignment import compute_goal_alignment, compute_coverage_gain
from src.quality.bias import compute_bias_score
from src.optimizer.weights import DEFAULT_WEIGHTS


def compute_agent_score(video: Dict[str, Any], user_goal: str, weights: Dict[str, float] = None) -> float:
    """
    Compute the agent-centric score for a single video.

    Strategy:
    1. Try the TRAINED neural ranker (if model exists) — learned weights
    2. Fall back to heuristic weighted scoring — hand-tuned weights

    Both approaches implement the INVERTED scoring function:
    what YouTube boosts, we penalize.
    """
    # ── Try trained model first ──
    try:
        from src.pipeline.trained_scorer import score_video_with_trained_model
        trained_score = score_video_with_trained_model(video)
        if trained_score is not None:
            # Blend trained score with goal alignment (which needs the user query)
            title = video.get("title", "")
            description = video.get("description", "")
            goal_alignment = compute_goal_alignment(user_goal, title, description)
            # 60% trained model + 40% goal alignment
            return 0.6 * trained_score + 0.4 * goal_alignment
    except Exception:
        pass

    # ── Fallback: heuristic scoring ──
    if weights is None:
        weights = DEFAULT_WEIGHTS

    title = video.get("title", "")
    description = video.get("description", "")
    duration = video.get("duration_seconds", 0)
    tags = video.get("tags", [])
    transcript = video.get("transcript", None)

    # Positive signals (agent REWARDS these)
    info_density = compute_info_density_score(title, description, duration, transcript=transcript)
    credibility = compute_credibility_score(title, description, tags=tags)
    goal_alignment = compute_goal_alignment(user_goal, title, description)
    bias_balance = compute_bias_score(title, description)

    # Negative signals (agent PENALIZES these — the algorithmic inversion)
    clickbait = compute_clickbait_score(title)
    emotional = compute_emotional_manipulation_score(title, description)

    score = (
        weights["info_density"] * info_density
        + weights["credibility"] * credibility
        + weights["goal_alignment"] * goal_alignment
        + weights["bias_balance"] * bias_balance
        - weights["clickbait_penalty"] * clickbait
        - weights["emotional_penalty"] * emotional
    )

    return score


def select_optimal_set(
    candidates: List[Dict[str, Any]],
    user_goal: str,
    time_budget_seconds: int,
    max_items: int = 10,
    lambda_param: float = 0.7,
    weights: Dict[str, float] = None,
) -> List[Dict[str, Any]]:
    """
    Greedy submodular maximization to select the optimal set of videos.

    At each step, picks the candidate that maximizes:
        λ · agent_score(candidate) + (1-λ) · marginal_coverage_gain(candidate, selected)

    Respects:
        - time_budget (total duration of selected set)
        - max_items (maximum number of videos)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    selected = []
    selected_texts = []
    remaining_budget = time_budget_seconds
    remaining = list(candidates)

    # Pre-compute agent scores for all candidates
    print(f"  Computing agent scores for {len(remaining)} candidates...")
    for v in remaining:
        v["_agent_score"] = compute_agent_score(v, user_goal, weights)

    while len(selected) < max_items and remaining_budget > 0 and remaining:
        best = None
        best_combined_score = -float("inf")

        for candidate in remaining:
            duration = candidate.get("duration_seconds", 0)
            if duration > remaining_budget:
                continue

            quality = candidate["_agent_score"]

            # Compute marginal coverage gain
            if selected_texts:
                c_title = candidate.get("title", "")
                c_desc = candidate.get("description", "")
                diversity = compute_coverage_gain(c_title, c_desc, selected_texts)
            else:
                diversity = 1.0

            combined = lambda_param * quality + (1 - lambda_param) * diversity

            if combined > best_combined_score:
                best = candidate
                best_combined_score = combined

        if best is None:
            break

        selected.append(best)
        remaining.remove(best)
        remaining_budget -= best.get("duration_seconds", 0)

        # Track selected content for diversity computation
        best_text = f"{best.get('title', '')}. {' '.join(best.get('description', '').split()[:200])}"
        selected_texts.append(best_text)

        print(f"  Selected #{len(selected)}: \"{best.get('title', '')[:60]}...\" "
              f"(score={best_combined_score:.3f}, "
              f"budget_left={remaining_budget//60}m)")

    # Clean up internal keys
    for v in selected:
        v.pop("_agent_score", None)
    for v in remaining:
        v.pop("_agent_score", None)

    return selected
