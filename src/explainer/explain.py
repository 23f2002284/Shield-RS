"""
Module 4: Explainability Layer

Generates human-readable explanations for WHY each video was
recommended or filtered. Transparency is a core value of the agent.

For each recommended video: why it was selected.
For each filtered video: why it was rejected.
"""

from typing import Dict, Any, List

from src.manipulation.clickbait import compute_clickbait_score, compute_tactic_vector
from src.manipulation.emotion import compute_emotional_manipulation_score
from src.quality.info_density import compute_info_density_score
from src.quality.credibility import compute_credibility_score
from src.quality.goal_alignment import compute_goal_alignment
from src.quality.bias import compute_bias_score


def _level(score: float) -> str:
    """Convert a 0–1 score to a human-readable level."""
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    elif score >= 0.15:
        return "Low"
    return "Very Low"


def explain_recommendation(video: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
    """
    Generate a full explanation for a recommended or filtered video.
    Returns a dict with scores and a natural-language reason.
    """
    title = video.get("title", "")
    description = video.get("description", "")
    duration = video.get("duration_seconds", 0)
    tags = video.get("tags", [])
    transcript = video.get("transcript", None)

    # Compute all scores
    clickbait = compute_clickbait_score(title)
    emotional = compute_emotional_manipulation_score(title, description)
    info_density = compute_info_density_score(title, description, duration, transcript=transcript)
    credibility = compute_credibility_score(title, description, tags=tags)
    goal_alignment = compute_goal_alignment(user_goal, title, description)
    bias_balance = compute_bias_score(title, description)
    tactics = compute_tactic_vector(title)

    # Build the explanation
    scores = {
        "goal_alignment": round(goal_alignment, 2),
        "info_density": round(info_density, 2),
        "credibility": round(credibility, 2),
        "bias_balance": round(bias_balance, 2),
        "clickbait_score": round(clickbait, 2),
        "emotional_manipulation": round(emotional, 2),
        "tactic_vector": {k: round(v, 2) for k, v in tactics.items()},
    }

    # Generate natural-language reason
    reasons = []

    if goal_alignment >= 0.5:
        reasons.append(f"Strong alignment with your goal ({_level(goal_alignment)})")
    elif goal_alignment >= 0.3:
        reasons.append(f"Moderate alignment with your goal ({_level(goal_alignment)})")

    if credibility >= 0.3:
        reasons.append(f"Credible source with cited references ({_level(credibility)})")

    if info_density >= 0.4:
        reasons.append(f"High information density — efficient use of your time")

    if bias_balance >= 0.5:
        reasons.append(f"Balanced, multi-perspective framing")

    if clickbait >= 0.3:
        # Build tactic-specific reasons
        tactic_reasons = []
        if tactics.get("curiosity_gap", 0) > 0.3:
            tactic_reasons.append("curiosity gap")
        if tactics.get("false_urgency", 0) > 0.3:
            tactic_reasons.append("false urgency")
        if tactics.get("emotional_bait", 0) > 0.3:
            tactic_reasons.append("emotional bait")
        if tactics.get("exaggeration", 0) > 0.3:
            tactic_reasons.append("exaggeration")

        tactic_str = ", ".join(tactic_reasons) if tactic_reasons else "clickbait patterns"
        reasons.append(f"⚠️ Clickbait detected: {tactic_str} (score: {clickbait:.2f})")

    if emotional >= 0.3:
        reasons.append(f"⚠️ Emotional manipulation detected (score: {emotional:.2f})")

    return {
        "video_id": video.get("video_id", ""),
        "title": title,
        "scores": scores,
        "reasons": reasons,
        "summary": "; ".join(reasons) if reasons else "No strong signals detected.",
    }


def explain_selection(
    recommended: List[Dict[str, Any]],
    filtered: List[Dict[str, Any]],
    user_goal: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate explanations for all recommended and filtered videos.
    """
    return {
        "recommended": [
            {**explain_recommendation(v, user_goal), "status": "✅ Recommended"}
            for v in recommended
        ],
        "filtered": [
            {**explain_recommendation(v, user_goal), "status": "❌ Filtered"}
            for v in filtered[:10]  # Limit filtered explanations to top 10
        ],
    }
