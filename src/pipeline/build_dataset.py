"""
Dataset Builder for Learning-to-Rank RS Training

Creates training data from our scraped YouTube videos by computing
two sets of relevance labels:

1. ENGAGEMENT labels (what YouTube optimizes for):
   - Based on view_count, like_count, like_ratio
   - This is the "human-centric RS" baseline

2. AGENT-CENTRIC labels (what our shield optimizes for):
   - Based on info_density, credibility, goal_alignment, bias_balance
   - Penalized by clickbait_score, emotional_manipulation
   - This is the "agent-centric RS" — our contribution

Same model architecture, different training labels → different rankings.
This IS the algorithmic inversion.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.manipulation.clickbait import compute_clickbait_score
from src.manipulation.emotion import compute_emotional_manipulation_score
from src.quality.info_density import compute_info_density_score
from src.quality.credibility import compute_credibility_score
from src.quality.bias import compute_bias_score


# ── Queries that define what a "user goal" looks like ──
QUERY_GOALS = {
    "learn_about_climate_change": "learn about climate change science and impact",
    "how_to_invest_for_beginners": "learn how to invest money as a beginner",
    "history_of_ancient_Rome": "learn about the history of ancient Rome",
    "machine_learning_tutorial": "learn machine learning concepts and tutorials",
    "healthy_meal_prep": "learn healthy meal preparation and recipes",
}


def compute_features(video: Dict, user_goal: str) -> Dict:
    """
    Compute all feature scores for a single video.
    Returns a dict of features ready for model training.
    """
    title = video.get("title", "")
    description = video.get("description", "")
    duration = video.get("duration_seconds", 0)
    tags = video.get("tags", [])
    transcript = video.get("transcript", None)

    # Agent-centric features (what our RS should value)
    info_density = compute_info_density_score(title, description, duration, transcript=transcript)
    credibility = compute_credibility_score(title, description, tags=tags)
    bias_balance = compute_bias_score(title, description)

    # Manipulation features (what our RS should penalize)
    clickbait = compute_clickbait_score(title)
    emotional = compute_emotional_manipulation_score(title, description)

    # Content metadata features
    duration_minutes = duration / 60 if duration > 0 else 0
    title_length = len(title)
    desc_length = len(description) if description else 0
    tag_count = len(tags) if isinstance(tags, list) else 0
    has_transcript = 1.0 if transcript else 0.0

    return {
        # Agent-centric quality features
        "info_density": info_density,
        "credibility": credibility,
        "bias_balance": bias_balance,
        # Manipulation features
        "clickbait_score": clickbait,
        "emotional_score": emotional,
        # Content metadata
        "duration_minutes": duration_minutes,
        "title_length": title_length,
        "desc_length": desc_length,
        "tag_count": tag_count,
        "has_transcript": has_transcript,
        # Engagement metrics (used for engagement labels, NOT as input features)
        "_view_count": video.get("view_count", 0),
        "_like_count": video.get("like_count", 0),
        "_subscriber_count": video.get("subscriber_count", 0),
    }


def compute_engagement_label(features: Dict) -> float:
    """
    Engagement-based relevance label (what YouTube optimizes for).
    Normalized to [0, 1] using log-scaling.
    """
    views = max(features["_view_count"], 1)
    likes = max(features["_like_count"], 1)
    # Log-normalized engagement score
    log_views = np.log10(views) / 8.0  # Normalize by ~100M max
    log_likes = np.log10(likes) / 6.0  # Normalize by ~1M max
    return float(np.clip(0.6 * log_views + 0.4 * log_likes, 0, 1))


def compute_agent_label(features: Dict) -> float:
    """
    Agent-centric relevance label (what our shield optimizes for).
    This is the INVERTED scoring: reward quality, penalize manipulation.
    """
    score = (
        0.25 * features["info_density"]
        + 0.25 * features["credibility"]
        + 0.15 * features["bias_balance"]
        - 0.20 * features["clickbait_score"]
        - 0.15 * features["emotional_score"]
    )
    # Shift to [0, 1] range
    return float(np.clip((score + 0.35) / 0.70, 0, 1))


def build_dataset(scrapes_dir: str, output_dir: str):
    """
    Build the full training dataset from scraped YouTube data.

    Produces:
    - features.csv: Feature matrix for all videos
    - For each query group: engagement labels and agent labels
    """
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for topic_file, user_goal in QUERY_GOALS.items():
        filepath = os.path.join(scrapes_dir, f"{topic_file}_final.json")
        if not os.path.exists(filepath):
            print(f"Skipping {topic_file} — file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {topic_file}")
        print(f"User goal: \"{user_goal}\"")

        with open(filepath, "r", encoding="utf-8") as f:
            videos = json.load(f)

        print(f"  Computing features for {len(videos)} videos...")
        for i, video in enumerate(videos):
            if i % 50 == 0:
                print(f"    {i}/{len(videos)}...")

            features = compute_features(video, user_goal)

            # Compute both labels
            engagement_label = compute_engagement_label(features)
            agent_label = compute_agent_label(features)

            row = {
                "video_id": video.get("video_id", ""),
                "title": video.get("title", ""),
                "query_group": topic_file,
                "user_goal": user_goal,
                # Input features for the model
                "info_density": features["info_density"],
                "credibility": features["credibility"],
                "bias_balance": features["bias_balance"],
                "clickbait_score": features["clickbait_score"],
                "emotional_score": features["emotional_score"],
                "duration_minutes": features["duration_minutes"],
                "title_length": features["title_length"],
                "desc_length": features["desc_length"],
                "tag_count": features["tag_count"],
                "has_transcript": features["has_transcript"],
                # Labels
                "engagement_label": engagement_label,
                "agent_label": agent_label,
                # Raw engagement metrics (for analysis, not training)
                "view_count": features["_view_count"],
                "like_count": features["_like_count"],
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Save full dataset
    output_path = os.path.join(output_dir, "rs_training_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Dataset built: {len(df)} samples")
    print(f"  Query groups: {df['query_group'].nunique()}")
    print(f"  Videos with transcripts: {df['has_transcript'].sum():.0f}")
    print(f"  Saved to: {output_path}")

    # Print label statistics
    print(f"\n  Engagement label — mean: {df['engagement_label'].mean():.3f}, std: {df['engagement_label'].std():.3f}")
    print(f"  Agent label      — mean: {df['agent_label'].mean():.3f}, std: {df['agent_label'].std():.3f}")
    print(f"  Correlation between labels: {df['engagement_label'].corr(df['agent_label']):.3f}")
    print(f"  (Low correlation = the two systems disagree = our project has a point!)")

    return df


if __name__ == "__main__":
    build_dataset(
        scrapes_dir="data/scrapes",
        output_dir="data/processed"
    )
