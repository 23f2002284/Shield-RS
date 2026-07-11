"""
End-to-End Pipeline: AgentShield

Orchestrates all modules into a single recommend() call:
1. Fetch candidates from YouTube API
2. Strip social proof signals
3. Run Manipulation Detector on all candidates
4. Run Content Quality Evaluator on all candidates
5. Run Goal-Constrained Optimizer to select the best subset
6. Generate explanations for selected + filtered items
7. Return curated set with explanations
"""

import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.scraper.youtube_scraper import scrape_topic
from src.manipulation.social_proof import strip_social_proof_batch
from src.manipulation.clickbait import compute_clickbait_score
from src.manipulation.emotion import compute_emotional_manipulation_score
from src.optimizer.selector import select_optimal_set
from src.optimizer.weights import DEFAULT_WEIGHTS, get_weights_for_preference
from src.explainer.explain import explain_selection

load_dotenv()


class AgentShield:
    """
    The Agent-as-Shield Recommender System.

    Usage:
        agent = AgentShield()
        result = agent.recommend(
            user_goal="learn about climate change",
            time_budget_minutes=30,
            preferences={"style": "scientific"}
        )
    """

    def recommend(
        self,
        user_goal: str,
        time_budget_minutes: int = 30,
        max_items: int = 10,
        preferences: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[Dict[str, Any]]] = None,
        fetch_count: int = 50,
    ) -> Dict[str, Any]:
        """
        Run the full Agent-as-Shield pipeline.

        Args:
            user_goal: What the user wants to learn/find.
            time_budget_minutes: Maximum total time for recommended videos.
            max_items: Maximum number of videos to recommend.
            preferences: User meta-preferences (e.g., {"style": "scientific"}).
            candidates: Pre-fetched candidates (skip YouTube API call if provided).
            fetch_count: Number of candidates to fetch from YouTube if not provided.

        Returns:
            Dict with recommended videos, filtered videos, explanations, and metrics.
        """
        if preferences is None:
            preferences = {}

        print(f"🛡️ Agent-as-Shield: Processing goal: \"{user_goal}\"")
        print(f"   Time budget: {time_budget_minutes} min | Max items: {max_items}")

        # ── Step 1: Get candidates ──
        if candidates is None:
            print(f"\n📡 Step 1: Fetching {fetch_count} candidates from YouTube...")
            candidates = scrape_topic(user_goal, max_results=fetch_count)
            print(f"   Fetched {len(candidates)} candidates.")
        else:
            print(f"\n📡 Step 1: Using {len(candidates)} pre-fetched candidates.")

        if not candidates:
            return {"error": "No candidates found.", "recommended": [], "filtered": [], "metrics": {}}

        # ── Step 2: Strip social proof signals ──
        print("\n🚫 Step 2: Stripping social proof signals (views, likes, subscribers)...")
        # Keep originals for reporting, use stripped for scoring
        originals = {v["video_id"]: v for v in candidates}
        stripped_candidates = strip_social_proof_batch(candidates)

        # ── Step 3: Score manipulation signals ──
        print("\n🔍 Step 3: Scoring manipulation signals...")
        for v in stripped_candidates:
            v["_clickbait_score"] = compute_clickbait_score(v.get("title", ""))
            v["_emotional_score"] = compute_emotional_manipulation_score(
                v.get("title", ""), v.get("description", "")
            )

        # ── Step 4: Get weights based on preferences ──
        style = preferences.get("style", "default")
        weights = get_weights_for_preference(style)
        print(f"\n⚙️ Step 4: Using '{style}' weight profile.")

        # ── Step 5: Run optimizer ──
        time_budget_seconds = time_budget_minutes * 60
        print(f"\n🎯 Step 5: Running goal-constrained optimizer...")
        recommended = select_optimal_set(
            candidates=stripped_candidates,
            user_goal=user_goal,
            time_budget_seconds=time_budget_seconds,
            max_items=max_items,
            weights=weights,
        )

        # ── Step 6: Identify filtered items ──
        recommended_ids = {v["video_id"] for v in recommended}
        filtered = [v for v in stripped_candidates if v["video_id"] not in recommended_ids]

        # Sort filtered by clickbait score (highest first) for the explanation
        filtered.sort(key=lambda v: v.get("_clickbait_score", 0), reverse=True)

        # ── Step 7: Generate explanations ──
        print(f"\n📝 Step 6: Generating explanations...")
        explanations = explain_selection(recommended, filtered, user_goal)

        # ── Step 8: Compute session metrics ──
        total_duration = sum(v.get("duration_seconds", 0) for v in recommended)
        avg_clickbait = (
            sum(v.get("_clickbait_score", 0) for v in recommended) / max(len(recommended), 1)
        )
        avg_emotional = (
            sum(v.get("_emotional_score", 0) for v in recommended) / max(len(recommended), 1)
        )

        # Restore original social proof data for display purposes
        for v in recommended:
            orig = originals.get(v["video_id"], {})
            v["view_count"] = orig.get("view_count", 0)
            v["like_count"] = orig.get("like_count", 0)
            v["subscriber_count"] = orig.get("subscriber_count", 0)

        metrics = {
            "total_videos_recommended": len(recommended),
            "total_videos_filtered": len(filtered),
            "total_duration_minutes": round(total_duration / 60, 1),
            "time_budget_minutes": time_budget_minutes,
            "avg_clickbait_score": round(avg_clickbait, 3),
            "avg_emotional_score": round(avg_emotional, 3),
        }

        # Clean up internal scoring keys
        for v in recommended + filtered:
            v.pop("_clickbait_score", None)
            v.pop("_emotional_score", None)
            v.pop("_agent_score", None)

        print(f"\n✅ Done! Recommended {len(recommended)} videos "
              f"({metrics['total_duration_minutes']} min) "
              f"from {len(candidates)} candidates.")

        return {
            "user_goal": user_goal,
            "recommended": recommended,
            "filtered_sample": filtered[:10],
            "explanations": explanations,
            "metrics": metrics,
        }
