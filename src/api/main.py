"""
src/api/main.py
================
Shield Product API — FastAPI Backend

Endpoints:
  GET  /api/health                     Health check
  POST /api/search                     Search YouTube + agent scoring
  POST /api/compare                    Side-by-side YouTube vs Shield
  GET  /api/catalog                    Browse pre-cached video catalog
  POST /api/users                      Create user
  GET  /api/users                      List users
  GET  /api/users/{user_id}            Get user profile
  PUT  /api/users/{user_id}/prefs      Update preferences
  POST /api/users/{user_id}/history    Log watch event
  GET  /api/users/{user_id}/history    Get watch history
  GET  /api/users/{user_id}/recommend  Personalized recommendations
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.database import (
    create_user, get_user, update_user_preferences,
    list_users, log_watch, get_history, get_user_topic_stats,
)

app = FastAPI(
    title="Shield",
    description="YouTube's algorithm is designed to hack your brain. Shield is designed to hack YouTube's algorithm.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazy-loaded scoring modules (avoid cold-start delay on import)
# ---------------------------------------------------------------------------

_models_loaded = False


def _ensure_models():
    """Pre-load ML models into memory on first use."""
    global _models_loaded
    if _models_loaded:
        return
    print("[Shield] Pre-loading ML scoring models...")
    t0 = time.time()
    try:
        from src.manipulation.clickbait import _get_classifier as get_cb
        get_cb()
        print("  [OK] Clickbait detector loaded")
    except Exception as e:
        print(f"  [WARN] Clickbait model failed: {e}")
    try:
        from src.quality.credibility import _get_classifier as get_cr
        get_cr()
        print("  [OK] Credibility classifier loaded")
    except Exception as e:
        print(f"  [WARN] Credibility model failed: {e}")
    _models_loaded = True
    print(f"[Shield] Models ready in {time.time()-t0:.1f}s")


def score_video(video: dict, user_goal: str = "") -> dict:
    """
    Score a single video with the full agent pipeline.
    Returns the video dict augmented with agent scores.
    """
    _ensure_models()

    title = video.get("title", "")
    description = video.get("description", "")

    # Clickbait score (0-1, higher = more clickbait)
    try:
        from src.manipulation.clickbait import compute_clickbait_score
        cb_score = compute_clickbait_score(title)
    except Exception:
        cb_score = 0.3  # fallback

    # Emotional manipulation score (0-1)
    try:
        from src.manipulation.emotion import compute_emotional_manipulation_score
        em_score = compute_emotional_manipulation_score(title, description)
    except Exception:
        em_score = 0.3

    # Credibility score (0-1, higher = more credible)
    try:
        from src.quality.credibility import compute_credibility_score
        cr_score = compute_credibility_score(
            title=title,
            description=description,
            channel_name=video.get("channel_name", video.get("channel", "")),
            tags=video.get("tags", [])
        )
    except Exception:
        cr_score = 0.5

    # Info density score (0-1, higher = more informative)
    try:
        from src.quality.info_density import compute_info_density_score
        id_score = compute_info_density_score(
            title=title,
            description=description,
            duration_seconds=video.get("duration_seconds", 0),
            transcript=video.get("transcript")
        )
    except Exception:
        id_score = 0.5

    # Goal alignment (0-1, higher = more relevant)
    ga_score = 0.5
    if user_goal:
        try:
            from src.quality.goal_alignment import compute_goal_alignment
            ga_score = compute_goal_alignment(user_goal, title, description)
        except Exception:
            ga_score = 0.5

    # Shield composite score (clickbait penalty boosted to 0.30 so
    # high-clickbait videos get ranked significantly lower)
    shield_score = (
        0.20 * cr_score
        + 0.20 * id_score
        + 0.15 * ga_score
        - 0.30 * cb_score     # STRONG clickbait penalty
        - 0.15 * em_score     # emotional manipulation penalty
    )
    # Rescale from [-0.45, 0.55] range to [0, 1]
    shield_score = max(0.0, min(1.0, (shield_score + 0.45) / 1.0))

    video["agent_scores"] = {
        "shield_score": round(shield_score, 3),
        "clickbait": round(cb_score, 3),
        "emotional_manipulation": round(em_score, 3),
        "credibility": round(cr_score, 3),
        "info_density": round(id_score, 3),
        "goal_alignment": round(ga_score, 3),
    }
    # Ensure social proof metrics are always present for the compare view
    if "view_count" not in video:
        video["view_count"] = video.get("view_count", 0)
    if "like_count" not in video:
        video["like_count"] = video.get("like_count", 0)
    if "subscriber_count" not in video:
        video["subscriber_count"] = video.get("subscriber_count", 0)
    return video


def search_youtube(query: str, max_results: int = 20) -> list[dict]:
    """Search YouTube and get video details."""
    try:
        from src.scraper.youtube_scraper import scrape_topic
        results = scrape_topic(query, max_results=max_results)
        return results
    except Exception as e:
        print(f"[ERROR] YouTube search failed: {e}")
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# Pre-cached catalog (scraped videos from 5 topics)
# ---------------------------------------------------------------------------

_catalog_cache: Optional[list[dict]] = None


def _load_catalog() -> list[dict]:
    """Load pre-scraped videos from data/scrapes/ as a browsable catalog."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    scrapes_dir = PROJECT_ROOT / "data" / "scrapes"
    catalog = []
    topic_map = {
        "machine_learning_tutorial": "Machine Learning",
        "learn_about_climate_change": "Climate Change",
        "how_to_invest_for_beginners": "Investing",
        "history_of_ancient_rome": "Ancient Rome",
        "healthy_meal_prep": "Healthy Cooking",
    }

    for filename in sorted(scrapes_dir.glob("*_final.json")):
        try:
            with open(filename, encoding="utf-8") as f:
                videos = json.load(f)

            # Infer topic from filename
            stem = filename.stem.replace("_final", "")
            topic_label = topic_map.get(stem, stem.replace("_", " ").title())

            for v in videos:
                v["topic"] = topic_label
                v["topic_key"] = stem
                # Ensure thumbnail exists
                if not v.get("thumbnail"):
                    vid = v.get("video_id", "")
                    v["thumbnail"] = f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
            catalog.extend(videos)
        except Exception as e:
            print(f"[WARN] Failed to load {filename}: {e}")

    _catalog_cache = catalog
    print(f"[Shield] Loaded catalog: {len(catalog)} videos from {len(topic_map)} topics")
    return catalog


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    max_results: int = 20
    time_budget_minutes: int = 60
    quality_preference: str = "balanced"  # "balanced" | "scientific" | "casual"


class CreateUserRequest(BaseModel):
    name: str
    preferences: Optional[Dict[str, Any]] = None


class UpdatePrefsRequest(BaseModel):
    preferences: Dict[str, Any]


class WatchEventRequest(BaseModel):
    video_id: str
    title: str = ""
    channel: str = ""
    thumbnail: str = ""
    duration_seconds: int = 0
    watch_pct: float = 1.0
    agent_scores: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "shield", "version": "2.0.0"}


@app.post("/api/search")
def search(req: SearchRequest):
    """
    Search YouTube + score with full Shield agent pipeline.
    Returns videos ranked by Shield score (not YouTube popularity).
    """
    t0 = time.time()
    print(f"\n[Search] Query: '{req.query}' | max={req.max_results}")

    # 1. Fetch from YouTube API
    raw_results = search_youtube(req.query, max_results=req.max_results)
    if not raw_results:
        # Fallback: search pre-cached catalog
        print("[Search] YouTube API returned nothing, searching catalog...")
        catalog = _load_catalog()
        query_lower = req.query.lower()
        raw_results = [
            v for v in catalog
            if query_lower in v.get("title", "").lower()
            or query_lower in v.get("description", "").lower()
            or query_lower in v.get("topic", "").lower()
        ][:req.max_results]

    if not raw_results:
        return {"videos": [], "metrics": {}, "elapsed_ms": 0}

    # 2. Score each video with agent pipeline
    scored = []
    for v in raw_results:
        scored_v = score_video(v.copy(), user_goal=req.query)
        scored.append(scored_v)

    # 3. Filter by time budget
    time_budget_s = req.time_budget_minutes * 60
    total_dur = 0
    within_budget = []
    for v in scored:
        dur = v.get("duration_seconds", 0)
        if total_dur + dur <= time_budget_s or not within_budget:
            within_budget.append(v)
            total_dur += dur

    # 4. Sort by Shield score (highest first)
    within_budget.sort(
        key=lambda v: v.get("agent_scores", {}).get("shield_score", 0),
        reverse=True,
    )

    elapsed = int((time.time() - t0) * 1000)

    # Compute aggregate metrics
    if within_budget:
        avg_shield = sum(v["agent_scores"]["shield_score"] for v in within_budget) / len(within_budget)
        avg_clickbait = sum(v["agent_scores"]["clickbait"] for v in within_budget) / len(within_budget)
        avg_credibility = sum(v["agent_scores"]["credibility"] for v in within_budget) / len(within_budget)
    else:
        avg_shield = avg_clickbait = avg_credibility = 0

    return {
        "videos": within_budget,
        "metrics": {
            "total_results": len(within_budget),
            "total_candidates_scored": len(scored),
            "avg_shield_score": round(avg_shield, 3),
            "avg_clickbait": round(avg_clickbait, 3),
            "avg_credibility": round(avg_credibility, 3),
            "total_duration_minutes": round(total_dur / 60, 1),
        },
        "elapsed_ms": elapsed,
    }


@app.post("/api/compare")
def compare(req: SearchRequest):
    """
    Side-by-side: YouTube ranking (by view count) vs Shield ranking (by agent score).
    Same videos, different ordering.
    """
    t0 = time.time()

    raw_results = search_youtube(req.query, max_results=req.max_results)
    if not raw_results:
        catalog = _load_catalog()
        query_lower = req.query.lower()
        raw_results = [
            v for v in catalog
            if query_lower in v.get("title", "").lower()
            or query_lower in v.get("description", "").lower()
            or query_lower in v.get("topic", "").lower()
        ][:req.max_results]

    scored = [score_video(v.copy(), user_goal=req.query) for v in raw_results]

    # YouTube ranking: by view count (engagement-driven)
    youtube_ranked = sorted(
        scored, key=lambda v: v.get("view_count", 0), reverse=True
    )

    # Shield ranking: by agent score (quality-driven)
    shield_ranked = sorted(
        scored,
        key=lambda v: v.get("agent_scores", {}).get("shield_score", 0),
        reverse=True,
    )

    def avg_metric(vids, key):
        vals = [v.get("agent_scores", {}).get(key, 0) for v in vids]
        return round(sum(vals) / max(len(vals), 1), 3)

    return {
        "query": req.query,
        "youtube": {
            "videos": youtube_ranked,
            "metrics": {
                "avg_shield_score": avg_metric(youtube_ranked, "shield_score"),
                "avg_clickbait": avg_metric(youtube_ranked, "clickbait"),
                "avg_credibility": avg_metric(youtube_ranked, "credibility"),
            },
        },
        "shield": {
            "videos": shield_ranked,
            "metrics": {
                "avg_shield_score": avg_metric(shield_ranked, "shield_score"),
                "avg_clickbait": avg_metric(shield_ranked, "clickbait"),
                "avg_credibility": avg_metric(shield_ranked, "credibility"),
            },
        },
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


@app.get("/api/catalog")
def get_catalog(topic: Optional[str] = None, limit: int = 50):
    """Browse the pre-cached video catalog (no YouTube API call needed)."""
    catalog = _load_catalog()
    if topic:
        catalog = [v for v in catalog if v.get("topic_key", "") == topic]

    # Score a subset for display
    scored = []
    for v in catalog[:limit]:
        scored_v = score_video(v.copy(), user_goal="")
        scored.append(scored_v)

    scored.sort(
        key=lambda v: v.get("agent_scores", {}).get("shield_score", 0),
        reverse=True,
    )

    topics = sorted(set(v.get("topic", "Unknown") for v in _load_catalog()))
    return {"videos": scored, "topics": topics, "total": len(catalog)}


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------

@app.post("/api/users")
def api_create_user(req: CreateUserRequest):
    return create_user(req.name, req.preferences)


@app.get("/api/users")
def api_list_users():
    return list_users()


@app.get("/api/users/{user_id}")
def api_get_user(user_id: str):
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["history"] = get_history(user_id, limit=20)
    user["topic_stats"] = get_user_topic_stats(user_id)
    return user


@app.put("/api/users/{user_id}/prefs")
def api_update_prefs(user_id: str, req: UpdatePrefsRequest):
    if not get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    update_user_preferences(user_id, req.preferences)
    return {"status": "updated"}


@app.post("/api/users/{user_id}/history")
def api_log_watch(user_id: str, req: WatchEventRequest):
    if not get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return log_watch(
        user_id=user_id,
        video_id=req.video_id,
        title=req.title,
        channel=req.channel,
        thumbnail=req.thumbnail,
        duration_seconds=req.duration_seconds,
        watch_pct=req.watch_pct,
        agent_scores=req.agent_scores,
    )


@app.get("/api/users/{user_id}/history")
def api_get_history(user_id: str, limit: int = 50):
    if not get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return get_history(user_id, limit=limit)


@app.get("/api/users/{user_id}/recommend")
def api_recommend(user_id: str, top_k: int = 10):
    """
    Personalized recommendations based on user's watch history.
    Uses the pre-cached catalog + collaborative filtering signals.
    """
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    history = get_history(user_id, limit=100)
    watched_ids = {h["video_id"] for h in history}

    # Infer preferred topics from history
    topic_counts = get_user_topic_stats(user_id)
    preferred_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)[:3] if topic_counts else []

    # Get catalog and filter to unwatched + preferred topics
    catalog = _load_catalog()
    candidates = [
        v for v in catalog
        if v.get("video_id") not in watched_ids
    ]

    # Boost preferred topics
    for v in candidates:
        topic = v.get("topic", "")
        v["_topic_boost"] = 1.5 if topic in [t for t in preferred_topics] else 1.0

    # Score and rank
    scored = []
    for v in candidates[:60]:  # Score top 60 candidates
        sv = score_video(v.copy(), user_goal="")
        shield = sv.get("agent_scores", {}).get("shield_score", 0)
        sv["_final_score"] = shield * v.get("_topic_boost", 1.0)
        scored.append(sv)

    scored.sort(key=lambda v: v.get("_final_score", 0), reverse=True)

    # Clean up internal keys
    for v in scored:
        v.pop("_topic_boost", None)
        v.pop("_final_score", None)

    return {
        "user_id": user_id,
        "recommendations": scored[:top_k],
        "preferred_topics": preferred_topics,
        "history_count": len(history),
    }


# ---------------------------------------------------------------------------
# Gemini Explainer
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini():
    """Lazy-load the Gemini client."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("[WARN] GEMINI_API_KEY not set, explanations disabled")
            return None
        _gemini_client = genai.Client(api_key=api_key)
        print("[Shield] Gemini client initialized")
        return _gemini_client
    except Exception as e:
        print(f"[WARN] Gemini init failed: {e}")
        return None


class ExplainRequest(BaseModel):
    title: str
    channel: str = ""
    description: str = ""
    agent_scores: Dict[str, float] = {}
    query: str = ""


@app.post("/api/explain")
def explain_video(req: ExplainRequest):
    """
    Generate a human-readable explanation of why Shield scored this video
    the way it did, using Gemini Flash (fast, cheap).
    """
    scores = req.agent_scores
    shield = scores.get("shield_score", 0)
    cb = scores.get("clickbait", 0)
    cred = scores.get("credibility", 0)
    info = scores.get("info_density", 0)
    emo = scores.get("emotional_manipulation", 0)
    goal = scores.get("goal_alignment", 0)

    prompt = f"""You are Shield, an AI agent that protects users from manipulative YouTube content.

A user searched for: "{req.query}"

Video: "{req.title}" by {req.channel}
Description: {req.description[:300]}

Shield Agent Scores (0-100 scale):
- Shield Score: {int(shield*100)} (composite quality score)
- Clickbait: {int(cb*100)} (higher = more clickbait)
- Credibility: {int(cred*100)} (higher = more credible)
- Info Density: {int(info*100)} (higher = more informative)
- Emotional Manipulation: {int(emo*100)} (higher = more manipulative)
- Goal Alignment: {int(goal*100)} (how relevant to the search)

Write a 2-3 sentence explanation for the user about WHY this video received this Shield Score.
Be specific about what's good or concerning. If clickbait is high, explain what makes it clickbait.
If credibility is high, mention why. Keep it conversational and helpful, not academic.
Do NOT use bullet points. Do NOT use markdown. Just plain sentences."""

    client = _get_gemini()
    if not client:
        # Fallback: generate a simple rule-based explanation
        return {"explanation": _fallback_explanation(scores, req.title)}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()
        return {"explanation": text}
    except Exception as e:
        print(f"[WARN] Gemini explain failed: {e}")
        return {"explanation": _fallback_explanation(scores, req.title)}


def _fallback_explanation(scores: dict, title: str) -> str:
    """Rule-based explanation when Gemini is unavailable."""
    parts = []
    cb = scores.get("clickbait", 0)
    cred = scores.get("credibility", 0)
    info = scores.get("info_density", 0)

    if cb > 0.7:
        parts.append(f"This title has strong clickbait signals (score: {int(cb*100)}%), which significantly lowers its Shield ranking.")
    elif cb > 0.4:
        parts.append(f"Some clickbait patterns detected in the title (score: {int(cb*100)}%).")
    else:
        parts.append(f"The title appears genuine with low clickbait (score: {int(cb*100)}%).")

    if cred > 0.6:
        parts.append(f"The content shows good credibility indicators ({int(cred*100)}%).")
    elif cred < 0.4:
        parts.append(f"Credibility indicators are below average ({int(cred*100)}%).")

    if info > 0.6:
        parts.append(f"High information density ({int(info*100)}%) suggests substantive content.")

    return " ".join(parts)


if __name__ == "__main__":
    import uvicorn
    print("[Shield] Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
