"""
src/api/main.py
================
Shield v3 — Product API

Endpoints:
  GET  /api/health                       Health check
  GET  /api/feed                         Personalized feed (instant, pre-scored)
  POST /api/search                       Live YouTube search + scoring
  POST /api/compare                      Side-by-side YouTube vs Shield
  POST /api/auth/register                Register new user
  POST /api/auth/login                   Login
  POST /api/users                        Legacy create user
  GET  /api/users                        List users
  GET  /api/users/{user_id}              Get user profile
  PUT  /api/users/{user_id}/settings     Update settings
  POST /api/users/{user_id}/history      Log watch event
  GET  /api/users/{user_id}/history      Get watch history
  GET  /api/users/{user_id}/recommend    Personalized recommendations
  POST /api/explain                      Gemini explanation
"""

import os
import sys
import json
import time
import random
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.database import (
    register_user, login_user, get_user, update_user_settings,
    list_users, log_watch, get_history, get_user_topic_stats,
)

app = FastAPI(
    title="Shield",
    description="AI-powered YouTube content quality filter",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pre-scored catalog (loaded once at startup)
# ---------------------------------------------------------------------------

_scored_catalog: Optional[dict] = None
_catalog_topics: list[str] = []


def _load_scored_catalog() -> dict:
    """Load pre-scored catalog from JSON (built by scripts/build_catalog.py)."""
    global _scored_catalog, _catalog_topics

    if _scored_catalog is not None:
        return _scored_catalog

    catalog_path = PROJECT_ROOT / "data" / "scored_catalog.json"
    if catalog_path.exists():
        with open(catalog_path, encoding="utf-8") as f:
            _scored_catalog = json.load(f)
        _catalog_topics = _scored_catalog.get("topics", [])
        print(f"[Shield] Loaded pre-scored catalog: {_scored_catalog.get('total_videos', 0)} videos, {len(_catalog_topics)} topics")
        return _scored_catalog

    # Fallback: load from raw scrapes and score on-the-fly
    print("[Shield] No pre-scored catalog found, loading from raw scrapes...")
    return _load_fallback_catalog()


def _load_fallback_catalog() -> dict:
    """Fallback: load raw scrapes when no pre-scored catalog exists."""
    global _scored_catalog, _catalog_topics

    scrapes_dir = PROJECT_ROOT / "data" / "scrapes"
    if not scrapes_dir.exists():
        _scored_catalog = {"videos": [], "topics": []}
        return _scored_catalog

    topic_map = {
        "machine_learning_tutorial": "Machine Learning / AI",
        "learn_about_climate_change": "Climate Change",
        "how_to_invest_for_beginners": "Investing / Finance",
        "history_of_ancient_Rome": "History",
        "healthy_meal_prep": "Healthy Cooking",
    }

    all_videos = []
    topics = []

    for filename in sorted(scrapes_dir.glob("*_final.json")):
        try:
            with open(filename, encoding="utf-8") as f:
                videos = json.load(f)
            stem = filename.stem.replace("_final", "")
            topic_label = topic_map.get(stem, stem.replace("_", " ").title())
            if topic_label not in topics:
                topics.append(topic_label)

            for v in videos:
                v["topic"] = topic_label
                v["topic_key"] = stem
                if not v.get("thumbnail"):
                    vid = v.get("video_id", "")
                    v["thumbnail"] = f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
            all_videos.extend(videos)
        except Exception as e:
            print(f"[WARN] Failed to load {filename}: {e}")

    _catalog_topics = topics
    _scored_catalog = {
        "videos": all_videos,
        "topics": topics,
        "total_videos": len(all_videos),
    }
    print(f"[Shield] Loaded fallback catalog: {len(all_videos)} videos from {len(topics)} topics")
    return _scored_catalog


# ---------------------------------------------------------------------------
# Lazy-loaded scoring modules
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
    """Score a single video with the full agent pipeline."""
    _ensure_models()

    title = video.get("title", "")
    description = video.get("description", "")

    # Clickbait score
    try:
        from src.manipulation.clickbait import compute_clickbait_score
        cb_score = compute_clickbait_score(title)
    except Exception:
        cb_score = 0.3

    # Emotional manipulation score
    try:
        from src.manipulation.emotion import compute_emotional_manipulation_score
        em_score = compute_emotional_manipulation_score(title, description)
    except Exception:
        em_score = 0.3

    # Credibility score
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

    # Info density score
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

    # Goal alignment
    ga_score = 0.5
    if user_goal:
        try:
            from src.quality.goal_alignment import compute_goal_alignment
            ga_score = compute_goal_alignment(user_goal, title, description)
        except Exception:
            ga_score = 0.5

    # Shield composite score
    shield_score = (
        0.20 * cr_score
        + 0.20 * id_score
        + 0.15 * ga_score
        - 0.30 * cb_score
        - 0.15 * em_score
    )
    shield_score = max(0.0, min(1.0, (shield_score + 0.45) / 1.0))

    video["agent_scores"] = {
        "shield_score": round(shield_score, 3),
        "clickbait": round(cb_score, 3),
        "emotional_manipulation": round(em_score, 3),
        "credibility": round(cr_score, 3),
        "info_density": round(id_score, 3),
        "goal_alignment": round(ga_score, 3),
    }

    # Ensure social metrics
    for key in ("view_count", "like_count", "subscriber_count"):
        if key not in video:
            video[key] = 0

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
# Request/Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    max_results: int = 20
    time_budget_minutes: int = 60
    quality_preference: str = "balanced"

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    location: str = ""
    language: str = "en"
    preferred_topics: List[str] = []
    content_strictness: str = "balanced"

class LoginRequest(BaseModel):
    email: str
    password: str

class UpdateSettingsRequest(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    language: Optional[str] = None
    preferred_topics: Optional[List[str]] = None
    content_strictness: Optional[str] = None
    dark_mode: Optional[bool] = None

class WatchEventRequest(BaseModel):
    video_id: str
    title: str = ""
    channel: str = ""
    thumbnail: str = ""
    duration_seconds: int = 0
    watch_pct: float = 1.0
    agent_scores: Optional[Dict[str, Any]] = None

class ExplainRequest(BaseModel):
    title: str
    channel: str = ""
    description: str = ""
    agent_scores: Dict[str, float] = {}
    query: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "shield", "version": "3.0.0"}


# ── Feed ──

@app.get("/api/feed")
def get_feed(user_id: Optional[str] = None, topic: Optional[str] = None, limit: int = 60):
    """
    Instant personalized feed from pre-scored catalog.
    No ML scoring at request time — everything is pre-computed.
    """
    catalog = _load_scored_catalog()
    videos = list(catalog.get("videos", []))

    # Filter by topic
    if topic and topic != "all":
        videos = [v for v in videos if v.get("topic") == topic]

    # Personalize if user is logged in
    if user_id:
        user = get_user(user_id)
        if user:
            preferred = user.get("preferred_topics", [])
            if preferred:
                # Boost preferred topics by putting them first
                preferred_vids = [v for v in videos if v.get("topic_key") in preferred]
                other_vids = [v for v in videos if v.get("topic_key") not in preferred]
                videos = preferred_vids + other_vids

    # If catalog has pre-computed scores, sort by shield score
    videos.sort(
        key=lambda v: v.get("agent_scores", {}).get("shield_score", 0),
        reverse=True,
    )

    # Diversify: don't show 10 videos from the same topic in a row
    diversified = _diversify_feed(videos, max_per_topic_consecutive=3)

    return {
        "videos": diversified[:limit],
        "topics": catalog.get("topics", []),
        "total": len(videos),
    }


def _diversify_feed(videos: list, max_per_topic_consecutive: int = 3) -> list:
    """Re-order to avoid too many videos from the same topic in a row."""
    if len(videos) <= max_per_topic_consecutive:
        return videos

    result = []
    remaining = list(videos)
    last_topics = []

    while remaining:
        placed = False
        for i, v in enumerate(remaining):
            topic = v.get("topic", "")
            recent = last_topics[-max_per_topic_consecutive:]
            if topic not in recent or len(remaining) <= max_per_topic_consecutive:
                result.append(remaining.pop(i))
                last_topics.append(topic)
                placed = True
                break
        if not placed:
            result.append(remaining.pop(0))

    return result


# ── Search ──

@app.post("/api/search")
def search(req: SearchRequest):
    """Live YouTube search + Shield scoring."""
    t0 = time.time()
    print(f"\n[Search] Query: '{req.query}' | max={req.max_results}")

    raw_results = search_youtube(req.query, max_results=req.max_results)
    if not raw_results:
        # Fallback: search catalog
        catalog = _load_scored_catalog()
        q_lower = req.query.lower()
        raw_results = [
            v for v in catalog.get("videos", [])
            if q_lower in v.get("title", "").lower()
            or q_lower in v.get("description", "").lower()
            or q_lower in v.get("topic", "").lower()
        ][:req.max_results]

    if not raw_results:
        return {"videos": [], "metrics": {}, "elapsed_ms": 0}

    scored = [score_video(v.copy(), user_goal=req.query) for v in raw_results]
    scored.sort(key=lambda v: v.get("agent_scores", {}).get("shield_score", 0), reverse=True)

    elapsed = int((time.time() - t0) * 1000)

    if scored:
        avg_shield = sum(v["agent_scores"]["shield_score"] for v in scored) / len(scored)
        avg_cb = sum(v["agent_scores"]["clickbait"] for v in scored) / len(scored)
        avg_cred = sum(v["agent_scores"]["credibility"] for v in scored) / len(scored)
        total_dur = sum(v.get("duration_seconds", 0) for v in scored)
    else:
        avg_shield = avg_cb = avg_cred = total_dur = 0

    return {
        "videos": scored,
        "metrics": {
            "total_results": len(scored),
            "avg_shield_score": round(avg_shield, 3),
            "avg_clickbait": round(avg_cb, 3),
            "avg_credibility": round(avg_cred, 3),
            "total_duration_minutes": round(total_dur / 60, 1),
        },
        "elapsed_ms": elapsed,
    }


# ── Compare ──

@app.post("/api/compare")
def compare(req: SearchRequest):
    """Side-by-side: YouTube ranking vs Shield ranking."""
    t0 = time.time()

    raw_results = search_youtube(req.query, max_results=req.max_results)
    if not raw_results:
        catalog = _load_scored_catalog()
        q_lower = req.query.lower()
        raw_results = [
            v for v in catalog.get("videos", [])
            if q_lower in v.get("title", "").lower()
            or q_lower in v.get("description", "").lower()
            or q_lower in v.get("topic", "").lower()
        ][:req.max_results]

    scored = [score_video(v.copy(), user_goal=req.query) for v in raw_results]

    youtube_ranked = sorted(scored, key=lambda v: v.get("view_count", 0), reverse=True)
    shield_ranked = sorted(scored, key=lambda v: v.get("agent_scores", {}).get("shield_score", 0), reverse=True)

    def avg_m(vids, key):
        vals = [v.get("agent_scores", {}).get(key, 0) for v in vids]
        return round(sum(vals) / max(len(vals), 1), 3)

    return {
        "query": req.query,
        "youtube": {
            "videos": youtube_ranked,
            "metrics": {
                "avg_shield_score": avg_m(youtube_ranked, "shield_score"),
                "avg_clickbait": avg_m(youtube_ranked, "clickbait"),
                "avg_credibility": avg_m(youtube_ranked, "credibility"),
            },
        },
        "shield": {
            "videos": shield_ranked,
            "metrics": {
                "avg_shield_score": avg_m(shield_ranked, "shield_score"),
                "avg_clickbait": avg_m(shield_ranked, "clickbait"),
                "avg_credibility": avg_m(shield_ranked, "credibility"),
            },
        },
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


# ── Auth ──

@app.post("/api/auth/register")
def api_register(req: RegisterRequest):
    try:
        user = register_user(
            name=req.name,
            email=req.email,
            password=req.password,
            location=req.location,
            language=req.language,
            preferred_topics=req.preferred_topics,
            content_strictness=req.content_strictness,
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
def api_login(req: LoginRequest):
    user = login_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return user


# ── Users ──

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


@app.put("/api/users/{user_id}/settings")
def api_update_settings(user_id: str, req: UpdateSettingsRequest):
    updated = update_user_settings(
        user_id=user_id,
        name=req.name,
        location=req.location,
        language=req.language,
        preferred_topics=req.preferred_topics,
        content_strictness=req.content_strictness,
        dark_mode=req.dark_mode,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return updated


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
    """Personalized recommendations from pre-scored catalog."""
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    history = get_history(user_id, limit=100)
    watched_ids = {h["video_id"] for h in history}

    preferred_topics = user.get("preferred_topics", [])

    catalog = _load_scored_catalog()
    candidates = [v for v in catalog.get("videos", []) if v.get("video_id") not in watched_ids]

    # Boost preferred topics
    for v in candidates:
        boost = 1.5 if v.get("topic_key") in preferred_topics else 1.0
        v["_score"] = v.get("agent_scores", {}).get("shield_score", 0) * boost

    candidates.sort(key=lambda v: v.get("_score", 0), reverse=True)

    for v in candidates:
        v.pop("_score", None)

    return {
        "user_id": user_id,
        "recommendations": candidates[:top_k],
        "preferred_topics": preferred_topics,
        "history_count": len(history),
    }


# ── Catalog (legacy compat) ──

@app.get("/api/catalog")
def get_catalog(topic: Optional[str] = None, limit: int = 50):
    catalog = _load_scored_catalog()
    videos = list(catalog.get("videos", []))
    if topic:
        videos = [v for v in videos if v.get("topic_key") == topic]
    return {
        "videos": videos[:limit],
        "topics": catalog.get("topics", []),
        "total": len(videos),
    }


# ---------------------------------------------------------------------------
# Gemini Explainer
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini():
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


@app.post("/api/explain")
def explain_video(req: ExplainRequest):
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
Be specific about what's good or concerning. Keep it conversational and helpful.
Do NOT use bullet points. Do NOT use markdown. Just plain sentences."""

    client = _get_gemini()
    if not client:
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


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    """Load catalog at startup so feed is instant."""
    _load_scored_catalog()


if __name__ == "__main__":
    import uvicorn
    print("[Shield] Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
