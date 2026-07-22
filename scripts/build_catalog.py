"""
scripts/build_catalog.py
=========================
Build the pre-scored video catalog for Shield.

Scrapes YouTube, fetches transcripts, scores with ML models,
generates Gemini explanations, and saves everything to
data/scored_catalog.json for instant frontend loading.

Usage:
    python scripts/build_catalog.py                    # Full build (all 20 topics)
    python scripts/build_catalog.py --topics 3         # First 3 topics only
    python scripts/build_catalog.py --max-per-topic 20 # 20 videos per topic
    python scripts/build_catalog.py --skip-existing    # Skip already-scraped topics
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ── 20 Topic Categories ──
TOPICS = [
    {"key": "machine_learning_ai", "query": "machine learning tutorial", "label": "Machine Learning / AI"},
    {"key": "climate_change", "query": "climate change explained", "label": "Climate Change"},
    {"key": "investing_finance", "query": "investing for beginners", "label": "Investing / Finance"},
    {"key": "history_ancient", "query": "ancient history documentary", "label": "History"},
    {"key": "healthy_cooking", "query": "healthy meal prep recipes", "label": "Healthy Cooking"},
    {"key": "fitness_workout", "query": "home workout routine", "label": "Fitness / Workout"},
    {"key": "psychology_mental_health", "query": "psychology explained", "label": "Psychology / Mental Health"},
    {"key": "space_astronomy", "query": "space documentary", "label": "Space / Astronomy"},
    {"key": "programming_webdev", "query": "web development tutorial", "label": "Programming / Web Dev"},
    {"key": "physics_math", "query": "physics explained", "label": "Physics / Math"},
    {"key": "music_theory", "query": "music theory for beginners", "label": "Music Theory"},
    {"key": "photography_film", "query": "filmmaking tutorial", "label": "Photography / Film"},
    {"key": "entrepreneurship", "query": "startup advice entrepreneur", "label": "Entrepreneurship"},
    {"key": "language_learning", "query": "language learning tips", "label": "Language Learning"},
    {"key": "philosophy_ethics", "query": "philosophy explained", "label": "Philosophy / Ethics"},
    {"key": "biology_medicine", "query": "biology explained science", "label": "Biology / Medicine"},
    {"key": "gaming_design", "query": "game design tutorial", "label": "Gaming / Game Design"},
    {"key": "politics_geopolitics", "query": "geopolitics explained", "label": "Politics / Geopolitics"},
    {"key": "art_design", "query": "digital art tutorial", "label": "Art / Design"},
    {"key": "diy_engineering", "query": "DIY engineering projects", "label": "DIY / Engineering"},
]


def scrape_topic_videos(query: str, max_results: int = 200) -> list[dict]:
    """Scrape videos from YouTube for a topic."""
    from src.scraper.youtube_scraper import scrape_topic
    return scrape_topic(query, max_results=max_results)


def fetch_transcripts_for_videos(videos: list[dict], max_tier: int = 1) -> list[dict]:
    """Fetch transcripts and attach to video dicts."""
    from src.scraper.transcript_fetcher import fetch_transcript

    total = len(videos)
    success = 0
    for i, v in enumerate(videos):
        if (i + 1) % 20 == 0:
            print(f"    Transcripts: {i+1}/{total}...")

        vid_id = v.get("video_id", "")
        if not vid_id:
            continue

        transcript = fetch_transcript(vid_id, max_tier=max_tier)
        if transcript:
            v["transcript"] = transcript
            success += 1

    print(f"    Transcripts: {success}/{total} fetched ({success/max(total,1)*100:.0f}%)")
    return videos


def score_videos(videos: list[dict], topic_query: str) -> list[dict]:
    """Score all videos with the ML pipeline."""
    from src.api.main import score_video, _ensure_models
    _ensure_models()

    scored = []
    total = len(videos)
    for i, v in enumerate(videos):
        if (i + 1) % 20 == 0:
            print(f"    Scoring: {i+1}/{total}...")

        try:
            sv = score_video(v.copy(), user_goal=topic_query)
            scored.append(sv)
        except Exception as e:
            print(f"    [WARN] Score failed for {v.get('video_id', '?')}: {e}")
            v["agent_scores"] = {
                "shield_score": 0.5, "clickbait": 0.5,
                "emotional_manipulation": 0.1, "credibility": 0.5,
                "info_density": 0.5, "goal_alignment": 0.5,
            }
            scored.append(v)

    return scored


def generate_explanations(videos: list[dict], topic_query: str, batch_size: int = 10) -> list[dict]:
    """Generate 1-sentence Gemini explanations for videos."""
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("    [SKIP] No GEMINI_API_KEY, skipping explanations")
            return videos

        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"    [SKIP] Gemini init failed: {e}")
        return videos

    total = len(videos)
    for i in range(0, total, batch_size):
        batch = videos[i:i + batch_size]
        if (i + batch_size) % 50 == 0:
            print(f"    Explanations: {min(i+batch_size, total)}/{total}...")

        # Build a batch prompt for efficiency
        lines = []
        for j, v in enumerate(batch):
            s = v.get("agent_scores", {})
            lines.append(
                f"{j+1}. \"{v.get('title', '?')}\" — "
                f"Shield:{int(s.get('shield_score',0)*100)} "
                f"CB:{int(s.get('clickbait',0)*100)} "
                f"Cred:{int(s.get('credibility',0)*100)}"
            )

        prompt = f"""For each YouTube video below (user searched "{topic_query}"), write exactly ONE sentence explaining its Shield score. Be specific about clickbait or quality signals. Format: numbered list matching input.

{chr(10).join(lines)}"""

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text.strip()

            # Parse numbered responses
            explanation_lines = [l.strip() for l in text.split("\n") if l.strip()]
            for j, v in enumerate(batch):
                if j < len(explanation_lines):
                    # Remove numbering prefix
                    exp = explanation_lines[j]
                    exp = exp.lstrip("0123456789.)-: ")
                    v["explanation"] = exp
                else:
                    v["explanation"] = ""

            # Rate limit: ~15 RPM for free tier
            time.sleep(4)

        except Exception as e:
            print(f"    [WARN] Gemini batch failed: {e}")
            for v in batch:
                v["explanation"] = ""
            time.sleep(2)

    return videos


def load_existing_scrapes() -> dict[str, list[dict]]:
    """Load any existing scraped data from data/scrape/."""
    scrapes_dir = PROJECT_ROOT / "data" / "scrape"
    existing = {}
    if not scrapes_dir.exists():
        return existing

    for f in scrapes_dir.glob("*.json"):
        try:
            with open(f, encoding="utf-8") as fp:
                videos = json.load(fp)
            key = f.stem.replace("_final", "").replace("_raw", "")
            existing[key] = videos
            print(f"  [CACHE] Loaded {len(videos)} existing videos for '{key}'")
        except Exception:
            pass

    return existing


# Map old topic keys to new ones
OLD_TO_NEW_KEY = {
    "machine_learning_tutorial": "machine_learning_ai",
    "learn_about_climate_change": "climate_change",
    "how_to_invest_for_beginners": "investing_finance",
    "history_of_ancient_Rome": "history_ancient",
    "healthy_meal_prep": "healthy_cooking",
}


def main():
    parser = argparse.ArgumentParser(description="Build Shield video catalog")
    parser.add_argument("--topics", type=int, default=len(TOPICS), help="Number of topics to process")
    parser.add_argument("--max-per-topic", type=int, default=200, help="Max videos per topic")
    parser.add_argument("--skip-existing", action="store_true", help="Skip topics with existing scrapes")
    parser.add_argument("--skip-transcripts", action="store_true", help="Skip transcript fetching")
    parser.add_argument("--skip-explanations", action="store_true", help="Skip Gemini explanations")
    parser.add_argument("--transcript-tier", type=int, default=1, help="Max transcript tier (1=free, 3=paid)")
    parser.add_argument("--output", type=str, default="data/scored_catalog.json", help="Output path")
    args = parser.parse_args()

    print("=" * 60)
    print("Shield Catalog Builder")
    print(f"Topics: {args.topics} | Max/topic: {args.max_per_topic}")
    print("=" * 60)

    # Load existing scraped data
    existing = load_existing_scrapes()

    all_videos = []
    if existing:
        topics_to_process = []
        for key in existing.keys():
            topics_to_process.append({
                "key": key,
                "query": key.replace("_", " "),
                "label": key.replace("_", " ").title()
            })
    else:
        topics_to_process = TOPICS[:args.topics]

    for idx, topic in enumerate(topics_to_process):
        print(f"\n[{idx+1}/{len(topics_to_process)}] {topic['label']} ({topic['key']})")
        t0 = time.time()

        # Check if we have existing data for this topic
        videos = None

        # Check old key mapping
        for old_key, new_key in OLD_TO_NEW_KEY.items():
            if new_key == topic["key"] and old_key in existing:
                videos = existing[old_key][:args.max_per_topic]
                print(f"  Using {len(videos)} existing videos (mapped from {old_key})")
                break

        # Check new key
        if videos is None and topic["key"] in existing:
            if args.skip_existing:
                videos = existing[topic["key"]][:args.max_per_topic]
                print(f"  Using {len(videos)} existing videos")
            else:
                videos = None  # Re-scrape

        # Scrape if needed
        if videos is None:
            print(f"  Scraping YouTube for '{topic['query']}'...")
            try:
                videos = scrape_topic_videos(topic["query"], args.max_per_topic)
                print(f"  Got {len(videos)} videos")
            except Exception as e:
                print(f"  [ERROR] Scrape failed: {e}")
                videos = []

        if not videos:
            print(f"  [SKIP] No videos for {topic['label']}")
            continue

        # Tag with topic
        for v in videos:
            v["topic"] = topic["label"]
            v["topic_key"] = topic["key"]
            if not v.get("thumbnail"):
                vid = v.get("video_id", "")
                v["thumbnail"] = f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"

        # Fetch transcripts
        if not args.skip_transcripts:
            print(f"  Fetching transcripts (tier {args.transcript_tier})...")
            videos = fetch_transcripts_for_videos(videos, max_tier=args.transcript_tier)

        # Score with ML models
        print(f"  Scoring {len(videos)} videos...")
        videos = score_videos(videos, topic["query"])

        elapsed = time.time() - t0
        print(f"  [OK] {topic['label']}: {len(videos)} videos scored in {elapsed:.1f}s")

        all_videos.extend(videos)

    # Generate Gemini explanations (batch after all scoring is done)
    if not args.skip_explanations and all_videos:
        print(f"\nGenerating Gemini explanations for {len(all_videos)} videos...")
        all_videos = generate_explanations(all_videos, "general content quality")

    # Sort by Shield score
    all_videos.sort(
        key=lambda v: v.get("agent_scores", {}).get("shield_score", 0),
        reverse=True,
    )

    # Save catalog
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "3.0",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_videos": len(all_videos),
            "topics": [t["label"] for t in topics_to_process],
            "topic_keys": [t["key"] for t in topics_to_process],
            "videos": all_videos,
        }, f, indent=2, ensure_ascii=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"  [OK] Catalog saved: {output_path}")
    print(f"  Videos: {len(all_videos)} | Size: {size_mb:.1f}MB")
    print(f"  Topics: {len(topics_to_process)}")

    # Print per-topic stats
    topic_stats = {}
    for v in all_videos:
        t = v.get("topic", "Unknown")
        topic_stats[t] = topic_stats.get(t, 0) + 1
    for t, count in sorted(topic_stats.items()):
        print(f"    {t}: {count} videos")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
