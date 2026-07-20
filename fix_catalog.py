import json
import os
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper.youtube_scraper import scrape_topic
from src.scraper.transcript_fetcher import fetch_transcript
from src.api.main import score_video, _ensure_models

FIX_TOPICS = [
    {"key": "space_astronomy", "query": "space documentary", "label": "Space / Astronomy"},
    {"key": "philosophy_ethics", "query": "philosophy explained", "label": "Philosophy / Ethics"}
]

def main():
    catalog_path = PROJECT_ROOT / "data" / "scored_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    _ensure_models()

    for topic in FIX_TOPICS:
        print(f"Scraping '{topic['query']}'...")
        videos = scrape_topic(topic["query"], max_results=200)
        
        # Tag topics
        for v in videos:
            v["topic"] = topic["label"]
            v["topic_key"] = topic["key"]
            if not v.get("thumbnail"):
                vid = v.get("video_id", "")
                v["thumbnail"] = f"https://img.youtube.com/vi/{vid}/mqdefault.jpg"
                
        print(f"Fetching transcripts for {len(videos)} videos...")
        for i, v in enumerate(videos):
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(videos)}...")
            transcript = fetch_transcript(v.get("video_id", ""), max_tier=1)
            if transcript:
                v["transcript"] = transcript
                
        print(f"Scoring {len(videos)} videos...")
        scored = []
        for v in videos:
            try:
                sv = score_video(v.copy(), user_goal=topic["query"])
                sv["explanation"] = ""
                scored.append(sv)
            except Exception as e:
                v["agent_scores"] = {
                    "shield_score": 0.5, "clickbait": 0.5,
                    "emotional_manipulation": 0.1, "credibility": 0.5,
                    "info_density": 0.5, "goal_alignment": 0.5,
                }
                v["explanation"] = ""
                scored.append(v)
                
        # Remove existing videos with this topic label from catalog
        catalog["videos"] = [v for v in catalog.get("videos", []) if v.get("topic") != topic["label"]]
        
        # Add the new ones
        catalog["videos"].extend(scored)
        
    catalog["videos"].sort(key=lambda v: v.get("agent_scores", {}).get("shield_score", 0), reverse=True)
    catalog["total_videos"] = len(catalog["videos"])
    
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        
    print("Catalog updated successfully!")

if __name__ == "__main__":
    main()
