import json
import os
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.scraper.youtube_scraper import scrape_topic
from src.scraper.transcript_fetcher import fetch_transcript

FIX_TOPICS = [
    {"key": "space_astronomy", "query": "space documentary", "label": "Space / Astronomy"},
    {"key": "philosophy_ethics", "query": "philosophy explained", "label": "Philosophy / Ethics"}
]

def main():
    catalog_path = PROJECT_ROOT / "data" / "scored_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    new_videos = []

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
            try:
                transcript = fetch_transcript(v.get("video_id", ""), max_tier=1)
                if transcript:
                    v["transcript"] = transcript
            except Exception as e:
                pass
                
        print(f"Applying default scores for {len(videos)} videos (Bypassing ML due to OOM)...")
        for v in videos:
            v["agent_scores"] = {
                "shield_score": 0.5, "clickbait": 0.3,
                "emotional_manipulation": 0.2, "credibility": 0.6,
                "info_density": 0.6, "goal_alignment": 0.5,
            }
            v["explanation"] = "Default score applied (ML bypassed due to system memory limits)."
            new_videos.append(v)
            
    # Remove existing videos with this topic label from catalog
    for topic in FIX_TOPICS:
        catalog["videos"] = [v for v in catalog.get("videos", []) if v.get("topic") != topic["label"]]
        
    # Add the new ones
    catalog["videos"].extend(new_videos)
    catalog["videos"].sort(key=lambda v: v.get("agent_scores", {}).get("shield_score", 0), reverse=True)
    catalog["total_videos"] = len(catalog["videos"])
    
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
        
    print(f"Catalog updated successfully! Total videos now: {catalog['total_videos']}")

if __name__ == "__main__":
    main()
