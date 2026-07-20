import json
import os
import requests
import time

FIX_TOPICS = [
    {"key": "space_astronomy", "query": "space documentary", "label": "Space / Astronomy"},
    {"key": "philosophy_ethics", "query": "philosophy explained", "label": "Philosophy / Ethics"}
]

def main():
    catalog_path = "data/scored_catalog.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    new_videos = []
    
    for topic in FIX_TOPICS:
        print(f"Calling API to search & score '{topic['query']}'...")
        try:
            resp = requests.post(
                "http://localhost:8000/api/search",
                json={"query": topic["query"], "max_results": 200},
                timeout=600 # 10 minutes timeout since scoring takes a while
            )
            resp.raise_for_status()
            data = resp.json()
            videos = data.get("videos", [])
            print(f"Got {len(videos)} scored videos from API.")
            
            # Tag topics
            for v in videos:
                v["topic"] = topic["label"]
                v["topic_key"] = topic["key"]
                
            new_videos.extend(videos)
        except Exception as e:
            print(f"Failed to fetch {topic['label']}: {e}")
            
    # Remove existing videos with this topic label from catalog (if any)
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
