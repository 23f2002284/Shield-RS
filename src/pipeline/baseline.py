"""
Baseline: YouTube-native ranking.

For a given query, this module fetches the top results from YouTube's API
(already ranked by YouTube's engagement-driven algorithm) and records them.
This serves as the "Human-Centric RS" baseline for our A/B comparison.

The agent-centric RS must demonstrably outperform this on our metrics:
goal alignment, information density, viewpoint diversity, etc.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)


def fetch_baseline_ranking(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Fetch YouTube's top results for a query — this IS the baseline.
    YouTube ranks these by its own engagement-optimized algorithm.
    We record the rank position so we can compare against our agent later.
    """
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_results:
        request = youtube.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=min(50, max_results - len(video_ids)),
            pageToken=next_page_token,
            order="relevance"  # YouTube's default ranking
        )
        response = request.execute()
        for item in response.get("items", []):
            video_ids.append(item["id"]["videoId"])
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    # Fetch full metadata
    results = []
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk)
        )
        response = request.execute()
        for rank, item in enumerate(response.get("items", []), start=len(results) + 1):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})
            duration_seconds = int(isodate.parse_duration(content.get("duration", "PT0S")).total_seconds())

            results.append({
                "youtube_rank": rank,
                "video_id": item["id"],
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_name": snippet.get("channelTitle", ""),
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
                "duration_seconds": duration_seconds,
                "published_at": snippet.get("publishedAt", ""),
                "tags": snippet.get("tags", []),
                "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "captions_available": content.get("caption", "false") == "true"
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube baseline rankings.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--max_results", type=int, default=50)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    print(f"Fetching YouTube baseline for: '{args.query}'")
    results = fetch_baseline_ranking(args.query, args.max_results)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} baseline results to {args.output}")


if __name__ == "__main__":
    main()
