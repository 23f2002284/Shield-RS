import os
import json
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY is not set in the .env file.")

youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

def search_videos(query: str, max_results: int = 200) -> List[str]:
    """Search for videos and return a list of video IDs."""
    video_ids = []
    next_page_token = None
    
    print(f"Searching for query: '{query}'...")
    while len(video_ids) < max_results:
        try:
            request = youtube.search().list(
                part="id",
                q=query,
                type="video",
                maxResults=min(50, max_results - len(video_ids)),
                pageToken=next_page_token
            )
            response = request.execute()
            
            for item in response.get("items", []):
                video_ids.append(item["id"]["videoId"])
                
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
                
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred: {e.content}")
            break
        except Exception as e:
            print(f"A network or system error occurred: {e}")
            break
            
    print(f"Found {len(video_ids)} video IDs for query '{query}'.")
    return video_ids

def get_video_details(video_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch metadata for a list of video IDs."""
    videos_metadata = []
    
    # The API supports up to 50 IDs per request
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        try:
            request = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(chunk)
            )
            response = request.execute()
            
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                content_details = item.get("contentDetails", {})
                
                duration_iso = content_details.get("duration", "PT0S")
                duration_seconds = int(isodate.parse_duration(duration_iso).total_seconds())
                
                video_data = {
                    "video_id": item["id"],
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_name": snippet.get("channelTitle", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "view_count": int(statistics.get("viewCount", 0)),
                    "like_count": int(statistics.get("likeCount", 0)) if "likeCount" in statistics else 0,
                    # We will omit subscriber_count here as it requires a separate call to channels().list()
                    "duration_seconds": duration_seconds,
                    "published_at": snippet.get("publishedAt", ""),
                    "tags": snippet.get("tags", []),
                    "category_id": int(snippet.get("categoryId", 0)),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "captions_available": content_details.get("caption", "false") == "true"
                }
                videos_metadata.append(video_data)
                
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred while fetching details: {e.content}")
            
    return videos_metadata

def fetch_channel_subscribers(channel_ids: List[str]) -> Dict[str, int]:
    """Fetch subscriber counts for a list of channel IDs."""
    subscriber_counts = {}
    unique_channels = list(set(channel_ids))
    
    for i in range(0, len(unique_channels), 50):
        chunk = unique_channels[i:i+50]
        try:
            request = youtube.channels().list(
                part="statistics",
                id=",".join(chunk)
            )
            response = request.execute()
            for item in response.get("items", []):
                subscriber_counts[item["id"]] = int(item.get("statistics", {}).get("subscriberCount", 0))
        except HttpError as e:
            print(f"An HTTP error occurred while fetching channels: {e.content}")
            
    return subscriber_counts

def scrape_topic(query: str, max_results: int = 200) -> List[Dict[str, Any]]:
    video_ids = search_videos(query, max_results)
    if not video_ids:
        return []
        
    videos = get_video_details(video_ids)
    
    # Get subscriber counts
    channel_ids = [v["channel_id"] for v in videos]
    subs = fetch_channel_subscribers(channel_ids)
    
    for v in videos:
        v["subscriber_count"] = subs.get(v["channel_id"], 0)
        
    return videos

def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube video metadata.")
    parser.add_argument("--query", type=str, required=True, help="Topic query to search for.")
    parser.add_argument("--max_results", type=int, default=200, help="Maximum number of videos to fetch.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file.")
    
    args = parser.parse_args()
    
    print(f"Starting scrape for: {args.query}")
    videos = scrape_topic(args.query, args.max_results)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(videos)} videos to {args.output}")

if __name__ == "__main__":
    main()
