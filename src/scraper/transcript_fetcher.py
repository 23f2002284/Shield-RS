"""
src/scraper/transcript_fetcher.py
==================================
Tiered transcript fetching for YouTube videos.

Tier 1: youtube-transcript-api (free, ~70% coverage)
Tier 2: YouTube Data API captions (free, manual captions)
Tier 3: Gemini audio transcription (paid fallback)
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

# Cache directory for transcripts
TRANSCRIPT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "transcripts"
TRANSCRIPT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(video_id: str) -> Path:
    return TRANSCRIPT_CACHE_DIR / f"{video_id}.txt"


def _load_cached(video_id: str) -> Optional[str]:
    """Load transcript from cache if it exists."""
    path = _cache_path(video_id)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _save_cache(video_id: str, transcript: str):
    """Save transcript to cache."""
    path = _cache_path(video_id)
    path.write_text(transcript, encoding="utf-8")


def _tier1_youtube_transcript_api(video_id: str, language: str = "en") -> Optional[str]:
    """
    Tier 1: Free youtube-transcript-api package.
    Works for ~70% of videos that have auto-generated or manual captions.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to get the requested language first
        try:
            transcript = transcript_list.find_transcript([language])
        except Exception:
            # Fall back to any available transcript, translate to English
            try:
                transcript = transcript_list.find_generated_transcript([language])
            except Exception:
                # Get any transcript and translate
                for t in transcript_list:
                    try:
                        if t.language_code != language:
                            transcript = t.translate(language)
                        else:
                            transcript = t
                        break
                    except Exception:
                        continue
                else:
                    return None

        entries = transcript.fetch()
        # Combine all text segments
        full_text = " ".join(
            entry.get("text", "") if isinstance(entry, dict) else str(entry)
            for entry in entries
        )
        # Clean up
        full_text = full_text.replace("\n", " ").replace("  ", " ").strip()
        return full_text if len(full_text) > 20 else None

    except Exception as e:
        # Common: TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
        return None


def _tier2_youtube_data_api(video_id: str) -> Optional[str]:
    """
    Tier 2: Use YouTube Data API to download manually-uploaded captions.
    Requires YOUTUBE_API_KEY env var. Costs 200 quota units per caption download.
    Only use if Tier 1 fails.
    """
    try:
        from googleapiclient.discovery import build
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return None

        youtube = build("youtube", "v3", developerKey=api_key)

        # List captions for the video
        caption_response = youtube.captions().list(
            part="snippet",
            videoId=video_id,
        ).execute()

        captions = caption_response.get("items", [])
        if not captions:
            return None

        # Find English caption track
        target_caption = None
        for cap in captions:
            lang = cap["snippet"].get("language", "")
            if lang.startswith("en"):
                target_caption = cap
                break

        if not target_caption:
            target_caption = captions[0]  # Take first available

        # Note: Actually downloading captions requires OAuth, not just API key
        # So this tier has limited usefulness without OAuth setup
        return None

    except Exception:
        return None


def _tier3_gemini_transcription(video_id: str) -> Optional[str]:
    """
    Tier 3: Use Gemini to transcribe from YouTube URL.
    Gemini 2.5 Flash supports video/audio input via URL.
    Costs ~$0.003 per minute of video.
    """
    try:
        from google import genai
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return None

        client = genai.Client(api_key=api_key)

        # Gemini can process YouTube URLs directly
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": "Transcribe the spoken content of this YouTube video. Output ONLY the transcript text, no timestamps, no formatting, no commentary."},
                        {"file_data": {"file_uri": youtube_url, "mime_type": "video/*"}},
                    ],
                }
            ],
        )

        text = response.text.strip()
        if len(text) > 50:
            return text
        return None

    except Exception as e:
        print(f"  [Tier3] Gemini transcription failed for {video_id}: {e}")
        return None


def fetch_transcript(
    video_id: str,
    language: str = "en",
    use_cache: bool = True,
    max_tier: int = 3,
) -> Optional[str]:
    """
    Fetch transcript for a YouTube video using tiered approach.

    Args:
        video_id: YouTube video ID
        language: Language code (default "en")
        use_cache: Whether to check/save cache
        max_tier: Maximum tier to try (1=free only, 2=+API, 3=+Gemini paid)

    Returns:
        Transcript text or None if unavailable
    """
    # Check cache first
    if use_cache:
        cached = _load_cached(video_id)
        if cached:
            return cached

    transcript = None

    # Tier 1: youtube-transcript-api (free)
    if max_tier >= 1:
        transcript = _tier1_youtube_transcript_api(video_id, language)
        if transcript:
            if use_cache:
                _save_cache(video_id, transcript)
            return transcript

    # Tier 2: YouTube Data API (free but limited)
    if max_tier >= 2:
        transcript = _tier2_youtube_data_api(video_id)
        if transcript:
            if use_cache:
                _save_cache(video_id, transcript)
            return transcript

    # Tier 3: Gemini transcription (paid)
    if max_tier >= 3:
        transcript = _tier3_gemini_transcription(video_id)
        if transcript:
            if use_cache:
                _save_cache(video_id, transcript)
            return transcript

    return None


def fetch_transcripts_batch(
    video_ids: list[str],
    language: str = "en",
    max_tier: int = 1,
    progress: bool = True,
) -> dict[str, Optional[str]]:
    """
    Fetch transcripts for multiple videos.
    Returns dict mapping video_id -> transcript (or None).
    """
    results = {}
    total = len(video_ids)

    for i, vid in enumerate(video_ids):
        if progress and (i + 1) % 10 == 0:
            print(f"  Transcripts: {i+1}/{total} fetched")

        results[vid] = fetch_transcript(vid, language, max_tier=max_tier)

    success = sum(1 for v in results.values() if v is not None)
    if progress:
        print(f"  Transcripts: {success}/{total} succeeded ({success/max(total,1)*100:.0f}%)")

    return results
