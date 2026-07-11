"""
Robust Transcript Fetcher with Proxy Support

Strategies to maximize transcript yield:
1. Try all English language codes + Hindi
2. Fall back to auto-generated captions
3. Fall back to ANY available language transcript
4. Retry with exponential backoff
5. Rate limiting with delays between requests
6. PROXY SUPPORT — rotating residential proxies to bypass IP blocks

Usage:
  # Without proxy (direct connection)
  python transcript_fetcher.py --input data.json --output data_out.json

  # With proxy
  python transcript_fetcher.py --input data.json --output data_out.json \
      --proxy-url http://user:pass@proxy-host:port

  # Or set proxy in .env file:
  PROXY_URL=http://user:pass@proxy-host:port
"""

import os
import json
import time
import random
import argparse
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

load_dotenv()

LANGUAGE_CODES = ["en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN", "hi"]


def create_api(proxy_url: str = None) -> YouTubeTranscriptApi:
    """
    Create a YouTubeTranscriptApi instance, optionally with proxy.

    Args:
        proxy_url: Proxy URL in format http://user:pass@host:port
                   If None, checks PROXY_URL in .env
                   If still None, connects directly
    """
    if proxy_url is None:
        proxy_url = os.getenv("PROXY_URL")

    if proxy_url:
        print(f"  Using proxy: {proxy_url.split('@')[-1] if '@' in proxy_url else proxy_url}")
        proxy_config = WebshareProxyConfig(
            http_urls
        )
        return YouTubeTranscriptApi(proxy_config=proxy_config)
    else:
        print("  No proxy configured — using direct connection")
        return YouTubeTranscriptApi()


def fetch_transcript_robust(api: YouTubeTranscriptApi, video_id: str, max_retries: int = 3) -> str:
    """
    Fetch transcript with multiple fallback strategies.

    Strategy order:
    1. Try manual English transcripts
    2. Try auto-generated English transcripts
    3. Try Hindi transcripts
    4. Try any available transcript
    5. Retry on transient failures with exponential backoff
    """
    for attempt in range(max_retries):
        try:
            # Strategy 1: Try with our preferred language codes
            transcript = api.fetch(video_id, languages=LANGUAGE_CODES)
            return " ".join(s.text for s in transcript.snippets)

        except NoTranscriptFound:
            # Strategy 2-4: List all available transcripts and try them
            try:
                transcript_list = api.list(video_id)

                # Try any English manual transcript
                for t in transcript_list:
                    if t.language_code.startswith("en") and not t.is_generated:
                        try:
                            fetched = t.fetch()
                            return " ".join(s.text for s in fetched.snippets)
                        except Exception:
                            continue

                # Try any English auto-generated transcript
                for t in transcript_list:
                    if t.language_code.startswith("en") and t.is_generated:
                        try:
                            fetched = t.fetch()
                            return " ".join(s.text for s in fetched.snippets)
                        except Exception:
                            continue

                # Try Hindi transcripts
                for t in transcript_list:
                    if t.language_code.startswith("hi"):
                        try:
                            fetched = t.fetch()
                            return " ".join(s.text for s in fetched.snippets)
                        except Exception:
                            continue

                # Try ANY available transcript as last resort
                for t in transcript_list:
                    try:
                        fetched = t.fetch()
                        return " ".join(s.text for s in fetched.snippets)
                    except Exception:
                        continue

                return None

            except (TranscriptsDisabled, VideoUnavailable):
                return None
            except Exception:
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(wait)
                    continue
                return None

        except (TranscriptsDisabled, VideoUnavailable):
            return None

        except Exception:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(wait)
                continue
            return None

    return None


def process_file(
    input_path: str,
    output_path: str,
    proxy_url: str = None,
    batch_size: int = 20,
    delay: float = 0.5,
    retry_failed: bool = False,
):
    """
    Process a JSON file, fetching transcripts with rate limiting.

    Args:
        retry_failed: If True, re-attempt videos that previously returned null.
                      Useful when switching from no-proxy to proxy.
    """
    api = create_api(proxy_url)

    print(f"Loading metadata from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    total = len(videos)
    already_have = sum(1 for v in videos if v.get("transcript"))
    need_fetch = total - already_have
    print(f"  Total videos: {total}")
    print(f"  Already have transcript: {already_have}")
    print(f"  Need to fetch: {need_fetch}")

    success_count = already_have
    new_fetched = 0
    failed_ids = []

    for i, video in enumerate(videos):
        # Skip videos that already have transcripts
        if video.get("transcript") and not retry_failed:
            continue

        # In retry mode, skip videos that already have transcripts
        if video.get("transcript"):
            continue

        vid = video["video_id"]
        transcript = fetch_transcript_robust(api, vid)
        video["transcript"] = transcript

        if transcript:
            new_fetched += 1
            success_count += 1
            word_count = len(transcript.split())
            print(f"  [{i+1}/{total}] OK: {vid} ({word_count} words)")
        else:
            failed_ids.append(vid)
            if len(failed_ids) % 50 == 0:
                print(f"  [{i+1}/{total}] Progress: {success_count} transcripts, {len(failed_ids)} failed so far")

        # Rate limiting
        time.sleep(delay + random.uniform(0, 0.3))

        # Batch cooldown
        if new_fetched > 0 and new_fetched % batch_size == 0:
            print(f"  --- Batch checkpoint: {success_count}/{total} transcripts ({new_fetched} new) ---")
            # Save intermediate progress
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(videos, f, indent=2, ensure_ascii=False)
            print(f"  --- Intermediate save complete ---")
            time.sleep(2)

    print(f"\nResults:")
    print(f"  Already had: {already_have}")
    print(f"  Newly fetched: {new_fetched}")
    print(f"  Total with transcript: {success_count}/{total} ({success_count/total*100:.1f}%)")
    print(f"  Failed: {len(failed_ids)}")

    # Final save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")
    return success_count, total


def main():
    parser = argparse.ArgumentParser(description="Robust transcript fetcher with proxy support")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--proxy-url", type=str, default=None,
                        help="Proxy URL: http://user:pass@host:port (or set PROXY_URL in .env)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between requests in seconds (default: 0.5)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Save progress every N new transcripts")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry videos that previously returned null")
    args = parser.parse_args()
    process_file(
        args.input, args.output,
        proxy_url=args.proxy_url,
        batch_size=args.batch_size,
        delay=args.delay,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
