"""
Robust Transcript Fetcher v2 - Anti-Block Edition

Built to survive YouTube IP blocks, rate limiting, and proxy failures.
Designed to scale to 10,000+ videos.

Key features:
  - Free proxy pool with rotation (proxies.txt, one IP:PORT per line)
  - Auto-refresh proxy list via Selenium scraping (optional dependency)
  - Per-operation timeouts (no more hanging on dead proxies)
  - Per-video proxy tracking (not global blacklisting)
  - Multi-threaded collaborative work-stealing
  - "No transcript" vs "network error" distinction  
  - Progress persistence with full resume support
  - Backward-compatible JSON input/output with transcript_fetcher.py

Usage:
  # Basic - direct connection only
  python -m src.scraper.transcript_fetcher_v2 --input data.json --output data_out.json

  # With proxy file
  python -m src.scraper.transcript_fetcher_v2 --input data.json --output data_out.json \\
      --proxy-file proxies.txt

  # With .env fallback proxy + multi-threading
  python -m src.scraper.transcript_fetcher_v2 --input data.json --output data_out.json \\
      --proxy-url http://user:pass@proxy-host:port --threads 8

  # Resume an interrupted run
  python -m src.scraper.transcript_fetcher_v2 --input data.json --output data_out.json \\
      --proxy-file proxies.txt --resume
"""

import json
import os
import random
import re
import sys
import signal
import threading
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

load_dotenv()

# Optional Selenium import for proxy auto-refresh
_HAS_SELENIUM = False
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service as ChromeService
    from webdriver_manager.chrome import ChromeDriverManager

    _HAS_SELENIUM = True
except ImportError:
    pass

# Optional yt-dlp import for transcript fallback
_HAS_YTDLP = False
try:
    import yt_dlp
    _HAS_YTDLP = True
except ImportError:
    pass


# =============================================================================
# Configuration
# =============================================================================

LANGUAGE_CODES = ["en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN", "hi"]

# Threading
DEFAULT_THREADS = 4
MAX_THREADS = 200
MIN_THREADS = 1

# Timeouts (seconds)
DEFAULT_TIMEOUT = 15  # Per-operation timeout
LIST_TIMEOUT = 15  # Timeout for listing available transcripts
FETCH_TIMEOUT = 15  # Timeout for downloading transcript content

# Proxy
MAX_PROXY_RETRIES = 20  # Max proxy rotations per video
MAX_PROXY_REFRESHES = 3  # Max times to auto-refresh proxy list
NO_TRANSCRIPT_VOTES_REQUIRED = 2  # Need 2 proxies to confirm "no transcript"

# yt-dlp fallback
YTDLP_FALLBACK = True   # Enabled by default when yt-dlp is installed
YTDLP_LANG_PREF = ["en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN"]

# Rate limiting
THREAD_DELAY_MIN = 0.5
THREAD_DELAY_MAX = 1.5

# Batch save interval
DEFAULT_BATCH_SIZE = 20

# Progress file
PROGRESS_SUFFIX = ".progress.json"


# =============================================================================
# Exceptions
# =============================================================================

class NoTranscriptError(Exception):
    """Video genuinely has no transcript -- not a proxy/network issue. Don't retry."""
    pass


# =============================================================================
# Proxy Pool (Thread-safe)
# =============================================================================

class ProxyPool:
    """
    Thread-safe proxy pool with rotation and per-proxy failure tracking.

    Design decisions (from transcript_helper.py):
    - We do NOT globally blacklist proxies. A proxy that fails on video A
      might work on video B. Per-video tracking is done in VideoWorkQueue.
    - Round-robin AND random selection available.
    - Supports reload (for auto-refresh) without restart.
    """

    def __init__(self, proxy_file: str = None, fallback_proxy_url: str = None):
        self.proxies: list[str] = []
        self.failed_proxies: set[str] = set()
        self.lock = threading.Lock()
        self.index = 0
        self._fallback_proxy_url = fallback_proxy_url

        if proxy_file and os.path.exists(proxy_file):
            self._load_from_file(proxy_file)
        elif fallback_proxy_url:
            # Use the single .env proxy as the only proxy in the pool
            self.proxies.append(fallback_proxy_url)
            print(f"  ProxyPool: Using single fallback proxy from .env")

        if self.proxies:
            random.shuffle(self.proxies)

    def _load_from_file(self, proxy_file: str):
        """Load proxies from a file (one IP:PORT per line)."""
        with open(proxy_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and ":" in line:
                    self.proxies.append(line)
        print(f"  ProxyPool: Loaded {len(self.proxies)} proxies from {os.path.basename(proxy_file)}")

    def get_proxy(self) -> str | None:
        """Get next working proxy (round-robin), skipping globally failed ones."""
        with self.lock:
            if not self.proxies:
                return None
            attempts = 0
            while attempts < len(self.proxies):
                proxy = self.proxies[self.index % len(self.proxies)]
                self.index += 1
                if proxy not in self.failed_proxies:
                    return proxy
                attempts += 1
            return None

    def get_random_proxy(self, exclude: set = None) -> str | None:
        """Get a random proxy not in the exclude set."""
        with self.lock:
            available = [
                p for p in self.proxies
                if p not in self.failed_proxies
                and (exclude is None or p not in exclude)
            ]
            return random.choice(available) if available else None

    def mark_failed(self, proxy: str):
        """Mark a proxy as globally failed."""
        with self.lock:
            self.failed_proxies.add(proxy)

    def reload(self, proxy_file: str):
        """Reload proxies from file, clearing all failure state."""
        with self.lock:
            self.proxies.clear()
            self.failed_proxies.clear()
            self.index = 0
        # Load outside lock (I/O)
        if os.path.exists(proxy_file):
            new_proxies = []
            with open(proxy_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and ":" in line:
                        new_proxies.append(line)
            with self.lock:
                self.proxies = new_proxies
                random.shuffle(self.proxies)
            print(f"  ProxyPool: Reloaded {len(new_proxies)} proxies")

    def get_available_count(self) -> int:
        with self.lock:
            return len(self.proxies) - len(self.failed_proxies)

    def has_proxies(self) -> bool:
        return len(self.proxies) > 0

    def get_stats(self) -> tuple[int, int]:
        """Return (total, failed) counts."""
        with self.lock:
            return len(self.proxies), len(self.failed_proxies)


# =============================================================================
# Proxy Auto-Refresh (Selenium - optional)
# =============================================================================

def download_fresh_proxies(proxy_file: str) -> int:
    """
    Download a fresh proxy list from free-proxy-list.net using Selenium.
    Returns the number of proxies downloaded, or 0 on failure.

    Requires: selenium, webdriver-manager (optional dependencies).
    If not installed, logs a warning and returns 0.
    """
    if not _HAS_SELENIUM:
        print("  [proxy-refresh] Selenium not installed -- cannot auto-refresh proxies.")
        print("  [proxy-refresh] Install with: pip install selenium webdriver-manager")
        return 0

    print("  [proxy-refresh] Downloading fresh proxy list...")
    driver = None
    try:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=options,
        )

        driver.get("https://free-proxy-list.net/en/")
        wait = WebDriverWait(driver, 10)

        button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a[title="Get raw list"]'))
        )
        button.click()

        textarea = wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "textarea.form-control"))
        )
        content = textarea.get_attribute("value")
        proxies = re.findall(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+", content)

        with open(proxy_file, "w", encoding="utf-8") as f:
            f.write("\n".join(proxies))

        print(f"  [proxy-refresh] OK: Downloaded {len(proxies)} proxies")
        return len(proxies)

    except Exception as e:
        print(f"  [proxy-refresh] FAIL: {e}")
        return 0
    finally:
        if driver:
            driver.quit()


# =============================================================================
# yt-dlp Transcript Fallback
# =============================================================================

def _parse_vtt(vtt_text: str) -> str:
    """
    Convert a WebVTT subtitle string to clean plain text.
    Removes timestamps, cue metadata, HTML tags, and duplicate lines.
    """
    lines = vtt_text.splitlines()
    seen = []
    for line in lines:
        line = line.strip()
        # Skip header, blank lines, timestamps, and cue IDs
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("NOTE"):
            continue
        if re.match(r"^\d{2}:\d{2}[:\.]\d{2}", line):  # timestamp
            continue
        if re.match(r"^\d+$", line):  # cue number
            continue
        # Strip HTML/VTT tags like <00:00:01.000>, <c>, </c>
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and (not seen or seen[-1] != line):
            seen.append(line)
    return " ".join(seen)


def _parse_json3(json3_text: str) -> str:
    """
    Convert a YouTube json3 subtitle format to clean plain text.
    """
    try:
        data = json.loads(json3_text)
        events = data.get("events", [])
        parts = []
        for event in events:
            for seg in event.get("segs", []):
                text = seg.get("utf8", "").strip()
                if text and text != "\n":
                    parts.append(text)
        return " ".join(parts)
    except Exception:
        return ""


def _fetch_via_ytdlp(video_id: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[str, str]:
    """
    Fetch transcript for a video using yt-dlp subtitle extraction.

    This is a fallback for when youtube_transcript_api fails. yt-dlp emulates
    a real browser and can access auto-generated captions that the API misses.

    Returns:
        (transcript_text, language_label) on success.
    Raises:
        NoTranscriptError if no subtitles found.
        RuntimeError on yt-dlp errors.
    """
    if not _HAS_YTDLP:
        raise RuntimeError("yt-dlp not installed")

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Phase 1: Extract subtitle info without downloading video
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": YTDLP_LANG_PREF + [".*"],  # preferred + any language
        "subtitlesformat": "vtt/json3",
        "quiet": True,
        "no_warnings": True,
        "logger": _YtdlpQuietLogger(),
        "socket_timeout": timeout,
    }

    def _extract():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info

    try:
        info = _run_with_timeout(_extract, timeout, label="ytdlp-extract")
    except TimeoutError:
        raise
    except Exception as e:
        raise RuntimeError(f"yt-dlp extraction failed: {e}")

    if not info:
        raise NoTranscriptError("yt-dlp: no video info returned")

    # Collect all available subtitle tracks: manual first, then auto-generated
    subtitles: dict = info.get("subtitles") or {}
    auto_subs: dict = info.get("automatic_captions") or {}

    # Build ordered list of (lang_code, formats_list, is_auto)
    candidates = []
    # Prefer manual subtitles in preferred languages
    for lang in YTDLP_LANG_PREF:
        if lang in subtitles:
            candidates.append((lang, subtitles[lang], False))
    # Then auto-generated in preferred languages
    for lang in YTDLP_LANG_PREF:
        if lang in auto_subs:
            candidates.append((lang, auto_subs[lang], True))
    # Then any manual language
    for lang, fmts in subtitles.items():
        if lang not in YTDLP_LANG_PREF:
            candidates.append((lang, fmts, False))
    # Then any auto language
    for lang, fmts in auto_subs.items():
        if lang not in YTDLP_LANG_PREF:
            candidates.append((lang, fmts, True))

    if not candidates:
        raise NoTranscriptError("yt-dlp: no subtitles available")

    # Phase 2: Download and parse the first working subtitle track
    import urllib.request

    for lang_code, formats, is_auto in candidates:
        # Pick best format: json3 > vtt > any
        url_to_fetch = None
        fmt_type = None
        for fmt_pref in ("json3", "vtt", None):
            for fmt in formats:
                if fmt_pref is None or fmt.get("ext") == fmt_pref:
                    url_to_fetch = fmt.get("url")
                    fmt_type = fmt.get("ext", "vtt")
                    break
            if url_to_fetch:
                break

        if not url_to_fetch:
            continue

        try:
            def _download_sub():
                req = urllib.request.Request(
                    url_to_fetch,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.read().decode("utf-8", errors="replace")

            raw = _run_with_timeout(_download_sub, timeout, label="ytdlp-sub-download")

            if fmt_type == "json3":
                text = _parse_json3(raw)
            else:
                text = _parse_vtt(raw)

            if not text or len(text.split()) < 5:
                continue

            lang_type = "auto-generated" if is_auto else "manual"
            return text, f"{lang_code} ({lang_type}, via yt-dlp)"

        except Exception:
            continue

    raise NoTranscriptError("yt-dlp: all subtitle tracks failed to download or were empty")


class _YtdlpQuietLogger:
    """Suppress all yt-dlp console output."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


# =============================================================================
# Timeout Wrappers
# =============================================================================

def _run_with_timeout(fn, timeout: int, label: str = "operation"):
    """
    Run a callable with a hard timeout. Kills hung threads.
    Returns the result or raises TimeoutError / the original exception.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn)
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"{label} timed out after {timeout}s")
    except Exception:
        executor.shutdown(wait=False)
        raise


# =============================================================================
# Transcript API Helpers
# =============================================================================

def _create_api(proxy: str = None) -> YouTubeTranscriptApi:
    """
    Create a YouTubeTranscriptApi instance with optional proxy.

    Args:
        proxy: Proxy string. Formats supported:
            - "IP:PORT" (free proxy, will be wrapped as http://IP:PORT)
            - "http://user:pass@host:port" (authenticated proxy, used as-is)
    """
    if not proxy:
        return YouTubeTranscriptApi()

    # Determine the proxy URL format
    if proxy.startswith("http://") or proxy.startswith("https://"):
        proxy_url = proxy
    else:
        proxy_url = f"http://{proxy}"

    proxy_config = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
    return YouTubeTranscriptApi(proxy_config=proxy_config)

def _check_transcript_availability(
    video_id: str, proxy: str = None, timeout: int = LIST_TIMEOUT
) -> tuple:
    """
    Check what transcripts are available for a video (with timeout).

    Returns (transcript_obj, language_info_str).
    Raises NoTranscriptError if the video genuinely has no transcript.
    Raises TimeoutError or other Exception on network issues.
    """

    def _inner():
        api = _create_api(proxy)
        transcript_list = api.list(video_id)

        # Strategy 1: Try English transcript
        try:
            transcript = transcript_list.find_transcript(LANGUAGE_CODES)
            lang_type = "auto-generated" if transcript.is_generated else "manual"
            lang_code = transcript.language_code
            return (transcript, f"{lang_code} ({lang_type})")
        except NoTranscriptFound:
            pass

        # Strategy 2: Try translatable transcript -> English
        for available in transcript_list:
            if available.is_translatable:
                translation_codes = [
                    lang.get("language_code", "")
                    for lang in available.translation_languages
                ]
                if "en" in translation_codes:
                    translated = available.translate("en")
                    return (translated, f"Translated from {available.language}")

        # Strategy 3: Fall back to ANY available transcript
        available_list = list(transcript_list)
        if available_list:
            t = available_list[0]
            return (t, f"{t.language} (no English available)")

        raise NoTranscriptError("No transcript available in any language")

    try:
        return _run_with_timeout(_inner, timeout, label="list-transcripts")
    except TranscriptsDisabled:
        raise NoTranscriptError("Transcripts are disabled for this video")
    except NoTranscriptFound:
        raise NoTranscriptError("No transcript found for this video")
    except VideoUnavailable:
        raise NoTranscriptError("Video is unavailable")


def _fetch_transcript_content(transcript_obj, timeout: int = FETCH_TIMEOUT) -> str:
    """
    Fetch the actual transcript text from a transcript object (with timeout).
    Returns the full text as a single string.
    """

    def _inner():
        fetched = transcript_obj.fetch()
        return " ".join(snippet.text for snippet in fetched)

    return _run_with_timeout(_inner, timeout, label="fetch-transcript")


# =============================================================================
# Single Video Fetch (with proxy rotation)
# =============================================================================

def fetch_single_video(
    video_id: str,
    proxy: str = None,
    timeout: int = DEFAULT_TIMEOUT,
    use_ytdlp: bool = True,
) -> dict:
    """
    Fetch transcript for a single video using ONE specific proxy.

    Returns a result dict:
      {
        "video_id": str,
        "transcript": str | None,
        "language": str | None,
        "method": "direct" | "proxy:IP" | "no_transcript" | "error",
        "error_detail": str | None,
      }
    """
    result = {
        "video_id": video_id,
        "transcript": None,
        "language": None,
        "method": None,
        "error_detail": None,
    }

    proxy_label = proxy.split(":")[0] if proxy and not proxy.startswith("http") else (
        proxy.split("@")[-1].split(":")[0] if proxy else "direct"
    )
    strategy = f"proxy:{proxy_label}" if proxy else "direct"

    try:
        # Step 1: Check availability
        transcript_obj, lang_info = _check_transcript_availability(
            video_id, proxy=proxy, timeout=timeout
        )

        # Step 2: Fetch content (using SAME proxy -- critical for consistency)
        text = _fetch_transcript_content(transcript_obj, timeout=timeout)

        result["transcript"] = text
        result["language"] = lang_info
        result["method"] = strategy
        return result

    except NoTranscriptError as e:
        # Fallback: try yt-dlp before giving up
        if use_ytdlp and _HAS_YTDLP and YTDLP_FALLBACK:
            try:
                text, lang_info = _fetch_via_ytdlp(video_id, timeout=timeout)
                result["transcript"] = text
                result["language"] = lang_info
                result["method"] = "ytdlp"
                return result
            except NoTranscriptError as ytdlp_e:
                result["method"] = "no_transcript"
                result["error_detail"] = f"API: {e} | yt-dlp: {ytdlp_e}"
                return result
            except Exception as ytdlp_e:
                # yt-dlp had a network/parse error -- still report no_transcript
                result["method"] = "no_transcript"
                result["error_detail"] = f"API: {e} | yt-dlp error: {ytdlp_e}"
                return result
        result["method"] = "no_transcript"
        result["error_detail"] = str(e)
        return result

    except TimeoutError as e:
        result["method"] = "error"
        result["error_detail"] = f"{strategy}: {e}"
        return result

    except Exception as e:
        result["method"] = "error"
        result["error_detail"] = f"{strategy}: {e}"
        return result


# =============================================================================
# Video Work Queue (Collaborative Multi-Proxy Download)
# =============================================================================

class VideoWorkQueue:
    """
    Thread-safe work queue where multiple threads collaborate on the same video
    with different proxies. Implements work-stealing pattern.

    Design (from transcript_helper.py):
    - Per-video proxy tracking (not global blacklisting)
    - "No transcript" needs 2 votes from different proxies to confirm
    - Auto-refresh proxy list when all exhausted (up to MAX_PROXY_REFRESHES)
    """

    def __init__(self, video_ids: list[str], proxy_pool: ProxyPool, proxy_file: str = None):
        self.video_ids = list(video_ids)
        self.proxy_pool = proxy_pool
        self.proxy_file = proxy_file
        self.lock = threading.Lock()
        self.proxy_refresh_count = 0
        self.refresh_in_progress = False

        # Video states: pending, in_progress, completed, no_transcript, failed
        self.video_states = {vid: "pending" for vid in video_ids}

        # Per-video: set of proxies already tried
        self.video_proxy_attempts = {vid: set() for vid in video_ids}

        # No-transcript votes (need 2 to confirm)
        self.no_transcript_votes = {vid: 0 for vid in video_ids}

        # Active worker count per video
        self.active_workers = {vid: 0 for vid in video_ids}

        # Results storage
        self.results: dict[str, dict] = {}

        # Counters
        self.completed_count = 0
        self.success_count = 0
        self.no_transcript_count = 0
        self.failed_count = 0
        self.total_proxy_attempts = 0

    def get_work(self) -> tuple[str, str] | None:
        """
        Get a (video_id, proxy) pair to work on.

        Strategy:
        1. Find a 'pending' video with an untried proxy
        2. Join an 'in_progress' video with a new proxy
        3. Refresh proxy list if exhausted
        4. Return None if all done or no proxies left
        """
        should_retry = False

        with self.lock:
            all_proxies = set(self.proxy_pool.proxies)

            # Strategy 1: Pending videos
            for vid in self.video_ids:
                if self.video_states[vid] == "pending":
                    available = all_proxies - self.video_proxy_attempts[vid]
                    if available:
                        proxy = random.choice(list(available))
                        self.video_states[vid] = "in_progress"
                        self.video_proxy_attempts[vid].add(proxy)
                        self.active_workers[vid] += 1
                        self.total_proxy_attempts += 1
                        return (vid, proxy)

            # Strategy 2: Join in-progress videos (prefer fewer workers)
            in_progress = []
            for vid in self.video_ids:
                if self.video_states[vid] == "in_progress":
                    avail_count = len(all_proxies - self.video_proxy_attempts[vid])
                    if avail_count > 0:
                        in_progress.append((vid, self.active_workers[vid], avail_count))

            in_progress.sort(key=lambda x: (x[1], -x[2]))

            for vid, _, _ in in_progress:
                available = all_proxies - self.video_proxy_attempts[vid]
                if available:
                    proxy = random.choice(list(available))
                    self.video_proxy_attempts[vid].add(proxy)
                    self.active_workers[vid] += 1
                    self.total_proxy_attempts += 1
                    return (vid, proxy)

            # Strategy 3: Check if we need to refresh
            remaining = [
                vid for vid, state in self.video_states.items()
                if state in ("pending", "in_progress")
            ]

            if not remaining:
                return None

            # Check for untried proxies on any remaining video
            any_available = any(
                len(all_proxies - self.video_proxy_attempts[vid]) > 0
                for vid in remaining
            )

            if any_available:
                should_retry = True
            elif self.refresh_in_progress:
                # Another thread is refreshing, wait briefly
                self.lock.release()
                time.sleep(0.5)
                self.lock.acquire()
                should_retry = True
            else:
                if self._refresh_proxies_internal():
                    should_retry = True
                else:
                    # Max refreshes reached -- mark remaining as failed
                    for vid in remaining:
                        if self.video_states[vid] not in ("completed", "no_transcript"):
                            self.video_states[vid] = "failed"
                            self.failed_count += 1
                            self.completed_count += 1
                    return None

        if should_retry:
            if self.is_all_done():
                return None
            return self.get_work()

        return None

    def _refresh_proxies_internal(self) -> bool:
        """
        Auto-refresh proxy list. Must be called with self.lock held.
        Releases lock during I/O, re-acquires after.
        """
        if self.proxy_refresh_count >= MAX_PROXY_REFRESHES:
            return False
        if self.refresh_in_progress:
            return False

        self.refresh_in_progress = True
        self.proxy_refresh_count += 1
        refresh_num = self.proxy_refresh_count

        # Release lock for I/O
        self.lock.release()
        try:
            print(f"\n  *** Auto-refreshing proxy list (attempt {refresh_num}/{MAX_PROXY_REFRESHES}) ***")
            count = 0
            if self.proxy_file:
                count = download_fresh_proxies(self.proxy_file)
        finally:
            self.lock.acquire()
            self.refresh_in_progress = False

        if count > 0 and self.proxy_file:
            self.proxy_pool.reload(self.proxy_file)
            # Clear per-video attempts so new proxies can be tried
            for vid in self.video_proxy_attempts:
                self.video_proxy_attempts[vid].clear()
            print(f"  *** Loaded {count} fresh proxies, cleared attempt history ***")
            return True

        return False

    def mark_completed(self, video_id: str, result: dict):
        """Mark video as successfully downloaded."""
        with self.lock:
            if self.video_states[video_id] in ("completed", "no_transcript", "failed"):
                return
            self.video_states[video_id] = "completed"
            self.results[video_id] = result
            self.completed_count += 1
            self.success_count += 1

    def mark_no_transcript(self, video_id: str, result: dict):
        """Vote: this video has no transcript. Needs 2 votes to confirm."""
        with self.lock:
            if self.video_states[video_id] in ("completed", "no_transcript", "failed"):
                return
            self.no_transcript_votes[video_id] += 1
            if self.no_transcript_votes[video_id] >= NO_TRANSCRIPT_VOTES_REQUIRED:
                self.video_states[video_id] = "no_transcript"
                self.results[video_id] = result
                self.completed_count += 1
                self.no_transcript_count += 1

    def mark_proxy_failed(self, video_id: str, proxy: str):
        """
        Note: Intentionally does NOT globally blacklist the proxy.
        Per-video tracking in video_proxy_attempts is sufficient.
        """
        pass

    def release_work(self, video_id: str):
        """Decrement active worker count when a thread finishes working on a video."""
        with self.lock:
            self.active_workers[video_id] = max(0, self.active_workers[video_id] - 1)

    def is_video_done(self, video_id: str) -> bool:
        with self.lock:
            return self.video_states[video_id] in ("completed", "no_transcript", "failed")

    def is_all_done(self) -> bool:
        with self.lock:
            return self.completed_count >= len(self.video_ids)

    def get_progress(self) -> tuple[int, int, int, int, int]:
        """Return (completed, success, no_transcript, failed, total)."""
        with self.lock:
            return (
                self.completed_count,
                self.success_count,
                self.no_transcript_count,
                self.failed_count,
                len(self.video_ids),
            )


# =============================================================================
# Progress Persistence (Thread-safe)
# =============================================================================

_progress_lock = threading.Lock()


def _get_progress_path(output_path: str) -> str:
    """Derive progress file path from output path."""
    base, _ = os.path.splitext(output_path)
    return base + PROGRESS_SUFFIX


def save_progress(output_path: str, videos: list[dict], completed_ids: set[str]):
    """Save current progress to a JSON sidecar file."""
    progress_path = _get_progress_path(output_path)
    with _progress_lock:
        data = {
            "output_path": output_path,
            "completed_ids": list(completed_ids),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  [progress] Warning: Could not save progress: {e}")


def load_progress(output_path: str) -> set[str] | None:
    """Load previously completed IDs from progress file."""
    progress_path = _get_progress_path(output_path)
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ids = set(data.get("completed_ids", []))
        print(f"  [resume] Found progress file: {len(ids)} videos already completed")
        return ids
    except Exception:
        return None


def clear_progress(output_path: str):
    """Delete the progress file after successful completion."""
    progress_path = _get_progress_path(output_path)
    if os.path.exists(progress_path):
        os.remove(progress_path)


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_file(
    input_path: str,
    output_path: str,
    proxy_file: str = None,
    proxy_url: str = None,
    threads: int = DEFAULT_THREADS,
    timeout: int = DEFAULT_TIMEOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume: bool = False,
    retry_failed: bool = False,
    use_ytdlp: bool = True,
):
    """
    Process a JSON file of video metadata, fetching transcripts with full
    anti-blocking protection.

    Supports two modes:
    - threads=1: Sequential processing (simple, for debugging)
    - threads>1: Multi-threaded collaborative work-stealing

    Args:
        input_path: Path to input JSON (list of dicts with "video_id" key)
        output_path: Path to write output JSON (input + "transcript" field)
        proxy_file: Path to proxies.txt (one IP:PORT per line)
        proxy_url: Single proxy URL fallback (from .env or CLI)
        threads: Number of concurrent workers
        timeout: Per-operation timeout in seconds
        batch_size: Save progress every N new transcripts
        resume: If True, resume from progress file
        retry_failed: If True, re-attempt videos that previously returned null
    """
    if proxy_url is None:
        proxy_url = os.getenv("PROXY_URL")

    # Load input
    print(f"\n{'='*60}")
    print(f"  Transcript Fetcher v2 - Anti-Block Edition")
    print(f"{'='*60}")
    print(f"\n  Loading: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    total = len(videos)

    # Load existing output if resuming or retrying
    existing_transcripts: dict[str, str] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for v in existing:
                if v.get("transcript"):
                    existing_transcripts[v["video_id"]] = v["transcript"]
        except Exception:
            pass

    # Load progress file for resume
    completed_ids: set[str] = set()
    if resume:
        prev = load_progress(output_path)
        if prev:
            completed_ids = prev

    # Determine which videos need fetching
    already_have = 0
    need_fetch_ids = []
    for v in videos:
        vid = v["video_id"]
        has_transcript = vid in existing_transcripts
        was_completed = vid in completed_ids

        if has_transcript and not retry_failed:
            already_have += 1
            continue
        if was_completed and not retry_failed:
            already_have += 1
            continue

        need_fetch_ids.append(vid)

    print(f"  Total videos:          {total}")
    print(f"  Already have:          {already_have}")
    print(f"  Need to fetch:         {len(need_fetch_ids)}")

    if not need_fetch_ids:
        print(f"\n  All videos already have transcripts!")
        # Still save output in case format changed
        _save_output(output_path, videos, existing_transcripts)
        return already_have, total

    # Initialize proxy pool
    proxy_pool = ProxyPool(proxy_file=proxy_file, fallback_proxy_url=proxy_url)
    total_proxies, _ = proxy_pool.get_stats()
    print(f"  Proxies loaded:        {total_proxies}")
    print(f"  Threads:               {threads}")
    print(f"  Timeout:               {timeout}s per operation")
    print(f"  Selenium auto-refresh: {'available' if _HAS_SELENIUM else 'not installed'}")
    if use_ytdlp and _HAS_YTDLP:
        print(f"  yt-dlp fallback:       enabled")
    elif not _HAS_YTDLP:
        print(f"  yt-dlp fallback:       not installed (pip install yt-dlp)")
    else:
        print(f"  yt-dlp fallback:       disabled (--no-ytdlp)")
    print(f"{'='*60}\n")

    # Set up interrupt handling
    stop_flag = threading.Event()
    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        print("\n\n  *** Interrupted! Saving progress... ***")
        stop_flag.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # Choose processing mode
    if threads <= 1 or not proxy_pool.has_proxies():
        # Sequential mode (no proxy pool, or single thread)
        success, new_fetched, new_transcripts = _process_sequential(
            need_fetch_ids, proxy_pool, timeout, batch_size,
            output_path, completed_ids, stop_flag, use_ytdlp=use_ytdlp,
        )
    else:
        # Multi-threaded collaborative mode
        success, new_fetched, new_transcripts = _process_parallel(
            need_fetch_ids, proxy_pool, proxy_file, threads, timeout,
            batch_size, output_path, completed_ids, stop_flag, use_ytdlp=use_ytdlp,
        )

    # Restore signal handler
    signal.signal(signal.SIGINT, original_sigint)

    # Merge results
    existing_transcripts.update(new_transcripts)
    total_success = already_have + success

    # Final save
    _save_output(output_path, videos, existing_transcripts)

    was_interrupted = stop_flag.is_set()
    if was_interrupted:
        save_progress(output_path, videos, completed_ids)
        print(f"\n  *** Progress saved. Resume with --resume flag. ***")
    else:
        clear_progress(output_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {'INTERRUPTED' if was_interrupted else 'COMPLETE'}")
    print(f"{'='*60}")
    print(f"  Already had:    {already_have}")
    print(f"  Newly fetched:  {new_fetched}")
    print(f"  Total success:  {total_success}/{total} ({total_success/total*100:.1f}%)")
    print(f"  Failed/skipped: {len(need_fetch_ids) - new_fetched - (success - new_fetched if success > new_fetched else 0)}")
    print(f"  Output:         {output_path}")
    print(f"{'='*60}\n")

    return total_success, total


def _process_sequential(
    video_ids: list[str],
    proxy_pool: ProxyPool,
    timeout: int,
    batch_size: int,
    output_path: str,
    completed_ids: set[str],
    stop_flag: threading.Event,
    use_ytdlp: bool = True,
) -> tuple[int, int, dict[str, str]]:
    """Sequential processing -- one video at a time, with proxy rotation."""
    success = 0
    new_fetched = 0
    transcripts: dict[str, str] = {}
    total = len(video_ids)

    for i, vid in enumerate(video_ids):
        if stop_flag.is_set():
            break

        # Try proxies for this video
        tried_proxies: set[str] = set()
        attempts = 0
        max_attempts = MAX_PROXY_RETRIES if proxy_pool.has_proxies() else 1

        while attempts < max_attempts and not stop_flag.is_set():
            proxy = None
            if proxy_pool.has_proxies():
                proxy = proxy_pool.get_random_proxy(exclude=tried_proxies)
                if proxy is None:
                    break
                tried_proxies.add(proxy)

            result = fetch_single_video(vid, proxy=proxy, timeout=timeout, use_ytdlp=use_ytdlp)

            if result["transcript"]:
                transcripts[vid] = result["transcript"]
                completed_ids.add(vid)
                success += 1
                new_fetched += 1
                word_count = len(result["transcript"].split())
                method_label = result['method']
                print(f"  [{i+1}/{total}] OK: {vid} ({word_count} words) via {method_label}")
                break

            elif result["method"] == "no_transcript":
                completed_ids.add(vid)
                success += 0  # Not a success, but done
                print(f"  [{i+1}/{total}] SKIP: {vid}: no transcript available")
                break

            else:
                # Proxy error -- try next
                attempts += 1

        # Rate limiting
        time.sleep(random.uniform(THREAD_DELAY_MIN, THREAD_DELAY_MAX))

        # Batch checkpoint
        if new_fetched > 0 and new_fetched % batch_size == 0:
            save_progress(output_path, [], completed_ids)
            print(f"  --- Checkpoint: {new_fetched} new transcripts saved ---")

    return success, new_fetched, transcripts


def _process_parallel(
    video_ids: list[str],
    proxy_pool: ProxyPool,
    proxy_file: str | None,
    threads: int,
    timeout: int,
    batch_size: int,
    output_path: str,
    completed_ids: set[str],
    stop_flag: threading.Event,
    use_ytdlp: bool = True,
) -> tuple[int, int, dict[str, str]]:
    """
    Multi-threaded collaborative processing.
    Multiple threads can work on the SAME video with DIFFERENT proxies.
    """
    work_queue = VideoWorkQueue(video_ids, proxy_pool, proxy_file)
    transcripts: dict[str, str] = {}
    transcripts_lock = threading.Lock()
    last_print_time = [time.time()]  # Mutable for closure
    print_lock = threading.Lock()

    def _print_progress():
        """Print progress at most every 2 seconds."""
        now = time.time()
        with print_lock:
            if now - last_print_time[0] < 2.0:
                return
            last_print_time[0] = now

        completed, success, no_trans, failed, total = work_queue.get_progress()
        pct = (completed / total * 100) if total > 0 else 0
        bar_filled = int(pct // 2.5)
        bar = "#" * bar_filled + "." * (40 - bar_filled)
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  "
            f"OK:{success} SKIP:{no_trans} FAIL:{failed} / {total}  "
            f"proxies tried: {work_queue.total_proxy_attempts}   "
        )
        sys.stdout.flush()

    def worker():
        """Worker thread: continuously fetch videos until queue is empty."""
        while not stop_flag.is_set():
            work = work_queue.get_work()
            if work is None:
                break

            vid, proxy = work

            # Another thread may have finished this video
            if work_queue.is_video_done(vid):
                work_queue.release_work(vid)
                continue

            # Small jitter to spread requests
            time.sleep(random.uniform(THREAD_DELAY_MIN, THREAD_DELAY_MAX))

            result = fetch_single_video(vid, proxy=proxy, timeout=timeout, use_ytdlp=use_ytdlp)

            # Check again (another thread may have finished while we were working)
            if work_queue.is_video_done(vid):
                work_queue.release_work(vid)
                continue

            if result["transcript"]:
                work_queue.mark_completed(vid, result)
                with transcripts_lock:
                    transcripts[vid] = result["transcript"]
                    completed_ids.add(vid)

            elif result["method"] == "no_transcript":
                work_queue.mark_no_transcript(vid, result)
                if work_queue.is_video_done(vid):
                    with transcripts_lock:
                        completed_ids.add(vid)

            else:
                work_queue.mark_proxy_failed(vid, proxy)

            work_queue.release_work(vid)
            _print_progress()

            # Periodic progress save
            completed, _, _, _, _ = work_queue.get_progress()
            if completed > 0 and completed % batch_size == 0:
                save_progress(output_path, [], completed_ids)

            # Stop all workers if done
            if work_queue.is_all_done():
                stop_flag.set()

    # Launch worker threads
    print(f"  Launching {threads} worker threads...\n")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(worker) for _ in range(threads)]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"\n  [worker-error] {e}")

    # Clear the progress line
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()

    # Final stats
    completed, success, no_trans, failed, total = work_queue.get_progress()
    new_fetched = success
    print(f"  Completed: {completed}/{total}")
    print(f"  Success: {success}, No transcript: {no_trans}, Failed: {failed}")
    print(f"  Total proxy attempts: {work_queue.total_proxy_attempts}")
    print(f"  Proxy refreshes: {work_queue.proxy_refresh_count}/{MAX_PROXY_REFRESHES}")

    return success, new_fetched, transcripts


def _save_output(output_path: str, videos: list[dict], transcripts: dict[str, str]):
    """Merge transcripts into video list and save to output file."""
    for v in videos:
        vid = v["video_id"]
        if vid in transcripts:
            v["transcript"] = transcripts[vid]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(videos, f, indent=2, ensure_ascii=False)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Robust Transcript Fetcher v2 - Anti-Block Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic (direct connection)
  python -m src.scraper.transcript_fetcher_v2 --input data.json --output out.json

  # With proxy file + 8 threads
  python -m src.scraper.transcript_fetcher_v2 --input data.json --output out.json \\
      --proxy-file proxies.txt --threads 8

  # Resume interrupted run
  python -m src.scraper.transcript_fetcher_v2 --input data.json --output out.json \\
      --proxy-file proxies.txt --resume
        """,
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument(
        "--proxy-file", type=str, default=None,
        help="Path to proxy list file (one IP:PORT per line)",
    )
    parser.add_argument(
        "--proxy-url", type=str, default=None,
        help="Single proxy URL: http://user:pass@host:port (or set PROXY_URL in .env)",
    )
    parser.add_argument(
        "--threads", type=int, default=DEFAULT_THREADS,
        help=f"Number of concurrent workers (default: {DEFAULT_THREADS})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-operation timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Save progress every N transcripts (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from a previous interrupted run",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-attempt videos that previously returned null",
    )
    parser.add_argument(
        "--no-ytdlp", action="store_true",
        help="Disable yt-dlp fallback (use youtube_transcript_api only)",
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help="(Legacy) Delay between requests -- ignored, jitter is automatic",
    )

    args = parser.parse_args()

    process_file(
        input_path=args.input,
        output_path=args.output,
        proxy_file=args.proxy_file,
        proxy_url=args.proxy_url,
        threads=args.threads,
        timeout=args.timeout,
        batch_size=args.batch_size,
        resume=args.resume,
        retry_failed=args.retry_failed,
        use_ytdlp=not args.no_ytdlp,
    )


if __name__ == "__main__":
    main()
