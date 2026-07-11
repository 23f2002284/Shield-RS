"""
Run the robust transcript fetcher on all 5 topics.

Usage:
  # Without proxy
  python fetch_all_transcripts.py

  # With proxy
  python fetch_all_transcripts.py --proxy-url http://user:pass@host:port

  # With proxy from .env (add PROXY_URL=http://... to .env)
  python fetch_all_transcripts.py

  # Retry failed videos with proxy
  python fetch_all_transcripts.py --proxy-url http://user:pass@host:port --retry-failed
"""
import subprocess
import sys
import argparse

topics = [
    "learn_about_climate_change",
    "how_to_invest_for_beginners",
    "history_of_ancient_Rome",
    "machine_learning_tutorial",
    "healthy_meal_prep",
]

parser = argparse.ArgumentParser()
parser.add_argument("--proxy-url", type=str, default=None)
parser.add_argument("--retry-failed", action="store_true")
parser.add_argument("--delay", type=float, default=0.5)
args = parser.parse_args()

for topic in topics:
    print(f"\n{'='*60}")
    print(f"TOPIC: {topic}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "src/scraper/transcript_fetcher.py",
        "--input", f"data/scrapes/{topic}_final.json",
        "--output", f"data/scrapes/{topic}_final.json",
        "--delay", str(args.delay),
        "--batch-size", "20",
    ]

    if args.proxy_url:
        cmd.extend(["--proxy-url", args.proxy_url])

    if args.retry_failed:
        cmd.append("--retry-failed")

    subprocess.run(cmd, check=True)

print("\n\nDONE! All topics processed.")
