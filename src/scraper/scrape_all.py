import os
import subprocess
import sys

TOPICS = [
    "learn about climate change",
    "how to invest for beginners",
    "history of ancient Rome",
    "machine learning tutorial",
    "healthy meal prep"
]

def main():
    # Make sure data/scrapes exists
    os.makedirs(os.path.join("data", "scrapes"), exist_ok=True)
    
    for topic in TOPICS:
        safe_name = topic.replace(" ", "_")
        raw_output = os.path.join("data", "scrapes", f"{safe_name}_raw.json")
        final_output = os.path.join("data", "scrapes", f"{safe_name}_final.json")
        
        print(f"=== Scraping metadata for: '{topic}' ===")
        # Run youtube_scraper.py
        subprocess.run([
            sys.executable, "-m", "src.scraper.youtube_scraper",
            "--query", topic,
            "--max_results", "200",
            "--output", raw_output
        ], check=True)
        
        print(f"=== Fetching transcripts for: '{topic}' ===")
        # Run transcript_fetcher.py
        subprocess.run([
            sys.executable, "-m", "src.scraper.transcript_fetcher",
            "--input", raw_output,
            "--output", final_output
        ], check=True)
        
        print(f"=== Completed: '{topic}' ===\n")

if __name__ == "__main__":
    main()
