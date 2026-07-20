import os
import subprocess
import sys

TOPICS = [
    "python programming for beginners",
    "rust programming crash course",
    "deep learning explained",
    "how to build a web app",
    "introduction to artificial intelligence",
    "latest space exploration news",
    "healthy diet tips",
    "best workout routines",
    "history of the universe",
    "ancient civilizations documentary",
    "quantum mechanics explained",
    "renewable energy future",
    "finance basics",
    "understanding the stock market",
    "cybersecurity basics",
    "how internet works",
    "introduction to blockchain"
]

def main():
    os.makedirs(os.path.join("data", "scrapes"), exist_ok=True)
    
    for topic in TOPICS:
        safe_name = topic.replace(" ", "_")
        raw_output = os.path.join("data", "scrapes", f"{safe_name}_raw.json")
        
        print(f"=== Scraping metadata for: '{topic}' ===")
        subprocess.run([
            sys.executable, "-m", "src.scraper.youtube_scraper",
            "--query", topic,
            "--max_results", "50",
            "--output", raw_output
        ], check=True)
        print(f"=== Completed: '{topic}' ===\n")

if __name__ == "__main__":
    main()
