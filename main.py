import yt_dlp

ydl_opts = {
    'skip_download': True,
    'writeautomaticsub': True,
    'subtitlesformat': 'vtt',
    'subtitleslangs': ['en'],
    'outtmpl': 'transcripts/%(title)s.%(ext)s',
    'ignoreerrors': True,  # CRITICAL for batch processing: prevents the script from crashing on bad URLs
}

# A list of multiple YouTube videos
urls = [
    "https://www.youtube.com/watch?v=DGFAltVowvo",
    "https://www.youtube.com/watch?v=jNQXAC9IVRw", # Me at the zoo (oldest video)
    "https://www.youtube.com/watch?v=invalid_url_for_testing" # Will be skipped gracefully
]

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    # yt-dlp natively accepts a list of URLs to iterate through
    print(f"Starting batch download for {len(urls)} videos...")
    ydl.download(urls)
    print("Batch download complete!")
