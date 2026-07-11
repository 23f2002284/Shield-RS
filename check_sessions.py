import json
from collections import Counter
sessions = json.load(open("data/processed/sessions.json", encoding="utf-8"))
lengths = [len(s["video_sequence"]) for s in sessions]
c = Counter(lengths)
print("Session length distribution:")
for l in sorted(c)[:15]:
    print(f"  len={l}: {c[l]:,} sessions")
print(f"Total sessions: {len(sessions):,}")
print(f"Multi-item (>=2): {sum(1 for s in sessions if len(s['video_sequence'])>=2):,}")
print(f"Multi-item (>=3): {sum(1 for s in sessions if len(s['video_sequence'])>=3):,}")
