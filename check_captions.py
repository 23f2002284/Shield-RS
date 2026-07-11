import json

topics = [
    'learn_about_climate_change',
    'how_to_invest_for_beginners', 
    'history_of_ancient_Rome',
    'machine_learning_tutorial',
    'healthy_meal_prep',
]

total = 0
caps = 0
fetched = 0

for t in topics:
    path = f"data/scrapes/{t}_final.json"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    n = len(d)
    c = sum(1 for v in d if v.get("captions_available"))
    fe = sum(1 for v in d if v.get("transcript"))
    total += n
    caps += c
    fetched += fe
    print(f"  {t}: {n} videos | {c} have captions | {fe} transcripts fetched | {c - fe} MISSED")

print(f"\nTotal: {total} videos | {caps} have captions | {fetched} transcripts fetched | {caps - fetched} MISSED")
print(f"Videos with NO captions at all: {total - caps}")
