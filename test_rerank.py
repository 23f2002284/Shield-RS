"""Quick smoke test for agent_rerank fix — loads catalog, runs rerank demo only."""
import pandas as pd
from pathlib import Path
from src.algorithms.matrix_factorization import agent_rerank

item_meta = pd.read_csv("data/processed/item_catalog.csv")

# Fake recommendations (real video IDs from catalog)
sample_vids = item_meta["video_id"].head(10).tolist()
fake_recs = [(vid, 0.5 + i * 0.05) for i, vid in enumerate(sample_vids)]

reranked = agent_rerank(fake_recs, item_meta)

print("Agent re-ranking demo:")
print(f"{'Rank':<5} {'CF Score':>10} {'Agent Score':>12}  Video ID")
for i, ((vid_cf, cf_sc), (vid_ag, ag_sc)) in enumerate(zip(fake_recs, reranked)):
    marker = "<-- reranked" if vid_cf != vid_ag else ""
    print(f"  {i+1:<3} {cf_sc:>10.4f} {ag_sc:>12.4f}  {vid_ag[:20]}  {marker}")

print("\nFix OK!")
