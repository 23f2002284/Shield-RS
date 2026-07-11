import pandas as pd
import json
from scipy.sparse import load_npz
from pathlib import Path

p = Path("data/processed")
print("=== OUTPUT FILE VERIFICATION ===")

# users.json
users = json.load(open(p / "users.json", encoding="utf-8"))
types = {}
for u in users:
    types[u["user_type"]] = types.get(u["user_type"], 0) + 1
print(f"users.json         : {len(users):,} users")
for k, v in sorted(types.items()):
    print(f"  {k:<10}: {v:,}")

# interactions.csv
df_i = pd.read_csv(p / "interactions.csv")
print(f"interactions.csv   : {len(df_i):,} rows | {df_i.shape[1]} cols")
print(f"  avg watch_pct    : {df_i['watch_pct'].mean():.3f}")
print(f"  skipped rate     : {df_i['skipped'].mean():.3f}")
print(f"  rows with rating : {df_i['explicit_rating'].notna().sum():,}")
topic_counts = df_i["topic"].value_counts().to_dict()
for t, c in topic_counts.items():
    print(f"  {t:<40}: {c:,}")

# user_item_matrix
m = load_npz(p / "user_item_matrix.npz")
sparsity = 1 - m.nnz / (m.shape[0] * m.shape[1])
print(f"user_item_matrix   : {m.shape} | nnz={m.nnz:,} | sparsity={sparsity:.4f}")

# BPR triplets
df_bpr = pd.read_csv(p / "pairwise_triplets.csv")
hard_count = len(df_bpr[df_bpr["negative_type"] == "hard"])
easy_count = len(df_bpr[df_bpr["negative_type"] == "easy"])
print(f"pairwise_triplets  : {len(df_bpr):,} rows (hard={hard_count:,} easy={easy_count:,})")

# sessions
sessions = json.load(open(p / "sessions.json", encoding="utf-8"))
seq_lens = [len(s["video_sequence"]) for s in sessions]
avg_len = sum(seq_lens) / len(seq_lens) if seq_lens else 0
print(f"sessions.json      : {len(sessions):,} sessions | avg_len={avg_len:.2f} videos/session")

# context_vectors
df_ctx = pd.read_csv(p / "context_vectors.csv")
print(f"context_vectors    : {len(df_ctx):,} rows | {df_ctx.shape[1]} cols")
print(f"  is_weekend rate  : {df_ctx['is_weekend'].mean():.3f}")
print(f"  is_evening rate  : {df_ctx['is_evening'].mean():.3f}")

# item_catalog
df_cat = pd.read_csv(p / "item_catalog.csv")
print(f"item_catalog       : {len(df_cat):,} videos | {df_cat.shape[1]} cols")
cat_topics = df_cat["topic"].value_counts().to_dict()
for t, c in cat_topics.items():
    print(f"  {t:<40}: {c:,}")

# File sizes
print("\n=== FILE SIZES ===")
for fname in ["users.json", "interactions.csv", "user_item_matrix.npz",
              "pairwise_triplets.csv", "sessions.json", "context_vectors.csv",
              "item_catalog.csv", "user_index.json", "item_index.json"]:
    fpath = p / fname
    if fpath.exists():
        size_mb = fpath.stat().st_size / 1024 / 1024
        print(f"  {fname:<30}: {size_mb:.2f} MB")

print("\n=== ALL FILES OK ===")
