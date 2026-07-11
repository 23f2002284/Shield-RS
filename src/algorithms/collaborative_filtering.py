"""
src/algorithms/collaborative_filtering.py
==========================================
User-Based and Item-Based Collaborative Filtering (memory-based RS).

Unlike MF models that learn latent factors, memory-based CF computes
similarity directly from the observed user-item matrix — no training needed.

User-Based CF:
    To recommend for user u:
    1. Find K most similar users (cosine similarity on watch_pct vectors)
    2. Aggregate their watch_pct scores: score(u, i) = Σ_v sim(u,v) * r_vi / Σ_v |sim(u,v)|
    3. Return top-N unobserved items by predicted score

Item-Based CF:
    To recommend for user u:
    1. For each item u has watched, find K most similar items (cosine on item columns)
    2. score(u, i) = Σ_j sim(i,j) * r_uj  (j = items u has already watched)
    3. Return top-N unobserved items

Item-Based CF is generally preferred in production:
  - Item similarities are stable over time (recompute weekly)
  - User similarities change rapidly (recompute hourly in production)
  - Item-CF recommendations are more explainable ("because you watched X")

Both models support agent-centric re-ranking via agent_rerank().

Usage:
    python -m src.algorithms.collaborative_filtering
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity

from src.algorithms.matrix_factorization import (
    ndcg_at_k, precision_at_k, recall_at_k, hit_rate_at_k,
    agent_rerank, load_data, train_test_split, build_matrix_from_df,
)

DATA_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# User-Based CF
# ---------------------------------------------------------------------------

class UserBasedCF:
    """
    Memory-based User-Based Collaborative Filtering.

    Prediction for user u on item i:
        score(u, i) = Σ_{v in N(u)} sim(u, v) * r_vi
                      / Σ_{v in N(u)} |sim(u, v)|

    Where N(u) = top-K most similar users who have rated item i.
    """

    def __init__(self, k_neighbors: int = 50, min_overlap: int = 3):
        """
        Args:
            k_neighbors : how many similar users to aggregate over
            min_overlap : minimum number of co-rated items to trust similarity
        """
        self.k           = k_neighbors
        self.min_overlap = min_overlap
        self.matrix      : Optional[csr_matrix] = None
        self.user_index  : dict[str, int] = {}
        self.item_index  : dict[str, int] = {}
        self.idx_to_user : dict[int, str] = {}
        self.idx_to_item : dict[int, str] = {}
        self.item_meta   : Optional[pd.DataFrame] = None

        # Pre-computed similarity matrix [n_users x n_users]
        self.similarity_matrix: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(
        self,
        matrix    : csr_matrix,
        user_index: dict[str, int],
        item_index: dict[str, int],
        idx_to_user: dict[int, str],
        idx_to_item: dict[int, str],
        item_meta  : Optional[pd.DataFrame] = None,
    ) -> "UserBasedCF":
        self.matrix     = matrix.toarray().astype(np.float32)
        self.user_index = user_index
        self.item_index = item_index
        self.idx_to_user= idx_to_user
        self.idx_to_item= idx_to_item
        self.item_meta  = item_meta

        n_users = self.matrix.shape[0]
        print(f"[UserBasedCF] Computing {n_users}x{n_users} user similarity matrix...")
        t0 = time.time()

        # Cosine similarity on dense matrix rows
        self.similarity_matrix = cosine_similarity(self.matrix).astype(np.float32)
        # Zero out self-similarity
        np.fill_diagonal(self.similarity_matrix, 0.0)

        elapsed = time.time() - t0
        self.is_fitted = True
        print(f"[UserBasedCF] Similarity computed in {elapsed:.2f}s")
        return self

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Weighted average of neighbor ratings for a single (user, item) pair."""
        sims = self.similarity_matrix[user_idx]  # [n_users]

        # Only consider neighbors who have rated this item
        rated_mask = self.matrix[:, item_idx] > 0  # [n_users] bool
        neighbor_sims = sims * rated_mask.astype(np.float32)

        # Top-K neighbors
        top_k_idx = np.argsort(neighbor_sims)[::-1][:self.k]
        top_k_sims   = neighbor_sims[top_k_idx]
        top_k_ratings = self.matrix[top_k_idx, item_idx]

        sim_sum = top_k_sims.sum()
        if sim_sum < 1e-9:
            return 0.0
        return float((top_k_sims * top_k_ratings).sum() / sim_sum)

    def recommend(
        self,
        user_id    : str,
        top_k      : int = 10,
        seen_items : Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        u = self.user_index.get(user_id)
        if u is None:
            return []

        sims = self.similarity_matrix[u]  # [n_users]

        # Weighted sum of neighbor rating vectors
        # For all items at once (vectorized)
        top_n_idx = np.argsort(sims)[::-1][:self.k]
        top_sims  = sims[top_n_idx]                     # [K]
        neighbor_mat = self.matrix[top_n_idx, :]        # [K x n_items]

        sim_sum = top_sims.sum()
        if sim_sum < 1e-9:
            return []

        scores = (top_sims @ neighbor_mat) / sim_sum    # [n_items]
        # Zero out already-seen items
        seen_mask = self.matrix[u, :] > 0
        scores[seen_mask] = -1.0

        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            vid = self.idx_to_item.get(int(idx))
            if vid and (not seen_items or vid not in seen_items):
                results.append((vid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def evaluate(
        self, test_df: pd.DataFrame, k_values: list[int] = [5, 10, 20],
        max_users: int = 300, seed: int = 42,
    ) -> dict[str, float]:
        return _evaluate_generic(self, test_df, k_values, max_users, seed)


# ---------------------------------------------------------------------------
# Item-Based CF
# ---------------------------------------------------------------------------

class ItemBasedCF:
    """
    Memory-based Item-Based Collaborative Filtering.

    Prediction for user u on item i:
        score(u, i) = Σ_{j in rated(u) ∩ N(i)} sim(i, j) * r_uj
                      / Σ_{j in rated(u) ∩ N(i)} |sim(i, j)|

    Where N(i) = top-K most similar items to item i.
    Item similarity is pre-computed (stable, reusable across users).
    """

    def __init__(self, k_neighbors: int = 30):
        self.k           = k_neighbors
        self.matrix      : Optional[np.ndarray] = None
        self.user_index  : dict[str, int] = {}
        self.item_index  : dict[str, int] = {}
        self.idx_to_item : dict[int, str] = {}
        self.item_meta   : Optional[pd.DataFrame] = None

        # Pre-computed item similarity [n_items x n_items]
        self.similarity_matrix : Optional[np.ndarray] = None
        # For each item, top-K neighbors: [n_items x K] indices and scores
        self.top_k_neighbors   : Optional[np.ndarray] = None
        self.top_k_sims        : Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(
        self,
        matrix    : csr_matrix,
        user_index: dict[str, int],
        item_index: dict[str, int],
        idx_to_user: dict[int, str],
        idx_to_item: dict[int, str],
        item_meta  : Optional[pd.DataFrame] = None,
    ) -> "ItemBasedCF":
        self.matrix     = matrix.toarray().astype(np.float32)
        self.user_index = user_index
        self.item_index = item_index
        self.idx_to_item= idx_to_item
        self.item_meta  = item_meta

        n_items = self.matrix.shape[1]
        print(f"[ItemBasedCF] Computing {n_items}x{n_items} item similarity matrix...")
        t0 = time.time()

        # Compute item-item cosine similarity on transposed matrix (items as rows)
        item_mat = self.matrix.T  # [n_items x n_users]
        self.similarity_matrix = cosine_similarity(item_mat).astype(np.float32)
        np.fill_diagonal(self.similarity_matrix, 0.0)

        # Pre-compute top-K neighbors for each item
        self.top_k_neighbors = np.argsort(self.similarity_matrix, axis=1)[:, ::-1][:, :self.k]
        self.top_k_sims      = np.take_along_axis(
            self.similarity_matrix, self.top_k_neighbors, axis=1
        )

        elapsed = time.time() - t0
        self.is_fitted = True
        print(f"[ItemBasedCF] Similarity computed in {elapsed:.2f}s | "
              f"top-{self.k} neighbors cached")
        return self

    def recommend(
        self,
        user_id    : str,
        top_k      : int = 10,
        seen_items : Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        u = self.user_index.get(user_id)
        if u is None:
            return []

        user_ratings = self.matrix[u, :]   # [n_items] — which items user rated
        rated_indices = np.where(user_ratings > 0)[0]
        if len(rated_indices) == 0:
            return []

        n_items = self.matrix.shape[1]
        scores = np.zeros(n_items, dtype=np.float32)
        sim_sums = np.zeros(n_items, dtype=np.float32)

        # For each rated item j, accumulate weighted similarity to all candidates
        for j in rated_indices:
            r_uj = user_ratings[j]
            neighbors = self.top_k_neighbors[j]    # [K] indices
            sims      = self.top_k_sims[j]         # [K] similarities
            scores[neighbors]   += sims * r_uj
            sim_sums[neighbors] += np.abs(sims)

        # Normalize
        valid = sim_sums > 0
        scores[valid] /= sim_sums[valid]
        # Mask already-rated items
        scores[user_ratings > 0] = -1.0

        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            vid = self.idx_to_item.get(int(idx))
            if vid and (not seen_items or vid not in seen_items):
                results.append((vid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def evaluate(
        self, test_df: pd.DataFrame, k_values: list[int] = [5, 10, 20],
        max_users: int = 300, seed: int = 42,
    ) -> dict[str, float]:
        return _evaluate_generic(self, test_df, k_values, max_users, seed)


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------

def _evaluate_generic(
    model, test_df, k_values, max_users, seed
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    test_users = test_df["user_id"].unique()
    if len(test_users) > max_users:
        test_users = rng.choice(test_users, size=max_users, replace=False)

    results: dict[str, list[float]] = {
        f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg", "hr"]
    }
    for user_id in test_users:
        if user_id not in model.user_index:
            continue
        user_test = test_df[test_df["user_id"] == user_id]
        relevant  = set(user_test[user_test["watch_pct"] > 0.5]["video_id"].tolist())
        if not relevant:
            continue
        recs    = model.recommend(user_id, top_k=max(k_values))
        rec_ids = [v for v, _ in recs]
        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(rec_ids, relevant, k))
            results[f"recall@{k}"].append(recall_at_k(rec_ids, relevant, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(rec_ids, relevant, k))
            results[f"hr@{k}"].append(hit_rate_at_k(rec_ids, relevant, k))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in results.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  COLLABORATIVE FILTERING: User-Based & Item-Based")
    print("=" * 60)

    matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item = load_data()
    train_df, test_df = train_test_split()
    train_matrix = build_matrix_from_df(train_df, user_idx, item_idx)

    results_rows = []

    # User-Based CF
    print("\n[Step 1] User-Based CF (K=50 neighbors)...")
    ubcf = UserBasedCF(k_neighbors=50)
    ubcf.fit(train_matrix, user_idx, item_idx, idx_to_user, idx_to_item, item_meta)
    print("[UserCF] Evaluating...")
    ubcf_metrics = ubcf.evaluate(test_df, k_values=[5, 10, 20], max_users=300)
    results_rows.append({"model": "UserBasedCF", **ubcf_metrics})
    for k in [5, 10, 20]:
        print(f"  @{k:<3}: P={ubcf_metrics[f'precision@{k}']:.4f} "
              f"R={ubcf_metrics[f'recall@{k}']:.4f} "
              f"NDCG={ubcf_metrics[f'ndcg@{k}']:.4f} "
              f"HR={ubcf_metrics[f'hr@{k}']:.4f}")

    # Item-Based CF
    print("\n[Step 2] Item-Based CF (K=30 neighbors)...")
    ibcf = ItemBasedCF(k_neighbors=30)
    ibcf.fit(train_matrix, user_idx, item_idx, idx_to_user, idx_to_item, item_meta)
    print("[ItemCF] Evaluating...")
    ibcf_metrics = ibcf.evaluate(test_df, k_values=[5, 10, 20], max_users=300)
    results_rows.append({"model": "ItemBasedCF", **ibcf_metrics})
    for k in [5, 10, 20]:
        print(f"  @{k:<3}: P={ibcf_metrics[f'precision@{k}']:.4f} "
              f"R={ibcf_metrics[f'recall@{k}']:.4f} "
              f"NDCG={ibcf_metrics[f'ndcg@{k}']:.4f} "
              f"HR={ibcf_metrics[f'hr@{k}']:.4f}")

    # Save
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    pd.DataFrame(results_rows).to_csv(out / "cf_evaluation.csv", index=False)
    print(f"\n[Saved] -> experiments/cf_evaluation.csv")
