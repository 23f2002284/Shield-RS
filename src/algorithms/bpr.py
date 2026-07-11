"""
src/algorithms/bpr.py
======================
Bayesian Personalized Ranking (Rendle et al. 2009).

BPR learns from pairwise preferences: given user u watched video i but NOT j,
we train u to score i > j. This is fundamentally better than pointwise MF
for implicit feedback because it directly optimizes ranking rather than
reconstruction error.

BPR Loss (per triplet):
    L = -log σ(x_ui - x_uj) + λ||Θ||²

    where x_ui = P_u · Q_i (dot product of user and item factors)

SGD update per triplet (u, i, j):
    δ = σ(x_uj - x_ui)    ← gradient signal (how wrong is the ranking?)
    P_u += lr * (δ * (Q_i - Q_j) - λ * P_u)
    Q_i += lr * (δ * P_u - λ * Q_i)
    Q_j += lr * (-δ * P_u - λ * Q_j)

Negative sampling strategy (from your pairwise_triplets.csv):
    - Hard negatives (same topic): makes model learn fine-grained topic distinctions
    - Easy negatives (different topic): makes model learn broad topic preferences
    Both types are essential for calibrated ranking.

Usage:
    python -m src.algorithms.bpr
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.algorithms.matrix_factorization import (
    ndcg_at_k, precision_at_k, recall_at_k, hit_rate_at_k,
    agent_rerank, load_data,
)

DATA_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# BPR Model
# ---------------------------------------------------------------------------

class BPRMatrixFactorization:
    """
    BPR-MF: Matrix Factorization optimized with BPR pairwise loss.

    Compared to ALS (pointwise):
    - ALS minimizes (watch_pct - predicted_score)^2 per interaction
    - BPR minimizes -log σ(score_pos - score_neg) per (pos, neg) pair
    BPR directly optimizes the ranking metric — more aligned with recommendation.

    Hard negatives (same topic) give a stronger training signal because
    the model must learn within-topic quality differences, not just topic affinity.
    """

    def __init__(
        self,
        n_factors     : int   = 64,
        n_epochs      : int   = 10,
        learning_rate : float = 0.01,
        regularization: float = 0.01,
        hard_neg_ratio: float = 0.5,   # fraction of hard negatives to use
        seed          : int   = 42,
        batch_size    : int   = 4096,  # mini-batch SGD
    ):
        self.n_factors      = n_factors
        self.n_epochs       = n_epochs
        self.lr             = learning_rate
        self.reg            = regularization
        self.hard_neg_ratio = hard_neg_ratio
        self.seed           = seed
        self.batch_size     = batch_size

        self.user_factors  : Optional[np.ndarray] = None
        self.item_factors  : Optional[np.ndarray] = None
        self.user_index    : dict[str, int] = {}
        self.item_index    : dict[str, int] = {}
        self.idx_to_item   : dict[int, str] = {}
        self.is_fitted     : bool = False
        self.training_auc  : list[float] = []

    def fit(
        self,
        triplets_df : pd.DataFrame,
        user_index  : dict[str, int],
        item_index  : dict[str, int],
        idx_to_item : dict[int, str],
    ) -> "BPRMatrixFactorization":
        """
        Train BPR on pairwise triplets.

        Args:
            triplets_df : DataFrame with columns [user_id, pos_video_id, neg_video_id, negative_type]
            user_index  : {user_id -> row_index}
            item_index  : {video_id -> col_index}
            idx_to_item : {col_index -> video_id}
        """
        self.user_index  = user_index
        self.item_index  = item_index
        self.idx_to_item = idx_to_item

        rng = np.random.default_rng(self.seed)
        n_users = len(user_index)
        n_items = len(item_index)

        # Initialize factors
        scale = 0.01
        self.user_factors = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)

        # Convert triplets to index arrays (filter unknowns)
        print(f"[BPR] Preparing {len(triplets_df):,} triplets...")
        users, pos_items, neg_items = self._prepare_triplets(triplets_df)
        print(f"[BPR] Valid triplets after filtering: {len(users):,}")

        n_triplets = len(users)
        print(f"[BPR] Training: {n_users} users x {n_items} items | "
              f"{self.n_factors} factors | {self.n_epochs} epochs")

        for epoch in range(self.n_epochs):
            t0 = time.time()

            # Shuffle triplets each epoch
            perm = rng.permutation(n_triplets)
            u_arr = users[perm]
            pi_arr = pos_items[perm]
            nj_arr = neg_items[perm]

            epoch_loss = 0.0
            epoch_correct = 0  # for AUC approximation

            # Mini-batch SGD
            for start in range(0, n_triplets, self.batch_size):
                end = min(start + self.batch_size, n_triplets)
                u_batch  = u_arr[start:end]
                pi_batch = pi_arr[start:end]
                nj_batch = nj_arr[start:end]

                # Compute scores
                x_ui = np.sum(self.user_factors[u_batch] * self.item_factors[pi_batch], axis=1)
                x_uj = np.sum(self.user_factors[u_batch] * self.item_factors[nj_batch], axis=1)
                x_uij = x_ui - x_uj

                # BPR gradient: sigmoid of negative difference
                sigma = self._sigmoid(-x_uij)  # δ = 1 - σ(x_uij) = σ(-x_uij)
                epoch_loss    += -np.log(1.0 - sigma + 1e-10).sum()
                epoch_correct += (x_uij > 0).sum()

                # Update factors (vectorized)
                # P_u update
                dP = (sigma[:, None] * (self.item_factors[pi_batch] - self.item_factors[nj_batch])
                      - self.reg * self.user_factors[u_batch])
                # Q_i (pos) update
                dQi = sigma[:, None] * self.user_factors[u_batch] - self.reg * self.item_factors[pi_batch]
                # Q_j (neg) update
                dQj = -sigma[:, None] * self.user_factors[u_batch] - self.reg * self.item_factors[nj_batch]

                np.add.at(self.user_factors, u_batch,  self.lr * dP)
                np.add.at(self.item_factors, pi_batch, self.lr * dQi)
                np.add.at(self.item_factors, nj_batch, self.lr * dQj)

            batch_count = n_triplets // self.batch_size + 1
            avg_loss = epoch_loss / n_triplets
            approx_auc = epoch_correct / n_triplets
            self.training_auc.append(float(approx_auc))
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:2d}/{self.n_epochs} | Loss: {avg_loss:.4f} | "
                  f"Approx AUC: {approx_auc:.4f} | Time: {elapsed:.1f}s")

        self.is_fitted = True
        return self

    def predict(self, user_id: str, video_id: str) -> float:
        u = self.user_index.get(user_id)
        v = self.item_index.get(video_id)
        if u is None or v is None:
            return 0.0
        return float(self.user_factors[u] @ self.item_factors[v])

    def recommend(
        self,
        user_id    : str,
        top_k      : int = 10,
        seen_items : Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        u = self.user_index.get(user_id)
        if u is None:
            return []
        scores = self.item_factors @ self.user_factors[u]
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            vid = self.idx_to_item.get(int(idx))
            if vid and (not seen_items or vid not in seen_items):
                results.append((vid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def evaluate(
        self,
        test_df  : pd.DataFrame,
        k_values : list[int] = [5, 10, 20],
        max_users: int = 300,
        seed     : int = 42,
    ) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        test_users = test_df["user_id"].unique()
        if len(test_users) > max_users:
            test_users = rng.choice(test_users, size=max_users, replace=False)

        results: dict[str, list[float]] = {
            f"{m}@{k}": []
            for k in k_values for m in ["precision", "recall", "ndcg", "hr"]
        }

        for user_id in test_users:
            if user_id not in self.user_index:
                continue
            user_test = test_df[test_df["user_id"] == user_id]
            relevant  = set(user_test[user_test["watch_pct"] > 0.5]["video_id"].tolist())
            if not relevant:
                continue
            recs    = self.recommend(user_id, top_k=max(k_values))
            rec_ids = [v for v, _ in recs]
            for k in k_values:
                results[f"precision@{k}"].append(precision_at_k(rec_ids, relevant, k))
                results[f"recall@{k}"].append(recall_at_k(rec_ids, relevant, k))
                results[f"ndcg@{k}"].append(ndcg_at_k(rec_ids, relevant, k))
                results[f"hr@{k}"].append(hit_rate_at_k(rec_ids, relevant, k))

        return {k: float(np.mean(v)) if v else 0.0 for k, v in results.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_triplets(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert string IDs to indices, filtering out unknowns."""
        users, pos_items, neg_items = [], [], []
        for _, row in df.iterrows():
            u  = self.user_index.get(row["user_id"])
            pi = self.item_index.get(row["pos_video_id"])
            nj = self.item_index.get(row["neg_video_id"])
            if u is not None and pi is not None and nj is not None:
                users.append(u)
                pos_items.append(pi)
                neg_items.append(nj)
        return (
            np.array(users,     dtype=np.int32),
            np.array(pos_items, dtype=np.int32),
            np.array(neg_items, dtype=np.int32),
        )

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.algorithms.matrix_factorization import train_test_split

    print("\n" + "=" * 60)
    print("  BPR: Bayesian Personalized Ranking")
    print("=" * 60)

    # Load shared indices
    matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item = load_data()
    _, test_df = train_test_split()

    # Load triplets — use a sample for speed (full 1.3M can be slow for demo)
    print("\n[BPR] Loading pairwise triplets...")
    triplets = pd.read_csv(DATA_DIR / "pairwise_triplets.csv")
    # Sample to keep training fast (use 200K triplets — still 10x more than interactions)
    sample_size = min(200_000, len(triplets))
    triplets    = triplets.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f"  Using {len(triplets):,} triplets "
          f"(hard: {(triplets.negative_type=='hard').sum():,} | "
          f"easy: {(triplets.negative_type=='easy').sum():,})")

    model = BPRMatrixFactorization(
        n_factors=64, n_epochs=10, learning_rate=0.01,
        regularization=0.01, batch_size=4096,
    )
    model.fit(triplets, user_idx, item_idx, idx_to_item)

    print("\n[BPR] Evaluating...")
    metrics = model.evaluate(test_df, k_values=[5, 10, 20], max_users=300)
    print("\nBPR Evaluation Results:")
    for k in [5, 10, 20]:
        print(f"  @{k:<3}: P={metrics[f'precision@{k}']:.4f} "
              f"R={metrics[f'recall@{k}']:.4f} "
              f"NDCG={metrics[f'ndcg@{k}']:.4f} "
              f"HR={metrics[f'hr@{k}']:.4f}")

    # Save metrics
    results_path = Path("experiments") / "bpr_evaluation.csv"
    results_path.parent.mkdir(exist_ok=True)
    pd.DataFrame([{"model": "BPR", **metrics}]).to_csv(results_path, index=False)
    print(f"\n[Saved] -> {results_path}")
