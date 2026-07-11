"""
src/algorithms/two_tower.py
============================
Two-Tower Neural Network Recommender System.

Architecture:
  User Tower:  user_features [32-dim] → Linear(32→64) → ReLU → Linear(64→32) → L2-normalize
  Item Tower:  item_features [15-dim] → Linear(15→64) → ReLU → Linear(64→32) → L2-normalize
  Score:       cosine_similarity(user_embedding, item_embedding)

Training:
  BPR loss over (user, pos_item, neg_item) triplets from pairwise_triplets.csv.
  Loss = -log σ(score(u, pos) - score(u, neg)) + λ||W||²

Why Two-Tower for Shield:
  1. Dense feature inputs allow cold-start recommendations (no interaction history needed)
  2. User features include manipulation_aversion, credibility_sensitivity
     → the AGENT's preferences are directly encoded in the user tower
  3. Item features have SIGNAL/CONFOUNDER labeled columns (from feature_encoder.py)
     → the item tower learns to weight quality signals vs. popularity confounders
  4. At inference, item embeddings are pre-computed offline (ANN search)
     → production-scale recommendation in <1ms per user

Implementation: pure numpy (no PyTorch).
  Forward pass:  matrix multiply + ReLU activation
  Backward pass: analytically derived gradients for BPR loss
  Optimizer:     Adam with weight decay

Usage:
    python -m src.algorithms.two_tower
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Numpy neural network primitives
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ReLU: 1 if x > 0 else 0."""
    return (x > 0).astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalization."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class AdamOptimizer:
    """
    Adam optimizer for a single parameter matrix.
    Maintains running moment estimates (m, v) per parameter.
    """
    def __init__(self, shape: tuple, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * param
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# Two-Tower Tower (shared structure for user & item towers)
# ---------------------------------------------------------------------------

class Tower:
    """
    A 2-layer MLP tower: input_dim → hidden_dim → embed_dim, with ReLU + L2-norm.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int,
                 lr: float = 1e-3, weight_decay: float = 1e-4, seed_offset: int = 0):
        rng   = np.random.default_rng(42 + seed_offset)
        scale = 0.01

        # Layer 1: input_dim → hidden_dim
        self.W1 = rng.normal(0, scale, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        # Layer 2: hidden_dim → embed_dim
        self.W2 = rng.normal(0, scale, (hidden_dim, embed_dim)).astype(np.float32)
        self.b2 = np.zeros(embed_dim, dtype=np.float32)

        # Adam optimizers per parameter
        self.opt_W1 = AdamOptimizer(self.W1.shape, lr, weight_decay=weight_decay)
        self.opt_b1 = AdamOptimizer(self.b1.shape, lr)
        self.opt_W2 = AdamOptimizer(self.W2.shape, lr, weight_decay=weight_decay)
        self.opt_b2 = AdamOptimizer(self.b2.shape, lr)

        # Cache for backward pass
        self._x  : Optional[np.ndarray] = None   # input
        self._h1 : Optional[np.ndarray] = None   # after layer 1 (pre-relu)
        self._a1 : Optional[np.ndarray] = None   # after relu
        self._h2 : Optional[np.ndarray] = None   # after layer 2 (pre-norm)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: [batch x input_dim]
        returns: [batch x embed_dim] (L2-normalized)
        """
        self._x  = x
        self._h1 = x @ self.W1 + self.b1
        self._a1 = relu(self._h1)
        self._h2 = self._a1 @ self.W2 + self.b2
        return l2_normalize(self._h2)

    def backward(self, d_embed: np.ndarray) -> None:
        """
        d_embed: [batch x embed_dim] upstream gradient from loss w.r.t. normalized output
        Updates W1, b1, W2, b2 using Adam.
        """
        batch = d_embed.shape[0]

        # Gradient through L2 normalization: d(x/||x||)/dx
        h2   = self._h2
        norms = np.linalg.norm(h2, axis=-1, keepdims=True) + 1e-8
        # d_norm: [batch x embed_dim]
        d_h2 = (d_embed / norms
                - h2 * np.sum(d_embed * h2, axis=-1, keepdims=True) / (norms ** 3))

        # Layer 2 gradients
        dW2 = (self._a1.T @ d_h2) / batch
        db2 = d_h2.mean(axis=0)
        d_a1 = d_h2 @ self.W2.T

        # ReLU gradient
        d_h1 = d_a1 * relu_grad(self._h1)

        # Layer 1 gradients
        dW1 = (self._x.T @ d_h1) / batch
        db1 = d_h1.mean(axis=0)

        # Adam updates
        self.W2 = self.opt_W2.step(self.W2, dW2)
        self.b2 = self.opt_b2.step(self.b2, db2)
        self.W1 = self.opt_W1.step(self.W1, dW1)
        self.b1 = self.opt_b1.step(self.b1, db1)


# ---------------------------------------------------------------------------
# Two-Tower Model
# ---------------------------------------------------------------------------

class TwoTowerModel:
    """
    Two-Tower Neural Recommender.

    User Tower:  user_features [32] → 64 → 32 (L2-normalized embedding)
    Item Tower:  item_features [15] → 64 → 32 (L2-normalized embedding)
    Score:       cosine_similarity = dot product (after L2-norm)

    Training:
        BPR loss over (user, pos_item, neg_item) triplets.
        Both towers updated jointly via backprop through the BPR loss.

    Key advantages for Shield:
        1. user_features include manipulation_aversion, credibility_sensitivity
           → model learns that users who want quality get quality recommendations
        2. item_features explicitly include SIGNAL vs. CONFOUNDER columns
           → the item tower learns to separate quality from popularity
        3. Cold-start: can recommend to new users given only their feature vector
           (no interaction history needed — unlike MF/CF/BPR)
    """

    def __init__(
        self,
        user_dim    : int   = 32,
        item_dim    : int   = 15,
        hidden_dim  : int   = 64,
        embed_dim   : int   = 32,
        n_epochs    : int   = 15,
        lr          : float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size  : int   = 512,
        seed        : int   = 42,
    ):
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.seed       = seed

        self.user_tower = Tower(user_dim, hidden_dim, embed_dim, lr, weight_decay, seed_offset=0)
        self.item_tower = Tower(item_dim, hidden_dim, embed_dim, lr, weight_decay, seed_offset=1)

        # Pre-computed item embeddings (set after fit)
        self.item_embeddings: Optional[np.ndarray] = None  # [n_items x embed_dim]

        # Index mappings
        self.user_index  : dict[str, int] = {}
        self.item_index  : dict[str, int] = {}
        self.idx_to_item : list[str]      = []

        self.training_loss: list[float] = []
        self.is_fitted = False

    def fit(
        self,
        triplets_df  : pd.DataFrame,
        user_features: np.ndarray,   # [n_users x user_dim]
        item_features: np.ndarray,   # [n_items x item_dim]
        user_index   : dict[str, int],
        item_index   : dict[str, int],
        idx_to_item  : list[str],
    ) -> "TwoTowerModel":
        rng = np.random.default_rng(self.seed)
        self.user_index  = user_index
        self.item_index  = item_index
        self.idx_to_item = idx_to_item

        # Build triplet index arrays
        print(f"[TwoTower] Preparing triplets...")
        u_arr, pi_arr, nj_arr = [], [], []
        for _, row in triplets_df.iterrows():
            u  = user_index.get(row["user_id"])
            pi = item_index.get(row["pos_video_id"])
            nj = item_index.get(row["neg_video_id"])
            if all(x is not None for x in [u, pi, nj]):
                u_arr.append(u)
                pi_arr.append(pi)
                nj_arr.append(nj)

        u_arr  = np.array(u_arr,  dtype=np.int32)
        pi_arr = np.array(pi_arr, dtype=np.int32)
        nj_arr = np.array(nj_arr, dtype=np.int32)
        n_triples = len(u_arr)
        print(f"[TwoTower] {n_triples:,} valid triplets | Training {self.n_epochs} epochs...")

        for epoch in range(self.n_epochs):
            t0   = time.time()
            perm = rng.permutation(n_triples)
            u_p  = u_arr[perm]
            pi_p = pi_arr[perm]
            nj_p = nj_arr[perm]

            epoch_loss  = 0.0
            epoch_corr  = 0

            for start in range(0, n_triples, self.batch_size):
                end = min(start + self.batch_size, n_triples)
                B   = end - start
                ub  = u_p[start:end]
                pb  = pi_p[start:end]
                nb  = nj_p[start:end]

                # --- USER TOWER (single forward) ---
                u_feats  = user_features[ub].astype(np.float32)   # [B x user_dim]
                u_emb    = self.user_tower.forward(u_feats)         # [B x embed_dim]

                # --- ITEM TOWER: process pos AND neg in ONE combined batch ---
                # Concatenate [pos | neg] → single forward → split → single backward
                pi_feats = item_features[pb].astype(np.float32)    # [B x item_dim]
                nj_feats = item_features[nb].astype(np.float32)    # [B x item_dim]
                combined_feats = np.vstack([pi_feats, nj_feats])   # [2B x item_dim]
                combined_emb   = self.item_tower.forward(combined_feats)  # [2B x embed_dim]
                pi_emb = combined_emb[:B]   # [B x embed_dim]
                nj_emb = combined_emb[B:]   # [B x embed_dim]

                # --- BPR loss ---
                s_pos = np.sum(u_emb * pi_emb, axis=1)   # [B] cosine sims
                s_neg = np.sum(u_emb * nj_emb, axis=1)   # [B]
                diff  = s_pos - s_neg
                # σ(-diff) = gradient signal: large when s_neg > s_pos (wrong ranking)
                delta = sigmoid(-diff)                    # [B]

                epoch_loss += -np.log(sigmoid(diff) + 1e-10).sum()
                epoch_corr += (diff > 0).sum()

                # --- Gradients (BPR loss, all signs verified) ---
                # delta = σ(s_neg - s_pos): large when ranking is WRONG
                #
                # d_loss/d(u_emb)  = delta * (nj_emb - pi_emb)
                #   → u_emb -= lr * delta*(nj-pi)  = u_emb += lr*delta*(pi-nj)
                #   → moves u_emb TOWARD pi, AWAY from nj  ✓
                #
                # d_loss/d(pi_emb) = -delta * u_emb
                #   → pi_emb -= lr * (-delta*u) = pi_emb += lr*delta*u
                #   → moves pi_emb TOWARD u_emb  ✓
                #
                # d_loss/d(nj_emb) = +delta * u_emb
                #   → nj_emb -= lr * (delta*u) = moves nj AWAY from u  ✓
                d_u_emb  = delta[:, None] * (nj_emb - pi_emb)   # [B x embed_dim]
                d_pi_emb = -delta[:, None] * u_emb               # [B x embed_dim]
                d_nj_emb =  delta[:, None] * u_emb               # [B x embed_dim]
                d_combined = np.vstack([d_pi_emb, d_nj_emb])     # [2B x embed_dim]

                # --- Backward through towers (one call each) ---
                self.user_tower.backward(d_u_emb)
                self.item_tower.backward(d_combined)

            avg_loss = epoch_loss / n_triples
            auc      = epoch_corr / n_triples
            self.training_loss.append(float(avg_loss))
            print(f"  Epoch {epoch+1:2d}/{self.n_epochs} | "
                  f"Loss: {avg_loss:.4f} | AUC: {auc:.4f} | Time: {time.time()-t0:.1f}s")

        # Pre-compute all item embeddings for fast inference
        print("[TwoTower] Pre-computing item embeddings...")
        self.item_embeddings = self.item_tower.forward(item_features.astype(np.float32))
        self.is_fitted = True
        print(f"[TwoTower] Item embeddings: {self.item_embeddings.shape}")
        return self

    def get_user_embedding(self, user_features_vec: np.ndarray) -> np.ndarray:
        """Get embedding for a user given their feature vector."""
        x = user_features_vec.reshape(1, -1).astype(np.float32)
        return self.user_tower.forward(x)[0]

    def recommend(
        self,
        user_id       : str,
        user_features : np.ndarray,   # [n_users x user_dim]
        top_k         : int = 10,
        seen_items    : Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        u = self.user_index.get(user_id)
        if u is None:
            return []
        u_emb  = self.get_user_embedding(user_features[u])  # [embed_dim]
        scores = self.item_embeddings @ u_emb                # [n_items] cosine sims

        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            vid = self.idx_to_item[int(idx)]
            if not seen_items or vid not in seen_items:
                results.append((vid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def recommend_cold_start(
        self,
        user_feature_vec: np.ndarray,   # [user_dim] — any user vector, no ID needed
        top_k           : int = 10,
    ) -> list[tuple[str, float]]:
        """
        Recommend for a cold-start user given ONLY their feature vector.
        No interaction history required — this is the key Two-Tower advantage.
        """
        u_emb  = self.get_user_embedding(user_feature_vec)
        scores = self.item_embeddings @ u_emb

        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.idx_to_item[int(i)], float(scores[i])) for i in ranked]

    def evaluate(
        self,
        test_df      : pd.DataFrame,
        user_features: np.ndarray,
        k_values     : list[int] = [5, 10, 20],
        max_users    : int = 300,
        seed         : int = 42,
    ) -> dict[str, float]:
        from src.algorithms.matrix_factorization import (
            ndcg_at_k, precision_at_k, recall_at_k, hit_rate_at_k
        )
        rng = np.random.default_rng(seed)
        test_users = test_df["user_id"].unique()
        if len(test_users) > max_users:
            test_users = rng.choice(test_users, size=max_users, replace=False)

        results: dict[str, list[float]] = {
            f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg", "hr"]
        }
        for user_id in test_users:
            if user_id not in self.user_index:
                continue
            user_test = test_df[test_df["user_id"] == user_id]
            relevant  = set(user_test[user_test["watch_pct"] > 0.5]["video_id"].tolist())
            if not relevant:
                continue
            recs    = self.recommend(user_id, user_features, top_k=max(k_values))
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
    print("  TWO-TOWER NEURAL RECOMMENDER")
    print("=" * 60)

    from src.algorithms.matrix_factorization import load_data, train_test_split

    # Load everything
    matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item = load_data()
    _, test_df = train_test_split()

    user_features = np.load(DATA_DIR / "user_features.npy")   # [4500 x 32]
    item_features = np.load(DATA_DIR / "item_features.npy")   # [ 865 x 15]
    idx_to_item_list = [idx_to_item[i] for i in range(len(idx_to_item))]

    print(f"\nUser features: {user_features.shape}")
    print(f"Item features:  {item_features.shape}")

    # Load a sample of BPR triplets for training
    print("\n[TwoTower] Loading BPR triplets...")
    triplets = pd.read_csv(DATA_DIR / "pairwise_triplets.csv")
    sample   = triplets.sample(n=min(100_000, len(triplets)), random_state=42).reset_index(drop=True)
    print(f"  Using {len(sample):,} triplets")

    # Train Two-Tower
    model = TwoTowerModel(
        user_dim=32, item_dim=15, hidden_dim=64, embed_dim=32,
        n_epochs=15, lr=5e-4, weight_decay=1e-4, batch_size=512,
    )
    model.fit(sample, user_features, item_features, user_idx, item_idx, idx_to_item_list)

    # Evaluate
    print("\n[TwoTower] Evaluating...")
    metrics = model.evaluate(test_df, user_features, k_values=[5, 10, 20], max_users=300)
    print("\nTwo-Tower Results:")
    for k in [5, 10, 20]:
        print(f"  @{k:<3}: P={metrics[f'precision@{k}']:.4f} "
              f"R={metrics[f'recall@{k}']:.4f} "
              f"NDCG={metrics[f'ndcg@{k}']:.4f} "
              f"HR={metrics[f'hr@{k}']:.4f}")

    # Cold-start demo: recommend for a brand new user using ONLY their feature vector
    print("\n[Cold-Start Demo] Recommending for a cold-start user (feature vector only)...")
    # Simulate a new user: high credibility_sensitivity, high manipulation_aversion, ML topic
    new_user_vec = user_features[0].copy()  # use first user as proxy
    cold_recs = model.recommend_cold_start(new_user_vec, top_k=5)
    print("  Top-5 cold-start recommendations:")
    for vid, score in cold_recs:
        print(f"    {vid[:16]}...  score={score:.4f}")

    # Save
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    pd.DataFrame([{"model": "TwoTower", **metrics}]).to_csv(
        out / "two_tower_evaluation.csv", index=False
    )
    print(f"\n[Saved] -> experiments/two_tower_evaluation.csv")
