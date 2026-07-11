"""
src/algorithms/matrix_factorization.py
=======================================
Matrix Factorization implementations for the Agent-as-Shield RS.

Three models, one common interface, all supporting the Shield project's
agent-centric evaluation (penalizes clickbait, rewards credibility).

+------------------+----------------------------------------------+----------------------------------+
| Model            | Algorithm                                    | Best for                         |
+------------------+----------------------------------------------+----------------------------------+
| TruncatedSVD     | scipy SVD on watch_pct matrix                | Fast baseline, initial EDA       |
| ALS              | Hu et al. 2008 implicit feedback MF          | Main CF experiments (no ratings) |
| BiasSVD          | Funk SVD with SGD + user/item biases         | When explicit ratings available  |
+------------------+----------------------------------------------+----------------------------------+

All models expose:
    .fit(matrix, user_index, item_index, item_metadata)
    .predict(user_id, video_id)  -> float score
    .recommend(user_id, top_k)   -> list[(video_id, score)]
    .evaluate(test_df, k_values) -> dict of metrics
    .get_user_embedding(user_id) -> np.ndarray
    .get_item_embedding(video_id)-> np.ndarray

Evaluation metrics (leave-one-out split):
    - Precision@K, Recall@K, NDCG@K, HR@K (Hit Rate)

Usage:
    from src.algorithms.matrix_factorization import ALSMatrixFactorization, load_data

    matrix, user_index, item_index, item_meta = load_data()
    model = ALSMatrixFactorization(n_factors=64, alpha=40, regularization=0.01)
    model.fit(matrix, user_index, item_index, item_meta)

    recs = model.recommend("some-user-uuid", top_k=10)
    for video_id, score in recs:
        print(video_id, score)

    metrics = model.evaluate(test_df, k_values=[5, 10, 20])
    print(metrics)

Run standalone:
    python -m src.algorithms.matrix_factorization
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from scipy.sparse.linalg import svds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_data(data_dir: Path = DATA_DIR) -> tuple:
    """
    Load all generated data for matrix factorization.

    Returns:
        matrix      : scipy csr_matrix [n_users x n_items]
        user_index  : dict {user_id -> row_index}
        item_index  : dict {video_id -> col_index}
        item_meta   : pd.DataFrame with video metadata + stub agent scores
    """
    matrix    = load_npz(data_dir / "user_item_matrix.npz")
    user_idx  = json.load(open(data_dir / "user_index.json"))
    item_idx  = json.load(open(data_dir / "item_index.json"))
    item_meta = pd.read_csv(data_dir / "item_catalog.csv")

    # Build reverse indices for fast lookup
    idx_to_user = {v: k for k, v in user_idx.items()}
    idx_to_item = {v: k for k, v in item_idx.items()}

    print(f"[load_data] Matrix: {matrix.shape} | nnz: {matrix.nnz:,} | "
          f"sparsity: {1 - matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.4f}")
    return matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item


def train_test_split(
    interactions_csv: Path = DATA_DIR / "interactions.csv",
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train/test using random holdout.
    Each user's test set = random test_fraction of their interactions.
    Ensures every user in test also appears in train (no cold-start leakage).
    """
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(interactions_csv)

    train_rows, test_rows = [], []
    for user_id, group in df.groupby("user_id"):
        if len(group) < 3:
            # Too few interactions — all go to train
            train_rows.append(group)
            continue
        n_test = max(1, int(len(group) * test_fraction))
        test_idx = rng.choice(group.index, size=n_test, replace=False)
        test_rows.append(group.loc[test_idx])
        train_rows.append(group.drop(index=test_idx))

    train = pd.concat(train_rows).reset_index(drop=True)
    test  = pd.concat(test_rows).reset_index(drop=True)
    print(f"[split] Train: {len(train):,} | Test: {len(test):,}")
    return train, test


def build_matrix_from_df(
    df: pd.DataFrame,
    user_index: dict[str, int],
    item_index: dict[str, int],
) -> csr_matrix:
    """Build a sparse user-item matrix (watch_pct values) from a DataFrame."""
    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        uid = user_index.get(row["user_id"])
        vid = item_index.get(row["video_id"])
        if uid is not None and vid is not None:
            rows.append(uid)
            cols.append(vid)
            data.append(float(row["watch_pct"]))
    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_index), len(item_index)),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    # Ideal DCG: all k relevant items at the top
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseMatrixFactorization(ABC):
    """
    Abstract base class for all MF models.
    Subclasses must implement: _fit_internal, predict_score.
    """

    def __init__(self, n_factors: int = 64):
        self.n_factors   = n_factors
        self.user_index  : dict[str, int] = {}
        self.item_index  : dict[str, int] = {}
        self.idx_to_user : dict[int, str] = {}
        self.idx_to_item : dict[int, str] = {}
        self.item_meta   : Optional[pd.DataFrame] = None
        self.is_fitted   : bool = False

        # Embedding matrices (set by subclass)
        self.user_factors: Optional[np.ndarray] = None   # [n_users x n_factors]
        self.item_factors: Optional[np.ndarray] = None   # [n_items x n_factors]

    def fit(
        self,
        matrix    : csr_matrix,
        user_index: dict[str, int],
        item_index: dict[str, int],
        item_meta : pd.DataFrame,
        idx_to_user: dict[int, str],
        idx_to_item: dict[int, str],
    ) -> "BaseMatrixFactorization":
        """Train the model on the user-item matrix."""
        self.user_index  = user_index
        self.item_index  = item_index
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item
        self.item_meta   = item_meta.set_index("video_id") if item_meta is not None else None
        print(f"[{self.__class__.__name__}] Fitting on {matrix.shape} matrix...")
        t0 = time.time()
        self._fit_internal(matrix)
        elapsed = time.time() - t0
        self.is_fitted = True
        print(f"[{self.__class__.__name__}] Fit complete in {elapsed:.2f}s")
        return self

    @abstractmethod
    def _fit_internal(self, matrix: csr_matrix) -> None:
        """Implement this in each subclass."""
        ...

    @abstractmethod
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Return the raw predicted score for (user_idx, item_idx)."""
        ...

    def predict(self, user_id: str, video_id: str) -> float:
        """Predict score for a (user_id, video_id) string pair."""
        u = self.user_index.get(user_id)
        v = self.item_index.get(video_id)
        if u is None or v is None:
            return 0.0
        return self.predict_score(u, v)

    def recommend(
        self,
        user_id    : str,
        top_k      : int = 10,
        exclude_seen: bool = True,
        seen_items : Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Return top_k recommendations for a user as [(video_id, score)].
        Optionally filters out already-seen items.
        """
        u = self.user_index.get(user_id)
        if u is None:
            return []

        # Score all items at once using embedding dot product
        user_vec = self.user_factors[u]   # [n_factors]
        scores   = self.item_factors @ user_vec  # [n_items]

        # Add biases if available
        if hasattr(self, "item_bias") and self.item_bias is not None:
            scores += self.item_bias
        if hasattr(self, "user_bias") and self.user_bias is not None:
            scores += self.user_bias[u]
        if hasattr(self, "global_mean") and self.global_mean is not None:
            scores += self.global_mean

        # Sort descending
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            vid = self.idx_to_item.get(int(idx))
            if vid is None:
                continue
            if exclude_seen and seen_items and vid in seen_items:
                continue
            results.append((vid, float(scores[idx])))
            if len(results) >= top_k:
                break

        return results

    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Return the learned latent factor vector for a user."""
        u = self.user_index.get(user_id)
        if u is None or self.user_factors is None:
            return None
        return self.user_factors[u]

    def get_item_embedding(self, video_id: str) -> Optional[np.ndarray]:
        """Return the learned latent factor vector for an item."""
        v = self.item_index.get(video_id)
        if v is None or self.item_factors is None:
            return None
        return self.item_factors[v]

    def evaluate(
        self,
        test_df  : pd.DataFrame,
        k_values : list[int] = [5, 10, 20],
        max_users: int = 500,
        seed     : int = 42,
    ) -> dict[str, float]:
        """
        Evaluate on held-out test interactions.

        For each user in test_df:
          - Their test interactions are the 'relevant' set
          - We recommend top_k from items NOT seen in test
          - Compute Precision@K, Recall@K, NDCG@K, HR@K

        Returns dict like:
          {'precision@5': 0.12, 'recall@5': 0.08, 'ndcg@5': 0.14, 'hr@5': 0.45, ...}
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling evaluate()")

        results: dict[str, list[float]] = {
            f"{metric}@{k}": []
            for k in k_values
            for metric in ["precision", "recall", "ndcg", "hr"]
        }

        rng = np.random.default_rng(seed)
        test_users = test_df["user_id"].unique()
        if len(test_users) > max_users:
            test_users = rng.choice(test_users, size=max_users, replace=False)

        for user_id in test_users:
            if user_id not in self.user_index:
                continue

            user_test  = test_df[test_df["user_id"] == user_id]
            # Relevant = items with watch_pct > 0.5 (considered as positive interactions)
            relevant   = set(user_test[user_test["watch_pct"] > 0.5]["video_id"].tolist())
            if not relevant:
                continue

            recs = self.recommend(user_id, top_k=max(k_values))
            rec_ids = [vid for vid, _ in recs]

            for k in k_values:
                results[f"precision@{k}"].append(precision_at_k(rec_ids, relevant, k))
                results[f"recall@{k}"].append(recall_at_k(rec_ids, relevant, k))
                results[f"ndcg@{k}"].append(ndcg_at_k(rec_ids, relevant, k))
                results[f"hr@{k}"].append(hit_rate_at_k(rec_ids, relevant, k))

        # Average across users
        return {key: float(np.mean(vals)) if vals else 0.0 for key, vals in results.items()}


# ---------------------------------------------------------------------------
# Model 1: Truncated SVD (fast baseline)
# ---------------------------------------------------------------------------

class TruncatedSVDMF(BaseMatrixFactorization):
    """
    Truncated SVD Matrix Factorization.

    Decomposes the watch_pct matrix: R ~ U * S * V^T
    Users:  user_factors = U * sqrt(S)    [n_users x n_factors]
    Items:  item_factors = V * sqrt(S)    [n_items x n_factors]
    Score:  user_factors[u] @ item_factors[v]

    Pros:  Extremely fast, no hyperparameters to tune
    Cons:  Treats all zeros as "zero preference" (not missing), suboptimal for implicit data
    Best:  Baseline, quick EDA, understanding latent structure
    """

    def __init__(self, n_factors: int = 64):
        super().__init__(n_factors)

    def _fit_internal(self, matrix: csr_matrix) -> None:
        # scipy svds returns singular vectors, NOT full SVD
        # k must be < min(matrix.shape) - 1
        k = min(self.n_factors, min(matrix.shape) - 1)
        U, sigma, Vt = svds(matrix.astype(np.float32), k=k)

        # Sort by singular value descending (svds returns ascending)
        order = np.argsort(sigma)[::-1]
        U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

        sqrt_sigma = np.sqrt(np.maximum(sigma, 0))
        self.user_factors = U  * sqrt_sigma[np.newaxis, :]   # [n_users x k]
        self.item_factors = Vt.T * sqrt_sigma[np.newaxis, :] # [n_items x k]
        self.singular_values = sigma

        print(f"  Explained variance (top factor): {sigma[0]:.4f} | "
              f"Top-{k} sum: {sigma.sum():.4f}")

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])


# ---------------------------------------------------------------------------
# Model 2: ALS — Alternating Least Squares (implicit feedback)
# ---------------------------------------------------------------------------

class ALSMatrixFactorization(BaseMatrixFactorization):
    """
    Alternating Least Squares for Implicit Feedback (Hu, Koren, Volinsky 2008).

    Key insight: treat watch_pct as CONFIDENCE, not rating.
    - Preference p_ui = 1 for any interaction (binary: watched or not)
    - Confidence c_ui = 1 + alpha * watch_pct
      → Higher watch_pct = more confident this user prefers this item

    ALS update (user u):
        x_u = (Y^T C^u Y + lambda*I)^-1 Y^T C^u p_u

    ALS update (item i):
        y_i = (X^T C^i X + lambda*I)^-1 X^T C^i p_i

    Pros:  Correctly handles implicit data, state-of-the-art for watch_pct
    Cons:  O(f^2 * n_items) per user per iteration, slower than SVD
    Best:  PRIMARY collaborative filtering model for this project
    """

    def __init__(
        self,
        n_factors     : int   = 64,
        n_iterations  : int   = 15,
        alpha         : float = 40.0,    # confidence scaling factor
        regularization: float = 0.01,    # L2 regularization lambda
        seed          : int   = 42,
    ):
        super().__init__(n_factors)
        self.n_iterations   = n_iterations
        self.alpha          = alpha
        self.regularization = regularization
        self.seed           = seed
        self.training_loss  : list[float] = []

    def _fit_internal(self, matrix: csr_matrix) -> None:
        rng = np.random.default_rng(self.seed)
        n_users, n_items = matrix.shape

        # Initialize factors with small random values
        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors)).astype(np.float32)

        # Confidence matrix: C = 1 + alpha * R (where R = watch_pct)
        # We keep the sparse structure: only store non-zero confidences
        # Zeros in the matrix → c_ui = 1.0 (handled implicitly in ALS update)
        C = matrix.copy().astype(np.float32)
        C.data = 1.0 + self.alpha * C.data  # c_ui for observed interactions

        # Transpose for item updates
        C_T = C.T.tocsr()

        lambda_I = self.regularization * np.eye(self.n_factors, dtype=np.float32)

        for iteration in range(self.n_iterations):
            t_iter = time.time()

            # --- Update user factors ---
            # YtY = Y^T Y (item-item gram matrix)
            YtY = self.item_factors.T @ self.item_factors  # [f x f]

            for u in range(n_users):
                # Get items that user u interacted with
                user_row = C[u]   # sparse row of confidences for user u
                if user_row.nnz == 0:
                    continue

                item_indices  = user_row.indices
                confidences_u = user_row.data  # c_ui for observed items

                # p_u = 1 for all observed items (binary preference)
                # C^u - I = diag(c_ui - 1) for observed items only
                # Y^T C^u Y = YtY + Y_obs^T * diag(c_ui - 1) * Y_obs
                Y_obs = self.item_factors[item_indices]          # [n_obs x f]
                Cu_minus_1 = confidences_u - 1.0                 # [n_obs]

                A = YtY + Y_obs.T @ (Cu_minus_1[:, None] * Y_obs) + lambda_I
                b = (confidences_u[:, None] * Y_obs).sum(axis=0) # Y^T C^u p_u

                try:
                    self.user_factors[u] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    self.user_factors[u] = np.linalg.lstsq(A, b, rcond=None)[0]

            # --- Update item factors ---
            XtX = self.user_factors.T @ self.user_factors  # [f x f]

            for v in range(n_items):
                item_col = C_T[v]
                if item_col.nnz == 0:
                    continue

                user_indices  = item_col.indices
                confidences_v = item_col.data

                X_obs = self.user_factors[user_indices]
                Cv_minus_1 = confidences_v - 1.0

                A = XtX + X_obs.T @ (Cv_minus_1[:, None] * X_obs) + lambda_I
                b = (confidences_v[:, None] * X_obs).sum(axis=0)

                try:
                    self.item_factors[v] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    self.item_factors[v] = np.linalg.lstsq(A, b, rcond=None)[0]

            elapsed = time.time() - t_iter
            loss = self._compute_loss(matrix, C, lambda_I)
            self.training_loss.append(loss)
            print(f"  Iter {iteration+1:2d}/{self.n_iterations} | "
                  f"Loss: {loss:.4f} | Time: {elapsed:.1f}s")

    def _compute_loss(
        self, R: csr_matrix, C: csr_matrix, lambda_I: np.ndarray
    ) -> float:
        """
        Approximate training loss (only over observed interactions for speed).
        Full loss = Σ c_ui (p_ui - x_u^T y_i)^2 + lambda(||X||^2 + ||Y||^2)
        """
        loss = 0.0
        sample_size = min(5000, R.nnz)
        nnz_indices = np.random.choice(R.nnz, size=sample_size, replace=False)

        rows, cols = R.nonzero()
        for idx in nnz_indices:
            u, v = rows[idx], cols[idx]
            pred  = float(self.user_factors[u] @ self.item_factors[v])
            c_uv  = 1.0 + self.alpha * float(R[u, v])
            loss += c_uv * (1.0 - pred) ** 2

        # Regularization term
        reg = self.regularization * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
        )
        return (loss / sample_size) + (reg / (R.shape[0] * R.shape[1]))

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])


# ---------------------------------------------------------------------------
# Model 3: BiasSVD — Funk SVD with user/item biases (SGD)
# ---------------------------------------------------------------------------

class BiasSVD(BaseMatrixFactorization):
    """
    Funk SVD with user bias, item bias, and global mean.
    Uses SGD over observed interactions.

    Score(u, i) = mu + b_u + b_i + P_u^T Q_i

    Where:
        mu  = global mean rating
        b_u = user bias (user's tendency to rate high/low)
        b_i = item bias (item's general quality/appeal)
        P_u = user latent factors
        Q_i = item latent factors

    Uses watch_pct as the "rating" signal, falls back to explicit_rating
    when available (via the interactions DataFrame).

    Pros:  Captures systematic biases, excellent with explicit ratings
    Cons:  Slower convergence than ALS, needs careful LR tuning
    Best:  When you want to explain WHY a score is high (bias decomposition)
    """

    def __init__(
        self,
        n_factors     : int   = 64,
        n_epochs      : int   = 20,
        learning_rate : float = 0.005,
        regularization: float = 0.02,
        seed          : int   = 42,
    ):
        super().__init__(n_factors)
        self.n_epochs      = n_epochs
        self.lr            = learning_rate
        self.regularization= regularization
        self.seed          = seed
        self.global_mean   : float = 0.0
        self.user_bias     : Optional[np.ndarray] = None
        self.item_bias     : Optional[np.ndarray] = None
        self.training_loss : list[float] = []

    def _fit_internal(self, matrix: csr_matrix) -> None:
        rng = np.random.default_rng(self.seed)
        n_users, n_items = matrix.shape

        # Global mean from observed values
        self.global_mean = float(matrix.data.mean())

        # Initialize factors and biases
        scale = 0.01
        self.user_factors = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.item_factors = rng.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)
        self.user_bias    = np.zeros(n_users, dtype=np.float32)
        self.item_bias    = np.zeros(n_items, dtype=np.float32)

        rows, cols = matrix.nonzero()
        ratings = matrix.data.astype(np.float32)
        n_obs   = len(ratings)

        for epoch in range(self.n_epochs):
            t_epoch = time.time()
            epoch_loss = 0.0

            # Shuffle training pairs each epoch
            perm = rng.permutation(n_obs)

            for idx in perm:
                u = rows[idx]
                v = cols[idx]
                r = ratings[idx]

                # Predict
                pred = (self.global_mean
                        + self.user_bias[u]
                        + self.item_bias[v]
                        + float(self.user_factors[u] @ self.item_factors[v]))

                err = r - pred
                epoch_loss += err ** 2

                # SGD updates
                # Biases
                self.user_bias[u] += self.lr * (err - self.regularization * self.user_bias[u])
                self.item_bias[v] += self.lr * (err - self.regularization * self.item_bias[v])

                # Factors (standard SGD update)
                pu_old = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (
                    err * self.item_factors[v] - self.regularization * self.user_factors[u]
                )
                self.item_factors[v] += self.lr * (
                    err * pu_old - self.regularization * self.item_factors[v]
                )

            rmse = np.sqrt(epoch_loss / n_obs)
            self.training_loss.append(rmse)
            elapsed = time.time() - t_epoch
            print(f"  Epoch {epoch+1:2d}/{self.n_epochs} | RMSE: {rmse:.4f} | Time: {elapsed:.1f}s")

    def predict_score(self, user_idx: int, item_idx: int) -> float:
        score = (self.global_mean
                 + self.user_bias[user_idx]
                 + self.item_bias[item_idx]
                 + float(self.user_factors[user_idx] @ self.item_factors[item_idx]))
        return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Agent-centric re-ranking (post-processing)
# ---------------------------------------------------------------------------

def agent_rerank(
    recommendations : list[tuple[str, float]],
    item_meta       : pd.DataFrame,
    credibility_w   : float = 0.3,
    manipulation_w  : float = 0.3,
    info_density_w  : float = 0.2,
    cf_score_w      : float = 0.2,
) -> list[tuple[str, float]]:
    """
    Re-rank MF recommendations using agent-centric stub scores.

    Final score = cf_score_w * cf_score
                + info_density_w * info_density_stub
                + credibility_w  * credibility_stub
                - manipulation_w * clickbait_score_stub

    This is the KEY integration point between your MF output and the
    Agent-as-Shield scoring function. When Module 1 & 2 replace the stubs
    with real scores, this function needs no changes.

    Args:
        recommendations : output of model.recommend() — [(video_id, cf_score)]
        item_meta       : item_catalog.csv loaded as DataFrame, indexed by video_id

    Returns:
        Re-ranked list [(video_id, agent_score)]
    """
    meta_idx = item_meta.set_index("video_id") if "video_id" in item_meta.columns else item_meta
    # Ensure no duplicate index (drop dupes keeping first)
    if meta_idx.index.duplicated().any():
        meta_idx = meta_idx[~meta_idx.index.duplicated(keep="first")]

    reranked = []
    for vid, cf_score in recommendations:
        if vid not in meta_idx.index:
            agent_score = cf_score * cf_score_w
        else:
            row = meta_idx.loc[vid]
            # loc returns a DataFrame if index has duplicates — take first row
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            info_density = float(row["info_density_stub"])  if not pd.isna(row.get("info_density_stub")) else 0.5
            credibility  = float(row["credibility_stub"])   if not pd.isna(row.get("credibility_stub"))   else 0.5
            clickbait    = float(row["clickbait_score_stub"]) if not pd.isna(row.get("clickbait_score_stub")) else 0.5

            agent_score = (
                cf_score_w   * cf_score
                + info_density_w * info_density
                + credibility_w  * credibility
                - manipulation_w * clickbait
            )

        reranked.append((vid, float(agent_score)))

    # Re-sort by agent_score descending
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    models     : dict[str, BaseMatrixFactorization],
    test_df    : pd.DataFrame,
    k_values   : list[int] = [5, 10, 20],
    max_users  : int = 300,
) -> pd.DataFrame:
    """
    Evaluate multiple models and return a comparison DataFrame.

    Args:
        models  : dict of {model_name: fitted_model}
        test_df : held-out test interactions

    Returns:
        DataFrame with models as rows, metrics as columns
    """
    rows = []
    for name, model in models.items():
        print(f"\n[Evaluation] {name}...")
        metrics = model.evaluate(test_df, k_values=k_values, max_users=max_users)
        rows.append({"model": name, **metrics})

    df = pd.DataFrame(rows).set_index("model")
    return df.round(4)


# ---------------------------------------------------------------------------
# Main: train all 3 models, evaluate, print comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  AGENT-AS-SHIELD: Matrix Factorization Training & Evaluation")
    print("=" * 65)

    # Load data
    print("\n[Step 1] Loading data...")
    matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item = load_data()

    # Train/test split
    print("\n[Step 2] Creating train/test split...")
    train_df, test_df = train_test_split()
    train_matrix = build_matrix_from_df(train_df, user_idx, item_idx)

    # ---------------------------------------------------------------
    # Model 1: TruncatedSVD (fast baseline)
    # ---------------------------------------------------------------
    print("\n[Step 3a] Training TruncatedSVD (baseline)...")
    svd_model = TruncatedSVDMF(n_factors=64)
    svd_model.fit(train_matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item)

    # ---------------------------------------------------------------
    # Model 2: ALS (main CF model)
    # ---------------------------------------------------------------
    print("\n[Step 3b] Training ALS (implicit feedback)...")
    als_model = ALSMatrixFactorization(
        n_factors=64, n_iterations=10, alpha=40.0, regularization=0.01
    )
    als_model.fit(train_matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item)

    # ---------------------------------------------------------------
    # Model 3: BiasSVD (with explicit-rating-compatible SGD)
    # ---------------------------------------------------------------
    print("\n[Step 3c] Training BiasSVD (Funk SVD with biases)...")
    bias_model = BiasSVD(
        n_factors=64, n_epochs=15, learning_rate=0.005, regularization=0.02
    )
    bias_model.fit(train_matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item)

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    print("\n[Step 4] Evaluating all models...")
    comparison = run_comparison(
        models={
            "TruncatedSVD" : svd_model,
            "ALS"          : als_model,
            "BiasSVD"      : bias_model,
        },
        test_df   = test_df,
        k_values  = [5, 10, 20],
        max_users = 300,
    )
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    print(comparison.to_string())

    # ---------------------------------------------------------------
    # Demo: ALS recommendations with agent re-ranking
    # ---------------------------------------------------------------
    print("\n[Step 5] Demo: Agent-centric re-ranking on ALS recommendations")
    sample_user = list(user_idx.keys())[50]  # pick a non-cold-start user
    raw_recs    = als_model.recommend(sample_user, top_k=10)
    agent_recs  = agent_rerank(raw_recs, item_meta)

    print(f"\nUser: {sample_user[:16]}...")
    print(f"\n{'Rank':<5} {'CF Score':>10} {'Agent Score':>12}  Video ID")
    for i, ((vid_cf, cf_sc), (vid_ag, ag_sc)) in enumerate(zip(raw_recs, agent_recs)):
        marker = "<-- reranked" if vid_cf != vid_ag else ""
        print(f"  {i+1:<3} {cf_sc:>10.4f} {ag_sc:>12.4f}  {vid_ag[:16]}... {marker}")

    # Save evaluation results
    results_path = Path("experiments") / "mf_evaluation.csv"
    results_path.parent.mkdir(exist_ok=True)
    comparison.to_csv(results_path)
    print(f"\n[Saved] Evaluation results -> {results_path}")
    print("\nDone.")
