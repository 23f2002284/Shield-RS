"""
src/algorithms/sequential_rs.py
================================
Sequential Recommender Systems — models that treat watch history as an
ordered sequence and predict the NEXT item a user will watch.

Standard RS (MF, CF, BPR) treat interactions as a bag of items — order
is ignored. Sequential RS captures:
  - "Users who watched A, then B, tend to watch C next"
  - Within-session momentum: recent watches matter more than old ones
  - Topic progression patterns: beginner → intermediate → advanced

Two models implemented (no PyTorch required):

1. MarkovChainRS (baseline):
   Models P(next_item | last_item) using first-order transition counts.
   score(i | last=j) = count(j → i) / count(j → *)
   Fast, interpretable, strong for stationary patterns.

2. FPMC (Factorized Personalized Markov Chains, Rendle et al. 2010):
   Combines MF (long-term user preference) with MC (short-term transitions).
   score(u, i | last=j) = P_u · Q_i  +  L_j · M_i
   Where:
     P_u, Q_i = user/item latent factors (long-term: WHO the user is)
     L_j, M_i = transition latent factors (short-term: WHAT they just watched)
   Trained with BPR loss on consecutive (u, j→i) triples from sessions.

Both models expose:
    .fit(sessions)
    .recommend(user_id, last_video_id, top_k) -> list[(video_id, score)]
    .evaluate(sessions_test, k_values)         -> dict of metrics

Evaluation approach (leave-last-out):
    For each session with ≥ 3 videos:
        train = first (n-1) videos
        test  = last video  ← can we predict it?

Usage:
    python -m src.algorithms.sequential_rs
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Session loading utilities
# ---------------------------------------------------------------------------

def load_sessions(
    path: Path = DATA_DIR / "sessions.json",
    min_length: int = 2,
) -> list[dict]:
    """Load sessions from sessions.json."""
    sessions = json.load(open(path, encoding="utf-8"))
    sessions = [s for s in sessions if len(s.get("video_sequence", [])) >= min_length]
    print(f"[Sessions] Loaded {len(sessions):,} sessions with >={min_length} items")
    return sessions


def load_user_sequences(
    interactions_path: Path = DATA_DIR / "interactions.csv",
    min_length: int = 5,
) -> list[dict]:
    """
    Build per-user ordered watch sequences from interactions.csv.
    Each 'session' = all of a user's interactions sorted by timestamp.
    This gives rich sequential data (50-200 items per user) vs. the
    sparse session.json which groups by time window (98% are length=1).

    This is the recommended input for sequential RS on this dataset.
    """
    print("[UserSequences] Loading interactions.csv...")
    df = pd.read_csv(interactions_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])

    sequences = []
    for user_id, group in df.groupby("user_id"):
        seq = group["video_id"].tolist()
        if len(seq) >= min_length:
            sequences.append({
                "user_id"       : user_id,
                "video_sequence": seq,
                "length"        : len(seq),
            })

    print(f"[UserSequences] {len(sequences):,} users with >={min_length} interactions")
    print(f"  Avg sequence length: {sum(s['length'] for s in sequences)/len(sequences):.1f}")
    return sequences


def split_sessions(sessions: list[dict], test_fraction: float = 0.2, seed: int = 42) -> tuple:
    """
    Train/test split for sequential RS.

    For user-level sequences (from load_user_sequences):
        Each user sequence is split temporally:
        - train = first (1 - test_fraction) of the sequence
        - test  = a set of (context, target) pairs from the last test_fraction items
                  Each test pair: context = all items before position p, target = item at p

    For session-level data (from load_sessions):
        Hold out test_fraction of sessions per user.
    """
    rng = np.random.default_rng(seed)

    # Detect if these are full user sequences (one per user, long) or short sessions
    avg_len = np.mean([len(s.get("video_sequence", [])) for s in sessions])
    is_user_sequences = avg_len > 10  # user sequences are long; session sequences are short

    if is_user_sequences:
        # Temporal split within each user's sequence
        train_sessions, test_sessions = [], []
        for s in sessions:
            seq = s["video_sequence"]
            uid = s["user_id"]
            n   = len(seq)
            n_train = max(int(n * (1 - test_fraction)), 5)

            # Train: use full sequence (model sees it all during fit)
            train_sessions.append({"user_id": uid, "video_sequence": seq[:n_train]})

            # Test: for each position from n_train onward, context = seq[:p], target = seq[p]
            for p in range(n_train, min(n, n_train + 10)):  # max 10 test pairs per user
                if p >= 2:
                    test_sessions.append({
                        "user_id"       : uid,
                        "video_sequence": seq[:p+1],  # last item is target
                    })

        print(f"[Split] User sequences -> Train: {len(train_sessions):,} | "
              f"Test pairs: {len(test_sessions):,}")
        return train_sessions, test_sessions

    # Session-level split (original logic)
    by_user: dict[str, list] = defaultdict(list)
    for s in sessions:
        by_user[s["user_id"]].append(s)

    train_sessions, test_sessions = [], []
    for user_id, user_sessions in by_user.items():
        n_test = max(1, int(len(user_sessions) * test_fraction))
        indices = set(rng.choice(len(user_sessions), size=n_test, replace=False).tolist())
        for i, s in enumerate(user_sessions):
            if i in indices and len(s["video_sequence"]) >= 3:
                test_sessions.append(s)
            else:
                train_sessions.append(s)

    print(f"[Sessions] Train: {len(train_sessions):,} | Test: {len(test_sessions):,}")
    return train_sessions, test_sessions



def sequential_metrics(
    recommended: list[str], target: str, k_values: list[int]
) -> dict[str, float]:
    """Compute HR@K and MRR@K for a single user with one target item."""
    results = {}
    for k in k_values:
        top_k = recommended[:k]
        results[f"hr@{k}"]  = 1.0 if target in top_k else 0.0
        # MRR: 1/rank if in top-k, else 0
        try:
            rank = top_k.index(target) + 1
            results[f"mrr@{k}"] = 1.0 / rank
        except ValueError:
            results[f"mrr@{k}"] = 0.0
    return results


# ---------------------------------------------------------------------------
# Model 1: First-Order Markov Chain RS
# ---------------------------------------------------------------------------

class MarkovChainRS:
    """
    First-Order Markov Chain Recommender.

    Learns transition probabilities from consecutive (item_j → item_i) pairs.
    P(next=i | last=j) = count(j, i) / sum_k count(j, k)

    Personalization: weighted mix of Markov transitions + user history.
    score(u, i | last=j) = alpha * P(i|j) + (1-alpha) * user_item_freq(u, i)

    Pros:  No training needed, pure counting, fully interpretable
    Cons:  Only captures first-order dependencies (no long-range patterns)
    Best:  Strong baseline for sequential recommendation
    """

    def __init__(self, alpha: float = 0.7, smoothing: float = 1e-6):
        """
        Args:
            alpha    : weight on Markov transitions vs. user history (0=pure history, 1=pure MC)
            smoothing: Laplace smoothing for zero-count transitions
        """
        self.alpha     = alpha
        self.smoothing = smoothing

        # transition_counts[j][i] = how often i follows j across all sessions
        self.transition_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # transition_sums[j] = total transitions from j (for normalization)
        self.transition_sums  : dict[str, float] = defaultdict(float)
        # user_item_freq[user_id][video_id] = how often user watched this video
        self.user_item_freq   : dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        self.all_items: list[str] = []
        self.is_fitted = False

    def fit(self, sessions: list[dict]) -> "MarkovChainRS":
        t0 = time.time()
        item_set: set[str] = set()

        for session in sessions:
            seq = session.get("video_sequence", [])
            uid = session["user_id"]

            # Count consecutive transitions
            for k in range(len(seq) - 1):
                j = seq[k]   # current item
                i = seq[k+1] # next item
                self.transition_counts[j][i] += 1.0
                self.transition_sums[j]       += 1.0
                item_set.add(j)
                item_set.add(i)

            # User-level item frequency
            for vid in seq:
                self.user_item_freq[uid][vid] += 1.0
                item_set.add(vid)

        self.all_items = sorted(item_set)
        self.is_fitted  = True
        print(f"[MarkovChainRS] Fitted on {len(sessions):,} sessions | "
              f"{len(self.all_items):,} unique items | {time.time()-t0:.2f}s")
        return self

    def recommend(
        self,
        user_id    : str,
        last_item  : str,
        top_k      : int = 10,
        seen_items : Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Return top_k next-item recommendations given the last watched item.
        score(i) = alpha * P(i | last) + (1-alpha) * normalized_user_freq(u, i)
        """
        # Markov scores: P(i | last_item)
        total_from_last = self.transition_sums.get(last_item, 0) + self.smoothing * len(self.all_items)
        mc_scores: dict[str, float] = {}
        for item in self.all_items:
            count = self.transition_counts.get(last_item, {}).get(item, 0.0)
            mc_scores[item] = (count + self.smoothing) / total_from_last

        # User history scores (normalized)
        user_counts = self.user_item_freq.get(user_id, {})
        max_count   = max(user_counts.values(), default=1.0)
        user_scores: dict[str, float] = {
            item: user_counts.get(item, 0.0) / max_count
            for item in self.all_items
        }

        # Blend
        scores = {
            item: self.alpha * mc_scores.get(item, 0.0) + (1-self.alpha) * user_scores.get(item, 0.0)
            for item in self.all_items
        }

        # Filter seen
        if seen_items:
            scores = {k: v for k, v in scores.items() if k not in seen_items}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def evaluate(
        self,
        test_sessions: list[dict],
        k_values     : list[int] = [5, 10, 20],
        max_sessions : int = 1000,
        seed         : int = 42,
    ) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        if len(test_sessions) > max_sessions:
            indices = rng.choice(len(test_sessions), size=max_sessions, replace=False)
            test_sessions = [test_sessions[i] for i in indices]

        metric_lists: dict[str, list[float]] = defaultdict(list)

        for session in test_sessions:
            seq = session.get("video_sequence", [])
            if len(seq) < 3:
                continue
            uid       = session["user_id"]
            last_seen = seq[-2]    # second-to-last
            target    = seq[-1]    # last item = ground truth
            seen      = set(seq[:-1])

            recs = self.recommend(uid, last_seen, top_k=max(k_values), seen_items=seen)
            rec_ids = [v for v, _ in recs]
            m = sequential_metrics(rec_ids, target, k_values)
            for k, v in m.items():
                metric_lists[k].append(v)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in metric_lists.items()}


# ---------------------------------------------------------------------------
# Model 2: FPMC — Factorized Personalized Markov Chains
# ---------------------------------------------------------------------------

class FPMCModel:
    """
    Factorized Personalized Markov Chains (Rendle et al. 2010).

    Decomposes the next-item score as:
        score(u, i | last=j) = P_u · Q_i  +  L_j · M_i

    P_u  [n_users x k] : user preference factors (long-term)
    Q_i  [n_items x k] : item preference factors
    L_j  [n_items x k] : item-as-context factors (what was just watched)
    M_i  [n_items x k] : item-as-target factors

    Training: BPR loss over (user, last_item, pos_next, neg_next) tuples.
    For each consecutive (j → i) pair in a session, sample a random neg item.

    Pros:  Captures both personal taste (P_u · Q_i) and local context (L_j · M_i)
    Cons:  More parameters than Markov, needs careful tuning
    Best:  When you want personalized next-item prediction
    """

    def __init__(
        self,
        n_factors     : int   = 32,
        n_epochs      : int   = 10,
        learning_rate : float = 0.01,
        regularization: float = 0.01,
        batch_size    : int   = 2048,
        seed          : int   = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr        = learning_rate
        self.reg       = regularization
        self.batch_size= batch_size
        self.seed      = seed

        self.user_index  : dict[str, int] = {}
        self.item_index  : dict[str, int] = {}
        self.idx_to_item : list[str]      = []

        # Factor matrices
        self.P : Optional[np.ndarray] = None  # [n_users x k] user factors
        self.Q : Optional[np.ndarray] = None  # [n_items x k] item-as-target factors
        self.L : Optional[np.ndarray] = None  # [n_items x k] item-as-context factors
        self.M : Optional[np.ndarray] = None  # [n_items x k] item-as-MC-target factors

        self.training_auc: list[float] = []
        self.is_fitted = False

    def fit(self, sessions: list[dict]) -> "FPMCModel":
        t0  = time.time()
        rng = np.random.default_rng(self.seed)

        # Build indices from all sessions
        users  = sorted({s["user_id"] for s in sessions})
        items  = sorted({v for s in sessions for v in s.get("video_sequence", [])})
        self.user_index  = {u: i for i, u in enumerate(users)}
        self.item_index  = {v: i for i, v in enumerate(items)}
        self.idx_to_item = items

        n_users, n_items = len(users), len(items)
        k = self.n_factors
        scale = 0.01

        self.P = rng.normal(0, scale, (n_users, k)).astype(np.float32)
        self.Q = rng.normal(0, scale, (n_items, k)).astype(np.float32)
        self.L = rng.normal(0, scale, (n_items, k)).astype(np.float32)
        self.M = rng.normal(0, scale, (n_items, k)).astype(np.float32)

        # Build training triples: (user_idx, last_item_idx, next_item_idx)
        print(f"[FPMC] Building training triples from {len(sessions):,} sessions...")
        triples = []
        for s in sessions:
            seq = s.get("video_sequence", [])
            uid = self.user_index.get(s["user_id"])
            if uid is None:
                continue
            for k_idx in range(len(seq) - 1):
                j_idx = self.item_index.get(seq[k_idx])
                i_idx = self.item_index.get(seq[k_idx + 1])
                if j_idx is not None and i_idx is not None:
                    triples.append((uid, j_idx, i_idx))

        triples_arr = np.array(triples, dtype=np.int32)
        n_triples   = len(triples_arr)
        print(f"[FPMC] {n_triples:,} training triples | "
              f"{n_users} users | {n_items} items | {k} factors")

        for epoch in range(self.n_epochs):
            t_ep = time.time()
            perm  = rng.permutation(n_triples)
            u_arr = triples_arr[perm, 0]
            j_arr = triples_arr[perm, 1]
            i_arr = triples_arr[perm, 2]   # positive next item
            # Sample random negative items
            neg_arr = rng.integers(0, n_items, size=n_triples).astype(np.int32)

            epoch_correct = 0
            for start in range(0, n_triples, self.batch_size):
                end = min(start + self.batch_size, n_triples)
                u_b = u_arr[start:end]
                j_b = j_arr[start:end]
                i_b = i_arr[start:end]
                n_b = neg_arr[start:end]

                # Compute FPMC scores
                # score(u, i, j) = P_u · Q_i + L_j · M_i
                s_pos = (np.sum(self.P[u_b] * self.Q[i_b], axis=1) +
                         np.sum(self.L[j_b] * self.M[i_b], axis=1))
                s_neg = (np.sum(self.P[u_b] * self.Q[n_b], axis=1) +
                         np.sum(self.L[j_b] * self.M[n_b], axis=1))

                diff  = s_pos - s_neg
                sigma = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))
                grad  = (1.0 - sigma)   # gradient signal

                epoch_correct += (diff > 0).sum()

                # Update P_u
                dP = grad[:, None] * (self.Q[i_b] - self.Q[n_b]) - self.reg * self.P[u_b]
                # Update Q_i (pos)
                dQi = grad[:, None] * self.P[u_b] - self.reg * self.Q[i_b]
                # Update Q_n (neg)
                dQn = -grad[:, None] * self.P[u_b] - self.reg * self.Q[n_b]
                # Update L_j
                dLj = grad[:, None] * (self.M[i_b] - self.M[n_b]) - self.reg * self.L[j_b]
                # Update M_i (pos)
                dMi = grad[:, None] * self.L[j_b] - self.reg * self.M[i_b]
                # Update M_n (neg)
                dMn = -grad[:, None] * self.L[j_b] - self.reg * self.M[n_b]

                np.add.at(self.P, u_b,  self.lr * dP)
                np.add.at(self.Q, i_b,  self.lr * dQi)
                np.add.at(self.Q, n_b,  self.lr * dQn)
                np.add.at(self.L, j_b,  self.lr * dLj)
                np.add.at(self.M, i_b,  self.lr * dMi)
                np.add.at(self.M, n_b,  self.lr * dMn)

            auc = epoch_correct / n_triples
            self.training_auc.append(float(auc))
            print(f"  Epoch {epoch+1:2d}/{self.n_epochs} | "
                  f"Approx AUC: {auc:.4f} | Time: {time.time()-t_ep:.1f}s")

        self.is_fitted = True
        print(f"[FPMC] Training complete in {time.time()-t0:.1f}s")
        return self

    def score(self, user_id: str, item_id: str, last_item_id: str) -> float:
        u = self.user_index.get(user_id)
        i = self.item_index.get(item_id)
        j = self.item_index.get(last_item_id)
        if any(x is None for x in [u, i, j]):
            return 0.0
        return float(self.P[u] @ self.Q[i] + self.L[j] @ self.M[i])

    def recommend(
        self,
        user_id    : str,
        last_item  : str,
        top_k      : int = 10,
        seen_items : Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        u = self.user_index.get(user_id)
        j = self.item_index.get(last_item)
        if u is None or j is None:
            return []

        # Vectorized score over all items
        # score(u, i, j) = P_u · Q_i + L_j · M_i
        s_pref = self.Q @ self.P[u]   # [n_items] — user preference component
        s_mc   = self.M @ self.L[j]   # [n_items] — Markov transition component
        scores = s_pref + s_mc

        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            vid = self.idx_to_item[int(idx)]
            if not seen_items or vid not in seen_items:
                results.append((vid, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def evaluate(
        self,
        test_sessions: list[dict],
        k_values     : list[int] = [5, 10, 20],
        max_sessions : int = 1000,
        seed         : int = 42,
    ) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        if len(test_sessions) > max_sessions:
            indices = rng.choice(len(test_sessions), size=max_sessions, replace=False)
            test_sessions = [test_sessions[i] for i in indices]

        metric_lists: dict[str, list[float]] = defaultdict(list)

        for session in test_sessions:
            seq = session.get("video_sequence", [])
            if len(seq) < 3:
                continue
            uid       = session["user_id"]
            last_seen = seq[-2]
            target    = seq[-1]
            seen      = set(seq[:-1])

            recs    = self.recommend(uid, last_seen, top_k=max(k_values), seen_items=seen)
            rec_ids = [v for v, _ in recs]
            m = sequential_metrics(rec_ids, target, k_values)
            for k, v in m.items():
                metric_lists[k].append(v)

        return {k: float(np.mean(v)) if v else 0.0 for k, v in metric_lists.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SEQUENTIAL RS: Markov Chain + FPMC")
    print("=" * 60)

    # Use per-user interaction sequences (much richer than sessions.json)
    all_sequences             = load_user_sequences(min_length=5)
    train_sessions, test_sess = split_sessions(all_sequences, test_fraction=0.2)

    # ---------------------------------------------------------------
    # Model 1: Markov Chain RS
    # ---------------------------------------------------------------
    print("\n[Step 1] Markov Chain RS (alpha=0.7)...")
    mc = MarkovChainRS(alpha=0.7)
    mc.fit(train_sessions)

    print("[MarkovChain] Evaluating...")
    mc_metrics = mc.evaluate(test_sess, k_values=[5, 10, 20], max_sessions=1000)
    print("Markov Chain Results:")
    for k in [5, 10, 20]:
        print(f"  @{k:<3}: HR={mc_metrics[f'hr@{k}']:.4f}  MRR={mc_metrics[f'mrr@{k}']:.4f}")

    # ---------------------------------------------------------------
    # Model 2: FPMC
    # ---------------------------------------------------------------
    print("\n[Step 2] FPMC (Factorized Personalized Markov Chains)...")
    fpmc = FPMCModel(n_factors=32, n_epochs=8, learning_rate=0.01, regularization=0.01)
    fpmc.fit(train_sessions)

    print("[FPMC] Evaluating...")
    fpmc_metrics = fpmc.evaluate(test_sess, k_values=[5, 10, 20], max_sessions=1000)
    print("FPMC Results:")
    for k in [5, 10, 20]:
        print(f"  @{k:<3}: HR={fpmc_metrics[f'hr@{k}']:.4f}  MRR={fpmc_metrics[f'mrr@{k}']:.4f}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  SEQUENTIAL RS COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'HR@5':>7} {'HR@10':>7} {'HR@20':>7} {'MRR@10':>8}")
    print("-" * 52)
    for name, m in [("MarkovChain", mc_metrics), ("FPMC", fpmc_metrics)]:
        print(f"  {name:<18} {m['hr@5']:>7.4f} {m['hr@10']:>7.4f} "
              f"{m['hr@20']:>7.4f} {m['mrr@10']:>8.4f}")

    # Save
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    rows = [
        {"model": "MarkovChain", **mc_metrics},
        {"model": "FPMC",        **fpmc_metrics},
    ]
    pd.DataFrame(rows).to_csv(out / "sequential_evaluation.csv", index=False)
    print(f"\n[Saved] -> experiments/sequential_evaluation.csv")
