"""
src/algorithms/mab.py
=====================
Multi-Armed Bandit algorithms for cold-start users.

Context: 500 users in your dataset have NO watch history (user_type="new").
Standard MF, BPR, and CF cannot recommend to these users — the cold-start problem.

MAB approach:
  - Each video = one "arm"
  - Reward = agent-centric score (NOT click-through rate)
              = info_density + credibility - clickbait
  - The bandit learns which videos have high agent-centric reward
    by balancing exploration (try unknown videos) vs. exploitation
    (recommend proven high-quality videos)

KEY DISTINCTION from standard MAB:
  - Standard MAB: reward = user clicked? (optimizes engagement)
  - Shield MAB  : reward = agent_score(video) (optimizes quality)
  This is exactly the "hack YouTube's algorithm" thesis:
  the exploration strategy finds quality, not virality.

Three algorithms:
1. UCB1 (Upper Confidence Bound):
     arm = argmax(Q(a) + sqrt(2 * ln(N) / n(a)))
     Pure exploration-exploitation with no user context.
     Best for: first few interactions (no context available yet)

2. Thompson Sampling:
     arm = argmax(sample from Beta(α_a, β_a))
     Bayesian approach — maintains uncertainty about each arm's quality.
     Best for: when reward is binary-ish (good/bad threshold on agent_score)

3. LinUCB (Contextual Bandit):
     arm = argmax(θ_a^T x_u + α * sqrt(x_u^T A_a^-1 x_u))
     Uses USER context features (from feature_encoder.py) to personalize.
     Best for: when user profile is available at cold-start (signup questionnaire)

Simulated rewards:
  Since new users have no interaction history, rewards are simulated
  from the item_catalog stub agent scores + Gaussian noise.
  In production, rewards would come from actual user feedback.

Usage:
    python -m src.algorithms.mab
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")


def compute_agent_reward(row: pd.Series, noise_std: float = 0.05) -> float:
    """
    Agent-centric reward for a video, derived from stub agent scores.
    This is the Shield thesis: reward is quality, not engagement.

    reward = 0.35 * info_density
           + 0.35 * credibility
           - 0.20 * clickbait
           - 0.10 * emotional_manipulation
    Clipped to [0, 1].
    """
    reward = (
        0.35 * float(row.get("info_density_stub", 0.5))
        + 0.35 * float(row.get("credibility_stub", 0.5))
        - 0.20 * float(row.get("clickbait_score_stub", 0.3))
        - 0.10 * float(row.get("emotional_manipulation_stub", 0.3))
    )
    noise = np.random.normal(0, noise_std)
    return float(np.clip(reward + noise, 0.0, 1.0))


# ---------------------------------------------------------------------------
# UCB1 Bandit
# ---------------------------------------------------------------------------

class UCB1Bandit:
    """
    Upper Confidence Bound 1 (UCB1) bandit.

    For each arm a:
        UCB(a) = Q(a) + c * sqrt(ln(N) / n(a))

    Where:
        Q(a) = mean observed reward
        N    = total pulls so far
        n(a) = pulls of arm a
        c    = exploration constant (default: sqrt(2))

    Non-contextual: same recommendation for all new users.
    Appropriate before any user signal is available.
    """

    def __init__(self, c: float = np.sqrt(2), seed: int = 42):
        self.c    = c
        self.seed = seed
        self.n_arms    : int = 0
        self.video_ids : list[str] = []
        self.Q         : np.ndarray = np.array([])   # mean rewards
        self.n_pulls   : np.ndarray = np.array([])   # pull counts
        self.N         : int = 0   # total pulls

    def initialize(self, video_ids: list[str]) -> None:
        """Set up arms from item catalog."""
        self.video_ids = video_ids
        self.n_arms    = len(video_ids)
        self.Q         = np.zeros(self.n_arms, dtype=np.float64)
        self.n_pulls   = np.zeros(self.n_arms, dtype=np.int64)
        self.N         = 0

    def select_arm(self) -> int:
        """Return the index of the arm to pull next."""
        # Pull each arm at least once first
        unpulled = np.where(self.n_pulls == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])

        # UCB scores
        ucb_scores = self.Q + self.c * np.sqrt(np.log(self.N) / self.n_pulls)
        return int(np.argmax(ucb_scores))

    def update(self, arm_idx: int, reward: float) -> None:
        """Update arm statistics after observing reward."""
        self.n_pulls[arm_idx] += 1
        self.N += 1
        # Incremental mean update
        n = self.n_pulls[arm_idx]
        self.Q[arm_idx] += (reward - self.Q[arm_idx]) / n

    def recommend(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return top-K arms by current UCB score."""
        if self.N == 0:
            # Before any pulls, return random
            indices = np.random.choice(self.n_arms, size=min(top_k, self.n_arms), replace=False)
        else:
            unpulled = np.where(self.n_pulls == 0)[0]
            if len(unpulled) >= top_k:
                indices = unpulled[:top_k]
            else:
                ucb_scores = self.Q.copy()
                pulled_mask = self.n_pulls > 0
                ucb_scores[pulled_mask] += (
                    self.c * np.sqrt(np.log(max(self.N, 1)) / self.n_pulls[pulled_mask])
                )
                indices = np.argsort(ucb_scores)[::-1][:top_k]

        return [(self.video_ids[i], float(self.Q[i])) for i in indices]

    def simulate(
        self,
        item_meta  : pd.DataFrame,
        n_rounds   : int = 500,
        noise_std  : float = 0.05,
    ) -> pd.DataFrame:
        """
        Simulate n_rounds of exploration.
        Returns a DataFrame tracking cumulative reward and regret over time.
        """
        meta_idx = item_meta.set_index("video_id")
        if meta_idx.index.duplicated().any():
            meta_idx = meta_idx[~meta_idx.index.duplicated(keep="first")]

        # Pre-compute true agent rewards for all arms (oracle)
        true_rewards = np.array([
            compute_agent_reward(meta_idx.loc[vid], noise_std=0.0)
            if vid in meta_idx.index else 0.5
            for vid in self.video_ids
        ])
        best_reward = true_rewards.max()

        history = []
        cumulative_reward = 0.0
        cumulative_regret = 0.0

        for t in range(1, n_rounds + 1):
            arm_idx = self.select_arm()
            vid = self.video_ids[arm_idx]
            reward = (
                compute_agent_reward(meta_idx.loc[vid], noise_std=noise_std)
                if vid in meta_idx.index
                else np.random.uniform(0.2, 0.5)
            )
            self.update(arm_idx, reward)

            cumulative_reward += reward
            cumulative_regret += best_reward - true_rewards[arm_idx]

            if t % 100 == 0 or t <= 10:
                history.append({
                    "round"              : t,
                    "arm_pulled"         : arm_idx,
                    "reward"             : round(reward, 4),
                    "cumulative_reward"  : round(cumulative_reward, 4),
                    "cumulative_regret"  : round(cumulative_regret, 4),
                    "avg_reward"         : round(cumulative_reward / t, 4),
                    "best_arm_q"         : round(self.Q.max(), 4),
                })

        return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Thompson Sampling
# ---------------------------------------------------------------------------

class ThompsonSamplingBandit:
    """
    Thompson Sampling (Bayesian bandit).

    Models reward as Bernoulli (good = reward > threshold, bad = below).
    Prior: Beta(1, 1) = Uniform
    Update: Beta(alpha + reward_is_good, beta + reward_is_bad)

    Naturally balances exploration and exploitation via posterior sampling:
    arms with high uncertainty get explored more.

    Key advantage over UCB1:
    - Handles correlated arms better
    - Converges faster in practice
    - Posterior is interpretable (confidence interval on quality)
    """

    def __init__(self, reward_threshold: float = 0.5, seed: int = 42):
        """
        Args:
            reward_threshold : agent_score above this = "good" arm
        """
        self.threshold = reward_threshold
        self.rng = np.random.default_rng(seed)
        self.n_arms   : int = 0
        self.video_ids: list[str] = []
        self.alpha    : np.ndarray = np.array([])  # successes + 1
        self.beta_    : np.ndarray = np.array([])  # failures  + 1

    def initialize(self, video_ids: list[str]) -> None:
        self.video_ids = video_ids
        self.n_arms    = len(video_ids)
        self.alpha     = np.ones(self.n_arms, dtype=np.float64)  # Beta(1,1) prior
        self.beta_     = np.ones(self.n_arms, dtype=np.float64)

    def select_arm(self) -> int:
        """Sample from each arm's posterior Beta distribution."""
        samples = self.rng.beta(self.alpha, self.beta_)
        return int(np.argmax(samples))

    def update(self, arm_idx: int, reward: float) -> None:
        """Update Beta posterior."""
        if reward >= self.threshold:
            self.alpha[arm_idx] += 1.0
        else:
            self.beta_[arm_idx] += 1.0

    def recommend(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return top-K arms by posterior mean E[Beta] = alpha/(alpha+beta)."""
        posterior_means = self.alpha / (self.alpha + self.beta_)
        top_indices = np.argsort(posterior_means)[::-1][:top_k]
        return [(self.video_ids[i], float(posterior_means[i])) for i in top_indices]

    def simulate(
        self, item_meta: pd.DataFrame, n_rounds: int = 500, noise_std: float = 0.05
    ) -> pd.DataFrame:
        meta_idx = item_meta.set_index("video_id")
        if meta_idx.index.duplicated().any():
            meta_idx = meta_idx[~meta_idx.index.duplicated(keep="first")]

        true_rewards = np.array([
            compute_agent_reward(meta_idx.loc[vid], noise_std=0.0)
            if vid in meta_idx.index else 0.5
            for vid in self.video_ids
        ])
        best_reward = true_rewards.max()

        history = []
        cumulative_reward = 0.0
        cumulative_regret = 0.0

        for t in range(1, n_rounds + 1):
            arm_idx = self.select_arm()
            vid = self.video_ids[arm_idx]
            reward = (
                compute_agent_reward(meta_idx.loc[vid], noise_std=noise_std)
                if vid in meta_idx.index
                else float(self.rng.uniform(0.2, 0.5))
            )
            self.update(arm_idx, reward)
            cumulative_reward += reward
            cumulative_regret += best_reward - true_rewards[arm_idx]

            if t % 100 == 0 or t <= 10:
                history.append({
                    "round"            : t,
                    "reward"           : round(reward, 4),
                    "cumulative_reward": round(cumulative_reward, 4),
                    "cumulative_regret": round(cumulative_regret, 4),
                    "avg_reward"       : round(cumulative_reward / t, 4),
                })

        return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# LinUCB (Contextual Bandit)
# ---------------------------------------------------------------------------

class LinUCBBandit:
    """
    LinUCB: Contextual Linear UCB Bandit (Li et al. 2010).

    Models reward as linear in user context:
        E[r_a | x] = θ_a^T x

    UCB for arm a given context x:
        UCB_a(x) = θ_a^T x + alpha * sqrt(x^T A_a^-1 x)

    Where:
        θ_a = A_a^-1 b_a  (OLS estimate of arm parameters)
        A_a = Σ x_t x_t^T + I  (design matrix, regularized)
        b_a = Σ r_t x_t         (reward-weighted contexts)

    Uses USER FEATURES from feature_encoder.py as context x.
    This personalizes cold-start recommendations based on user profile
    (age, topic preferences, language, credibility sensitivity, etc.)

    This is the most powerful MAB for the Shield project:
    even without interaction history, we can use the signup questionnaire
    answers (mapped to user features) to personalize recommendations.
    """

    def __init__(self, context_dim: int, alpha: float = 1.0, seed: int = 42):
        """
        Args:
            context_dim : dimensionality of user feature vector
            alpha       : exploration constant (higher = more exploration)
        """
        self.context_dim = context_dim
        self.alpha       = alpha
        self.seed        = seed
        self.rng         = np.random.default_rng(seed)

        self.n_arms   : int = 0
        self.video_ids: list[str] = []

        # Per-arm matrices: A_a [d x d], b_a [d]
        self.A: Optional[np.ndarray] = None  # [n_arms x d x d]
        self.b: Optional[np.ndarray] = None  # [n_arms x d]

    def initialize(self, video_ids: list[str]) -> None:
        self.video_ids = video_ids
        self.n_arms    = len(video_ids)
        d = self.context_dim
        self.A = np.array([np.eye(d, dtype=np.float64) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms, d), dtype=np.float64)

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm with highest UCB given user context."""
        x = context.astype(np.float64)
        best_ucb = -np.inf
        best_arm = 0

        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.context_dim))
            theta = A_inv @ self.b[a]
            # Exploitation + exploration bonus
            ucb = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = a

        return best_arm

    def select_top_k(self, context: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Select top-K arms by UCB score for a given user context."""
        x = context.astype(np.float64)
        ucb_scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.context_dim))
            theta = A_inv @ self.b[a]
            ucb_scores[a] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)

        top_indices = np.argsort(ucb_scores)[::-1][:top_k]
        return [(self.video_ids[i], float(ucb_scores[i])) for i in top_indices]

    def update(self, arm_idx: int, context: np.ndarray, reward: float) -> None:
        """Update A_a and b_a after observing reward."""
        x = context.astype(np.float64)
        self.A[arm_idx] += np.outer(x, x)
        self.b[arm_idx] += reward * x

    def simulate(
        self,
        item_meta   : pd.DataFrame,
        user_features: np.ndarray,   # [n_users x d] — use cold-start users
        user_ids    : list[str],
        n_rounds    : int = 200,
        noise_std   : float = 0.05,
    ) -> pd.DataFrame:
        """
        Simulate n_rounds with randomly sampled user contexts.
        Each round: pick a random new user, get UCB recommendation, observe reward.
        """
        meta_idx = item_meta.set_index("video_id")
        if meta_idx.index.duplicated().any():
            meta_idx = meta_idx[~meta_idx.index.duplicated(keep="first")]

        true_rewards = np.array([
            compute_agent_reward(meta_idx.loc[vid], noise_std=0.0)
            if vid in meta_idx.index else 0.5
            for vid in self.video_ids
        ])
        best_reward = true_rewards.max()

        history = []
        cumulative_reward = 0.0
        cumulative_regret = 0.0
        n_users = len(user_ids)

        for t in range(1, n_rounds + 1):
            # Sample a random cold-start user
            u_idx   = self.rng.integers(0, n_users)
            context = user_features[u_idx]

            arm_idx = self.select_arm(context)
            vid     = self.video_ids[arm_idx]
            reward  = (
                compute_agent_reward(meta_idx.loc[vid], noise_std=noise_std)
                if vid in meta_idx.index
                else float(self.rng.uniform(0.2, 0.5))
            )
            self.update(arm_idx, context, reward)
            cumulative_reward += reward
            cumulative_regret += best_reward - true_rewards[arm_idx]

            if t % 50 == 0 or t <= 5:
                history.append({
                    "round"            : t,
                    "reward"           : round(reward, 4),
                    "cumulative_reward": round(cumulative_reward, 4),
                    "cumulative_regret": round(cumulative_regret, 4),
                    "avg_reward"       : round(cumulative_reward / t, 4),
                })

        return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  MAB: Multi-Armed Bandits for Cold-Start Users")
    print("=" * 65)

    item_meta = pd.read_csv(DATA_DIR / "item_catalog.csv")

    # Use a representative sample of items for bandit arms
    # (full 1000 arms makes LinUCB matrix ops slow for demo — use 100)
    sample_items = item_meta.sample(n=min(100, len(item_meta)), random_state=42)
    video_ids    = sample_items["video_id"].tolist()

    print(f"\nBandit setup: {len(video_ids)} arms (videos), agent-centric reward")
    print(f"Reward = 0.35*info_density + 0.35*credibility - 0.20*clickbait - 0.10*manipulation")

    N_ROUNDS = 500

    # ---------------------------------------------------------------
    # UCB1
    # ---------------------------------------------------------------
    print(f"\n[Step 1] UCB1 ({N_ROUNDS} rounds)...")
    ucb = UCB1Bandit(c=np.sqrt(2), seed=42)
    ucb.initialize(video_ids)
    ucb_history = ucb.simulate(item_meta, n_rounds=N_ROUNDS, noise_std=0.05)
    ucb_final = ucb_history.iloc[-1]
    print(f"  Final avg reward : {ucb_final['avg_reward']:.4f}")
    print(f"  Total regret     : {ucb_final['cumulative_regret']:.4f}")
    top_ucb = ucb.recommend(top_k=5)
    print(f"  Top-5 arms: {[(v[:12], f'{s:.3f}') for v, s in top_ucb]}")

    # ---------------------------------------------------------------
    # Thompson Sampling
    # ---------------------------------------------------------------
    print(f"\n[Step 2] Thompson Sampling ({N_ROUNDS} rounds)...")
    ts = ThompsonSamplingBandit(reward_threshold=0.5, seed=42)
    ts.initialize(video_ids)
    ts_history = ts.simulate(item_meta, n_rounds=N_ROUNDS, noise_std=0.05)
    ts_final = ts_history.iloc[-1]
    print(f"  Final avg reward : {ts_final['avg_reward']:.4f}")
    print(f"  Total regret     : {ts_final['cumulative_regret']:.4f}")
    top_ts = ts.recommend(top_k=5)
    print(f"  Top-5 arms: {[(v[:12], f'{s:.3f}') for v, s in top_ts]}")

    # ---------------------------------------------------------------
    # LinUCB (contextual)
    # ---------------------------------------------------------------
    print(f"\n[Step 3] LinUCB Contextual Bandit (200 rounds)...")
    user_feat_path = DATA_DIR / "user_features.npy"
    user_idx_path  = DATA_DIR / "user_index.json"

    if user_feat_path.exists():
        user_feats = np.load(user_feat_path)
        user_idx   = json.load(open(user_idx_path))
        # Get new users (cold start) — those with user_type = "new"
        users_data = json.load(open(DATA_DIR / "users.json", encoding="utf-8"))
        new_user_ids = [u["user_id"] for u in users_data if u.get("user_type") == "new"]
        # New users have no interactions -> not in user_index -> not in user_feats matrix
        # Fall back: use all users' features to simulate cold-start context distribution
        new_user_indices = [user_idx[uid] for uid in new_user_ids if uid in user_idx]
        if len(new_user_indices) == 0:
            # All new users excluded from matrix — use full feature matrix
            cold_start_feats = user_feats
            sim_user_ids = list(user_idx.keys())
            print(f"  Note: new users have no matrix rows (no interactions). Using all {len(sim_user_ids)} users as simulation pool.")
        else:
            cold_start_feats = user_feats[new_user_indices]
            sim_user_ids = new_user_ids
        print(f"  Cold-start simulation pool: {len(sim_user_ids)} users | Feature dim: {user_feats.shape[1]}")

        linucb = LinUCBBandit(context_dim=user_feats.shape[1], alpha=1.0, seed=42)
        linucb.initialize(video_ids)
        linucb_history = linucb.simulate(
            item_meta, cold_start_feats, sim_user_ids, n_rounds=200, noise_std=0.05
        )
        linucb_final = linucb_history.iloc[-1]
        print(f"  Final avg reward : {linucb_final['avg_reward']:.4f}")
        print(f"  Total regret     : {linucb_final['cumulative_regret']:.4f}")

        # Demo recommendation for a sample cold-start user
        sample_ctx = cold_start_feats[0]
        linucb_recs = linucb.select_top_k(sample_ctx, top_k=5)
        print(f"  Sample user recs: {[(v[:12], f'{s:.3f}') for v, s in linucb_recs]}")
    else:
        print("  [SKIP] user_features.npy not found. Run feature_encoder.py first.")
        linucb_history = pd.DataFrame()

    # Save
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    ucb_history.to_csv(out / "mab_ucb1.csv", index=False)
    ts_history.to_csv(out / "mab_thompson.csv", index=False)
    if not linucb_history.empty:
        linucb_history.to_csv(out / "mab_linucb.csv", index=False)
    print(f"\n[Saved] MAB histories -> experiments/mab_*.csv")

    # Summary comparison
    print("\n" + "=" * 45)
    print("  MAB Summary (final avg reward after simulation)")
    print("=" * 45)
    print(f"  UCB1              : {ucb_final['avg_reward']:.4f}")
    print(f"  Thompson Sampling : {ts_final['avg_reward']:.4f}")
    if not linucb_history.empty:
        print(f"  LinUCB            : {linucb_final['avg_reward']:.4f}")
    print()
    print("  Best possible (oracle) reward:")
    meta_idx = item_meta.set_index("video_id")
    if meta_idx.index.duplicated().any():
        meta_idx = meta_idx[~meta_idx.index.duplicated(keep="first")]
    true_rewards = [
        compute_agent_reward(meta_idx.loc[vid], noise_std=0.0)
        if vid in meta_idx.index else 0.5
        for vid in video_ids
    ]
    print(f"  Max oracle reward : {max(true_rewards):.4f}")
    print(f"  Mean oracle reward: {np.mean(true_rewards):.4f}")
