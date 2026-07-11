"""
src/algorithms/causal_rs.py
============================
Debiased Recommender System using Causal Inference.

MOTIVATION (the core of your Shield thesis):
  Standard RS (MF, CF, BPR) trains on OBSERVED interactions.
  But observed interactions are CONFOUNDED by popularity bias:
    - Popular videos (high view_count, high subscriber_count) are
      MORE LIKELY TO BE RECOMMENDED by YouTube → MORE LIKELY TO BE CLICKED
    - This inflates their estimated quality in training data
    - Model learns: "popular = good" (wrong! popular = heavily promoted)

  The Shield algorithm must DEBIAS this:
    "Estimate what a user's true preference would be if they had
     equal exposure probability to all videos" — not just the ones
     YouTube's algorithm chose to show them.

CAUSAL FRAMEWORK:
  Treatment:  T = video was exposed to the user (T ∈ {0,1})
  Outcome:    Y = watch_pct (user's engagement if exposed)
  Confounder: X = popularity features (view_count, subscriber_count, clickbait_score)

  Propensity score: P(T=1 | X) = P(video was exposed | popularity features)
    → Estimated via logistic regression on interaction data
    → High view_count → high propensity (popular = more likely shown by YouTube)

THREE ESTIMATORS:

1. Naive (biased baseline):
     score(u, i) = watch_pct(u, i) / 1.0
     Just the raw observed rating. Subject to popularity bias.

2. IPS (Inverse Propensity Scoring):
     score(u, i) = watch_pct(u, i) / propensity(i)
     Divides by the probability of exposure. High-propensity (popular)
     items get DISCOUNTED — their inflated training signal is corrected.

     Problem: High variance when propensity is near 0 (unseen items).
     Solution: Clip propensity to [clip_min, 1.0].

3. DR (Doubly Robust):
     score(u, i) = imputed(u, i) + (watch_pct - imputed(u, i)) / propensity(i)
     Combines a direct model (imputed) with IPS correction.
     Consistent if EITHER the propensity OR imputed model is correct.
     Lowest variance of the three — recommended for production.

IMPLEMENTATION:
  1. Propensity model: logistic regression P(observed | view_count, subscriber_count, clickbait)
  2. Imputed model: linear regression predicting watch_pct from item features
  3. DR estimator: combines both
  4. Re-rank any model's recommendations using DR-debiased scores
  5. Compare: naive ALS vs. DR-debiased ALS (this is your paper's key result)

EXPECTED RESULT:
  DR-debiased recommendations will have:
  - LOWER average view_count (less popularity-biased)
  - HIGHER average credibility_stub / info_density_stub (more quality-aligned)
  - LOWER average clickbait_score_stub (fewer manipulative videos)

Usage:
    python -m src.algorithms.causal_rs
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.algorithms.matrix_factorization import (
    load_data, train_test_split,
    ndcg_at_k, precision_at_k, recall_at_k, hit_rate_at_k,
)

DATA_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Propensity Model
# ---------------------------------------------------------------------------

class PropensityModel:
    """
    Estimates P(T=1 | X) = P(video was exposed to user | video's popularity features).

    We model T = 1 as "this (user, video) pair appears in the training data".
    Confounders X = [log_view_count, log_subscriber_count, clickbait_score, duration_log]

    Since we only observe interactions (T=1), we need negative examples.
    Strategy: for each user, sample videos they did NOT interact with as T=0.
    This gives us a binary classification problem.
    """

    def __init__(self, clip_min: float = 0.05):
        """
        Args:
            clip_min : minimum propensity value to prevent extreme IPS weights
        """
        self.clip_min  = clip_min
        self.model     = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        self.scaler    = StandardScaler()
        self.is_fitted = False
        self.feature_names = [
            "log_view_count",
            "log_subscriber_count",
            "log_like_count",
            "clickbait_score_stub",
            "emotional_manipulation_stub",
            "duration_log_norm",
        ]

    def fit(
        self,
        interactions_df: pd.DataFrame,
        item_catalog   : pd.DataFrame,
        neg_sample_ratio: float = 1.0,
        seed           : int = 42,
    ) -> "PropensityModel":
        """
        Build training data and fit logistic regression.

        Positive examples (T=1): rows from interactions_df
        Negative examples (T=0): (user, item) pairs NOT in interactions
        """
        rng = np.random.default_rng(seed)
        print("[PropensityModel] Building training set...")

        # Build item feature lookup
        meta = item_catalog.copy()
        if meta.index.name != "video_id":
            meta = meta.set_index("video_id")
        if meta.index.duplicated().any():
            meta = meta[~meta.index.duplicated(keep="first")]

        def get_item_features(video_id: str) -> Optional[np.ndarray]:
            if video_id not in meta.index:
                return None
            row = meta.loc[video_id]
            vc  = max(float(row.get("view_count", 0)), 0)
            sc  = max(float(row.get("subscriber_count", 0)), 0)
            lc  = max(float(row.get("like_count", 0)), 0)
            dur = max(float(row.get("duration_seconds", 300)), 1)
            return np.array([
                np.log1p(vc),
                np.log1p(sc),
                np.log1p(lc),
                float(row.get("clickbait_score_stub", 0.3)),
                float(row.get("emotional_manipulation_stub", 0.3)),
                np.log1p(dur) / np.log1p(7200),
            ], dtype=np.float32)

        # Positive examples
        pos_features, neg_features = [], []

        all_video_ids = list(meta.index)
        observed_pairs = set(zip(interactions_df["user_id"], interactions_df["video_id"]))
        # Sample a manageable subset of positive interactions
        sample_pos = interactions_df.sample(n=min(20_000, len(interactions_df)), random_state=seed)

        for _, row in sample_pos.iterrows():
            feats = get_item_features(row["video_id"])
            if feats is not None:
                pos_features.append(feats)

        # Negative examples: same users, random unobserved items
        n_neg = int(len(pos_features) * neg_sample_ratio)
        user_ids = sample_pos["user_id"].unique()
        neg_count = 0
        while neg_count < n_neg:
            u = str(rng.choice(user_ids))
            v = str(rng.choice(all_video_ids))
            if (u, v) not in observed_pairs:
                feats = get_item_features(v)
                if feats is not None:
                    neg_features.append(feats)
                    neg_count += 1

        X = np.vstack(pos_features + neg_features)
        y = np.array([1] * len(pos_features) + [0] * len(neg_features))

        print(f"[PropensityModel] Fitting on {len(X):,} examples "
              f"({len(pos_features):,} pos, {len(neg_features):,} neg)...")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Report
        train_acc = self.model.score(X_scaled, y)
        print(f"[PropensityModel] Train accuracy: {train_acc:.4f}")
        return self

    def predict_propensity(self, video_id: str, meta: pd.DataFrame) -> float:
        """Return P(T=1 | video_id) clipped to [clip_min, 1.0]."""
        if not self.is_fitted:
            return 0.5
        if video_id not in meta.index:
            return 0.5

        row = meta.loc[video_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        vc  = max(float(row.get("view_count", 0)), 0)
        sc  = max(float(row.get("subscriber_count", 0)), 0)
        lc  = max(float(row.get("like_count", 0)), 0)
        dur = max(float(row.get("duration_seconds", 300)), 1)

        x = np.array([[
            np.log1p(vc),
            np.log1p(sc),
            np.log1p(lc),
            float(row.get("clickbait_score_stub", 0.3)),
            float(row.get("emotional_manipulation_stub", 0.3)),
            np.log1p(dur) / np.log1p(7200),
        ]])
        x_scaled = self.scaler.transform(x)
        prob = float(self.model.predict_proba(x_scaled)[0, 1])
        return float(np.clip(prob, self.clip_min, 1.0))

    def batch_propensities(
        self, video_ids: list[str], meta: pd.DataFrame
    ) -> np.ndarray:
        """Vectorized propensity prediction for a list of video_ids."""
        if meta.index.name != "video_id":
            meta = meta.set_index("video_id")
        if meta.index.duplicated().any():
            meta = meta[~meta.index.duplicated(keep="first")]

        rows = []
        for vid in video_ids:
            if vid in meta.index:
                row = meta.loc[vid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                vc  = max(float(row.get("view_count", 0)), 0)
                sc  = max(float(row.get("subscriber_count", 0)), 0)
                lc  = max(float(row.get("like_count", 0)), 0)
                dur = max(float(row.get("duration_seconds", 300)), 1)
                rows.append([
                    np.log1p(vc), np.log1p(sc), np.log1p(lc),
                    float(row.get("clickbait_score_stub", 0.3)),
                    float(row.get("emotional_manipulation_stub", 0.3)),
                    np.log1p(dur) / np.log1p(7200),
                ])
            else:
                rows.append([0, 0, 0, 0.3, 0.3, 0.5])

        X = np.array(rows, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return np.clip(probs, self.clip_min, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Empirical Propensity Model (Strong Popularity Debiasing)
# ---------------------------------------------------------------------------

class EmpiricalPropensityModel:
    """
    Computes propensity empirically based on how often an item is interacted with.
    P(item_i exposed) ∝ count(item_i in training data).

    This definitively captures the popularity bias (which the logistic regression
    struggled to isolate from topic-affinity). Dividing by this propensity forces
    the algorithm to penalize items strictly based on their raw popularity.
    """
    def __init__(self, clip_min: float = 0.01):
        self.clip_min = clip_min
        self.item_propensities: dict[str, float] = {}
        self.is_fitted = False

    def fit(self, interactions_df: pd.DataFrame) -> "EmpiricalPropensityModel":
        print("[EmpiricalPropensity] Counting item frequencies...")
        counts = interactions_df["video_id"].value_counts()
        max_count = counts.max()

        # Normalize so the most popular item has propensity ~1.0
        # and others have proportionally less.
        for vid, count in counts.items():
            self.item_propensities[str(vid)] = float(count) / max_count

        self.is_fitted = True
        return self

    def predict_propensity(self, video_id: str, meta: pd.DataFrame) -> float:
        if not self.is_fitted:
            return 0.5
        prob = self.item_propensities.get(video_id, self.clip_min)
        return float(np.clip(prob, self.clip_min, 1.0))

    def batch_propensities(
        self, video_ids: list[str], meta: pd.DataFrame
    ) -> np.ndarray:
        probs = [self.predict_propensity(v, meta) for v in video_ids]
        return np.array(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Imputed (direct) model for DR estimator
# ---------------------------------------------------------------------------

class ImputedModel:
    """
    Direct model predicting watch_pct from item features.
    Used as the baseline estimate in the Doubly Robust estimator.

    Simple Ridge regression: watch_pct ~ item_features
    (In production, this would be the pointwise MF model.)
    """

    def __init__(self, alpha: float = 1.0):
        self.model  = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self, interactions_df: pd.DataFrame, item_catalog: pd.DataFrame, seed: int = 42
    ) -> "ImputedModel":
        meta = item_catalog.set_index("video_id") if item_catalog.index.name != "video_id" else item_catalog
        if meta.index.duplicated().any():
            meta = meta[~meta.index.duplicated(keep="first")]

        sample = interactions_df.sample(n=min(30_000, len(interactions_df)), random_state=seed)
        X_rows, y_vals = [], []

        for _, row in sample.iterrows():
            vid = row["video_id"]
            if vid not in meta.index:
                continue
            item = meta.loc[vid]
            if isinstance(item, pd.DataFrame):
                item = item.iloc[0]

            feats = [
                float(item.get("clickbait_score_stub", 0.3)),
                float(item.get("info_density_stub", 0.5)),
                float(item.get("credibility_stub", 0.5)),
                float(item.get("emotional_manipulation_stub", 0.3)),
                np.log1p(max(float(item.get("view_count", 0)), 0)),
                np.log1p(max(float(item.get("subscriber_count", 0)), 0)),
                np.log1p(max(float(item.get("duration_seconds", 300)), 1)) / np.log1p(7200),
            ]
            X_rows.append(feats)
            y_vals.append(float(row["watch_pct"]))

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_vals, dtype=np.float32)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        print(f"[ImputedModel] Fitted on {len(X):,} examples | "
              f"R^2: {self.model.score(X_scaled, y):.4f}")
        return self

    def predict(self, video_id: str, meta: pd.DataFrame) -> float:
        if not self.is_fitted or video_id not in meta.index:
            return 0.5
        item = meta.loc[video_id]
        if isinstance(item, pd.DataFrame):
            item = item.iloc[0]
        feats = np.array([[
            float(item.get("clickbait_score_stub", 0.3)),
            float(item.get("info_density_stub", 0.5)),
            float(item.get("credibility_stub", 0.5)),
            float(item.get("emotional_manipulation_stub", 0.3)),
            np.log1p(max(float(item.get("view_count", 0)), 0)),
            np.log1p(max(float(item.get("subscriber_count", 0)), 0)),
            np.log1p(max(float(item.get("duration_seconds", 300)), 1)) / np.log1p(7200),
        ]])
        x_s = self.scaler.transform(feats)
        return float(np.clip(self.model.predict(x_s)[0], 0.0, 1.0))


# ---------------------------------------------------------------------------
# Causal Recommender (IPS / DR debiasing)
# ---------------------------------------------------------------------------

class CausalRecommender:
    """
    Wraps any base RS model (e.g., ALS) and applies causal debiasing at
    recommendation time via IPS or DR estimators.

    The base model provides:
        base_score(u, i) → collaborative filtering score

    The causal layer corrects:
        ips_score(u, i)  = base_score / propensity(i)
        dr_score(u, i)   = imputed(i) + (base_score - imputed(i)) / propensity(i)

    Re-ranking using DR:
        1. Get top-K*2 candidates from base model (cast wide net)
        2. Rerank by DR score (which discounts high-propensity popular items)
        3. Return top-K after debiasing
    """

    def __init__(
        self,
        propensity_model, # Can be PropensityModel or EmpiricalPropensityModel
        imputed_model   : ImputedModel,
        estimator       : str = "dr",   # "naive" | "ips" | "dr"
    ):
        assert estimator in ("naive", "ips", "dr"), f"Unknown estimator: {estimator}"
        self.propensity = propensity_model
        self.imputed    = imputed_model
        self.estimator  = estimator

    def debias_score(
        self,
        base_score: float,
        video_id  : str,
        meta      : pd.DataFrame,
    ) -> float:
        """
        Apply the selected estimator to debias a single (user, item) score.
        """
        if self.estimator == "naive":
            return base_score

        propensity = self.propensity.predict_propensity(video_id, meta)

        if self.estimator == "ips":
            return base_score / propensity

        # DR estimator
        imputed = self.imputed.predict(video_id, meta)
        return imputed + (base_score - imputed) / propensity

    def rerank(
        self,
        base_recommendations: list[tuple[str, float]],
        meta                : pd.DataFrame,
        top_k               : int = 10,
    ) -> list[tuple[str, float]]:
        """
        Rerank a list of (video_id, base_score) using the DR/IPS estimator.
        Returns top_k items after debiasing.
        """
        debiased = []
        for vid, score in base_recommendations:
            db_score = self.debias_score(score, vid, meta)
            debiased.append((vid, db_score))

        debiased.sort(key=lambda x: x[1], reverse=True)
        return debiased[:top_k]


# ---------------------------------------------------------------------------
# Analysis: compare biased vs. debiased recommendations
# ---------------------------------------------------------------------------

def analyze_bias_reduction(
    biased_recs  : list[tuple[str, float]],
    debiased_recs: list[tuple[str, float]],
    meta         : pd.DataFrame,
    top_k        : int = 10,
) -> dict:
    """
    Compare quality signals between biased and debiased recommendation lists.
    This is the key result for your paper:
    Show that DR debiasing reduces average popularity and increases average quality.
    """
    if meta.index.name != "video_id":
        meta = meta.set_index("video_id")
    if meta.index.duplicated().any():
        meta = meta[~meta.index.duplicated(keep="first")]

    def stats(recs):
        vids = [v for v, _ in recs[:top_k] if v in meta.index]
        if not vids:
            return {}
        rows = meta.loc[vids]
        return {
            "avg_view_count"    : float(rows["view_count"].mean()),
            "avg_credibility"   : float(rows["credibility_stub"].mean()),
            "avg_info_density"  : float(rows["info_density_stub"].mean()),
            "avg_clickbait"     : float(rows["clickbait_score_stub"].mean()),
            "avg_manipulation"  : float(rows["emotional_manipulation_stub"].mean()),
        }

    b  = stats(biased_recs)
    db = stats(debiased_recs)

    return {
        "biased"              : b,
        "debiased"            : db,
        "delta_view_count"    : db.get("avg_view_count", 0) - b.get("avg_view_count", 0),
        "delta_credibility"   : db.get("avg_credibility", 0) - b.get("avg_credibility", 0),
        "delta_info_density"  : db.get("avg_info_density", 0) - b.get("avg_info_density", 0),
        "delta_clickbait"     : db.get("avg_clickbait", 0) - b.get("avg_clickbait", 0),
        "delta_manipulation"  : db.get("avg_manipulation", 0) - b.get("avg_manipulation", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  CAUSAL RS: IPS & Doubly Robust Debiasing")
    print("=" * 65)

    # Load data
    matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item = load_data()
    train_df, test_df = train_test_split()

    meta_indexed = item_meta.set_index("video_id")
    if meta_indexed.index.duplicated().any():
        meta_indexed = meta_indexed[~meta_indexed.index.duplicated(keep="first")]

    # Step 1: Fit propensity model
    print("\n[Step 1] Fitting Empirical Propensity Model (strict popularity penalty)...")
    propensity_model = EmpiricalPropensityModel(clip_min=0.01)
    propensity_model.fit(train_df)

    # Step 2: Fit imputed model
    print("\n[Step 2] Fitting Imputed Model (for DR estimator)...")
    imputed_model = ImputedModel(alpha=1.0)
    imputed_model.fit(train_df, item_meta, seed=42)

    # Step 3: Load ALS model for base scores
    # (Re-train a quick ALS rather than re-load, since we don't persist models yet)
    print("\n[Step 3] Training ALS base model...")
    from src.algorithms.matrix_factorization import (
        ALSMatrixFactorization, build_matrix_from_df
    )
    train_matrix = build_matrix_from_df(train_df, user_idx, item_idx)
    als = ALSMatrixFactorization(n_factors=64, n_iterations=5, alpha=40.0, regularization=0.01)
    als.fit(train_matrix, user_idx, item_idx, item_meta, idx_to_user, idx_to_item)

    # Step 4: Build causal recommenders
    causal_ips = CausalRecommender(propensity_model, imputed_model, estimator="ips")
    causal_dr  = CausalRecommender(propensity_model, imputed_model, estimator="dr")

    # Step 5: Compare on sample users
    print("\n[Step 4] Comparing biased vs. debiased recommendations...")
    sample_users = list(user_idx.keys())[100:130]   # 30 users
    bias_analysis_rows = []

    for user_id in sample_users:
        # Get base ALS recs (top-20 candidates)
        biased_recs = als.recommend(user_id, top_k=20)
        if not biased_recs:
            continue

        # Apply DR debiasing (top-10 after reranking)
        dr_recs = causal_dr.rerank(biased_recs, meta_indexed, top_k=10)

        analysis = analyze_bias_reduction(biased_recs[:10], dr_recs, meta_indexed, top_k=10)
        bias_analysis_rows.append({
            "user_id": user_id,
            **{f"biased_{k}": v for k, v in analysis["biased"].items()},
            **{f"debiased_{k}": v for k, v in analysis["debiased"].items()},
            **{k: v for k, v in analysis.items() if k.startswith("delta_")},
        })

    df_bias = pd.DataFrame(bias_analysis_rows)

    print("\n" + "=" * 65)
    print("  BIAS ANALYSIS: Biased (ALS) vs. DR-Debiased (Causal RS)")
    print("=" * 65)
    print(f"\n  Averaged over {len(df_bias)} users:\n")
    metrics = ["avg_view_count", "avg_credibility", "avg_info_density",
               "avg_clickbait", "avg_manipulation"]
    for m in metrics:
        b_val  = df_bias[f"biased_{m}"].mean()
        db_val = df_bias[f"debiased_{m}"].mean()
        delta  = db_val - b_val
        arrow  = "UP" if delta > 0 else "DOWN"
        good   = (delta > 0) if m in ["avg_credibility", "avg_info_density"] else (delta < 0)
        marker = "[GOOD]" if good else "[BAD]"
        print(f"  {m:<25}: biased={b_val:>10.2f}  debiased={db_val:>10.2f}  "
              f"delta={delta:>+8.2f}  {arrow} {marker}")

    # Step 6: Evaluate debiased model on test set
    print("\n[Step 5] Evaluating DR-debiased vs. naive ALS on test set...")
    rng = np.random.default_rng(42)
    test_users = rng.choice(test_df["user_id"].unique(), size=200, replace=False)

    results = {"naive_als": [], "dr_debiased": []}
    k = 10
    for user_id in test_users:
        if user_id not in user_idx:
            continue
        user_test = test_df[test_df["user_id"] == user_id]
        relevant  = set(user_test[user_test["watch_pct"] > 0.5]["video_id"].tolist())
        if not relevant:
            continue

        # Naive
        naive_recs = als.recommend(user_id, top_k=k)
        naive_ids  = [v for v, _ in naive_recs]
        results["naive_als"].append(ndcg_at_k(naive_ids, relevant, k))

        # DR debiased (rerank from top-20)
        base_20   = als.recommend(user_id, top_k=20)
        dr_recs   = causal_dr.rerank(base_20, meta_indexed, top_k=k)
        dr_ids    = [v for v, _ in dr_recs]
        results["dr_debiased"].append(ndcg_at_k(dr_ids, relevant, k))

    naive_ndcg = float(np.mean(results["naive_als"]))
    dr_ndcg    = float(np.mean(results["dr_debiased"]))
    print(f"\n  NDCG@{k} Naive ALS      : {naive_ndcg:.4f}")
    print(f"  NDCG@{k} DR-Debiased   : {dr_ndcg:.4f}")
    delta = dr_ndcg - naive_ndcg
    print(f"  Delta               : {delta:+.4f}  "
          f"({'DR better' if delta >= 0 else 'naive better — check propensity model'})")

    # Save
    out = Path("experiments")
    out.mkdir(exist_ok=True)
    df_bias.to_csv(out / "causal_bias_analysis.csv", index=False)
    pd.DataFrame([
        {"model": "Naive ALS", "ndcg@10": naive_ndcg},
        {"model": "DR-Debiased ALS", "ndcg@10": dr_ndcg},
    ]).to_csv(out / "causal_evaluation.csv", index=False)
    print(f"\n[Saved] -> experiments/causal_*.csv")
