"""
Neural Learning-to-Rank Model for Agent-as-Shield RS

This is the core RS model. It's a neural ranker that learns to score
and rank videos. The same architecture is trained twice:

1. Train on ENGAGEMENT labels → produces a human-centric RS (baseline)
2. Train on AGENT labels → produces our agent-centric RS (contribution)

The model takes video features as input and outputs a relevance score.
We use a ListNet-style listwise loss for training, which is a standard
Learning-to-Rank approach used in production search/recommendation systems.

Architecture:
    Input (10 features) → FC(64) → ReLU → Dropout → FC(32) → ReLU → FC(1) → Score
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import json

# ── Feature columns used as model input ──
FEATURE_COLS = [
    "info_density",
    "credibility",
    "bias_balance",
    "clickbait_score",
    "emotional_score",
    "duration_minutes",
    "title_length",
    "desc_length",
    "tag_count",
    "has_transcript",
]


class VideoRankingDataset(Dataset):
    """
    Dataset for listwise Learning-to-Rank.
    Groups videos by query and returns (features, labels) per query group.
    """

    def __init__(self, df: pd.DataFrame, label_col: str, max_list_size: int = 50):
        self.groups = []
        self.label_col = label_col

        for _, group_df in df.groupby("query_group"):
            features = group_df[FEATURE_COLS].values.astype(np.float32)
            labels = group_df[label_col].values.astype(np.float32)

            # Pad or truncate to fixed list size
            n = len(features)
            if n > max_list_size:
                # Sample top items + random for diversity
                idx = np.argsort(-labels)[:max_list_size]
                features = features[idx]
                labels = labels[idx]
            elif n < max_list_size:
                pad_n = max_list_size - n
                features = np.vstack([features, np.zeros((pad_n, features.shape[1]))])
                labels = np.concatenate([labels, np.zeros(pad_n)])

            self.groups.append((
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.float32),
            ))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]


class NeuralRanker(nn.Module):
    """
    Neural scoring function for Learning-to-Rank.

    Takes a feature vector for a single video and outputs a relevance score.
    Applied independently to each video in a list, then scores are used
    to produce a ranking.
    """

    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch_size, list_size, num_features)
        Returns:
            scores: (batch_size, list_size)
        """
        batch_size, list_size, num_features = x.shape
        x_flat = x.view(-1, num_features)
        scores_flat = self.network(x_flat)
        scores = scores_flat.view(batch_size, list_size)
        return scores


def listnet_loss(predicted_scores: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
    """
    ListNet loss: Cross-entropy between the predicted probability distribution
    (softmax of scores) and the ground truth distribution (softmax of labels).

    This is a standard listwise LTR loss function used in RS research.
    """
    # Convert to probability distributions
    pred_probs = torch.softmax(predicted_scores, dim=-1)
    true_probs = torch.softmax(true_labels, dim=-1)

    # Cross-entropy loss
    loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=-1)
    return loss.mean()


def ndcg_at_k(predicted_scores: np.ndarray, true_labels: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@k — the standard RS evaluation metric.
    Measures how well the predicted ranking matches the ideal ranking.
    """
    # Get predicted ranking
    predicted_order = np.argsort(-predicted_scores)[:k]
    # Get ideal ranking
    ideal_order = np.argsort(-true_labels)[:k]

    # DCG
    dcg = sum(
        (2**true_labels[idx] - 1) / np.log2(rank + 2)
        for rank, idx in enumerate(predicted_order)
    )
    # Ideal DCG
    idcg = sum(
        (2**true_labels[idx] - 1) / np.log2(rank + 2)
        for rank, idx in enumerate(ideal_order)
    )

    return dcg / idcg if idcg > 0 else 0.0


def train_ranker(
    df: pd.DataFrame,
    label_col: str,
    model_name: str,
    epochs: int = 100,
    lr: float = 0.001,
    save_dir: str = "models",
) -> Tuple[NeuralRanker, Dict]:
    """
    Train a neural ranker on the given label column.

    Args:
        df: Training dataframe with features and labels
        label_col: Which label to train on ('engagement_label' or 'agent_label')
        model_name: Name for saving the model
        epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save model weights

    Returns:
        Trained model and training history
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Label: {label_col}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # ── Normalize features ──
    feature_means = df[FEATURE_COLS].mean()
    feature_stds = df[FEATURE_COLS].std().replace(0, 1)
    df_norm = df.copy()
    df_norm[FEATURE_COLS] = (df_norm[FEATURE_COLS] - feature_means) / feature_stds

    # Save normalization stats for inference
    norm_stats = {
        "means": feature_means.to_dict(),
        "stds": feature_stds.to_dict(),
    }
    with open(os.path.join(save_dir, f"{model_name}_norm_stats.json"), "w") as f:
        json.dump(norm_stats, f, indent=2)

    # ── Create dataset ──
    dataset = VideoRankingDataset(df_norm, label_col)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ── Initialize model ──
    model = NeuralRanker(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # ── Training loop ──
    history = {"loss": [], "ndcg@5": [], "ndcg@10": []}
    best_ndcg = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            scores = model(features)
            loss = listnet_loss(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # ── Evaluate ──
        model.eval()
        ndcg5_scores = []
        ndcg10_scores = []

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(device)
                scores = model(features).cpu().numpy().flatten()
                labels_np = labels.numpy().flatten()

                ndcg5_scores.append(ndcg_at_k(scores, labels_np, k=5))
                ndcg10_scores.append(ndcg_at_k(scores, labels_np, k=10))

        avg_loss = epoch_loss / len(dataloader)
        avg_ndcg5 = np.mean(ndcg5_scores)
        avg_ndcg10 = np.mean(ndcg10_scores)

        history["loss"].append(avg_loss)
        history["ndcg@5"].append(avg_ndcg5)
        history["ndcg@10"].append(avg_ndcg10)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | NDCG@5: {avg_ndcg5:.4f} | NDCG@10: {avg_ndcg10:.4f}")

        # Save best model
        if avg_ndcg10 > best_ndcg:
            best_ndcg = avg_ndcg10
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}.pt"))

    print(f"\n  Best NDCG@10: {best_ndcg:.4f}")
    print(f"  Model saved to: {save_dir}/{model_name}.pt")

    # Save history
    with open(os.path.join(save_dir, f"{model_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def train_both_models(data_path: str = "data/processed/rs_training_data.csv"):
    """
    Train both the engagement-based and agent-based rankers.
    This is the core experiment: same architecture, different objectives.
    """
    print("Loading training data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples across {df['query_group'].nunique()} query groups")

    # ── Train Model 1: Engagement-based RS (YouTube-like) ──
    engagement_model, engagement_history = train_ranker(
        df=df,
        label_col="engagement_label",
        model_name="ranker_engagement",
        epochs=100,
    )

    # ── Train Model 2: Agent-centric RS (our contribution) ──
    agent_model, agent_history = train_ranker(
        df=df,
        label_col="agent_label",
        model_name="ranker_agent",
        epochs=100,
    )

    # ── Compare: How different are the rankings? ──
    print(f"\n{'='*60}")
    print("COMPARISON: Engagement RS vs Agent RS")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engagement_model.eval()
    agent_model.eval()

    # Load normalization stats
    with open("models/ranker_engagement_norm_stats.json") as f:
        eng_stats = json.load(f)
    with open("models/ranker_agent_norm_stats.json") as f:
        agent_stats = json.load(f)

    for query_group in df["query_group"].unique():
        group_df = df[df["query_group"] == query_group].copy()

        # Normalize features
        for col in FEATURE_COLS:
            group_df[f"{col}_eng_norm"] = (group_df[col] - eng_stats["means"][col]) / eng_stats["stds"][col]
            group_df[f"{col}_agent_norm"] = (group_df[col] - agent_stats["means"][col]) / agent_stats["stds"][col]

        eng_features = torch.tensor(
            group_df[[f"{c}_eng_norm" for c in FEATURE_COLS]].values, dtype=torch.float32
        ).unsqueeze(0).to(device)
        agent_features = torch.tensor(
            group_df[[f"{c}_agent_norm" for c in FEATURE_COLS]].values, dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            eng_scores = engagement_model(eng_features).cpu().numpy().flatten()
            agent_scores = agent_model(agent_features).cpu().numpy().flatten()

        # Compare top-5 rankings
        eng_top5 = np.argsort(-eng_scores)[:5]
        agent_top5 = np.argsort(-agent_scores)[:5]

        overlap = len(set(eng_top5) & set(agent_top5))

        print(f"\n  Query: {query_group}")
        print(f"  Top-5 overlap: {overlap}/5 videos in common")
        print(f"  Engagement Top-5 avg clickbait: {group_df.iloc[eng_top5]['clickbait_score'].mean():.3f}")
        print(f"  Agent Top-5 avg clickbait:      {group_df.iloc[agent_top5]['clickbait_score'].mean():.3f}")
        print(f"  Engagement Top-5 avg credibility: {group_df.iloc[eng_top5]['credibility'].mean():.3f}")
        print(f"  Agent Top-5 avg credibility:      {group_df.iloc[agent_top5]['credibility'].mean():.3f}")


if __name__ == "__main__":
    train_both_models()
