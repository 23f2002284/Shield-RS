import pandas as pd
from pathlib import Path
import json

def generate_comparison_table():
    print("============================================================")
    print("  SHIELD: FINAL EXPERIMENT RESULTS COMPARISON")
    print("============================================================\n")

    exp_dir = Path("experiments")

    # 1. Ranking Models (MF, CF, BPR, Two-Tower)
    print("### 1. TOP-K RANKING MODELS")
    print("Task: Recommend 10 items a user is likely to enjoy.")
    print("-" * 65)
    print(f"{'Model':<15} | {'HR@10':>6} | {'HR@20':>6} | {'NDCG@10':>7} | {'Train Time':>10}")
    print("-" * 65)

    ranking_rows = []

    # MF models
    if (exp_dir / "mf_evaluation.csv").exists():
        df = pd.read_csv(exp_dir / "mf_evaluation.csv")
        for _, row in df.iterrows():
            ranking_rows.append((row['model'], row['hr@10'], row['hr@20'], row['ndcg@10'], row['train_time_sec']))

    # BPR
    if (exp_dir / "bpr_evaluation.csv").exists():
        df = pd.read_csv(exp_dir / "bpr_evaluation.csv")
        ranking_rows.append(("BPR", df['hr@10'].mean(), df['hr@20'].mean(), df['ndcg@10'].mean(), 9.0)) # approx time

    # CF
    if (exp_dir / "cf_evaluation.csv").exists():
        df = pd.read_csv(exp_dir / "cf_evaluation.csv")
        for _, row in df.iterrows():
            ranking_rows.append((row['model'], row['hr@10'], row['hr@20'], row['ndcg@10'], 0.1))

    # Two-Tower
    if (exp_dir / "two_tower_evaluation.csv").exists():
        df = pd.read_csv(exp_dir / "two_tower_evaluation.csv")
        ranking_rows.append(("Two-Tower", df['hr@10'].mean(), df['hr@20'].mean(), df['ndcg@10'].mean(), 13.0))

    ranking_rows.sort(key=lambda x: x[2], reverse=True) # sort by HR@20

    for name, hr10, hr20, ndcg, time in ranking_rows:
        print(f"{name:<15} | {hr10:6.3f} | {hr20:6.3f} | {ndcg:7.3f} | {time:6.1f}s")


    # 2. Sequential Models
    print("\n### 2. SEQUENTIAL MODELS")
    print("Task: Predict the EXACT next video in a user's sequence.")
    print("-" * 60)
    print(f"{'Model':<15} | {'HR@5':>6} | {'HR@10':>6} | {'HR@20':>6} | {'MRR@10':>7}")
    print("-" * 60)
    if (exp_dir / "sequential_evaluation.csv").exists():
        df = pd.read_csv(exp_dir / "sequential_evaluation.csv")
        df = df.sort_values(by="hr@20", ascending=False)
        for _, row in df.iterrows():
            print(f"{row['model']:<15} | {row['hr@5']:6.3f} | {row['hr@10']:6.3f} | {row['hr@20']:6.3f} | {row['mrr@10']:7.3f}")


    # 3. Cold-Start Bandits
    print("\n### 3. COLD-START BANDITS")
    print("Task: Maximize Agent-Centric Reward for new users.")
    print("Reward = 0.35*InfoDensity + 0.35*Credibility - 0.2*Clickbait - 0.1*Manipulation")
    print("-" * 65)
    print(f"{'Bandit':<20} | {'Avg Reward':>12} | {'Total Regret':>12}")
    print("-" * 65)

    bandit_rows = []
    # UCB1
    if (exp_dir / "mab_ucb1.csv").exists():
        df = pd.read_csv(exp_dir / "mab_ucb1.csv")
        last = df.iloc[-1]
        bandit_rows.append(("UCB1", last['avg_reward'], last['cumulative_regret']))
    # Thompson
    if (exp_dir / "mab_thompson.csv").exists():
        df = pd.read_csv(exp_dir / "mab_thompson.csv")
        last = df.iloc[-1]
        bandit_rows.append(("Thompson Sampling", last['avg_reward'], last['cumulative_regret']))
    # LinUCB
    if (exp_dir / "mab_linucb.csv").exists():
        df = pd.read_csv(exp_dir / "mab_linucb.csv")
        last = df.iloc[-1]
        bandit_rows.append(("LinUCB (Contextual)", last['avg_reward'], last['cumulative_regret']))

    bandit_rows.sort(key=lambda x: x[1], reverse=True) # Sort by avg reward
    for name, rwd, reg in bandit_rows:
        print(f"{name:<20} | {rwd:12.4f} | {reg:12.4f}")


    # 4. Causal Debiasing (The Core Thesis)
    print("\n### 4. CAUSAL DEBIASING (Shield Thesis Proof)")
    print("Hypothesis: DR-Debiasing reduces average view_count and increases quality.")
    print("-" * 75)
    if (exp_dir / "causal_bias_analysis.csv").exists():
        df = pd.read_csv(exp_dir / "causal_bias_analysis.csv")
        print(f"Results averaged over {len(df)} sample users:")
        print(f"  Avg View Count      : Biased = {df['biased_avg_view_count'].mean():10.0f} | Debiased = {df['debiased_avg_view_count'].mean():10.0f}  (Delta: {df['delta_view_count'].mean():+8.0f})")
        print(f"  Avg Credibility     : Biased = {df['biased_avg_credibility'].mean():10.3f} | Debiased = {df['debiased_avg_credibility'].mean():10.3f}  (Delta: {df['delta_credibility'].mean():+8.3f})")
        print(f"  Avg Info Density    : Biased = {df['biased_avg_info_density'].mean():10.3f} | Debiased = {df['debiased_avg_info_density'].mean():10.3f}  (Delta: {df['delta_info_density'].mean():+8.3f})")
        print(f"  Avg Clickbait       : Biased = {df['biased_avg_clickbait'].mean():10.3f} | Debiased = {df['debiased_avg_clickbait'].mean():10.3f}  (Delta: {df['delta_clickbait'].mean():+8.3f})")

        print("\nRanking Impact:")
        if (exp_dir / "causal_evaluation.csv").exists():
            eval_df = pd.read_csv(exp_dir / "causal_evaluation.csv")
            naive = eval_df[eval_df['model'] == 'Naive ALS']['ndcg@10'].values[0]
            dr = eval_df[eval_df['model'] == 'DR-Debiased ALS']['ndcg@10'].values[0]
            print(f"  NDCG@10 Naive ALS    : {naive:.4f}")
            print(f"  NDCG@10 DR-Debiased  : {dr:.4f}  (Delta: {dr-naive:+.4f})")
    else:
        print("  Causal analysis results not found. Run causal_rs.py first.")

    print("\n============================================================")

if __name__ == "__main__":
    generate_comparison_table()
