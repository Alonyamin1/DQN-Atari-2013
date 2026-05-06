"""
Plot Results from All 3 Training Runs

Run this after completing all 3 training runs to:
1. Generate individual learning curves
2. Generate combined plot
3. Compute final statistics for the report

USAGE:
    python plot_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def load_results(save_dir="results"):
    """Load results from all 3 runs."""
    all_rewards = []
    all_steps = []

    for run_id in [1, 2, 3]:
        rewards_path = f"{save_dir}/eval_rewards_run{run_id}.npy"
        steps_path = f"{save_dir}/eval_steps_run{run_id}.npy"

        if os.path.exists(rewards_path) and os.path.exists(steps_path):
            rewards = np.load(rewards_path)
            steps = np.load(steps_path)
            all_rewards.append(rewards)
            all_steps.append(steps)
            print(f"Loaded run {run_id}: {len(rewards)} evaluations")
        else:
            print(f"Run {run_id} not found")

    return all_rewards, all_steps


def plot_all_runs(all_rewards, all_steps, save_dir="results"):
    """Plot all 3 runs on the same graph."""
    plt.figure(figsize=(12, 7))

    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    labels = ["Run 1", "Run 2", "Run 3"]

    for i, (rewards, steps) in enumerate(zip(all_rewards, all_steps)):
        plt.plot(steps, rewards, color=colors[i], label=labels[i], linewidth=2)

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Evaluation Reward", fontsize=12)
    plt.title("DQN on Robotank - All Training Runs", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_runs_combined.png", dpi=150)
    plt.close()
    print(f"Saved combined plot to {save_dir}/all_runs_combined.png")


def compute_statistics(all_rewards):
    """Compute final statistics for the report."""
    final_rewards = []

    print("\n" + "=" * 50)
    print("FINAL RESULTS FOR REPORT")
    print("=" * 50)

    for i, rewards in enumerate(all_rewards):
        final = rewards[-1]  # Last evaluation
        final_rewards.append(final)
        print(f"Run {i+1} Final Reward: {final:.2f}")

    print("-" * 50)
    print(f"Average across 3 runs: {np.mean(final_rewards):.2f}")
    print(f"Standard deviation: {np.std(final_rewards):.2f}")
    print("=" * 50)

    return final_rewards


if __name__ == "__main__":
    all_rewards, all_steps = load_results()

    if len(all_rewards) > 0:
        plot_all_runs(all_rewards, all_steps)
        compute_statistics(all_rewards)
    else:
        print("No results found. Make sure to run training first.")
