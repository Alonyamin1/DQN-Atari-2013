"""
Hyperparameter Tuning Script

This script tests different hyperparameter combinations to find the best settings.
Uses shorter training runs to quickly compare configurations.

USAGE:
    python hypertuning.py

After finding good hyperparameters, update train.py and run the full 3 training runs.
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import itertools
from datetime import datetime

# Register ALE environments
gym.register_envs(ale_py)

from dqn_agent import DQNAgent
from preprocessing import FrameStack, AtariWrapper


def make_env():
    env = gym.make("QbertNoFrameskip-v4")
    env = AtariWrapper(env, skip=4)
    return env


def evaluate(agent, frame_stack, n_episodes=3):
    """Quick evaluation."""
    env = make_env()
    total_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = frame_stack.step(obs)
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()
    return np.mean(total_rewards)


def train_with_config(config, tuning_steps=100000, eval_freq=10000):
    """
    Train with a specific configuration and return final performance.

    Args:
        config: Dictionary of hyperparameters
        tuning_steps: Shorter training for quick comparison
        eval_freq: Evaluation frequency

    Returns:
        Final evaluation reward
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()
    n_actions = env.action_space.n

    frame_stack = FrameStack(k=4)
    eval_frame_stack = FrameStack(k=4)

    agent = DQNAgent(
        n_actions=n_actions,
        device=device,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        epsilon_start=1.0,
        epsilon_end=config["epsilon_end"],
        epsilon_decay_steps=config["epsilon_decay_steps"],
    )

    eval_rewards = []
    obs, _ = env.reset()
    state = frame_stack.reset(obs)

    for step in range(1, tuning_steps + 1):
        action = agent.select_action(state, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = frame_stack.step(next_obs)

        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state

        if done:
            obs, _ = env.reset()
            state = frame_stack.reset(obs)

        if step % eval_freq == 0:
            avg_reward = evaluate(agent, eval_frame_stack, n_episodes=3)
            eval_rewards.append(avg_reward)

    env.close()

    # Return best reward achieved (not just final)
    return max(eval_rewards) if eval_rewards else 0, eval_rewards


def run_hypertuning():
    """
    Test different hyperparameter combinations.
    """
    # Define hyperparameter search space
    # Based on paper values, testing variations
    # Note: 2013 paper uses single Q-network (no target network)
    param_grid = {
        "learning_rate": [0.00025, 0.0001, 0.00005],  # Paper: 0.00025
        "batch_size": [32, 64],                        # Paper: 32
        "epsilon_end": [0.1, 0.05],                    # Paper: 0.1
        "epsilon_decay_steps": [500000, 1000000],      # Paper: 1000000
        "gamma": [0.99],                               # Paper: 0.99
        "buffer_size": [100000],                       # Paper: 1000000 (reduced)
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Testing {len(combinations)} hyperparameter combinations")
    print("=" * 60)

    results = []
    best_reward = float("-inf")
    best_config = None

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))

        print(f"\n[{i+1}/{len(combinations)}] Testing config:")
        print(f"  lr={config['learning_rate']}, batch={config['batch_size']}, "
              f"eps_end={config['epsilon_end']}, eps_decay={config['epsilon_decay_steps']}")

        try:
            best_eval, eval_history = train_with_config(
                config,
                tuning_steps=100000,  # Shorter runs for tuning
                eval_freq=10000
            )

            results.append({
                "config": config,
                "best_reward": best_eval,
                "history": eval_history
            })

            print(f"  Best reward: {best_eval:.2f}")

            if best_eval > best_reward:
                best_reward = best_eval
                best_config = config
                print(f"  *** NEW BEST! ***")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Print final results
    print("\n" + "=" * 60)
    print("HYPERTUNING RESULTS")
    print("=" * 60)

    # Sort by reward
    results.sort(key=lambda x: x["best_reward"], reverse=True)

    print("\nTop 5 configurations:")
    for i, r in enumerate(results[:5]):
        print(f"\n{i+1}. Reward: {r['best_reward']:.2f}")
        for k, v in r["config"].items():
            print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION (copy to train.py):")
    print("=" * 60)
    if best_config:
        print(f"""
agent = DQNAgent(
    n_actions=n_actions,
    device=device,
    learning_rate={best_config['learning_rate']},
    gamma={best_config['gamma']},
    buffer_size={best_config['buffer_size']},
    batch_size={best_config['batch_size']},
    epsilon_start=1.0,
    epsilon_end={best_config['epsilon_end']},
    epsilon_decay_steps={best_config['epsilon_decay_steps']},
)
""")

    # Save results
    os.makedirs("tuning_results", exist_ok=True)

    # Plot top configs
    plt.figure(figsize=(12, 6))
    for i, r in enumerate(results[:5]):
        steps = [(j+1) * 10000 for j in range(len(r["history"]))]
        label = f"lr={r['config']['learning_rate']}, batch={r['config']['batch_size']}"
        plt.plot(steps, r["history"], label=label, linewidth=2)

    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Reward")
    plt.title("Hyperparameter Tuning - Top 5 Configurations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("tuning_results/tuning_comparison.png", dpi=150)
    plt.close()

    print("\nPlot saved to tuning_results/tuning_comparison.png")

    return best_config


if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    best = run_hypertuning()
