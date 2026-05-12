"""
Evaluate a trained DQN agent over multiple episodes.

USAGE:
    python evaluate.py --model results/final_run3/results/model_run3.pt --episodes 100
"""

import gymnasium as gym
import ale_py
import torch
import argparse
import numpy as np

gym.register_envs(ale_py)

from model import DQN
from preprocessing import FrameStack, AtariWrapper


def evaluate(model_path, episodes=100):
    """Evaluate the agent over multiple episodes."""

    import random

    # Use repeat_action_probability for stochastic evaluation (standard ALE setting)
    env = gym.make("QbertNoFrameskip-v4", repeat_action_probability=0.25)
    env = AtariWrapper(env, skip=4)

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DQN(n_actions).to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["q_network"])
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Evaluating over {episodes} episodes...")
    print(f"Device: {device}")
    print("-" * 40)

    frame_stack = FrameStack(k=4)
    rewards = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()

        # Random no-ops at start (1-30) for stochastic evaluation
        for _ in range(random.randint(1, 30)):
            obs, _, terminated, truncated, _ = env.step(0)  # NOOP action
            if terminated or truncated:
                obs, _ = env.reset()

        state = frame_stack.reset(obs)
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = frame_stack.step(obs)
            episode_reward += reward

        rewards.append(episode_reward)
        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes}: Reward = {episode_reward:.0f}, Running Avg = {np.mean(rewards):.2f}")

    env.close()

    print("-" * 40)
    print(f"Results over {episodes} episodes:")
    print(f"  Mean:   {np.mean(rewards):.2f}")
    print(f"  Std:    {np.std(rewards):.2f}")
    print(f"  Min:    {np.min(rewards):.0f}")
    print(f"  Max:    {np.max(rewards):.0f}")
    print(f"  Median: {np.median(rewards):.0f}")

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    args = parser.parse_args()

    evaluate(args.model, args.episodes)
