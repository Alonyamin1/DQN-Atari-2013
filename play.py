"""
Watch a trained DQN agent play Q*bert.

USAGE:
    python play.py --model results/final_runs/results/model_run1.pt
    python play.py --model results/final_runs/results/model_run1.pt --episodes 5
"""

import gymnasium as gym
import ale_py
import torch
import argparse
import time

gym.register_envs(ale_py)

from model import DQN
from preprocessing import FrameStack, AtariWrapper


def play(model_path, episodes=3, delay=0.02):
    """Watch the agent play."""

    # Create environment with rendering
    env = gym.make("QbertNoFrameskip-v4", render_mode="human")
    env = AtariWrapper(env, skip=4)

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DQN(n_actions).to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["q_network"])
    model.eval()

    print(f"Loaded model from {model_path}")
    print(f"Playing {episodes} episodes...")
    print(f"Device: {device}")
    print("-" * 40)

    frame_stack = FrameStack(k=4)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action (greedy)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = frame_stack.step(obs)
            episode_reward += reward
            steps += 1

            time.sleep(delay)

        print(f"Episode {ep}: Reward = {episode_reward:.0f}, Steps = {steps}")

    env.close()
    print("-" * 40)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained DQN agent play")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between frames")
    args = parser.parse_args()

    play(args.model, args.episodes, args.delay)
