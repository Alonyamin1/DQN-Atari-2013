"""
Training Script for DQN with Target Network on Qbert (2015 Nature Paper)

Key difference from train.py:
- Uses DQNAgentWithTargetNetwork (separate target network)
- Adds target_update_freq hyperparameter (C=10,000 in paper)

USAGE:
    python train_WtargetNetwork.py --run_id 4

Use run_ids 4, 5, 6 to distinguish from the 2013 paper runs (1, 2, 3).
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import time
from datetime import datetime

# Register ALE environments
gym.register_envs(ale_py)

from dqn_agent_WtargetNetwork import DQNAgentWithTargetNetwork
from preprocessing import FrameStack, AtariWrapper


def make_logger(log_path):
    """Return a log() function that writes to both stdout and a file."""
    log_file = open(log_path, "a", buffering=1)

    def log(msg=""):
        print(msg)
        log_file.write(str(msg) + "\n")

    return log, log_file


def make_env():
    """Create the Qbert environment with proper wrappers."""
    env = gym.make("QbertNoFrameskip-v4")
    env = AtariWrapper(env, skip=4)
    return env


def collect_fixed_states(n_states=1000):
    """Collect fixed states using random policy for Q-value tracking."""
    env = make_env()
    frame_stack = FrameStack(k=4)
    states = []

    obs, _ = env.reset()
    state = frame_stack.reset(obs)

    for _ in range(n_states):
        states.append(state.copy())
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
            state = frame_stack.reset(obs)
        else:
            state = frame_stack.step(obs)

    env.close()
    return np.array(states)


def compute_avg_max_q(agent, fixed_states):
    """Compute average max Q-value over fixed states."""
    with torch.no_grad():
        states_tensor = torch.FloatTensor(fixed_states).to(agent.device)
        q_values = agent.q_network(states_tensor)
        max_q = q_values.max(dim=1)[0]
        return max_q.mean().item()


def evaluate(agent, frame_stack, n_episodes=5):
    """Evaluate the agent without exploration."""
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


def save_checkpoint(path, agent, step, eval_rewards, eval_steps, avg_max_q_values, losses, fixed_states):
    """Save training checkpoint for resuming later."""
    checkpoint = {
        "agent_state": {
            "q_network": agent.q_network.state_dict(),
            "target_network": agent.target_network.state_dict(),
            "optimizer": agent.optimizer.state_dict(),
            "epsilon": agent.epsilon,
            "steps": agent.steps,
        },
        "training_state": {
            "step": step,
            "eval_rewards": eval_rewards,
            "eval_steps": eval_steps,
            "avg_max_q_values": avg_max_q_values,
            "losses": losses,
        },
        "fixed_states": fixed_states,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, agent):
    """Load training checkpoint."""
    checkpoint = torch.load(path, weights_only=False)

    # Restore agent state
    agent.q_network.load_state_dict(checkpoint["agent_state"]["q_network"])
    agent.target_network.load_state_dict(checkpoint["agent_state"]["target_network"])
    agent.optimizer.load_state_dict(checkpoint["agent_state"]["optimizer"])
    agent.epsilon = checkpoint["agent_state"]["epsilon"]
    agent.steps = checkpoint["agent_state"]["steps"]

    return (
        checkpoint["training_state"]["step"],
        checkpoint["training_state"]["eval_rewards"],
        checkpoint["training_state"]["eval_steps"],
        checkpoint["training_state"]["avg_max_q_values"],
        checkpoint["training_state"].get("losses", []),
        checkpoint["fixed_states"],
    )


def train(
    total_steps=2500000,
    eval_freq=10000,
    eval_episodes=5,
    final_eval_episodes=100,
    run_id=4,
    save_dir="results",
    resume=False,
    learning_rate=0.00025,
    gamma=0.99,
    buffer_size=500000,
    batch_size=32,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=1000000,
    target_update_freq=10000,
):
    """
    Main training function with Target Network.

    Args:
        total_steps: Total number of environment steps
        eval_freq: Evaluate every N steps
        eval_episodes: Number of episodes per evaluation
        run_id: Identifier for this training run (use 4+ for target network runs)
        save_dir: Directory to save results
        resume: Whether to resume from checkpoint
        target_update_freq: Steps between target network updates (C in 2015 paper)
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = f"{save_dir}/checkpoint_run{run_id}.pt"
    log_path = f"{save_dir}/train_run{run_id}.log"

    log, log_file = make_logger(log_path)
    log("=" * 60)
    log(f"Training run {run_id} (WITH TARGET NETWORK) started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    env = make_env()
    n_actions = env.action_space.n
    log("Game: Qbert")
    log(f"Number of actions: {n_actions}")
    log("Architecture: DQN with TARGET NETWORK (2015 Nature Paper)")

    frame_stack = FrameStack(k=4)
    eval_frame_stack = FrameStack(k=4)

    hyperparams = dict(
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        target_update_freq=target_update_freq,
    )
    agent = DQNAgentWithTargetNetwork(n_actions=n_actions, device=device, **hyperparams)

    log("Hyperparameters:")
    for k, v in hyperparams.items():
        log(f"  {k}: {v}")
    log(f"  total_steps: {total_steps}")
    log(f"  eval_freq: {eval_freq}")
    log(f"  eval_episodes: {eval_episodes}")
    log(f"  final_eval_episodes: {final_eval_episodes}")

    start_step = 1
    if resume and os.path.exists(checkpoint_path):
        log(f"Resuming from checkpoint: {checkpoint_path}")
        start_step, eval_rewards, eval_steps, avg_max_q_values, losses, fixed_states = load_checkpoint(
            checkpoint_path, agent
        )
        start_step += 1
        log(f"Resumed at step {start_step}, epsilon={agent.epsilon:.4f}")
    else:
        log("Collecting fixed states for Q-value tracking...")
        fixed_states = collect_fixed_states(n_states=1000)
        log(f"Collected {len(fixed_states)} fixed states")

        eval_rewards = []
        eval_steps = []
        avg_max_q_values = []
        losses = []

    interval_losses = []
    interval_episode_rewards = []
    interval_start_time = time.time()
    current_episode_reward = 0

    obs, _ = env.reset()
    state = frame_stack.reset(obs)

    log("")
    log(f"Starting training run {run_id} (WITH TARGET NETWORK)...")
    log(f"Total steps: {total_steps} (starting from {start_step})")
    log(f"Evaluation every {eval_freq} steps")
    log(f"Target network update every {target_update_freq} steps")
    log("-" * 50)

    for step in range(start_step, total_steps + 1):
        action = agent.select_action(state, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = frame_stack.step(next_obs)
        agent.store_transition(state, action, reward, next_state, done)

        loss = agent.train_step()
        if loss is not None:
            interval_losses.append(loss)

        current_episode_reward += reward
        state = next_state

        if done:
            interval_episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            obs, _ = env.reset()
            state = frame_stack.reset(obs)

        if step % eval_freq == 0:
            avg_reward = evaluate(agent, eval_frame_stack, n_episodes=eval_episodes)
            avg_q = compute_avg_max_q(agent, fixed_states)
            eval_rewards.append(avg_reward)
            avg_max_q_values.append(avg_q)
            eval_steps.append(step)

            mean_loss = float(np.mean(interval_losses)) if interval_losses else 0.0
            losses.append(mean_loss)

            elapsed = time.time() - interval_start_time
            steps_per_sec = eval_freq / elapsed if elapsed > 0 else 0.0
            n_episodes = len(interval_episode_rewards)
            mean_train_reward = (
                float(np.mean(interval_episode_rewards)) if interval_episode_rewards else 0.0
            )

            log(
                f"Step {step:>7} | "
                f"Eval Reward: {avg_reward:>8.2f} | "
                f"Avg Max Q: {avg_q:>6.2f} | "
                f"Loss: {mean_loss:>8.4f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer):>7} | "
                f"Train Eps: {n_episodes:>3} | "
                f"Train R: {mean_train_reward:>7.2f} | "
                f"{steps_per_sec:>5.1f} steps/s"
            )

            np.save(f"{save_dir}/eval_rewards_run{run_id}.npy", eval_rewards)
            np.save(f"{save_dir}/eval_steps_run{run_id}.npy", eval_steps)
            np.save(f"{save_dir}/avg_max_q_run{run_id}.npy", avg_max_q_values)
            np.save(f"{save_dir}/losses_run{run_id}.npy", losses)

            save_checkpoint(
                checkpoint_path, agent, step, eval_rewards, eval_steps, avg_max_q_values, losses, fixed_states
            )

            interval_losses = []
            interval_episode_rewards = []
            interval_start_time = time.time()

    env.close()

    log("")
    log(f"Final evaluation ({final_eval_episodes} episodes)...")
    final_reward = evaluate(agent, eval_frame_stack, n_episodes=final_eval_episodes)
    log(f"Final average reward ({final_eval_episodes} episodes): {final_reward:.2f}")
    np.save(f"{save_dir}/final_reward_run{run_id}.npy", np.array([final_reward]))

    agent.save(f"{save_dir}/model_run{run_id}.pt")
    log(f"Model saved to {save_dir}/model_run{run_id}.pt")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(eval_steps, eval_rewards)
    axes[0].set_xlabel("Training Steps")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title(f"Average Reward on Qbert (Target Network) - Run {run_id}")
    axes[0].grid(True)

    axes[1].plot(eval_steps, avg_max_q_values)
    axes[1].set_xlabel("Training Steps")
    axes[1].set_ylabel("Average Max Q")
    axes[1].set_title(f"Average Q on Qbert (Target Network) - Run {run_id}")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curve_run{run_id}.png", dpi=150)
    plt.close()

    log("")
    log(f"Results saved to {save_dir}/")
    log(f"Run {run_id} finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_file.close()

    return final_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN with Target Network on Qbert")
    parser.add_argument("--run_id", type=int, default=4, help="Training run ID (4, 5, 6 for target network)")
    parser.add_argument("--steps", type=int, default=2500000, help="Total training steps")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--final_eval_episodes", type=int, default=100, help="Episodes for final evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--save_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay_steps", type=int, default=1000000)
    parser.add_argument("--target_update_freq", type=int, default=10000, help="Target network update frequency (C)")
    args = parser.parse_args()

    final_reward = train(
        total_steps=args.steps,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        final_eval_episodes=args.final_eval_episodes,
        run_id=args.run_id,
        save_dir=args.save_dir,
        resume=args.resume,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_update_freq=args.target_update_freq,
    )

    print(f"\n{'='*50}")
    print(f"Training Run {args.run_id} (WITH TARGET NETWORK) Complete!")
    print(f"Final Reward: {final_reward:.2f}")
    print(f"{'='*50}")
