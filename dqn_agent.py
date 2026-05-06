"""
DQN Agent (2013 Paper Implementation)

This is the brain of the system. It:
1. Decides which action to take (epsilon-greedy)
2. Stores experiences in replay buffer
3. Learns from random batches of experiences

Based on: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- Uses a single Q-network (no target network - that was added in 2015 Nature paper)
- No gradient clipping (not in original paper)
- Reward clipping to {-1, 0, +1}
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        n_actions,
        device,
        learning_rate=0.0001,
        gamma=0.99,
        buffer_size=100000,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100000,
    ):
        """
        Initialize the DQN Agent.

        Args:
            n_actions: Number of possible actions in the game
            device: 'cuda' or 'cpu'
            learning_rate: How fast the network learns (too high = unstable, too low = slow)
            gamma: Discount factor (0.99 means future rewards matter almost as much as immediate)
            buffer_size: How many experiences to remember
            batch_size: How many experiences to learn from at once
            epsilon_start: Initial exploration rate (1.0 = fully random)
            epsilon_end: Final exploration rate (0.01 = mostly greedy)
            epsilon_decay_steps: How many steps to decay epsilon over
        """
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Q-Network
        self.q_network = DQN(n_actions).to(device)

        # Optimizer - Paper uses RMSprop
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=learning_rate)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Step counter (for target network updates)
        self.steps = 0

    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.

        EPSILON-GREEDY EXPLAINED:
        - With probability epsilon: take random action (EXPLORE)
        - With probability 1-epsilon: take best action (EXPLOIT)

        Early in training: high epsilon = lots of exploration
        Later: low epsilon = mostly using what we learned

        Args:
            state: Current state (4, 84, 84)
            training: If False, use greedy policy (for evaluation)

        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.n_actions)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer with reward clipping."""
        # Paper: "clipped all positive rewards at 1 and all negative rewards at -1"
        # This means discrete {-1, 0, +1}, not a range
        if reward > 0:
            clipped_reward = 1.0
        elif reward < 0:
            clipped_reward = -1.0
        else:
            clipped_reward = 0.0
        self.replay_buffer.push(state, action, clipped_reward, next_state, done)

    def train_step(self):
        """
        Perform one training step.

        THE DQN UPDATE:
        1. Sample batch from replay buffer
        2. Compute current Q-values: Q(s, a)
        3. Compute target: r + gamma * max(Q_target(s', a'))
        4. Compute loss: (Q(s,a) - target)^2
        5. Backpropagate

        Returns:
            Loss value (for logging), or None if buffer too small
        """
        # Train as soon as we have enough samples for a batch
        # (2013 paper Algorithm 1 trains every step from the start)
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample random batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a)
        # We get Q-values for all actions, then select the ones for actions we took
        current_q_values = self.q_network(states)  # (batch, n_actions)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (2013 paper: use same Q-network, no separate target network)
        with torch.no_grad():
            # Get max Q-value for next states: max_a' Q(s', a')
            next_q_values = self.q_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]

            # Compute target: r + gamma * max_a' Q(s', a')
            # If done, there's no next state, so target is just r
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute loss
        loss = self.loss_fn(current_q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability (hyperparameter choice)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1)
        self.optimizer.step()

        # Update step counter
        self.steps += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        return loss.item()

    def save(self, path):
        """Save the model weights."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path):
        """Load the model weights."""
        checkpoint = torch.load(path, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
