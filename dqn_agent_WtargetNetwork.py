"""
DQN Agent with Target Network (2015 Nature Paper Implementation)

Key difference from 2013 paper:
- Uses a separate TARGET NETWORK for computing TD targets
- Target network is updated every C steps (frozen between updates)
- This provides stable targets and prevents oscillation

Based on: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgentWithTargetNetwork:
    def __init__(
        self,
        n_actions,
        device,
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
        Initialize the DQN Agent with Target Network.

        Args:
            n_actions: Number of possible actions in the game
            device: 'cuda' or 'cpu'
            learning_rate: How fast the network learns
            gamma: Discount factor (0.99 = future rewards matter)
            buffer_size: How many experiences to remember
            batch_size: How many experiences to learn from at once
            epsilon_start: Initial exploration rate (1.0 = fully random)
            epsilon_end: Final exploration rate
            epsilon_decay_steps: How many steps to decay epsilon over
            target_update_freq: Steps between target network updates (C in paper)
        """
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Online Q-Network (updated every step)
        self.q_network = DQN(n_actions).to(device)

        # Target Q-Network (frozen, updated every C steps)
        self.target_network = DQN(n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Always in eval mode, no gradients

        # Optimizer - Paper uses RMSprop
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=learning_rate)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Step counter
        self.steps = 0

    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state (4, 84, 84)
            training: If False, use greedy policy (for evaluation)

        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer with reward clipping."""
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

        KEY DIFFERENCE FROM 2013:
        - Target Q-values computed using TARGET network (frozen)
        - Target network updated every C steps

        Returns:
            Loss value (for logging), or None if buffer too small
        """
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

        # Current Q-values from ONLINE network: Q(s, a)
        current_q_values = self.q_network(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values from TARGET network (KEY CHANGE from 2013)
        with torch.no_grad():
            # Use TARGET network for next state Q-values
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]

            # Compute target: r + gamma * max_a' Q_target(s', a')
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        # Compute loss
        loss = self.loss_fn(current_q_values, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (can be more relaxed with target network)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()

        # Update step counter
        self.steps += 1

        # Update target network every C steps
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        return loss.item()

    def save(self, path):
        """Save the model weights."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
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
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]
