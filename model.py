"""
Q-Network Architecture (from DQN paper)

WHY THIS ARCHITECTURE?
- CNNs are great for processing images (the game screen)
- The network takes 4 stacked frames as input (84x84x4)
- Output is Q-value for each possible action

The agent will pick the action with highest Q-value.
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network as described in the 2013 DQN paper.

    Input: 4 stacked grayscale frames (4, 84, 84)
    Output: Q-value for each action
    """

    def __init__(self, n_actions):
        super(DQN, self).__init__()

        # Convolutional layers - extract features from frames
        # Conv1: 16 filters, 8x8 kernel, stride 4
        # Input: (batch, 4, 84, 84) -> Output: (batch, 16, 20, 20)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)

        # Conv2: 32 filters, 4x4 kernel, stride 2
        # Input: (batch, 16, 20, 20) -> Output: (batch, 32, 9, 9)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        # Fully connected layers
        # 32 * 9 * 9 = 2592 input features
        self.fc1 = nn.Linear(32 * 9 * 9, 256)

        # Output layer: one Q-value per action
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape (batch, 4, 84, 84) - 4 stacked frames
               Values in range [0, 255]

        Returns:
            Q-values for each action, shape (batch, n_actions)
        """
        # Normalize pixel values to [0, 1]
        # Note: Not explicitly in 2013 paper, but common practice for neural networks
        x = x / 255.0

        # Convolutional layers with ReLU activation
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch, 2592)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)  # No activation - Q-values can be any real number

        return q_values
