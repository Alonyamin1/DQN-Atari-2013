"""
Experience Replay Buffer

WHY DO WE NEED THIS?
1. Neural networks expect i.i.d. (independent) data
2. But game frames are highly correlated (frame t looks like frame t+1)
3. Training on consecutive frames causes instability

SOLUTION: Store experiences and sample RANDOMLY
- This breaks the correlation
- The network sees diverse experiences each batch
- Much more stable learning!
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Stores transitions (state, action, reward, next_state, done)
    and provides random sampling for training.
    """

    def __init__(self, capacity):
        """
        Args:
            capacity: Maximum number of transitions to store.
                      When full, oldest transitions are removed.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state: Current state (4 stacked frames)
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.

        WHY RANDOM? To break correlation between samples.
        If we sampled consecutive frames, the network would
        overfit to recent experience and forget old lessons.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
            States are converted to float32 here (stored as uint8 to save memory)
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),      # Convert uint8 -> float32
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32), # Convert uint8 -> float32
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
