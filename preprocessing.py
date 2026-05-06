"""
Atari Preprocessing (as specified in 2013 DQN paper)

From the paper:
1. RGB to grayscale, downsample to 110x84, crop to 84x84
2. Stack 4 frames
3. Frame skip: repeat action for k frames (k=4)

WHY FRAME SKIP?
- Atari runs at 60 FPS - too many decisions per second
- Repeat each action 4 times, reduces to ~15 decisions/second
- Also speeds up training significantly
"""

import cv2
import numpy as np
from collections import deque
import gymnasium as gym


def preprocess_frame(frame):
    """
    Preprocess a single Atari frame (EXACTLY as in DQN paper).

    Steps from paper:
    1. Convert RGB to grayscale
    2. Downsample to 110x84 (maintains aspect ratio better)
    3. Crop to 84x84 (removes score/lives UI at top/bottom)

    Args:
        frame: Raw RGB frame from Atari (210, 160, 3)

    Returns:
        Processed frame (84, 84), values 0-255
    """
    # Step 1: Convert RGB to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Step 2: Downsample to 110x84
    # Width 84, Height 110 (note: cv2.resize takes (width, height))
    resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)

    # Step 3: Crop to 84x84 - take the middle/bottom portion
    # This removes the score display at the top
    # Crop from row 18 to 102 (110 - 84 = 26, we take 18 from top, 8 from bottom)
    cropped = resized[18:102, :]  # shape: (84, 84)

    return cropped


class FrameStack:
    """
    Maintains a stack of the last 4 frames.

    WHY 4 FRAMES?
    - The paper uses 4 frames
    - This gives the network temporal information
    - It can detect velocity, direction of movement, etc.
    """

    def __init__(self, k=4):
        """
        Args:
            k: Number of frames to stack (default 4)
        """
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame):
        """
        Reset the frame stack with a new initial frame.
        Called at the start of each episode.

        We fill the stack with copies of the first frame
        (since we don't have previous frames yet).
        """
        processed = preprocess_frame(frame)
        for _ in range(self.k):
            self.frames.append(processed)
        return self._get_state()

    def step(self, frame):
        """
        Add a new frame to the stack.

        The oldest frame is automatically removed (deque maxlen).
        """
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self._get_state()

    def _get_state(self):
        """
        Return the stacked frames as a numpy array.

        Returns:
            Array of shape (4, 84, 84), values 0-255 as uint8
            (saves memory for large replay buffers - 1M states)
            (conversion to float32 and normalization done in model.py)
        """
        state = np.array(self.frames, dtype=np.uint8)
        return state


class AtariWrapper(gym.Wrapper):
    """
    Wrapper that implements frame skipping as in 2013 DQN paper.

    From paper:
    - Repeat each action for k frames (frame skip)
    - "the agent sees and selects actions on every k-th frame instead
       of every frame, and its last action is repeated on skipped frames"
    """

    def __init__(self, env, skip=4):
        """
        Args:
            env: The Atari environment
            skip: Number of frames to skip (repeat action)
        """
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        """
        Repeat action for 'skip' frames and return the last frame.
        Rewards are summed across all skipped frames.
        """
        total_reward = 0.0
        done = False
        obs = None

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset and return initial observation."""
        return self.env.reset(**kwargs)
