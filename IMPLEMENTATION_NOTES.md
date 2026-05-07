# DQN Implementation Notes

## Overview
This document describes the implementation choices, challenges encountered, and solutions applied while implementing the DQN algorithm from the 2013 paper "Playing Atari with Deep Reinforcement Learning" (Mnih et al.).

---

## Game Selection

### Initial Choice: Robotank
We initially chose Robotank as our training environment.

### Problem: Q-Value Explosion
During testing, we observed severe Q-value instability:
```
Step    2000 | Avg Max Q: 414222.16
Step    4000 | Avg Max Q: 1557277.75
Step   10000 | Avg Max Q: 1963736.25
```
Q-values reached millions, indicating divergence.

### Root Cause
The 2013 paper uses a **single Q-network** without a target network. This is known to be unstable with function approximation (neural networks). The 2015 Nature paper added target networks specifically to fix this issue.

Additionally, **Robotank was not one of the 7 games tested in the 2013 paper**. The paper tested: Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders.

### Solution: Switched to Q*bert
We switched to **Q*bert** because:
1. It was one of the original 7 games in the 2013 paper
2. The authors confirmed stable training on this game
3. Score targets: minimum=163.9, good grade=613.5, bonus=10596

---

## Stability Modifications

### 1. Gradient Clipping
**Problem:** Without gradient clipping, large TD-errors caused exploding gradients.

**Solution:** Added gradient clipping as a hyperparameter:
```python
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1)
```

**Justification:** While not explicitly in the 2013 paper, gradient clipping is a common training technique that can be considered a hyperparameter choice. The assignment allows modifying hyperparameters.

### 2. Pixel Normalization
**Problem:** Raw pixel values (0-255) can cause large activations and unstable gradients.

**Solution:** Normalize pixels to [0, 1] in the model's forward pass:
```python
def forward(self, x):
    x = x / 255.0  # Normalize to [0, 1]
    ...
```

**Note:** This is not explicitly mentioned in the 2013 paper but is common practice for neural network training.

---

## Memory Optimization

### Problem: 1 Million Buffer Memory Requirements
The paper uses a replay buffer of 1 million transitions. Memory requirement:
- Each state: 4 frames × 84 × 84 pixels = 28,224 values
- Each transition stores: state + next_state + action + reward + done
- With float32: ~56 KB per transition → 56 GB for 1M transitions
- With uint8: ~14 KB per transition → 14 GB for 1M transitions

### Solution: uint8 Storage

**Store states as uint8 instead of float32:**
```python
# preprocessing.py
state = np.array(self.frames, dtype=np.uint8)  # 1 byte per pixel

# replay_buffer.py - convert to float32 only when sampling
np.array(states, dtype=np.float32)
```

Memory savings: 4x reduction (uint8 vs float32)

**For local PC (16 GB RAM):** buffer_size=100,000 (~1.4 GB)
**For Kaggle (30 GB RAM):** buffer_size=1,000,000 (~14 GB) - full paper size

---

## Hyperparameters

### Paper Values vs Our Values

| Parameter | Paper (2013) | Local PC | Kaggle | Reason |
|-----------|--------------|----------|--------|--------|
| Buffer size | 1,000,000 | 100,000 | 1,000,000 | Memory constraints |
| Training steps | 10M frames | 1M steps | 2.5M steps | Time/resource constraints |
| Batch size | 32 | 32 | 32 | Same as paper |
| Learning rate | 0.00025 | 0.00025 | 0.00025 | Same as paper |
| Gamma | 0.99 | 0.99 | 0.99 | Same as paper |
| Epsilon start | 1.0 | 1.0 | 1.0 | Same as paper |
| Epsilon end | 0.1 | 0.1 | 0.1 | Same as paper |
| Epsilon decay | 1M frames | 1M frames | 1M frames | Same as paper |
| Optimizer | RMSprop | RMSprop | RMSprop | Same as paper |
| Gradient clipping | Not mentioned | max_norm=1 | max_norm=1 | Stability |
| Target network | No | No | No | Following 2013 paper |

---

## Architecture (Exactly as Paper)

```
Input: 4 × 84 × 84 (4 stacked grayscale frames)
    ↓
Conv1: 16 filters, 8×8, stride 4, ReLU
    ↓
Conv2: 32 filters, 4×4, stride 2, ReLU
    ↓
FC1: 256 units, ReLU
    ↓
Output: n_actions Q-values (6 for Q*bert)
```

---

## Preprocessing (Exactly as Paper)

1. **RGB to Grayscale**
2. **Resize to 110×84**
3. **Crop to 84×84** (removes score display)
4. **Stack 4 frames** (temporal information)
5. **Frame skip k=4** (repeat action, sum rewards)

---

## Training Process

1. **Collect transitions** with epsilon-greedy policy
2. **Store in replay buffer** (rewards clipped to {-1, 0, +1})
3. **Sample random minibatch** (size 32)
4. **Compute TD target:** y = r + γ × max_a' Q(s', a')
5. **Compute loss:** MSE(Q(s,a), y)
6. **Backpropagate** with gradient clipping
7. **Update weights** with RMSprop

Training starts immediately after batch_size samples (per Algorithm 1 in paper).

---

## Evaluation

- **Every 10,000 steps:** Run evaluation episodes
- **Metrics tracked:**
  - Average episode reward (noisy, as paper notes)
  - Average max Q over fixed states (smoother, per paper Section 5.1)
- **Evaluation policy:** Greedy (epsilon=0)
- **Rewards:** Unclipped (real game scores)

---

## Files Structure

```
├── dqn_agent.py      # DQN agent with epsilon-greedy, replay, training
├── model.py          # CNN architecture (Q-network)
├── replay_buffer.py  # Experience replay with uint8 storage
├── preprocessing.py  # Frame preprocessing and stacking
├── train.py          # Main training script
├── hypertuning.py    # Hyperparameter search
└── plot_results.py   # Visualization of results
```

---

## System Requirements

### Local PC
- GPU: NVIDIA RTX 3060 Laptop (6 GB VRAM)
- RAM: 16 GB
- Buffer: 100,000 transitions (~1.4 GB)
- Training time: ~2-3 hours per run (1M steps)

### Kaggle (Recommended for full paper parameters)
- GPU: Tesla P100 or T4 (16 GB VRAM)
- RAM: ~30 GB
- Buffer: 1,000,000 transitions (~14 GB) - full paper size
- Training time: ~6-8 hours per run (2.5M steps)

**Resource usage:**
- VRAM: ~500 MB (model is small)
- Disk: ~50 MB per checkpoint

---

## Resume Capability

Training can be interrupted and resumed:
```bash
# Start training
python train.py --run_id 1 --steps 1000000

# If interrupted, resume with:
python train.py --run_id 1 --steps 1000000 --resume
```

**What is saved in checkpoint:**
- Model weights (Q-network)
- Optimizer state
- Epsilon value
- Training step
- Evaluation history
- Fixed states for Q-tracking

**What is NOT saved:**
- Replay buffer (too large ~5.5 GB)
- On resume, buffer starts fresh but agent retains learned weights

---

## Lessons Learned

1. **DQN without target network is unstable** - The 2013 paper's claim of "no divergence" may have been specific to their exact setup or lucky random seeds.

2. **Game choice matters** - Stick to games tested in the original paper for best results.

3. **Memory is a bottleneck** - uint8 storage is essential for large replay buffers.

4. **Gradient clipping helps** - Even if not in the paper, it's a practical necessity for stability.

5. **Pixel normalization is important** - Neural networks train better with normalized inputs.
