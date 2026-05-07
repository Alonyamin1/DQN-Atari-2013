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

- **Every 10,000 steps:** Run 5 evaluation episodes (for fast checkpoints during training)
- **Final reported result:** 30 evaluation episodes (for stable score reporting)
- **Evaluation policy:** Greedy (epsilon=0) — no exploration during evaluation
- **Rewards:** Unclipped (real game scores)
- **Metrics tracked:**
  - Average episode reward (noisy, as paper notes)
  - Average max Q over fixed states (smoother, per paper Section 5.1)

**Note:** Standard practice in the literature is 100 episodes. We use 30 for the final result as a balance between statistical stability and compute time. This is clearly documented per assignment requirements.

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

## Kaggle Training Environment

### Environment Specs (First Training Run)
- **GPU:** Tesla T4, 15.6 GB VRAM
- **RAM:** 33.7 GB
- **CPU:** 4 cores
- **Disk:** 20.9 GB free
- **Training:** 3 runs of 2,500,000 steps (= 10M frames with frame skip k=4)

### Memory Crash on Kaggle

**Problem:** When we moved training from our local PC to Kaggle, the notebook crashed with "out of memory" error before training even started.

**Root Cause:** `preprocessing.py` stored frame stacks as float32 (4 bytes per pixel). With a 1,000,000 transition replay buffer storing 2 states per transition (state + next_state), each with 4 frames of 84×84 pixels:

```
Memory per state = 4 frames × 84 × 84 pixels × 4 bytes = 112,896 bytes
Memory per transition = 2 states × 112,896 bytes = 225,792 bytes
Total buffer memory = 1,000,000 × 225,792 = ~225 GB... wait, let me recalculate:
  - 2 states × 4 frames × 84 × 84 × 4 bytes × 1M = ~56 GB
```

This exceeded Kaggle's 33.7 GB RAM limit.

**Fix:** Changed `preprocessing.py` to store frames as uint8 (1 byte per pixel) instead of float32:
```python
state = np.array(self.frames, dtype=np.uint8)  # Changed from float32
```

This reduced buffer memory from ~56 GB to ~14 GB. The conversion to float32 still happens at sampling time in the agent, so training is mathematically identical.

### Hyperparameters for First Kaggle Run

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.00025 | Same as paper |
| Gamma | 0.99 | Same as paper |
| Buffer size | 1,000,000 | Full paper size |
| Batch size | 32 | Same as paper |
| Epsilon start | 1.0 | Same as paper |
| Epsilon end | 0.1 | Same as paper |
| Epsilon decay steps | 1,000,000 | Same as paper |
| Total steps | 2,500,000 | = 10M frames / 4 (frame skip) |
| Number of runs | 3 | For averaging results |
| Optimizer | RMSprop | Same as paper |
| Gradient clipping | max_norm=1 | For stability |
| Target network | No | Following 2013 paper |

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

## Kaggle Cloud Training

### Why We Moved to Kaggle

Local PC had insufficient RAM and CPU for training with the paper's full hyperparameters, so we moved to Kaggle's free cloud GPUs.

### Kaggle Environment Specs

| Resource | Value |
|----------|-------|
| GPU | Tesla T4, 15.6 GB VRAM |
| RAM | 30 GB |
| CPU | 4 cores |
| Disk | 20.9 GB free |
| Session limit | 12 hours |

### Problem 1 — OOM Crash (float32 buffer)

**Symptom:** The notebook crashed immediately with "out of memory" before training even started.

**Root Cause:** `preprocessing.py` stored frames as float32 (4 bytes/pixel). With 1M buffer × 2 states × 4 frames × 84×84 pixels:
```
Memory = 1,000,000 × 2 × 4 × 84 × 84 × 4 bytes = ~56 GB RAM required
```

This exceeded Kaggle's 30 GB RAM limit.

**Fix:** Changed `preprocessing.py` to store frames as uint8 (1 byte/pixel), reducing buffer memory to ~14 GB:
```python
state = np.array(self.frames, dtype=np.uint8)  # Changed from float32
```

### Problem 2 — OOM Crash at Step 810,000

**Symptom:** Even with the uint8 fix, training crashed at step 810,000 with another OOM error.

**Impact:** The checkpoint at step 810,000 was corrupted mid-save and could not be loaded. We lost that run entirely.

**Root Cause:** 14 GB buffer + other process memory still pushed close to the 30 GB limit.

### Final Solution

Reduced replay buffer from 1,000,000 to **500,000 transitions** (~7 GB RAM) — well within the 30 GB limit.

| Buffer Size | Memory | Status |
|-------------|--------|--------|
| 1,000,000 (paper) | ~14 GB | OOM at 810k steps |
| 500,000 (final) | ~7 GB | Stable |
| 100,000 (local) | ~1.4 GB | Too small |

**Trade-off:** 500,000 is half the paper value but 5x what we used locally. This provides a good balance between experience diversity and memory safety.

### Run 1 Results

| Metric | Value |
|--------|-------|
| Final reward | 425.00 (average over 30 evaluation episodes) |
| Total steps | 2,500,000 |
| Buffer size | 500,000 |

**Observations:**
- Reward was noisy throughout training — many 0.00 evaluations mixed with peaks of 700-800
- Minimum score (163.9) ✅ beaten consistently
- Good grade score (613.5) ✅ hit several times but not consistently
- Bonus score (10,596) ❌ not reached — expected since we use 2013 paper (no target network). The 2013 paper itself scored ~1,952 on Q*bert

### Decisions for Run 2

Based on Run 1's reward instability, we adjusted hyperparameters:

| Parameter | Run 1 | Run 2 | Reason |
|-----------|-------|-------|--------|
| epsilon_end | 0.1 | 0.05 | More exploitation |
| epsilon_decay_steps | 1,000,000 | 500,000 | Faster decay, agent exploits sooner |
| learning_rate | 0.00025 | 0.0001 | More stable updates, less oscillation |
| buffer_size | 500,000 | 600,000 | More experience diversity |

**Note on buffer size:** Risky increase — crashed at ~810k previously with 1M buffer. 600k is estimated to be within safe range (~8.4 GB).

**Goal:** Reduce the reward instability seen in Run 1 and push average reward closer to 613.5.

---

## Lessons Learned

1. **DQN without target network is unstable** - The 2013 paper's claim of "no divergence" may have been specific to their exact setup or lucky random seeds.

2. **Game choice matters** - Stick to games tested in the original paper for best results.

3. **Memory is a bottleneck** - uint8 storage is essential for large replay buffers.

4. **Gradient clipping helps** - Even if not in the paper, it's a practical necessity for stability.

5. **Pixel normalization is important** - Neural networks train better with normalized inputs.
