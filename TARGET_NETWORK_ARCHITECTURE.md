# Target Network Architecture (2015 Nature DQN)

## Motivation

After completing 3 training runs with the 2013 DQN architecture (no target network), our best results were:
- Max evaluation reward: 1150
- Average evaluation reward: ~350-450
- High variance with many 0-reward episodes

The **bonus score of 10,596** requires the 2015 Nature DQN improvements. Initially, we believed we were restricted to the 2013 paper architecture. However, we later understood that **the 2013 paper was provided as background context**, and only the **preprocessing steps are mandatory** (grayscale, resize to 84x84, frame stacking). The network architecture and training algorithm can be modified.

---

## Key Difference: Target Network

### 2013 Paper (Current Implementation)
```
Target = r + γ * max_a' Q(s', a')  ← uses SAME network for both
```

**Problem:** The Q-values we're trying to match are computed by the same network we're updating. This creates a "moving target" that causes instability and oscillation.

### 2015 Nature Paper (Proposed)
```
Target = r + γ * max_a' Q_target(s', a')  ← uses SEPARATE frozen network
```

**Solution:** Use a separate "target network" that is updated less frequently. This provides stable targets during training.

---

## Implementation Changes

### 1. Add Target Network to Agent (`dqn_agent.py`)

```python
class DQNAgent:
    def __init__(self, ...):
        # Q-Network (online network - updated every step)
        self.q_network = DQN(n_actions).to(device)
        
        # Target Network (frozen copy - updated every C steps)
        self.target_network = DQN(n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Always in eval mode
        
        # Target network update frequency
        self.target_update_freq = 10000  # Paper value: C = 10,000 steps
```

### 2. Modify Training Step

```python
def train_step(self):
    # ... (sampling same as before)
    
    # Current Q-values (from online network)
    current_q_values = self.q_network(states)
    current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Target Q-values (from TARGET network, not online)
    with torch.no_grad():
        next_q_values = self.target_network(next_states)  # <-- CHANGED
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + self.gamma * max_next_q * (1 - dones)
    
    # ... (loss and optimization same as before)
    
    # Update target network every C steps
    if self.steps % self.target_update_freq == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())
```

### 3. Update Save/Load

```python
def save(self, path):
    torch.save({
        "q_network": self.q_network.state_dict(),
        "target_network": self.target_network.state_dict(),  # <-- ADD
        "optimizer": self.optimizer.state_dict(),
        "epsilon": self.epsilon,
        "steps": self.steps,
    }, path)

def load(self, path):
    checkpoint = torch.load(path, weights_only=False)
    self.q_network.load_state_dict(checkpoint["q_network"])
    self.target_network.load_state_dict(checkpoint["target_network"])  # <-- ADD
    # ... rest same
```

---

## Proposed Hyperparameters for Target Network Runs

| Parameter | 2015 Paper | Our Value | Notes |
|-----------|------------|-----------|-------|
| target_update_freq | 10,000 | 10,000 | Steps between target network updates |
| learning_rate | 0.00025 | 0.00025 | Same as before |
| gamma | 0.99 | 0.99 | Same as before |
| buffer_size | 1,000,000 | 500,000 | Kaggle memory limit |
| batch_size | 32 | 32 | Same as before |
| epsilon_start | 1.0 | 1.0 | Same as before |
| epsilon_end | 0.1 | 0.1 | Same as before |
| epsilon_decay_steps | 1,000,000 | 1,000,000 | Same as before |
| total_steps | 10,000,000 | 2,500,000 | Kaggle time limit |
| gradient_clipping | Not mentioned | max_norm=10 | Less aggressive, network is more stable |

### Changes from 2013 Implementation

1. **Target Network Update Frequency (C=10,000):** This is the key hyperparameter from the 2015 paper. Every 10,000 steps, copy weights from online network to target network.

2. **Gradient Clipping Relaxed:** With target network providing stable targets, we can relax gradient clipping from max_norm=1 to max_norm=10 (or remove entirely).

3. **No other hyperparameter changes:** The target network itself should provide the stability needed.

---

## Expected Improvements

| Metric | 2013 (Current) | 2015 (Expected) |
|--------|----------------|-----------------|
| Q-value stability | High variance, occasional explosion | Stable growth |
| Training stability | Oscillating rewards | Smoother learning curve |
| Max reward | ~1150 | 5000-10000+ |
| Average reward | ~350-450 | 1000-2000+ |

The 2015 Nature paper reported **10,596 average** on Q*bert with the target network architecture.

---

## Files to Modify

1. `dqn_agent.py` - Add target network, modify train_step
2. `train.py` - Add target_update_freq hyperparameter
3. `hypertuning.py` - Add target_update_freq to search space (optional)

---

## Memory Impact

Adding a target network doubles the model size in GPU memory:
- Current: ~500 MB for Q-network
- With target: ~1 GB total

This is well within Kaggle's T4 GPU (15.6 GB VRAM).

---

## References

- Mnih et al., 2015: "Human-level control through deep reinforcement learning" (Nature)
- Key contribution: Target network + larger replay buffer = stable training
