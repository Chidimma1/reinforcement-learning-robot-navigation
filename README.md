# Reinforcement Learning for Autonomous Robot Navigation

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-orange)
![Course](https://img.shields.io/badge/ELET%206303-Applied%20Neural%20Networks-red)
![UH](https://img.shields.io/badge/University%20of%20Houston-Cullen%20College-C8102E)

## Abstract

This project applies reinforcement learning to autonomous robot navigation in a
simulated 2D grid environment. A 10×10 GridWorld with 15 randomly-placed obstacles,
a fixed start (0,0) and goal (9,9), and BFS-guaranteed solvability serves as the
testbed. Two RL methods — tabular Q-Learning and Deep Q-Network (DQN) — are
implemented from scratch and benchmarked against an optimal Breadth-First Search
baseline (18 steps).

Q-Learning converged to 100% path efficiency, exactly matching the BFS optimum.
DQN initially diverged due to insufficient state representation and an overly
aggressive learning rate, but was subsequently fixed with one-hot state encoding,
Adam (lr=1e-3), and gradient clipping — also achieving 100% efficiency.
Four systematic parameter experiments confirmed the chosen baseline and identified
one concrete improvement: dense step shaping (step=−2) reduces path length by 13%
with no loss of success rate.

## Tech Stack

| Category       | Tools / Libraries                                      |
|----------------|--------------------------------------------------------|
| Language       | Python 3.x                                             |
| RL Algorithms  | Tabular Q-Learning, Deep Q-Network (DQN)               |
| DQN Framework  | PyTorch (Linear layers, Adam optimizer, MSE loss)      |
| Baseline       | Breadth-First Search (BFS)                             |
| Visualization  | Matplotlib, Matplotlib.animation (PillowWriter), Pygame|
| Utilities      | NumPy, collections.deque, random, math                 |
| Environment    | Jupyter Notebook / Google Colab                        |

## Environment

A custom `GridWorld` class was built from scratch (Person A) with the following configuration:

| Parameter          | Value                                      |
|--------------------|--------------------------------------------|
| Grid size          | 10×10 (100 discrete states)                |
| Obstacles          | 15 randomly placed, BFS-solvability checked|
| Start / Goal       | (0,0) top-left → (9,9) bottom-right        |
| Action space       | 4 — UP, DOWN, LEFT, RIGHT                  |
| Seed               | 42 (fully reproducible)                    |
| Max steps/episode  | 200                                        |

**Reward structure:**

| Event         | Reward |
|---------------|--------|
| Reach goal    | +100   |
| Hit obstacle  | −10    |
| Hit wall      | −5     |
| Each step     | −1     |

**Bonus features implemented:**
- `dynamic_obstacles=True` — obstacles shift each step (30% probability); solvability re-checked after each shift
- `slip_probability` — intended action overridden randomly with probability p, modelling actuator noise
- LiDAR-style sensor — 10-value vector (8 directional distances + goal distance + goal angle)
- Pygame real-time visualizer with live sensor display, speed controls, and pause functionality

## Algorithms & Results

### Baseline: BFS Optimal Path
BFS on the seed=42 grid finds the shortest path in **18 steps** — the theoretical ceiling for RL comparison.

---

### Q-Learning (tabular)

**Hyperparameters:** α=0.1 | γ=0.99 | ε-start=1.0 | ε-min=0.01 | ε-decay=0.995 | Episodes=1,000

| Episode | Avg Reward | Avg Steps | Success Rate | Epsilon |
|---------|------------|-----------|--------------|---------|
| 200     | +32.0      | 37.7      | 100%         | 0.367   |
| 400     | +72.6      | 22.0      | 100%         | 0.135   |
| 600     | +79.1      | 19.3      | 100%         | 0.049   |
| 800     | +81.8      | 18.4      | 100%         | 0.018   |
| 1000    | +82.5      | 18.1      | 100%         | 0.010   |

Final greedy episode: **18 steps — 100% path efficiency (matches BFS optimum exactly)**

---

### DQN (Deep Q-Network)

**Architecture:** Input(1) → Linear(64, ReLU) → Linear(64, ReLU) → Output(4) | 4,548 parameters  
**Fixed version:** one-hot state encoding | Adam lr=1e-3 | gradient clipping | hidden=128

| Metric            | Q-Learning | DQN    | BFS     |
|-------------------|------------|--------|---------|
| Avg reward (last 100) | +82.5  | Converged | N/A |
| Avg steps to goal | 18.1       | 18     | 18      |
| Success rate      | 100%       | 100%   | 100%    |
| Path efficiency   | 100%       | 100%   | 100%    |

**Key lesson:** Q-Learning is the right algorithm for a 100-state grid. DQN is designed for large or continuous state spaces — algorithm selection matters more than algorithm complexity.

## Parameter Variation Experiments

Four systematic experiments were conducted (Person C), each changing exactly one variable
while holding all others at baseline. Training budget: 1,000 episodes.

### Experiment 1 — Learning Rate (α)

| α Value       | Final Avg Reward | Success Rate | Notes                          |
|---------------|------------------|--------------|--------------------------------|
| 0.001         | −76.7            | 74%          | Failed — too slow to converge  |
| 0.01          | +68.2            | 100%         | Works but still trending at ep 1000 |
| **0.1 (baseline)** | **+82.7**   | **100%**     | **Best — Goldilocks zone**     |
| 0.5           | +82.2            | 100%         | Fast but noisy (Q-value overshooting) |

**Finding:** α=0.1 confirmed as optimal for 1,000 episodes on this grid.

---

### Experiment 2 — Epsilon Decay Rate

| Decay  | ε hits 0.01 at | Final Avg Reward | Success Rate | Notes                        |
|--------|----------------|------------------|--------------|------------------------------|
| 0.99   | Episode 449    | +82.6            | 100%         | Exploits too early           |
| **0.995 (baseline)** | **Episode 917** | **+82.7** | **100%** | **Best — perfectly calibrated** |
| 0.999  | Never (in 1000) | +53.4           | 100%         | 37% random at episode 1000   |

**Finding:** Epsilon decay must match the episode budget — even a 0.005 difference compounds dramatically.

---

### Experiment 3 — Reward Shaping

| Configuration      | Reward Structure             | Final Avg Reward | Success | Avg Steps |
|--------------------|------------------------------|------------------|---------|-----------|
| Baseline           | step=−1                      | +82.7            | 100%    | 18.2      |
| Sparse             | goal-only                    | —                | 100%    | ~18       |
| No step cost       | step=0                       | —                | 100%    | ~18       |
| **Dense shaping**  | **step=−2**                  | +65.6*           | **100%**| **15.8**  |

*Lower raw reward due to higher per-step deduction — not worse performance.

**Finding:** Dense step shaping (step=−2) reduces path length by **13%** with no loss in success rate. Recommended improvement over baseline.

---

### Experiment 4 — Slip Probability (Motion Uncertainty)

| Slip Probability | Success Rate (50 test eps) | Avg Steps | Step Increase |
|------------------|----------------------------|-----------|---------------|
| 0% (deterministic) | 100%                     | 18.0      | Baseline      |
| 10%              | 100%                       | 19.9      | +10.6%        |
| 20%              | 100%                       | 23.0      | +27.8%        |

**Finding:** Q-Learning maintains 100% success up to 20% motion noise. The agent emergently learns
to keep wider clearance from obstacle clusters — a safety-conscious policy that was never explicitly programmed.

## Bonus Features

| Feature              | Result                                                        |
|----------------------|---------------------------------------------------------------|
| Dynamic obstacles    | Agent solved maze in 18 steps with obstacles shifting per step|
| 20×20 grid           | 4,000 episodes → 38 steps (100% efficiency vs BFS 38 steps)  |
| 25×25 grid           | 6,000 episodes → ~52 steps (100% efficiency vs BFS ~52 steps) |
| Slip probability 20% | 100% success maintained; path length +27.8% (18 → 23 steps)  |
| Pygame visualizer    | Live training window with LiDAR beams, HUD, speed/pause controls |

## Project Structure

```
RL-Robot-Navigation/
├── Final_Project_Clean.ipynb          # Full annotated source code
│                                      # (learning curves, policy visualizations,
│                                      #  4-panel dashboards, animated GIFs,
│                                      #  Pygame visualizer, all experiment plots)
├── Final_Project_Report.pdf           # Full written report (Parts 1, 2, 3)
├── Visualizations_Supplement.pdf      # All figures paired with report sections
└── README.md
```

All code is organized section-by-section in the notebook with comments explaining each block.
Part 1 (environment/tools), Part 2 (Q-Learning & DQN), and Part 3 (parameter experiments)
are self-contained — Parts 2 and 3 contain only calls into the Part 1 module.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/RL-Robot-Navigation.git
```

2. Open the notebook in Google Colab or Jupyter:
```
Final_Project_Clean.ipynb
```

3. Run all cells sequentially. Dependencies (NumPy, Matplotlib) are available
   in Colab by default. For the Pygame visualizer, a local Jupyter environment
   with a display is required.

> **Note:** All experiments use `seed=42` for full reproducibility.
> The 10×10 grid with 15 obstacles will be identical across all runs.

## Team Contributions

| Member                              | Role                   | Contribution                                                      |
|-------------------------------------|------------------------|-------------------------------------------------------------------|
| Bhone Kyaw Si Hein                  | Environment & Tools    | GridWorld class, LiDAR sensor, all visualization functions, BFS baseline, GIF animations, Pygame visualizer |
| Person B                            | Q-Learning & DQN       | Q-Learning, DQN (with replay buffer & target network), BFS comparison, bonus features (dynamic obstacles, larger grids) |
| Chidimma Ukoha                      | Parameter Study        | Four parameter variation experiments (α, ε-decay, reward shaping, slip probability), analysis, final report writing |

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279–292.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.

## Course Info

**Course:** ELET 6303 — Applied Neural Networks  
**Program:** MS in Engineering Data Science & Artificial Intelligence  
**Institution:** University of Houston, Cullen College of Engineering  
**Semester:** Spring 2026
