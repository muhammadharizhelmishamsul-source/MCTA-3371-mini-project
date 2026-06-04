# Intelligent Mobile Robot Navigation
## MCTA3371 / MCTE4322 – Mini Project

---

## Project Structure

```
robot_nav/
├── environment.py    ← Core grid world + robot model + sensor system
├── visualizer.py     ← Matplotlib-based live & static visualizer
├── demo.py           ← Quick test with a random agent
├── requirements.txt  ← Python dependencies
└── README.md
```

> **Next steps** (future files you will add):
> - `fuzzy_controller.py`  – Mamdani fuzzy logic controller
> - `ga_optimizer.py`      – Genetic Algorithm to tune fuzzy rules / weights
> - `hybrid_controller.py` – Fuzzy-GA hybrid
> - `evaluate.py`          – Batch evaluation & comparison across maps

---

## Setup (PyCharm)

1. Open **PyCharm → File → Open** and select this folder.
2. Create a virtual environment:
   - **PyCharm → Settings → Project → Python Interpreter → Add → Virtualenv**
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the demo:
   ```bash
   python demo.py
   ```

---

## Environment Design

### Maps

| | Simple Map | Complex Map |
|---|---|---|
| Size | 10 × 10 | 15 × 15 |
| Layout | Scattered wall blocks | Maze-like corridors |
| Difficulty | Low | High |

**Grid cell values:**

| Value | Meaning |
|-------|---------|
| `0`   | Free space |
| `1`   | Obstacle / wall |
| `2`   | Start position (S) |
| `3`   | Goal position (G) |

### Robot Model

- **Point robot** – occupies one grid cell
- **Movement** – discrete 8-directional (N, NE, E, SE, S, SW, W, NW)
- **Collision rule** – invalid moves (into wall or boundary) are rejected; robot stays in place and receives a penalty reward

### Sensor System (8-ray scan)

The robot carries 8 directional proximity sensors:

```
  NW  N  NE
   W  R   E
  SW  S  SE
```

Each ray reports:
- **Distance** – number of free cells before hitting an obstacle (0 = blocked next to robot, 5 = max range, clear)
- **Blocked** – boolean flag

---

## Controller Interface

Your soft-computing controller receives an **observation dict** each step:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `distance_to_goal` | float | ≥ 0 | Euclidean distance to goal |
| `angle_to_goal` | float | [−π, π] | Angle from robot to goal |
| `sensor_readings` | list[8] | [0, 5] | Obstacle distance per ray |
| `sensor_blocked` | list[8] | bool×8 | True = ray blocked |
| `normalized_dist` | float | [0, 1] | Distance ÷ map diagonal |
| `normalized_sensors` | list[8] | [0, 1] | Sensors ÷ max range |
| `position` | tuple | (row, col) | Current robot cell |
| `goal` | tuple | (row, col) | Goal cell |

And must return an **action integer**:

| Action | Direction |
|--------|-----------|
| 0 | UP (−row) |
| 1 | DOWN (+row) |
| 2 | LEFT (−col) |
| 3 | RIGHT (+col) |
| 4 | UP-LEFT |
| 5 | UP-RIGHT |
| 6 | DOWN-LEFT |
| 7 | DOWN-RIGHT |

---

## Reward Signal

| Event | Reward |
|-------|--------|
| Each step | −0.1 (step cost) |
| Moving closer to goal | +2 × Δdistance |
| Collision / invalid move | −5.0 |
| Reaching goal | +100.0 |

---

## Evaluation Metrics

Tracked automatically in `RobotEnvironment`:

- **Success rate** – did the robot reach the goal?
- **Path length** – total steps taken
- **Collisions** – number of invalid moves
- **Total reward** – cumulative reward signal

---

## Usage Example

```python
from environment import make_env, N_ACTIONS
from visualizer  import LiveVisualizer
import numpy as np

env = make_env(1)          # 1 = simple map, 2 = complex
obs = env.reset()

viz = LiveVisualizer(env, interval_ms=300)
viz.start()

for _ in range(500):
    # Replace with your fuzzy / GA / hybrid controller:
    action = np.random.randint(0, N_ACTIONS)

    obs, reward, done, info = env.step(action)
    viz.update()

    if done:
        break

viz.close()
print("Steps:", info["steps"], "| Collisions:", info["collisions"])
```
