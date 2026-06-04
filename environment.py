"""
environment.py
==============
2D Grid-based environment for mobile robot navigation.

Grid cell values:
    0 = free space
    1 = obstacle
    2 = start position
    3 = goal position

Author: Mini Project – MCTA3371 / MCTE4322
"""

import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Map Definitions
# ---------------------------------------------------------------------------

# MAP 1 – Simple map (open with scattered obstacles)
MAP_SIMPLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
], dtype=int)

# MAP 2 – Complex map (maze-like corridors)
MAP_COMPLEX = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
], dtype=int)


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

# 4-directional movement only: (row_delta, col_delta)
ACTIONS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}

ACTION_NAMES = {
    0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"
}

N_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# Environment Class
# ---------------------------------------------------------------------------

class RobotEnvironment:
    """
    A 2D discrete grid world for mobile robot navigation.

    The robot (point robot) moves one cell at a time.
    Sensor readings and goal direction are returned as the state
    so any soft-computing controller can consume them.
    """

    # Sensor ray directions (N, NE, E, SE, S, SW, W, NW)
    SENSOR_DIRS = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    SENSOR_MAX  = 5   # maximum sensor range (cells)

    def __init__(self, grid: np.ndarray, map_name: str = "map"):
        self.map_name  = map_name
        self.grid      = grid.copy()
        self.n_rows, self.n_cols = grid.shape

        # locate start & goal from grid markers
        starts = list(zip(*np.where(grid == 2)))
        goals  = list(zip(*np.where(grid == 3)))
        assert len(starts) == 1, "Map must have exactly one start cell (value=2)"
        assert len(goals)  == 1, "Map must have exactly one goal cell (value=3)"

        self.start_pos: Tuple[int,int] = starts[0]
        self.goal_pos:  Tuple[int,int] = goals[0]

        self.robot_pos: Tuple[int,int] = self.start_pos
        self.steps:     int = 0
        self.collisions:int = 0
        self.done:      bool = False
        self.path:      List[Tuple[int,int]] = [self.start_pos]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset the environment to the start state and return initial obs."""
        self.robot_pos = self.start_pos
        self.steps     = 0
        self.collisions= 0
        self.done      = False
        self.path      = [self.start_pos]
        return self._get_observation()

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """
        Execute one action.

        Parameters
        ----------
        action : int
            Index into ACTIONS dict (0-7).

        Returns
        -------
        obs    : dict  – sensor state (see _get_observation)
        reward : float – shaped reward signal
        done   : bool  – True when goal reached or step limit hit
        info   : dict  – diagnostics
        """
        assert not self.done, "Episode finished – call reset() first."
        assert action in ACTIONS, f"Invalid action: {action}"

        dr, dc = ACTIONS[action]
        nr     = self.robot_pos[0] + dr
        nc     = self.robot_pos[1] + dc

        collision = False
        if self._out_of_bounds(nr, nc) or self.grid[nr, nc] == 1:
            # Invalid move – stay in place
            collision   = True
            self.collisions += 1
            reward      = -5.0
        else:
            prev_dist   = self._dist_to_goal(self.robot_pos)
            self.robot_pos = (nr, nc)
            self.path.append(self.robot_pos)
            new_dist    = self._dist_to_goal(self.robot_pos)
            reward      = (prev_dist - new_dist) * 2.0 - 0.1  # progress – step cost

        self.steps += 1

        if self.robot_pos == self.goal_pos:
            reward   += 100.0
            self.done = True

        max_steps = self.n_rows * self.n_cols * 2
        if self.steps >= max_steps:
            self.done = True

        obs  = self._get_observation()
        info = {
            "steps":      self.steps,
            "collisions": self.collisions,
            "collision":  collision,
            "position":   self.robot_pos,
            "path_length":len(self.path),
        }
        return obs, reward, self.done, info

    # ------------------------------------------------------------------
    # State / Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> dict:
        """
        Build the full sensor state returned to the controller.

        Returns
        -------
        obs : dict with keys
            distance_to_goal   – Euclidean distance (float)
            angle_to_goal      – angle in radians, relative to grid +col axis
            sensor_readings    – list of 8 floats, obstacle distance per ray
            sensor_blocked     – list of 8 bools
            normalized_dist    – distance_to_goal scaled to [0,1]
            normalized_sensors – sensor_readings scaled to [0,1]
            position           – current (row, col)
            goal               – goal (row, col)
        """
        r, c       = self.robot_pos
        gr, gc     = self.goal_pos
        dist_goal  = self._dist_to_goal(self.robot_pos)
        max_diag   = np.hypot(self.n_rows, self.n_cols)

        # angle: 0 = right (+col), pi/2 = up (-row)
        angle_goal = np.arctan2(-(gr - r), gc - c)

        # 8-ray sensor scan
        sensor_dists  = []
        sensor_blocked= []
        for (sdr, sdc) in self.SENSOR_DIRS:
            d, blocked = self._cast_ray(r, c, sdr, sdc)
            sensor_dists.append(d)
            sensor_blocked.append(blocked)

        obs = {
            "distance_to_goal":   float(dist_goal),
            "angle_to_goal":      float(angle_goal),
            "sensor_readings":    sensor_dists,
            "sensor_blocked":     sensor_blocked,
            "normalized_dist":    float(dist_goal / max_diag),
            "normalized_sensors": [s / self.SENSOR_MAX for s in sensor_dists],
            "position":           self.robot_pos,
            "goal":               self.goal_pos,
        }
        return obs

    def _cast_ray(self, r: int, c: int,
                  dr: int, dc: int) -> Tuple[float, bool]:
        """Cast a ray in direction (dr,dc) and return (distance, is_blocked)."""
        for step in range(1, self.SENSOR_MAX + 1):
            nr, nc = r + dr * step, c + dc * step
            if self._out_of_bounds(nr, nc) or self.grid[nr, nc] == 1:
                return float(step - 1), True
        return float(self.SENSOR_MAX), False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _dist_to_goal(self, pos: Tuple[int,int]) -> float:
        return np.hypot(pos[0] - self.goal_pos[0],
                        pos[1] - self.goal_pos[1])

    def _out_of_bounds(self, r: int, c: int) -> bool:
        return r < 0 or r >= self.n_rows or c < 0 or c >= self.n_cols

    def get_free_cells(self) -> List[Tuple[int,int]]:
        """Return all walkable (non-obstacle) cells."""
        return [(r, c)
                for r in range(self.n_rows)
                for c in range(self.n_cols)
                if self.grid[r, c] != 1]

    def render_ascii(self) -> str:
        """Return an ASCII string representation of the current state."""
        symbols = {0: ".", 1: "#", 2: "S", 3: "G"}
        rows = []
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                if (r, c) == self.robot_pos:
                    row += "R"
                else:
                    row += symbols.get(self.grid[r, c], "?")
            rows.append(row)
        return "\n".join(rows)

    def print_state(self):
        """Pretty-print the current grid + sensor info."""
        print(f"\n{'='*40}")
        print(f"Map: {self.map_name}  |  Step: {self.steps}")
        print(self.render_ascii())
        obs = self._get_observation()
        print(f"\nRobot @ {obs['position']}  |  Goal @ {obs['goal']}")
        print(f"Distance to goal : {obs['distance_to_goal']:.2f}  "
              f"(norm: {obs['normalized_dist']:.3f})")
        print(f"Angle to goal    : {np.degrees(obs['angle_to_goal']):.1f}°")
        dirs = ["N","NE","E","SE","S","SW","W","NW"]
        sensor_str = "  ".join(
            f"{d}:{obs['sensor_readings'][i]:.0f}"
            f"{'!' if obs['sensor_blocked'][i] else ''}"
            for i, d in enumerate(dirs)
        )
        print(f"Sensors          : {sensor_str}")
        print(f"Collisions so far: {self.collisions}")
        print(f"{'='*40}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_env(map_id: int = 1) -> RobotEnvironment:
    """
    Convenience factory.

    Parameters
    ----------
    map_id : 1 → simple map, 2 → complex map
    """
    if map_id == 1:
        return RobotEnvironment(MAP_SIMPLE,  map_name="Simple Map")
    elif map_id == 2:
        return RobotEnvironment(MAP_COMPLEX, map_name="Complex Map")
    else:
        raise ValueError(f"Unknown map_id={map_id}. Use 1 (simple) or 2 (complex).")