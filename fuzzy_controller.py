"""
fuzzy_controller.py  –  v4  (4-directional only)
Actions: 0=UP  1=DOWN  2=LEFT  3=RIGHT
"""

import numpy as np

N_PARAMS = 4   # one repulsion weight per cardinal sensor direction
DEFAULT_PARAMS = np.array([0.6, 0.6, 0.6, 0.6], dtype=float)

# Which sensor (N/S/E/W) blocks which actions
# Sensors: 0=N, 1=S, 2=E, 3=W  (only 4 cardinal rays now)
SENSOR_BLOCKS = {
    0: [0],   # N sensor blocks UP
    1: [1],   # S sensor blocks DOWN
    2: [3],   # E sensor blocks RIGHT
    3: [2],   # W sensor blocks LEFT
}


class FuzzyController:

    def __init__(self, params: np.ndarray = None):
        if params is None:
            params = DEFAULT_PARAMS.copy()
        self.params = np.clip(params, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Angle → 4-direction scores
    # ------------------------------------------------------------------
    def _angle_scores(self, angle_rad: float) -> np.ndarray:
        a = angle_rad
        s = np.zeros(4)
        s[0] = max(0.0,  np.sin(a))   # UP    — angle points upward
        s[1] = max(0.0, -np.sin(a))   # DOWN
        s[3] = max(0.0,  np.cos(a))   # RIGHT — angle points right
        s[2] = max(0.0, -np.cos(a))   # LEFT
        return s

    # ------------------------------------------------------------------
    # Obstacle repulsion (cardinal directions only)
    # ------------------------------------------------------------------
    def _obstacle_penalty(self, obs: dict) -> np.ndarray:
        """
        Use the 4 cardinal sensor readings (N=idx0, NE=1, E=2, SE=3,
        S=4, SW=5, W=6, NW=7) — we only use N(0), E(2), S(4), W(6).
        """
        sensors = obs["normalized_sensors"]
        penalty = np.zeros(4)
        # cardinal sensor indices in the 8-ray list
        card = {0: 0,   # N  → action UP(0)
                2: 3,   # E  → action RIGHT(3)
                4: 1,   # S  → action DOWN(1)
                6: 2}   # W  → action LEFT(2)
        weights = self.params  # [wN, wS, wE, wW]
        w_map   = {0: weights[0], 4: weights[1],
                   2: weights[2], 6: weights[3]}

        for s_idx, action in card.items():
            closeness = max(0.0, 1.0 - sensors[s_idx])
            if closeness > 0.05:
                penalty[action] += w_map[s_idx] * closeness * 2.5
        return penalty

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------
    def _compute_scores(self, obs: dict) -> np.ndarray:
        angle      = obs["angle_to_goal"]
        dist_cells = obs["distance_to_goal"]

        base = self._angle_scores(angle)

        # Goal proximity override — within 2 cells, ignore obstacles
        if dist_cells <= 2.0:
            return base * 3.0

        penalty       = self._obstacle_penalty(obs)
        sensors       = obs["normalized_sensors"]
        avg_closeness = float(np.mean([max(0, 1 - sensors[i])
                                       for i in [0, 2, 4, 6]]))
        penalty_scale = 0.4 + avg_closeness * 1.2
        scores        = base - penalty * penalty_scale

        # If completely boxed in, fall back to angle only
        if np.max(scores) <= 0:
            scores = base

        return scores

    def decide(self, obs: dict) -> int:
        return int(np.argmax(self._compute_scores(obs)))

    def decide_with_scores(self, obs: dict):
        scores = self._compute_scores(obs)
        return int(np.argmax(scores)), scores