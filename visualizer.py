"""
visualizer.py
=============
Matplotlib-based visualizer for the robot navigation environment.

Draws the grid, robot position, sensor rays, and path history.
Works in both interactive (live animation) and static (save-frame) modes.

Author: Mini Project – MCTA3371 / MCTE4322
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional, List

from robot_nav.environment import RobotEnvironment

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
CMAP = ListedColormap(["#F0F4F8", "#2D3748", "#68D391", "#F6E05E"])
#                       free       obstacle   start      goal

ROBOT_COLOR  = "#E53E3E"   # red
PATH_COLOR   = "#63B3ED"   # blue
SENSOR_COLOR = "#FC8181"   # light red


# ---------------------------------------------------------------------------
# Static visualizer
# ---------------------------------------------------------------------------

def plot_grid(env: RobotEnvironment,
              show_sensors: bool = True,
              show_path: bool    = True,
              title: str         = "",
              ax: Optional[plt.Axes] = None,
              save_path: Optional[str] = None) -> plt.Figure:
    """
    Draw a single frame of the environment.

    Parameters
    ----------
    env          : RobotEnvironment instance
    show_sensors : overlay 8-ray sensor beams
    show_path    : draw the path the robot has taken
    title        : optional title string
    ax           : existing Axes to draw on (creates new figure if None)
    save_path    : if given, saves the figure to this path
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, env.n_cols), max(8, env.n_rows)))
    else:
        fig = ax.get_figure()

    n_rows, n_cols = env.n_rows, env.n_cols

    # ---- draw grid ----
    display = env.grid.astype(float)
    ax.imshow(display, cmap=CMAP, vmin=0, vmax=3,
              origin="upper", extent=[-0.5, n_cols - 0.5, n_rows - 0.5, -0.5])

    # ---- grid lines ----
    for r in range(n_rows + 1):
        ax.axhline(r - 0.5, color="#A0AEC0", lw=0.4)
    for c in range(n_cols + 1):
        ax.axvline(c - 0.5, color="#A0AEC0", lw=0.4)

    # ---- path ----
    if show_path and len(env.path) > 1:
        pr = [p[0] for p in env.path]
        pc = [p[1] for p in env.path]
        ax.plot(pc, pr, color=PATH_COLOR, lw=2, zorder=3, alpha=0.7,
                marker=".", markersize=4)

    # ---- sensor rays ----
    if show_sensors:
        obs = env._get_observation()
        rr, rc = env.robot_pos
        for i, (sdr, sdc) in enumerate(env.SENSOR_DIRS):
            dist    = obs["sensor_readings"][i]
            blocked = obs["sensor_blocked"][i]
            end_r   = rr + sdr * dist
            end_c   = rc + sdc * dist
            lw      = 1.2 if blocked else 0.6
            alpha   = 0.7 if blocked else 0.35
            ax.annotate("", xy=(end_c, end_r), xytext=(rc, rr),
                        arrowprops=dict(arrowstyle="->",
                                        color=SENSOR_COLOR,
                                        lw=lw, alpha=alpha))

    # ---- robot ----
    rr, rc = env.robot_pos
    circle = plt.Circle((rc, rr), 0.35,
                         color=ROBOT_COLOR, zorder=5)
    ax.add_patch(circle)
    ax.text(rc, rr, "R", ha="center", va="center",
            color="white", fontsize=9, fontweight="bold", zorder=6)

    # ---- labels ----
    sr, sc = env.start_pos
    gr, gc = env.goal_pos
    ax.text(sc, sr, "S", ha="center", va="center",
            color="#276749", fontsize=10, fontweight="bold", zorder=4)
    ax.text(gc, gr, "G", ha="center", va="center",
            color="#975A16", fontsize=10, fontweight="bold", zorder=4)

    # ---- axes styling ----
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    t = title or f"{env.map_name}  |  Step {env.steps}"
    obs = env._get_observation()
    subtitle = (f"Robot {env.robot_pos}  |  "
                f"Dist {obs['distance_to_goal']:.2f}  |  "
                f"Collisions {env.collisions}")
    ax.set_title(f"{t}\n{subtitle}", fontsize=11)

    # legend
    legend_handles = [
        mpatches.Patch(color="#F0F4F8", label="Free space",   ec="#A0AEC0"),
        mpatches.Patch(color="#2D3748", label="Obstacle"),
        mpatches.Patch(color="#68D391", label="Start"),
        mpatches.Patch(color="#F6E05E", label="Goal"),
        mpatches.Patch(color=ROBOT_COLOR,  label="Robot"),
        mpatches.Patch(color=PATH_COLOR,   label="Path"),
        mpatches.Patch(color=SENSOR_COLOR, label="Sensors"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[Visualizer] Saved frame → {save_path}")
    return fig


def plot_two_maps(env1: RobotEnvironment, env2: RobotEnvironment,
                  save_path: Optional[str] = None) -> plt.Figure:
    """Side-by-side view of both maps."""
    max_cols = max(env1.n_cols, env2.n_cols)
    fig, axes = plt.subplots(1, 2,
                             figsize=(max_cols * 1.2, max(env1.n_rows, env2.n_rows)))
    plot_grid(env1, ax=axes[0], show_sensors=False)
    plot_grid(env2, ax=axes[1], show_sensors=False)
    fig.suptitle("Environment Maps Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Live / animated visualizer
# ---------------------------------------------------------------------------

class LiveVisualizer:
    """
    Real-time step-by-step visualizer.

    Usage
    -----
        viz = LiveVisualizer(env)
        viz.start()
        for step in range(200):
            action = my_controller(obs)
            obs, reward, done, info = env.step(action)
            viz.update()
            if done:
                break
        viz.close()
    """

    def __init__(self, env: RobotEnvironment,
                 interval_ms: int = 300,
                 show_sensors: bool = True):
        self.env          = env
        self.interval_ms  = interval_ms
        self.show_sensors = show_sensors
        self.fig: Optional[plt.Figure] = None
        self.ax:  Optional[plt.Axes]   = None

    def start(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(
            figsize=(max(8, self.env.n_cols),
                     max(8, self.env.n_rows))
        )
        self.update()

    def update(self):
        if self.ax is None:
            return
        self.ax.cla()
        plot_grid(self.env, show_sensors=self.show_sensors,
                  ax=self.ax)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.interval_ms / 1000)

    def close(self):
        plt.ioff()
        if self.fig:
            plt.close(self.fig)


# ---------------------------------------------------------------------------
# Performance plots
# ---------------------------------------------------------------------------

def plot_performance(rewards: List[float],
                     path_lengths: List[int],
                     collisions: List[int],
                     title: str = "Navigation Performance",
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot episode rewards, path lengths, and collision counts.

    Parameters
    ----------
    rewards      : total reward per episode
    path_lengths : number of steps per episode
    collisions   : number of collisions per episode
    """
    episodes = range(1, len(rewards) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(episodes, rewards,     color="#4299E1", lw=1.5)
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, path_lengths, color="#68D391", lw=1.5)
    axes[1].set_ylabel("Path Length (steps)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(episodes, collisions,  color="#FC8181", lw=1.5)
    axes[2].set_ylabel("Collisions")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
