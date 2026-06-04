"""
demo.py
=======
Quick demonstration of the environment + visualizer.

Runs a random agent on both maps so you can verify everything works
before plugging in your soft-computing controller.

Run:
    python demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

from environment import make_env, ACTIONS, ACTION_NAMES, N_ACTIONS
from visualizer  import plot_two_maps, LiveVisualizer, plot_performance


# ---------------------------------------------------------------------------
# Helper: run one episode with a random agent
# ---------------------------------------------------------------------------

def run_random_episode(map_id: int,
                       max_steps: int = 300,
                       live: bool = False,
                       verbose: bool = True) -> dict:
    """
    Run one episode of random navigation and return performance metrics.

    Parameters
    ----------
    map_id    : 1 = simple, 2 = complex
    max_steps : hard step ceiling (overrides env default if smaller)
    live      : show live animation
    verbose   : print step-by-step state

    Returns
    -------
    metrics : dict with keys success, steps, collisions, path_length,
              total_reward, path
    """
    env = make_env(map_id)
    obs = env.reset()

    viz = LiveVisualizer(env, interval_ms=200) if live else None
    if viz:
        viz.start()

    total_reward = 0.0
    success      = False

    print(f"\n{'='*50}")
    print(f"Running RANDOM agent on {env.map_name}")
    print(f"Start: {env.start_pos}  →  Goal: {env.goal_pos}")
    print(f"{'='*50}")

    for step in range(max_steps):
        action = np.random.randint(0, N_ACTIONS)

        if verbose:
            print(f"  Step {step+1:3d} | Action: {ACTION_NAMES[action]:10s} | "
                  f"Pos: {env.robot_pos} | "
                  f"Dist: {obs['distance_to_goal']:.2f} | "
                  f"Sensors: {[f'{s:.0f}' for s in obs['sensor_readings']]}")

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if viz:
            viz.update()

        if done:
            success = (env.robot_pos == env.goal_pos)
            break

    if viz:
        viz.close()

    status = "✓ GOAL REACHED" if success else "✗ timed out"
    print(f"\n{status}  |  Steps: {env.steps}  |  "
          f"Collisions: {env.collisions}  |  "
          f"Total reward: {total_reward:.1f}")

    return {
        "success":      success,
        "steps":        env.steps,
        "collisions":   env.collisions,
        "path_length":  len(env.path),
        "total_reward": total_reward,
        "path":         env.path.copy(),
    }


# ---------------------------------------------------------------------------
# Helper: display observation keys clearly
# ---------------------------------------------------------------------------

def display_observation_info(map_id: int = 1):
    """Print the controller input/output specification."""
    env = make_env(map_id)
    obs = env.reset()

    print("\n" + "="*55)
    print(" CONTROLLER INPUT / OUTPUT SPECIFICATION")
    print("="*55)
    print("\nINPUTS (observation dict keys):\n")
    print(f"  {'Key':<22} {'Type':<10} {'Shape/Range':<20} Description")
    print(f"  {'-'*22} {'-'*10} {'-'*20} {'-'*30}")

    rows = [
        ("distance_to_goal",   "float", "≥ 0",          "Euclidean distance to goal"),
        ("angle_to_goal",      "float", "[-π, π] rad",  "Angle from robot to goal"),
        ("sensor_readings",    "list",  "8 × [0,5]",    "Obstacle dist per ray (N NE E SE S SW W NW)"),
        ("sensor_blocked",     "list",  "8 × bool",     "True if ray hit obstacle"),
        ("normalized_dist",    "float", "[0, 1]",       "Distance scaled by map diagonal"),
        ("normalized_sensors", "list",  "8 × [0, 1]",  "Sensor distances scaled to [0,1]"),
        ("position",           "tuple", "(row, col)",   "Current robot grid position"),
        ("goal",               "tuple", "(row, col)",   "Goal grid position"),
    ]
    for name, typ, rng, desc in rows:
        print(f"  {name:<22} {typ:<10} {rng:<20} {desc}")

    print("\nOUTPUTS (action space):\n")
    for idx, name in {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT",
                      4:"UP-LEFT",5:"UP-RIGHT",
                      6:"DOWN-LEFT",7:"DOWN-RIGHT"}.items():
        print(f"  Action {idx}: {name}")

    print(f"\nSample observation values (from map {map_id} start):")
    for k, v in obs.items():
        if isinstance(v, list):
            fmt = [f"{x:.2f}" if isinstance(x, float) else str(x) for x in v]
            print(f"  {k:<22}: {fmt}")
        elif isinstance(v, float):
            print(f"  {k:<22}: {v:.4f}")
        else:
            print(f"  {k:<22}: {v}")
    print("="*55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # 1) Show both maps side-by-side (static)
    env1 = make_env(1)
    env2 = make_env(2)
    env1.reset()
    env2.reset()
    fig = plot_two_maps(env1, env2, save_path="maps_overview.png")
    plt.show(block=False)
    plt.pause(1)

    # 2) Print controller I/O specification
    display_observation_info(map_id=1)

    # 3) Run random agent on simple map (no live animation, no verbose)
    print("\n--- Random agent on Simple Map ---")
    m1 = run_random_episode(map_id=1, max_steps=200, verbose=False)

    # 4) Run random agent on complex map
    print("\n--- Random agent on Complex Map ---")
    m2 = run_random_episode(map_id=2, max_steps=500, verbose=False)

    # 5) Visualize final state of each
    for map_id, label in [(1, "simple"), (2, "complex")]:
        env = make_env(map_id)
        obs = env.reset()
        # replay 50 random steps so path is visible
        for _ in range(50):
            obs, _, done, _ = env.step(np.random.randint(0, N_ACTIONS))
            if done:
                break
        from visualizer import plot_grid
        fig2, ax = plt.subplots(figsize=(10, 10))
        plot_grid(env, ax=ax, show_sensors=True,
                  title=f"Final State – {env.map_name}")
        fig2.savefig(f"final_state_{label}.png", dpi=150)
        plt.show(block=False)
        plt.pause(0.5)

    print("\n[demo.py] Done. Saved: maps_overview.png, "
          "final_state_simple.png, final_state_complex.png")
    plt.show()
