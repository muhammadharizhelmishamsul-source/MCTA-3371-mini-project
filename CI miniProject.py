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