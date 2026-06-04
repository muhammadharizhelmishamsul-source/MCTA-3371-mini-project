"""
ga_optimizer.py  –  Fixed version
GA now tunes 6 obstacle-repulsion weights in FuzzyController.
"""

import numpy as np
import time
from typing import List, Tuple, Callable, Optional

from environment      import make_env, N_ACTIONS
from fuzzy_controller import FuzzyController, N_PARAMS, DEFAULT_PARAMS

MAX_STEPS     = 300
N_EVAL_MAPS   = [1, 2]
N_EPISODES    = 3


def evaluate_individual(params: np.ndarray,
                         max_steps: int = MAX_STEPS,
                         maps: List[int] = N_EVAL_MAPS,
                         n_eps: int = N_EPISODES,
                         verbose: bool = False) -> float:
    ctrl          = FuzzyController(params)
    total_fitness = 0.0
    episodes_run  = 0

    for map_id in maps:
        env      = make_env(map_id)
        max_dist = (env.n_rows**2 + env.n_cols**2) ** 0.5

        for ep in range(n_eps):
            obs = env.reset()

            for _ in range(max_steps):
                action = ctrl.decide(obs)
                obs, reward, done, info = env.step(action)
                if done:
                    break

            success    = (env.robot_pos == env.goal_pos)
            final_dist = ((env.robot_pos[0] - env.goal_pos[0])**2 +
                          (env.robot_pos[1] - env.goal_pos[1])**2) ** 0.5

            fitness = (
                (200.0 if success else 0.0)
                - env.steps * 0.3
                - env.collisions * 5.0
                + 100.0 * (1.0 - final_dist / max_dist)
            )
            total_fitness += fitness
            episodes_run  += 1

            if verbose:
                print(f"  Map {map_id} Ep {ep+1}: "
                      f"{'OK' if success else 'XX'}  "
                      f"steps={env.steps}  coll={env.collisions}  "
                      f"fit={fitness:.1f}")

    return total_fitness / max(episodes_run, 1)


def _clip(p): return np.clip(p, 0.0, 1.0)

def _tournament(pop, fits, k=3):
    idx  = np.random.choice(len(pop), k, replace=False)
    best = idx[np.argmax([fits[i] for i in idx])]
    return pop[best].copy()

def _crossover(p1, p2, prob=0.5):
    mask = np.random.random(len(p1)) < prob
    return np.where(mask, p1, p2), np.where(mask, p2, p1)

def _mutate(p, prob=0.2, sigma=0.1):
    mask  = np.random.random(len(p)) < prob
    noise = np.random.normal(0, sigma, len(p))
    return _clip(p + mask * noise)


class GeneticOptimizer:

    def __init__(self, pop_size=30, n_generations=20,
                 elite_frac=0.15, cx_prob=0.5,
                 mut_prob=0.2, sigma=0.1,
                 callback: Optional[Callable] = None):
        self.pop_size      = pop_size
        self.n_generations = n_generations
        self.n_elite       = max(1, int(pop_size * elite_frac))
        self.cx_prob       = cx_prob
        self.mut_prob      = mut_prob
        self.sigma         = sigma
        self.callback      = callback
        self.best_params   = None
        self.best_fitness  = -np.inf
        self.history: List[Tuple[float, float]] = []

    def _init_pop(self):
        pop = [DEFAULT_PARAMS.copy()]
        for _ in range(self.pop_size - 1):
            pop.append(_clip(DEFAULT_PARAMS + np.random.normal(0, 0.2, N_PARAMS)))
        return pop

    def run(self, verbose=True):
        print(f"\nGA: pop={self.pop_size}  gen={self.n_generations}")
        pop  = self._init_pop()
        fits = [evaluate_individual(ind) for ind in pop]

        for gen in range(1, self.n_generations + 1):
            ranked   = sorted(zip(fits, pop), key=lambda x: x[0], reverse=True)
            fits     = [r[0] for r in ranked]
            pop      = [r[1] for r in ranked]
            gen_best = fits[0]
            gen_mean = float(np.mean(fits))
            self.history.append((gen_mean, gen_best))

            if gen_best > self.best_fitness:
                self.best_fitness = gen_best
                self.best_params  = pop[0].copy()

            if verbose:
                print(f"  Gen {gen:3d}  best={gen_best:7.2f}  "
                      f"mean={gen_mean:7.2f}  allbest={self.best_fitness:7.2f}")

            if self.callback:
                self.callback(gen, self.best_fitness, self.best_params)

            new_pop = [pop[i].copy() for i in range(self.n_elite)]
            while len(new_pop) < self.pop_size:
                p1 = _tournament(pop, fits)
                p2 = _tournament(pop, fits)
                c1, c2 = _crossover(p1, p2, self.cx_prob)
                new_pop.append(_mutate(c1, self.mut_prob, self.sigma))
                if len(new_pop) < self.pop_size:
                    new_pop.append(_mutate(c2, self.mut_prob, self.sigma))

            pop  = new_pop
            fits = [evaluate_individual(ind) for ind in pop]

        print(f"GA done. Best fitness = {self.best_fitness:.2f}")
        return self.best_params

    def save(self, path="best_params.npy"):
        if self.best_params is not None:
            np.save(path, self.best_params)

    @staticmethod
    def load(path="best_params.npy"):
        return np.load(path)


if __name__ == "__main__":
    ga   = GeneticOptimizer(pop_size=20, n_generations=10)
    best = ga.run(verbose=True)
    ga.save("best_params.npy")
    print("\nVerbose eval of best:")
    evaluate_individual(best, verbose=True)