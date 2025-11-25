"""
Simple PSO test on the Ackley benchmark to validate optimizer behavior.

Generates convergence curves and basic statistics for multiple runs.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ensure project root is on sys.path so `Utilities` can be imported when
# running this script from the Run/ folder or from elsewhere
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from Utilities.benchmark_functions import ackley


class SimplePSO:
    def __init__(self, dim, bounds, n_particles=30, w=0.7, c1=1.5, c2=1.5, vmax=None, rng=None):
        self.dim = dim
        self.bounds = np.asarray(bounds)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = rng if rng is not None else np.random.default_rng()

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.pos = self.rng.uniform(low, high, size=(n_particles, dim))
        self.vel = np.zeros((n_particles, dim)) if vmax is None else self.rng.uniform(-vmax, vmax, (n_particles, dim))
        self.vmax = vmax if vmax is not None else (high - low)

        self.pbest_pos = self.pos.copy()
        self.pbest_val = np.array([np.inf] * n_particles)
        self.gbest_pos = None
        self.gbest_val = np.inf

    def step(self, fitness_func):
        r1 = self.rng.random((self.n_particles, self.dim))
        r2 = self.rng.random((self.n_particles, self.dim))

        for i in range(self.n_particles):
            val = fitness_func(self.pos[i])
            if val < self.pbest_val[i]:
                self.pbest_val[i] = val
                self.pbest_pos[i] = self.pos[i].copy()
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest_pos = self.pos[i].copy()

        cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
        social = self.c2 * r2 * (self.gbest_pos - self.pos)
        self.vel = self.w * self.vel + cognitive + social

        # clamp velocity
        vmax = np.abs(self.vmax)
        self.vel = np.clip(self.vel, -vmax, vmax)

        # update positions
        self.pos = self.pos + self.vel

        # handle boundaries by clamping
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        self.pos = np.clip(self.pos, low, high)

        return self.gbest_val


def run_experiment(dim=2, runs=10, particles=30, iters=100):
    bounds = np.tile(np.array([-32.768, 32.768]), (dim, 1))

    all_best_hist = []
    best_final = []
    times = []

    for run in range(runs):
        seed = int(time.time() * 1000) % 2**32 + run
        rng = np.random.default_rng(seed)
        pso = SimplePSO(dim=dim, bounds=bounds, n_particles=particles, rng=rng)

        best_hist = []
        start = time.time()
        for it in range(iters):
            best = pso.step(lambda x: ackley(x))
            best_hist.append(best)
        elapsed = time.time() - start

        all_best_hist.append(best_hist)
        best_final.append(best_hist[-1])
        times.append(elapsed)
        print(f'Run {run+1}/{runs} finished: final best={best_final[-1]:.6e}, time={elapsed:.2f}s')

    return np.array(all_best_hist), np.array(best_final), np.array(times)


def plot_results(all_hist, dim, outpath):
    median = np.median(all_hist, axis=0)
    mean = np.mean(all_hist, axis=0)
    std = np.std(all_hist, axis=0)

    iters = np.arange(all_hist.shape[1])
    plt.figure(figsize=(8, 6))
    plt.plot(iters, mean, label='mean best')
    plt.fill_between(iters, mean - std, mean + std, alpha=0.3, label='±1 std')
    plt.plot(iters, median, label='median', linestyle='--')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Best fitness (log scale)')
    plt.title(f'PSO on Ackley (dim={dim})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    print('Saved plot to', outpath)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'pso_ackley_results')
    os.makedirs(out_dir, exist_ok=True)

    # Quick experiments: dim=2 and dim=5
    for dim in (2, 5):
        print('\nRunning PSO on Ackley, dim=', dim)
        all_hist, best_final, times = run_experiment(dim=dim, runs=8, particles=30, iters=120)

        print(f'Final stats dim={dim}: best mean={best_final.mean():.4e}, std={best_final.std():.4e}, min={best_final.min():.4e}')
        plot_path = os.path.join(out_dir, f'ackley_pso_dim{dim}.png')
        plot_results(all_hist, dim, plot_path)

        # save a small csv with final results
        np.savetxt(os.path.join(out_dir, f'ackley_pso_dim{dim}_finals.csv'), best_final, header='final_best')
        print('Saved finals CSV and times. Mean time per run:', times.mean())


if __name__ == '__main__':
    main()
