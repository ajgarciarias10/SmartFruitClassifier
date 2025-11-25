"""
Compare PSO and Artificial Bee Colony (ABC) on the Ackley benchmark (2D and 3D).

Produces CSV summary and convergence plots for parameter grids of iterations
and number of agents (particles / bees). Run multiple repetitions to collect
mean/std statistics.

Usage: run this file from project root or from Run/ (it adjusts sys.path).
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

# make sure project root is importable when running from Run/
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from Utilities.benchmark_functions import ackley


class SimplePSO:
    def __init__(self, dim, bounds, n_particles=30, w=0.7, c1=1.5, c2=1.5, rng=None):
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
        self.vel = np.zeros((n_particles, dim))

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

        # clamp velocity to reasonable range
        vmax = (self.bounds[:, 1] - self.bounds[:, 0])
        self.vel = np.clip(self.vel, -vmax, vmax)

        # update positions and clamp to bounds
        self.pos = np.clip(self.pos + self.vel, self.bounds[:, 0], self.bounds[:, 1])
        return self.gbest_val


class ArtificialBeeColony:
    """A minimal ABC implementation suitable for benchmarking.

    - population = number of food sources (employed bees)
    - onlookers = same number as employed by default
    - limit = iterations without improvement before scout replaces
    """
    def __init__(self, dim, bounds, n_bees=30, limit=None, rng=None):
        self.dim = dim
        self.bounds = np.asarray(bounds)
        self.n_bees = n_bees
        self.rng = rng if rng is not None else np.random.default_rng()
        self.limit = limit if limit is not None else max(20, int(0.6 * n_bees))

        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        # food sources positions
        self.pos = self.rng.uniform(low, high, size=(n_bees, dim))
        self.fitness = np.full(n_bees, np.inf)
        self.trial = np.zeros(n_bees, dtype=int)  # trial counters

        # evaluate initial
        for i in range(n_bees):
            self.fitness[i] = ackley(self.pos[i])

    def _neighbor(self, i):
        # generate neighbor for food source i by modifying a random dimension
        k = self.rng.integers(0, self.n_bees)
        while k == i:
            k = self.rng.integers(0, self.n_bees)
        phi = self.rng.uniform(-1, 1, size=self.dim)
        new = self.pos[i] + phi * (self.pos[i] - self.pos[k])
        # clamp
        new = np.clip(new, self.bounds[:, 0], self.bounds[:, 1])
        return new

    def step(self, fitness_func):
        # Employed bees phase
        for i in range(self.n_bees):
            new = self._neighbor(i)
            val = fitness_func(new)
            if val < self.fitness[i]:
                self.pos[i] = new
                self.fitness[i] = val
                self.trial[i] = 0
            else:
                self.trial[i] += 1

        # Calculate selection probabilities for onlookers (fitness-proportional)
        # lower fitness -> higher probability, so transform
        f = self.fitness
        # avoid division by zero
        maxf = f.max()
        probs = (maxf - f + 1e-12)
        probs = probs / probs.sum()

        # Onlooker bees: allocate same number as employed
        for _ in range(self.n_bees):
            i = self.rng.choice(self.n_bees, p=probs)
            new = self._neighbor(i)
            val = fitness_func(new)
            if val < self.fitness[i]:
                self.pos[i] = new
                self.fitness[i] = val
                self.trial[i] = 0
            else:
                self.trial[i] += 1

        # Scout phase: replace any source with trial > limit
        for i in range(self.n_bees):
            if self.trial[i] > self.limit:
                low = self.bounds[:, 0]
                high = self.bounds[:, 1]
                self.pos[i] = self.rng.uniform(low, high, size=self.dim)
                self.fitness[i] = fitness_func(self.pos[i])
                self.trial[i] = 0

        # return best value
        return float(self.fitness.min())


def run_grid(alg_name, dim, iters, agents, runs=8):
    bounds = np.tile(np.array([-32.768, 32.768]), (dim, 1))
    all_finals = []
    all_hist = []
    times = []
    for run in range(runs):
        seed = int(time.time() * 1000) % 2 ** 32 + run
        rng = np.random.default_rng(seed)

        if alg_name == 'PSO':
            opt = SimplePSO(dim=dim, bounds=bounds, n_particles=agents, rng=rng)
            hist = []
            start = time.time()
            for it in range(iters):
                val = opt.step(lambda x: ackley(x))
                hist.append(val)
            elapsed = time.time() - start
            final = hist[-1]
        elif alg_name == 'ABC':
            opt = ArtificialBeeColony(dim=dim, bounds=bounds, n_bees=agents, rng=rng)
            hist = []
            start = time.time()
            for it in range(iters):
                val = opt.step(lambda x: ackley(x))
                hist.append(val)
            elapsed = time.time() - start
            final = hist[-1]
        else:
            raise ValueError('Unknown algorithm: ' + alg_name)

        all_finals.append(final)
        all_hist.append(hist)
        times.append(elapsed)
        print(f'{alg_name} run {run+1}/{runs} dim={dim} agents={agents} iters={iters} final={final:.4e} time={elapsed:.2f}s')

    return np.array(all_hist), np.array(all_finals), np.array(times)


def save_summary_csv(out_dir, rows, filename='summary.csv'):
    path = os.path.join(out_dir, filename)
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['algorithm', 'dim', 'iters', 'agents', 'runs', 'best_mean', 'best_std', 'best_min', 'mean_time'])
        for r in rows:
            writer.writerow(r)
    print('Saved summary to', path)


def plot_convergence(all_hist, alg_name, dim, agents, iters, outpath):
    all_hist = np.asarray(all_hist)
    mean = all_hist.mean(axis=0)
    std = all_hist.std(axis=0)
    iters_idx = np.arange(all_hist.shape[1])
    plt.figure(figsize=(8, 5))
    plt.plot(iters_idx, mean, label='mean best')
    plt.fill_between(iters_idx, mean - std, mean + std, alpha=0.25)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Best fitness (log)')
    plt.title(f'{alg_name} on Ackley (dim={dim}) agents={agents} iters={iters}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'optimizer_results')
    os.makedirs(out_dir, exist_ok=True)

    algorithms = ['PSO', 'ABC']
    dims = [2, 3]
    iters_list = [50, 100, 200]
    agents_list = [10, 30, 50]
    runs = 8

    summary_rows = []

    for alg in algorithms:
        for dim in dims:
            for agents in agents_list:
                for iters in iters_list:
                    all_hist, finals, times = run_grid(alg, dim, iters, agents, runs=runs)

                    mean_best = float(finals.mean())
                    std_best = float(finals.std())
                    min_best = float(finals.min())
                    mean_time = float(times.mean())

                    summary_rows.append([alg, dim, iters, agents, runs, mean_best, std_best, min_best, mean_time])

                    # save convergence plot and finals CSV
                    plot_path = os.path.join(out_dir, f'{alg}_dim{dim}_agents{agents}_iters{iters}.png')
                    plot_convergence(all_hist, alg, dim, agents, iters, plot_path)
                    np.savetxt(os.path.join(out_dir, f'{alg}_dim{dim}_agents{agents}_iters{iters}_finals.csv'), finals, header='final_best')
                    print('Saved results for', alg, dim, agents, iters)

    save_summary_csv(out_dir, summary_rows)


if __name__ == '__main__':
    main()
