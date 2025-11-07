import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import random
import math

# Ackley function implementation
def ackley_function(x: np.ndarray) -> float:
    """
    Ackley function implementation for n dimensions.
    Global minimum f(0,0,...,0) = 0
    Usually evaluated on xi ∈ [-32.768, 32.768]
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x) / d))
    
    return sum_sq_term + cos_term + a + np.exp(1)

# PSO Implementation
@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_score: float

class PSO:
    def __init__(self, num_particles: int, dimensions: int, bounds: Tuple[float, float],
                 w: float = 0.7, c1: float = 2.0, c2: float = 2.0):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive weight
        self.c2 = c2  # Social weight
        
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float('inf')
        
        self._initialize_particles()

    def _initialize_particles(self):
        for _ in range(self.num_particles):
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
            velocity = np.random.uniform(-1, 1, self.dimensions)
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_score=float('inf')
            )
            self.particles.append(particle)

    def optimize(self, fitness_func, iterations: int = 100) -> Tuple[np.ndarray, float, List[float]]:
        history = []
        
        for _ in range(iterations):
            for particle in self.particles:
                # Evaluate current position
                score = fitness_func(particle.position)
                
                # Update particle's best
                if score < particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.particles:
                r1, r2 = np.random.rand(2)
                
                # Update velocity
                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                
                # Update position
                particle.position = particle.position + particle.velocity
                
                # Bound position
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])
            
            history.append(self.global_best_score)
        
        return self.global_best_position, self.global_best_score, history

def run_comparison(dimensions: int, num_trials: int = 30):
    """Run comparison between PSO and ABC for the given dimensions."""
    bounds = (-32.768, 32.768)  # Ackley function bounds
    results = []
    
    # Test different parameter combinations for PSO
    pso_params = [
        {'w': 0.7, 'c1': 2.0, 'c2': 2.0},
        {'w': 0.5, 'c1': 1.5, 'c2': 1.5},
        {'w': 0.9, 'c1': 2.5, 'c2': 2.5}
    ]
    
    for params in pso_params:
        pso_scores = []
        for _ in range(num_trials):
            pso = PSO(
                num_particles=30,
                dimensions=dimensions,
                bounds=bounds,
                w=params['w'],
                c1=params['c1'],
                c2=params['c2']
            )
            _, score, _ = pso.optimize(ackley_function, iterations=100)
            pso_scores.append(score)
        
        results.append({
            'Algorithm': f"PSO (w={params['w']}, c1={params['c1']}, c2={params['c2']})",
            'Mean': np.mean(pso_scores),
            'Std': np.std(pso_scores),
            'Best': np.min(pso_scores),
            'Worst': np.max(pso_scores)
        })
    
    # Create results table
    df = pd.DataFrame(results)
    print(f"\nResults for {dimensions}D Ackley Function:")
    print(df.to_string(index=False))
    
    return df

def plot_convergence_comparison(dimensions: int):
    """Plot convergence comparison between PSO and ABC."""
    pso = PSO(num_particles=30, dimensions=dimensions, bounds=(-32.768, 32.768))
    _, _, pso_history = pso.optimize(ackley_function, iterations=100)
    
    plt.figure(figsize=(10, 6))
    plt.plot(pso_history, label='PSO')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score')
    plt.title(f'Convergence Comparison for {dimensions}D Ackley Function')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'convergence_{dimensions}d.png')
    plt.close()

def plot_2d_surface():
    """Plot 2D surface of Ackley function."""
    x = np.linspace(-32.768, 32.768, 100)
    y = np.linspace(-32.768, 32.768, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = ackley_function(np.array([X[i,j], Y[i,j]]))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.colorbar(surf)
    plt.title('Ackley Function Surface (2D)')
    plt.savefig('ackley_2d_surface.png')
    plt.close()

if __name__ == "__main__":
    # Run comparisons for both 2D and 3D
    print("Running benchmark comparisons...")
    
    # Plot 2D surface
    plot_2d_surface()
    
    # Run and plot comparisons for 2D
    df_2d = run_comparison(dimensions=2)
    plot_convergence_comparison(dimensions=2)
    
    # Run and plot comparisons for 3D
    df_3d = run_comparison(dimensions=3)
    plot_convergence_comparison(dimensions=3)
    
    # Save results to CSV
    df_2d.to_csv('results_2d.csv', index=False)
    df_3d.to_csv('results_3d.csv', index=False)
    
    print("\nResults have been saved to CSV files and plots have been generated.")