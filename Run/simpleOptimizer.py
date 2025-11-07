import random
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Union, Callable
from FruitDetector import FruitDetector

# Parameter ranges to optimize with their limits and types
PARAM_RANGES = {
    "learning_rate": (1e-5, 1e-2, "log"),      # Learning rate
    "batch_size": (16, 64, "int"),             # Batch size
    "max_epochs": (10, 30, "int"),             # Maximum epochs
}

# Tipos para anotaciones
Params = Dict[str, Union[int, float]]
FitnessFunc = Callable[[Params], float]

@dataclass
class Bee:
    """Represents a bee in the colony."""
    parameters: Params           # Parameters that this solution represents
    fitness: float = float('-inf')  # Fitness value
    trials: int = 0             # Number of trials without improvement

class SimpleOptimizer:
    """Simplified optimizer based on the ABC (Artificial Bee Colony) algorithm."""
    
    def __init__(self, colony_size: int = 10, max_trials: int = 5):
        """
        Initialize the optimizer.
        
        Args:
            colony_size: Number of bees in the colony
            max_trials: Maximum number of trials before abandoning a solution
        """
        self.colony_size = colony_size
        self.max_trials = max_trials
        self.colony: List[Bee] = []
        self.best_solution: Bee = None
        self.fitness_function: FitnessFunc = None

    def _create_random_solution(self) -> Params:
        """Generates a random solution within the allowed ranges."""
        solution = {}
        for param, (min_val, max_val, param_type) in PARAM_RANGES.items():
            if param_type == "log":
                value = 10 ** random.uniform(math.log10(min_val), math.log10(max_val))
            elif param_type == "int":
                value = int(random.uniform(min_val, max_val))
            else:  # float
                value = random.uniform(min_val, max_val)
            solution[param] = value
        return solution

    def initialize(self, fitness_function: FitnessFunc):
        """
        Initialize the colony and set the fitness function.
        
        Args:
            fitness_function: Function that evaluates the quality of a solution
        """
        self.fitness_function = fitness_function
        self.colony = []
        # Create initial population
        for _ in range(self.colony_size):
            bee = Bee(parameters=self._create_random_solution())
            bee.fitness = self.fitness_function(bee.parameters)
            self.colony.append(bee)
        
        # Identify initial best solution
        self.best_solution = max(self.colony, key=lambda b: b.fitness)

    def _generate_neighbor_solution(self, current: Params) -> Params:
        """Generates a neighboring solution by modifying parameters."""
        neighbor = current.copy()
        # Modify a random parameter
        param = random.choice(list(PARAM_RANGES.keys()))
        min_val, max_val, param_type = PARAM_RANGES[param]
        
        # Calculate new value
        
        current_val = current[param]
        variation = random.uniform(-1, 1)  # Variation factor
        if param_type == "log":
            new_val = current_val * (1 + variation)
        else:
            range_size = max_val - min_val
            new_val = current_val + range_size * variation

        # Ensure the value is within bounds
        new_val = max(min_val, min(max_val, new_val))
        if param_type == "int":
            new_val = int(round(new_val))
            
        neighbor[param] = new_val
        return neighbor

    def optimize(self, iterations: int = 50) -> Bee:
        """
        Executes the optimization process.
        
        Args:
            iterations: Number of algorithm iterations
            
        Returns:
            The best solution found
        """
        for _ in range(iterations):
            # Employed bees phase
            for i in range(self.colony_size):
                bee = self.colony[i]
                neighbor = self._generate_neighbor_solution(bee.parameters)
                neighbor_fitness = self.fitness_function(neighbor)
                
                # Update if we find a better solution
                if neighbor_fitness > bee.fitness:
                    self.colony[i] = Bee(parameters=neighbor, fitness=neighbor_fitness)
                    if neighbor_fitness > self.best_solution.fitness:
                        self.best_solution = self.colony[i]
                else:
                    bee.trials += 1

            # Onlooker bees phase
            # Select better solutions for further exploration
            fitness_sum = sum(max(0.0001, bee.fitness) for bee in self.colony)
            for i in range(self.colony_size):
                if random.random() < self.colony[i].fitness / fitness_sum:
                    neighbor = self._generate_neighbor_solution(self.colony[i].parameters)
                    neighbor_fitness = self.fitness_function(neighbor)
                    if neighbor_fitness > self.colony[i].fitness:
                        self.colony[i] = Bee(parameters=neighbor, fitness=neighbor_fitness)
                        if neighbor_fitness > self.best_solution.fitness:
                            self.best_solution = self.colony[i]

            # Scout bees phase
            # Replace stagnated solutions
            for i in range(self.colony_size):
                if self.colony[i].trials >= self.max_trials:
                    self.colony[i] = Bee(parameters=self._create_random_solution())
                    self.colony[i].fitness = self.fitness_function(self.colony[i].parameters)

        return self.best_solution

def example_usage():
    # Ejemplo de uso del optimizador
    def dummy_fitness(params: Params) -> float:
        """Función de fitness de ejemplo."""
        return -sum((p - 0.5) ** 2 for p in params.values())

    # Crear y ejecutar el optimizador
    optimizer = SimpleOptimizer(colony_size=10, max_trials=5)
    optimizer.initialize(dummy_fitness)
    best = optimizer.optimize(iterations=20)
    
    print("Mejor solución encontrada:")
    for param, value in best.parameters.items():
        print(f"{param}: {value}")
    print(f"Fitness: {best.fitness}")

def run_optimizer_and_apply(train_dir: str, val_dir: str, num_classes: int = 5, img_size: int = 224):
    """
    Runs the optimizer and applies the best hyperparameters found to the model.
    
    Args:
        train_dir: Directory with training data
        val_dir: Directory with validation data
        num_classes: Number of classes to classify
        img_size: Input image size
    
    Returns:
        tuple: (trained model, training history)
    """
    def fitness_function(params: Params) -> float:
        """Fitness function that trains and evaluates the model with the given parameters."""
        detector = FruitDetector(img_size, num_classes)
        
        # Crear generadores de datos
        train_gen, val_gen = detector.create_data_generators(
            train_dir, 
            int(params['batch_size']), 
            val_dir
        )
        
        # Build model with only learning rate parameter
        model = detector.build_model(params['learning_rate'])
        
        # Train for a few epochs for quick evaluation
        history = detector.train(
            train_gen, 
            val_gen, 
            epochs=int(params['max_epochs'])
        )
        
        # Use the final validation accuracy as fitness metric
        return history.history['val_accuracy'][-1]

    # Crear y ejecutar el optimizador
    optimizer = SimpleOptimizer(colony_size=10, max_trials=5)
    optimizer.initialize(fitness_function)
    best_solution = optimizer.optimize(iterations=20)
    
    print("\nBest hyperparameters found:")
    for param, value in best_solution.parameters.items():
        print(f"{param}: {value}")
    print(f"Best validation accuracy: {best_solution.fitness:.4f}")
    
    # Apply the best parameters found to the final model
    detector = FruitDetector(img_size, num_classes)
    train_gen, val_gen = detector.create_data_generators(
        train_dir, 
        int(best_solution.parameters['batch_size']), 
        val_dir
    )
    
    model = detector.build_model(best_solution.parameters['learning_rate'])
    
    # Train the final model with the best parameters
    history = detector.train(
        train_gen, 
        val_gen, 
        epochs=int(best_solution.parameters['max_epochs'])
    )
    
    # Visualize results
    detector.plot_training_history()
    
    return detector, history

if __name__ == "__main__":
    # Path configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'train', 'Fruit')
    VAL_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'val', 'Fruit')
    
    # Run optimization and training
    detector, history = run_optimizer_and_apply(TRAIN_DIR, VAL_DIR)