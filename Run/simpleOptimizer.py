import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import tensorflow as tf

from FruitDetector import FruitDetector

# Parameter ranges to optimize with their limits and types
PARAM_RANGES = {
    "learning_rate": (1e-5, 5e-3, "log"),      # Learning rate
    "batch_size": (16, 64, "int"),             # Batch size
    "max_epochs": (10, 40, "int"),             # Maximum epochs
}

Params = Dict[str, Union[int, float, bool]]
FitnessFunc = Callable[[Params], float]


@dataclass
class Bee:
    """Represents a solution/food source in the ABC algorithm."""

    parameters: Params
    fitness: float
    trials: int = 0


class ArtificialBeeOptimizer:
    """Artificial Bee Colony optimizer specialized for hyperparameter search."""
#region Main functions of the ABC optimizer
    def __init__(self, colony_size: int , max_trials: int ):
        if colony_size < 2:
            raise ValueError("colony_size must be at least 2.")
        self.colony_size = colony_size
        self.max_trials = max_trials
        self.fitness_function: FitnessFunc = None
        self.colony: List[Bee] = []
        self.best_bee: Bee = None
        self._fitness_cache: Dict[Tuple[Tuple[str, Union[int, float, bool]], ...], float] = {}

    def _cache_key(self, params: Params) -> Tuple[Tuple[str, Union[int, float, bool]], ...]:
        """Create a hashable key for caching fitness evaluations."""
        normalized_items = []
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, float):
                value = round(value, 6)
            normalized_items.append((key, value))
        return tuple(normalized_items)

    def _evaluate(self, params: Params) -> float:
        """Evaluate (with memoization) the fitness of a parameter set."""
        key = self._cache_key(params)
        if key not in self._fitness_cache:
            self._fitness_cache[key] = self.fitness_function(params)
        return self._fitness_cache[key]

    @staticmethod
    def _sample_param(min_val, max_val, param_type):
        if param_type == "log":
            return 10 ** random.uniform(math.log10(min_val), math.log10(max_val))
        if param_type == "int":
            return random.randint(int(min_val), int(max_val))
        if param_type == "bool":
            return bool(random.getrandbits(1))
        return random.uniform(min_val, max_val)

    def _create_random_solution(self) -> Params:
        return {
            param: self._sample_param(*PARAM_RANGES[param])
            for param in PARAM_RANGES
        }

    @staticmethod
    def _clamp_value(value, min_val, max_val, param_type):
        value = max(min_val, min(max_val, value))
        if param_type == "int":
            value = int(round(value))
        elif param_type == "bool":
            value = bool(round(value))
        return value

    def _mutate(self, source_idx: int, target_idx: int = None) -> Params:
        """Generate a neighbor solution using the ABC perturbation rule."""
        source = self.colony[source_idx].parameters
        neighbor = source.copy()
        param = random.choice(list(PARAM_RANGES.keys()))
        min_val, max_val, param_type = PARAM_RANGES[param]

        if param_type == "bool":
            neighbor[param] = not neighbor[param]
            return neighbor

        if target_idx is None:
            candidates = [i for i in range(self.colony_size) if i != source_idx]
            target_idx = random.choice(candidates)
        target = self.colony[target_idx].parameters
        phi = random.uniform(-1, 1)

        current_val = source[param]
        partner_val = target[param]
        if param_type == "log":
            current_log = math.log10(current_val)
            partner_log = math.log10(partner_val)
            mutated_log = current_log + phi * (current_log - partner_log)
            new_val = 10 ** mutated_log
        else:
            new_val = current_val + phi * (current_val - partner_val)

        neighbor[param] = self._clamp_value(new_val, min_val, max_val, param_type)
        return neighbor

    def initialize(self, fitness_function: FitnessFunc, initial_solutions: List[Params] = None):
        """Initialize food sources (bees) and evaluate first fitnesses."""
        self.fitness_function = fitness_function
        self.colony = []
        self._fitness_cache.clear()

        initial_solutions = initial_solutions or []
        for params in initial_solutions[: self.colony_size]:
            candidate = params.copy()
            fitness = self._evaluate(candidate)
            bee = Bee(parameters=candidate, fitness=fitness)
            self.colony.append(bee)

        while len(self.colony) < self.colony_size:
            params = self._create_random_solution()
            fitness = self._evaluate(params)
            bee = Bee(parameters=params, fitness=fitness)
            self.colony.append(bee)

        self.best_bee = max(self.colony, key=lambda b: b.fitness)

    def _probabilities(self) -> List[float]:
        """Compute selection probabilities for onlooker bees."""
        min_fit = min(bee.fitness for bee in self.colony)
        #Shift is added to ensure all fitnesses are positive because probabilities must be non-negative
        shift = -min_fit + 1e-6 if min_fit < 0 else 1e-6
        scores = [bee.fitness + shift for bee in self.colony]
        total = sum(scores)
        return [score / total for score in scores]

    @staticmethod
    def _roulette_wheel(probabilities: List[float]) -> int:
        r = random.random()
        cumulative = 0.0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return idx
        return len(probabilities) - 1

    def _greedy_select(self, idx: int, candidate_params: Params):
        """Replace the bee solution if the candidate is better."""
        candidate_fitness = self._evaluate(candidate_params)
        bee = self.colony[idx]
        if candidate_fitness > bee.fitness:
            self.colony[idx] = Bee(parameters=candidate_params, fitness=candidate_fitness)
            if candidate_fitness > self.best_bee.fitness:
                self.best_bee = self.colony[idx]
        else:
            bee.trials += 1
            self.colony[idx] = bee

    def optimize(self, iterations: int = 5) -> Bee:
        """Run the ABC optimization loop and return the best bee found."""
        if not self.colony:
            raise RuntimeError("Call initialize() with a fitness function before optimize().")

        for _ in range(iterations):
            # Employed bees phase
            for idx in range(self.colony_size):
                neighbor = self._mutate(idx)
                self._greedy_select(idx, neighbor)

            # Onlooker bees phase
            probabilities = self._probabilities()
            for _ in range(self.colony_size):
                selected_idx = self._roulette_wheel(probabilities)
                neighbor = self._mutate(selected_idx)
                self._greedy_select(selected_idx, neighbor)

            # Scout bees phase
            for idx in range(self.colony_size):
                if self.colony[idx].trials >= self.max_trials:
                    params = self._create_random_solution()
                    fitness = self._evaluate(params)
                    self.colony[idx] = Bee(parameters=params, fitness=fitness, trials=0)
                    if fitness > self.best_bee.fitness:
                        self.best_bee = self.colony[idx]

        return self.best_bee

#endregion
DEFAULT_BASELINE_PARAMS: Params = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 13,
}
def run_optimizer_and_apply(
    train_dir: str,
    val_dir: str,
    num_classes: int ,
    img_size: int,
    colony_size: int,
    iterations:int,
    train_epochs_override:int,
    max_trials: int,
    data_fraction: float 
):

    def fitness_function(params: Params   ) -> float:
        detector = FruitDetector(img_size, num_classes)

        # Keep image augmentation fixed; only CNN hyperparameters are optimized.
        train_gen, val_gen = detector.create_data_generators(
            train_dir,
            int(params["batch_size"]),
            val_dir
        )

        detector.build_model(params["learning_rate"])

        effective_epochs = int(params["max_epochs"])

        if train_epochs_override is not None:
            effective_epochs = min(effective_epochs, int(train_epochs_override))
        effective_epochs = max(1, effective_epochs)

        try:
            history = detector.train(
                train_gen,
                val_gen,
                epochs=effective_epochs,
                data_fraction=data_fraction,
            )
            fitness = max(history.history.get("val_accuracy", [0.0]))
        finally:
            #In case of any error, clear the session to avoid GPU memory leaks
            tf.keras.backend.clear_session()
            
        return fitness

    optimizer = ArtificialBeeOptimizer(colony_size=colony_size, max_trials=max_trials)
    seed_params =  DEFAULT_BASELINE_PARAMS

    
    optimizer.initialize(fitness_function
        , initial_solutions=[seed_params])
    best_bee = optimizer.optimize(iterations=iterations)


    detector = FruitDetector(img_size, num_classes)

    train_gen, val_gen = detector.create_data_generators(
        train_dir,
        int(best_bee.parameters.get("batch_size", 32)),
        val_dir
    )

    detector.build_model(best_bee.parameters.get("learning_rate", 1e-3))

    final_epochs = int(best_bee.parameters.get("max_epochs", 10))
    final_epochs = min(final_epochs, 10)
    history = detector.train(
        train_gen,
        val_gen,
        epochs=max(1, final_epochs),
        data_fraction=1.0,
    )

    detector.plot_training_history()

    return detector, history, best_bee.parameters


if __name__ == "__main__":
    # Path configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
    TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train", "Fruit")
    VAL_DIR = os.path.join(PROJECT_ROOT, "dataset", "val", "Fruit")
    # Run optimization and training
    detector, history = run_optimizer_and_apply(TRAIN_DIR, VAL_DIR)
