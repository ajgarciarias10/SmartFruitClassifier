import os
import sys

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Ensure project root is on the import path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FruitDetector import FruitDetector
from simpleOptimizer import run_optimizer_and_apply



# --- PARAMETER CONFIGURATION ---
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5  # apple, banana, cucumber, Grapefruit, PomeGranate
LEARNING_RATE = 0.001

# Optimizer Parameters (ABC Algorithm)
COLONY_SIZE = 3
ITERATIONS = 2
TRAIN_EPCHS_OVERRIDE = 4
DATA_FRACTION = 0.2
MAX_TRIALS = 4

# Dataset Paths
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train', 'Fruit')
VAL_DIR = os.path.join(DATASET_ROOT, 'val', 'Fruit')
TEST_DIR = os.path.join(DATASET_ROOT, 'test', 'Fruit')

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Fruit Classification Model \n")

    print("Hyperparameter Optimization (ABC) Starting...")
    
    detector, history, best_parameters = run_optimizer_and_apply(
        TRAIN_DIR,
        VAL_DIR,
        NUM_CLASSES,
        IMG_SIZE,
        COLONY_SIZE,
        ITERATIONS,
        TRAIN_EPCHS_OVERRIDE,
        MAX_TRIALS,     
        DATA_FRACTION  
    )
    
    print("Optimization completed. Using the best hyperparameters found.")
    print(f"Best parameters: {best_parameters}")

    # Plot results
    detector.plot_training_history()

    print("\n=== Training Complete ===")
    
    # Test the model (Only if test directory exists)
    if os.path.exists(TEST_DIR):
        print("\nEvaluating on Test Set...")
        try:
            # Note: Ensure batch_size is an integer
            test_loss, test_acc, test_prec, test_rec = detector.evaluate_on_test(
                TEST_DIR, 
                int(best_parameters['batch_size'])
            )
            print(f"Test -> loss: {test_loss:.4f}, acc: {test_acc:.4f}, prec: {test_prec:.4f}, rec: {test_rec:.4f}")
        except Exception as e:
            print(f"Error during testing: {e}")
    else:
        print(f"Warning: Test directory not found at {TEST_DIR}")

    print("\n=== Evaluation Complete ===")

    # Save final model
    save_path = os.path.join(BASE_DIR, 'final_fruit_model.h5')
    detector.save_model(save_path)
    print(f"Model saved at: {save_path}")

    print("\n=== Program Finished ===")