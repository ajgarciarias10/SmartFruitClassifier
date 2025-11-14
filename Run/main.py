import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Ensure project root is on the import path when running from VS Code or other IDEs
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FruitDetector import FruitDetector
from simpleOptimizer import run_optimizer_and_apply
from Utilities.DatasetManagement.utils import (
    validate_dataset_structure,
    print_dataset_summary,
    check_model_file,
)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5  # apple, banana, cucumber, Grapefruit, PomeGranate
LEARNING_RATE = 0.001

# Dataset paths
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset')
TRAIN_DIR = os.path.join(DATASET_ROOT, 'train', 'Fruit')
VAL_DIR = os.path.join(DATASET_ROOT, 'val', 'Fruit')
TEST_DIR = os.path.join(DATASET_ROOT, 'test', 'Fruit')

# Main execution
print("Fruit Classification Model \n")

print("Particle Swarm Optimization of Hyperparameters Starting...")
detector, history = run_optimizer_and_apply(
    TRAIN_DIR,
    VAL_DIR,
    NUM_CLASSES,
    IMG_SIZE
)
print("Optimizing completed. Using the best hyperparameters found.")

# Plot results
detector.plot_training_history()


print("\n=== Training Complete ===")

# Show final dataset summary
print("\n Final Dataset Summary:")
dataset_results = validate_dataset_structure(DATASET_ROOT)
print_dataset_summary(dataset_results)

# Check saved model
model_info = check_model_file(os.path.join(PROJECT_ROOT, 'final_fruit_model.h5'))
print(f"\n {model_info['message']}")

# Example prediction (uncomment to use)
train_gen, _ = detector.create_data_generators(TRAIN_DIR, BATCH_SIZE, VAL_DIR)
class_names = list(train_gen.class_indices.keys())
predicted_fruit, confidence, probs = detector.predict_image(
    'test_image.jpg',
    class_names
)
print(f"\nPredicted: {predicted_fruit} (Confidence: {confidence:.2f}%)")
