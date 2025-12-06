import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))

# Ensure project root is on the import path when running from VS Code or other IDEs
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FruitDetector import FruitDetector
from simpleOptimizer import run_optimizer_and_apply
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Save the best model (callbacks restore best weights during training)
best_model_path = os.path.join(PROJECT_ROOT, 'best_fruit_model.keras')
detector.save_model(best_model_path)
print(f"Best model saved to: {best_model_path}")

# Plot results
detector.plot_training_history()


print("\n=== Training Complete ===")

# Show final dataset summary
print("\n Final Dataset Summary:")
dataset_results = validate_dataset_structure(DATASET_ROOT)
print_dataset_summary(dataset_results)

# Check saved model
model_info = check_model_file(best_model_path)
print(f"\n {model_info['message']}")

# Test evaluation on TEST_DIR
print("\n" + "="*60)
print("TESTING ON TEST DATASET")
print("="*60)

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model on test set
print("\nEvaluating model on test dataset...")
test_results = detector.evaluate(test_generator)

# Save the model after test evaluation
test_model_filename = f"fruit_model_test_acc{test_results[1]:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
test_model_path = os.path.join(PROJECT_ROOT, test_model_filename)
detector.save_model(test_model_path)
print(f"\nTest model saved to: {test_model_filename}")
