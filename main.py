from Utilities.FruitDetector import FruitDetector
from Utilities.DatasetManagement.utils import validate_dataset_structure, print_dataset_summary, check_model_file
import os #Used to manage file paths

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 5  # apple, banana, cucumber, grapefruit, pomegranate
LEARNING_RATE = 0.001

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train', 'Fruit')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'val', 'Fruit')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test', 'Fruit')

# Main execution
print("Fruit Classification  Model \n")

# Initialize detector
detector = FruitDetector(IMG_SIZE, NUM_CLASSES)

# Create data generators
print("Loading data...")
train_gen, val_gen = detector.create_data_generators(TRAIN_DIR, BATCH_SIZE, VAL_DIR)

# Print class names
class_names = list(train_gen.class_indices.keys())
print(f"\nClasses: {class_names}")
print(f"Number of training samples: {train_gen.samples}")
print(f"Number of validation samples: {val_gen.samples}")


# Build model
print("\nBuilding model...")
model = detector.build_model(LEARNING_RATE)
model.summary()

# Train model
print("\nStarting training...")
history = detector.train(train_gen, val_gen, epochs=EPOCHS)

# Plot results
detector.plot_training_history()


print("\n=== Training Complete ===")

# Show final dataset summary
print("\n📊 Final Dataset Summary:")
dataset_results = validate_dataset_structure('dataset')
print_dataset_summary(dataset_results)

# Check saved model
model_info = check_model_file('final_fruit_model.h5')
print(f"\n💾 {model_info['message']}")

    # Example prediction (uncomment to use)
    # predicted_fruit, confidence, probs = detector.predict_image(
    #     'test_image.jpg',
    #     class_names
    # )
    # print(f"\nPredicted: {predicted_fruit} (Confidence: {confidence:.2f}%)")