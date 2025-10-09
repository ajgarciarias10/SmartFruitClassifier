from FruitDetector import FruitDetector  # Fixed import

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 5  # apple, banana, cucumber, grapefruit, pomegranate
LEARNING_RATE = 0.001

# Dataset paths - adjust these to your directory structure
TRAIN_DIR = 'dataset/train'
VAL_DIR = 'dataset/val'
TEST_DIR = 'dataset/test'

# Main execution
if __name__ == "__main__":
    print("=== Fruit Detection Model ===\n")

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

    # Debug: check shapes of data and labels
    x_batch, y_batch = next(train_gen)
    print(f"Train batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
    x_val_batch, y_val_batch = next(val_gen)
    print(f"Val batch shape: {x_val_batch.shape}, Labels shape: {y_val_batch.shape}")

    # Build model
    print("\nBuilding model...")
    model = detector.build_model(LEARNING_RATE, use_transfer_learning=True)  # Added LEARNING_RATE parameter
    model.summary()

    # Train model
    print("\nStarting training...")
    history = detector.train(train_gen, val_gen, epochs=EPOCHS)

    # Plot results
    detector.plot_training_history()

    # Save model
    detector.save_model('final_fruit_model.h5')

    print("\n=== Training Complete ===")

    # Example prediction (uncomment to use)
    # predicted_fruit, confidence, probs = detector.predict_image(
    #     'test_image.jpg',
    #     class_names
    # )
    # print(f"\nPredicted: {predicted_fruit} (Confidence: {confidence:.2f}%)")