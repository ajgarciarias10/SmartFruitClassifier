"""
Simple Transfer Learning Training Script
Train EfficientNet-B0 with frozen base on fruit dataset
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from random_search import random_search
from transfer_model import TransferLearningModel

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 5

# Hyperparameters to test (you can modify these)
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DENSE_UNITS = 512
DROPOUT_RATE = 0.5
EPOCHS = 20

# Paths
DATASET_ROOT = PROJECT_ROOT / 'dataset'
TRAIN_DIR = DATASET_ROOT / 'train' / 'Fruit'
VAL_DIR = DATASET_ROOT / 'val' / 'Fruit'
TEST_DIR = DATASET_ROOT / 'test' / 'Fruit'

#Hyperammeters for random search
N_TRIALS = 20 # Depending on the GPU


def main():
    print("\n" + "="*60)
    print("TRANSFER LEARNING - EFFICIENTNET-B0")
    print("="*60)
    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Dense Units: {DENSE_UNITS}")
    print(f"  Dropout Rate: {DROPOUT_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print("="*60 + "\n")
    random_search
    best_hparams = random_search(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        n_trials=N_TRIALS,     # puedes subir/bajar según recursos
        epochs=EPOCHS,       # menos epochs en la búsqueda, más rápidos los trials
        augmentation=True,
        output_dir=PROJECT_ROOT
    )
    # Create model
    model = TransferLearningModel(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
    
    
    # Build with frozen base
    model.build_model(
        learning_rate=best_hparams["learning_rate"],
        dense_units=best_hparams["dense_units"],
        dropout_rate=best_hparams["dropout_rate"],
        freeze_base=True,  # Freeze EfficientNet base
        l2_reg=best_hparams.get("l2_reg", 1e-4)
    )
    
    # Create data generators
    print("Loading data...")
    train_gen, val_gen = model.create_data_generators(
        train_dir=str(TRAIN_DIR),
        val_dir=str(VAL_DIR),
        batch_size=best_hparams["batch_size"],
        augmentation=True
    )
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}\n")
    
    # Train
    history = model.train(train_gen, val_gen, epochs=EPOCHS)
    
    # Plot results
    plot_path = PROJECT_ROOT / 'training_history_transfer.png'
    model.plot_history(save_path=str(plot_path))
    
    # Evaluate on test set
    if TEST_DIR.exists():
        print("\nEvaluating on test set...")
        test_gen = model.create_data_generators(
            train_dir=str(TEST_DIR),
            val_dir=str(TEST_DIR),
            batch_size=best_hparams["batch_size"],
            augmentation=False
        )[0]
        
        model.evaluate(test_gen)
    
    # Save final model
    save_path = PROJECT_ROOT / 'models' / 'efficientnet-b0' / 'final_model.keras'
    model.save_model(str(save_path))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best model saved at: {PROJECT_ROOT / 'models' / 'efficientnet-b0' / 'best_model.keras'}")
    print(f"Final model saved at: {save_path}")
    print(f"Training plot: {plot_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
