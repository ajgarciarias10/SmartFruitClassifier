"""
Test script to verify random_search with Excel export and plotting
Run this to see the full tuning pipeline with results
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from random_search import random_search

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 5
N_TRIALS = 5  # Use small number for testing
EPOCHS = 3    # Use few epochs for quick test

# Paths
DATASET_ROOT = PROJECT_ROOT / 'dataset'
TRAIN_DIR = DATASET_ROOT / 'train' / 'Fruit'
VAL_DIR = DATASET_ROOT / 'val' / 'Fruit'

def main():
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH RANDOM SEARCH")
    print("="*60)
    print(f"Test Configuration:")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Epochs per trial: {EPOCHS}")
    print(f"  Output directory: {PROJECT_ROOT}")
    print("="*60 + "\n")
    
    # Run random search
    best_hparams = random_search(
        train_dir=str(TRAIN_DIR),
        val_dir=str(VAL_DIR),
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        n_trials=N_TRIALS,
        epochs=EPOCHS,
        augmentation=True,
        output_dir=PROJECT_ROOT
    )
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nBest Hyperparameters Found:")
    for key, value in best_hparams.items():
        print(f"  {key}: {value}")
    
    print(f"\nOutput Files Generated:")
    print(f"  ✓ Excel file: {PROJECT_ROOT / 'hyperparameter_tuning_results.xlsx'}")
    print(f"  ✓ Plots saved in: {PROJECT_ROOT / 'tuning_plots'}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
