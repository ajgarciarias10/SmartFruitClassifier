"""
Transfer Learning Training Script with Random Search
Train EfficientNet-B0 with frozen base and hyperparameter search
Usage:
    python Run/train_simple.py --trials 20 --epochs 15
"""

import sys
import argparse
import json
import random
import math
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transfer_model import TransferLearningModel

# Configuration
IMG_SIZE = 224
NUM_CLASSES = 5
DEFAULT_TRIALS = 1
DEFAULT_EPOCHS = 20

# Paths
DATASET_ROOT = PROJECT_ROOT / 'dataset'
TRAIN_DIR = DATASET_ROOT / 'train' / 'Fruit'
VAL_DIR = DATASET_ROOT / 'val' / 'Fruit'
TEST_DIR = DATASET_ROOT / 'test' / 'Fruit'
TEST_RESULTS_DIR = BASE_DIR / 'test_results'
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Samplers for hyperparameter space
def sample_loguniform(low, high):
    return float(10 ** random.uniform(math.log10(low), math.log10(high)))

def sample_categorical(options):
    return random.choice(options)

def sample_uniform(low, high):
    return float(random.uniform(low, high))


def compute_class_weights(gen):
    labels = getattr(gen, 'classes', None)
    if labels is None:
        counts = {}
        for fname in gen.filenames:
            cls = fname.split('/')[0]
            counts[cls] = counts.get(cls, 0) + 1
        classes = sorted(list(gen.class_indices.keys()))
        counts_arr = [counts.get(c, 0) for c in classes]
        total = sum(counts_arr)
        weights = {i: total / (len(classes) * (counts_arr[i] if counts_arr[i] > 0 else 1)) for i in range(len(classes))}
        return weights
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]
    weights = {int(u): float(total / (len(unique) * c)) for u, c in zip(unique, counts)}
    return weights


def main(trials=DEFAULT_TRIALS, epochs=DEFAULT_EPOCHS):
    print("\n" + "="*70)
    print("TRANSFER LEARNING - EFFICIENTNET-B0 WITH RANDOM SEARCH")
    print("="*70)
    print(f"Random Search Configuration:")
    print(f"  Trials: {trials}")
    print(f"  Epochs per trial: {epochs}")
    print("="*70 + "\n")
    
    results = []
    best_val_acc = -1.0
    best_model_path = None
    
    for trial_num in range(1, trials + 1):
        # Sample hyperparameters
        lr = sample_loguniform(1e-5, 1e-3)
        batch_size = sample_categorical([16, 32, 64])
        dense_units = sample_categorical([128, 256, 512])
        dropout = sample_uniform(0.1, 0.6)
        l2_reg = sample_loguniform(1e-6, 1e-3) if random.random() < 0.5 else 0.0
        label_smoothing = sample_uniform(0.0, 0.1) if random.random() < 0.5 else 0.0
        
        print(f"\n{'='*70}")
        print(f"TRIAL {trial_num}/{trials}")
        print(f"{'='*70}")
        print(f"Hyperparameters:")
        print(f"  LR={lr:.6g}, Batch={batch_size}, Dense={dense_units}")
        print(f"  Dropout={dropout:.3f}, L2={l2_reg:.6g}, Label_smooth={label_smoothing:.3f}")
        print(f"{'='*70}\n")
        
        # Build model
        model = TransferLearningModel(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
        model.build_model(
            learning_rate=lr,
            dense_units=dense_units,
            dropout_rate=dropout,
            freeze_base=True,
            l2_reg=l2_reg,
            label_smoothing=label_smoothing
        )
        
        # Load data
        train_gen, val_gen = model.create_data_generators(
            train_dir=str(TRAIN_DIR),
            val_dir=str(VAL_DIR),
            batch_size=batch_size,
            augmentation=True
        )
        
        print(f"Data loaded:")
        print(f"  Train: {train_gen.samples} samples")
        print(f"  Val: {val_gen.samples} samples")
        print(f"  Classes: {list(train_gen.class_indices.keys())}\n")
        
        # Compute class weights
        class_weight = compute_class_weights(train_gen)
        
        # Train
        history = model.train(train_gen, val_gen, epochs=epochs, class_weight=class_weight)
        
        # Get best validation accuracy
        val_accs = history.history.get('val_accuracy', [])
        trial_best_val_acc = max(val_accs) if val_accs else 0.0
        
        # Save model for this trial
        model_name = f"trial_{trial_num}_valacc_{trial_best_val_acc:.4f}.keras"
        model_path = TEST_RESULTS_DIR / model_name
        model.save_model(str(model_path))
        
        # Save trial result
        trial_result = {
            'trial': trial_num,
            'params': {
                'learning_rate': lr,
                'batch_size': batch_size,
                'dense_units': dense_units,
                'dropout': dropout,
                'l2_reg': l2_reg,
                'label_smoothing': label_smoothing
            },
            'best_val_accuracy': float(trial_best_val_acc),
            'model_path': str(model_path)
        }
        results.append(trial_result)
        
        # Update best
        if trial_best_val_acc > best_val_acc:
            best_val_acc = trial_best_val_acc
            best_model_path = model_path
            try:
                import shutil
                shutil.copyfile(model_path, TEST_RESULTS_DIR / 'best_model.keras')
                print(f"\n✓ New best model! Val accuracy: {trial_best_val_acc:.4f}")
            except Exception as e:
                print(f"Warning: Could not copy best model: {e}")
        
        # Save partial results
        with open(TEST_RESULTS_DIR / 'random_search_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'best_val_accuracy': best_val_acc,
                'best_model': str(best_model_path),
                'trials_completed': trial_num
            }, f, indent=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("RANDOM SEARCH COMPLETED!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved: {TEST_RESULTS_DIR / 'best_model.keras'}")
    print(f"All results saved: {TEST_RESULTS_DIR / 'random_search_results.json'}")
    print(f"{'='*70}\n")
    
    # Show top 5 trials
    sorted_results = sorted(results, key=lambda x: x['best_val_accuracy'], reverse=True)
    print("\nTop 5 trials:")
    for i, result in enumerate(sorted_results[:5], 1):
        params = result['params']
        print(f"  {i}. Val Acc: {result['best_val_accuracy']:.4f} | "
              f"LR={params['learning_rate']:.6g}, Batch={params['batch_size']}, "
              f"Dense={params['dense_units']}, Dropout={params['dropout']:.3f}")
    
    # Evaluate best model on TEST_DIR
    if TEST_DIR.exists():
        print(f"\n{'='*70}")
        print("EVALUATING BEST MODEL ON TEST SET")
        print(f"{'='*70}\n")
        
        best_model = TransferLearningModel(img_size=IMG_SIZE, num_classes=NUM_CLASSES)
        from tensorflow.keras.models import load_model
        best_model.model = load_model(TEST_RESULTS_DIR / 'best_model.keras')
        
        test_gen, _ = best_model.create_data_generators(
            train_dir=str(TEST_DIR),
            val_dir=str(TEST_DIR),
            batch_size=32,
            augmentation=False
        )
        
        print(f"Test samples: {test_gen.samples}\n")
        test_results = best_model.evaluate(test_gen)
        
        # Save test results to JSON
        test_metrics = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'best_model_path': str(TEST_RESULTS_DIR / 'best_model.keras'),
            'best_val_accuracy': best_val_acc
        }
        with open(TEST_RESULTS_DIR / 'test_evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"\n✓ Test results saved: {TEST_RESULTS_DIR / 'test_evaluation_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning with Random Search')
    parser.add_argument('--trials', type=int, default=DEFAULT_TRIALS,
                        help=f'Number of random search trials (default: {DEFAULT_TRIALS})')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help=f'Epochs per trial (default: {DEFAULT_EPOCHS})')
    args = parser.parse_args()
    
    main(trials=args.trials, epochs=args.epochs)
