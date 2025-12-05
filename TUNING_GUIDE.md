# Hyperparameter Tuning with Random Search

## Overview
The improved `random_search.py` now includes:
- **Random Search**: Tests different hyperparameter combinations
- **Excel Export**: All results saved to `hyperparameter_tuning_results.xlsx`
- **Sensitivity Analysis**: 6 plots showing how each hyperparameter affects accuracy
- **Accuracy Tracking**: Comparison of train vs validation metrics
- **Distribution Analysis**: Histograms of hyperparameter distributions

## Features

### 1. Excel Output (`hyperparameter_tuning_results.xlsx`)
Contains all trial results with columns:
- **Trial**: Trial number
- **Dense_Units**: Number of neurons in dense layer (128, 256, 384, 512)
- **Dropout_Rate**: Dropout rate (0.5, 0.6, 0.7)
- **Learning_Rate**: Learning rate (10^-4.5 to 10^-3.5)
- **Batch_Size**: Batch size (32, 64)
- **L2_Reg**: L2 regularization (10^-5 to 10^-3)
- **Train_Accuracy**: Training accuracy for that trial
- **Val_Accuracy**: Validation accuracy for that trial
- **Train_Loss**: Training loss
- **Val_Loss**: Validation loss
- **Overfitting_Gap**: Difference between train and validation accuracy
- **Adjusted_Score**: Accuracy penalized by overfitting gap

### 2. Visualization Plots (`tuning_plots/`)

#### a) `sensitivity_analysis.png`
Six scatter plots showing impact of each hyperparameter:
1. Dense Units vs Validation Accuracy
2. Dropout Rate vs Validation Accuracy
3. Learning Rate vs Validation Accuracy
4. Batch Size vs Validation Accuracy
5. L2 Regularization vs Validation Accuracy
6. Overfitting Gap vs Validation Accuracy

Color intensity indicates validation accuracy (darker = better)

#### b) `accuracy_comparison.png`
Two line plots:
1. Train vs Validation Accuracy across trials
2. Train vs Validation Loss across trials

#### c) `hyperparameter_distributions.png`
Six histograms showing the distribution of:
1. Dense Units sampled
2. Dropout Rates sampled
3. Learning Rates sampled
4. Batch Sizes sampled
5. L2 Regularization values sampled
6. Overfitting Gaps observed

## Usage

### Option 1: Full Training Pipeline
```python
from random_search import random_search

best_hparams = random_search(
    train_dir="path/to/train",
    val_dir="path/to/val",
    img_size=224,
    num_classes=5,
    n_trials=20,
    epochs=20,
    augmentation=True,
    output_dir="output_directory"
)
```

### Option 2: Quick Test
Run the test script:
```bash
python test_random_search.py
```

This runs with small values (5 trials, 3 epochs) to verify everything works.

## How to Interpret Results

### Sensitivity Analysis
- **Peak accuracies**: Look for the hyperparameter values where points cluster at the top
- **Sweet spot**: Find the region with high accuracy and low overfitting gap
- **Color gradient**: Darker colors indicate better validation accuracy

### Accuracy Comparison
- **Converging lines**: Good generalization (train and val accuracy are close)
- **Diverging lines**: Overfitting (train accuracy high, val accuracy low)
- **Loss curves**: Should both decrease; if val loss increases while train loss decreases = overfitting

### Distributions
- Help understand which hyperparameter ranges were explored
- Useful for planning future searches with tighter ranges

## Best Practices

1. **Monitor Overfitting Gap**: The system penalizes models with high gap (train_acc - val_acc)
2. **Check Excel First**: Sort by "Adjusted_Score" to find best models
3. **Review Plots**: Visual confirmation of best regions for each hyperparameter
4. **Increase Trials**: More trials = better coverage of search space (but slower)
5. **Use L2 Regularization**: Helps prevent overfitting in dense layers

## File Locations

After running random_search, you'll find:
```
SmartFruitClassifier/
├── hyperparameter_tuning_results.xlsx    # Complete results table
├── best_random_search_model.keras        # Best model found
└── tuning_plots/
    ├── sensitivity_analysis.png
    ├── accuracy_comparison.png
    └── hyperparameter_distributions.png
```

## Next Steps

After finding good hyperparameters:
1. Use best_hparams to train final model with more epochs
2. Fine-tune around the best hyperparameter values
3. Consider expanding search space based on plots if better regions are visible
