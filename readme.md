# SmartFruit Classifier

Computer vision system that classifies fruit images using a custom CNN optimized with bio-inspired algorithms. The pipeline covers dataset preparation, automated hyperparameter search, and training of deployment-ready models.

## Supported Fruits

- Apple
- Banana
- Cucumber
- Grapefruit
- Pomegranate

## Key Features

- **Specialized CNN** (`Run/FruitDetector.py`): sequential network with four convolutional blocks, `Dropout` regularization, and accuracy/precision/recall metrics.
- **Configurable data augmentation**: rotation, translation, zoom, shear, and horizontal flip controlled by the optimized hyperparameters.
- **Robust training**: `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks save `best_fruit_model.keras` and the training curves in `training_history.png`.
- **Automatic optimization** (`Run/simpleOptimizer.py`): Artificial Bee Colony (ABC) implementation that tunes learning rate, batch size, max epochs, and augmentation parameters using fractions of the dataset to speed up the search.
- **Dataset utilities** (`Utilities/DatasetManagement`): structure validation, Open Images/FiftyOne downloads, and helper scripts to fill gaps with Fruits360.
- **System diagnostics** (`Utilities/System/check_dependencies.py`): checks Python 3.13+, required packages, and GPU availability.
- **Metaheuristic benchmarks** (`Run/compare_pso_abc.py`, `Run/benchmark_comparison.py`, `Run/test_ackley_pso.py`): compare ABC vs PSO on Ackley/Rastrigin functions.

## Requirements

- Python **3.13 or newer** (verified with `python Utilities/System/check_dependencies.py`).
- Required packages: `tensorflow`, `numpy<2.0`, `matplotlib`, `pillow`.
- Optional packages: `fiftyone`, `opencv-python`, `kaggle` (needed only for dataset scripts).

## Project Structure

```
SmartFruitClassifier/
|-- Run/
|   |-- main.py
|   |-- FruitDetector.py
|   |-- simpleOptimizer.py
|   |-- benchmark_comparison.py | compare_pso_abc.py | test_ackley_pso.py
|-- Utilities/
|   |-- DatasetManagement/  # Dataset download and validation
|   |-- System/             # Environment checks
|-- dataset/
|   |-- train/Fruit/<Class>/
|   |-- val/Fruit/<Class>/
|   |-- test/Fruit/<Class>/
|-- training_history.png
|-- best_fruit_model.keras (generated after training)
```

## Dataset and Licenses

- **Kaggle (Fruits360)**: https://www.kaggle.com/datasets/moltean/fruits
- **Open Images v7**: https://g.co/dataset/open-images

> Google licenses Open Images under CC BY 4.0; individual assets often use CC BY 2.0. Confirm the license of each sub-dataset before production use.

Recommended dataset split (about 10,000 images per class combining sources):
- 80% training -> `dataset/train/Fruit`
- 10% validation -> `dataset/val/Fruit`
- 10% test -> `dataset/test/Fruit`

Scripts `Utilities/DatasetManagement/loadFiftyOne.py` and `import_local_fruits360.py` download and organize the data into those folders.

## Quick Start

1. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv fruit_classifier_env
   # Linux / macOS
   source fruit_classifier_env/bin/activate
   # Windows
   .\fruit_classifier_env\Scripts\activate
   ```
2. **Install minimum dependencies**  
   ```bash
   pip install "numpy<2.0" tensorflow matplotlib pillow
   ```
   Add `fiftyone opencv-python kaggle` when you need the dataset helpers.
3. **Verify the environment**  
   ```bash
   python Utilities/System/check_dependencies.py
   ```
4. **Prepare the dataset**  
   - Inspect per-class counts with `python Utilities/DatasetManagement/checkHowManyFruits.py`.
   - (Optional) Download from Open Images via `python Utilities/DatasetManagement/loadFiftyOne.py`.
   - Fill remaining gaps with Fruits360 using `python Utilities/DatasetManagement/import_local_fruits360.py`.
5. **Train and optimize**  
   ```bash
   python Run/main.py
   ```
   This runs the ABC optimizer, applies the best hyperparameters, and trains the final model.

## Recommended Workflow

1. `validate_dataset_structure` (called inside `Run/main.py`) ensures every class has files in train/val/test.
2. The optimizer (`run_optimizer_and_apply`) defaults to `data_fraction=0.15` and up to 4 epochs per trial to explore quickly.
3. After selecting the best colony, the final model trains on the full dataset, writes `training_history.png`, and saves weights to `best_fruit_model.keras`.
4. Use `FruitDetector.predict_image(...)` for inference on new samples; see the commented example at the end of `Run/main.py`.

## Outputs

- `training_history.png`: accuracy and loss curves for train/val.
- `best_fruit_model.keras`: final model weights (native Keras format).
- `final_fruit_model.h5`: legacy format for compatibility.
- `Run/optimizer_results/` and `Run/pso_ackley_results/`: logs from optimization experiments.

## Troubleshooting

- **TensorFlow with NumPy 2.x**: if `_ARRAY_API` warnings appear, reinstall with `pip install "numpy<2.0" tensorflow`.
- **No GPU detected**: ensure drivers and CUDA/cuDNN versions match the TensorFlow build.
- **Incomplete dataset**: `Utilities/DatasetManagement/utils.py` exposes `validate_dataset_structure` and `print_dataset_summary` to find missing classes.

## Next Steps

- Extend the optimizer to search over architecture choices (filters/layers).
- Provide a REST API or CLI for batch inference.
- Track experiments (learning rate, accuracy, etc.) in MLflow or another experiment tracker.
