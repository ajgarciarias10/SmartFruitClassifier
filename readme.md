# 🍎 SmartFruit Classifier

Computer vision system based on Transfer Learning with **EfficientNet-B0** that classifies fruit images with high accuracy. The project includes an interactive web application for real-time predictions, hyperparameter optimization using bio-inspired algorithms (ABC), and a complete pipeline from dataset preparation to deployment.

## 🌐 Live Web Application

### 🚀 Live Demo
**Try the application here:** [https://huggingface.co/spaces/toniariasss/SmartFruitClassiffier](https://huggingface.co/spaces/toniariasss/SmartFruitClassiffier)

The web application enables visual and interactive fruit classification through a modern interface developed with Gradio and deployed on Hugging Face Spaces:

### Main Interface
![Main Interface](websiteResults/Captura%20de%20pantalla%202025-12-09%20181752.png)

### Prediction Example - Apple
![Apple Prediction](websiteResults/Captura%20de%20pantalla%202025-12-09%20181808.png)

### Prediction Example - Grapefruit
![Grapefruit Prediction](websiteResults/Captura%20de%20pantalla%202025-12-09%20181830.png)

**Application Features:**
- ✅ **Deployed on Hugging Face Spaces** - 24/7 public access
- ✅ Responsive web interface with modern design (Gradio)
- ✅ Image upload via drag & drop or file selection
- ✅ Real-time prediction with probability visualization
- ✅ Support for 5 fruit classes
- ✅ Prediction confidence for each class
- ✅ Automatic preprocessing compatible with EfficientNet-B0

## 🧠 Model Architecture

### 📊 Evolution: Previous Model vs Current Model

| Aspect | Previous Model (Custom CNN) | Current Model (Transfer Learning) |
|---------|------------------------------|-----------------------------------|
| **Base Architecture** | Custom CNN from scratch | Pre-trained EfficientNet-B0 |
| **Total Parameters** | ~2-3M parameters | 4.38M parameters |
| **Trainable Parameters** | All (~2-3M) | Only top layers (~329K) |
| **Pre-training** | ❌ No | ✅ Yes (ImageNet) |
| **Test Accuracy** | ~85-90% | **>95%** |
| **Training Time** | Longer (more epochs) | Shorter (fast convergence) |
| **Overfitting Risk** | High | Low (robust features) |
| **Optimization** | ABC (bio-inspired) | Random Search + callbacks |
| **Preprocessing** | Manual normalization | EfficientNet preprocessing |
| **Regularization** | Basic Dropout | Dropout + L2 + Label Smoothing |

**🏆 Conclusion:** The Transfer Learning model (EfficientNet-B0) outperforms the custom CNN by **+5-10% accuracy** with **fewer trainable parameters**, demonstrating the effectiveness of transfer learning.

### 🎯 Current Model: Transfer Learning with EfficientNet-B0

The model uses **Transfer Learning** with **EfficientNet-B0** as a pre-trained backbone on ImageNet, combined with custom classification layers optimized through Random Search.

### Technical Specifications

**Base Model:** EfficientNet-B0 (frozen)
- Pre-trained on ImageNet
- 4,049,571 parameters (frozen)
- Input shape: 224×224×3
- Global Average Pooling

**Classification Head (trainable):**
```
Input (224×224×3)
    ↓
EfficientNet Preprocessing
    ↓
EfficientNet-B0 Base (frozen)
    ↓
Global Average Pooling
    ↓
Dropout (0.3-0.5)
    ↓
Dense (128-512 units, ReLU, L2 regularization)
    ↓
Dropout (0.18-0.3)
    ↓
Dense (5 units, Softmax)
```

**Total Parameters:** ~4,378,792
- **Trainable:** 329,221 (top layers)
- **Frozen:** 4,049,571 (EfficientNet base)

**Optimization:**
- Optimizer: Adam
- Learning Rate: 0.0001 - 0.001 (optimized)
- Loss: Categorical Crossentropy with label smoothing
- Metrics: Accuracy, Precision, Recall

**Regularization:**
- Dropout in top layers
- L2 weight decay
- Label smoothing
- Early Stopping with patience
- ReduceLROnPlateau

### Model Results

The final model achieves:
- **Test Accuracy:** >95%
- **Average Precision:** ~95%
- **Average Recall:** ~95%
- **Convergence:** ~15-25 epochs with early stopping

## 🍇 Supported Classes

- 🍎 Apple
- 🍌 Banana
- 🥒 Cucumber
- 🍊 Grapefruit
- 🍓 Pomegranate

## ⚡ Key Features

- **🎯 Transfer Learning with EfficientNet-B0** (`Run/transfer_model.py`): Pre-trained architecture on ImageNet with custom classification layers optimized through Random Search. **Outperforms previous CNN model by +5-10% accuracy.**
- **🌐 Deployed Web Application** (`Run/web_app/app.py` + Hugging Face): Modern web interface with Gradio deployed on [Hugging Face Spaces](https://huggingface.co/spaces/toniariasss/SmartFruitClassiffier) for public access.
- **🔬 Previous Model (Custom CNN)** (`Run/FruitDetector.py`): Custom CNN with 4 convolutional blocks as comparison baseline.
- **🔍 Automatic Hyperparameter Optimization**: 
  - Random Search to explore learning rate, dense units, dropout, L2 regularization and label smoothing
  - Artificial Bee Colony (ABC) for bio-inspired optimization (`Run/simpleOptimizer.py`)
  - PSO (Particle Swarm Optimization) implemented for benchmarking
- **📊 Configurable Data Augmentation**: rotation, translation, zoom, shear and horizontal flip controlled by optimized hyperparameters.
- **💪 Robust Training**: 
  - `EarlyStopping` to prevent overfitting
  - `ReduceLROnPlateau` for dynamic learning rate adjustment
  - `ModelCheckpoint` saves the best model in `best_model.keras`
  - Automatic generation of training curves
- **📈 Dataset Analysis** (`Run/analyze_dataset.py`): script to analyze dataset distribution and structure.
- **📋 Final Evaluation** (`Run/final_evaluation.py`): detailed comparison between models with confusion matrices.

## Requirements

- Python **3.10 or newer**
- Required packages: `tensorflow`, `numpy<2.0`, `matplotlib`, `pillow`, `gradio`
- Optional packages: `scikit-learn`, `seaborn` (for evaluation and visualization)

## 📁 Project Structure

```
SmartFruitClassifier/
|-- Run/
|   |-- transfer_model.py          # Transfer Learning EfficientNet-B0 model
|   |-- train_simple.py             # Training script with Random Search
|   |-- final_evaluation.py         # Model evaluation and comparison
|   |-- analyze_dataset.py          # Dataset distribution analysis
|   |-- web_app/
|   |   |-- app.py                  # Flask web application (local)
|   |   |-- templates/              # HTML templates
|   |   |-- static/                 # CSS and assets
|-- models/
|   |-- efficientnet-b0/
|   |   |-- best_model.keras        # Trained model (4.38M parameters)
|-- dataset/
|   |-- train/Fruit/<Class>/        # 80% training data
|   |-- val/Fruit/<Class>/          # 10% validation data
|   |-- test/Fruit/<Class>/         # 10% test data
|-- results/
|   |-- mconf/                      # Generated confusion matrices
|-- websiteResults/                 # Web application screenshots
|-- pretrained_results.json         # Experiment results
|-- readme.md
```

## Dataset and Licenses

- **Kaggle (Fruits360)**: https://www.kaggle.com/datasets/moltean/fruits
- **Open Images v7**: https://g.co/dataset/open-images

> Google licenses Open Images under CC BY 4.0; individual assets often use CC BY 2.0. Confirm the license of each sub-dataset before production use.

Recommended dataset split:
- 80% training -> `dataset/train/Fruit`
- 10% validation -> `dataset/val/Fruit`
- 10% test -> `dataset/test/Fruit`

The dataset must be organized in this structure before training the model.

## Quick Start

1. **Create a virtual environment (recommended)**  
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```

2. **Install dependencies**  
   ```bash
   pip install tensorflow numpy matplotlib pillow scikit-learn seaborn gradio
   ```

3. **Prepare the dataset**  
   - Organize images in `dataset/train/Fruit/<Class>/`, `dataset/val/Fruit/<Class>/` and `dataset/test/Fruit/<Class>/`
   - Analyze distribution with: `python Run/analyze_dataset.py`

4. **Train the model with Transfer Learning**  
   ```bash
   cd Run
   python train_simple.py
   ```
   This runs Random Search to optimize hyperparameters and trains the EfficientNet-B0 model.

5. **Evaluate the model**
   ```bash
   python Run/final_evaluation.py
   ```
   Generates confusion matrices and detailed metrics.

## Recommended Workflow

1. **Dataset Analysis**: Run `python Run/analyze_dataset.py` to verify class distribution
2. **Training**: Run `python Run/train_simple.py` which uses Random Search to find the best hyperparameters
3. **Evaluation**: Run `python Run/final_evaluation.py` to get detailed metrics and confusion matrices
4. **Local Deployment**: Start Flask app with `python Run/web_app/app.py`
5. **Cloud Deployment**: Use the deployed version on [Hugging Face Spaces](https://huggingface.co/spaces/toniariasss/SmartFruitClassiffier)

## 📤 Generated Outputs

- `models/efficientnet-b0/best_model.keras`: Final trained model (4.3M parameters, native Keras format)
- `training_history.png`: Accuracy and loss curves for train/val with metrics visualization
- `Run/optimizer_results/`: ABC optimization experiment logs
- `Run/pso_ackley_results/`: PSO benchmarking results on test functions
- `websiteResults/`: Screenshots of the web application in action
- Evaluation files: detailed per-class metrics (precision, recall, F1-score)

## 🚀 Web Application Usage

### Option 1: Live Demo (Recommended)
Access the deployed application directly:
**[https://huggingface.co/spaces/toniariasss/SmartFruitClassiffier](https://huggingface.co/spaces/toniariasss/SmartFruitClassiffier)**

### Option 2: Local Execution

1. **Start the local Flask server:**
   ```bash
   cd Run/web_app
   python app.py
   ```
   The application will be available at `http://localhost:5000`

2. **Make predictions:**
   - Drag a fruit image to the upload zone
   - Or click to select an image from your computer
   - The model will process the image and display:
     - **Predicted class** with highest probability
     - **Confidence distribution** for all classes (0-100%)
     - Inference time

3. **Supported formats:**
   - JPG, JPEG, PNG
   - Any resolution (automatically resized to 224×224)

## Troubleshooting

- **TensorFlow with NumPy 2.x**: If `_ARRAY_API` warnings appear, reinstall with `pip install "numpy<2.0" tensorflow`
- **No GPU detected**: Ensure that drivers and CUDA/cuDNN versions match your TensorFlow build
- **Incomplete dataset**: Use `python Run/analyze_dataset.py` to verify that all classes have images in train/val/test
- **Error loading .h5 model**: Old .h5 models may be corrupted, use the EfficientNet-B0 .keras model

## 🔮 Next Steps and Improvements

- ✅ **Cloud deployment**: ~~Migrate to Hugging Face Spaces~~ **COMPLETED** - [Demo available](https://huggingface.co/spaces/toniariasss/SmartFruitClassiffier)
- **REST API**: Implement endpoint for batch inference and integration with other systems
- **Progressive fine-tuning**: Unfreeze upper EfficientNet layers for incremental improvements
- **More classes**: Expand dataset to include more fruit varieties
- **Experiment tracking**: Integration with MLflow or Weights & Biases for advanced tracking
- **Architecture optimization**: Extend ABC/PSO to search over architectural decisions (number of layers, filters)
- **Mobile model**: Conversion to TFLite for deployment on mobile devices
- **Explainability**: Implement Grad-CAM to visualize which image regions the model uses

## 📊 Benchmarks and Comparisons

The project includes complete implementations of:
- **Artificial Bee Colony (ABC)**: Bio-inspired optimization for hyperparameter search
- **Particle Swarm Optimization (PSO)**: Swarm algorithm for comparison
- **Test functions**: Ackley and Rastrigin to validate metaheuristic convergence
- **Random Search**: Exhaustive exploration of hyperparameter space

Results demonstrate that Transfer Learning with EfficientNet-B0 + hyperparameter optimization **significantly outperforms custom CNN architectures:**
- **Custom CNN Model**: ~85-90% accuracy with ~2-3M trainable parameters
- **Transfer Learning Model**: **>95% accuracy** with only ~329K trainable parameters
- **Improvement**: +5-10% accuracy with 87% fewer trainable parameters

## 📝 Technical Notes

**Why EfficientNet-B0?**
- Optimal balance between accuracy and computational efficiency
- ImageNet pre-training provides robust features
- Only 4M parameters in the base (vs. 20M+ in ResNet50)
- Well-optimized compound scaling architecture

**Transfer Learning Strategy:**
- Frozen base throughout training
- Only custom classification layers are trained (~329K parameters)
- EfficientNet-specific preprocessing applied automatically
- Enables fast training with small-to-medium datasets

**Overfitting Management:**
- Multiple dropout in top layers
- L2 regularization in Dense layers
- Label smoothing in loss function
- Early stopping based on validation loss
- Data augmentation during training
