# SmartFruit Classifier

## Project Overview

Kyrilo and Antonio have developed a machine learning system for fruit classification using deep learning. The system can identify:

- 🍎 Apples
- 🍌 Bananas
- 🥒 Cucumber
- 🍊 Pomegranate
- 🟠 Grapefruit

## Requirements

#### Python Version: 3.8+ (recommended 3.10+)

### 🛠️ Quick Setup

1. **Check dependencies**: `python check_dependencies.py`
2. **Install requirements**: `pip install tensorflow numpy matplotlib pillow`
3. **Check dataset**: `python checkHowManyFruits.py`
4. **Train model**: `python main.py`

### Datasets : From Kaggle and OpenImagesV7

#### Name of dataset: OpenImages v7

[URL of dataset](https://g.co/dataset/open-)

License of dataset: licensed by Google Inc. under CC BY 4.0 license.

The images are listed as having a CC BY 2.0 license.

Short description of dataset and use case(s): bigger than ImageNet with 61M image level labels, 16M bounding boxes, 3M visual relationships, 2.7M instance segmentation masks, 600k localized narratives (synchronized audio and text caption, with mouse trace), and 66M point labels.

## 📁 Project Structure
#TODO correct this

```
SmartFruitClassifier/
├── 🤖 FruitDetector.py      # Main CNN model class
├── 🎯 main.py               # Training script
├── 🔍 check_dependencies.py # System requirements check
├── 📊 checkHowManyFruits.py # Dataset status checker
├── 📥 loadFiftyOne.py       # Data download script
├── 🛠️ utils.py              # Common utilities
├── 🏗️ dataset/              # Training data
│   ├── train/Fruit/         # Training images
│   ├── val/Fruit/           # Validation images
│   └── test/Fruit/          # Test images
└── 🧠 *.h5                  # Trained models
```

## 🚀 Usage

1. **First time setup**: Run `python check_dependencies.py`
2. **Check dataset**: Run `python checkHowManyFruits.py`
3. **Download more data** (optional): Run `python loadFiftyOne.py`
4. **Train model**: Run `python main.py`

## ✨ Features
## TODO CORRECT THIS
- **CNN Learning** with MobileNetV2
- **Data Augmentation** for better generalization
- **Early Stopping** to prevent overfitting
- **Automatic model checkpointing**
- **Comprehensive training metrics** (accuracy, precision, recall)
- **Visualization** of training progress

## 🔧 Recent Improvements

- ✅ Fixed path handling (now uses absolute paths)
- ✅ Added input validation and error checking
- ✅ Improved code organization with utils.py
- ✅ Better error messages and user feedback
- ✅ Dependency checking system
- ✅ Dataset validation tools

##### FiftyOne Installation Guide

[Official Documentation](https://docs.voxel51.com/tutorials/open_images.html)

##### Data Download Script

- [loadFiftyOne.py](Utilities/DatasetManagement/loadFiftyOne.py) - Downloads images from OpenImages V7
