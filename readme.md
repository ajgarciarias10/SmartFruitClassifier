# SmartFruit Classifier

## Project Overview

Antonio and  Kyrylo have developed a machine learning system for fruit classification using deep learning. The system can identify:

- 🍎 Apples
- 🍌 Bananas
- 🥒 Cucumber
- 🍊 Pomegranate
- 🟠 Grapefruit

## Requirements

#### Python Version: 3.8+ (recommended 3.10+)

### 🛠️ Quick Setup

1. **Create virtual enviroment**: 
   ```
   python -m venv fruit_classifier_env
   source fruit_classifier_env/bin/activate  # En Linux/Mac
   .\fruit_classifier_env\Scripts\activate   # En Windows
   ```

2. **Train Model**: 
   ```
   python main.py
   ```

3. **Test the model**: 
   ```
   python test_model.py
   ```



## 📁 Project Structure

```
SmartFruitClassifier/
|    
├── 🤖 Run/ # Folder used to run the code          
│   └── FruitDetector.py   #Class that has all necessary methods from training and testing
|   └── main.py
|   └── test_model.py
| 
├── �️ dataset/            
│   ├── train/Fruit/       # Images for training
│   ├── val/Fruit/         # Images for validation
│   └── test/Fruit/        # Images for testing
├──            
├── 🧪 test_model.py       # Script para probar el modelo
├── 🧠 best_fruit_model.h5 # Mejor modelo guardado
└── 📊 training_history.png # Gráficas de entrenamiento
```
### Datasets : From Kaggle and OpenImagesV7

#### Name of dataset: OpenImages v7

[URL of dataset](https://g.co/dataset/open-)

License of dataset: licensed by Google Inc. under CC BY 4.0 license.

The images are listed as having a CC BY 2.0 license.

Short description of dataset and use case(s): bigger than ImageNet with 61M image level labels, 16M bounding boxes, 3M visual relationships, 2.7M instance segmentation masks, 600k localized narratives (synchronized audio and text caption, with mouse trace), and 66M point labels.

## 🚀 Usage

1. **First time setup**: Run `python check_dependencies.py`
2. **Check dataset**: Run `python checkHowManyFruits.py`
3. **Download more data** (optional): Run `python loadFiftyOne.py`
4. **Train model**: Run `python main.py`

## ✨ Features

- **CNN Personalised**: CNN architecture designed specifically for classifying fruits
- **Data Augmentation**: Rotation, zoom, and other transformations to improve generalization
- **Early Stopping**: Preventing over-adjustment by monitoring the loss of validation
- **Automatic Save**: Saving the best training model
- **Complete Metrics**: Following  accuracy, precision and recall
- **Visualization**: Detailed graphics about the training
- **Simple prediction**: Interfaz fácil de usar para clasificar nuevas imágenes

## 🔧 Recent Improvements

- ✅ Fixed path handling (now uses absolute paths)
- ✅ Added input validation and error checking
- ✅ Improved code organization with utils.py
- ✅ Better error messages and user feedback
- ✅ Dependency checking system
- ✅ Dataset validation tools

##### FiftyOne Installation Guide

[Official Documentation](https://docs.voxel51.com/tutorials/open_images.html)

##### Data Download 
Downloads images from OpenImages V7
