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

1. **Crear entorno virtual**: 
   ```bash
   python -m venv fruit_classifier_env
   source fruit_classifier_env/bin/activate  # En Linux/Mac
   .\fruit_classifier_env\Scripts\activate   # En Windows
   ```

2. **Instalar dependencias**: 
   ```bash
   pip install -r requirements.txt
   ```

3. **Entrenar modelo**: 
   ```bash
   python main.py
   ```

4. **Probar modelo**: 
   ```bash
   python test_model.py
   ```

### Datasets : From Kaggle and OpenImagesV7

#### Name of dataset: OpenImages v7

[URL of dataset](https://g.co/dataset/open-)

License of dataset: licensed by Google Inc. under CC BY 4.0 license.

The images are listed as having a CC BY 2.0 license.

Short description of dataset and use case(s): bigger than ImageNet with 61M image level labels, 16M bounding boxes, 3M visual relationships, 2.7M instance segmentation masks, 600k localized narratives (synchronized audio and text caption, with mouse trace), and 66M point labels.

## 📁 Project Structure

```
SmartFruitClassifier/
├── 🤖 Utilities/           # Utilidades del proyecto
│   └── FruitDetector.py   # Clase principal del clasificador CNN
├── �️ dataset/            # Datos de entrenamiento
│   ├── train/Fruit/       # Imágenes de entrenamiento
│   ├── val/Fruit/         # Imágenes de validación
│   └── test/Fruit/        # Imágenes de prueba
├── � main.py             # Script principal de entrenamiento
├── 🧪 test_model.py       # Script para probar el modelo
├── 📝 requirements.txt    # Dependencias del proyecto
├── ❗ TROUBLESHOOTING.md  # Guía de resolución de problemas
├── 🧠 best_fruit_model.h5 # Mejor modelo guardado
└── 📊 training_history.png # Gráficas de entrenamiento
```

## 🚀 Usage

1. **First time setup**: Run `python check_dependencies.py`
2. **Check dataset**: Run `python checkHowManyFruits.py`
3. **Download more data** (optional): Run `python loadFiftyOne.py`
4. **Train model**: Run `python main.py`

## ✨ Features

- **CNN Personalizada**: Arquitectura de red neuronal convolucional diseñada específicamente para clasificación de frutas
- **Aumento de Datos**: Rotación, zoom, y otras transformaciones para mejorar la generalización
- **Early Stopping**: Prevención de sobreajuste monitorizando la pérdida de validación
- **Guardado Automático**: Almacenamiento del mejor modelo durante el entrenamiento
- **Métricas Completas**: Seguimiento de accuracy, precision y recall
- **Visualización**: Gráficas detalladas del progreso del entrenamiento
- **Predicción Simple**: Interfaz fácil de usar para clasificar nuevas imágenes

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
