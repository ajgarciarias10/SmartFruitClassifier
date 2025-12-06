# Resumen de Conversación - SmartFruitClassifier
**Fecha:** 6 de diciembre de 2025

## Problema Inicial
- Usuario reportó que la web app mostraba predicciones incorrectas (manzana → "Banana 69%")
- Las matrices de confusión offline mostraban buenos resultados, pero la web app no coincidía

## Soluciones Implementadas

### 1. Diagnóstico y Corrección de Rutas
**Problema:** El modelo estaba en `Run/test_results/best_model.keras` pero la app buscaba en `test_results/`

**Solución:**
- Corregido `Run/web_app/app.py` para usar rutas correctas:
  ```python
  APP_DIR = Path(__file__).resolve().parent
  RUN_DIR = APP_DIR.parent
  TEST_RESULTS_DIR = RUN_DIR / 'test_results'
  ```
- Actualizado `Run/debug_model_mapping.py` con las mismas rutas

**Archivos modificados:**
- `Run/web_app/app.py`
- `Run/debug_model_mapping.py`

### 2. Mapeo de Clases Robusto
**Problema:** Índices del modelo no coincidían con nombres de clases

**Solución:** Implementado mapeo robusto con prioridades:
1. Cargar desde `*_metrics_*.json` (orden usado en evaluación)
2. Usar `generator.class_indices` (flow_from_directory)
3. Leer carpetas del dataset
4. Fallback a nombres por defecto

**Mapeo correcto establecido:**
```python
CLASS_NAMES = ['Apple', 'Banana', 'Cucumber', 'Grapefruit', 'Pomegranate']
# Índice 0 = Apple
# Índice 1 = Banana
# Índice 2 = Cucumber
# Índice 3 = Grapefruit
# Índice 4 = Pomegranate
```

### 3. Scripts de Prueba Creados
**Creados:**
- `Run/test_all_classes.py` - Test del modelo con todas las clases
- `Run/quick_test.py` - Test rápido con primera imagen
- `Run/test_one.py` - Test con una sola imagen

### 4. Optimización de Hiperparámetros

#### Pregunta del Usuario
"¿Qué modelo es mejor para mejorar hiperparámetros: PSO, Random Search o Grid Search?"

#### Recomendación Dada
**Random Search** como primer paso:
- Más eficiente que Grid Search para espacios grandes
- Más simple que PSO y Bayesian
- Buena relación coste/beneficio

**Comparación:**
| Método | Pros | Contras | Cuándo usar |
|--------|------|---------|-------------|
| Grid Search | Exhaustivo, simple | Lento, escala mal | Espacios pequeños (2-3 parámetros) |
| Random Search | Rápido, eficiente | No usa conocimiento previo | Exploración inicial ✅ |
| PSO | Búsqueda poblacional | Costoso, puede sobreajustar | Si Random/Bayesian no funcionan |
| Bayesian (Optuna) | Muy eficiente | Más complejo | Presupuesto limitado (recomendado para afinar) |
| Hyperband/BOHB | Super eficiente | Requiere early stopping | Búsquedas caras |

### 5. Implementación de Random Search

#### Script Independiente (Inicial)
**Creado:** `Run/random_search.py`
- Búsqueda aleatoria de hiperparámetros
- Guarda cada trial y el mejor modelo
- Uso: `python Run/random_search.py --trials 20 --epochs 15`

#### Integración en train_simple.py (Final)
**Modificado:** `Run/train_simple.py`

**Características:**
- Random Search integrado directamente
- Hiperparámetros muestreados:
  - `learning_rate`: log-uniform [1e-5, 1e-3]
  - `batch_size`: categorical [16, 32, 64]
  - `dense_units`: categorical [128, 256, 512]
  - `dropout`: uniform [0.1, 0.6]
  - `l2_reg`: log-uniform [1e-6, 1e-3] (50% probabilidad)
  - `label_smoothing`: uniform [0.0, 0.1] (50% probabilidad)

- **Class weights automático** (balancea clases desiguales)
- **Callbacks incluidos:**
  - `EarlyStopping(patience=10, restore_best_weights=True)`
  - `ModelCheckpoint(save_best_only=True)`
  - `ReduceLROnPlateau(factor=0.5, patience=5)`

- **Evaluación en TEST_DIR** al final
- Guarda top-5 trials

**Uso:**
```powershell
# Prueba rápida
python Run\train_simple.py --trials 3 --epochs 5

# Búsqueda recomendada
python Run\train_simple.py --trials 15 --epochs 12

# Búsqueda completa
python Run\train_simple.py --trials 20 --epochs 15
```

**Resultados generados:**
- `models/efficientnet-b0/best_model.keras` - Mejor modelo (guardado automáticamente durante entrenamiento)
- `Run/test_results/trial_{i}_valacc_{acc}.keras` - Cada trial
- `Run/test_results/random_search_results.json` - Métricas de todos los trials
- `Run/test_results/test_evaluation_results.json` - Evaluación final en test set

### 6. Resultados del Smoke Test
```
Epoch 1: val_accuracy improved from None to 0.88100
Epoch 2: val_accuracy improved from 0.88100 to 0.90000
Best val accuracy: 0.9000
```

**Conclusión:** El modelo con Random Search alcanzó 90% de precisión en validación en solo 2 epochs, confirmando que el enfoque funciona bien.

## Archivos Finales del Proyecto

### Scripts de Entrenamiento
- `Run/train_simple.py` ⭐ - Entrenamiento con Random Search integrado
- `Run/transfer_model.py` - Clase TransferLearningModel

### Scripts de Evaluación/Prueba
- `Run/test_all_classes.py` - Test con todas las clases
- `Run/compare_models.py` - Comparación entre modelos
- `Run/evaluate_test_results.py` - Evaluación de múltiples modelos
- `Run/debug_model_mapping.py` - Diagnóstico de mapeo

### Web App
- `Run/web_app/app.py` - Flask backend
- `Run/web_app/templates/index.html` - Frontend con cámara y upload

### Modelos
- `models/efficientnet-b0/best_model.keras` - Mejor modelo (guardado automáticamente)
- `models/efficientnet-b0/final_model.keras` - Modelo final
- `Run/test_results/best_model.keras` - Copia del mejor (opcional)

## Comandos Principales

### Entrenar con Random Search
```powershell
cd c:\Users\ajgar\Escritorio\SmartFruitClassifier
python Run\train_simple.py --trials 15 --epochs 12
```

### Ejecutar Web App
```powershell
cd c:\Users\ajgar\Escritorio\SmartFruitClassifier
python Run\web_app\app.py
# Abrir: http://127.0.0.1:5000
```

### Test de Todas las Clases
```powershell
python Run\test_all_classes.py
```

## Próximos Pasos Sugeridos

1. **Ejecutar búsqueda completa:**
   ```powershell
   python Run\train_simple.py --trials 20 --epochs 15
   ```

2. **Analizar resultados:**
   ```powershell
   Get-Content Run\test_results\random_search_results.json
   Get-Content Run\test_results\test_evaluation_results.json
   ```

3. **Probar mejor modelo en web app:**
   - Ejecutar `python Run\web_app\app.py`
   - Subir imagen de manzana
   - Verificar que ahora clasifica correctamente

4. **Opcional - Fine-tuning:**
   Si quieres mejorar aún más, descongelar capas superiores de EfficientNet y entrenar con LR menor

5. **Opcional - Bayesian Optimization:**
   Si necesitas máximo rendimiento, implementar Optuna sobre la mejor región encontrada por Random Search

## Tecnologías Usadas
- Python 3.11
- TensorFlow/Keras
- EfficientNet-B0 (Transfer Learning)
- Flask (Web App)
- ImageDataGenerator (Augmentación)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## Notas Importantes
- El modelo usa preprocesamiento de EfficientNet (`preprocess_input`)
- Imágenes redimensionadas a 224×224
- 5 clases: Apple, Banana, Cucumber, Grapefruit, Pomegranate
- Dataset: train (12795), val (1000), test (1521)
- Class weights automáticos para clases desbalanceadas
- Base de EfficientNet congelada (freeze_base=True) durante búsqueda

## Decisiones de Diseño
1. **Random Search en lugar de Grid/PSO/Bayesian:**
   - Más rápido y eficiente para exploración inicial
   - Fácil de implementar y entender
   - Suficiente para encontrar buenas configuraciones

2. **Guardar best_model en models/ (no test_results/):**
   - ModelCheckpoint lo guarda automáticamente
   - Evita duplicación innecesaria
   - Más limpio y estándar

3. **Evaluación en TEST_DIR al final:**
   - Valida rendimiento en datos no vistos
   - Previene sobreajuste en validación
   - Métricas finales guardadas en JSON

4. **Class weights automáticos:**
   - Balancea clases desiguales
   - Mejora predicción en clases minoritarias
   - Calculado dinámicamente del generator
