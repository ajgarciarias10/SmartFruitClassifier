import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "test_results"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "test" / "Fruit"
IMG_SIZE = 224
BATCH_SIZE = 32
PHASE2 = RESULTS_ROOT / "fruit_model_test_acc0.6759_20251205_140653.h5"
PHASE3 = RESULTS_ROOT / "best_model.keras"

# 1) Generador de test (mismas dimensiones que el entrenamiento)
test_datagen = ImageDataGenerator()
test_gen = test_datagen.flow_from_directory(
    DATASET_ROOT,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# 2) Cargar modelos
model_tl = tf.keras.models.load_model(PHASE3)       # con transfer learning
model_no_tl = tf.keras.models.load_model(PHASE2)    # sin transfer learning

# 3) Evaluar en el mismo conjunto de test (pueden venir mas de dos metricas)
raw_tl = model_tl.evaluate(test_gen, verbose=0)
raw_no_tl = model_no_tl.evaluate(test_gen, verbose=0)

results_tl = dict(zip(model_tl.metrics_names, raw_tl))
results_no_tl = dict(zip(model_no_tl.metrics_names, raw_no_tl))


def extract_accuracy(results, raw_values):
    """Devuelve accuracy si existe; si no, el segundo valor de evaluate."""
    for key in ("accuracy", "acc", "categorical_accuracy", "binary_accuracy"):
        if key in results:
            return results[key]
    if isinstance(raw_values, (list, tuple)) and len(raw_values) > 1:
        return raw_values[1]
    return None


acc_tl = extract_accuracy(results_tl, raw_tl)
acc_no_tl = extract_accuracy(results_no_tl, raw_no_tl)

# 4) Predicciones para metricas mas detalladas
y_prob_tl = model_tl.predict(test_gen)
y_prob_no_tl = model_no_tl.predict(test_gen)

y_true = test_gen.classes  # ya estan indexadas segun class_indices

y_pred_tl = np.argmax(y_prob_tl, axis=1)
y_pred_no_tl = np.argmax(y_prob_no_tl, axis=1)

# Calcular accuracy si no vino en evaluate
acc_pred_tl = (y_pred_tl == y_true).mean()
acc_pred_no_tl = (y_pred_no_tl == y_true).mean()

if acc_tl is None:
    acc_tl = acc_pred_tl
if acc_no_tl is None:
    acc_no_tl = acc_pred_no_tl

print(f"Accuracy modelo con TL: {acc_tl:.4f}" if acc_tl is not None else "Accuracy modelo con TL: N/A")
print(f"Accuracy modelo sin TL: {acc_no_tl:.4f}" if acc_no_tl is not None else "Accuracy modelo sin TL: N/A")

print("=== Modelo con Transfer Learning ===")
print(classification_report(y_true, y_pred_tl))

print("=== Modelo sin Transfer Learning ===")
print(classification_report(y_true, y_pred_no_tl))

cm_tl = confusion_matrix(y_true, y_pred_tl)
cm_no_tl = confusion_matrix(y_true, y_pred_no_tl)

print("Matriz de confusion (TL):")
print(cm_tl)
print("Matriz de confusion (sin TL):")
print(cm_no_tl)

# 5) Visualizar matrices de confusion
class_names = [cls for cls, idx in sorted(test_gen.class_indices.items(), key=lambda item: item[1])]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title in zip(axes, [cm_tl, cm_no_tl], ["TL", "Sin TL"]):
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Matriz de confusion ({title})")
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Prediccion")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

plt.tight_layout()
plt.show()
