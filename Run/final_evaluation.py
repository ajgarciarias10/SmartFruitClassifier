"""
Final Model Evaluation & Comparison
Compares transfer learning model (.keras) vs previous model (.h5) on independent test set
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

IMG_SIZE = 224
NUM_CLASSES = 5
TEST_DIR = PROJECT_ROOT / 'dataset' / 'test' / 'Fruit'

print("\n" + "="*80)
print("FINAL MODEL EVALUATION & COMPARISON")
print("Transfer Learning (.keras) vs Previous Model (.h5)")
print("="*80 + "\n")

# Models to compare
keras_model_path = PROJECT_ROOT / 'models' / 'efficientnet-b0' / 'best_model.keras'

# Search for .h5 models in models directory (including subdirectories)
models_dir = PROJECT_ROOT / 'models'
h5_models = list(models_dir.glob('**/*.h5'))

models = []
if keras_model_path.exists():
    models.append(('Transfer Learning (.keras)', str(keras_model_path)))

# Add the first .h5 model found
if h5_models:
    h5_model_path = h5_models[0]
    models.append((f'Previous Model ({h5_model_path.name})', str(h5_model_path)))

if not models:
    print("ERROR: No models found")
    sys.exit(1)

print("Models to compare:")
for name, path in models:
    print(f"  • {name}: {Path(path).name}")
print()

# Load test dataset
print("Loading test dataset...")
datagen = ImageDataGenerator()  # NO rescale - EfficientNet has preprocessing built-in
test_gen = datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_gen.class_indices.keys())
true_labels = test_gen.classes
print(f"✓ Test set loaded: {test_gen.samples} images")
print(f"  Classes: {class_names}\n")

# Evaluate each model
results = {}

for model_name, model_path in models:
    print("="*80)
    print(f"EVALUATING: {model_name}")
    print("="*80 + "\n")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded\n")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        print(f"Skipping {model_name}\n")
        continue
    
    # Predict
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence = np.max(predictions, axis=1)
    
    # Metrics
    accuracy = np.mean(predicted_classes == true_labels)
    cm = confusion_matrix(true_labels, predicted_classes)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*80}\n")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Average Confidence: {confidence.mean():.4f}\n")
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        total = (true_labels == i).sum()
        correct = cm[i][i]
        class_acc = correct / total if total > 0 else 0
        print(f"  {class_name:15}: {correct:4d}/{total:4d} = {class_acc:.4f} ({class_acc*100:.1f}%)")
    
    # Confusion Matrix (text)
    print(f"\nConfusion Matrix:")
    print(f"{'':15}", end="")
    for name in class_names:
        print(f"{name[:10]:>10}", end=" ")
    print()
    print("-" * 80)
    for i, name in enumerate(class_names):
        print(f"{name[:15]:<15}", end="")
        for j in range(NUM_CLASSES):
            print(f"{cm[i][j]:>10}", end=" ")
        print()
    
    # Confusion Matrix (graphical)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure in results/mconf/
    mconf_dir = PROJECT_ROOT / 'results' / 'mconf'
    mconf_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '_')
    cm_path = mconf_dir / f'confusion_matrix_{safe_name}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    # Check collapse
    pred_unique, pred_counts = np.unique(predicted_classes, return_counts=True)
    max_ratio = max(pred_counts) / len(predicted_classes)
    
    # Create a dictionary for pred_counts by class index
    pred_count_dict = dict(zip(pred_unique, pred_counts))
    
    print(f"\nPrediction Distribution:")
    for cls_idx in range(NUM_CLASSES):
        count = pred_count_dict.get(cls_idx, 0)
        pct = 100.0 * count / len(predicted_classes)
        print(f"  {class_names[cls_idx]:15}: {count:4d} predictions ({pct:5.1f}%)")
    
    if max_ratio > 0.9:
        print(f"\n⚠ WARNING: Model COLLAPSED ({max_ratio:.1%} predictions in one class)")
    
    results[model_name] = {
        'accuracy': accuracy,
        'confidence': confidence.mean(),
        'confusion_matrix': cm,
        'collapsed': max_ratio > 0.9
    }
    
    print("\n")

# Final Summary
if not results:
    print("="*80)
    print("ERROR: No models could be evaluated")
    print("="*80)
    sys.exit(1)

if len(results) == 1:
    print("="*80)
    print("SINGLE MODEL EVALUATION COMPLETE")
    print("="*80 + "\n")
    model_name = list(results.keys())[0]
    r = results[model_name]
    print(f"Model: {model_name}")
    print(f"Accuracy: {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
    print(f"Avg Confidence: {r['confidence']:.4f}")
    print(f"Status: {'⚠ COLLAPSED' if r['collapsed'] else '✓ OK'}")
    print(f"\n{'='*80}\n")

# Comparison
elif len(results) == 2:
    print("="*80)
    print("COMPARISON")
    print("="*80 + "\n")
    
    names = list(results.keys())
    r1, r2 = results[names[0]], results[names[1]]
    
    print(f"{'Metric':<25} {names[0]:<30} {names[1]:<30}")
    print("-" * 80)
    print(f"{'Accuracy':<25} {r1['accuracy']:.4f} ({r1['accuracy']*100:.2f}%){'':<15} {r2['accuracy']:.4f} ({r2['accuracy']*100:.2f}%)")
    print(f"{'Avg Confidence':<25} {r1['confidence']:.4f}{'':<25} {r2['confidence']:.4f}")
    print(f"{'Status':<25} {'COLLAPSED' if r1['collapsed'] else 'OK':<30} {'COLLAPSED' if r2['collapsed'] else 'OK'}")
    
    acc_diff = r1['accuracy'] - r2['accuracy']
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}\n")
    
    if r1['collapsed'] and not r2['collapsed']:
        print(f"🏆 WINNER: {names[1]} (not collapsed)")
    elif r2['collapsed'] and not r1['collapsed']:
        print(f"🏆 WINNER: {names[0]} (not collapsed)")
    elif r1['accuracy'] > r2['accuracy']:
        print(f"🏆 WINNER: {names[0]}")
        print(f"   Accuracy improved by {acc_diff*100:.2f}%")
    elif r2['accuracy'] > r1['accuracy']:
        print(f"🏆 WINNER: {names[1]}")
        print(f"   Accuracy improved by {-acc_diff*100:.2f}%")
    else:
        print("🤝 TIE")
    
    # Comparison visualization
    print(f"\nGenerating comparison visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # First model confusion matrix
    disp1 = ConfusionMatrixDisplay(confusion_matrix=r1['confusion_matrix'], display_labels=class_names)
    disp1.plot(cmap='Blues', values_format='d', ax=axes[0])
    axes[0].set_title(f'{names[0]}\nAccuracy: {r1["accuracy"]:.4f}', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Second model confusion matrix
    disp2 = ConfusionMatrixDisplay(confusion_matrix=r2['confusion_matrix'], display_labels=class_names)
    disp2.plot(cmap='Oranges', values_format='d', ax=axes[1])
    axes[1].set_title(f'{names[1]}\nAccuracy: {r2["accuracy"]:.4f}', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    mconf_dir = PROJECT_ROOT / 'results' / 'mconf'
    mconf_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = mconf_dir / 'confusion_matrix_comparison.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison saved: {comparison_path}")
    plt.show()

print(f"\n{'='*80}\n")
