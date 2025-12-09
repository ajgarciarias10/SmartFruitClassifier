"""
Quick training script with simpler approach
"""
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATASET_ROOT = PROJECT_ROOT / 'dataset'
TRAIN_DIR = DATASET_ROOT / 'train' / 'Fruit'
VAL_DIR = DATASET_ROOT / 'val' / 'Fruit'
TEST_DIR = DATASET_ROOT / 'test' / 'Fruit'
MODELS_DIR = PROJECT_ROOT / 'models' / 'efficientnet-b0'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
IMG_SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

print("="*60)
print("QUICK TRAINING")
print("="*60)

# Create data generators (no augmentation to avoid issues)
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    str(TRAIN_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    str(VAL_DIR),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nData loaded:")
print(f"  Train: {train_gen.samples} samples")
print(f"  Val: {val_gen.samples} samples")
print(f"  Classes: {list(train_gen.class_indices.keys())}\n")

# Build model
print("Building model...")
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel compiled!")
print(f"Trainable params: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n")

# Setup callbacks
checkpoint_path = MODELS_DIR / 'best_model.keras'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

# Train
print("="*60)
print("Starting training...")
print("="*60)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_model_path = MODELS_DIR / 'final_model.keras'
model.save(str(final_model_path))

print("\n" + "="*60)
print("Training complete!")
print("="*60)
print(f"Best model saved to: {checkpoint_path}")
print(f"Final model saved to: {final_model_path}")

# Print best results
best_val_acc = max(history.history['val_accuracy'])
print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
