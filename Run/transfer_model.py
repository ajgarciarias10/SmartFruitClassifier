"""
Transfer Learning Model with EfficientNet-B0
Freeze base, train only top layers
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path


class TransferLearningModel:
    """Transfer Learning with frozen EfficientNet-B0 base"""
    
    def __init__(self, img_size: int = 224, num_classes: int = 5):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.base_model = None
        
    def build_model(self, learning_rate, dense_units, dropout_rate, freeze_base, l2_reg=0.0, label_smoothing=0.0):
        """Build model with frozen base and custom top layers

        Args:
            learning_rate: optimizer learning rate
            dense_units: units in the top dense layer
            dropout_rate: dropout rate applied to top layers
            freeze_base: whether to freeze the EfficientNet base
            l2_reg: L2 weight decay applied to Dense layers (float)
            label_smoothing: label smoothing factor for categorical loss
        """
        
        # Load pretrained base
        self.base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )
        
        # Freeze base
        self.base_model.trainable = not freeze_base
        
        # Build model
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))
        x = keras.applications.efficientnet.preprocess_input(inputs)
        x = self.base_model(x, training=False)
        x = layers.Dropout(dropout_rate)(x)
        if l2_reg and l2_reg > 0.0:
            reg = tf.keras.regularizers.l2(l2_reg)
        else:
            reg = None

        x = layers.Dense(dense_units, activation='relu', kernel_regularizer=reg)(x)
        x = layers.Dropout(dropout_rate * 0.6)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', kernel_regularizer=reg)(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"\n{'='*60}")
        print(f"Model built: Base frozen={freeze_base}, LR={learning_rate}")
        print(f"Dense units={dense_units}, Dropout={dropout_rate}")
        print(f"Trainable params: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        print(f"{'='*60}\n")
        
        return self.model
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32, augmentation=True):
        """Create data generators"""
        
        if augmentation:
            train_datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.15,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        val_datagen = ImageDataGenerator()
        
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_gen, val_gen
    
    def train(self, train_gen, val_gen, epochs, class_weight=None):
        """Train model"""
        
        # Setup callbacks
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = project_root / 'models' / 'efficientnet-b0' / 'best_model.keras'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        print(f"\n{'='*60}")
        print(f"Training: {epochs} epochs, {len(train_gen)} steps/epoch")
        print(f"{'='*60}\n")
        
        fit_kwargs = dict(
            x=train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        if class_weight is not None:
            fit_kwargs['class_weight'] = class_weight

        self.history = self.model.fit(**fit_kwargs)
        
        return self.history
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        if not self.history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        h = self.history.history
        
        # Accuracy
        axes[0, 0].plot(h['accuracy'], label='Train')
        axes[0, 0].plot(h['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(h['loss'], label='Train')
        axes[0, 1].plot(h['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(h['precision'], label='Train')
        axes[1, 0].plot(h['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(h['recall'], label='Train')
        axes[1, 1].plot(h['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✓ Plot saved: {save_path}")
        
        plt.show()
    
    def evaluate(self, test_gen):
        """Evaluate on test set"""
        results = self.model.evaluate(test_gen, verbose=1)
        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"  Loss: {results[0]:.4f}")
        print(f"  Accuracy: {results[1]:.4f}")
        print(f"  Precision: {results[2]:.4f}")
        print(f"  Recall: {results[3]:.4f}")
        print(f"{'='*60}\n")
        return results
    
    def save_model(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"✓ Model saved: {filepath}")


if __name__ == "__main__":
    print("Transfer Learning Model - Ready to use")
