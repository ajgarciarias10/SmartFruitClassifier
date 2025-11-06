import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
# import os
from pathlib import Path
class FruitDetector:
    # Initialize the FruitDetector class
    def __init__(self, img_size, num_classes):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None

    # Create data generators for training and validation datasets
    def create_data_generators(self, TRAIN_DIR, BATCH_SIZE, VAL_DIR):
    
        # Training data augmentation
        #This part is used to make differences between images of the same class
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
             # Shear is like inclinate  the image 20%
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            #Is to fill in new pixels that may appear after a transformation
            fill_mode='nearest'
        )

        # Validation/Test data - only rescaling
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # Load training data
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            # Is because is a clasification problem with more than 2 classes
            class_mode='categorical',
            shuffle=True
        )

        # Load validation data
        val_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=(self.img_size, self.img_size),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, val_generator

    def build_model(self,LEARNING_RATE):
            # Custom CNN architecture
        
        model = keras.Sequential([
                #Arguments (net size , filter size, activation function, input shape)
                layers.Conv2D(32, (3, 3), activation='relu', #relu  because for images is a good option
                              input_shape=(self.img_size, self.img_size, 3)),
                layers.MaxPooling2D(2, 2),

                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),

                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),

                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),

                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')]) #The output is a probability distribution vector when each element is the probability of each class

        # Compile model
        model.compile(
            #The optimizer is Adam, a popular optimization algorithm that adapts the learning rate for each parameter.
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            #Calculation loss function with categorical crossentropy
            #Categorical crossentropy is used for multi-class classification problems 
            # where the labels are one-hot encoded.
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        self.model = model
        return model

    def setup_callbacks(self):
        """Setup training callbacks"""
        #Early stopping to prevent overfitting
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            #Save the best model during training 
            ModelCheckpoint(
                'best_fruit_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
        ]
        return callbacks

    def train(self, train_generator, val_generator, epochs):
        """Train the model"""
        callbacks = self.setup_callbacks()

        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available. Train the model first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

    def evaluate(self, test_generator):
        """Evaluate model on test data"""
        results = self.model.evaluate(test_generator, verbose=1)

        print("\n=== Test Results ===")
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        print(f"Test Precision: {results[2]:.4f}")
        print(f"Test Recall: {results[3]:.4f}")

        return results
    def predict_image(self, image_path, class_names):
        """Predict a single image and return class name and confidence"""
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(
            image_path,
            target_size=(self.img_size, self.img_size)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        #This does expand the dimensions of the image to match the input shape of the model and normalizes 
        # the pixel values to be between 0 and 1
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Get predictions
        predictions = self.model.predict(img_array)
        #Convert the predictions to class name and confidence
        idx = np.argmax(predictions[0])
        predicted_class = class_names[idx]
        confidence = predictions[0][idx] * 100

        return predicted_class, confidence

    def save_model(self, filepath='fruit_detector_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='fruit_detector_model.h5'):
        """Load a saved model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")