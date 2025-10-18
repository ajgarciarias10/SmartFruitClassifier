"""
Simple test script for the pre-trained fruit classifier model
This script works with the existing model without needing TensorFlow training
"""
import os
import sys
from pathlib import Path

def test_existing_model():
    """Test if we can load and use the existing model"""
    print("🍎 Smart Fruit Classifier - Model Test")
    print("=" * 50)
    
    # Check if model file exists
    model_path = "fruit_classifier_apple_banana_avocado.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Get model file size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Found model: {model_path}")
    print(f"📁 Model size: {size_mb:.1f} MB")
    
    # Check dataset structure
    print("\n📊 Dataset Status:")
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    splits = ["train", "val", "test"]
    total_images = 0
    
    for split in splits:
        split_path = os.path.join(dataset_dir, split, "Fruit")
        if os.path.exists(split_path):
            classes = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
            split_total = 0
            
            print(f"\n  {split.upper()}:")
            for class_name in sorted(classes):
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    split_total += count
                    status = "✅" if count > 100 else "⚠️" if count > 10 else "❌"
                    print(f"    {status} {class_name}: {count} images")
            
            total_images += split_total
            print(f"    Total {split}: {split_total} images")
    
    print(f"\n📈 Total dataset: {total_images} images")
    
    # Try to import TensorFlow (optional)
    print(f"\n🔧 Dependencies:")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        # Try to load the model
        print("\n🧠 Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Show model summary
        print("\n📋 Model Architecture:")
        model.summary()
        
        return True
        
    except ImportError:
        print("❌ TensorFlow not available - can't load model")
        print("💡 Install with: pip install 'numpy<2.0.0' tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def predict_sample_image():
    """Try to predict a sample image if TensorFlow is available"""
    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        
        # Load model
        model_path = "fruit_classifier_apple_banana_avocado.h5"
        model = tf.keras.models.load_model(model_path)
        
        # Class names (based on your dataset)
        class_names = ["Apple", "Banana", "Cucumber", "Grapefruit", "Pomegranate"]
        
        # Look for a sample image
        test_dir = os.path.join("dataset", "test", "Fruit")
        sample_image = None
        
        if os.path.exists(test_dir):
            for class_name in class_names:
                class_path = os.path.join(test_dir, class_name)
                if os.path.exists(class_path):
                    images = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if images:
                        sample_image = os.path.join(class_path, images[0])
                        actual_class = class_name
                        break
        
        if sample_image and os.path.exists(sample_image):
            print(f"\n🔍 Testing with sample image:")
            print(f"📸 Image: {sample_image}")
            print(f"🏷️  Actual class: {actual_class}")
            
            # Load and preprocess image
            img = Image.open(sample_image)
            img = img.resize((224, 224))  # MobileNetV2 input size
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            print(f"🤖 Predicted: {predicted_class} ({confidence:.2f}% confidence)")
            
            if predicted_class == actual_class:
                print("✅ Correct prediction!")
            else:
                print("❌ Incorrect prediction")
            
            # Show all class probabilities
            print(f"\n📊 All class probabilities:")
            for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
                print(f"  {class_name}: {prob*100:.2f}%")
        else:
            print("\n⚠️  No sample images found for testing")
            
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")

if __name__ == "__main__":
    print("SmartFruit Classifier - Quick Test")
    print("This script tests the existing model and dataset")
    print("=" * 60)
    
    success = test_existing_model()
    
    if success:
        print("\n" + "=" * 60)
        predict_sample_image()
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    
    if not success:
        print("\n💡 To fix TensorFlow issues:")
        print("1. pip install 'numpy<2.0.0'")
        print("2. pip install tensorflow")
        print("3. python test_model.py")