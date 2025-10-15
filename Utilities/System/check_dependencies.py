"""
Check system requirements and dependencies for Smart Fruit Classifier
"""
import sys
import subprocess

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
        
    try:
        __import__(import_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name} - NOT INSTALLED")
        return False
    except AttributeError as e:
        if "_ARRAY_API" in str(e) and package_name == "tensorflow":
            print(f"⚠️  {package_name} - COMPATIBILITY ISSUE (NumPy 2.x)")
            return False
        else:
            print(f"❌ {package_name} - ERROR: {e}")
            return False
    except Exception as e:
        print(f"❌ {package_name} - ERROR: {e}")
        return False

def check_tensorflow_gpu():
    """Check TensorFlow GPU availability"""
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 TensorFlow GPU support: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("⚠️  TensorFlow: CPU only (no GPU detected)")
    except ImportError as e:
        print("❌ TensorFlow not available")
    except AttributeError as e:
        if "_ARRAY_API" in str(e):
            print("⚠️  TensorFlow/NumPy compatibility issue detected")
            print("   Try: pip install 'numpy<2.0.0' tensorflow")
        else:
            print(f"❌ TensorFlow error: {e}")
    except Exception as e:
        print(f"❌ TensorFlow error: {e}")

def main():
    print("=" * 60)
    print("SMART FRUIT CLASSIFIER - DEPENDENCY CHECK")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    
    print("\n📦 Required packages:")
    
    # Core ML packages
    packages = [
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),  
        ("matplotlib", "matplotlib"),
        ("Pillow (PIL)", "PIL"),
    ]
    
    all_ok = python_ok
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_ok = False
    
    print("\n🔧 Optional packages:")
    
    # Optional packages
    optional_packages = [
        ("fiftyone", "fiftyone"),
        ("opencv-python", "cv2"),
    ]
    
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name)
    
    # Check GPU support
    print("\n🎮 GPU Support:")
    check_tensorflow_gpu()
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("✅ All required dependencies are installed!")
        print("You can run the fruit classifier.")
    else:
        print("❌ Some required dependencies are missing.")
        print("\n🔧 RECOMMENDED FIX:")
        print("pip install -r requirements.txt")
        print("\n💡 Alternative (manual installation):")
        print('pip install "numpy<2.0.0" tensorflow matplotlib pillow')
        print("\n📦 Optional packages:")
        print("pip install fiftyone opencv-python")
    
    print("=" * 60)

if __name__ == "__main__":
    main()