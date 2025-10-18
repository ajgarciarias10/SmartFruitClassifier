"""
Check system requirements and dependencies for the Smart Fruit Classifier project.
"""
import sys


REQUIRED_PYTHON = (3, 13)


def check_python_version() -> bool:
    """Ensure Python 3.13+ is active."""
    version = sys.version_info
    print(f"Python version detected: {version.major}.{version.minor}.{version.micro}")

    meets_requirement = (version.major, version.minor) >= REQUIRED_PYTHON
    if meets_requirement:
        print("OK  Python version requirement satisfied (>= 3.13)")
    else:
        print("ERR Python 3.13 or newer is required")
    return meets_requirement


def check_package(package_name: str, import_name: str | None = None) -> bool:
    """Check whether a package can be imported."""
    module_name = import_name or package_name

    try:
        __import__(module_name)
        print(f"OK  {package_name}")
        return True
    except ImportError:
        print(f"ERR {package_name} not installed")
        return False
    except AttributeError as exc:
        if "_ARRAY_API" in str(exc) and module_name == "tensorflow":
            print("WARN tensorflow compatibility issue (NumPy 2.x)")
        else:
            print(f"ERR {package_name} attribute error: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"ERR {package_name} unexpected error: {exc}")
        return False


def check_tensorflow_gpu() -> None:
    """Report TensorFlow GPU availability."""
    try:
        import tensorflow as tf  # type: ignore

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            print(f"INFO TensorFlow GPU support: {len(gpus)} GPU(s) available")
            for idx, gpu in enumerate(gpus):
                print(f"     GPU {idx}: {gpu.name}")
        else:
            print("WARN TensorFlow: CPU only (no GPU detected)")
    except ImportError:
        print("ERR TensorFlow not available")
    except AttributeError as exc:
        if "_ARRAY_API" in str(exc):
            print("WARN TensorFlow/NumPy compatibility issue detected")
            print("     Try: pip install 'numpy<2.0.0' tensorflow")
        else:
            print(f"ERR TensorFlow attribute error: {exc}")
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"ERR TensorFlow unexpected error: {exc}")


def main() -> None:
    print("=" * 60)
    print("SMART FRUIT CLASSIFIER - DEPENDENCY CHECK")
    print("=" * 60)

    python_ok = check_python_version()

    print("\nRequired packages:")
    required_packages = [
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("Pillow (PIL)", "PIL"),
    ]

    all_ok = python_ok
    for package_label, module_name in required_packages:
        if not check_package(package_label, module_name):
            all_ok = False

    print("\nOptional packages:")
    optional_packages = [
        ("fiftyone", "fiftyone"),
        ("opencv-python", "cv2"),
        ("kaggle", "kaggle"),
    ]
    for package_label, module_name in optional_packages:
        check_package(package_label, module_name)

    print("\nGPU support check:")
    check_tensorflow_gpu()

    print("\n" + "=" * 60)

    if all_ok:
        print("OK  All required dependencies are installed!")
        print("You can run the fruit classifier.")
    else:
        print("ERR Some required dependencies are missing.")
        print("\nRecommended fix:")
        print("pip install -r requirements.txt")
        print("\nAlternative (manual installation):")
        print('pip install "numpy<2.0.0" tensorflow matplotlib pillow')
        print("\nOptional packages:")
        print("pip install fiftyone opencv-python")

    print("=" * 60)


if __name__ == "__main__":
    main()
