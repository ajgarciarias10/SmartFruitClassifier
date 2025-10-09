import os
import time
try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F
    FIFTYONE_AVAILABLE = True
except ImportError:
    print("⚠️  FiftyOne not installed. Install with: pip install fiftyone")
    FIFTYONE_AVAILABLE = False
    F = None

from utils import clean_view_fiftyone, validate_dataset_structure, print_dataset_summary


def clean_view(ds, oi_name):
    """
    Wrapper for the common clean_view function
    """
    if not FIFTYONE_AVAILABLE:
        raise ImportError("FiftyOne is required for this function")
    
    return clean_view_fiftyone(ds, oi_name, F)


def check_dataset_status():
    """
    Check current dataset status without downloading anything
    """
    print("Checking current dataset status...")
    
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    results = validate_dataset_structure(dataset_dir)
    print_dataset_summary(results)


if __name__ == "__main__":
    check_dataset_status()
