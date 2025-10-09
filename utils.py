"""
Utility functions for the Smart Fruit Classifier project
"""
import os
from pathlib import Path

def clean_view_fiftyone(ds, oi_name, ViewField):
    """
    Common function for filtering FiftyOne datasets
    
    Args:
        ds: FiftyOne dataset
        oi_name: Target class name (e.g., "Apple")  
        ViewField: FiftyOne ViewField class (imported as F)
        
    Returns:
        Filtered dataset view containing only target class images
        with at most 2 labels (allowing for some noise)
    """
    labels = ViewField("positive_labels.classifications").map(ViewField("label"))
    
    # Must contain target fruit AND have at most 2 labels
    cond = labels.contains(oi_name) & (labels.length() <= 2)
    
    return ds.match(cond)


def validate_dataset_structure(base_dir):
    """
    Validate that the dataset has the expected structure
    
    Args:
        base_dir: Path to dataset root directory
        
    Returns:
        dict: Validation results with counts per split
    """
    results = {}
    
    expected_classes = ["Apple", "Banana", "Cucumber", "Grapefruit", "Pomegranate"]
    splits = ["train", "val", "test"]
    
    for split in splits:
        results[split] = {}
        split_path = os.path.join(base_dir, split, "Fruit")
        
        if not os.path.exists(split_path):
            print(f"⚠️  Warning: {split} directory not found at {split_path}")
            continue
            
        for class_name in expected_classes:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                # Count image files
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                results[split][class_name] = image_count
            else:
                results[split][class_name] = 0
                
    return results


def print_dataset_summary(dataset_results):
    """
    Print a formatted summary of dataset validation results
    
    Args:
        dataset_results: Results from validate_dataset_structure()
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    for split, classes in dataset_results.items():
        if not classes:
            continue
            
        print(f"\n{split.upper()}:")
        print(f"{'Class':<15} {'Images':>10} {'Status':>15}")
        print("-" * 40)
        
        for class_name, count in classes.items():
            if count > 500:
                status = "✅ Good"
            elif count > 100:
                status = "⚠️  Low"
            elif count > 0:
                status = "❌ Very Low" 
            else:
                status = "❌ Missing"
                
            print(f"{class_name:<15} {count:>10} {status:>15}")
    
    print("\n" + "=" * 60)


def check_model_file(model_path):
    """
    Check if a model file exists and get its info
    
    Args:
        model_path: Path to model file
        
    Returns:
        dict: Model file information
    """
    if not os.path.exists(model_path):
        return {"exists": False, "size": 0, "message": f"Model file not found: {model_path}"}
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    
    return {
        "exists": True,
        "size": size_mb,
        "message": f"Model found: {model_path} ({size_mb:.1f} MB)"
    }