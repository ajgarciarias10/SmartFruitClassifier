import os
from pathlib import Path

# Path configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_ROOT = PROJECT_ROOT / 'dataset'

print("="*60)
print("DATASET ANALYSIS")
print("="*60)

total_images = 0

for split in ['train', 'val', 'test']:
    split_path = DATASET_ROOT / split / 'Fruit'
    
    if not split_path.exists():
        print(f"\n{split.upper()}: Directory not found!")
        continue
    
    print(f"\n{split.upper()}:")
    split_total = 0
    
    classes = sorted([d for d in split_path.iterdir() if d.is_dir()])
    
    for cls_dir in classes:
        images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.jpeg')) + list(cls_dir.glob('*.png'))
        count = len(images)
        split_total += count
        print(f"  {cls_dir.name}: {count:,} images")
    
    print(f"  TOTAL: {split_total:,} images")
    total_images += split_total

print(f"\n{'='*60}")
print(f"GRAND TOTAL: {total_images:,} images")
print(f"{'='*60}")
