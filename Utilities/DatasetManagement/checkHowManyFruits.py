"""
Report the current dataset counts and highlight how many images are still
needed per split/class to reach the targets defined in loadFiftyOne.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _import_loader():
    """Import the loadFiftyOne module regardless of execution context."""
    try:
        from . import loadFiftyOne as loader  # type: ignore
    except ImportError:  # pragma: no cover - script mode fallback
        import loadFiftyOne as loader  # type: ignore
    return loader


def _resolve_dataset_root(loader) -> Path:
    """Resolve the dataset root directory relative to the project root."""
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / loader.OUT_ROOT
    return dataset_root


def _count_images(folder: Path) -> int:
    """Count image files in the provided folder."""
    if not folder.exists():
        return 0
    return sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def _gather_counts(loader, dataset_root: Path) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
    """
    Collect counts per split and class.

    Returns a nested dict of the form:
      {split: {class_name: (current, target, pending)}}
    """
    results: Dict[str, Dict[str, Tuple[int, int, int]]] = {}

    for split in loader.SPLITS:
        split_folder = dataset_root / loader.split_dir_name(split) / "Fruit"
        results[split] = {}

        for class_name in loader.TARGET_CLASS_NAMES:
            class_folder = split_folder / class_name
            current = _count_images(class_folder)
            target = loader.TARGET_COUNTS_PER_CLASS_SPLIT[split][class_name]
            pending = max(target - current, 0)
            results[split][class_name] = (current, target, pending)

    return results


def print_summary(counts: Dict[str, Dict[str, Tuple[int, int, int]]]) -> None:
    """Pretty print the counts and deficits."""
    total_current = 0
    total_target = 0
    total_pending = 0

    print("\n" + "=" * 80)
    print("DATASET STATUS")
    print("=" * 80)

    for split, class_data in counts.items():
        print(f"\n{split.upper()}:")
        print(f"{'Class':<15} {'Current':>10} {'Target':>10} {'Pending':>10}")
        print("-" * 50)

        for class_name, (current, target, pending) in class_data.items():
            print(f"{class_name:<15} {current:>10} {target:>10} {pending:>10}")
            total_current += current
            total_target += target
            total_pending += pending

    print("\n" + "-" * 50)
    print(f"{'TOTAL':<15} {total_current:>10} {total_target:>10} {total_pending:>10}")
    print("=" * 80 + "\n")


def main() -> None:
    loader = _import_loader()
    dataset_root = _resolve_dataset_root(loader)

    if not dataset_root.exists():
        print("ERR Dataset directory not found.")
        print(f"Expected path: {dataset_root}")
        sys.exit(1)

    counts = _gather_counts(loader, dataset_root)
    print_summary(counts)


if __name__ == "__main__":
    main()
