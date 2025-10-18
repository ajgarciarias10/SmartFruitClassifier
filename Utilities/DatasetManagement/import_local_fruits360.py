"""
Fill dataset deficits using local Fruits 360 images.

This script copies images from the extracted Fruits 360 Training directory
at C:\\Users\\ajgar\\Downloads\\archive\\fruits-360_100x100\\fruits-360\\Training
into the project's dataset folders until all targets defined in
loadFiftyOne.py are satisfied.
"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SOURCE_TRAIN_ROOT = Path(
    r"C:\Users\ajgar\Downloads\archive\fruits-360_100x100\fruits-360\Training"
)


def _import_loader():
    try:
        from . import loadFiftyOne as loader  # type: ignore
    except ImportError:  # pragma: no cover
        import loadFiftyOne as loader  # type: ignore
    return loader


def _resolve_dataset_root(loader) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / loader.OUT_ROOT


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(
        1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _collect_pending(loader, dataset_root: Path) -> Dict[str, Dict[str, int]]:
    pending: Dict[str, Dict[str, int]] = {}

    for split in loader.SPLITS:
        split_folder = dataset_root / loader.split_dir_name(split) / "Fruit"
        pending[split] = {}

        for class_name in loader.TARGET_CLASS_NAMES:
            target = loader.TARGET_COUNTS_PER_CLASS_SPLIT[split][class_name]
            current = _count_images(split_folder / class_name)
            deficit = max(target - current, 0)
            pending[split][class_name] = deficit

    return pending


def _collect_candidates(class_name: str, keywords: Iterable[str]) -> List[Path]:
    matches: List[Path] = []
    if not SOURCE_TRAIN_ROOT.exists():
        return matches

    for subdir in SOURCE_TRAIN_ROOT.iterdir():
        if not subdir.is_dir():
            continue
        lower_name = subdir.name.lower()
        if not any(keyword in lower_name for keyword in keywords):
            continue

        for img_path in subdir.glob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                matches.append(img_path)

    random.shuffle(matches)
    return matches


def _plan_copy(
    loader,
    pending: Dict[str, Dict[str, int]],
    keyword_map: Dict[str, List[str]],
) -> Dict[str, Dict[str, List[Path]]]:
    plan: Dict[str, Dict[str, List[Path]]] = {}
    used_files: Dict[str, Set[Path]] = {cls: set() for cls in loader.TARGET_CLASS_NAMES}
    cache: Dict[str, List[Path]] = {}

    for class_name in loader.TARGET_CLASS_NAMES:
        keywords = [k.lower() for k in keyword_map.get(class_name, [])]
        cache[class_name] = _collect_candidates(class_name, keywords)

    for split in loader.SPLITS:
        plan[split] = {}
        for class_name, deficit in pending[split].items():
            if deficit <= 0:
                plan[split][class_name] = []
                continue

            pool = cache[class_name]
            pool_iter = (item for item in pool if item not in used_files[class_name])
            chosen: List[Path] = []

            for item in pool_iter:
                chosen.append(item)
                used_files[class_name].add(item)
                if len(chosen) >= deficit:
                    break

            plan[split][class_name] = chosen

    return plan


def _copy_images(
    dataset_root: Path,
    loader,
    plan: Dict[str, Dict[str, List[Path]]],
) -> Dict[str, Dict[str, int]]:
    copied_summary: Dict[str, Dict[str, int]] = {}

    for split, split_plan in plan.items():
        split_folder = dataset_root / loader.split_dir_name(split) / "Fruit"
        split_folder.mkdir(parents=True, exist_ok=True)
        copied_summary[split] = {}

        for class_name, files in split_plan.items():
            target_dir = split_folder / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            for src in files:
                if not src.exists():
                    continue

                new_name = f"{class_name.lower()}_{src.stem}_{random.getrandbits(32):08x}{src.suffix.lower()}"
                dest = target_dir / new_name

                try:
                    shutil.copy2(src, dest)
                    copied += 1
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"  !! Error copiando {src} -> {dest}: {exc}")

            copied_summary[split][class_name] = copied

    return copied_summary


def _print_plan(plan: Dict[str, Dict[str, List[Path]]]) -> None:
    print("\nPLAN DE COPIA:")
    for split, split_plan in plan.items():
        print(f"\n{split.upper()}:")
        for class_name, files in split_plan.items():
            print(f"  {class_name:<15} -> {len(files)} archivos seleccionados")


def _print_summary(summary: Dict[str, Dict[str, int]]) -> None:
    print("\nRESUMEN DE COPIA:")
    for split, split_summary in summary.items():
        print(f"\n{split.upper()}:")
        for class_name, count in split_summary.items():
            print(f"  {class_name:<15} +{count}")
    print()


def main() -> None:
    loader = _import_loader()
    dataset_root = _resolve_dataset_root(loader)

    if not SOURCE_TRAIN_ROOT.exists():
        print("ERR Ruta de origen no encontrada:")
        print(f"    {SOURCE_TRAIN_ROOT}")
        sys.exit(1)

    if not dataset_root.exists():
        print("ERR Ruta de destino no encontrada:")
        print(f"    {dataset_root}")
        sys.exit(1)

    pending = _collect_pending(loader, dataset_root)
    keyword_map = getattr(loader, "KAGGLE_KEYWORDS", None)
    if keyword_map is None:
        keyword_map = {
            "Apple": ["apple"],
            "Banana": ["banana"],
            "Cucumber": ["cucumber"],
            "Pomegranate": ["pomegranate"],
            "Grapefruit": ["grapefruit"],
        }

    plan = _plan_copy(loader, pending, keyword_map)
    _print_plan(plan)

    summary = _copy_images(dataset_root, loader, plan)
    _print_summary(summary)

    print("Proceso completado. Ejecuta checkHowManyFruits.py para verificar los nuevos conteos.")


if __name__ == "__main__":
    random.seed(42)
    main()
