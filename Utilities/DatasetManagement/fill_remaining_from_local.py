"""
Fill the remaining dataset deficits by copying images from the local
Fruits 360 extraction located in:

  C:\\Users\\ajgar\\Downloads\\archive\\fruits-360_100x100\\fruits-360\\Training
  C:\\Users\\ajgar\\Downloads\\archive\\fruits-360_100x100\\fruits-360\\Test

The script reuses the target counts defined in loadFiftyOne.py, copying
images (with possible repetition) until each split/class reaches its goal.
"""
from __future__ import annotations

import itertools
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List
from uuid import uuid4

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SOURCE_DIRECTORIES = [
    Path(r"C:\Users\ajgar\Downloads\archive\fruits-360_100x100\fruits-360\Training"),
    Path(r"C:\Users\ajgar\Downloads\archive\fruits-360_100x100\fruits-360\Test"),
]


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
            pending[split][class_name] = max(target - current, 0)
    return pending


def _gather_candidates(class_name: str, keywords: Iterable[str]) -> List[Path]:
    collected: List[Path] = []
    lowered_keywords = [kw.lower() for kw in keywords]

    for root in SOURCE_DIRECTORIES:
        if not root.exists():
            continue
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue
            folder_name = subdir.name.lower()
            if not any(keyword in folder_name for keyword in lowered_keywords):
                continue

            for img_path in subdir.glob("*"):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    collected.append(img_path)

    random.shuffle(collected)
    return collected


def _copy_for_deficit(
    dataset_root: Path,
    loader,
    split: str,
    class_name: str,
    deficit: int,
    candidates: List[Path],
) -> int:
    if deficit <= 0:
        return 0
    if not candidates:
        print(f"  !! No se encontraron imagenes locales para {class_name}")
        return 0

    dest_dir = dataset_root / loader.split_dir_name(split) / "Fruit" / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    copies = 0
    pool = itertools.cycle(candidates)
    for _ in range(deficit):
        src = next(pool)
        if not src.exists():
            continue
        new_name = f"{class_name.lower()}_{split}_local_{uuid4().hex}{src.suffix.lower()}"
        dest = dest_dir / new_name
        try:
            shutil.copy2(src, dest)
            copies += 1
        except Exception as exc:  # pragma: no cover
            print(f"    !! Error copiando {src} -> {dest}: {exc}")
    return copies


def main() -> None:
    loader = _import_loader()
    dataset_root = _resolve_dataset_root(loader)

    missing_sources = [str(path) for path in SOURCE_DIRECTORIES if not path.exists()]
    if missing_sources:
        print("ERR No se encuentran los directorios de origen:")
        for path in missing_sources:
            print(f"    {path}")
        sys.exit(1)

    if not dataset_root.exists():
        print("ERR Directorio destino (dataset) inexistente:")
        print(f"    {dataset_root}")
        sys.exit(1)

    pending = _collect_pending(loader, dataset_root)
    keyword_map = getattr(loader, "KAGGLE_KEYWORDS", None)
    if keyword_map is None:
        keyword_map = {
            "Apple": ["apple"],
            "Banana": ["banana"],
            "Cucumber": ["cucumber"],
            "Pear": ["pear"],
            "Tomato": ["tomato"],
        }

    candidates_cache: Dict[str, List[Path]] = {}
    for class_name in loader.TARGET_CLASS_NAMES:
        candidates_cache[class_name] = _gather_candidates(
            class_name, keyword_map.get(class_name, [])
        )
        print(
            f"Candidatos disponibles para {class_name}: "
            f"{len(candidates_cache[class_name])} archivos"
        )

    total_copied = 0
    for split in loader.SPLITS:
        print(f"\nProcesando split: {split.upper()}")
        for class_name, deficit in pending[split].items():
            if deficit <= 0:
                print(f"  {class_name:<15} objetivo alcanzado, nada que copiar")
                continue

            copied = _copy_for_deficit(
                dataset_root,
                loader,
                split,
                class_name,
                deficit,
                candidates_cache[class_name],
            )
            total_copied += copied
            print(
                f"  {class_name:<15} faltaban {deficit:>4} -> copiados {copied:>4}"
                f" | pendientes ahora: {max(deficit - copied, 0)}"
            )

    print("\n" + "-" * 60)
    print(f"Total de imagenes copiadas: {total_copied}")
    print("Vuelve a ejecutar checkHowManyFruits.py para verificar el resultado.")


if __name__ == "__main__":
    random.seed(1234)
    main()
