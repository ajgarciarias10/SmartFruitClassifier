import math
import os
import time

try:
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    FIFTYONE_AVAILABLE = True
except ImportError:
    print("Warning: FiftyOne not installed. Install with: pip install fiftyone")
    FIFTYONE_AVAILABLE = False
    F = None

try:
    # Running as module within package
    from .utils import clean_view_fiftyone  # type: ignore
except ImportError:
    # Running as standalone script
    from utils import clean_view_fiftyone

# Dataset configuration ----------------------------------------------------- #
SPLITS = ["train", "validation", "test"]
SPLIT_NAME_MAP = {"train": "train", "validation": "val", "test": "test"}

TARGET_TO_OI = {
    "Apple": "Apple",
    "Banana": "Banana",
    "Cucumber": "Cucumber",
    "Pear": "Pear",
    "Tomato": "Tomato",
}

TARGET_CLASS_NAMES = list(TARGET_TO_OI.keys())

TOTAL_IMAGES_TARGET = 10000
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}

OUT_ROOT = "dataset"
BATCH_MULTIPLIERS = [3, 5, 8, 12, 20, 30]
MAX_SAMPLES_PER_ATTEMPT = 50000
DOWNLOAD_COOLDOWN_SECONDS = 1
ERROR_COOLDOWN_SECONDS = 2
DATASET_SOURCES = ["open-images-v7"]


def _compute_split_targets(total_images, splits, ratios):
    """Compute integer image targets per split honoring the desired ratios."""
    raw_values = [total_images * ratios[split] for split in splits]
    base_values = [math.floor(val) for val in raw_values]
    remainder = total_images - sum(base_values)

    # Assign remaining images to splits with the largest fractional parts
    fractional_order = sorted(
        range(len(splits)),
        key=lambda idx: raw_values[idx] - base_values[idx],
        reverse=True,
    )

    for idx in fractional_order[:remainder]:
        base_values[idx] += 1

    return {split: base_values[i] for i, split in enumerate(splits)}


def _distribute_per_class(total_count, class_names):
    """Distribute split totals evenly across classes while staying in integers."""
    if total_count <= 0:
        return {name: 0 for name in class_names}

    base = total_count // len(class_names)
    remainder = total_count - (base * len(class_names))

    distribution = {name: base for name in class_names}
    for index, name in enumerate(class_names):
        if index < remainder:
            distribution[name] += 1

    return distribution


SPLIT_TOTAL_TARGETS = _compute_split_targets(
    TOTAL_IMAGES_TARGET, SPLITS, SPLIT_RATIOS
)

TARGET_COUNTS_PER_CLASS_SPLIT = {
    split: _distribute_per_class(total, TARGET_CLASS_NAMES)
    for split, total in SPLIT_TOTAL_TARGETS.items()
}

TOTAL_REQUESTED_IMAGES = TOTAL_IMAGES_TARGET

# Helpers ------------------------------------------------------------------- #
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def split_dir_name(split):
    return SPLIT_NAME_MAP[split]


def count_existing_images(path):
    if not os.path.exists(path):
        return 0
    return sum(
        1 for fname in os.listdir(path) if os.path.isfile(os.path.join(path, fname))
    )


def clean_view(ds, oi_name):
    """
    Wrapper for the common clean_view function.
    """
    if not FIFTYONE_AVAILABLE:
        raise ImportError("FiftyOne is required for this function")

    return clean_view_fiftyone(ds, oi_name, F)


def export_view(view, export_dir, target_name):
    """Exporta las imagenes filtradas al disco con etiquetas normalizadas."""
    ds_base = view._dataset

    if "ground_truth" not in ds_base.get_field_schema():
        ds_base.add_sample_field(
            "ground_truth",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification,
        )

    print("      -> Asignando etiquetas...", end=" ", flush=True)
    for sample in view.iter_samples(progress=False):
        sample["ground_truth"] = fo.Classification(label=target_name)
        sample.save()
    print("OK")

    print("      -> Exportando a disco...", end=" ", flush=True)
    view.export(
        export_dir=export_dir,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
    )
    print("OK")


def download_one_class_for_split(target_name, oi_name, split, target_count):
    """
    Descarga imagenes limpias hasta alcanzar el objetivo exacto por clase/split.
    """
    split_dir = ensure_dir(os.path.join(OUT_ROOT, split_dir_name(split), "Fruit"))
    target_dir = ensure_dir(os.path.join(split_dir, target_name))

    current_count = count_existing_images(target_dir)

    if current_count == target_count:
        print(
            f"\nOK. [{split}] {target_name}: ya existen {current_count} imagenes "
            f"(objetivo alcanzado)."
        )
        return

    if current_count > target_count:
        print(
            f"\nWarning. [{split}] {target_name}: existen {current_count} imagenes, "
            f"por encima del objetivo de {target_count}. Se omite la descarga."
        )
        return

    print("\n" + "=" * 75)
    print(f"[{split}] {oi_name} -> {target_name}")
    print(
        f"Objetivo: {target_count} imagenes limpias | Descargadas previamente: "
        f"{current_count}"
    )
    print("=" * 75)

    for source_index, dataset_name in enumerate(DATASET_SOURCES, start=1):
        remaining = target_count - current_count
        if remaining <= 0:
            break

        print(
            f"\nFuente {source_index}/{len(DATASET_SOURCES)}: {dataset_name}"
        )

        for attempt, multiplier in enumerate(BATCH_MULTIPLIERS, start=1):
            remaining = target_count - current_count
            if remaining <= 0:
                break

            max_samples = min(
                max(remaining * multiplier, remaining), MAX_SAMPLES_PER_ATTEMPT
            )

            print(
                f"\n--> Intento {attempt}/{len(BATCH_MULTIPLIERS)} "
                f"desde {dataset_name}: solicitando hasta {max_samples} imagenes..."
            )

            try:
                ds = foz.load_zoo_dataset(
                    dataset_name,
                    split=split,
                    label_types=["classifications"],
                    classes=[oi_name],
                    max_samples=max_samples,
                    only_matching=True,
                    shuffle=True,
                    seed=42 + attempt + (source_index * 100),
                    persistent=False,
                )
            except Exception as exc:
                print(
                    f"   !! No se pudo cargar {dataset_name} para {split}: {exc}"
                )
                time.sleep(ERROR_COOLDOWN_SECONDS)
                break

            total_downloaded = len(ds)
            print(f"   -> Descargadas: {total_downloaded} imagenes totales")

            if total_downloaded == 0:
                print("   !! No hay mas imagenes disponibles para esta clase/split.")
                ds.delete()
                break

            clean = clean_view(ds, oi_name)
            clean_count = clean.count()
            percentage = (
                (clean_count / total_downloaded) * 100 if total_downloaded > 0 else 0.0
            )
            print(
                f"   -> Imagenes limpias: {clean_count} ({percentage:.1f}% del lote)"
            )

            if clean_count == 0:
                print("   !! Lote sin imagenes limpias, buscando otro...")
                ds.delete()
                if attempt >= 3:
                    print(
                        f"   !! No se encontraron imagenes limpias tras {attempt} intentos."
                    )
                    break
                time.sleep(DOWNLOAD_COOLDOWN_SECONDS)
                continue

            to_export = min(clean_count, remaining)
            if to_export == 0:
                print("   -> No se necesitan mas imagenes para este objetivo.")
                ds.delete()
                break

            if to_export < clean_count:
                print(
                    f"   -> Seleccionando {to_export} de {clean_count} para cumplir el objetivo."
                )
                clean = clean.take(to_export, seed=42)

            print(f"   -> Exportando {to_export} imagenes filtradas...")
            export_view(clean, split_dir, target_name)
            ds.delete()

            current_count = count_existing_images(target_dir)
            print(
                f"   -> Progreso acumulado: {current_count}/{target_count} imagenes."
            )

            if current_count >= target_count:
                print(
                    f"\nOK. Objetivo alcanzado para {target_name} en {split}: "
                    f"{current_count} imagenes limpias."
                )
                break

            if total_downloaded < max_samples:
                print(
                    "\n!! La fuente devolvio menos imagenes de las solicitadas. "
                    "Puede que no queden mas disponibles."
                )
                break

            time.sleep(DOWNLOAD_COOLDOWN_SECONDS)

        if current_count >= target_count:
            break

    final_count = count_existing_images(target_dir)
    print("\n" + "-" * 75)
    if final_count >= target_count:
        print(
            f"OK. Objetivo final cumplido: {final_count} imagenes "
            f"para {target_name} en {split}."
        )
    else:
        missing = target_count - final_count
        print(
            f"!! Objetivo incompleto: {final_count}/{target_count} imagenes "
            f"({missing} pendientes)."
        )
    print("-" * 75)


def main():
    if not FIFTYONE_AVAILABLE:
        print("Error: FiftyOne is not installed.")
        print("Please install it with: pip install fiftyone")
        return

    print("\n" + "=" * 80)
    print("DESCARGA BALANCEADA DE IMAGENES LIMPIAS")
    print("=" * 80)
    print(f"Clases objetivo: {TARGET_CLASS_NAMES}")
    print(f"Splits: {SPLITS}")
    print(f"Porcentajes por split: {SPLIT_RATIOS}")
    print(f"Objetivo total: {TOTAL_REQUESTED_IMAGES} imagenes")
    print("Objetivo por split:", SPLIT_TOTAL_TARGETS)
    print("Objetivo por clase y split:")
    for split in SPLITS:
        targets = TARGET_COUNTS_PER_CLASS_SPLIT[split]
        print(f"  - {split}: {targets}")
    print("Filtro aplicado: solo la fruta objetivo, sin objetos extra.")
    print("=" * 80 + "\n")

    total_start = time.time()

    for split in SPLITS:
        print("\n" + "#" * 80)
        print(f"# SPLIT: {split.upper()}")
        print("#" * 80)

        for target_name, oi_name in TARGET_TO_OI.items():
            target_count = TARGET_COUNTS_PER_CLASS_SPLIT[split][target_name]
            download_one_class_for_split(
                target_name=target_name,
                oi_name=oi_name,
                split=split,
                target_count=target_count,
            )

    total_time_minutes = (time.time() - total_start) / 60

    print("\n" + "=" * 80)
    print("PROCESO COMPLETADO")
    print("=" * 80)
    print(f"Tiempo total: {total_time_minutes:.1f} minutos")
    print(f"Directorio de salida: {os.path.abspath(OUT_ROOT)}")

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)

    for split in SPLITS:
        split_path = os.path.join(OUT_ROOT, split_dir_name(split), "Fruit")
        if not os.path.exists(split_path):
            print(f"\n{split.upper()}: sin datos.")
            continue

        print(f"\n{split.upper()}:")
        print(f"{'Clase':<15} {'Descargadas':>12} {'Estado':>12}")
        print("-" * 45)

        for target_name in TARGET_TO_OI.keys():
            class_path = os.path.join(split_path, target_name)
            count = count_existing_images(class_path)
            expected = TARGET_COUNTS_PER_CLASS_SPLIT[split][target_name]

            if count == expected:
                status = "OK"
            elif count > expected:
                status = "Sobre"
            elif count == 0:
                status = "Vacio"
            else:
                status = "Faltan"

            print(f"{target_name:<15} {count:>12} {status:>12} (objetivo {expected})")

    print("\nObjetivos totales:", TOTAL_REQUESTED_IMAGES)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
