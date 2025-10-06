##Code Using FiftyOne to load all the images
import os
import time
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


PER_CLASS_PER_SPLIT = 500  # objetivo: 500 por clase y por split
SPLITS = ["train", "validation", "test"]  # mapearemos validation->val al exportar

# Tus clases destino y cómo mapear al nombre en Open Images
TARGET_TO_OI = {
    "Apple": "Apple",
    "Avocado": "Avocado",
    "Banana": "Banana",
    "Pomegranate": "Pomegranate",
    "GrapeFruit": "Grapefruit",   # <<— exportará carpeta 'GrapeFruit'
}

# Para el filtro de “solo esa fruta, sin otras”
OTHER_OI = list({oi for oi in TARGET_TO_OI.values()})

# Raíz de exportación
OUT_ROOT = "dataset"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def split_dir_name(split):
    return {"train": "train", "validation": "val", "test": "test"}[split]


def clean_view(ds, oi_name):
    labels = F("positive_labels.classifications").map(F("label"))

    # Empieza con: contiene la clase objetivo
    cond = labels.contains(oi_name)

    # Y añade: NO contiene ninguna de las otras frutas
    others = [c for c in OTHER_OI if c != oi_name]
    for o in others:
        cond = cond & (~labels.contains(o))

    return ds.match(cond)
def export_view(view, export_dir, target_name):
    ds_base = view._dataset  # dataset subyacente

    # 1) Crea el campo si no existe
    if "ground_truth" not in ds_base.get_field_schema():
        ds_base.add_sample_field(
            "ground_truth",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification
        )

    # 2) Asigna etiqueta unaria a CADA muestra de la vista limpia
    # (evitamos set_field para no chocar con esquemas/expresiones)
    for s in view.iter_samples(progress=True):
        s["ground_truth"] = fo.Classification(label=target_name)
        s.save()

    # 3) Exporta a árbol de carpetas
    view.export(
        export_dir=export_dir,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
    )

def download_one_class_for_split(target_name, oi_name, split, needed):
    """
    Descarga progresivamente hasta conseguir `needed` imágenes limpias para (clase, split)
    Exporta en OUT_ROOT/<split_dir>/Fruit/<target_name>/
    """
    split_dir = ensure_dir(os.path.join(OUT_ROOT, split_dir_name(split), "Fruit"))
    target_dir = ensure_dir(os.path.join(split_dir, target_name))

    got = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
    if got >= needed:
        print(f"[{split}] {target_name}: ya hay {got} archivos en disco (>= {needed}), salto descarga.")
        return

    # Progresión de max_samples para intentar reunir suficientes limpias
    attempts = [needed, int(needed*1.6), int(needed*2.2), int(needed*3.2)]
    total_clean = 0

    for max_samp in attempts:
        print(f"\n[{split}] {oi_name} -> {target_name} | intentando con max_samples={max_samp}")
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["classifications"],
            classes=[oi_name],
            max_samples=max_samp,
            shuffle=True,
            seed=51,
            persistent=False,
        )
        clean = clean_view(ds, oi_name)
        len_clean = clean.count()  # ← mide ANTES de borrar ds
        print(f"Descargadas: {len(ds)} | Limpias (solo {oi_name}): {len_clean}")

        remaining = needed - got
        if len_clean > remaining:
            clean = clean.take(remaining, seed=51)
            len_clean = remaining  # ← actualiza el número real

        # Exporta y actualiza el contador en disco
        export_view(clean, split_dir, target_name)
        ds.delete()  # ← AHORA borras el dataset

        got = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
        print(f"Export acumulado [{split}/{target_name}]: {got}/{needed}")

        if got >= needed:
            break

        time.sleep(1.0)

def main():
    print("Clases objetivo:", list(TARGET_TO_OI.keys()))
    for split in SPLITS:
        for target_name, oi_name in TARGET_TO_OI.items():
            download_one_class_for_split(
                target_name=target_name,
                oi_name=oi_name,
                split=split,
                needed=PER_CLASS_PER_SPLIT,
            )
    print("\n✅ Listo. Revisa carpetas en:", os.path.abspath(OUT_ROOT))

if __name__ == "__main__":
    main()
