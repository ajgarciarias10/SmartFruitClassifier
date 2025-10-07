import os
import time
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# CONFIGURACIÓN DE LÍMITES
MIN_IMAGES_PER_CLASS = 1000  # Mínimo de imágenes limpias por clase/split
MAX_IMAGES_PER_CLASS = 2000  # Máximo de imágenes limpias por clase/split

SPLITS = ["train", "validation", "test"]
## hola
TARGET_TO_OI = {
    "Apple": "Apple",
    "Cucumber": "Cucumber",
    "Banana": "Banana",
    "Pomegranate": "Pomegranate",
    "GrapeFruit": "Grapefruit",
}

OUT_ROOT = "dataset"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def split_dir_name(split):
    return {"train": "train", "validation": "val", "test": "test"}[split]


def clean_view(ds, oi_name):
    """
    Filtra para obtener imágenes que contienen SOLO la clase objetivo
    Sin ningún otro objeto en la imagen
    """
    labels = F("positive_labels.classifications").map(F("label"))

    # Exactamente UNA etiqueta y debe ser la fruta objetivo
    cond = (labels.length() == 1) & labels.contains(oi_name)

    return ds.match(cond)


def export_view(view, export_dir, target_name):
    """Exporta las imágenes limpias al disco"""
    ds_base = view._dataset

    if "ground_truth" not in ds_base.get_field_schema():
        ds_base.add_sample_field(
            "ground_truth",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Classification
        )

    print("      📝 Asignando etiquetas...", end=" ", flush=True)
    for s in view.iter_samples(progress=False):
        s["ground_truth"] = fo.Classification(label=target_name)
        s.save()
    print("✓")

    print("      💾 Exportando a disco...", end=" ", flush=True)
    view.export(
        export_dir=export_dir,
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
    )
    print("✓")


def download_one_class_for_split(target_name, oi_name, split, min_needed, max_needed):
    """
    Descarga imágenes limpias hasta conseguir entre min_needed y max_needed
    Se detiene cuando alcanza max_needed o cuando no hay más imágenes disponibles
    """
    split_dir = ensure_dir(os.path.join(OUT_ROOT, split_dir_name(split), "Fruit"))
    target_dir = ensure_dir(os.path.join(split_dir, target_name))

    # Cuenta cuántas ya existen
    got = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])

    if got >= max_needed:
        print(f"\n✅ [{split}] {target_name}: ya hay {got} imágenes (≥ máximo {max_needed}), omitiendo.")
        return

    if got >= min_needed:
        print(f"\n✅ [{split}] {target_name}: ya hay {got} imágenes (≥ mínimo {min_needed}), omitiendo.")
        return

    print(f"\n{'=' * 75}")
    print(f"[{split}] {oi_name} → {target_name}")
    print(f"Rango objetivo: {min_needed}-{max_needed} | Ya descargadas: {got}")
    print(f"{'=' * 75}")

    remaining = max_needed - got

    # Estrategia de descargas progresivas
    # Asumiendo que ~10-20% de las imágenes son limpias
    batch_multipliers = [3, 5, 8, 12, 20, 30]

    for attempt, multiplier in enumerate(batch_multipliers, 1):
        if got >= max_needed:
            print(f"\n🎉 ¡Máximo alcanzado! ({got}/{max_needed})")
            break

        max_samp = int((max_needed - got) * multiplier)

        # Limitar a un máximo razonable por intento (50k imágenes)
        max_samp = min(max_samp, 50000)

        print(f"\n📥 Intento {attempt}/{len(batch_multipliers)}: descargando hasta {max_samp} imágenes...")

        try:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["classifications"],
                classes=[oi_name],
                max_samples=max_samp,
                only_matching=True,  # Solo imágenes que contienen esta clase
                shuffle=True,
                seed=42 + attempt,  # Seed diferente por intento
                persistent=False,
            )

            total_downloaded = len(ds)
            print(f"   📊 Descargadas: {total_downloaded} imágenes totales")

            if total_downloaded == 0:
                print(f"   ⚠️  No hay más imágenes disponibles en Open Images V7")
                ds.delete()
                break

            # Filtra las limpias (solo la fruta, sin otros objetos)
            clean = clean_view(ds, oi_name)
            len_clean = clean.count()

            percentage = (len_clean / total_downloaded * 100) if total_downloaded > 0 else 0
            print(f"   ✅ Limpias encontradas: {len_clean} ({percentage:.1f}%)")

            if len_clean == 0:
                print(f"   ⚠️  No se encontraron imágenes limpias en este lote")
                ds.delete()

                # Si llevamos muchos intentos sin encontrar limpias, paramos
                if attempt >= 3:
                    print(f"   ⛔ No se encuentran más imágenes limpias después de {attempt} intentos")
                    break
                continue

            # Toma solo las necesarias (sin pasarse del máximo)
            still_needed = max_needed - got
            to_export = min(len_clean, still_needed)

            if to_export < len_clean:
                print(f"   ✂️  Tomando {to_export} de {len_clean} (para no exceder el máximo)")
                clean = clean.take(to_export, seed=42)

            # Exporta
            print(f"   🔄 Exportando {to_export} imágenes...")
            export_view(clean, split_dir, target_name)

            # Limpia memoria
            ds.delete()

            # Actualiza contador
            got = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
            print(f"   📈 Total acumulado: {got}/{max_needed}")

            # Verifica si alcanzamos el objetivo
            if got >= max_needed:
                print(f"\n🎉 ¡Máximo alcanzado para {target_name} en {split}! ({got} imágenes)")
                break
            elif got >= min_needed:
                print(f"\n✅ Mínimo alcanzado para {target_name} en {split}! ({got} imágenes)")
                print(f"   Continuando para intentar alcanzar el máximo ({max_needed})...")

            # Si descargamos menos del límite, significa que no hay más
            if total_downloaded < max_samp:
                print(f"\n   ℹ️  Se alcanzó el límite de imágenes disponibles en Open Images V7")
                break

            time.sleep(1)  # Pausa entre intentos

        except Exception as e:
            print(f"   ❌ Error en intento {attempt}: {str(e)}")
            time.sleep(2)
            continue

    # Resumen final
    final_count = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])

    print(f"\n{'─' * 75}")
    if final_count >= max_needed:
        print(f"✅ COMPLETADO: {final_count} imágenes (máximo alcanzado)")
    elif final_count >= min_needed:
        print(f"✅ ACEPTABLE: {final_count} imágenes (mínimo alcanzado)")
        print(f"   No se pudieron obtener más imágenes limpias de Open Images V7")
    else:
        print(f"⚠️  INSUFICIENTE: {final_count} imágenes (mínimo: {min_needed})")
        print(f"   Faltan: {min_needed - final_count} imágenes")
    print(f"{'─' * 75}")


def main():
    print("\n" + "=" * 80)
    print("DESCARGA DE IMÁGENES LIMPIAS CON LÍMITES")
    print("=" * 80)
    print(f"Clases: {list(TARGET_TO_OI.keys())}")
    print(f"Objetivo por clase/split: {MIN_IMAGES_PER_CLASS}-{MAX_IMAGES_PER_CLASS} imágenes")
    print(f"Filtro: SOLO la fruta objetivo, sin otros objetos")
    print("=" * 80 + "\n")

    total_start = time.time()

    for split in SPLITS:
        print(f"\n{'#' * 80}")
        print(f"# SPLIT: {split.upper()}")
        print(f"{'#' * 80}")

        for target_name, oi_name in TARGET_TO_OI.items():
            download_one_class_for_split(
                target_name=target_name,
                oi_name=oi_name,
                split=split,
                min_needed=MIN_IMAGES_PER_CLASS,
                max_needed=MAX_IMAGES_PER_CLASS,
            )

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO")
    print("=" * 80)
    print(f"Tiempo total: {total_time / 60:.1f} minutos")
    print(f"Directorio: {os.path.abspath(OUT_ROOT)}")

    # Resumen final detallado
    print("\n" + "=" * 80)
    print("RESUMEN FINAL DE DESCARGA:")
    print("=" * 80)

    for split in SPLITS:
        split_name = split_dir_name(split)
        split_path = os.path.join(OUT_ROOT, split_name, "Fruit")

        if os.path.exists(split_path):
            print(f"\n{split.upper()}:")
            print(f"{'Clase':<15} {'Descargadas':>12} {'Estado':>15}")
            print("─" * 50)

            for target_name in TARGET_TO_OI.keys():
                target_path = os.path.join(split_path, target_name)
                if os.path.exists(target_path):
                    count = len([f for f in os.listdir(target_path)
                                 if os.path.isfile(os.path.join(target_path, f))])

                    if count >= MAX_IMAGES_PER_CLASS:
                        status = "✅ Máximo"
                    elif count >= MIN_IMAGES_PER_CLASS:
                        status = "✅ Mínimo OK"
                    else:
                        status = "⚠️ Insuficiente"

                    print(f"{target_name:<15} {count:>12} {status:>15}")
                else:
                    print(f"{target_name:<15} {0:>12} {'❌ Sin datos':>15}")

    print("\n" + "=" * 80)
    print("LEYENDA:")
    print("  ✅ Máximo      = Se alcanzó el límite máximo (2000 imágenes)")
    print("  ✅ Mínimo OK   = Se alcanzó el mínimo requerido (1000 imágenes)")
    print("  ⚠️ Insuficiente = No se alcanzó el mínimo (< 1000 imágenes)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()