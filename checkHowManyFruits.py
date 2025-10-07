import os
import time
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def clean_view(ds, oi_name):
    """
    Filtra para obtener imágenes que:
    1. Contienen SOLO la clase objetivo
    2. NO contienen ninguna otra clase (ni frutas ni otros objetos)
    """
    labels = F("positive_labels.classifications").map(F("label"))

    # Condición 1: Debe contener exactamente UNA etiqueta
    cond = labels.length() == 1

    # Condición 2: Esa única etiqueta debe ser la clase objetivo
    cond = cond & labels.contains(oi_name)

    # En clean_view():
    cond = labels.contains(oi_name)
    # Permitir 1-2 etiquetas si una es la fruta
    cond = cond & (labels.length() <= 2)

    return ds.match(cond)
