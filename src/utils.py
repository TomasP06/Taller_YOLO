"""
utils.py
Utilidades del taller: conversión de formatos, verificación de dataset
y visualización de anotaciones YOLO.
"""

import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml


# ──────────────────────────────────────────────
# 1. VERIFICACIÓN DEL DATASET
# ──────────────────────────────────────────────

def verificar_dataset(data_yaml: str | Path) -> dict:
    """
    Verifica la coherencia entre imágenes y etiquetas YOLO.
    Retorna un dict con estadísticas.
    """
    data_yaml = Path(data_yaml)
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = data_yaml.parent / root

    stats = {}
    for split in ("train", "val"):
        img_dir = root / cfg[split]
        lbl_dir = img_dir.parent.parent / "labels" / img_dir.name

        imgs   = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        labels = sorted(lbl_dir.glob("*.txt"))

        sin_etiqueta = [i.stem for i in imgs if not (lbl_dir / (i.stem + ".txt")).exists()]
        sin_imagen   = [l.stem for l in labels if not any((img_dir / (l.stem + ext)).exists()
                                                           for ext in (".jpg", ".png"))]

        stats[split] = {
            "imagenes": len(imgs),
            "etiquetas": len(labels),
            "sin_etiqueta": sin_etiqueta,
            "sin_imagen": sin_imagen,
        }

        print(f"\n[{split.upper()}]")
        print(f"  Imágenes : {len(imgs)}")
        print(f"  Etiquetas: {len(labels)}")
        if sin_etiqueta:
            print(f"  ⚠️  Sin etiqueta: {sin_etiqueta}")
        if sin_imagen:
            print(f"  ⚠️  Sin imagen  : {sin_imagen}")
        if not sin_etiqueta and not sin_imagen:
            print("  ✅ Todo en orden")

    return stats


# ──────────────────────────────────────────────
# 2. VISUALIZACIÓN DE ANOTACIONES YOLO
# ──────────────────────────────────────────────

def yolo_bbox_a_pixel(bbox_yolo: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    """
    Convierte [cx, cy, bw, bh] (normalizados) a [x1, y1, x2, y2] en píxeles.
    """
    cx, cy, bw, bh = bbox_yolo
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2


def visualizar_anotacion(img_path: str | Path, lbl_path: str | Path,
                          nombres_clases: list[str] | None = None,
                          mostrar: bool = True) -> np.ndarray:
    """
    Dibuja los bounding boxes YOLO sobre la imagen y la muestra/retorna.

    Args:
        img_path: ruta a la imagen (.jpg/.png)
        lbl_path: ruta al archivo de etiqueta YOLO (.txt)
        nombres_clases: lista de nombres de clases (ej: ['house'])
        mostrar: si True abre ventana con matplotlib

    Returns:
        imagen BGR con boxes dibujados
    """
    img_path = Path(img_path)
    lbl_path = Path(lbl_path)
    nombres_clases = nombres_clases or ["house"]

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.imshow(img_rgb)
    ax.set_title(img_path.name, fontsize=13)
    ax.axis("off")

    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            lineas = f.read().strip().splitlines()

        colores = plt.cm.get_cmap("tab10", len(nombres_clases))

        for linea in lineas:
            partes = linea.strip().split()
            if len(partes) < 5:
                continue
            clase_id = int(partes[0])
            bbox     = list(map(float, partes[1:5]))
            x1, y1, x2, y2 = yolo_bbox_a_pixel(bbox, w, h)

            color  = colores(clase_id)
            nombre = nombres_clases[clase_id] if clase_id < len(nombres_clases) else str(clase_id)

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x1, max(y1 - 5, 0), nombre,
                    color="white", fontsize=10, fontweight="bold",
                    bbox=dict(facecolor=color, alpha=0.7, pad=2))
    else:
        ax.set_title(f"{img_path.name}  (sin etiqueta)", color="red", fontsize=13)

    if mostrar:
        plt.tight_layout()
        plt.show()

    # Retornar imagen con boxes para uso externo
    img_out = img.copy()
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for linea in f.read().strip().splitlines():
                partes = linea.strip().split()
                if len(partes) < 5:
                    continue
                bbox = list(map(float, partes[1:5]))
                x1, y1, x2, y2 = yolo_bbox_a_pixel(bbox, w, h)
                cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 200, 100), 2)
    return img_out


def visualizar_muestra(data_yaml: str | Path, n: int = 4, split: str = "train") -> None:
    """
    Muestra una cuadrícula de N imágenes aleatorias del split con sus anotaciones.
    """
    data_yaml = Path(data_yaml)
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = data_yaml.parent / root

    img_dir = root / cfg[split]
    lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
    nombres  = cfg.get("names", {})
    if isinstance(nombres, dict):
        nombres = [nombres[k] for k in sorted(nombres)]

    imagenes = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not imagenes:
        print(f"❌ No hay imágenes en {img_dir}")
        return

    muestra = random.sample(imagenes, min(n, len(imagenes)))
    cols    = min(2, len(muestra))
    rows    = (len(muestra) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = np.array(axes).flatten() if len(muestra) > 1 else [axes]

    for ax, img_path in zip(axes, muestra):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img      = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(img_path.name, fontsize=10)
        ax.axis("off")

        if lbl_path.exists():
            colores = plt.cm.get_cmap("tab10", max(len(nombres), 1))
            with open(lbl_path, "r") as f:
                for linea in f.read().strip().splitlines():
                    partes = linea.strip().split()
                    if len(partes) < 5:
                        continue
                    clase_id = int(partes[0])
                    bbox = list(map(float, partes[1:5]))
                    x1, y1, x2, y2 = yolo_bbox_a_pixel(bbox, w, h)
                    color  = colores(clase_id)
                    nombre = nombres[clase_id] if clase_id < len(nombres) else str(clase_id)
                    rect   = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor=color, facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(x1, max(y1 - 5, 0), nombre, color="white", fontsize=9,
                            bbox=dict(facecolor=color, alpha=0.7, pad=1))

    # Ocultar ejes sobrantes
    for ax in axes[len(muestra):]:
        ax.set_visible(False)

    plt.suptitle(f"Muestra del split '{split}'", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 3. CONVERSIÓN: PASCAL VOC → YOLO
# ──────────────────────────────────────────────

def voc_a_yolo(xmin: float, ymin: float, xmax: float, ymax: float,
               img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convierte coordenadas Pascal VOC a formato YOLO normalizado."""
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)


def yolo_a_voc(cx: float, cy: float, bw: float, bh: float,
               img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convierte coordenadas YOLO normalizadas a Pascal VOC en píxeles."""
    xmin = int((cx - bw / 2) * img_w)
    ymin = int((cy - bh / 2) * img_h)
    xmax = int((cx + bw / 2) * img_w)
    ymax = int((cy + bh / 2) * img_h)
    return xmin, ymin, xmax, ymax


# ──────────────────────────────────────────────
# 4. RESUMEN RÁPIDO DEL DATASET
# ──────────────────────────────────────────────

def resumen_dataset(data_yaml: str | Path) -> None:
    """Imprime un resumen completo del dataset."""
    print("\n" + "=" * 50)
    print("  RESUMEN DEL DATASET")
    print("=" * 50)
    verificar_dataset(data_yaml)
    print("=" * 50 + "\n")


# ──────────────────────────────────────────────
# Demo rápido si se ejecuta directamente
# ──────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR  = Path(__file__).resolve().parent.parent
    DATA_YAML = BASE_DIR / "data.yaml"

    if DATA_YAML.exists():
        resumen_dataset(DATA_YAML)
        visualizar_muestra(DATA_YAML, n=4, split="train")
    else:
        print(f"No se encontró {DATA_YAML}. Asegúrate de estar en la raíz del proyecto.")
