"""
train_yolo.py
Script de entrenamiento para detección de casas colombianas con YOLOv8.
Uso: python src/train_yolo.py
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO


# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent   # raíz del proyecto
DATA_YAML  = BASE_DIR / "casas.v1i.yolov8" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"

HYPERPARAMS = {
    "model"  : "yolov8n.pt",   # modelo preentrenado (nano = liviano)
    "data"   : str(DATA_YAML),
    "epochs" : 50,
    "imgsz"  : 640,
    "batch"  : 8,              # reducir a 4 si hay OOM en GPU
    "device" : 0,              # 0 = primera GPU, 'cpu' para CPU
    "workers": 2,
    "cache"  : False,
    "project": str(MODELS_DIR),
    "name"   : "house-yolo",
    "exist_ok": True,
}


def verificar_dataset(data_yaml: Path) -> bool:
    """Verifica que el dataset tenga imágenes antes de entrenar."""
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root

    train_path = root / cfg["train"]
    val_path   = root / cfg["val"]

    n_train = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
    n_val   = len(list(val_path.glob("*.jpg")))   + len(list(val_path.glob("*.png")))

    print(f"📂 Train: {n_train} imágenes | Val: {n_val} imágenes")

    if n_train == 0:
        print("❌ No hay imágenes de entrenamiento. Agrega imágenes a dataset/images/train/")
        return False
    if n_val == 0:
        print("⚠️  No hay imágenes de validación. Agrega imágenes a dataset/images/val/")
        return False
    return True


def entrenar():
    print("=" * 55)
    print("  Taller YOLO — Detección de Casas Colombianas")
    print("=" * 55)

    if not verificar_dataset(DATA_YAML):
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar modelo preentrenado
    model = YOLO(HYPERPARAMS["model"])
    print(f"\n✅ Modelo cargado: {HYPERPARAMS['model']}")
    print(f"   Épocas: {HYPERPARAMS['epochs']} | imgsz: {HYPERPARAMS['imgsz']} | batch: {HYPERPARAMS['batch']}\n")

    # Entrenamiento
    results = model.train(**{k: v for k, v in HYPERPARAMS.items() if k != "model"})

    # Guardar pesos finales en models/
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    dest = MODELS_DIR / "house-yolo.pt"
    if best_weights.exists():
        import shutil
        shutil.copy(best_weights, dest)
        print(f"\n✅ Pesos guardados en: {dest}")
    else:
        print(f"\n⚠️  No se encontró best.pt en {best_weights}")

    print("\n📊 Métricas finales:")
    print(f"   mAP@0.5     : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   Precision   : {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"   Recall      : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    entrenar()
