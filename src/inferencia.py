"""
inferencia.py
Script de inferencia: carga pesos entrenados y detecta casas en imágenes.
Uso: python src/inferencia.py --imagen ruta/imagen.jpg
     python src/inferencia.py --carpeta ruta/carpeta/
"""

import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
PESOS_DEFAULT = BASE_DIR / "models" / "house-yolo.pt"
SALIDA_DIR   = BASE_DIR / "resultados"

CONFIDENCE = 0.25   # umbral mínimo de confianza
IOU        = 0.45   # umbral NMS IoU


def cargar_modelo(pesos: Path) -> YOLO:
    """Carga el modelo YOLO desde los pesos entrenados."""
    if not pesos.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos: {pesos}\n"
            "Entrena primero con: python src/train_yolo.py"
        )
    model = YOLO(str(pesos))
    print(f"✅ Modelo cargado: {pesos.name}")
    return model


def inferir_imagen(model: YOLO, ruta_imagen: Path, guardar: bool = True) -> None:
    """Ejecuta detección sobre una imagen y muestra/guarda el resultado."""
    if not ruta_imagen.exists():
        print(f"❌ Imagen no encontrada: {ruta_imagen}")
        return

    results = model.predict(
        source=str(ruta_imagen),
        conf=CONFIDENCE,
        iou=IOU,
        save=False,          # controlamos el guardado nosotros
        verbose=False,
    )

    result = results[0]
    img_bgr = result.orig_img.copy()
    n_detecciones = len(result.boxes)

    print(f"\n🔍 {ruta_imagen.name} — {n_detecciones} casa(s) detectada(s)")

    # Dibujar bounding boxes manualmente para más control
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        label = f"house {conf:.2f}"

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(
            img_bgr, label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 200, 100), 2,
        )

    if guardar:
        SALIDA_DIR.mkdir(parents=True, exist_ok=True)
        salida = SALIDA_DIR / f"det_{ruta_imagen.name}"
        cv2.imwrite(str(salida), img_bgr)
        print(f"   💾 Guardado en: {salida}")

    # Mostrar si hay entorno visual disponible
    try:
        cv2.imshow("Detección — house YOLO", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        pass   # sin entorno gráfico (ej. servidor), solo guardar


def inferir_carpeta(model: YOLO, ruta_carpeta: Path) -> None:
    """Corre inferencia sobre todas las imágenes en una carpeta."""
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imagenes = [p for p in ruta_carpeta.iterdir() if p.suffix.lower() in extensiones]

    if not imagenes:
        print(f"❌ No se encontraron imágenes en: {ruta_carpeta}")
        return

    print(f"📂 Procesando {len(imagenes)} imagen(es) en {ruta_carpeta}…")
    for img_path in sorted(imagenes):
        inferir_imagen(model, img_path, guardar=True)

    print(f"\n✅ Resultados guardados en: {SALIDA_DIR}")


def main():
    global CONFIDENCE
    parser = argparse.ArgumentParser(
        description="Inferencia YOLO — Detección de casas colombianas"
    )
    parser.add_argument("--imagen",   type=str, help="Ruta a una imagen individual")
    parser.add_argument("--carpeta",  type=str, help="Ruta a una carpeta de imágenes")
    parser.add_argument("--pesos",    type=str, default=str(PESOS_DEFAULT),
                        help="Ruta al archivo .pt de pesos (default: models/house-yolo.pt)")
    parser.add_argument("--conf",     type=float, default=CONFIDENCE,
                        help="Umbral de confianza (default: 0.25)")
    args = parser.parse_args()

    CONFIDENCE = args.conf

    model = cargar_modelo(Path(args.pesos))

    if args.imagen:
        inferir_imagen(model, Path(args.imagen))
    elif args.carpeta:
        inferir_carpeta(model, Path(args.carpeta))
    else:
        parser.print_help()
        print("\n💡 Ejemplo: python src/inferencia.py --imagen dataset/images/val/casa01.jpg")


if __name__ == "__main__":
    main()
