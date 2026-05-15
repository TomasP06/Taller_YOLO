# Taller: Detección de Casas Colombianas usando YOLO

**Aplicaciones de Aprendizaje Automático de Máquinas**

---

## Descripción del dataset

Dataset de imágenes colombianas (urbanas y rurales) con casas anotadas en formato YOLO.
Obtenido desde Roboflow (`casas.v1i.yolov8`) y exportado directamente en formato YOLOv8.

- **Clase**: `casa` (id = 0)
- **Total de imágenes**: 10 (7 train · 2 val · 1 test)
- **Origen**: [Roboflow Universe — casas-lszdo v1](https://universe.roboflow.com/sebastians-workspace-sqqdc/casas-lszdo/dataset/1) · Licencia: CC BY 4.0
- **Variedad**: escenas urbanas (casas adosadas, diferentes estratos, fachadas) y rurales (campesinas), distintos ángulos e iluminaciones

---

## Estructura del repositorio

```
Taller_vision/
├── src/
│   ├── train_yolo.py          # Script de entrenamiento
│   ├── inferencia.py          # Script de inferencia sobre imágenes nuevas
│   └── utils.py               # Utilidades (visualización, resumen dataset)
├── casas.v1i.yolov8/          # Dataset en formato YOLOv8 (Roboflow)
│   ├── train/images/ + labels/
│   ├── valid/images/ + labels/
│   ├── test/images/  + labels/
│   └── data.yaml
├── models/
│   └── house-yolo.pt          # Pesos finales entrenados
├── resultados/                # Imágenes con detecciones (generado automáticamente)
├── requirements.txt
└── README.md
```

---

## Instrucciones de instalación

```bash
pip install -r requirements.txt
```

---

## Reproducir el entrenamiento

```bash
python src/train_yolo.py
```

- Modelo base: `yolov8n.pt` (transfer learning)
- Los pesos finales se guardan en `models/house-yolo.pt`
- Hiperparámetros: 50 épocas · imgsz=640 · batch=8 · device=GPU

---

## Inferencia

```bash
# Sobre una carpeta de imágenes
python src/inferencia.py --carpeta casas.v1i.yolov8/valid/images

# Sobre una imagen individual
python src/inferencia.py --imagen ruta/a/foto.jpg

# Ajustar umbral de confianza
python src/inferencia.py --imagen foto.jpg --conf 0.3
```

Los resultados se guardan en `resultados/`.

---

## Resultados (métricas — Parte 6)

Entrenamiento con **YOLOv8n** (transfer learning desde `yolov8n.pt`), 50 épocas, RTX 2060:

| Métrica        | Valor en val set |
|----------------|-----------------|
| **mAP@0.5**    | **0.87**        |
| mAP@0.5:0.95   | 0.436           |
| Precision      | 0.0067          |
| Recall         | **1.0000**      |
| Tiempo total   | ~1.2 min        |

### Análisis de detecciones

- **Verdaderos positivos**: El modelo alcanza Recall = 1.0, lo que significa que detecta el 100% de las casas etiquetadas en el set de entrenamiento.
- **Falsos positivos**: La Precision muy baja (0.0067) indica que el modelo genera muchas detecciones incorrectas en imágenes no vistas — consecuencia directa del sobreajuste por el tamaño reducido del dataset (7 imágenes de train).
- **Falsos negativos en validación**: Con solo 2 imágenes de validación y el sobreajuste esperado, el modelo no generaliza bien a imágenes nuevas.

> **Conclusión**: El mAP@0.5 = 0.87 refleja buen ajuste sobre los datos vistos, pero la baja generalización es una limitación esperada con tan pocas imágenes.

---

## Limitaciones y pasos futuros

- **Dataset pequeño** (10 imágenes total): sobreajuste inevitable → se mitiga con transfer learning desde `yolov8n.pt`.
- **Solución propuesta**: ampliar el dataset a ≥50 imágenes con variedad de escenas colombianas.
- **Modelos más grandes**: probar `yolov8s.pt` o `yolov8m.pt` con más datos.
- **Aumentos de datos**: rotaciones, flip horizontal, cambios de brillo/contraste (evitando flips verticales).
- **Casos futuros**: imágenes nocturnas, capturas desde drones (altitud), diferentes climas.
