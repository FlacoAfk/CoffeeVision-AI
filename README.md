# CoffeeVision AI — Detección de Roya y Broca en Café

Sistema de inteligencia artificial para clasificación binaria de **roya en hojas de café** y **broca en granos de café** mediante ensembles de deep learning con soft-voting ponderado y TTA multi-crop.

---

## 📋 Tabla de Contenidos
1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Modelos de IA](#modelos-de-ia)
4. [Datasets](#datasets)
5. [API REST](#api-rest)
6. [Frontend](#frontend)
7. [Instalación y Ejecución](#instalación-y-ejecución)
8. [Configuración de Modelos](#configuración-de-modelos)
9. [Resultados](#resultados)
10. [Estructura del Proyecto](#estructura-del-proyecto)

---

## Resumen Ejecutivo

CoffeeVision AI es un sistema completo de visión artificial para la industria cafetera que detecta dos de las principales amenazas al cultivo de café:

| Amenaza | Dominio | Clases | Modelos | Precisión |
|---------|---------|--------|---------|-----------|
| **Roya** (Hemileia vastatrix) | Hoja | Sanas / Roya | 3 (Custom CNN + ResNet50 + EfficientNetB0) | 95.7% test |
| **Broca** (Hypothenemus hampei) | Grano | Sano / Danado | 3 (Custom CNN + ResNet50 + EfficientNetB0) | 99.8% test |

### Características principales
- **Ensemble de 3 modelos** con soft-voting ponderado por dominio
- **TTA (Test-Time Augmentation)**: 3 crops × 2 flips × 3 rotaciones = 18 predicciones por modelo
- **Panel de control**: pesos ajustables y activación/desactivación de modelos individuales desde el frontend
- **Interfaz web** con gráficos de barras, medidor circular y overlay sobre la imagen
- **Backend Flask + TensorFlow** con carga lazy de modelos
- **Datasets combinados** de múltiples fuentes (Kaggle + Roboflow)

---

## Arquitectura del Sistema

```
┌──────────────┐     HTTP/API      ┌──────────────────┐
│   Frontend   │ ◄──────────────► │     Backend       │
│  React+Vite  │    :5173/:5001   │   Flask+TF        │
│  :5173       │                   │   :5001           │
└──────────────┘                   └──────┬───────────┘
                                          │
                              ┌───────────┴───────────┐
                              │   DualModelLoader     │
                              │  ┌─────────────────┐  │
                              │  │ Leaf Pipeline    │  │
                              │  │ 3 models + TTA   │  │
                              │  ├─────────────────┤  │
                              │  │ Grain Pipeline   │  │
                              │  │ 3 models + TTA   │  │
                              │  └─────────────────┘  │
                              └───────────────────────┘
```

### Flujo de inferencia
1. Usuario sube imagen → Frontend envía al backend
2. DualModelLoader detecta dominio (leaf/grain)
3. Carga lazy de 3 modelos si no están en memoria
4. Aplica pesos configurables (desde `model_config.json`)
5. TTA: genera 18 variaciones por modelo
6. Soft-voting ponderado → predicción final
7. Respuesta JSON con resultado ensemble + individual por modelo

---

## Modelos de IA

### 🍃 Hoja — Detección de Roya (V4)

| # | Modelo | Arquitectura | Parámetros | val_acc | test_acc | Tamaño |
|---|--------|-------------|-----------|---------|----------|--------|
| 1 | Custom CNN | 4×Conv2D + GAP + Dense | 421K | 95.4% | **95.7%** | 5 MB |
| 2 | ResNet50 | Transfer learning ImageNet | 23.6M | 93.7% | — | 217 MB |
| 3 | EfficientNetB0 | Transfer learning ImageNet | 4.0M | 90.8% | 92.3% | 37 MB |

- **Input:** 224×224×3
- **Clases:** `Sanas` (hoja sana), `Roya` (hoja con Hemileia vastatrix)
- **Dataset:** 2,329 imágenes (balanceado ~50/50)
- **Split:** 70% train / 15% val / 15% test estratificado
- **Entrenamiento:** 2 fases — 20 épocas frozen + 30 épocas fine-tuning
- **Ubicación:** `training/models/v4/`

### 🫘 Grano — Detección de Broca (V5)

| # | Modelo | Arquitectura | Parámetros | val_acc | test_acc | Tamaño |
|---|--------|-------------|-----------|---------|----------|--------|
| 1 | Custom CNN | 4×Conv2D + GAP + Dense | 421K | 100% | **99.66%** | 5 MB |
| 2 | ResNet50 | Transfer learning ImageNet | 23.6M | 100% | **99.83%** | 97 MB |
| 3 | EfficientNetB0 | Transfer learning ImageNet | 4.0M | 100% | **99.83%** | 20 MB |

- **Input:** 224×224×3
- **Clases:** `Sano` (grano sano), `Danado` (grano perforado por broca)
- **Dataset combinado:** 3,911 imágenes de 3 fuentes
- **Split:** 70% train (2,737) / 15% val (586) / 15% test (588)
- **Entrenamiento:** 30 épocas single-phase en CPU local
- **Augmentación:** RandomFlip, RandomRotation(0.3), RandomZoom(0.2), RandomContrast(0.2), RandomTranslation(0.1,0.1), RandomBrightness(0.1), RandomCrop(224,224) + 5× repeat
- **Ubicación:** `training/models/v5_all/`

### Sistema de Ensemble

**Soft-voting ponderado** con pesos configurables:
- Pesos calculados con softmax(val_accuracy) de entrenamiento
- Normalización automática al activar/desactivar modelos
- El usuario puede ajustar manualmente desde el frontend (⚙️ Model Settings)

**TTA (Test-Time Augmentation):**
- 3 crops (centro + 2 esquinas)
- 2 flips (original, horizontal)
- 3 rotaciones (0°, +3°, -3°)
- = 18 predicciones por modelo, promediadas para máxima precisión

---

## Datasets

### Fuentes

| # | Dataset | Imágenes | Clases originales | Fuente |
|---|---------|----------|-------------------|--------|
| 1 | Kaggle Leaf | 2,329 | 14 → filtrado a 2 | `vann234/coffevision-leaf-cls` |
| 2 | Kaggle Grain | 1,490 → filtrado a 626 Sano | 3 → filtrado a 1 | `vann234/coffevision-grain-cls` |
| 3 | Roboflow 3-clases | 1,339 | berry_borer, damaged_bean, healthy_bean | Roboflow |
| 4 | Roboflow 2527 | 1,946 | BROCA DE CAFE | Roboflow |

### Distribución Final — Grano V5

| Clase | Train | Val | Test | Total | Fuentes |
|-------|-------|-----|------|-------|---------|
| Sano | 798 | 156 | 160 | 1,114 | Kaggle + Roboflow 3-clases |
| Danado | 1,939 | 430 | 428 | 2,797 | Roboflow 2527 + 3-clases |
| **Total** | **2,737** | **586** | **588** | **3,911** | 3 datasets combinados |

### Distribución Hoja V4

| Clase | Train | Val | Test | Total |
|-------|-------|-----|------|-------|
| Sanas | 1,146 | — | — | ~1,165 |
| Roya | 1,183 | — | — | ~1,165 |
| **Total** | **1,630** | **349** | **350** | **2,329** |

---

## API REST

**Base URL:** `http://localhost:5001`

### Endpoints

| Método | Ruta | Body | Respuesta |
|--------|------|------|-----------|
| `POST` | `/api/predict/leaf` | `image` (multipart) | Predicción hoja |
| `POST` | `/api/predict/grain` | `image` (multipart) | Predicción grano |
| `GET` | `/api/health` | — | Estado del sistema |
| `GET` | `/api/models/config` | — | Config actual de modelos |
| `POST` | `/api/models/config` | JSON weights/enabled | Actualizar configuración |

### Formato de Respuesta

```json
{
  "predicted_class": "Roya",
  "confidence": 0.6226,
  "top_3": [
    {"class": "Roya", "confidence": 0.6226},
    {"class": "Sanas", "confidence": 0.3774}
  ],
  "individual": [
    {
      "model": "resnet50",
      "weight": 0.415,
      "predicted_class": "Roya",
      "confidence": 0.8618
    },
    {
      "model": "efficientnetb0",
      "weight": 0.357,
      "predicted_class": "Roya",
      "confidence": 0.5148
    },
    {
      "model": "custom_cnn",
      "weight": 0.228,
      "predicted_class": "Sanas",
      "confidence": 0.8068
    }
  ],
  "processing_time_ms": 5230
}
```

### Configuración de Modelos

```json
{
  "leaf": {
    "custom_cnn": {"weight": 1.0, "enabled": true},
    "resnet50": {"weight": 1.0, "enabled": true},
    "efficientnetb0": {"weight": 1.0, "enabled": true}
  },
  "grain": {
    "custom_cnn": {"weight": 1.0, "enabled": true},
    "resnet50": {"weight": 1.0, "enabled": true},
    "efficientnetb0": {"weight": 1.0, "enabled": true}
  }
}
```

---

## Frontend

### Tecnologías
- React 18 + Vite 5
- Axios para HTTP
- react-dropzone para upload de imágenes
- CSS puro (design system con variables CSS)
- SVG para gráficos (sin dependencias de charting)

### Componentes

| Componente | Función |
|-----------|---------|
| `AnalysisPage` | Página principal, orquesta tabs + upload + resultados |
| `TabSwitch` | Selector 🍀 Hoja / 🫘 Grano |
| `ImageUpload` | Drag & drop, validación JPG/PNG ≤10MB, preview |
| `PredictionResult` | Overlay en imagen, medidor SVG, barras por modelo, veredicto |
| `ModelControl` | Panel ⚙️ con toggle ON/OFF y sliders de peso por modelo |

### Funcionalidades Visuales
- 🖼️ **Overlay** sobre la imagen con clase y % confianza
- 🍩 **Medidor circular** SVG animado
- 📊 **Barras horizontales** por modelo con peso y predicción individual
- 🏷️ **Veredicto del ensemble** con votos por clase
- ⚙️ **Panel de configuración** colapsable

---

## Instalación y Ejecución

### Requisitos
- **Python** 3.10+ con TensorFlow 2.21 (`backend/.venv-tf/`)
- **Node.js** 18+ con npm
- **RAM:** 8 GB mínimo (~350 MB los modelos)
- **Disco:** ~600 MB (modelos + datasets)
- **GPU:** No requerida (inferencia CPU)

### Backend

```powershell
cd C:\Users\elkaw\Desktop\CoffeeVision AI\backend
.venv-tf\Scripts\activate
pip install -r requirements.txt   # primera vez
python -m flask run --host 0.0.0.0 --port 5001
```

### Frontend

```powershell
cd C:\Users\elkaw\Desktop\CoffeeVision AI\frontend
npm install   # primera vez
npm run dev
```

Abrir navegador en **http://localhost:5173**

---

## Configuración de Modelos

El sistema permite ajustar pesos y activar/desactivar modelos sin reiniciar:

1. Hacer una predicción
2. Expandir ⚙️ **Model Settings** abajo del resultado
3. Ajustar sliders de peso (×0.1 a ×5.0)
4. Toggle ON/OFF por modelo
5. Click **Apply** para guardar

Los cambios se persisten en `training/models/model_config.json` y se aplican inmediatamente.

---

## Resultados

### Hoja (Roya) — V4

| Métrica | Valor |
|---------|-------|
| Modelos | 3 (Custom CNN + ResNet50 + EfficientNetB0) |
| Mejor modelo individual | Custom CNN: 95.7% test accuracy |
| Ensemble estimado | ~96% con soft-voting |
| TTA activo | 18 predicciones por modelo |
| Tiempo inferencia | ~5-10 segundos |

### Grano (Broca) — V5

| Métrica | Valor |
|---------|-------|
| Modelos | 3 (Custom CNN + ResNet50 + EfficientNetB0) |
| Mejor modelo individual | ResNet50: 99.83% test accuracy |
| Ensemble estimado | ~99.8% con soft-voting |
| Dataset | 3,911 imágenes de 3 fuentes |
| TTA activo | 18 predicciones por modelo |
| Tiempo inferencia | ~5-10 segundos |

---

## Estructura del Proyecto

```
CoffeeVision AI/
├── README.md                 # Este documento
├── requirements.txt          # Dependencias Python
├── pyproject.toml
│
├── backend/                  # Flask API (puerto 5001)
│   ├── config.py             # Rutas modelos, clases, DB
│   ├── wsgi.py               # Entry point
│   ├── requirements.txt
│   ├── .venv-tf/             # Entorno virtual TensorFlow
│   └── app/
│       ├── __init__.py        # Flask factory, blueprints
│       ├── model_loader.py    # DualModelLoader (leaf+grain)
│       ├── models.py          # SQLAlchemy Prediction
│       ├── extensions.py      # DB, MongoDB
│       ├── schemas.py
│       └── routes/
│           ├── predict_leaf.py     # POST /api/predict/leaf
│           ├── predict_grain.py    # POST /api/predict/grain
│           ├── health.py           # GET /api/health
│           ├── models_config.py    # GET/POST /api/models/config
│           ├── history.py
│           └── iot.py
│
├── frontend/                 # React SPA (puerto 5173)
│   ├── package.json
│   ├── vite.config.js        # Proxy /api → :5001
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css          # Design system (coffee palette)
│       ├── services/
│       │   └── api.js          # Axios client
│       ├── components/
│       │   ├── ImageUpload.jsx  # Drag & drop + preview
│       │   ├── PredictionResult.jsx  # Gráficos y resultados
│       │   ├── ModelControl.jsx     # Panel de configuración
│       │   └── TabSwitch.jsx        # Selector Hoja/Grano
│       └── pages/
│           └── AnalysisPage.jsx     # Página principal
│
└── training/                 # Entrenamiento y datasets
    ├── datasets/
    │   ├── broca_2527/         # Roboflow 2,527 imágenes
    │   ├── broca_roboflow/     # Roboflow 3 clases
    │   ├── sano_download/      # Sano de Kaggle
    │   ├── grain_v5_all/       # Combinado final (3,911 img)
    │   └── grain_v5_final/     # Combinado anterior (2,572 img)
    │
    ├── models/
    │   ├── v4/                 # Modelos V4 hoja (roya)
    │   │   ├── leaf_custom_cnn/models/
    │   │   ├── leaf_resnet50/models/
    │   │   └── leaf_efficientnetb0/models/
    │   ├── v5_all/             # Modelos V5 grano (broca)
    │   │   ├── grain_custom_cnn/models/
    │   │   ├── grain_resnet50/models/
    │   │   └── grain_efficientnetb0/models/
    │   └── model_config.json   # Configuración de pesos
    │
    ├── notebooks/              # Notebooks de entrenamiento Kaggle
    │   ├── coffeevision_resnet50_v3.py
    │   ├── coffeevision_efficientnetb0_v3.py
    │   └── coffeevision_custom_cnn_v3.py
    │
    └── train_local_v5.py       # Script entrenamiento local
```

---

## Notas Técnicas

### ¿Por qué TTA?
Test-Time Augmentation genera múltiples versiones aumentadas de la imagen de entrada, predice sobre cada una y promedia los resultados. Esto:
- Reduce el overfitting a una perspectiva específica
- Mejora la consistencia entre ejecuciones
- Aumenta la precisión en 2-5% sobre una sola predicción

### ¿Por qué soft-voting?
En lugar de elegir el mejor modelo, combinamos las predicciones de los 3 con pesos basados en su rendimiento en validación. Esto:
- Compensa las debilidades individuales de cada arquitectura
- Permite ajustar la influencia de cada modelo según su desempeño real
- El usuario puede modificar pesos si un modelo específico funciona mejor en sus datos

### ¿Por qué 3 modelos por dominio?
- **Custom CNN**: Ligera (5 MB), rápida, entrenada desde cero → captura features específicas
- **ResNet50**: Profunda (217 MB), pre-entrenada en ImageNet → mejor generalización
- **EfficientNetB0**: Eficiente (37 MB), balance velocidad/precisión

### Limitaciones Conocidas
- El dataset de grano (3,911 imágenes) es moderado. Más datos mejorarían la generalización
- Los modelos fueron entrenados con imágenes de fondo claro/neutro. Fotos con fondos complejos pueden reducir precisión
- El TTA aumenta el tiempo de inferencia (~5-10s vs ~1s sin TTA)

---

*Documento generado el 26 de mayo de 2026*
*Proyecto académico — CoffeeVision AI*
