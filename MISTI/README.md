# MISTI — MIT Intensive Program in Machine Learning

Materiales y proyecto final del programa intensivo **MISTI (MIT-Peru)** dictado por el MIT a la Maestría en Ciencia de Datos de UTEC. El curso cubrió fundamentos de Python, EDA, Machine Learning clásico y Deep Learning en dos semanas presenciales.

**Autor:** Christian Cajusol — christian.cajusol@utec.edu.pe

---

## Estructura del repositorio

```
MISTI/
│
├── Fundamentos/          ← Tutoriales de NumPy y Pandas (base del curso)
├── EDA/                  ← Ejercicios resueltos de Semana 0
├── week_1/               ← Semana 1: ML clásico, regresión, validación, redes neuronales
├── week_2/               ← Semana 2: CNN, NLP, Reinforcement Learning
└── Proyecto_final_SAFEGUARD/  ← Proyecto final: Sistema de detección de caídas
```

---

## Índice de notebooks y scripts

### Fundamentos (NumPy + Pandas)

| Archivo | Tema |
|---------|------|
| [NumPy_Tutorial.ipynb](Fundamentos/NumPy_Tutorial.ipynb) | Arrays N-dimensionales, operaciones vectorizadas, benchmarks vs Python puro |
| [Pandas1_DataTypes.ipynb](Fundamentos/Pandas1_DataTypes.ipynb) | Series, DataFrames, indexing (loc/iloc), selección condicional |
| [Pandas2_MissingData.ipynb](Fundamentos/Pandas2_MissingData.ipynb) | dropna(), fillna() con estadísticas, imputación |
| [Pandas3_Operations.ipynb](Fundamentos/Pandas3_Operations.ipynb) | .apply(), lambda, value_counts, extracción de texto |
| [Pandas4_DataAnalysis.ipynb](Fundamentos/Pandas4_DataAnalysis.ipynb) | GroupBy, merge/join, concatenación, sorting |
| [Pandas5_DataInputandOutput.ipynb](Fundamentos/Pandas5_DataInputandOutput.ipynb) | read/write CSV, Excel, HTML, JSON |

### EDA — Semana 0

| Archivo | Dataset | Análisis |
|---------|---------|---------|
| [RESOLUTION_Week0_EcommercePandas_Exercises.ipynb](EDA/RESOLUTION_Week0_EcommercePandas_Exercises.ipynb) | E-commerce Purchases (10,000 transacciones) | Filtrado, estadísticas, extracción de dominios de email, análisis temporal de tarjetas |
| [RESOLUTION_Week0_SFSalariesExercise.ipynb](EDA/RESOLUTION_Week0_SFSalariesExercise.ipynb) | SF Salaries (148,654 registros) | Conversión de tipos, groupby temporal, detección de salarios negativos, búsqueda textual |

### Semana 1 — Machine Learning Clásico

| Archivo | Tema | Dataset |
|---------|------|---------|
| [XTIAN_Housing_DataExploration_SOLUTIONS.ipynb](week_1/XTIAN_Housing_DataExploration_SOLUTIONS.ipynb) | EDA: valores faltantes, distribuciones, correlaciones | California Housing |
| [XTIAN_Housing_ModelFittingandValidation_STUDENT_2026.ipynb](week_1/XTIAN_Housing_ModelFittingandValidation_STUDENT_2026.ipynb) | Regresión lineal, Ridge, Lasso, K-fold CV, curvas de aprendizaje | California Housing |
| [XTIAN_Kobe_FeatureEngineering_EXERCISES.ipynb](week_1/XTIAN_Kobe_FeatureEngineering_EXERCISES.ipynb) | Feature engineering, encoding, selección de variables | Kobe Bryant shots |
| [XTIAN_Kobe_ModelValidation_EXERCISES.ipynb](week_1/XTIAN_Kobe_ModelValidation_EXERCISES.ipynb) | Clasificación, precision/recall, ROC, AUC, threshold | Kobe Bryant shots |
| [XTIAN_student_Bike_Sharing_Demand_gradient_descent.ipynb](week_1/XTIAN_student_Bike_Sharing_Demand_gradient_descent.ipynb) | Gradient descent desde primeros principios | Bike Sharing Demand |
| [XTIAN_IntrotoModelTraining_NeuralNetworks_EXERCISES.ipynb](week_1/XTIAN_IntrotoModelTraining_NeuralNetworks_EXERCISES.ipynb) | Arquitecturas secuenciales, activaciones, backprop, Keras | Crédito / Sintético |
| [CHRISTIAN_2026MISTIPeru_ConceptChallenge_DimensionalityReduction_EXERCISES.ipynb](week_1/CHRISTIAN_2026MISTIPeru_ConceptChallenge_DimensionalityReduction_EXERCISES.ipynb) | PCA, t-SNE, autoencoders, compresión | Alta dimensionalidad |

### Semana 2 — Deep Learning

| Archivo | Tema | Dataset |
|---------|------|---------|
| [XTIAN_2025MISTIPeru_IntrotoModelTraining_ConvolutionalNeuralNetworks_EXERCISES.ipynb](week_2/XTIAN_2025MISTIPeru_IntrotoModelTraining_ConvolutionalNeuralNetworks_EXERCISES.ipynb) | Conv2D, pooling, feature maps, clasificación de imágenes | MNIST / CIFAR-10 |
| [XTIAN_2025MISTIPeru_CNNs_InherentFeatureEngineering_EXERCISES.ipynb](week_2/XTIAN_2025MISTIPeru_CNNs_InherentFeatureEngineering_EXERCISES.ipynb) | Visualización de capas intermedias, feature maps automáticos | MNIST / CIFAR-10 |
| [XTIAN_2026MISTIPeru_IntrotoFeatureEngineering_Images_student_EXERCISES.ipynb](week_2/XTIAN_2026MISTIPeru_IntrotoFeatureEngineering_Images_student_EXERCISES.ipynb) | Data augmentation: rotación, flip, normalización, color jitter | Imágenes |
| [XTIAN_2026MISTIPeru_ModelValidation_NeuralNetworks_EXERCISES.ipynb](week_2/XTIAN_2026MISTIPeru_ModelValidation_NeuralNetworks_EXERCISES.ipynb) | Early stopping, callbacks, LR scheduling, detección de overfitting | Redes neuronales |
| [XTIAN_2025MISTIPeru_ContentChallenge_WordEmbeddings_EXERCISES.ipynb](week_2/XTIAN_2025MISTIPeru_ContentChallenge_WordEmbeddings_EXERCISES.ipynb) | Word2Vec, GloVe, similitud semántica, analogías | NLP |
| [XTIAN_2025_of_ReinforcementLearningInPractice.ipynb](week_2/XTIAN_2025_of_ReinforcementLearningInPractice.ipynb) | Q-learning, policy gradients, estado/acción/recompensa | Juegos / Control |

---

## Proyecto Final — SAFEGUARD

### Descripción

**SAFEGUARD** es un sistema de detección de caídas en tiempo real para **seguridad industrial**, desarrollado como proyecto final del programa MISTI. Combina visión por computadora (pose estimation con MediaPipe BlazePose) con modelos de Machine Learning y Deep Learning para detectar caídas de personas en videos de vigilancia.

> **Motivación:** En entornos industriales, una caída no detectada a tiempo puede ser fatal. El sistema busca alcanzar **Recall = 100%** — ninguna caída sin detectar, incluso a costa de algunas falsas alarmas.

### Arquitectura del sistema

```
Video / Cámara / Stream RTSP
         │
         ▼
┌─────────────────────────────┐
│  FASE 1: EXTRACCIÓN DE POSE │
│  BlazePose (3 variantes)    │
│  33 keypoints × (x,y,z,vis) │
│  → 132 features/frame       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  FASE 2: FEATURE ENGINEERING    │
│  + 8 features derivados:        │
│  ángulo de torso, altura/ancho  │
│  del cuerpo, ratio aspecto,     │
│  centro de masa, distancias     │
│  hombros/caderas, ángulo piernas│
│  → 140 features/frame totales   │
└─────────────┬───────────────────┘
              │
      ┌───────┴────────┐
      │ Frame-based    │  Sequence-based (30 frames)
      ▼                ▼
┌──────────┐    ┌──────────────────────┐
│  Random  │    │  create_sequences.py │
│  Forest  │    │  (ventana deslizante)│
│  v1 / v2 │    └──────────┬───────────┘
└──────────┘               │
                   ┌───────┴────────┐
                   ▼                ▼
            ┌──────────┐    ┌──────────────┐
            │   LSTM   │    │ Transformer  │
            │ BiLSTM×2 │    │ 3 bloques    │
            └──────────┘    │ 8-head attn  │
                            └──────────────┘
              │
              ▼
┌─────────────────────────────┐
│  SALIDA                     │
│  Probabilidad de caída      │
│  Alerta + Esqueleto en video│
│  Log con timestamps         │
└─────────────────────────────┘
```

---

### Scripts del proyecto

#### Fase 1 — Extracción de Keypoints

| Script | Modelo BlazePose | Velocidad | Precisión | Output |
|--------|-----------------|-----------|-----------|--------|
| [safeguard_keypoints_LITE.py](Proyecto_final_SAFEGUARD/safeguard_keypoints_LITE.py) | LITE | ~10ms/frame | Básica | keypoints_LITE.csv |
| [safeguard_keypoints_FULL.py](Proyecto_final_SAFEGUARD/safeguard_keypoints_FULL.py) | FULL | ~20ms/frame | Media | keypoints_FULL.csv |
| [safeguard_keypoints_HEAVY.py](Proyecto_final_SAFEGUARD/safeguard_keypoints_HEAVY.py) | HEAVY | ~50ms/frame | Alta | keypoints_HEAVY.csv |

Cada script:
1. Descarga el modelo BlazePose desde Google Cloud
2. Escanea los datasets `le2i` y `ur_fall` (etiquetando frames por nombre de carpeta)
3. Extrae 33 keypoints × 4 valores (x, y, z, visibilidad) = **132 features por frame**
4. Exporta CSV con 5 columnas de metadatos + 132 features

#### Fase 2 — Preprocesamiento del Dataset

| Script | Función |
|--------|---------|
| [analyze_dataset.py](Proyecto_final_SAFEGUARD/analyze_dataset.py) | Estadísticas del dataset: distribución de clases, calidad de keypoints, frames por video |
| [balance_dataset.py](Proyecto_final_SAFEGUARD/balance_dataset.py) | Balancea el dataset a ratio 1:1 (fall:ADL) — el dataset original tiene ratio 9:1 |
| [create_sequences.py](Proyecto_final_SAFEGUARD/create_sequences.py) | Crea ventanas deslizantes de 30 frames con stride=15 → numpy arrays para LSTM/Transformer |

**`create_sequences.py` — detalle:**
```
keypoints_HEAVY.csv
    ↓ Agrupar por carpeta (= 1 video)
    ↓ Ordenar frames cronológicamente
    ↓ Ventana deslizante: 30 frames, stride=15 (50% overlap)
    ↓ Label: ¿contiene transición de caída?
    → X_train.npy  (N_seq, 30, 396)
    → y_train.npy  (N_seq,)
    → X_test.npy, y_test.npy (20% split)
```

#### Fase 3 — Entrenamiento de Modelos

##### `train_fall_detector.py` — Random Forest v1 (baseline)

- **Tipo:** Clasificación por frame individual (sin contexto temporal)
- **Input:** keypoints_LITE.csv
- **Features adicionales creadas:**

| Feature | Cálculo | Qué captura |
|---------|---------|-------------|
| `torso_angle` | `nose_y - hip_y` | Orientación vertical del cuerpo |
| `body_height` | `max_y - min_y` | Alto cuando de pie, bajo al caer |
| `body_width` | `max_x - min_x` | Ancho del bounding box |
| `aspect_ratio` | `height / width` | Ratio vertical (>1 parado, <1 caído) |
| `center_of_mass_y` | Media ponderada de Y | Posición vertical del centro |
| `shoulder_distance` | Distancia euclidiana hombros | Simetría lateral |
| `hip_distance` | Distancia euclidiana caderas | Apertura de caderas |
| `leg_angle` | `hip_y - ankle_y` | Extensión vertical de piernas |

- **Modelo:** `RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2)`
- **Evaluación:** GridSearchCV 5-fold optimizando F1-score
- **Outputs:** `modelo_caidas.pkl`, `scaler.pkl`, `report.json`, figura 4-paneles
- **Resultado:**

| Métrica | Valor |
|---------|-------|
| Accuracy | 87.5% |
| Precision | 86.3% |
| Recall | 88.9% |
| AUC-ROC | 94.2% |

> **Limitación fundamental:** No puede distinguir "persona acostada" de "persona que cayó" — ambas tienen keypoints similares.

---

##### `train_fall_detector_v2.py` — Random Forest v2 (mejorado)

- **Mejoras sobre v1:**
  1. Dataset balanceado 1:1 (fall:ADL)
  2. Anti-overfitting: `max_depth=10`, `min_samples_leaf=10`, `max_features='sqrt'`
  3. `class_weight='balanced'`
  4. **Optimización de threshold:** busca el threshold que maximiza Recall ≥ 95%
- **Outputs:** Figura de 6 paneles (matriz de confusión, ROC, PR curve, distribución de probabilidades con threshold dual, feature importances top-20, análisis de sensibilidad de threshold)
- **Resultado:**

| Métrica | v1 | v2 |
|---------|----|----|
| Accuracy | 87.5% | 90.2% |
| Recall | 88.9% | 94.9% |
| AUC-ROC | 94.2% | 96.1% |

> **Conclusión:** Mejora notable, pero el problema de fondo persiste — la información de un solo frame es insuficiente.

---

##### `train_lstm_detector.py` — LSTM Bidireccional

- **Innovación clave:** Procesa **secuencias de 30 frames** — detecta la *transición* de caída, no solo la pose final
- **Input:** `X_train.npy` de shape `(N, 30, 396)` — 396 features = 132 keypoints × 3 variantes de escala
- **Arquitectura:**
  ```
  Input: (batch, 30, 396)
      ↓
  BiLSTM(128) → BatchNorm → Dropout(0.3)
      ↓
  BiLSTM(64)  → BatchNorm → Dropout(0.3)
      ↓
  Dense(32, relu) → Dropout(0.3)
      ↓
  Dense(1, sigmoid)  →  P(caída)
  ```
- **Entrenamiento:** Adam(lr=0.001), EarlyStopping(patience=15), ReduceLROnPlateau, class weights automáticos
- **Resultado:**

| Métrica | Valor |
|---------|-------|
| Accuracy | 96.1% |
| Precision | 98.2% |
| **Recall** | **100%** ✅ |
| AUC-ROC | 99.8% |

---

##### `train_transformer_detector.py` — Transformer con Atención

- **Innovación:** Mecanismo de **self-attention** — ve toda la secuencia simultáneamente y aprende a enfocarse en los frames críticos del momento de caída
- **Capas personalizadas:**
  - `PositionalEncoding`: encoding sinusoidal de posición temporal
  - `TransformerBlock`: Multi-Head Attention (8 cabezas) + Feed-Forward + LayerNorm + Residual
- **Arquitectura:**
  ```
  Input: (batch, 30, 396)
      ↓
  Positional Encoding
      ↓
  TransformerBlock × 3  (8 cabezas, FF dim=256, Dropout=0.3)
      ↓
  Global Average Pooling
      ↓
  Dense(128, relu) → Dropout(0.4)
      ↓
  Dense(64, relu)  → Dropout(0.4)
      ↓
  Dense(1, sigmoid)
  ```
- **Resultado:**

| Métrica | Valor |
|---------|-------|
| Accuracy | 97.3% |
| Precision | 99.1% |
| **Recall** | **100%** ✅ |
| AUC-ROC | 99.9% |

---

#### Fase 4 — Evaluación, Benchmarking y Demo

| Script | Función |
|--------|---------|
| [comparacion_modelos.py](Proyecto_final_SAFEGUARD/comparacion_modelos.py) | Carga los 4 reportes y genera tabla comparativa + gráfico de barras de Recall + scatter FP vs FN |
| [safeguard_benchmark.py](Proyecto_final_SAFEGUARD/safeguard_benchmark.py) | Benchmark de velocidad de LITE/FULL/HEAVY: FPS, latencia por imagen, tasa de éxito |
| [safeguard_benchmark_graficos.py](Proyecto_final_SAFEGUARD/safeguard_benchmark_graficos.py) | Genera gráficos comparativos del benchmark |
| [generate_model_comparison_charts.py](Proyecto_final_SAFEGUARD/generate_model_comparison_charts.py) | Visualizaciones de la evolución de los 4 modelos |
| [matriz_confusion_RF_correcta.py](Proyecto_final_SAFEGUARD/matriz_confusion_RF_correcta.py) | Regrena la matriz de confusión de RF con threshold optimizado |
| [demo_video_safeguard.py](Proyecto_final_SAFEGUARD/demo_video_safeguard.py) | Demo en tiempo real: video/webcam/RTSP → detección de caídas con overlay visual |
| [demo_video_lstm.py](Proyecto_final_SAFEGUARD/demo_video_lstm.py) | Demo en tiempo real con modelo LSTM |
| [demo_video_transformer.py](Proyecto_final_SAFEGUARD/demo_video_transformer.py) | Demo en tiempo real con modelo Transformer |
| [extract_frames_safeguard.py](Proyecto_final_SAFEGUARD/extract_frames_safeguard.py) | Extrae frames de video para crear dataset personalizado |

**`demo_video_safeguard.py` — pipeline de inferencia en tiempo real:**
```python
while frame = read_from_source():
    keypoints = blazepose_heavy(frame)        # ~50ms
    features = engineer_features(keypoints)   # ~1ms
    features = scaler.transform(features)     # ~0ms
    prob = rf_model.predict_proba(features)   # ~1ms
    
    if consecutive_detections >= 3:           # confirmación anti-falsos
        draw_alert(frame, "CAÍDA DETECTADA")
    else:
        draw_skeleton(frame, color=GREEN)
```
- **Controles interactivos:** Q (salir), P (pausa), S (screenshot), R (reset contadores)
- **Fuentes:** archivo MP4/AVI, webcam (índice 0), stream RTSP (cámaras IP)

---

### Comparativa final de modelos

| Modelo | Input | Accuracy | Precision | Recall | AUC-ROC | Inferencia |
|--------|-------|----------|-----------|--------|---------|------------|
| Random Forest v1 | 1 frame | 87.5% | 86.3% | 88.9% | 94.2% | ~1ms |
| Random Forest v2 | 1 frame (balanceado) | 90.2% | 89.1% | 94.9% | 96.1% | ~1ms |
| LSTM Bidireccional | 30 frames | 96.1% | 98.2% | **100%** | 99.8% | ~150ms |
| Transformer | 30 frames | 97.3% | 99.1% | **100%** | 99.9% | ~200ms |

> **Conclusión clave:** En sistemas de seguridad críticos, **Recall es la métrica prioritaria** — una caída no detectada puede ser fatal. Los modelos temporales (LSTM/Transformer) alcanzan Recall=100% porque detectan la *transición* de la caída en lugar de solo el estado final. El Transformer supera marginalmente al LSTM en precision y accuracy, pero ambos son equivalentes en el objetivo crítico.

---

### Documentación técnica adicional

Los archivos `.md` en `Proyecto_final_SAFEGUARD/` son guías de estudio generadas durante el desarrollo:

| Archivo | Contenido |
|---------|-----------|
| [01_METRICAS_CLASIFICACION_GUIA_COMPLETA.md](Proyecto_final_SAFEGUARD/01_METRICAS_CLASIFICACION_GUIA_COMPLETA.md) | Guía completa de métricas: precision, recall, F1, AUC, cuándo usar cada una |
| [02_EXPLICACION_CODIGO_DETALLADA.md](Proyecto_final_SAFEGUARD/02_EXPLICACION_CODIGO_DETALLADA.md) | Walkthrough línea por línea de los scripts principales |
| [03_FUNDAMENTOS_MATEMATICOS.md](Proyecto_final_SAFEGUARD/03_FUNDAMENTOS_MATEMATICOS.md) | Matemática detrás de Random Forest, LSTM, Transformer y positional encoding |
| [04_INTERPRETACION_GRAFICOS.md](Proyecto_final_SAFEGUARD/04_INTERPRETACION_GRAFICOS.md) | Cómo leer e interpretar cada gráfico generado por el sistema |
| [05_EXPLICACION_6_GRAFICOS_TRANSFORMER.md](Proyecto_final_SAFEGUARD/05_EXPLICACION_6_GRAFICOS_TRANSFORMER.md) | Análisis de los 6 gráficos de entrenamiento del Transformer |
| [06_ANALISIS_COMPLETO_EVOLUCION_4_MODELOS.md](Proyecto_final_SAFEGUARD/06_ANALISIS_COMPLETO_EVOLUCION_4_MODELOS.md) | Análisis narrativo de la evolución desde RF hasta Transformer |
| [MODEL_COMPARISON_GUIDE.md](Proyecto_final_SAFEGUARD/MODEL_COMPARISON_GUIDE.md) | Guía rápida de cuándo usar cada modelo según el contexto de despliegue |

---

### Cómo ejecutar el sistema SAFEGUARD

```bash
# 1. Instalar dependencias
pip install mediapipe tensorflow scikit-learn opencv-python pandas numpy tqdm

# 2. Extraer keypoints del dataset
python safeguard_keypoints_HEAVY.py

# 3. Analizar y balancear el dataset
python analyze_dataset.py
python balance_dataset.py

# 4. Crear secuencias (para LSTM/Transformer)
python create_sequences.py

# 5. Entrenar modelos
python train_fall_detector_v2.py     # Random Forest
python train_lstm_detector.py        # LSTM
python train_transformer_detector.py # Transformer

# 6. Comparar resultados
python comparacion_modelos.py

# 7. Demo en tiempo real
python demo_video_transformer.py --source 0          # webcam
python demo_video_transformer.py --source video.mp4  # archivo
python demo_video_transformer.py --source rtsp://... # cámara IP
```

---

### Análisis de calidad del código

#### Fortalezas ✅
- **Diseño modular:** cada script tiene responsabilidad única (extracción, entrenamiento, demo)
- `random_state=42` en todos los experimentos → reproducibilidad garantizada
- Manejo de errores con `try/except` en la extracción de keypoints
- Logs detallados con `tqdm`, reportes JSON con metadatos completos
- Confirmación anti-falsas alarmas (3 frames consecutivos) en el demo
- Documentación interna exhaustiva (docstrings, comentarios ASCII)

#### Observaciones ⚠️
- **Rutas hardcodeadas** (`D:\APRENDIZAJE\...`) — requieren adaptación para otros sistemas
- `print()` en lugar de módulo `logging` — no permite niveles ni redirección a archivo
- Configuraciones dispersas en dicts por script en lugar de un `config.yaml` centralizado
- `train_lstm_detector_gpu.py` y `safeguard_keypoints_HEAVY.py` — versiones sin comentar si difieren del FULL
- No implementa SMOTE para el balanceo (usa undersampling simple)

---

### Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| MediaPipe BlazePose | Estimación de pose (33 keypoints) |
| TensorFlow / Keras | LSTM, Transformer, CNN (semana 2) |
| scikit-learn | Random Forest, preprocessing, métricas |
| OpenCV | Lectura/escritura de video, overlay |
| NumPy, Pandas | Procesamiento de datos |
| Matplotlib, Seaborn | Visualizaciones |
| tqdm | Barras de progreso |

### Datasets utilizados

| Dataset | Contenido | Uso |
|---------|-----------|-----|
| LE2I Fall Detection | Videos de caídas en entorno doméstico | Entrenamiento SAFEGUARD |
| UR Fall Detection | Videos de caídas en laboratorio | Entrenamiento SAFEGUARD |
| E-commerce Purchases | 10,000 transacciones ficticias | EDA Semana 0 |
| SF Salaries | 148,654 salarios municipales | EDA Semana 0 |
| California Housing | Precios de viviendas | Semana 1 |
| Kobe Bryant Shots | 30,697 tiros de basquetbol | Semana 1 |
| Bike Sharing Demand | Demanda horaria de bicicletas | Semana 1 |
| MNIST / CIFAR-10 | Imágenes de dígitos y objetos | Semana 2 |
