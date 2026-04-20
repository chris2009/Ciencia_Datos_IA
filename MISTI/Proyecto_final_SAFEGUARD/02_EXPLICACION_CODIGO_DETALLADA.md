# 💻 Explicación Detallada del Código

## SafeGuard Vision AI | MIT Global Teaching Labs 2025
 
### Análisis Línea por Línea de Cada Script

---

## 📑 Tabla de Contenidos

1. [Arquitectura General del Proyecto](#1-arquitectura-general-del-proyecto)
2. [Script 1: create_sequences.py](#2-script-1-create_sequencespy)
3. [Script 2: train_lstm_detector.py](#3-script-2-train_lstm_detectorpy)
4. [Script 3: train_transformer_detector.py](#4-script-3-train_transformer_detectorpy)
5. [Script 4: demo_video_lstm.py](#5-script-4-demo_video_lstmpy)
6. [Componentes Comunes](#6-componentes-comunes)
7. [Flujo de Datos Completo](#7-flujo-de-datos-completo)

---

## 1. Arquitectura General del Proyecto

### Estructura de Archivos

```
SafeGuard Vision AI/
│
├── 📊 DATOS
│   ├── keypoints_HEAVY.csv          # Dataset original (16,930 frames)
│   └── safeguard_sequences/
│       ├── X_train.npy              # Secuencias de entrenamiento
│       ├── X_test.npy               # Secuencias de prueba
│       ├── y_train.npy              # Etiquetas de entrenamiento
│       ├── y_test.npy               # Etiquetas de prueba
│       ├── norm_mean.npy            # Media para normalización
│       └── norm_std.npy             # Desviación estándar
│
├── 🧠 MODELOS
│   ├── safeguard_model_lstm/
│   │   ├── modelo_lstm.h5           # Modelo LSTM entrenado
│   │   └── lstm_report.json         # Métricas y configuración
│   │
│   └── safeguard_model_transformer/
│       ├── modelo_transformer.h5    # Modelo Transformer entrenado
│       └── transformer_report.json  # Métricas y configuración
│
├── 📜 SCRIPTS
│   ├── create_sequences.py          # Paso 1: Preprocesamiento
│   ├── train_lstm_detector.py       # Paso 2a: Entrenar LSTM
│   ├── train_transformer_detector.py # Paso 2b: Entrenar Transformer
│   ├── demo_video_lstm.py           # Paso 3a: Demo LSTM
│   └── demo_video_transformer.py    # Paso 3b: Demo Transformer
│
└── 📈 VISUALIZACIONES
    └── model_comparison_charts/     # Gráficos comparativos
```

### Pipeline de Procesamiento

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE COMPLETO                               │
└─────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │   Video/     │
     │   Imagen     │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │  BlazePose   │  ← Extrae 33 keypoints (x, y, z, visibility)
     │  (MediaPipe) │     = 132 features por frame
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │   CSV con    │  ← keypoints_HEAVY.csv
     │  Keypoints   │     16,930 filas (frames)
     └──────┬───────┘
            │
            ▼
┌───────────────────────┐
│  create_sequences.py  │  ← Agrupa en secuencias de 30 frames
│                       │     Añade velocidad y aceleración
│  Entrada: CSV         │     = 396 features por frame
│  Salida: .npy files   │
└───────────┬───────────┘
            │
            ▼
     ┌──────────────┐
     │  Secuencias  │  ← Shape: (N, 30, 396)
     │   .npy       │     N = número de secuencias
     └──────┬───────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌───────────┐ ┌───────────┐
│   LSTM    │ │Transformer│
│  Training │ │  Training │
└─────┬─────┘ └─────┬─────┘
      │             │
      ▼             ▼
┌───────────┐ ┌───────────┐
│  modelo_  │ │  modelo_  │
│  lstm.h5  │ │transf.h5  │
└─────┬─────┘ └─────┬─────┘
      │             │
      └──────┬──────┘
             │
             ▼
      ┌──────────────┐
      │  Demo en     │  ← Detección en tiempo real
      │  Tiempo Real │
      └──────────────┘
```

---

## 2. Script 1: create_sequences.py

### Propósito
Convertir frames individuales en **secuencias temporales** para que los modelos puedan detectar **transiciones** (movimiento de caída).

### Código Explicado por Secciones

#### 2.1 Configuración Inicial

```python
# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

INPUT_CSV = r"G:\Mi unidad\safeguard_keypoints_HEAVY\keypoints_HEAVY.csv"
OUTPUT_FOLDER = r"G:\Mi unidad\safeguard_sequences"

SEQUENCE_LENGTH = 30      # Número de frames por secuencia
SEQUENCE_STRIDE = 15      # Desplazamiento entre secuencias (50% overlap)
TEST_SIZE = 0.2           # 20% para test
RANDOM_STATE = 42         # Semilla para reproducibilidad
```

**Explicación:**
- `SEQUENCE_LENGTH = 30`: Cada secuencia contiene 30 frames consecutivos (~1 segundo a 30fps)
- `SEQUENCE_STRIDE = 15`: Las secuencias se superponen 50% para aumentar datos
- `TEST_SIZE = 0.2`: 80% entrenamiento, 20% prueba

```
Visualización del overlap:

Frames:    1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 ...
           ├──────────────Secuencia 1───────────────┤
                             ├──────────────Secuencia 2───────────────┤
                                                ├──────────────Secuencia 3──...

Con STRIDE=15 y LENGTH=30, cada secuencia comparte 15 frames con la siguiente.
Esto aumenta el número de muestras de entrenamiento.
```

#### 2.2 Carga de Datos

```python
def load_and_prepare_data(csv_path):
    """
    Carga el CSV y prepara los datos.
    """
    print(f"📂 Cargando: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Identificar columnas de keypoints (x, y, z, visibility para 33 puntos)
    keypoint_cols = [col for col in df.columns if col.startswith('kp_')]
    
    # Columnas importantes
    # - video_id: Identificador del video de origen
    # - frame_number: Número de frame dentro del video
    # - label: 0 = ADL, 1 = Caída
    
    print(f"   Total frames: {len(df)}")
    print(f"   Features: {len(keypoint_cols)}")  # Debería ser 132 (33 × 4)
    print(f"   Videos únicos: {df['video_id'].nunique()}")
    
    return df, keypoint_cols
```

**Explicación:**
- Lee el CSV con todos los keypoints extraídos por BlazePose
- Identifica las columnas de keypoints (132 columnas: 33 puntos × 4 valores)
- Preserva `video_id` para no mezclar frames de diferentes videos

#### 2.3 Creación de Secuencias

```python
def create_sequences_from_video(video_frames, keypoint_cols, seq_length, stride):
    """
    Crea secuencias a partir de los frames de un video.
    
    Args:
        video_frames: DataFrame con frames de UN solo video
        keypoint_cols: Lista de columnas de keypoints
        seq_length: Longitud de cada secuencia (30)
        stride: Desplazamiento entre secuencias (15)
    
    Returns:
        sequences: Lista de arrays (seq_length, n_features)
        labels: Lista de etiquetas (0 o 1)
    """
    sequences = []
    labels = []
    
    # Ordenar por número de frame
    video_frames = video_frames.sort_values('frame_number')
    
    # Extraer keypoints como array numpy
    keypoints = video_frames[keypoint_cols].values  # Shape: (n_frames, 132)
    frame_labels = video_frames['label'].values     # Shape: (n_frames,)
    
    # Crear secuencias con ventana deslizante
    for start_idx in range(0, len(keypoints) - seq_length + 1, stride):
        end_idx = start_idx + seq_length
        
        # Extraer secuencia de keypoints
        sequence = keypoints[start_idx:end_idx]  # Shape: (30, 132)
        
        # Determinar etiqueta de la secuencia
        seq_labels = frame_labels[start_idx:end_idx]
        label = determine_sequence_label(seq_labels)
        
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels
```

**Explicación Visual:**

```
Video con 100 frames:

Frame:  0    15   30   45   60   75   90   100
        │    │    │    │    │    │    │    │
        ├────┴────┤                          Secuencia 1 (frames 0-29)
             ├────┴────┤                     Secuencia 2 (frames 15-44)
                  ├────┴────┤                Secuencia 3 (frames 30-59)
                       ├────┴────┤           Secuencia 4 (frames 45-74)
                            ├────┴────┤      Secuencia 5 (frames 60-89)
                                 ├────┴─┤    Secuencia 6 (frames 75-100) ✗ muy corta

Resultado: 5 secuencias de 30 frames cada una
```

#### 2.4 Determinación de Etiquetas

```python
def determine_sequence_label(seq_labels):
    """
    Determina la etiqueta de una secuencia completa.
    
    Lógica:
    1. Si detectamos TRANSICIÓN (empieza normal, termina en caída) → Caída
    2. Si mayoría es caída (>80%) → Caída
    3. Si mayoría es ADL (<20% caídas) → ADL
    4. Caso mixto → Usar etiqueta del último frame
    """
    fall_ratio = np.mean(seq_labels)  # Proporción de frames con caída
    
    # Primera mitad vs segunda mitad
    first_half = seq_labels[:len(seq_labels)//2]
    second_half = seq_labels[len(seq_labels)//2:]
    
    first_half_fall_ratio = np.mean(first_half)
    second_half_fall_ratio = np.mean(second_half)
    
    # CASO 1: Detectar TRANSICIÓN (el caso más importante)
    # Primera mitad mayormente ADL, segunda mitad mayormente caída
    if first_half_fall_ratio < 0.3 and second_half_fall_ratio > 0.7:
        return 1  # ¡TRANSICIÓN DETECTADA! → Caída
    
    # CASO 2: Mayoría caída (ya en el suelo)
    if fall_ratio > 0.8:
        return 1  # Caída
    
    # CASO 3: Mayoría ADL
    if fall_ratio < 0.2:
        return 0  # ADL
    
    # CASO 4: Mixto - usar último frame
    return seq_labels[-1]
```

**Explicación Visual de Casos:**

```
CASO 1: TRANSICIÓN (lo que queremos detectar)
Frames:  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
          ├─────── Primera mitad ───────┤├────── Segunda mitad ──────┤
          ADL (90%)                      Caída (100%)
          → Etiqueta: 1 (CAÍDA - transición detectada)

CASO 2: MAYORMENTE CAÍDA (persona ya en el suelo)
Frames:  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
          fall_ratio = 100%
          → Etiqueta: 1 (CAÍDA)

CASO 3: MAYORMENTE ADL (persona normal)
Frames:  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
          fall_ratio = 0%
          → Etiqueta: 0 (ADL)

CASO 4: MIXTO (ambiguo)
Frames:  [0,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0]
          fall_ratio = 27%
          → Etiqueta: 0 (último frame)
```

#### 2.5 Añadir Features Temporales

```python
def add_temporal_features(sequences):
    """
    Añade velocidad y aceleración a las secuencias.
    
    Entrada: (N, 30, 132) - posición solamente
    Salida:  (N, 30, 396) - posición + velocidad + aceleración
    
    Esto TRIPLICA los features pero añade información CRUCIAL
    sobre el MOVIMIENTO.
    """
    enhanced_sequences = []
    
    for sequence in sequences:
        # sequence shape: (30, 132)
        
        # VELOCIDAD = cambio de posición entre frames
        # v[t] = pos[t] - pos[t-1]
        velocity = np.zeros_like(sequence)
        velocity[1:] = sequence[1:] - sequence[:-1]
        
        # ACELERACIÓN = cambio de velocidad entre frames
        # a[t] = v[t] - v[t-1]
        acceleration = np.zeros_like(sequence)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        
        # Concatenar: [posición | velocidad | aceleración]
        enhanced = np.concatenate([sequence, velocity, acceleration], axis=1)
        # enhanced shape: (30, 396)
        
        enhanced_sequences.append(enhanced)
    
    return np.array(enhanced_sequences)
```

**Explicación Física:**

```
¿Por qué velocidad y aceleración?

POSICIÓN sola:
- Frame 1: Persona parada (pose vertical)
- Frame 30: Persona en el suelo (pose horizontal)
- Problema: No sabemos si CAYÓ o si ya estaba acostada

Con VELOCIDAD:
- Si velocidad ≈ 0: La persona estaba quieta (ya acostada)
- Si velocidad alta hacia abajo: La persona está CAYENDO

Con ACELERACIÓN:
- Aceleración constante hacia abajo ≈ gravedad → CAÍDA LIBRE
- Aceleración ≈ 0: Movimiento controlado o quietud

Ejemplo numérico (eje Y de la cadera):

Persona acostada (no caída):
  Frame:    1     2     3    ...   30
  Pos Y:   0.8   0.8   0.8   ...  0.8
  Vel Y:   0.0   0.0   0.0   ...  0.0  ← Sin movimiento
  Acc Y:   0.0   0.0   0.0   ...  0.0

Persona cayendo:
  Frame:    1     2     3    ...   30
  Pos Y:   0.3   0.35  0.42  ...  0.9
  Vel Y:   0.0   0.05  0.07  ...  0.1  ← Velocidad aumentando
  Acc Y:   0.0   0.05  0.02  ...  0.0  ← Aceleración (gravedad)
```

#### 2.6 División Train/Test

```python
def split_by_video(df, sequences, labels, video_ids, test_size=0.2):
    """
    Divide los datos por VIDEO, no por secuencia.
    
    ¿Por qué? Para evitar DATA LEAKAGE.
    
    Si dividimos por secuencia, secuencias del mismo video
    podrían estar en train Y test, haciendo que el modelo
    "memorice" videos en lugar de aprender patrones generales.
    """
    unique_videos = np.unique(video_ids)
    
    # Dividir videos, no secuencias
    train_videos, test_videos = train_test_split(
        unique_videos, 
        test_size=test_size, 
        random_state=RANDOM_STATE
    )
    
    # Asignar secuencias según su video de origen
    train_mask = np.isin(video_ids, train_videos)
    test_mask = np.isin(video_ids, test_videos)
    
    X_train = sequences[train_mask]
    X_test = sequences[test_mask]
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    
    return X_train, X_test, y_train, y_test
```

**Explicación Visual:**

```
INCORRECTO (data leakage):
┌─────────────────────────────────────────┐
│              Video A                     │
│  Sec1  Sec2  Sec3  Sec4  Sec5  Sec6     │
│   ↓     ↓     ↓     ↓     ↓     ↓       │
│ Train Test Train Test Train Test        │  ← Frames similares en ambos!
└─────────────────────────────────────────┘

CORRECTO (split por video):
┌─────────────────────────────────────────┐
│              Video A                     │
│  Sec1  Sec2  Sec3  Sec4  Sec5  Sec6     │
│   ↓     ↓     ↓     ↓     ↓     ↓       │
│ Train Train Train Train Train Train     │  ← Todo el video en train
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│              Video B                     │
│  Sec1  Sec2  Sec3  Sec4  Sec5  Sec6     │
│   ↓     ↓     ↓     ↓     ↓     ↓       │
│ Test  Test  Test  Test  Test  Test      │  ← Todo el video en test
└─────────────────────────────────────────┘
```

---

## 3. Script 2: train_lstm_detector.py

### Propósito
Entrenar una red neuronal **LSTM Bidireccional** para clasificar secuencias como "Caída" o "No Caída".

### Código Explicado por Secciones

#### 3.1 Arquitectura del Modelo

```python
def build_lstm_model(input_shape, config):
    """
    Construye el modelo LSTM.
    
    Args:
        input_shape: (sequence_length, n_features) = (30, 396)
        config: Diccionario con hiperparámetros
    
    Returns:
        model: Modelo Keras compilado
    """
    model = Sequential()
    
    # ═══════════════════════════════════════════════════════════════════
    # CAPA 1: LSTM Bidireccional (128 unidades)
    # ═══════════════════════════════════════════════════════════════════
    if config["bidirectional"]:
        model.add(Bidirectional(
            LSTM(config["lstm_units_1"], return_sequences=True),
            input_shape=input_shape
        ))
    else:
        model.add(LSTM(
            config["lstm_units_1"], 
            return_sequences=True,
            input_shape=input_shape
        ))
    
    model.add(BatchNormalization())  # Normaliza activaciones
    model.add(Dropout(config["dropout_rate"]))  # Previene overfitting
    
    # ═══════════════════════════════════════════════════════════════════
    # CAPA 2: LSTM Bidireccional (64 unidades)
    # ═══════════════════════════════════════════════════════════════════
    if config["bidirectional"]:
        model.add(Bidirectional(LSTM(config["lstm_units_2"])))
    else:
        model.add(LSTM(config["lstm_units_2"]))
    
    model.add(BatchNormalization())
    model.add(Dropout(config["dropout_rate"]))
    
    # ═══════════════════════════════════════════════════════════════════
    # CAPA 3: Dense (32 unidades)
    # ═══════════════════════════════════════════════════════════════════
    model.add(Dense(config["dense_units"], activation='relu'))
    model.add(Dropout(config["dropout_rate"]))
    
    # ═══════════════════════════════════════════════════════════════════
    # CAPA 4: Salida (1 neurona con sigmoid)
    # ═══════════════════════════════════════════════════════════════════
    model.add(Dense(1, activation='sigmoid'))
    
    # Compilar
    optimizer = Adam(learning_rate=config["learning_rate"])
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model
```

**Explicación de Cada Componente:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA LSTM                                    │
└─────────────────────────────────────────────────────────────────────────┘

Input: (batch_size, 30 frames, 396 features)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL LSTM (128 unidades)                          │
│                                                                         │
│   Forward LSTM:   Frame 1 → Frame 2 → ... → Frame 30                   │
│                   Aprende patrones de IZQUIERDA a DERECHA              │
│                                                                         │
│   Backward LSTM:  Frame 30 → Frame 29 → ... → Frame 1                  │
│                   Aprende patrones de DERECHA a IZQUIERDA              │
│                                                                         │
│   Output: Concatenación de ambos → 256 unidades (128 × 2)              │
│   return_sequences=True → Devuelve salida para cada frame              │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              BATCH NORMALIZATION                                        │
│                                                                         │
│   Normaliza las activaciones para estabilizar el entrenamiento         │
│   Acelera convergencia y reduce dependencia de inicialización          │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DROPOUT (30%)                                              │
│                                                                         │
│   "Apaga" aleatoriamente 30% de las neuronas en cada batch             │
│   Previene overfitting al forzar redundancia                           │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              BIDIRECTIONAL LSTM (64 unidades)                           │
│                                                                         │
│   Segunda capa LSTM, más pequeña                                        │
│   return_sequences=False → Solo devuelve salida del último frame       │
│   Output: 128 unidades (64 × 2)                                        │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DENSE (32 unidades, ReLU)                                  │
│                                                                         │
│   Capa fully-connected para combinar features                          │
│   ReLU: f(x) = max(0, x) - introduce no-linealidad                     │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DENSE (1 unidad, Sigmoid)                                  │
│                                                                         │
│   Neurona de salida                                                     │
│   Sigmoid: f(x) = 1/(1+e^(-x)) → Output entre 0 y 1                    │
│   Interpretación: Probabilidad de que sea una caída                    │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
Output: Probabilidad de caída (0.0 a 1.0)
```

#### 3.2 Callbacks de Entrenamiento

```python
callbacks = [
    # ═══════════════════════════════════════════════════════════════════
    # EARLY STOPPING
    # ═══════════════════════════════════════════════════════════════════
    EarlyStopping(
        monitor='val_recall',        # Monitorea Recall en validación
        patience=15,                 # Espera 15 épocas sin mejora
        mode='max',                  # Queremos MAXIMIZAR recall
        restore_best_weights=True,   # Restaura el mejor modelo
        verbose=1
    ),
    
    # ═══════════════════════════════════════════════════════════════════
    # REDUCE LEARNING RATE
    # ═══════════════════════════════════════════════════════════════════
    ReduceLROnPlateau(
        monitor='val_loss',          # Monitorea loss en validación
        factor=0.5,                  # Reduce LR a la mitad
        patience=5,                  # Espera 5 épocas sin mejora
        min_lr=1e-6,                 # LR mínimo
        verbose=1
    ),
    
    # ═══════════════════════════════════════════════════════════════════
    # MODEL CHECKPOINT
    # ═══════════════════════════════════════════════════════════════════
    ModelCheckpoint(
        os.path.join(output_folder, "best_model.h5"),
        monitor='val_recall',        # Guarda cuando mejora recall
        mode='max',
        save_best_only=True,         # Solo guarda el mejor
        verbose=1
    )
]
```

**Explicación Visual:**

```
EARLY STOPPING - Previene Overfitting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

val_recall
    │
1.0 │                    ●──●──●──●──●  ← Mejor modelo (se guarda)
    │                 ●/
    │              ●/
0.9 │           ●/
    │        ●/
    │     ●/
0.8 │  ●/                     ← 15 épocas sin mejora → STOP
    │●
    └────────────────────────────────────────────────────── época
    1  5  10  15  20  25  30  35  40  45
                              │
                              └── Early stopping aquí


REDUCE LR ON PLATEAU - Ajusta Learning Rate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

val_loss
    │
0.5 │●
    │ ●
    │  ●
0.3 │   ●                          
    │    ●──●──●──●──●  ← Estancado 5 épocas
    │                 ●  ← LR × 0.5, mejora de nuevo
0.1 │                  ●
    │                   ●●●
    └────────────────────────────────────────────────────── época
```

#### 3.3 Class Weights

```python
def calculate_class_weights(y_train):
    """
    Calcula pesos para manejar desbalance de clases.
    
    Si hay más ejemplos de ADL que de Caída, el modelo
    tiende a predecir siempre "ADL". Los pesos compensan esto.
    """
    n_samples = len(y_train)
    n_positive = sum(y_train)      # Número de caídas
    n_negative = n_samples - n_positive  # Número de ADL
    
    # Peso inversamente proporcional a la frecuencia
    weight_positive = n_samples / (2 * n_positive)
    weight_negative = n_samples / (2 * n_negative)
    
    class_weights = {0: weight_negative, 1: weight_positive}
    
    return class_weights

# Ejemplo numérico:
# n_samples = 100
# n_positive = 20 (caídas)
# n_negative = 80 (ADL)
#
# weight_positive = 100 / (2 × 20) = 2.5
# weight_negative = 100 / (2 × 80) = 0.625
#
# Interpretación: Cada ejemplo de caída "vale" 4× más que uno de ADL
```

---

## 4. Script 3: train_transformer_detector.py

### Propósito
Entrenar un **Transformer** que usa **Self-Attention** para analizar toda la secuencia a la vez.

### Código Explicado por Secciones

#### 4.1 Positional Encoding

```python
class PositionalEncoding(layers.Layer):
    """
    Añade información de POSICIÓN a la secuencia.
    
    ¿Por qué es necesario?
    El Transformer procesa todos los frames EN PARALELO.
    Sin positional encoding, no sabría que Frame 1 viene ANTES de Frame 30.
    """
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
        # Crear encodings posicionales usando senos y cosenos
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe = np.zeros((sequence_length, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term[:embed_dim//2 + embed_dim%2])
        pe[:, 1::2] = np.cos(position * div_term[:embed_dim//2])
        
        self.positional_encoding = tf.constant(pe, dtype=tf.float32)
    
    def call(self, x):
        # Suma el encoding posicional a la entrada
        return x + self.positional_encoding[:tf.shape(x)[1], :tf.shape(x)[2]]
```

**Explicación Visual:**

```
Sin Positional Encoding:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  [Frame1] [Frame2] [Frame3] ... [Frame30]
            │        │        │            │
            ▼        ▼        ▼            ▼
        ┌────────────────────────────────────┐
        │         TRANSFORMER                │  ← No sabe el ORDEN
        └────────────────────────────────────┘

Con Positional Encoding:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:  [Frame1] [Frame2] [Frame3] ... [Frame30]
            +        +        +            +
        [PE_1]   [PE_2]   [PE_3]  ... [PE_30]  ← Posición codificada
            │        │        │            │
            ▼        ▼        ▼            ▼
        ┌────────────────────────────────────┐
        │         TRANSFORMER                │  ← Ahora sabe el ORDEN
        └────────────────────────────────────┘

Donde PE_i usa funciones sin/cos de diferente frecuencia:
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

#### 4.2 Transformer Block

```python
class TransformerBlock(layers.Layer):
    """
    Un bloque del Transformer Encoder.
    
    Componentes:
    1. Multi-Head Self-Attention
    2. Feed-Forward Network
    3. Layer Normalization
    4. Residual Connections
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        
        # Multi-Head Attention
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,           # 8 cabezas
            key_dim=embed_dim // num_heads  # 64 dimensiones por cabeza
        )
        
        # Feed-Forward Network
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),  # Expansión
            layers.Dense(embed_dim),                   # Contracción
        ])
        
        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        # ════════════════════════════════════════════════════════════
        # PARTE 1: Self-Attention con Residual Connection
        # ════════════════════════════════════════════════════════════
        attn_output = self.att(inputs, inputs)  # Query=Key=Value=inputs
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual + Norm
        
        # ════════════════════════════════════════════════════════════
        # PARTE 2: Feed-Forward con Residual Connection
        # ════════════════════════════════════════════════════════════
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Residual + Norm
```

**Explicación Visual de Self-Attention:**

```
SELF-ATTENTION: Cada frame "mira" a TODOS los demás
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Frame 1 ──────┬──────┬──────┬──────┬─────► Weighted combination
              │      │      │      │
Frame 2 ◄─────┼──────┼──────┼──────┤
              │      │      │      │
Frame 3 ◄─────┼──────┼──────┼──────┤       Q × K^T / √d = Attention
              │      │      │      │
  ...         │      │      │      │       Attention × V = Output
              │      │      │      │
Frame30 ◄─────┴──────┴──────┴──────┘


MULTI-HEAD: 8 tipos diferentes de "atención"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Head 1: "¿Qué frames tienen pose similar?"
Head 2: "¿Qué frames tienen velocidad alta?"
Head 3: "¿Qué frames están cerca temporalmente?"
Head 4: "¿Qué frames muestran transición?"
...
Head 8: "¿Qué frames son el inicio/fin?"

Cada cabeza aprende a enfocarse en DIFERENTES relaciones.
Se concatenan al final: 8 × 64 = 512 dimensiones.
```

#### 4.3 Modelo Completo

```python
def build_transformer_model(input_shape, config):
    """
    Construye el modelo Transformer completo.
    """
    sequence_length, n_features = input_shape  # (30, 396)
    
    # Input
    inputs = layers.Input(shape=input_shape)
    
    # Proyección a dimensión del modelo
    embed_dim = config["num_heads"] * config["head_size"]  # 8 × 64 = 512
    x = layers.Dense(embed_dim)(inputs)
    
    # Positional Encoding
    x = PositionalEncoding(sequence_length, embed_dim)(x)
    
    # Stack de Transformer Blocks
    for i in range(config["num_transformer_blocks"]):  # 3 bloques
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=config["num_heads"],
            ff_dim=config["ff_dim"],
            dropout_rate=config["dropout_rate"]
        )(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # MLP Head
    for units in config["mlp_units"]:  # [128, 64]
        x = layers.Dense(units, activation="gelu")(x)
        x = layers.Dropout(config["mlp_dropout"])(x)
    
    # Output
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs, outputs)
    
    return model
```

**Arquitectura Visual:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA TRANSFORMER                             │
└─────────────────────────────────────────────────────────────────────────┘

Input: (batch, 30 frames, 396 features)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              PROYECCIÓN LINEAL                                          │
│              Dense(512)                                                 │
│              (30, 396) → (30, 512)                                      │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              POSITIONAL ENCODING                                        │
│              Añade información de posición temporal                     │
│              (30, 512) → (30, 512)                                      │
└─────────────────────────────────────────────────────────────────────────┘
                    │
          ┌────────┴────────┐
          │                 │
          ▼                 │
┌─────────────────────┐     │
│  Transformer Block  │     │
│  ┌───────────────┐  │     │
│  │ Multi-Head    │  │     │
│  │ Attention     │  │     │  × 3 bloques
│  │ (8 heads)     │  │     │
│  └───────┬───────┘  │     │
│          ▼          │     │
│  ┌───────────────┐  │     │
│  │ Feed Forward  │  │     │
│  │ (256 → 512)   │  │     │
│  └───────────────┘  │     │
└─────────────────────┘     │
          │                 │
          └────────┬────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              GLOBAL AVERAGE POOLING                                     │
│              Promedio sobre todos los frames                            │
│              (30, 512) → (512,)                                         │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              MLP HEAD                                                   │
│              Dense(128) → GELU → Dropout                                │
│              Dense(64) → GELU → Dropout                                 │
│              Dense(1) → Sigmoid                                         │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
Output: Probabilidad de caída (0.0 - 1.0)
```

---

## 5. Script 4: demo_video_lstm.py

### Propósito
Ejecutar detección de caídas en **tiempo real** usando el modelo entrenado.

### Código Explicado por Secciones

#### 5.1 Clase Principal del Detector

```python
class LSTMFallDetector:
    """
    Detector de caídas en tiempo real usando LSTM.
    """
    
    def __init__(self, model_folder, sequences_folder, sequence_length=30):
        # Buffer circular para almacenar los últimos 30 frames
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Historial de predicciones para confirmación
        self.fall_history = deque(maxlen=CONFIG['confirmation_frames'])
        
        # Cargar modelo y normalización
        self._load_model()
        self._load_normalization()
        self._init_blazepose()
```

**Explicación del Buffer:**

```
BUFFER CIRCULAR (deque con maxlen=30)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tiempo t=1:  [F1]
Tiempo t=2:  [F1, F2]
Tiempo t=3:  [F1, F2, F3]
...
Tiempo t=30: [F1, F2, F3, ..., F30]  ← Buffer lleno, primera predicción
Tiempo t=31: [F2, F3, F4, ..., F31]  ← F1 sale, F31 entra (ventana deslizante)
Tiempo t=32: [F3, F4, F5, ..., F32]  ← F2 sale, F32 entra
...

El buffer siempre tiene los 30 frames más recientes.
Esto permite detección CONTINUA en tiempo real.
```

#### 5.2 Procesamiento de Frame

```python
def process_frame(self, frame):
    """
    Procesa un frame y determina si hay caída.
    """
    # 1. Extraer keypoints con BlazePose
    keypoints, landmarks = self.extract_keypoints(frame)
    
    if keypoints is None:
        return {"state": "no_person", ...}
    
    # 2. Añadir al buffer
    self.frame_buffer.append(keypoints)
    
    # 3. Si buffer no está lleno, seguir buffering
    if len(self.frame_buffer) < self.sequence_length:
        return {"state": "buffering", ...}
    
    # 4. Predecir con el modelo
    prediction, probability = self.predict_sequence()
    
    # 5. Confirmación (evitar falsas alarmas por 1 frame)
    self.fall_history.append(prediction)
    recent_falls = sum(self.fall_history)
    
    if recent_falls >= CONFIG['confirmation_frames']:
        return {"state": "fall_confirmed", ...}
    elif prediction == 1:
        return {"state": "fall_possible", ...}
    else:
        return {"state": "normal", ...}
```

**Explicación de la Confirmación:**

```
SISTEMA DE CONFIRMACIÓN (evita falsas alarmas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

confirmation_frames = 2

Predicciones:  [0, 0, 0, 0, 1, 1, 1, 1, 1, ...]
                              │  │
                              │  └── Segunda predicción de caída
                              └───── Primera predicción de caída
                              
fall_history:  [0]     → Estado: normal
               [0, 1]  → Estado: fall_possible (1 predicción)
               [1, 1]  → Estado: fall_confirmed ✓ (2 consecutivas)
               
¿Por qué?
- Un solo frame con predicción "caída" podría ser ruido
- Requerir 2 predicciones consecutivas aumenta confiabilidad
- Trade-off: +0.07 segundos de latencia vs menos falsas alarmas
```

#### 5.3 Manejo de Pausa (Corrección Importante)

```python
# ════════════════════════════════════════════════════════════════════
# CORRECCIÓN: Variables para mantener estado durante pausa
# ════════════════════════════════════════════════════════════════════

current_frame = None    # Frame actual
current_result = None   # Resultado del procesamiento
display_frame = None    # Frame CON keypoints dibujados

while True:
    # Solo procesar si NO está pausado
    if not paused:
        ret, frame = cap.read()
        
        current_frame = cv2.resize(frame, ...)
        current_result = detector.process_frame(current_frame)
        
        # Crear frame de visualización (con keypoints)
        display_frame = current_frame.copy()
        display_frame = visualizer.draw_skeleton(display_frame, ...)
    
    # Siempre mostrar el frame guardado (no reprocesar)
    if display_frame is not None:
        show_frame = display_frame.copy()
        show_frame = visualizer.draw_status_panel(show_frame, ...)
        cv2.imshow("SafeGuard", show_frame)
```

**Por qué esta corrección era necesaria:**

```
ANTES (bug):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
while True:
    if not paused:
        ret, frame = cap.read()
    
    # ⚠️ ESTO SE EJECUTABA SIEMPRE, INCLUSO PAUSADO
    result = detector.process_frame(frame)  # Reprocesa
    frame = draw_skeleton(frame, result)    # Redibuja
    
Problema: BlazePose tiene variabilidad entre ejecuciones.
         Al reprocesar el mismo frame, los keypoints
         "parpadean" porque son ligeramente diferentes.


DESPUÉS (corregido):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
while True:
    if not paused:
        frame = cap.read()
        result = detector.process_frame(frame)
        display_frame = draw_skeleton(frame, result)  # Solo aquí
    
    # Muestra el frame GUARDADO, no reprocesa
    cv2.imshow(display_frame)
    
Resultado: Keypoints estáticos durante pausa.
```

---

## 6. Componentes Comunes

### 6.1 BlazePose (MediaPipe)

```python
def _init_blazepose(self):
    """
    Inicializa el detector de poses BlazePose.
    
    BlazePose detecta 33 keypoints del cuerpo humano:
    - Cara: 11 puntos (ojos, nariz, orejas, boca)
    - Torso: 4 puntos (hombros, caderas)
    - Brazos: 6 puntos (codos, muñecas)
    - Piernas: 8 puntos (rodillas, tobillos)
    - Manos: 4 puntos (pulgares, índices)
    """
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
    
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,                      # Detectar 1 persona
        min_pose_detection_confidence=0.5, # Umbral de detección
        min_pose_presence_confidence=0.5,  # Umbral de presencia
        min_tracking_confidence=0.5        # Umbral de tracking
    )
    
    self.pose_detector = vision.PoseLandmarker.create_from_options(options)
```

**Los 33 Keypoints de BlazePose:**

```
                    0 (nariz)
                       │
            ┌────2─────┼─────1────┐
           (ojo L)     │      (ojo R)
        ┌──4──┐    ┌───3───┐    ┌──5──┐
      (oreja L)   (boca)      (oreja R)
            │                     │
           11 ─────────────────── 12
        (hombro L)            (hombro R)
            │                     │
           13                    14
        (codo L)              (codo R)
            │                     │
           15                    16
        (muñeca L)            (muñeca R)
            │                     │
        17-22                  17-22
        (mano L)              (mano R)
            
           23 ─────────────────── 24
        (cadera L)            (cadera R)
            │                     │
           25                    26
        (rodilla L)           (rodilla R)
            │                     │
           27                    28
        (tobillo L)           (tobillo R)
            │                     │
        29-32                  29-32
        (pie L)               (pie R)
```

### 6.2 Normalización

```python
def normalize_sequence(self, sequence):
    """
    Normaliza la secuencia usando media y desviación estándar
    calculadas en el entrenamiento.
    
    Fórmula: X_norm = (X - mean) / std
    
    ¿Por qué normalizar?
    1. Los keypoints están en rango [0, 1] (coordenadas normalizadas)
    2. La velocidad puede ser muy pequeña (~0.01 por frame)
    3. Sin normalización, el modelo daría más peso a valores grandes
    """
    if self.norm_mean is not None:
        sequence = (sequence - self.norm_mean) / (self.norm_std + 1e-8)
        # +1e-8 evita división por cero
    
    return sequence
```

---

## 7. Flujo de Datos Completo

### Visualización End-to-End

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FLUJO DE DATOS COMPLETO                              │
└─────────────────────────────────────────────────────────────────────────┘

ENTRENAMIENTO (Offline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌───────────┐    ┌───────────┐    ┌───────────────┐    ┌───────────────┐
│  Videos   │───►│ BlazePose │───►│ keypoints.csv │───►│ Secuencias    │
│  (le2i,   │    │ 33 points │    │ 16,930 frames │    │ (N, 30, 396)  │
│  ur_fall) │    │ × 4 vals  │    │ × 132 features│    │ + vel + acc   │
└───────────┘    └───────────┘    └───────────────┘    └───────┬───────┘
                                                               │
                                         ┌─────────────────────┴─────────┐
                                         │                               │
                                         ▼                               ▼
                                  ┌─────────────┐                ┌─────────────┐
                                  │    LSTM     │                │ Transformer │
                                  │  Training   │                │  Training   │
                                  └──────┬──────┘                └──────┬──────┘
                                         │                               │
                                         ▼                               ▼
                                  ┌─────────────┐                ┌─────────────┐
                                  │modelo_lstm  │                │modelo_trans │
                                  │    .h5      │                │    .h5      │
                                  └─────────────┘                └─────────────┘


INFERENCIA (Real-time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌───────────┐
│  Webcam/  │
│   Video   │
└─────┬─────┘
      │ Frame (1280×720)
      ▼
┌─────────────┐
│  BlazePose  │ ~30ms
└─────┬───────┘
      │ 132 features
      ▼
┌─────────────┐
│   Buffer    │
│ (30 frames) │
└─────┬───────┘
      │ (30, 132)
      ▼
┌─────────────────┐
│ Add Temporal    │
│ vel + acc       │
└─────┬───────────┘
      │ (30, 396)
      ▼
┌─────────────────┐
│  Normalize      │
└─────┬───────────┘
      │ (1, 30, 396)
      ▼
┌─────────────────┐
│   LSTM/Trans    │ ~5ms
│   Predict       │
└─────┬───────────┘
      │ Probability (0.0-1.0)
      ▼
┌─────────────────┐
│  Confirmation   │
│  (2 frames)     │
└─────┬───────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│                    OUTPUT                        │
│                                                 │
│   Estado: NORMAL / FALL_POSSIBLE / FALL_CONFIRMED │
│   Probabilidad: 0.87                            │
│   Latencia: ~35ms (~28 FPS)                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 📌 Resumen de Archivos

| Script | Input | Output | Tiempo |
|--------|-------|--------|--------|
| `create_sequences.py` | CSV (16,930 frames) | .npy (secuencias) | ~2 min |
| `train_lstm_detector.py` | .npy secuencias | modelo_lstm.h5 | ~15-20 min |
| `train_transformer_detector.py` | .npy secuencias | modelo_transformer.h5 | ~20-30 min |
| `demo_video_lstm.py` | Video + modelo | Detección tiempo real | Real-time |
| `demo_video_transformer.py` | Video + modelo | Detección tiempo real | Real-time |

---

*Documento preparado para MIT Global Teaching Labs 2025*

*SafeGuard Vision AI - Industry 4.0 Zero Accident Initiative*
