# 🧮 Fundamentos Matemáticos de los Modelos

## SafeGuard Vision AI | MIT Global Teaching Labs 2025

### La Matemática Detrás de Random Forest, LSTM y Transformer

---

## 📑 Tabla de Contenidos

1. [Introducción: ¿Por Qué Matemáticas?](#1-introducción-por-qué-matemáticas)
2. [Random Forest: Ensemble de Árboles](#2-random-forest-ensemble-de-árboles)
3. [LSTM: Long Short-Term Memory](#3-lstm-long-short-term-memory)
4. [Transformer: Self-Attention](#4-transformer-self-attention)
5. [Comparación Matemática](#5-comparación-matemática)
6. [Funciones de Activación](#6-funciones-de-activación)
7. [Función de Pérdida: Binary Cross-Entropy](#7-función-de-pérdida-binary-cross-entropy)
8. [Optimización: Adam](#8-optimización-adam)
9. [Preguntas de Entrevista MIT](#9-preguntas-de-entrevista-mit)

---

## 1. Introducción: ¿Por Qué Matemáticas?

### El Problema Formal

Dado un conjunto de datos de entrenamiento:

$$\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), ..., (\mathbf{x}_n, y_n)\}$$

Donde:
- $\mathbf{x}_i \in \mathbb{R}^d$ es un vector de características (features)
- $y_i \in \{0, 1\}$ es la etiqueta (0 = ADL, 1 = Caída)

**Objetivo:** Encontrar una función $f: \mathbb{R}^d \rightarrow [0, 1]$ que minimice el error de predicción.

### En Nuestro Caso Específico

```
Para Random Forest:
    x ∈ ℝ¹³² (132 features de un frame)
    
Para LSTM/Transformer:
    X ∈ ℝ³⁰ˣ³⁹⁶ (30 frames × 396 features)
```

---

## 2. Random Forest: Ensemble de Árboles

### 2.1 Árbol de Decisión: La Unidad Básica

Un árbol de decisión particiona el espacio de features recursivamente.

**Criterio de División (Gini Impurity):**

$$Gini(S) = 1 - \sum_{i=1}^{C} p_i^2$$

Donde:
- $S$ = conjunto de muestras en un nodo
- $C$ = número de clases (2 en nuestro caso)
- $p_i$ = proporción de muestras de clase $i$

**Ejemplo Numérico:**

```
Nodo con 100 muestras:
- 70 ADL (clase 0)
- 30 Caída (clase 1)

p₀ = 70/100 = 0.7
p₁ = 30/100 = 0.3

Gini = 1 - (0.7² + 0.3²)
     = 1 - (0.49 + 0.09)
     = 1 - 0.58
     = 0.42

Interpretación: Gini = 0.42 indica impureza moderada.
Gini = 0 → Nodo puro (todas las muestras de una clase)
Gini = 0.5 → Máxima impureza (50/50)
```

**Ganancia de Información:**

$$Gain(S, A) = Gini(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Gini(S_v)$$

El algoritmo selecciona el atributo $A$ y valor de corte que **maximiza** la ganancia.

### 2.2 De Árbol a Bosque (Forest)

**Random Forest** entrena múltiples árboles con dos tipos de aleatoriedad:

1. **Bagging (Bootstrap Aggregating):**
   - Cada árbol se entrena con una muestra bootstrap (con reemplazo)
   - De $n$ muestras, se seleccionan $n$ con reemplazo

2. **Random Feature Selection:**
   - En cada división, solo se considera un subconjunto de features
   - Típicamente $\sqrt{d}$ features de $d$ totales

**Predicción Final:**

$$\hat{y} = \text{mode}\{h_1(\mathbf{x}), h_2(\mathbf{x}), ..., h_T(\mathbf{x})\}$$

O para probabilidades:

$$P(y=1|\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} h_t(\mathbf{x})$$

Donde $h_t$ es el árbol $t$ y $T$ es el número total de árboles.

### 2.3 Visualización del Proceso

```
RANDOM FOREST CON 3 ÁRBOLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dataset Original (100 muestras):
[1, 2, 3, 4, 5, ..., 100]

Bootstrap Sample 1:        Bootstrap Sample 2:        Bootstrap Sample 3:
[3, 7, 7, 12, 15, ...]    [1, 1, 4, 9, 23, ...]     [5, 8, 8, 11, ...]
        │                         │                         │
        ▼                         ▼                         ▼
   ┌─────────┐               ┌─────────┐               ┌─────────┐
   │ Árbol 1 │               │ Árbol 2 │               │ Árbol 3 │
   └────┬────┘               └────┬────┘               └────┬────┘
        │                         │                         │
    h₁(x)=0.3                 h₂(x)=0.7                 h₃(x)=0.6
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                    P(caída) = (0.3 + 0.7 + 0.6) / 3 = 0.53
                    
                    Si threshold = 0.5: Predicción = CAÍDA
```

### 2.4 Limitación Fundamental para Nuestro Problema

```
Random Forest procesa UN VECTOR a la vez:
    x ∈ ℝ¹³² → f(x) → ŷ ∈ {0, 1}

NO tiene noción de SECUENCIA ni TIEMPO.

Frame t=1: x₁ = [pose parada]     → f(x₁) = 0 (ADL)
Frame t=2: x₂ = [pose cayendo]    → f(x₂) = 0.5 (?)
Frame t=30: x₃₀ = [pose en suelo] → f(x₃₀) = 1 (Caída)

Problema: Frame 30 y "persona durmiendo" tienen la MISMA pose.
          Random Forest NO puede distinguirlos.
```

---

## 3. LSTM: Long Short-Term Memory

### 3.1 El Problema de las RNN Clásicas

Las RNN (Recurrent Neural Networks) procesan secuencias pero sufren de **vanishing gradients**:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial h_T} \cdot \prod_{t=1}^{T-1} \frac{\partial h_{t+1}}{\partial h_t}$$

Si $|\frac{\partial h_{t+1}}{\partial h_t}| < 1$, el producto tiende a 0 para secuencias largas.

### 3.2 Arquitectura LSTM

LSTM resuelve esto con **gates** (compuertas) y un **cell state** separado.

**Componentes:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CELDA LSTM                                      │
└─────────────────────────────────────────────────────────────────────────┘

Entradas:
    xₜ = input en tiempo t
    hₜ₋₁ = hidden state anterior
    cₜ₋₁ = cell state anterior

Salidas:
    hₜ = hidden state actual
    cₜ = cell state actual
```

**Ecuaciones Matemáticas:**

**1. Forget Gate (¿Qué olvidar del pasado?):**
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**2. Input Gate (¿Qué nueva información guardar?):**
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**3. Candidate Cell State (Nueva información propuesta):**
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**4. Cell State Update (Actualizar memoria):**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**5. Output Gate (¿Qué parte de la memoria mostrar?):**
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**6. Hidden State (Salida final):**
$$h_t = o_t \odot \tanh(c_t)$$

Donde:
- $\sigma$ = función sigmoid: $\sigma(x) = \frac{1}{1+e^{-x}}$
- $\odot$ = multiplicación elemento a elemento (Hadamard product)
- $[a, b]$ = concatenación de vectores

### 3.3 Visualización del Flujo

```
                    Cell State (cₜ₋₁ → cₜ)
    ═══════════════════╦═══════════════════════════════════════════════►
                       ║              ┌─────────┐
                  ┌────╨────┐         │         │
                  │    ×    │◄────────┤  Forget │◄──┐
                  └────┬────┘   fₜ    │  Gate σ │   │
                       │              └─────────┘   │
                       │                            │
                       │    ┌─────────┐  ┌───────┐ │
                  ┌────▼────┤         │  │       │ │
                  │    +    │◄──×─────┤ Input   │◄┤
                  └────┬────┘   │     │ Gate σ  │ │
                       │        │     └─────────┘ │
                       │        │                 │
                       │     ┌──┴──┐  ┌─────────┐ │
                       │     │ĉₜ   │◄─┤ Candi-  │◄┤
                       │     │tanh │  │ date    │ │
                       │     └─────┘  │ tanh    │ │
                       │              └─────────┘ │
                       │                          │
    ═══════════════════╬══════════════════════════╬═══════════════════►
                       │                          │
                  ┌────▼────┐  ┌─────────┐       │
                  │  tanh   │  │ Output  │       │
                  └────┬────┘  │ Gate σ  │◄──────┤
                       │       └────┬────┘       │
                  ┌────▼────┐       │            │
                  │    ×    │◄──────┘            │
                  └────┬────┘   oₜ              │
                       │                         │
                       ▼                         │
                      hₜ ───────────────────────►│
                                                 │
                      xₜ ────────────────────────┘
                      hₜ₋₁ ──────────────────────┘
```

### 3.4 Ejemplo Numérico Simplificado

```
Supongamos dimensiones pequeñas para ilustrar:
- Input: xₜ ∈ ℝ² 
- Hidden: hₜ ∈ ℝ³

Concatenación: [hₜ₋₁, xₜ] ∈ ℝ⁵

Matrices de pesos (cada gate):
    Wf, Wi, Wc, Wo ∈ ℝ³ˣ⁵
    bf, bi, bc, bo ∈ ℝ³

Ejemplo en t=1:
    xₜ = [0.5, -0.3]
    hₜ₋₁ = [0.1, 0.2, 0.0]  (inicializado en ceros típicamente)
    
    [hₜ₋₁, xₜ] = [0.1, 0.2, 0.0, 0.5, -0.3]
    
    fₜ = σ(Wf · [0.1, 0.2, 0.0, 0.5, -0.3] + bf)
       = σ([0.8, 0.3, -0.1])  (después de multiplicación)
       = [0.69, 0.57, 0.48]   (después de sigmoid)
    
    Interpretación: 
    - Dimensión 1: retener 69% de la memoria
    - Dimensión 2: retener 57% de la memoria
    - Dimensión 3: retener 48% de la memoria
```

### 3.5 LSTM Bidireccional

```
Bidirectional = Forward LSTM + Backward LSTM

Forward:  Frame 1 → Frame 2 → ... → Frame 30
          h₁→    h₂→         h₃₀→

Backward: Frame 30 → Frame 29 → ... → Frame 1
          ←h₃₀   ←h₂₉         ←h₁

Output en cada tiempo t:
    hₜ = [h→ₜ ; h←ₜ]  (concatenación)

Para LSTM(128) bidireccional:
    Forward: 128 unidades
    Backward: 128 unidades
    Output: 256 unidades por frame
```

### 3.6 ¿Por Qué LSTM Funciona para Caídas?

```
Secuencia de caída:
    Frame 1-10: Persona parada (pose vertical)
    Frame 11-20: Persona cayendo (transición)
    Frame 21-30: Persona en suelo (pose horizontal)

El Cell State RECUERDA:
    c₁₀: "La persona estaba parada"
    c₂₀: "Hubo una transición rápida hacia abajo"
    c₃₀: "Ahora está en el suelo DESPUÉS de caer"

El Forget Gate aprende:
    "No olvidar que estaba parada al inicio"

El Input Gate aprende:
    "Es importante recordar la velocidad de caída"

Resultado: h₃₀ codifica TODA la historia, no solo el estado final.
```

---

## 4. Transformer: Self-Attention

### 4.1 La Intuición de Attention

En lugar de procesar secuencialmente, **Attention** permite que cada elemento "mire" a todos los demás.

**Pregunta fundamental:** "¿Qué partes de la secuencia son relevantes para entender este frame?"

### 4.2 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Donde:
- $Q$ (Query): "¿Qué estoy buscando?"
- $K$ (Key): "¿Qué información tengo disponible?"
- $V$ (Value): "¿Qué información devuelvo?"
- $d_k$ = dimensión de las keys

### 4.3 Desglose Paso a Paso

**Paso 1: Crear Q, K, V**

Para cada frame $x_i$ en la secuencia:
$$q_i = W^Q x_i, \quad k_i = W^K x_i, \quad v_i = W^V x_i$$

**Paso 2: Calcular Attention Scores**

$$e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

Esto mide "cuánto debería el frame $i$ atender al frame $j$".

**Paso 3: Normalizar con Softmax**

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{n} \exp(e_{ik})}$$

Los $\alpha_{ij}$ suman 1 y representan pesos de atención.

**Paso 4: Weighted Sum**

$$z_i = \sum_{j=1}^{n} \alpha_{ij} v_j$$

La salida $z_i$ es una combinación ponderada de todos los valores.

### 4.4 Ejemplo Numérico

```
Secuencia de 3 frames (simplificado):
    x₁ = [1, 0]  (parado)
    x₂ = [0.5, 0.5]  (cayendo)
    x₃ = [0, 1]  (en suelo)

Supongamos d_k = 2 y matrices W identidad:
    Q = K = V = X = [[1, 0], [0.5, 0.5], [0, 1]]

Paso 2: Attention Scores (Q × Kᵀ / √2)

    e = QKᵀ / √2 = [[1·1 + 0·0,    1·0.5 + 0·0.5,   1·0 + 0·1],
                     [0.5·1 + 0.5·0, 0.5·0.5 + 0.5·0.5, 0.5·0 + 0.5·1],
                     [0·1 + 1·0,    0·0.5 + 1·0.5,   0·0 + 1·1]] / √2
    
        = [[1.0,  0.5,  0.0],
           [0.5,  0.5,  0.5],
           [0.0,  0.5,  1.0]] / 1.41
           
        ≈ [[0.71, 0.35, 0.00],
           [0.35, 0.35, 0.35],
           [0.00, 0.35, 0.71]]

Paso 3: Softmax por filas

    α = softmax(e) ≈ [[0.47, 0.33, 0.20],   ← Frame 1 atiende más a sí mismo
                      [0.33, 0.33, 0.33],   ← Frame 2 atiende igual a todos
                      [0.20, 0.33, 0.47]]   ← Frame 3 atiende más a sí mismo

Paso 4: Output (α × V)

    z₁ = 0.47·[1,0] + 0.33·[0.5,0.5] + 0.20·[0,1] = [0.64, 0.37]
    z₂ = 0.33·[1,0] + 0.33·[0.5,0.5] + 0.33·[0,1] = [0.50, 0.50]
    z₃ = 0.20·[1,0] + 0.33·[0.5,0.5] + 0.47·[0,1] = [0.37, 0.64]

Interpretación:
- z₂ (frame de transición) incorpora información de AMBOS extremos
- Esto es exactamente lo que necesitamos para detectar caídas
```

### 4.5 Multi-Head Attention

En lugar de una sola attention, usamos múltiples "cabezas":

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Donde:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**¿Por qué múltiples cabezas?**

```
Head 1: Aprende a enfocarse en POSES similares
Head 2: Aprende a enfocarse en VELOCIDAD
Head 3: Aprende a enfocarse en frames TEMPORALMENTE cercanos
Head 4: Aprende a detectar TRANSICIONES
...
Head 8: Aprende patrones complejos

Cada cabeza "mira" la secuencia de manera diferente.
La concatenación combina todas las perspectivas.
```

### 4.6 Positional Encoding

El Transformer no tiene noción inherente de orden. Añadimos posición:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

```
Ejemplo para secuencia de 30 frames, d=4:

Pos  | PE[:,0]  | PE[:,1]  | PE[:,2]  | PE[:,3]
-----|----------|----------|----------|----------
  0  | sin(0)   | cos(0)   | sin(0)   | cos(0)
     | 0.000    | 1.000    | 0.000    | 1.000
-----|----------|----------|----------|----------
  1  | sin(1)   | cos(1)   | sin(0.01)| cos(0.01)
     | 0.841    | 0.540    | 0.010    | 1.000
-----|----------|----------|----------|----------
 29  | sin(29)  | cos(29)  | sin(0.29)| cos(0.29)
     | -0.664   | -0.748   | 0.286    | 0.958

Las frecuencias varían: dimensiones bajas cambian rápido,
dimensiones altas cambian lento. Esto codifica posición a
múltiples escalas.
```

### 4.7 Transformer Block Completo

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRANSFORMER BLOCK                               │
└─────────────────────────────────────────────────────────────────────────┘

Input: X ∈ ℝ³⁰ˣ⁵¹² (30 frames × 512 dimensiones)
           │
           ├───────────────────────────┐
           │                           │ (Residual Connection)
           ▼                           │
    ┌─────────────────┐               │
    │ Multi-Head      │               │
    │ Self-Attention  │               │
    │ (8 heads)       │               │
    └────────┬────────┘               │
             │                         │
             ▼                         │
    ┌─────────────────┐               │
    │    Add & Norm   │◄──────────────┘
    │  (LayerNorm)    │
    └────────┬────────┘
             │
             ├───────────────────────────┐
             │                           │ (Residual Connection)
             ▼                           │
    ┌─────────────────┐               │
    │  Feed-Forward   │               │
    │  Dense(256)     │               │
    │  GELU           │               │
    │  Dense(512)     │               │
    └────────┬────────┘               │
             │                         │
             ▼                         │
    ┌─────────────────┐               │
    │    Add & Norm   │◄──────────────┘
    │  (LayerNorm)    │
    └────────┬────────┘
             │
             ▼
Output: X' ∈ ℝ³⁰ˣ⁵¹²
```

---

## 5. Comparación Matemática

### Tabla Comparativa

| Aspecto | Random Forest | LSTM | Transformer |
|---------|---------------|------|-------------|
| **Input** | $x \in \mathbb{R}^{132}$ | $X \in \mathbb{R}^{30 \times 396}$ | $X \in \mathbb{R}^{30 \times 396}$ |
| **Tipo** | Ensemble de árboles | Recurrente | Attention-based |
| **Temporal** | ❌ No | ✅ Secuencial | ✅ Paralelo |
| **Complejidad** | $O(n \cdot \log n \cdot T)$ | $O(T \cdot d^2)$ | $O(T^2 \cdot d)$ |
| **Parámetros** | ~5,000 | ~150,000 | ~500,000 |
| **Memoria** | $O(1)$ por frame | $O(d)$ cell state | $O(T^2)$ attention |

Donde $T$ = longitud de secuencia, $d$ = dimensión de features.

### Capacidad de Modelar Dependencias

```
RANDOM FOREST:
    Frame 1 ───► Predicción
    (No hay conexión entre frames)

LSTM:
    Frame 1 ─h₁→ Frame 2 ─h₂→ ... ─h₂₉→ Frame 30 ───► Predicción
    (Dependencias propagadas secuencialmente)
    
    Problema: Información de Frame 1 debe pasar por 29 pasos
              para llegar a Frame 30. Puede degradarse.

TRANSFORMER:
    Frame 1 ←────────────────────────────────→ Frame 30
    Frame 1 ←────────────────────→ Frame 15
    Frame 15 ←───────────────────→ Frame 30
    (Todas las dependencias en UN paso de attention)
    
    Ventaja: Frame 1 conecta directamente con Frame 30.
```

---

## 6. Funciones de Activación

### 6.1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```
Propiedades:
- Rango: (0, 1)
- Uso: Gates en LSTM, capa de salida para clasificación binaria
- Derivada: σ'(x) = σ(x)(1 - σ(x))

        1.0 │          ════════════
            │        ╱
            │      ╱
        0.5 │────╱────────────────
            │  ╱
            │╱
        0.0 ╘════════════════════════
           -6  -4  -2   0   2   4   6
```

### 6.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

```
Propiedades:
- Rango: (-1, 1)
- Uso: Cell state en LSTM
- Centrada en cero (mejor que sigmoid para hidden states)

        1.0 │          ════════════
            │        ╱
        0.0 │──────╱──────────────
            │    ╱
       -1.0 │════
           -6  -4  -2   0   2   4   6
```

### 6.3 ReLU

$$\text{ReLU}(x) = \max(0, x)$$

```
Propiedades:
- Rango: [0, ∞)
- Uso: Capas densas intermedias
- Simple y eficiente

            │            ╱
            │          ╱
            │        ╱
        0.0 │══════╱
            │
           -2  -1   0   1   2   3
```

### 6.4 GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x)$$

Donde $\Phi(x)$ es la CDF de la distribución normal estándar.

Aproximación:
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

```
Propiedades:
- Similar a ReLU pero suave
- Usado en Transformers (mejor que ReLU en práctica)
- No tiene "muerte de neuronas" como ReLU

            │            ╱
            │          ╱
            │       ╱╱
        0.0 │═════╱
            │   ╱  (curva suave, no esquina)
           -2  -1   0   1   2   3
```

---

## 7. Función de Pérdida: Binary Cross-Entropy

### Definición

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Donde:
- $y_i \in \{0, 1\}$ = etiqueta real
- $\hat{y}_i \in (0, 1)$ = probabilidad predicha
- $N$ = número de muestras

### Interpretación

```
Si y = 1 (caída real):
    L = -log(ŷ)
    
    ŷ = 0.9 → L = -log(0.9) = 0.105  (bajo, bueno)
    ŷ = 0.5 → L = -log(0.5) = 0.693  (moderado)
    ŷ = 0.1 → L = -log(0.1) = 2.303  (alto, malo)

Si y = 0 (ADL real):
    L = -log(1 - ŷ)
    
    ŷ = 0.1 → L = -log(0.9) = 0.105  (bajo, bueno)
    ŷ = 0.5 → L = -log(0.5) = 0.693  (moderado)
    ŷ = 0.9 → L = -log(0.1) = 2.303  (alto, malo)
```

### Gradiente

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1-y_i}{1-\hat{y}_i} = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}$$

---

## 8. Optimización: Adam

### El Algoritmo

Adam = **Ada**ptive **M**oment Estimation

Combina:
1. **Momentum** (promedios móviles del gradiente)
2. **RMSprop** (adaptación del learning rate por parámetro)

**Ecuaciones:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$

Corrección de sesgo:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

Actualización:
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### Hiperparámetros Típicos

```
α (learning_rate) = 0.001
β₁ = 0.9   (momentum del gradiente)
β₂ = 0.999 (momentum del gradiente²)
ε = 1e-8   (estabilidad numérica)
```

---

## 9. Preguntas de Entrevista MIT

### P1: "¿Cuál es la complejidad computacional de Self-Attention?"

**Respuesta:**
> "La complejidad es $O(T^2 \cdot d)$ donde $T$ es la longitud de la secuencia y $d$ la dimensión. El término $T^2$ viene de calcular attention scores entre todos los pares de frames. Para nuestra secuencia de 30 frames, esto significa 900 comparaciones, lo cual es manejable. Para secuencias muy largas (>1000), existen variantes eficientes como Linformer o Performer."

### P2: "¿Por qué LSTM usa tanh en el cell state pero sigmoid en los gates?"

**Respuesta:**
> "Los gates necesitan valores entre 0 y 1 para actuar como 'interruptores' que controlan el flujo de información (0 = bloquear, 1 = permitir). Sigmoid produce exactamente ese rango. El cell state usa tanh porque necesita representar información que puede ser positiva o negativa, y tanh está centrada en cero con rango (-1, 1), lo que permite que el cell state aumente o disminuya."

### P3: "¿Cómo previene el Forget Gate el vanishing gradient?"

**Respuesta:**
> "En RNN clásicas, el gradiente se multiplica repetidamente por la matriz de pesos, causando que tienda a cero. En LSTM, el cell state se actualiza con $c_t = f_t \odot c_{t-1} + ...$, donde $f_t$ puede ser cercano a 1. Esto crea un 'camino de gradiente' casi lineal que permite que la información fluya sin degradarse significativamente a través de muchos pasos temporales."

### P4: "¿Por qué escalar por $\sqrt{d_k}$ en attention?"

**Respuesta:**
> "Sin escalamiento, el producto punto $q \cdot k$ crece con la dimensionalidad. Para vectores aleatorios de dimensión $d$, la varianza del producto punto es aproximadamente $d$. Valores muy grandes hacen que softmax sature (gradientes cercanos a cero). Dividir por $\sqrt{d_k}$ normaliza la varianza a 1, manteniendo los gradientes en un rango saludable."

### P5: "¿Cuál es la diferencia entre Gini y Entropía para árboles de decisión?"

**Respuesta:**
> "Ambas miden impureza. Gini: $1 - \sum p_i^2$, Entropía: $-\sum p_i \log p_i$. Matemáticamente, Gini es una aproximación de segundo orden de la entropía. En práctica, producen árboles muy similares. Gini es ligeramente más rápido de computar (no requiere logaritmos). Scikit-learn usa Gini por defecto."

---

## 📌 Resumen de Fórmulas Clave

### Random Forest
$$\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$$
$$\hat{y} = \frac{1}{T}\sum_{t=1}^{T} h_t(x)$$

### LSTM
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

### Transformer
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Binary Cross-Entropy
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}[y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

---

*Documento preparado para MIT Global Teaching Labs 2025*

*SafeGuard Vision AI - Industry 4.0 Zero Accident Initiative*
