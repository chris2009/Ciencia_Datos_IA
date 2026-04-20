# 📊 Análisis Completo de Gráficos: Evolución de Modelos

## SafeGuard Vision AI | MIT Global Teaching Labs 2025

### De Random Forest a Transformer: Una Historia de Mejora Continua

---

## 📑 Tabla de Contenidos

1. [Resumen de la Evolución](#1-resumen-de-la-evolución)
2. [MODELO 1: Random Forest (Sin Balancear)](#2-modelo-1-random-forest-sin-balancear)
3. [MODELO 2: Random Forest (Balanceado + Anti-Overfitting)](#3-modelo-2-random-forest-balanceado--anti-overfitting)
4. [MODELO 3: LSTM (Análisis Temporal)](#4-modelo-3-lstm-análisis-temporal)
5. [MODELO 4: Transformer (Self-Attention)](#5-modelo-4-transformer-self-attention)
6. [Comparación Final: La Evolución Completa](#6-comparación-final-la-evolución-completa)
7. [Glosario de Métricas](#7-glosario-de-métricas)

---

## 1. Resumen de la Evolución

### La Historia en Números

| Modelo | Recall | FN | Problema Identificado | Solución Aplicada |
|--------|--------|----|-----------------------|-------------------|
| RF Sin Balancear | 88.9% | 38 | Dataset 5.5:1 desbalanceado | → Balancear datos |
| RF Balanceado | 100%* | 0 | *Threshold=0.01, muchos FP | → Modelos temporales |
| LSTM | 100% | 0 | - | ✅ Solución óptima |
| Transformer | 100% | 0 | - | ✅ Solución óptima |

### El Insight Principal

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  Random Forest: Analiza UN frame → No detecta TRANSICIONES               ║
║                                                                           ║
║  LSTM/Transformer: Analizan 30 frames → Detectan el MOVIMIENTO de caída  ║
║                                                                           ║
║  Una caída NO es una pose. Es un EVENTO TEMPORAL.                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. MODELO 1: Random Forest (Sin Balancear)

### 📍 Archivo: `model_evaluation.png`

Este fue nuestro **punto de partida** - el modelo baseline.

---

### 2.1 Matriz de Confusión

```
                    PREDICCIÓN
                   ADL      Fall
              ┌──────────┬──────────┐
       ADL    │   1819   │     3    │  → Solo 3 falsas alarmas
   REAL       ├──────────┼──────────┤
       Fall   │    38    │   303    │  → 38 CAÍDAS NO DETECTADAS ⚠️
              └──────────┴──────────┘
              
TN = 1819 | FP = 3 | FN = 38 | TP = 303
Total = 2163 muestras
```

### Interpretación

| Celda | Valor | Significado |
|-------|-------|-------------|
| **TN = 1819** | Correcto | Actividades normales bien clasificadas |
| **FP = 3** | Error menor | Solo 3 falsas alarmas (excelente Precision) |
| **FN = 38** | ⚠️ **CRÍTICO** | **38 caídas que NO detectamos** |
| **TP = 303** | Correcto | Caídas correctamente detectadas |

### Cálculo de Métricas

```
Recall = TP / (TP + FN) = 303 / (303 + 38) = 303/341 = 88.86%

Precision = TP / (TP + FP) = 303 / (303 + 3) = 303/306 = 99.02%

Accuracy = (TN + TP) / Total = (1819 + 303) / 2163 = 98.10%
```

### 🔴 El Problema

> **38 False Negatives = 38 caídas sin detectar**
>
> En un entorno industrial con 341 caídas, 38 trabajadores se quedarían en el suelo sin ayuda.
>
> **Recall del 88.86% NO ES ACEPTABLE para seguridad industrial.**

---

### 2.2 Curva ROC

```
     TPR (Recall)
     1.0 │      ●━━━━━━━━━━━━━━━●
         │     ╱
         │    ╱
     0.5 │   ╱      Área azul = AUC = 0.998
         │  ╱  
         │ ╱.........(línea diagonal = aleatorio)
     0.0 │╱
         └────────────────────────────
         0        0.5              1.0
                  FPR
```

### Interpretación

| Métrica | Valor | Significado |
|---------|-------|-------------|
| **AUC = 0.998** | Excelente | El modelo PUEDE separar las clases |
| **Curva "abraza" esquina** | Bueno | Alta TPR con baja FPR posible |

### 🤔 La Paradoja

> AUC = 0.998 parece excelente, pero aún tenemos 38 FN.
>
> ¿Por qué? Porque AUC mide la **capacidad** de separar clases, no el rendimiento con un threshold específico.
>
> El problema es el **desbalance**: 1822 ADL vs 341 Falls (5.3:1)

---

### 2.3 Top 20 Features más Importantes

```
right_hip_z      ████████████████████████████ 0.027
left_hip_z       ███████████████████████ 0.023
left_heel_y      ███████████████████ 0.019
left_hip_x       ██████████████████ 0.018
left_foot_index_x ████████████████ 0.016
right_hip_y      ███████████████ 0.015
body_width       ██████████████ 0.014
left_ankle_x     █████████████ 0.013
...
```

### Interpretación

| Feature | Por qué es importante |
|---------|----------------------|
| **right_hip_z, left_hip_z** | La profundidad de la cadera indica si la persona está de pie o acostada |
| **left_heel_y, left_hip_x** | Posición vertical del talón y horizontal de cadera |
| **body_width** | Ancho del cuerpo (feature derivada) |

### 🔑 Observación Clave

> **Todos son KEYPOINTS ORIGINALES (azul), no features derivadas.**
>
> El modelo está usando la POSE ESTÁTICA, no el movimiento.
>
> Esto explica por qué confunde "persona acostada voluntariamente" con "persona que cayó".

---

### 2.4 Distribución de Probabilidades

```
     Frecuencia
    1200│■
    1000│■
     800│■
     600│■    ADL (verde)           
     400│■    concentrado           Falls (rojo)
     200│■■   cerca de 0            dispersos 0.5-1.0
       0└────────────┼─────────────────────────
        0.0   0.2   0.5   0.7   0.9   1.0
                     │
               Threshold=0.5
```

### Interpretación

| Observación | Significado |
|-------------|-------------|
| **ADL muy concentrado en 0** | El modelo está muy seguro de que NO son caídas |
| **Falls dispersos entre 0.5-1.0** | El modelo tiene MENOS confianza en las caídas |
| **Algunos Falls cerca del threshold** | Estos son los 38 que se clasifican mal |

### 🔴 El Problema Visual

> Las barras rojas (caídas) NO están completamente separadas.
> Algunas caídas tienen probabilidad < 0.5 y se clasifican como ADL.
> Esto genera los 38 False Negatives.

---

## 3. MODELO 2: Random Forest (Balanceado + Anti-Overfitting)

### 📍 Archivo: `model_evaluation_v2.png`

Aplicamos **dos mejoras**:
1. Balanceo del dataset a 1:1
2. Parámetros anti-overfitting

---

### 3.1 Matriz de Confusión (con threshold=0.01)

```
                    PREDICCIÓN
                   ADL      Fall
              ┌──────────┬──────────┐
       ADL    │     0    │   333    │  → TODAS clasificadas como Fall
   REAL       ├──────────┼──────────┤
       Fall   │     0    │   333    │  → TODAS las caídas detectadas
              └──────────┴──────────┘
              
Con threshold = 0.01 (muy bajo)
```

### ⚠️ ¿Qué pasó aquí?

> Usamos threshold = 0.01 para **maximizar Recall**.
>
> Resultado: Recall = 100% pero Precision = 50%
>
> **Todo se clasifica como caída** - incluidas las actividades normales.

### ¿Por qué threshold = 0.01?

El gráfico **"Threshold vs Métricas"** (abajo-derecha) lo explica:

```
Score
1.0 │──●━━━━━━━━━━●            ← Recall (verde) = 100% con threshold bajo
    │              ╲
0.9 │               ●━━━━━━━━  ← Recall cae al subir threshold
    │                    ╲
0.8 │                     ╲
    │     ╱━━━━━━━━━━━━━━━━●  ← Precision (azul) sube con threshold alto
0.6 │    ╱
    │   ╱
    └──┼──────────────────────
      0.01  0.2   0.5   0.8  1.0
           │
      Óptimo (según el gráfico)
```

### 🔴 El Trade-off

> **No podemos tener Recall=100% Y Precision alta con este modelo.**
>
> Si bajamos threshold → Recall sube, Precision baja
> Si subimos threshold → Precision sube, Recall baja
>
> Este trade-off existe porque **el modelo no puede distinguir bien las clases**.

---

### 3.2 Curva ROC

```
AUC = 0.990 (ligeramente menor que antes: 0.998)
```

### Interpretación

| Comparación | RF Original | RF Balanceado |
|-------------|-------------|---------------|
| **AUC** | 0.998 | 0.990 |
| **¿Por qué bajó?** | - | Dataset más difícil (balanceado) |

> El AUC bajó ligeramente porque ahora el modelo ve igual cantidad de cada clase.
> Antes "hacía trampa" prediciendo mayormente ADL.

---

### 3.3 Curva Precision-Recall

```
Precision
1.0 │●━━━━━━━━━━━━━━━━━━━●
    │                     ╲
0.9 │                      ╲
    │                       ╲
0.8 │                        ╲
    │                         ●
0.5 │.........................│.....(Recall objetivo 95%)
    └─────────────────────────┼────
    0        0.5        0.95  1.0
                              │
                         Recall
```

### Interpretación

> La curva muestra que para lograr **Recall = 95%**, la Precision cae significativamente.
>
> **No hay un punto donde tengamos ambas métricas altas.**
>
> Esto indica que necesitamos un enfoque diferente.

---

### 3.4 Distribución de Probabilidades (CON DOS THRESHOLDS)

```
     Frecuencia
     50│
     40│■                              ■
     30│■■                            ■■
     20│■■■    ADL (verde)    Falls (rojo)
     10│■■■■■                     ■■■■■
      0└───────┼───────┼─────────────────
       0.0    0.01   0.5              1.0
              │       │
         Óptimo   Default
        threshold  threshold
```

### Interpretación

| Línea | Significado |
|-------|-------------|
| **Naranja (0.01)** | Threshold que da 100% Recall pero 50% Precision |
| **Azul (0.5)** | Threshold estándar - balance entre P y R |

### 🔴 El Problema Fundamental

> **HAY OVERLAP entre las distribuciones.**
>
> Las barras verdes (ADL) y rojas (Falls) se mezclan en el centro.
>
> No importa qué threshold elijamos, siempre habrá errores.
>
> **Necesitamos un modelo que SEPARE MEJOR las clases.**

---

### 3.5 Top 20 Features

```
left_foot_index_x  ████████████████████████████ 0.032
left_heel_x        ███████████████████████ 0.028
left_heel_y        ██████████████████████ 0.027
right_hip_z        █████████████████████ 0.025
right_heel_x       ████████████████████ 0.024
...
```

### Observación

> Los features importantes siguen siendo **keypoints de posición**.
>
> **No hay información temporal** (velocidad, aceleración).
>
> El modelo sigue analizando POSES, no MOVIMIENTOS.

---

## 4. MODELO 3: LSTM (Análisis Temporal)

### 📍 Archivo: `training_history.png`

**El cambio de paradigma:** De frames individuales a secuencias de 30 frames.

---

### 4.1 Función de Pérdida (Loss)

```
     Loss
     0.5 │●
         │ ╲
     0.4 │  ╲  Train (azul)
         │   ╲
     0.3 │    ╲
         │     ╲
     0.2 │      ╲___
         │          ╲___●___●___●___●  ← Converge ~0.05
     0.1 │●━━━━●━━━━━━━━━━━━━━━━━━━━━  ← Validation (naranja)
         │
     0.0 └────────────────────────────
         0   2   4   6   8  10  12  14
                      Epoch
```

### Interpretación

| Observación | Significado |
|-------------|-------------|
| **Train loss baja rápido** | El modelo está aprendiendo |
| **Validation loss baja también** | El modelo GENERALIZA (no memoriza) |
| **No hay divergencia** | **NO hay overfitting** |
| **Validation más baja que Train** | Normal con Dropout (regularización) |

### 🟢 Señal Positiva

> Ambas curvas convergen hacia valores bajos (~0.05).
>
> Si hubiera overfitting, veríamos: Train ↓ mientras Validation ↑
>
> **Este modelo generaliza bien.**

---

### 4.2 Precisión (Accuracy)

```
     Accuracy
     1.00│        ●━━━━━━━━━━━━━━━━━━  ← Validation = 100%
         │       ╱
     0.98│      ╱
         │     ╱  ●━━━━━━━━━━━━━━━━━━  ← Train converge a ~99%
     0.96│    ╱
         │   ╱
     0.94│  ╱
         │ ●
         └────────────────────────────
         0   2   4   6   8  10  12  14
```

### Interpretación

> **Validation Accuracy alcanza 100% y se mantiene.**
>
> El modelo clasifica correctamente TODAS las muestras de validación.

---

### 4.3 Recall (Detección de Caídas) - LA MÉTRICA CLAVE

```
     Recall
     1.00│    ●━━━━━━━━━━━━━━━━━━━━━━  ← Validation = 100% desde época 2
         │   ╱
     0.98│  ╱
         │ ╱    ●━━━━━━●━━━━━━━━━━━━━  ← Train fluctúa pero alto
     0.96│╱
         │
     0.94│
         │
         └────────────────────────────
         0   2   4   6   8  10  12  14
```

### Interpretación

| Observación | Significado |
|-------------|-------------|
| **Validation Recall = 100%** | **TODAS las caídas detectadas** |
| **Estable desde época 2** | El modelo aprende rápido el patrón |
| **Train ligeramente menor** | Normal - entrena con más variedad |

### 🟢 ¡BREAKTHROUGH!

> **Recall de validación = 100% = FN = 0**
>
> Por primera vez, nuestro modelo detecta TODAS las caídas.
>
> **Ningún trabajador se queda sin ayuda.**

---

### 4.4 Curva ROC

```
     TPR
     1.0 │■■■■■■■■■■■■■■■■■■■■■■■■
         │■
         │■
     0.5 │■       AUC = 1.000
         │■      (PERFECTO)
         │■
     0.0 │■_______________________
         0        0.5           1.0
                  FPR
```

### Interpretación

> **AUC = 1.000** significa separación PERFECTA entre clases.
>
> Existe un threshold donde tenemos 100% TPR con 0% FPR.
>
> El modelo puede distinguir completamente las caídas de las actividades normales.

---

### 4.5 Distribución de Predicciones

```
     Frecuencia
     4 │
     3 │     ■■              ■■■■  ← Pico en ~1.0
     2 │   ■■■■■            ■■■■
     1 │ ■■■■■■■■    ■     ■■■■■
     0 └────────────┼─────────────
       0.25  0.35  0.5  0.7  0.85 1.0
                    │
              Threshold
       
       ■ Verde = ADL     ■ Rojo = Falls
```

### Interpretación

| Observación | Significado |
|-------------|-------------|
| **ADL concentrado en 0.25-0.45** | Modelo asigna baja probabilidad |
| **Falls concentrado en 0.7-1.0** | Modelo asigna alta probabilidad |
| **Clara SEPARACIÓN** | ¡Las distribuciones NO se solapan! |

### 🟢 La Diferencia Clave vs Random Forest

> En Random Forest había **overlap** entre las distribuciones.
>
> En LSTM hay **separación clara**.
>
> Por eso LSTM logra 100% Recall sin sacrificar Precision.

---

### 4.6 Matriz de Confusión

```
                    PREDICCIÓN
                   ADL      Fall
              ┌──────────┬──────────┐
       ADL    │    16    │     1    │  → Solo 1 falsa alarma
   REAL       ├──────────┼──────────┤
       Fall   │     0    │    16    │  → CERO caídas perdidas ✓
              └──────────┴──────────┘
              
TN = 16 | FP = 1 | FN = 0 | TP = 16
Total = 33 secuencias de test
```

### Cálculo de Métricas

```
Recall = TP / (TP + FN) = 16 / (16 + 0) = 100% ✓

Precision = TP / (TP + FP) = 16 / (16 + 1) = 94.12%

Accuracy = (TN + TP) / Total = (16 + 16) / 33 = 96.97%

F1-Score = 2 × (P × R) / (P + R) = 2 × (0.9412 × 1.0) / 1.9412 = 96.97%
```

### 🟢 Resultado Final LSTM

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Recall** | **100%** | Todas las caídas detectadas |
| **Precision** | 94.12% | 1 falsa alarma de 17 predicciones |
| **FN** | **0** | Cero caídas perdidas |
| **AUC** | 1.000 | Separación perfecta |

---

## 5. MODELO 4: Transformer (Self-Attention)

### 📍 Archivo: `training_history_transformer.png`

Validamos nuestro enfoque temporal con una arquitectura diferente.

---

### 5.1 Función de Pérdida

```
     Loss
     0.6 │●
         │ ╲
     0.5 │  ╲  Train (azul)
         │   ╲
     0.4 │    ╲
         │     ╲    ╱╲
     0.2 │      ╲__╱  ╲__
         │              ╲__●___●___●  ← Converge ~0.01
     0.1 │●━━●━━━━━━━━━━━━━━━━━━━━━━  ← Validation (naranja)
         │
     0.0 └────────────────────────────
         0   2   4   6   8  10  12  14
```

### Comparación con LSTM

| Aspecto | LSTM | Transformer |
|---------|------|-------------|
| **Convergencia** | Gradual | Más rápida |
| **Inestabilidad** | Menor | Pico en época 5-6 |
| **Loss final** | ~0.05 | ~0.01 |

### Interpretación

> El Transformer tiene **más inestabilidad** inicial (pico en época 5-6).
>
> Pero converge a un **loss más bajo** (~0.01 vs ~0.05).
>
> La inestabilidad es normal - el mecanismo de atención necesita "encontrar" los patrones correctos.

---

### 5.2 Precisión (Precision)

```
     Precision
     1.00│          ●━━━━━━━━━━━━━━━━  ← Ambas convergen a ~100%
         │         ╱
     0.95│    ╱╲  ╱
         │   ╱  ╲╱   ← Validation fluctúa inicialmente
     0.90│  ╱
         │ ╱
     0.75│●
         └────────────────────────────
         0   2   4   6   8  10  12  14
```

### Interpretación

> **Fluctuación inicial** en validation (épocas 3-7).
>
> Esto coincide con la inestabilidad en la loss.
>
> Después de época 8, **se estabiliza en ~100%**.

---

### 5.3 Recall (Detección de Caídas)

```
     Recall
     1.00│  ●━━━━━━━━━━━━━━━━━━━━━━━━  ← Validation = 100% desde época 1
         │ ╱
     0.95│╱      ╱╲
         │      ╱  ╲    ●━━━━━━━━━━━━  ← Train converge a ~98%
     0.90│     ╱    ╲  ╱
         │    ╱      ╲╱
     0.80│   ╱
         │  ╱
     0.75│●╱
         └────────────────────────────
         0   2   4   6   8  10  12  14
```

### Comparación con LSTM

| Aspecto | LSTM | Transformer |
|---------|------|-------------|
| **Val Recall** | 100% | 100% |
| **Desde época** | 2 | 1 |
| **Train Recall** | ~99% | ~98% |

### 🟢 Mismo Resultado

> **Validation Recall = 100%** igual que LSTM.
>
> El Transformer aprende el patrón temporal incluso más rápido (época 1).
>
> Esto **valida** que el enfoque temporal es correcto.

---

### 5.4 Curva ROC

```
AUC = 1.000 (igual que LSTM)
```

### Interpretación

> Ambos modelos temporales logran **AUC perfecta**.
>
> Esto no es coincidencia - es evidencia de que el patrón temporal de caída es **muy distintivo**.

---

### 5.5 Distribución de Predicciones

```
     Frecuencia
     7 │■
     6 │■
     5 │■     ADL (verde)          Falls (rojo)
     4 │■     cerca de 0           cerca de 1
     3 │■■                              ■■
     2 │■■■        ■              ■    ■■
     1 │■■■■■   ■  ■         ■    ■    ■■
     0 └────────────┼─────────────────────
       0.0   0.2   0.5   0.7   0.9   1.0
                    │
              Threshold
```

### Comparación con LSTM

| Aspecto | LSTM | Transformer |
|---------|------|-------------|
| **ADL** | 0.25-0.45 | 0.0-0.4 |
| **Falls** | 0.7-1.0 | 0.8-1.0 |
| **Separación** | Clara | **Más clara** |

### 🟢 Observación

> El Transformer tiene distribuciones **más concentradas** en los extremos.
>
> ADL más cerca de 0, Falls más cerca de 1.
>
> Esto indica **mayor confianza** en las predicciones.

---

### 5.6 Matriz de Confusión

```
                    PREDICCIÓN
                   ADL      Fall
              ┌──────────┬──────────┐
       ADL    │    16    │     1    │
   REAL       ├──────────┼──────────┤
       Fall   │     0    │    16    │
              └──────────┴──────────┘
              
TN = 16 | FP = 1 | FN = 0 | TP = 16
```

### Resultado

> **IDÉNTICO a LSTM:**
> - Recall = 100%
> - FN = 0
> - Precision = 94.12%

---

## 6. Comparación Final: La Evolución Completa

### 6.1 Tabla Comparativa de Métricas

| Métrica | RF Original | RF Balanceado* | LSTM | Transformer |
|---------|-------------|----------------|------|-------------|
| **Recall** | 88.86% | 100%* | **100%** | **100%** |
| **Precision** | 99.02% | 50%* | 94.12% | 94.12% |
| **F1-Score** | 93.67% | 66.67%* | 96.97% | 96.97% |
| **AUC** | 0.998 | 0.990 | 1.000 | 1.000 |
| **FN** | 38 | 0* | **0** | **0** |
| **FP** | 3 | 333* | 1 | 1 |

*Con threshold=0.01, no viable en producción

### 6.2 Evolución Visual de la Matriz de Confusión

```
RF ORIGINAL              RF BALANCEADO           LSTM/TRANSFORMER
(threshold=0.5)          (threshold=0.01)        (threshold=0.5)

┌────────┬────────┐      ┌────────┬────────┐     ┌────────┬────────┐
│  1819  │    3   │      │    0   │  333   │     │   16   │    1   │
├────────┼────────┤      ├────────┼────────┤     ├────────┼────────┤
│   38   │  303   │      │    0   │  333   │     │    0   │   16   │
└────────┴────────┘      └────────┴────────┘     └────────┴────────┘
    ⚠️ 38 FN            ⚠️ 333 FP              ✅ Balance óptimo

"Pierde caídas"       "Todo es caída"        "Detecta todo,
                                              pocas falsas alarmas"
```

### 6.3 Evolución de la Distribución de Probabilidades

```
RF ORIGINAL: Overlap significativo
─────────────────────────────────
     ADL: ████████████████████
    Fall:          ████████████████
                   ^overlap^

RF BALANCEADO: Aún hay overlap  
─────────────────────────────────
     ADL: ██████████████████
    Fall:       ██████████████████
                ^overlap^

LSTM/TRANSFORMER: Separación clara
─────────────────────────────────
     ADL: ██████████
    Fall:                  ██████████
          (sin overlap)
```

### 6.4 ¿Por Qué los Modelos Temporales Funcionan?

```
RANDOM FOREST ve esto:
┌─────────────────────┐
│                     │
│   Frame único:      │     ¿Caída o persona
│   Pose horizontal   │ →   acostada?
│                     │     NO SE PUEDE SABER
└─────────────────────┘

LSTM/TRANSFORMER ve esto:
┌────┬────┬────┬────┬────┬────┬────┐
│ F1 │ F2 │... │F15 │... │F29 │F30 │
│    │    │    │    │    │    │    │
│ 🧍 │ 🧍 │    │ ⤵️ │    │ 🧎 │ 🛌 │
└────┴────┴────┴────┴────┴────┴────┘
  │                              │
  Parado                    En el suelo
  
  → TRANSICIÓN DETECTADA = CAÍDA
```

---

## 7. Glosario de Métricas

### 7.1 Métricas Básicas

| Métrica | Fórmula | Pregunta que Responde |
|---------|---------|----------------------|
| **Recall (Sensibilidad)** | TP / (TP + FN) | "¿Qué % de caídas detecté?" |
| **Precision** | TP / (TP + FP) | "¿Qué % de mis alarmas son reales?" |
| **Accuracy** | (TP + TN) / Total | "¿Qué % total acerté?" |
| **F1-Score** | 2×P×R / (P+R) | "¿Cuál es el balance P-R?" |

### 7.2 Matriz de Confusión

| Celda | Nombre | Significado |
|-------|--------|-------------|
| **TN** | True Negative | ADL correctamente identificado |
| **FP** | False Positive | Falsa alarma (ADL → Caída) |
| **FN** | False Negative | **Caída perdida** (Caída → ADL) ⚠️ |
| **TP** | True Positive | Caída correctamente detectada |

### 7.3 Curva ROC y AUC

| Término | Definición |
|---------|------------|
| **ROC** | Receiver Operating Characteristic - Gráfico de TPR vs FPR |
| **TPR** | True Positive Rate = Recall |
| **FPR** | False Positive Rate = FP / (FP + TN) |
| **AUC** | Area Under Curve - Capacidad de separar clases (1.0 = perfecto) |

### 7.4 Threshold

| Concepto | Explicación |
|----------|-------------|
| **Threshold** | Punto de corte para decidir Caída vs ADL |
| **Default (0.5)** | Si P(caída) > 0.5 → Clasificar como caída |
| **Bajo (0.01)** | Más sensible, más Recall, más FP |
| **Alto (0.9)** | Más específico, más Precision, más FN |

---

## 📌 Resumen Ejecutivo

### Lo que Aprendimos

```
1. El DESBALANCE de datos causa problemas (RF original: 38 FN)

2. BALANCEAR ayuda pero no resuelve el problema fundamental

3. El problema fundamental es que Random Forest no ve MOVIMIENTO

4. LSTM y Transformer analizan SECUENCIAS y detectan TRANSICIONES

5. Ambos modelos temporales logran:
   - Recall = 100% (cero caídas perdidas)
   - Precision > 94% (pocas falsas alarmas)
   - AUC = 1.0 (separación perfecta)
```

### La Frase Clave

> **"Una caída no es una pose, es una transición. Solo los modelos temporales pueden detectar transiciones."**

---

*Documento preparado para MIT Global Teaching Labs 2025*

*SafeGuard Vision AI - Industry 4.0 Zero Accident Initiative*
