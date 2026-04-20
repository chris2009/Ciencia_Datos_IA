# 📊 Guía Completa de Métricas de Clasificación

## SafeGuard Vision AI | MIT Global Teaching Labs 2025

### Preparación para Evaluación Técnica

---

## 📑 Tabla de Contenidos

1. [Introducción: El Problema de Clasificación Binaria](#1-introducción-el-problema-de-clasificación-binaria)
2. [Las Clases en Nuestro Problema](#2-las-clases-en-nuestro-problema)
3. [La Matriz de Confusión: El Origen de Todo](#3-la-matriz-de-confusión-el-origen-de-todo)
4. [Métrica 1: Accuracy (Exactitud)](#4-métrica-1-accuracy-exactitud)
5. [Métrica 2: Precision (Precisión)](#5-métrica-2-precision-precisión)
6. [Métrica 3: Recall (Sensibilidad)](#6-métrica-3-recall-sensibilidad)
7. [Métrica 4: F1-Score](#7-métrica-4-f1-score)
8. [Métrica 5: AUC-ROC](#8-métrica-5-auc-roc)
9. [¿Cuándo Usar Cada Métrica?](#9-cuándo-usar-cada-métrica)
10. [¿Por Qué Recall es Crítico en Nuestro Caso?](#10-por-qué-recall-es-crítico-en-nuestro-caso)
11. [Trade-offs Entre Métricas](#11-trade-offs-entre-métricas)
12. [Preguntas Frecuentes del MIT](#12-preguntas-frecuentes-del-mit)

---

## 1. Introducción: El Problema de Clasificación Binaria

### ¿Qué es Clasificación Binaria?

La **clasificación binaria** es un tipo de problema de Machine Learning donde el modelo debe decidir entre **exactamente dos opciones** (clases).

```
Entrada (X) ──────► [ MODELO ] ──────► Salida: Clase A o Clase B
```

### Ejemplos de Clasificación Binaria:

| Dominio | Clase Positiva (1) | Clase Negativa (0) |
|---------|-------------------|-------------------|
| Email | Spam | No Spam |
| Medicina | Enfermo | Sano |
| Fraude | Fraudulento | Legítimo |
| **Nuestro caso** | **Caída** | **No Caída (ADL)** |

### ¿Por qué "Positiva" y "Negativa"?

No tiene que ver con "bueno" o "malo". Es una **convención**:

- **Clase Positiva (1):** El evento que queremos **DETECTAR**
- **Clase Negativa (0):** La ausencia de ese evento

En nuestro caso:
- **Positiva = Caída** (queremos detectarla)
- **Negativa = ADL** (Activities of Daily Living - actividades normales)

---

## 2. Las Clases en Nuestro Problema

### Clase 0: ADL (No Caída)

```
ADL = Activities of Daily Living
```

Incluye todas las actividades normales donde NO hay emergencia:

| Actividad | Descripción |
|-----------|-------------|
| Caminar | Desplazamiento normal |
| Sentarse | En silla, sofá, cama |
| Agacharse | Recoger objetos |
| Acostarse | Voluntariamente en cama/sofá |
| Trabajar | Actividades laborales normales |
| Estirarse | Ejercicios, yoga |

### Clase 1: Caída (Fall)

Una **transición no controlada** de una posición alta a una baja, típicamente al suelo.

| Tipo de Caída | Descripción |
|---------------|-------------|
| Tropiezo | Obstáculo en el camino |
| Resbalón | Superficie mojada/lisa |
| Desmayo | Pérdida de consciencia |
| Mareo | Desequilibrio súbito |
| Colapso | Debilidad física |

### ⚠️ El Desafío Principal

```
PROBLEMA VISUAL:

    Persona acostada          Persona que cayó
    voluntariamente           (emergencia)
         
         😴                        😵
        ════                      ════
        
    Pose: Horizontal          Pose: Horizontal
    Clase: 0 (ADL)            Clase: 1 (Caída)
    
    ¡¡¡ LA POSE ES IDÉNTICA !!!
```

**Por eso necesitamos análisis TEMPORAL:** No es la pose final, es la **TRANSICIÓN**.

---

## 3. La Matriz de Confusión: El Origen de Todo

### ¿Qué es la Matriz de Confusión?

Es una tabla de 2×2 que muestra **todas las posibles combinaciones** entre:
- Lo que el modelo **predijo**
- Lo que **realmente era**

### Estructura de la Matriz

```
                        PREDICCIÓN DEL MODELO
                    ┌─────────────┬─────────────┐
                    │  Predijo 0  │  Predijo 1  │
                    │  (No Caída) │   (Caída)   │
        ┌───────────┼─────────────┼─────────────┤
        │ Real 0    │             │             │
 VERDAD │ (No Caída)│     TN      │     FP      │
 REAL   ├───────────┼─────────────┼─────────────┤
        │ Real 1    │             │             │
        │ (Caída)   │     FN      │     TP      │
        └───────────┴─────────────┴─────────────┘
```

### Los 4 Cuadrantes Explicados

#### ✅ TN - True Negative (Verdadero Negativo)
```
Realidad: NO hubo caída
Predicción: NO hubo caída
Resultado: ¡CORRECTO! ✓

Ejemplo: Persona caminando → Modelo dice "Normal" → Correcto
```

#### ✅ TP - True Positive (Verdadero Positivo)
```
Realidad: SÍ hubo caída
Predicción: SÍ hubo caída
Resultado: ¡CORRECTO! ✓

Ejemplo: Persona cayó → Modelo dice "Caída" → Correcto (¡Vida salvada!)
```

#### ❌ FP - False Positive (Falso Positivo) - "Falsa Alarma"
```
Realidad: NO hubo caída
Predicción: SÍ hubo caída
Resultado: ERROR (Tipo I)

Ejemplo: Persona se agacha → Modelo dice "Caída" → Falsa alarma
Consecuencia: Molestia, desconfianza en el sistema
```

#### ❌ FN - False Negative (Falso Negativo) - "Caída Perdida"
```
Realidad: SÍ hubo caída
Predicción: NO hubo caída
Resultado: ERROR (Tipo II) ⚠️ CRÍTICO

Ejemplo: Persona cayó → Modelo dice "Normal" → ¡NO SE DETECTÓ!
Consecuencia: Persona sin ayuda, posible muerte
```

### 🚨 En Seguridad Industrial: FN es INACEPTABLE

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   FP (Falsa Alarma) = Molestia, pérdida de tiempo                        ║
║                       → Costo: $100 - $1,000                             ║
║                                                                           ║
║   FN (Caída Perdida) = Persona sin ayuda, posible muerte                 ║
║                       → Costo: INCALCULABLE (vida humana)                ║
║                                                                           ║
║   Por lo tanto: FN >> FP en términos de gravedad                         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Ejemplo Numérico Real de Nuestro Proyecto

**Random Forest Balanceado:**
```
                    Predicción
                  ADL    │  Caída
              ┌─────────┼─────────┐
    Real ADL  │   155   │    12   │  → 12 falsas alarmas
              ├─────────┼─────────┤
    Real Caída│     9   │   157   │  → 9 caídas NO detectadas ⚠️
              └─────────┴─────────┘
              
Total: 333 muestras
Correctos: 155 + 157 = 312
Errores: 12 + 9 = 21
```

**LSTM (100% Recall):**
```
                    Predicción
                  ADL    │  Caída
              ┌─────────┼─────────┐
    Real ADL  │    16   │     2   │  → 2 falsas alarmas
              ├─────────┼─────────┤
    Real Caída│     0   │    16   │  → 0 caídas perdidas ✓
              └─────────┴─────────┘
              
¡CERO False Negatives! Todas las caídas detectadas.
```

---

## 4. Métrica 1: Accuracy (Exactitud)

### Definición

```
                    Predicciones Correctas
Accuracy = ─────────────────────────────────────
                    Total de Predicciones


                TN + TP
Accuracy = ─────────────────
            TN + TP + FP + FN
```

### En Palabras Simples

> "De todas las predicciones que hice, ¿qué porcentaje fue correcto?"

### Ejemplo Numérico

```
TN = 155, TP = 157, FP = 12, FN = 9

Accuracy = (155 + 157) / (155 + 157 + 12 + 9)
         = 312 / 333
         = 0.9369
         = 93.69%
```

### ⚠️ EL PROBLEMA CON ACCURACY: La Paradoja del Dataset Desbalanceado

Imagina este escenario extremo:

```
Dataset con 1000 muestras:
- 950 son NO caída (95%)
- 50 son Caída (5%)

Modelo TONTO que SIEMPRE predice "NO caída":

                    Predicción
                  ADL    │  Caída
              ┌─────────┼─────────┐
    Real ADL  │   950   │     0   │
              ├─────────┼─────────┤
    Real Caída│    50   │     0   │  ← ¡TODAS las caídas perdidas!
              └─────────┴─────────┘

Accuracy = (950 + 0) / 1000 = 95% 🎉... ¿éxito?

¡NO! El modelo es INÚTIL. No detecta NINGUNA caída.
```

### ¿Cuándo SÍ Usar Accuracy?

| Situación | ¿Usar Accuracy? |
|-----------|-----------------|
| Dataset balanceado (50/50) | ✅ Sí |
| Ambos errores igual de graves | ✅ Sí |
| Dataset muy desbalanceado | ❌ No |
| Un error es mucho peor que otro | ❌ No |

### Ejemplos de Uso Apropiado

- **Reconocimiento de dígitos (0-9):** Cada dígito tiene ~10% del dataset
- **Clasificación de imágenes generales:** Gatos vs Perros balanceado
- **Tests A/B:** Comparar versiones con grupos iguales

---

## 5. Métrica 2: Precision (Precisión)

### Definición

```
                        Verdaderos Positivos
Precision = ─────────────────────────────────────────
            Verdaderos Positivos + Falsos Positivos


                   TP
Precision = ─────────────
             TP + FP
```

### En Palabras Simples

> "De todas las veces que dije 'CAÍDA', ¿cuántas veces realmente era una caída?"

O también:

> "¿Qué tan confiables son mis alarmas?"

### Ejemplo Numérico

```
TP = 157 (caídas correctamente detectadas)
FP = 12 (falsas alarmas)

Precision = 157 / (157 + 12)
          = 157 / 169
          = 0.929
          = 92.9%
```

**Interpretación:** "El 92.9% de las veces que el modelo dice 'Caída', realmente es una caída."

### ¿Qué Mide Realmente?

```
Precision ALTA (→ 100%):
├── Pocas falsas alarmas
├── Cuando dice "Caída", casi siempre es verdad
└── El sistema es CONFIABLE

Precision BAJA (→ 0%):
├── Muchas falsas alarmas
├── El sistema grita "¡Lobo!" constantemente
└── Usuarios ignoran las alarmas (peligroso)
```

### ¿Cuándo es CRÍTICA la Precision?

| Dominio | Por qué importa Precision |
|---------|--------------------------|
| **Filtro de Spam** | FP = Email importante en spam → Pierdes negocios |
| **Búsqueda de Google** | FP = Resultados irrelevantes → Usuario frustrado |
| **Sistema Judicial** | FP = Inocente en prisión → Injusticia grave |
| **Recomendaciones** | FP = Sugerencias malas → Usuario pierde confianza |

### El Trade-off con Recall

```
Si subo mucho la Precision (reduzco FP):
→ Me vuelvo muy "exigente" para declarar caída
→ Algunas caídas reales no las detecto (aumenta FN)
→ El Recall BAJA

¡No puedo maximizar ambas simultáneamente!
```

---

## 6. Métrica 3: Recall (Sensibilidad)

### 🔥 LA MÉTRICA MÁS IMPORTANTE PARA NUESTRO PROYECTO

### Definición

```
                    Verdaderos Positivos
Recall = ─────────────────────────────────────────
         Verdaderos Positivos + Falsos Negativos


               TP
Recall = ─────────────
          TP + FN
```

### En Palabras Simples

> "De todas las caídas REALES, ¿cuántas logré detectar?"

O también:

> "¿Qué porcentaje de caídas NO se me escaparon?"

### Otros Nombres

- **Sensibilidad** (Sensitivity)
- **Tasa de Verdaderos Positivos** (True Positive Rate - TPR)
- **Tasa de Detección** (Detection Rate)
- **Hit Rate**

### Ejemplo Numérico

**Random Forest Balanceado:**
```
TP = 157 (caídas detectadas)
FN = 9 (caídas NO detectadas)

Recall = 157 / (157 + 9)
       = 157 / 166
       = 0.9458
       = 94.58%
```

**Interpretación:** "Detectamos el 94.58% de las caídas. Se nos escapó el 5.42%."

**LSTM:**
```
TP = 16
FN = 0

Recall = 16 / (16 + 0)
       = 16 / 16
       = 1.0
       = 100%
```

**Interpretación:** "Detectamos el 100% de las caídas. No se escapó ninguna."

### ¿Por Qué Recall = Detección de Caídas?

```
                    ┌─────────────────────────────────┐
                    │     TODAS LAS CAÍDAS REALES     │
                    │         (TP + FN = 166)         │
                    │                                 │
                    │   ┌─────────────────────────┐   │
                    │   │                         │   │
                    │   │   Caídas DETECTADAS     │   │
                    │   │      (TP = 157)         │   │
                    │   │                         │   │
                    │   │       = 94.58%          │   │
                    │   │                         │   │
                    │   └─────────────────────────┘   │
                    │                                 │
                    │   Caídas PERDIDAS (FN = 9)      │
                    │        = 5.42%                  │
                    └─────────────────────────────────┘

Recall = Área detectada / Área total = TP / (TP + FN)
```

### ¿Cuándo es CRÍTICO el Recall?

| Dominio | Por qué importa Recall |
|---------|------------------------|
| **Detección de cáncer** | FN = Cáncer no detectado → Paciente muere |
| **Seguridad aeroportuaria** | FN = Bomba no detectada → Catástrofe |
| **Detección de fraude** | FN = Fraude no detectado → Pérdidas millonarias |
| **🛡️ NUESTRO CASO** | FN = Caída no detectada → Trabajador muere |

### 100% Recall: El Santo Grial para Seguridad

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   Recall = 100%                                                           ║
║                                                                           ║
║   Significa:                                                              ║
║   • FN = 0 (Cero False Negatives)                                        ║
║   • TODAS las caídas fueron detectadas                                   ║
║   • NINGUNA caída pasó desapercibida                                     ║
║   • En seguridad industrial: NADIE se queda sin ayuda                    ║
║                                                                           ║
║   LSTM y Transformer lograron esto.                                       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 7. Métrica 4: F1-Score

### Definición

```
                   2 × Precision × Recall
F1-Score = ─────────────────────────────────
                  Precision + Recall
```

Esta es la **media armónica** de Precision y Recall.

### ¿Por Qué Media Armónica y No Promedio Simple?

**Promedio simple (aritmético):**
```
Si Precision = 100% y Recall = 0%:
Promedio = (100 + 0) / 2 = 50%

¡Parece aceptable, pero el modelo es INÚTIL!
(No detecta ninguna caída)
```

**Media armónica (F1):**
```
Si Precision = 100% y Recall = 0%:
F1 = 2 × (1.0 × 0) / (1.0 + 0) = 0 / 1 = 0%

¡Correctamente indica que el modelo es malo!
```

### Propiedad Clave del F1

> La media armónica **penaliza** los valores extremadamente bajos.
> 
> Si cualquiera de las dos (Precision o Recall) es muy baja, el F1 también será bajo.

### Ejemplo Numérico

**Random Forest Balanceado:**
```
Precision = 0.929
Recall = 0.9458

F1 = 2 × (0.929 × 0.9458) / (0.929 + 0.9458)
   = 2 × 0.8785 / 1.8748
   = 1.757 / 1.8748
   = 0.937
   = 93.7%
```

**LSTM:**
```
Precision = 0.9412
Recall = 1.0

F1 = 2 × (0.9412 × 1.0) / (0.9412 + 1.0)
   = 2 × 0.9412 / 1.9412
   = 1.8824 / 1.9412
   = 0.9697
   = 96.97%
```

### Interpretación Visual

```
                    F1-Score = Balance
                    
        Precision                     Recall
           │                            │
           │         ┌──────┐           │
           │         │      │           │
           └────────►│  F1  │◄──────────┘
                     │      │
                     └──────┘
                     
   Si uno es bajo, F1 es bajo.
   Solo es alto si AMBOS son altos.
```

### ¿Cuándo Usar F1-Score?

| Situación | ¿Usar F1? |
|-----------|-----------|
| Precision y Recall igual de importantes | ✅ Sí |
| Dataset desbalanceado | ✅ Sí |
| Quieres una métrica única para comparar | ✅ Sí |
| Un tipo de error es MUCHO más grave | ❌ No, usa Precision o Recall |

### F1 vs Recall en Nuestro Proyecto

```
¿Por qué priorizamos Recall sobre F1?

Porque FN (caída no detectada) >> FP (falsa alarma)

Un F1 alto con Recall bajo es INACEPTABLE:
- F1 = 95% pero Recall = 85%
- Significa que 15% de caídas no se detectan
- En una fábrica con 1000 caídas/año = 150 sin detectar
- Potencialmente 150 trabajadores sin ayuda

Preferimos:
- F1 = 90% pero Recall = 100%
- Más falsas alarmas, pero CERO caídas perdidas
```

---

## 8. Métrica 5: AUC-ROC

### Conceptos Previos

#### Umbral de Decisión (Threshold)

El modelo no dice directamente "Caída" o "No caída". Da una **probabilidad**:

```
Modelo: "Esta secuencia tiene 73% probabilidad de ser caída"

¿Es caída o no? Depende del UMBRAL:

Si umbral = 0.5:  73% > 50% → Caída ✓
Si umbral = 0.8:  73% < 80% → No caída ✗
Si umbral = 0.3:  73% > 30% → Caída ✓
```

#### La Curva ROC

ROC = **R**eceiver **O**perating **C**haracteristic

Es un gráfico que muestra cómo cambia el rendimiento del modelo al variar el umbral.

```
Ejes:
- X: FPR (False Positive Rate) = FP / (FP + TN) = Tasa de falsas alarmas
- Y: TPR (True Positive Rate) = TP / (TP + FN) = Recall

        TPR (Recall)
        │
    1.0 ┤        ╭────────────
        │       ╱
        │      ╱
        │     ╱   ← Curva ROC
        │    ╱
    0.5 ┤   ╱
        │  ╱
        │ ╱
        │╱............  ← Línea diagonal = modelo aleatorio
    0.0 ┼────────────────────
        0        0.5        1.0
                 FPR
```

### ¿Qué es AUC?

AUC = **A**rea **U**nder the **C**urve

Es el **área bajo la curva ROC**.

```
        TPR
        │
    1.0 ┤████████████████████
        │███████████████████╱
        │██████████████████╱
        │█████████████████╱  ← Área sombreada = AUC
        │████████████████╱
        │███████████████╱
        │██████████████╱
        │█████████████╱
    0.0 ┼────────────────────
        0                  1.0
                 FPR
```

### Interpretación de AUC

| AUC | Significado |
|-----|-------------|
| 1.0 | Modelo PERFECTO - separa las clases completamente |
| 0.9-1.0 | Excelente |
| 0.8-0.9 | Bueno |
| 0.7-0.8 | Aceptable |
| 0.5-0.7 | Pobre |
| 0.5 | No mejor que lanzar una moneda |
| < 0.5 | Peor que aleatorio (¡invertir predicciones!) |

### Ejemplo: Nuestros Modelos

```
Random Forest: AUC = 0.9896 (Excelente)
LSTM:          AUC = 1.0000 (PERFECTO)
Transformer:   AUC = 1.0000 (PERFECTO)
```

**AUC = 1.0 significa:** Existe un umbral donde el modelo separa PERFECTAMENTE las caídas de las no caídas.

### ¿Por Qué Usar AUC?

1. **Independiente del umbral:** Evalúa el modelo en general, no para un umbral específico
2. **Funciona con desbalance:** No se ve afectada por clases desbalanceadas
3. **Comparable:** Fácil comparar modelos diferentes

### Limitación de AUC

```
AUC mide la CAPACIDAD de separar clases, pero no dice:
- Qué umbral usar en producción
- Cuál es el balance óptimo Precision/Recall

Por eso también usamos otras métricas.
```

---

## 9. ¿Cuándo Usar Cada Métrica?

### Tabla de Decisión

| Situación | Métrica Principal | Razón |
|-----------|-------------------|-------|
| Dataset balanceado, errores equivalentes | **Accuracy** | Simple y efectiva |
| Dataset desbalanceado | **F1-Score** o **AUC** | Accuracy engañosa |
| FN es CRÍTICO (seguridad, medicina) | **Recall** | Minimizar eventos perdidos |
| FP es CRÍTICO (spam, justicia) | **Precision** | Minimizar falsas acusaciones |
| Quieres comparar modelos general | **AUC-ROC** | Independiente del umbral |
| Balance entre P y R igualmente importante | **F1-Score** | Media armónica |

### Ejemplos por Dominio

```
╔════════════════════════════════════════════════════════════════════════════╗
║ DOMINIO              │ MÉTRICA CLAVE │ RAZÓN                               ║
╠══════════════════════╪═══════════════╪═════════════════════════════════════╣
║ Detección de cáncer  │ Recall        │ FN = Paciente muere                 ║
║ Filtro de spam       │ Precision     │ FP = Email importante perdido       ║
║ Reconocimiento facial│ Accuracy      │ Ambos errores son malos             ║
║ Detección de fraude  │ Recall + F1   │ FN = Pérdida económica grande       ║
║ Recomendaciones      │ Precision     │ FP = Usuario pierde confianza       ║
║ 🛡️ CAÍDAS INDUSTRIAL │ RECALL        │ FN = Trabajador muere               ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 10. ¿Por Qué Recall es Crítico en Nuestro Caso?

### El Análisis de Costo-Beneficio

```
Costo de un FALSE POSITIVE (Falsa Alarma):
├── Se activa alarma innecesaria
├── Personal de seguridad acude
├── Interrupción de trabajo: ~15 minutos
├── Costo estimado: $50 - $200
└── Molestia, pero NADIE RESULTA HERIDO

Costo de un FALSE NEGATIVE (Caída No Detectada):
├── Trabajador en el suelo sin ayuda
├── Posibles lesiones se agravan
├── "Golden Hour" perdida en emergencias médicas
├── Costo si hay lesión grave: $50,000 - $500,000
├── Costo si hay muerte: INCALCULABLE + demandas
└── Consecuencia: VIDA HUMANA EN RIESGO
```

### Comparación Matemática

```
Ratio de Gravedad:

Costo FN     $100,000 (promedio lesión grave)
────────── = ──────────────────────────────── = 1000:1
Costo FP     $100 (promedio falsa alarma)

Un FN es ~1000 veces peor que un FP.
Por lo tanto, MINIMIZAR FN es la prioridad absoluta.
RECALL mide exactamente esto.
```

### El Objetivo: Recall = 100%

```
Recall = 100%  ⟺  FN = 0  ⟺  Cero caídas perdidas

Este es nuestro objetivo y LO LOGRAMOS con LSTM y Transformer.
```

### ¿Qué Sacrificamos por 100% Recall?

```
Random Forest Balanceado:
- Recall: 94.89%
- Precision: 92.94%
- Falsas alarmas: ~7%

LSTM:
- Recall: 100% ✓
- Precision: 94.12%
- Falsas alarmas: ~6%

¡En realidad no sacrificamos casi nada!
La precision incluso MEJORÓ ligeramente.
```

---

## 11. Trade-offs Entre Métricas

### El Trade-off Fundamental: Precision vs Recall

```
No puedes maximizar ambas simultáneamente.
Al subir una, la otra tiende a bajar.

           Precision
              ▲
              │     ╲
              │      ╲
              │       ╲
         100% │        ╲
              │         ╲
              │          ╲
              │           ╲
              │            ╲
              │             ╲
              └──────────────────► Recall
                           100%
                           
Cada punto en la curva = un umbral diferente
```

### Ajustando el Umbral

```
Umbral BAJO (ej: 0.3):
├── Más fácil declarar "Caída"
├── Recall ALTO (detectamos más)
├── Precision BAJA (más falsas alarmas)
└── Menos FN, más FP

Umbral ALTO (ej: 0.8):
├── Más difícil declarar "Caída"
├── Recall BAJO (detectamos menos)
├── Precision ALTA (menos falsas alarmas)
└── Más FN, menos FP
```

### ¿Cómo Elegir el Umbral?

Depende del **costo relativo de los errores**:

```
Si FN >> FP (nuestro caso):
→ Usar umbral BAJO
→ Priorizar Recall

Si FP >> FN (ej: filtro de spam):
→ Usar umbral ALTO
→ Priorizar Precision
```

### Curva Precision-Recall

```
        Precision
              ▲
         100% │╲
              │ ╲
              │  ╲
              │   ╲
              │    ╲
              │     ╲
              │      ╲
              │       ╲________
              │               ╲
              └────────────────────► Recall
                             100%
                             
Área bajo esta curva = AP (Average Precision)
```

---

## 12. Preguntas Frecuentes del MIT

### P1: "¿Por qué no usaron solo Accuracy?"

**Respuesta:**
> "Accuracy puede ser engañosa con datasets desbalanceados. Nuestro dataset original tenía ratio 5.5:1 (ADL vs Caídas). Un modelo que siempre predice 'No Caída' tendría ~85% accuracy pero no detectaría ninguna caída. En seguridad industrial, donde un False Negative puede significar una vida perdida, el Recall es la métrica crítica."

### P2: "¿Cómo justifican priorizar Recall sobre Precision?"

**Respuesta:**
> "Realizamos un análisis de costo-beneficio. El costo de un False Negative (caída no detectada) es del orden de $100,000 o más si resulta en lesiones graves, sin mencionar el valor incalculable de una vida humana. El costo de un False Positive (falsa alarma) es aproximadamente $100 en tiempo perdido. Con un ratio de gravedad de 1000:1, minimizar FN (maximizar Recall) es la prioridad obvia."

### P3: "¿Por qué su AUC es 1.0? ¿No es eso sospechoso?"

**Respuesta:**
> "Un AUC de 1.0 en el test set puede indicar tres cosas: (1) overfitting, (2) data leakage, o (3) que el problema es resuelble con los features disponibles. En nuestro caso, creemos que es lo tercero porque: (a) usamos train/test split estratificado, (b) las secuencias de train y test provienen de videos diferentes, y (c) la combinación de features temporales (posición + velocidad + aceleración) proporciona información suficiente para distinguir caídas de ADL. Lo validaremos con videos completamente nuevos en la demo."

### P4: "¿Qué es el F1-Score y por qué lo reportan?"

**Respuesta:**
> "F1-Score es la media armónica de Precision y Recall. La reportamos porque es una métrica estándar que permite comparar modelos cuando ambas métricas importan. Sin embargo, para nuestro caso específico, priorizamos Recall porque los False Negatives tienen consecuencias mucho más graves que los False Positives."

### P5: "¿Cuál es la diferencia entre Recall y Sensitivity?"

**Respuesta:**
> "Son exactamente la misma métrica. 'Recall' es el término usado en Machine Learning e Information Retrieval. 'Sensitivity' o 'Sensibilidad' es el término usado en estadística y medicina. Ambos miden la proporción de positivos reales que fueron correctamente identificados: TP / (TP + FN)."

### P6: "Si el Recall es lo más importante, ¿por qué no poner el umbral en 0?"

**Respuesta:**
> "Con umbral = 0, el modelo diría 'Caída' para todo, logrando 100% Recall pero 0% Precision (todas serían falsas alarmas excepto las caídas reales). Esto causaría 'fatiga de alarmas' donde los operadores ignoran las alertas. Buscamos el máximo Recall posible manteniendo una Precision aceptable (>90%). Nuestro LSTM logra Recall=100% con Precision=94%, un excelente balance."

### P7: "¿Cómo se calcula la Matriz de Confusión para multiclase?"

**Respuesta:**
> "Para multiclase, la matriz se extiende a NxN donde N es el número de clases. Cada celda (i,j) muestra cuántas muestras de clase i fueron predichas como clase j. La diagonal principal contiene las predicciones correctas. Sin embargo, nuestro problema es binario (Caída vs ADL), así que usamos la matriz 2x2 estándar."

---

## 📌 Resumen Final

### Las 5 Métricas en Una Tabla

| Métrica | Fórmula | Pregunta que Responde | Cuándo Usarla |
|---------|---------|----------------------|---------------|
| **Accuracy** | (TP+TN)/(Total) | ¿Qué % acerté en general? | Dataset balanceado |
| **Precision** | TP/(TP+FP) | ¿Qué % de mis alarmas son reales? | FP es grave |
| **Recall** | TP/(TP+FN) | ¿Qué % de caídas detecté? | FN es grave |
| **F1-Score** | 2×P×R/(P+R) | ¿Cuál es el balance P-R? | Ambos importan igual |
| **AUC-ROC** | Área bajo ROC | ¿Qué tan bien separo las clases? | Comparar modelos |

### Para SafeGuard Vision AI

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   MÉTRICA DETERMINANTE: RECALL                                            ║
║                                                                           ║
║   Razón: FN (caída no detectada) puede significar una muerte             ║
║                                                                           ║
║   Objetivo: Recall = 100% (LOGRADO con LSTM y Transformer)               ║
║                                                                           ║
║   Validación: También mantuvimos Precision > 94%                          ║
║               para evitar fatiga de alarmas                               ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

*Documento preparado para MIT Global Teaching Labs 2025*

*SafeGuard Vision AI - Industry 4.0 Zero Accident Initiative*
