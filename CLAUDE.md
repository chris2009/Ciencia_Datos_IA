# CLAUDE.md — Contexto de trabajo para este repositorio

## Propósito del repositorio
Portafolio personal de scripts de **Ciencia de Datos e Inteligencia Artificial** organizado por área temática.
Autor: Christian Cajusol — christian.cajusol@utec.edu.pe — UTEC

---

## Convenciones de estructura

Cada área temática vive en su propia carpeta con:
- El script principal (`.ipynb` o `.py`)
- Un `README.md` propio explicando dataset, estructura y hallazgos

---

## Reglas de trabajo — AUTOMÁTICAS, sin necesidad de que el usuario las pida

Tras **cualquier** cambio en este repo, ejecutar siempre estos 4 pasos en el mismo turno:

1. Actualizar **este archivo** (`CLAUDE.md`) — tabla de áreas, contadores
2. Actualizar el **`README.md` raíz** — índice y tablas de notebooks
3. Actualizar **`memory/project_context.md`** — estado actual de áreas
4. Hacer un **commit descriptivo** — qué se agregó, qué algoritmos/datasets, por qué

**Estas 4 acciones son el Definition of Done de cualquier tarea. No hay tarea terminada sin ellas.**

---

## Áreas actuales

| Área | Carpeta | Notebooks | Descripción | Stack |
|------|---------|-----------|-------------|-------|
| EDA | `EDA/` | 4 | EDA, transformaciones, outliers multivariados, calidad de datos, feature engineering, Visual Data Discovery | Python, Pandas, Seaborn, Scipy, Scikit-learn, PySpark, Databricks |
| Machine Learning | `Machine_Learning/` | 14 | Modelos supervisados (regresión, clasificación) y no supervisados (clustering, reducción de dimensionalidad), todos from scratch + sklearn | Python, NumPy, Matplotlib, Scikit-learn, imbalanced-learn, UMAP, TensorFlow |
| MISTI — MIT Intensive | `MISTI/` | 13 notebooks + 16 scripts | Programa MIT-Peru: fundamentos NumPy/Pandas, EDA, ML clásico, CNN, NLP, RL y proyecto final SAFEGUARD (detección de caídas con BlazePose + Random Forest + LSTM + Transformer, Recall=100%) | Python, TensorFlow, MediaPipe, OpenCV, scikit-learn |

---

## Stack tecnológico principal
- **Plataforma:** Databricks
- **Lenguajes:** Python 3.x, Spark SQL
- **Librerías:** pandas, numpy, matplotlib, seaborn, scipy, PySpark
- **Control de versiones:** Git / GitHub (rama principal: `main`)
