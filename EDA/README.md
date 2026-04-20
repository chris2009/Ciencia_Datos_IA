# Exploratory Data Analysis (EDA)

Colección de notebooks de Análisis Exploratorio de Datos abarcando distintos datasets, técnicas y niveles de profundidad.
Todos los notebooks pertenecen al curso **MCD8009: Data Discovery** de la Maestría en Ciencia de Datos — UTEC.

---

## Notebooks incluidos

| # | Archivo | Dataset | Tema principal |
|---|---------|---------|----------------|
| 1 | [EDA.ipynb](EDA.ipynb) | Loan Prediction | EDA completo en Databricks + Spark SQL |
| 2 | [EDA_ejercicio_practico_Boston_Housing.ipynb](EDA_ejercicio_practico_Boston_Housing.ipynb) | Boston Housing + Brazil Tourism + Salaries | EDA, missingness y outliers univariados |
| 3 | [Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb](Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb) | Salaries SF + Ames House Prices + Solicitudes de crédito | Transformaciones, outliers multivariados, calidad de datos |
| 4 | [Feature_Engineering_Data_Visualization.ipynb](Feature_Engineering_Data_Visualization.ipynb) | Ames Housing + Telco Customer Churn (5 CSVs) | Feature engineering, Visual Data Discovery, análisis ejecutivo de churn |

---

## Flujo general del EDA (presente en todos los notebooks)

```
RAW DATA
    │
    ▼
1. CARGA Y CONFIGURACIÓN
   └─ Importar librerías, leer dataset (CSV / Spark / OpenML)
   └─ Vista previa: shape, dtypes, head()
    │
    ▼
2. EXPLORACIÓN INICIAL
   └─ Clasificar variables: numéricas vs categóricas
   └─ Estadísticas descriptivas (describe, value_counts)
    │
    ▼
3. CALIDAD DE DATOS
   ├─ Duplicados → eliminar
   ├─ Valores faltantes → tabla de % por columna + visualización
   └─ Reglas de negocio (dominio, rangos, formatos, coherencia)
    │
    ▼
4. ANÁLISIS UNIVARIADO
   ├─ Numéricas: histograma + KDE, boxplot, skewness
   ├─ Categóricas: barras, frecuencias relativas
   └─ Outliers: IQR, Z-score, MAD, IQR asimétrico
    │
    ▼
5. ANÁLISIS BIVARIADO / MULTIVARIADO
   ├─ Correlación: heatmap, pairplot
   ├─ Categórica vs target: crosstab + barras apiladas
   └─ Outliers multivariados: Mahalanobis, Isolation Forest, LOF, DBSCAN
    │
    ▼
6. TRANSFORMACIONES (cuando hay sesgo)
   ├─ Logarítmica  ─ reduce cola derecha
   ├─ Raíz cuadrada ─ corrección moderada
   ├─ Box-Cox      ─ encuentra λ óptimo (solo valores > 0)
   └─ Yeo-Johnson  ─ como Box-Cox pero acepta negativos y ceros
    │
    ▼
7. TRATAMIENTO DE DATOS
   ├─ Imputación: moda (categ.), mediana (num.), regresión, KNN, MICE
   ├─ Codificación: label encoding / mapeo manual
   └─ Feature engineering: nuevas variables derivadas (TotalIncome, IMC)
    │
    ▼
8. CONCLUSIONES Y RECOMENDACIONES
   └─ Hallazgos clave + próximos pasos para modelado
```

---

## Detalle por notebook

---

### 1. `EDA.ipynb` — Loan Prediction Dataset (Databricks)

**Contexto:** EDA completo sobre solicitudes de préstamos bancarios, ejecutado en Databricks con PySpark.

**Secciones:**
1. Carga desde Spark y conversión a Pandas
2. Exploración inicial (shape, dtypes, clasificación de variables)
3. Calidad de datos: duplicados (7 eliminados) + valores faltantes (1.87%)
4. Análisis univariado: estadísticas descriptivas, distribuciones, outliers IQR
5. Análisis multivariado: correlación, pairplot, scatter plots, variable objetivo
6. Tratamiento: imputación moda/mediana + label encoding
7. Spark SQL: 5 consultas analíticas

**Dataset:** 614 registros × 13 variables | Target: `Loan_Status` (Y/N)

**Hallazgos clave:**
- `Credit_History` es el predictor más importante
- Dataset desbalanceado (~68% aprobados)
- Outliers severos en `ApplicantIncome` (8.1%) y `Loan_Amount_Term` (14.7%)

---

### 2. `EDA_ejercicio_practico_Boston_Housing.ipynb` — Lab 2

**Contexto:** Laboratorio académico con 4 partes sobre distintos datasets y técnicas.

#### Parte 1 — EDA (Boston Housing)
- Pairplot de todas las variables con conclusiones interpretativas
- Media, mediana, desviación estándar de precios (`MEDV`)
- Proporción de viviendas cerca del río Charles (`CHAS`)
- Comparación de precios por `CHAS` (boxplot)
- Correlación de `RM` con precio → ~0.70 (correlación fuerte positiva)
- Variables más correlacionadas positivamente: `TAX` ↔ `RAD`
- Variables más correlacionadas negativamente: `NOX` ↔ `DIS`

#### Parte 2 — Missingness conceptual
- Reflexión sobre riesgos de imputar datos MNAR con la media
- Diferencia entre MAR, MCAR y MNAR

#### Parte 3 — Missingness aplicado (Brazil Tourism)
- Visualización de valores faltantes con `missingno`
- Comparación de 4 métodos de imputación en variable `Income`:
  - Media / Mediana / Moda → generan pico artificial
  - Regresión lineal → preserva mejor la distribución
  - KNN → interpola por vecinos similares
  - MICE → imputación múltiple por ecuaciones encadenadas
- Tratamiento correcto de `Logged_income` como variable derivada

#### Parte 4 — Outliers univariados (Salaries SF)
- Feature engineering: cálculo de IMC desde Peso y Altura
- Análisis univariado del IMC (boxplot + histograma)
- Comparación de métodos estándar vs robustos:
  - **Estándar:** IQR 1.5× + Z-score
  - **Robustos:** IQR asimétrico (factor diferencial en cola derecha) + MAD

---

### 3. `Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb` — Lab 3

**Contexto:** Laboratorio con 3 partes sobre transformaciones, outliers multivariados y calidad de datos.

#### Parte 1 — Transformaciones (Salaries SF 2014, Full Time)
Aplicación de 4 transformaciones sobre `TotalPayBenefits` (sesgo derecho pronunciado):

| Transformación | λ estimado | Skewness resultante | Escala |
|----------------|-----------|---------------------|--------|
| Original | — | alta (sesgo derecho) | ~138K |
| Logarítmica | — | baja | ~12 |
| Raíz cuadrada | — | moderada | ~370 |
| Box-Cox | −0.327 | muy baja | ~3 |
| Yeo-Johnson | −0.327 | muy baja | ~3 |

- Box-Cox y Yeo-Johnson producen resultados casi idénticos porque todos los valores son positivos
- La raíz cuadrada es la más interpretable pero la que menos corrige

#### Parte 2 — Outliers multivariados (Ames House Prices: LotArea × GrLivArea)
Comparación de 4 métodos con visualización de scatter (azul=normal, rojo=outlier):

| Método | Parámetros | Outliers detectados |
|--------|-----------|---------------------|
| Mahalanobis | α=0.025, χ²(df=2) | 33 |
| Isolation Forest | contamination=0.05 | 73 |
| LOF | n_neighbors=20 | 43 |
| DBSCAN | eps=350, min_samples=5 | 87 |

- Mahalanobis detecta los más extremos basándose en la distribución conjunta
- DBSCAN es el más agresivo (detecta puntos en zonas de baja densidad)
- Los outliers visualmente obvios (lotes enormes con área habitable baja) son capturados por todos los métodos

#### Parte 3 — Calidad de datos (Solicitudes de crédito, n=200)
Implementación de 10 reglas de calidad con reporte de incumplimientos:

| Regla | Descripción | Incumplimientos |
|-------|-------------|-----------------|
| R1 | DNI: exactamente 8 dígitos numéricos | 5 (2.5%) |
| R2 | Email: formato `usuario@dominio.ext` | 4 (2.0%) |
| R3 | Teléfono: 9 dígitos, inicia en 9 | 3 (1.5%) |
| R4 | Edad: entre 18 y 80 | 3 (1.5%) |
| R5 | Ingreso mensual > 0 | 2 (1.0%) |
| R6 | Tipo empleo en dominio válido | 2 (1.0%) |
| R7 | Empresa obligatoria si `dependiente` | 1 (0.5%) |
| R8 | Fecha resolución ≥ fecha solicitud | 3 (1.5%) |
| R9 | Edad coherente con fecha nacimiento (±1 año) | 3 (1.5%) |
| R10 | Cuota ≤ 30% del ingreso (norma SBS Perú) | 50 (25.0%) |

---

## Análisis de calidad del código

### `EDA.ipynb` ✅ Sólido
- Código bien organizado con secciones claramente delimitadas
- Funciones reutilizables (`analizar_valores_faltantes`, `detectar_outliers_iqr`)
- Buenas prácticas: copia del DataFrame antes de tratamiento, verificación post-imputación
- **Observación:** `Credit_History` es clasificada como numérica por su dtype `float64`, pero es una variable binaria — debería analizarse como categórica en el univariado

### `EDA_ejercicio_practico_Boston_Housing.ipynb` ✅ Bueno con observaciones
- Análisis completo y bien comentado
- Comparación de múltiples métodos de imputación con justificación
- **Observación 1:** La función `leer_csv_inteligente` viene de un trabajo anterior — bien reutilizada, pero el notebook no es autocontenido sin esa función
- **Observación 2:** Ruta hardcodeada `/content/boston.csv` (Google Colab) — no portable
- **Observación 3:** El dataset Boston Housing fue retirado de `sklearn` en v1.2 por sesgos estadísticos; se recomienda usar `fetch_openml("boston")` o el dataset de Ames

### `Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb` ✅ Muy bueno
- El más completo y mejor documentado de los tres
- Comentarios en cada bloque explicando el *por qué*, no solo el *qué*
- Implementación correcta de los 4 métodos de outliers multivariados
- Las 10 reglas de calidad cubren casos reales de negocio (norma SBS, formato peruano)
- **Observación:** R10 marca `SOL-00026` (ingreso negativo = -800) como violación de la regla de cuota. Matemáticamente es correcto (`-51.70 > 0.30 × -800 = -240`), pero semánticamente ese registro ya viola R5. Sería más robusto excluir de R10 los registros con ingreso ≤ 0

### `Feature_Engineering_Data_Visualization.ipynb` ✅ Excelente — el más maduro
- Paleta de colores constante definida al inicio (`CHURN_COLOR`, `RETAIN_COLOR`, `NEUTRAL_COLOR`) — coherencia visual total
- Visualizaciones ejecutivas con etiquetas directas en barras, sin leyendas innecesarias
- `observed=True` en todos los `groupby` con categóricas — buena práctica pandas
- `sample(min(2000, len(df)), random_state=42)` para reproducibilidad en scatter geográfico
- Merge encadenado de 5 CSVs limpio y sin redundancias (elimina columnas `Count` duplicadas)
- Feature engineering con justificación de negocio en cada variable
- Recomendaciones accionables sustentadas en evidencia de los gráficos
- **Observación 1:** Min-Max scaling implementado manualmente — correcto, pero `MinMaxScaler` de sklearn es más seguro ante edge cases (división por cero si min == max)
- **Observación 2:** `MUTED_COLOR = '#C0C0C0'` se redefine dentro de la Figura 1 en lugar de usar la constante global — inconsistencia menor
- **Observación 3:** `warnings.filterwarnings('ignore')` global — suprime todas las advertencias, puede ocultar problemas reales en producción

---

### 4. `Feature_Engineering_Data_Visualization.ipynb` — Lab 4

**Contexto:** Laboratorio sobre Feature Engineering y Visual Data Discovery. Trabajo colaborativo (equipo). Curso MCD8009 — UTEC.

#### Parte 1 — Feature Engineering (Ames Housing)
6 nuevas variables creadas sobre el dataset de viviendas de Ames:

| # | Variable creada | Tipo | Técnica | Variable fuente |
|---|-----------------|------|---------|-----------------|
| 1 | `CentralAir_Enc` | Encoding binario | `.map({'Y':1,'N':0})` | `Central Air` |
| 2 | `LotArea_Scaled` | Escalado | Min-Max manual | `Lot Area` |
| 3 | `YearBuilt_Bin` | Discretización | `pd.cut()` en 4 bins | `Year Built` |
| 4 | `TotalSF` | Variable compuesta | Suma de 3 áreas | `1st Flr SF + 2nd Flr SF + Total Bsmt SF` |
| 5 | `HouseAgeAtSale` | Variable temporal | Diferencia de años | `Yr Sold - Year Built` |
| 6 | `KitchenQual_Ord` | Encoding ordinal | Mapeo con orden lógico | `Kitchen Qual` (Ex/Gd/TA/Fa/Po → 5…1) |

**Visualizaciones de los nuevos features:**
- Scatter: `TotalSF` vs `SalePrice` → correlación positiva fuerte (más área = mayor precio)
- Boxplot: `YearBuilt_Bin` vs `SalePrice` → viviendas nuevas tienen precios medianos más altos y mayor variabilidad
- Countplot: distribución de `KitchenQual_Ord` → concentración en calidad media-alta (3-4)

#### Parte 2 — Teoría de Visualización
Respuestas conceptuales sobre:
- Sobrecarga visual y cuándo se vuelve contraproducente
- Elección de tipo de gráfico según pregunta analítica
- Principios de visualización efectiva (Tufte, data-ink ratio)

#### Parte 3 — Visual Data Discovery: Telco Customer Churn (trabajo integrador)

**Fuente de datos:** 5 archivos CSV fusionados en un solo DataFrame de 7,043 clientes × múltiples variables.

| Archivo CSV | Contenido |
|-------------|-----------|
| `Telco_customer_churn_demographics.csv` | Edad, género, estado civil |
| `Telco_customer_churn_services.csv` | Contratos, servicios contratados |
| `Telco_customer_churn_status.csv` | Churn label, razón, categoría |
| `Telco_customer_churn_location.csv` | Ciudad, latitud, longitud, ZIP |
| `Telco_customer_churn_population.csv` | Población por código postal |

**Feature engineering sobre el dataset integrado:**
- `Tenure_Segment`: `pd.cut()` en 5 intervalos (0-6m, 7-12m, 13-24m, 25-48m, 48+m)
- `Pop_Segment`: `pd.qcut()` en 4 cuartiles de densidad poblacional

**EDA previo a visualizaciones:**
- Distribuciones de Age, Tenure, Monthly Charge, Total Charges por estado de churn
- Top 10 razones de churn + categorías
- Tasa de churn por tipo de contrato y tipo de internet
- Efecto de 6 servicios adicionales sobre la tasa de churn
- Análisis geográfico: ciudades con mayor concentración de clientes y churn

**5 visualizaciones ejecutivas** (paleta: rojo=#E74C3C, verde=#2ECC71, gris=#C0C0C0):

| # | Título | Insight principal |
|---|--------|-------------------|
| 1 | Tasa de Churn por Tipo de Contrato | Mes-a-Mes: 45.8% churn vs 2.5% en contratos de 2 años |
| 2 | Antigüedad del Cliente vs Churn | Mayor riesgo en primeros 6 meses; >48m casi no churnan |
| 3 | Categorías y Razones de Churn | Competencia (precio/oferta) > 40% de los churns |
| 4 | Servicios como Anclas de Retención | Sin Online Security/Support: churn 40-50%; con servicio: ~15% |
| 5 | Densidad Poblacional × Churn | Zonas media-baja y alta densidad: mayor tasa de abandono |

**3 recomendaciones de negocio sustentadas:**
1. Migrar clientes mes-a-mes a contratos largos con incentivos
2. Bundle de servicios de seguridad/soporte para clientes de los primeros 6 meses
3. Programa de contraoferta ante la competencia (precio + datos)

---

## Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| Python 3.x | Lenguaje base |
| pandas, numpy | Manipulación de datos |
| matplotlib, seaborn | Visualización |
| scipy.stats | Pruebas estadísticas, Box-Cox |
| scikit-learn | Imputación (KNN, MICE), outliers (IsolationForest, LOF, DBSCAN) |
| missingno | Visualización de valores faltantes |
| PySpark / Databricks | Procesamiento distribuido (notebook 1) |

---

## Datasets utilizados

| Dataset | Fuente | Registros | Notebooks |
|---------|--------|-----------|-----------|
| Loan Prediction | CSV local | 614 | Lab 1 |
| Boston Housing | CSV local / OpenML | ~506 | Lab 2 |
| Brazil Tourism | OpenML | — | Lab 2 |
| San Francisco Salaries | CSV local | 148,654 | Lab 2, Lab 3 |
| Ames House Prices | OpenML / CSV local | 1,460 | Lab 3, Lab 4 |
| Solicitudes de crédito | CSV local | 200 | Lab 3 |
| Telco Customer Churn | CSV local (5 archivos) | 7,043 | Lab 4 |
