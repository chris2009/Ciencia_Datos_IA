# Exploratory Data Analysis (EDA) - Loan Prediction Dataset

## Descripción

Script de análisis exploratorio de datos sobre un dataset de **solicitudes de préstamos bancarios**. El objetivo es comprender la estructura del dataset, identificar problemas de calidad de datos, detectar patrones relevantes y preparar los datos para un posterior modelado predictivo de aprobación de créditos.

El análisis fue desarrollado en **Databricks** usando **PySpark + Pandas**, lo que permite combinar el poder de procesamiento distribuido de Spark con las capacidades analíticas de Python.

---

## Finalidad

> Responder la pregunta: **¿Qué factores determinan si un préstamo será aprobado o rechazado?**

El EDA sienta las bases para construir un modelo de clasificación (`Loan_Status`: Y/N) al revelar qué variables tienen mayor poder predictivo, cómo se distribuyen los datos y qué tratamiento necesitan antes del modelado.

---

## Dataset

**Archivo:** `train.csv`  
**Registros:** 614 (tras eliminación de duplicados)  
**Variables:** 13 originales + 1 derivada (`TotalIncome`)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `Loan_ID` | Categórica | Identificador único del préstamo |
| `Gender` | Categórica | Género del solicitante |
| `Married` | Categórica | Estado civil |
| `Dependents` | Categórica | Número de dependientes |
| `Education` | Categórica | Nivel educativo |
| `Self_Employed` | Categórica | Trabajador independiente |
| `ApplicantIncome` | Numérica | Ingreso del solicitante |
| `CoapplicantIncome` | Numérica | Ingreso del co-solicitante |
| `LoanAmount` | Numérica | Monto del préstamo (miles) |
| `Loan_Amount_Term` | Numérica | Plazo del préstamo (meses) |
| `Credit_History` | Binaria | Historial crediticio (1=bueno, 0=malo) |
| `Property_Area` | Categórica | Área de la propiedad (Urban/Semiurban/Rural) |
| `Loan_Status` | **Target** | Estado del préstamo — Y=Aprobado, N=Rechazado |

---

## Estructura del Notebook

El análisis está organizado en **7 secciones** más un bloque de conclusiones:

### 1. Carga y Configuración Inicial
- Importación de librerías (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`)
- Lectura del dataset desde una tabla Spark en Databricks y conversión a Pandas

### 2. Exploración Inicial de Datos
- Vista previa de los primeros registros (`head`)
- Dimensiones, tipos de datos e información general del dataset
- Clasificación automática de variables numéricas vs categóricas

### 3. Análisis de Calidad de Datos
- Detección y eliminación de **duplicados** (se encontraron 7 registros duplicados)
- Análisis de **valores faltantes** por columna con porcentaje y visualización por barras de colores (verde / naranja / rojo según severidad)
- Total de valores faltantes: **149 celdas** (1.87% del dataset)

### 4. Análisis Univariado
- Estadísticas descriptivas para variables numéricas y categóricas
- Distribución de frecuencias de variables categóricas (tablas + gráficos de barras)
- Histogramas con curva KDE y línea de media para variables numéricas
- Boxplots para visualizar la dispersión
- Detección de **outliers** usando el método IQR con resumen cuantitativo

### 5. Análisis Multivariado
- **Matriz de correlación** con heatmap para variables numéricas
- **Pairplot** por estado de préstamo (`Loan_Status`) para ver separación entre clases
- Diagramas de dispersión: Ingreso vs Monto de Préstamo (individual y total)
- Análisis de la **variable objetivo**: distribución de clases con gráfico de barras y pie chart
- Gráficos de barras apiladas para comparar tasa de aprobación por cada variable categórica

### 6. Tratamiento de Datos
- **Imputación de variables categóricas** con la moda
- **Imputación de variables numéricas** con la mediana
- **Codificación** de variables categóricas a valores numéricos usando mapeos explícitos
- Verificación de que no quedan valores faltantes tras el tratamiento

### 7. Análisis con Spark SQL
Cinco consultas SQL ejecutadas sobre una vista temporal de Spark:
- Vista general del dataset
- Conteo y porcentaje de préstamos por estado de aprobación
- Estadísticas y tasa de aprobación por área de propiedad
- Análisis cruzado por nivel educativo y estado civil
- Top 10 solicitudes con mayor monto de préstamo

---

## Principales Hallazgos

| Hallazgo | Detalle |
|----------|---------|
| Dataset desbalanceado | ~68% préstamos aprobados vs ~32% rechazados |
| Predictor más fuerte | `Credit_History` (historial crediticio) |
| Outliers severos | `ApplicantIncome` (8.1%) y `Loan_Amount_Term` (14.7%) |
| Plazo más común | 360 meses (la gran mayoría de los préstamos) |
| Área con mayor aprobación | Semiurban > Urban > Rural |
| Graduados | Mayor tasa de aprobación que no graduados |

---

## Recomendaciones para Modelado

- Aplicar **balanceo de clases** (SMOTE o undersampling) por el desbalance del target
- Considerar **transformación logarítmica** en `ApplicantIncome` y `LoanAmount`
- Priorizar `Credit_History` como feature en el modelo
- Usar la variable derivada `TotalIncome = ApplicantIncome + CoapplicantIncome`
- Evaluar el **ratio deuda/ingreso** como feature adicional

---

## Requisitos

- Python 3.x
- pandas, numpy, matplotlib, seaborn, scipy
- Apache Spark (Databricks) — para la carga inicial y las consultas SQL

---

## Autor

**christian.cajusol@utec.edu.pe**
