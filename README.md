# Ciencia de Datos e IA — Portafolio Personal

Repositorio de scripts y notebooks de **Ciencia de Datos e Inteligencia Artificial** organizados por área temática.
Cada carpeta contiene los scripts y su propio README con descripción detallada.

**Autor:** Christian Cajusol | UTEC | christian.cajusol@utec.edu.pe

---

## Áreas

| # | Área | Carpeta | Notebooks | Descripción | Stack |
|---|------|---------|-----------|-------------|-------|
| 1 | Exploratory Data Analysis | [`EDA/`](EDA/) | 4 | EDA, transformaciones, outliers multivariados, calidad de datos, feature engineering y Visual Data Discovery | Python, Pandas, Seaborn, Scipy, Scikit-learn, PySpark, Databricks |

---

## Detalle de notebooks por área

### EDA

| # | Notebook | Dataset(s) | Temas |
|---|----------|-----------|-------|
| 1 | [EDA.ipynb](EDA/EDA.ipynb) | Loan Prediction | EDA completo, Spark SQL, imputación, encoding |
| 2 | [EDA_ejercicio_practico_Boston_Housing.ipynb](EDA/EDA_ejercicio_practico_Boston_Housing.ipynb) | Boston Housing, Brazil Tourism, Salaries SF | EDA, missingness (KNN/MICE/Regresión), outliers univariados robustos |
| 3 | [Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb](EDA/Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb) | Salaries SF, Ames House Prices, Solicitudes crédito | Transformaciones (Box-Cox, Yeo-Johnson), outliers multivariados (Mahalanobis/IF/LOF/DBSCAN), 10 reglas de calidad |
| 4 | [Feature_Engineering_Data_Visualization.ipynb](EDA/Feature_Engineering_Data_Visualization.ipynb) | Ames Housing, Telco Customer Churn (5 CSVs) | 6 técnicas de feature engineering, 5 visualizaciones ejecutivas, análisis de churn con recomendaciones de negocio |

---

## Estructura del repositorio

```
Ciencia_Datos_IA/
│
├── EDA/
│   ├── EDA.ipynb
│   ├── EDA_ejercicio_practico_Boston_Housing.ipynb
│   ├── Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb
│   ├── Feature_Engineering_Data_Visualization.ipynb
│   ├── train.csv
│   └── README.md
│
├── CLAUDE.md
└── README.md
```

---

## Cómo navegar el repositorio

Cada área tiene su propio `README.md` con:
- **Flujo general** del proceso (diagrama paso a paso)
- **Detalle por notebook**: secciones, datasets, hallazgos clave
- **Análisis de calidad del código**: observaciones y mejoras posibles
- **Stack tecnológico** y datasets utilizados
