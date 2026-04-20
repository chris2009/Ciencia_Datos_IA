# Ciencia de Datos e IA — Portafolio Personal

Repositorio de scripts y notebooks de **Ciencia de Datos e Inteligencia Artificial** organizados por área temática.
Cada carpeta contiene los scripts y su propio README con descripción detallada.

**Autor:** Christian Cajusol | UTEC | christian.cajusol@utec.edu.pe

---

## Áreas

| # | Área | Carpeta | Notebooks | Descripción | Stack |
|---|------|---------|-----------|-------------|-------|
| 1 | Exploratory Data Analysis | [`EDA/`](EDA/) | 4 | EDA, transformaciones, outliers multivariados, calidad de datos, feature engineering y Visual Data Discovery | Python, Pandas, Seaborn, Scipy, Scikit-learn, PySpark, Databricks |
| 2 | Machine Learning | [`Machine_Learning/`](Machine_Learning/) | 14 | Modelos supervisados y no supervisados implementados from scratch con NumPy, comparados contra scikit-learn | Python, NumPy, Matplotlib, Scikit-learn, imbalanced-learn, UMAP, TensorFlow |

---

## Detalle de notebooks por área

### EDA

| # | Notebook | Dataset(s) | Temas |
|---|----------|-----------|-------|
| 1 | [EDA.ipynb](EDA/EDA.ipynb) | Loan Prediction | EDA completo, Spark SQL, imputación, encoding |
| 2 | [EDA_ejercicio_practico_Boston_Housing.ipynb](EDA/EDA_ejercicio_practico_Boston_Housing.ipynb) | Boston Housing, Brazil Tourism, Salaries SF | EDA, missingness (KNN/MICE/Regresión), outliers univariados robustos |
| 3 | [Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb](EDA/Transformaciones_Valores_Atpicos_Calidad_Datos.ipynb) | Salaries SF, Ames House Prices, Solicitudes crédito | Transformaciones (Box-Cox, Yeo-Johnson), outliers multivariados, 10 reglas de calidad |
| 4 | [Feature_Engineering_Data_Visualization.ipynb](EDA/Feature_Engineering_Data_Visualization.ipynb) | Ames Housing, Telco Customer Churn | Feature engineering, 5 visualizaciones ejecutivas, análisis de churn |

### Machine Learning

| # | Notebook | Algoritmo(s) | Tipo |
|---|----------|-------------|------|
| 1 | [Session2.0_Gradient Descent](Machine_Learning/Session2.0_Gradient%20Descent.ipynb) | Gradient Descent | Optimización |
| 2 | [Session2.0_OLS](Machine_Learning/Session2.0_OLS.ipynb) | OLS + análisis de patologías | Regresión |
| 3 | [Session2.1_Linear model](Machine_Learning/Session2.1_Linear%20model.ipynb) | OLS, Ridge, Lasso, GD, Polinomial | Regresión (BandGap) |
| 4 | [Session3.0_LDA_Logistic](Machine_Learning/Session3.0_LDA_Logistic.ipynb) | LDA, Logística (Newton + SGD) | Clasificación |
| 5 | [Session3.1_SVM](Machine_Learning/Session3.1_SVM.ipynb) | SVM lineal, Kernel SVM, SVR | Clasificación / Regresión |
| 6 | [Session3.2_Random Forest](Machine_Learning/Session3.2_Random%20Forest.ipynb) | Random Forest | Clasificación |
| 7 | [Session4.0_kMeans](Machine_Learning/Session4.0_kMeans.ipynb) | K-Means, K-Means++, MiniBatch | Clustering |
| 8 | [Session4.0_kmedoids](Machine_Learning/Session4.0_kmedoids.ipynb) | PAM, CLARA | Clustering |
| 9 | [Session4.0_Mean Shift](Machine_Learning/Session4.0_Mean%20Shift.ipynb) | KDE, Mean Shift | Clustering |
| 10 | [Session4.0_EM](Machine_Learning/Session4.0_EM.ipynb) | GMM + EM, AIC/BIC | Clustering |
| 11 | [Session4.1_Agglomerative](Machine_Learning/Session4.1_Agglomerative.ipynb) | Agglomerative (4 linkages) + Dendrogramas | Clustering |
| 12 | [PCA](Machine_Learning/PCA.ipynb) | PCA from scratch | Reducción de dimensionalidad |
| 13 | [Dimensional reduction](Machine_Learning/Dimensional%20reduction.ipynb) | PCA, t-SNE, UMAP | Reducción de dimensionalidad |
| 14 | [Unbalanced](Machine_Learning/Unbalanced.ipynb) | Undersampling, Tomek Links, Oversampling | Clases desbalanceadas |

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
│   ├── Dataset/  (CSVs)
│   └── README.md
│
├── Machine_Learning/
│   ├── Session2.0_Gradient Descent.ipynb
│   ├── Session2.0_OLS.ipynb
│   ├── Session2.1_Linear model.ipynb
│   ├── Session3.0_LDA_Logistic.ipynb
│   ├── Session3.1_SVM.ipynb
│   ├── Session3.2_Random Forest.ipynb
│   ├── Session4.0_EM.ipynb
│   ├── Session4.0_Mean Shift.ipynb
│   ├── Session4.0_kMeans.ipynb
│   ├── Session4.0_kmedoids.ipynb
│   ├── Session4.1_Agglomerative.ipynb
│   ├── Dimensional reduction.ipynb
│   ├── PCA.ipynb
│   ├── Unbalanced.ipynb
│   ├── Dataset/  (BandGap, Lab1-4 folds)
│   └── README.md
│
├── CLAUDE.md
└── README.md
```

---

## Cómo navegar el repositorio

Cada área tiene su propio `README.md` con:
- **Índice de notebooks** con algoritmos, tipo y dataset
- **Detalle por notebook**: concepto central, implementación, experimentos y hallazgos
- **Análisis de calidad del código**: fortalezas y observaciones
- **Stack tecnológico** y datasets utilizados
