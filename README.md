# Ciencia de Datos e IA — Portafolio Personal

Repositorio de scripts y notebooks de **Ciencia de Datos e Inteligencia Artificial** organizados por área temática.
Cada carpeta contiene los scripts y su propio README con descripción detallada.

**Autor:** Christian Cajusol | UTEC | christian.cajusol@utec.edu.pe

---

## Áreas

| # | Área | Carpeta | Archivos | Descripción |
|---|------|---------|----------|-------------|
| 1 | Exploratory Data Analysis | [`EDA/`](EDA/) | 4 notebooks | EDA, transformaciones, outliers, calidad de datos, feature engineering, Visual Data Discovery |
| 2 | Machine Learning | [`Machine_Learning/`](Machine_Learning/) | 14 notebooks | Modelos supervisados y no supervisados implementados from scratch con NumPy + validación sklearn |
| 3 | MISTI — MIT Intensive | [`MISTI/`](MISTI/) | 13 notebooks + 16 scripts | Programa intensivo MIT-Peru: fundamentos, ML clásico, Deep Learning y proyecto final SAFEGUARD |

---

## Detalle por área

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
| 2 | [Session2.0_OLS](Machine_Learning/Session2.0_OLS.ipynb) | OLS + patologías | Regresión |
| 3 | [Session2.1_Linear model](Machine_Learning/Session2.1_Linear%20model.ipynb) | OLS, Ridge, Lasso, GD, Polynomial | Regresión (BandGap) |
| 4 | [Session3.0_LDA_Logistic](Machine_Learning/Session3.0_LDA_Logistic.ipynb) | LDA, Logística (Newton+SGD) | Clasificación |
| 5 | [Session3.1_SVM](Machine_Learning/Session3.1_SVM.ipynb) | SVM lineal/kernel, SVR | Clasificación/Regresión |
| 6 | [Session3.2_Random Forest](Machine_Learning/Session3.2_Random%20Forest.ipynb) | Random Forest | Clasificación |
| 7–11 | Sessions 4.x | K-Means, K-Medoids, Mean Shift, GMM-EM, Agglomerative | Clustering |
| 12–13 | PCA / Dimensional reduction | PCA, t-SNE, UMAP | Reducción de dim. |
| 14 | [Unbalanced](Machine_Learning/Unbalanced.ipynb) | Undersampling, Tomek, Oversampling | Clases desbalanceadas |

### MISTI — MIT Intensive Program

| Módulo | Contenido |
|--------|-----------|
| **Fundamentos** | 6 notebooks: NumPy arrays vectorizados, Pandas Series/DataFrame, missing data, operaciones, GroupBy/merge, I/O |
| **EDA Semana 0** | 2 ejercicios resueltos: E-commerce (10K transacciones), SF Salaries (148K registros) |
| **Semana 1** | 7 notebooks: California Housing (EDA + regresión + validación), Kobe shots (feature engineering + clasificación), Bike Sharing (gradient descent), intro redes neuronales, reducción de dimensionalidad |
| **Semana 2** | 6 notebooks: CNN (arquitectura + feature maps), data augmentation para imágenes, validación de redes neuronales, Word Embeddings (Word2Vec/GloVe), Reinforcement Learning (Q-learning) |
| **Proyecto SAFEGUARD** | Sistema completo de detección de caídas: 4 modelos (RF v1/v2 → LSTM → Transformer), pipeline BlazePose + feature engineering, demo en tiempo real, Recall=100% |

---

## Estructura del repositorio

```
Ciencia_Datos_IA/
│
├── EDA/
│   ├── *.ipynb  (4 notebooks)
│   ├── *.csv    (datasets)
│   └── README.md
│
├── Machine_Learning/
│   ├── Session*.ipynb  (14 notebooks)
│   ├── *.ipynb         (PCA, Dim. Reduction, Unbalanced)
│   ├── Dataset/        (BandGap, Lab1-4 folds)
│   └── README.md
│
├── MISTI/
│   ├── Fundamentos/    (6 notebooks NumPy + Pandas)
│   ├── EDA/            (2 ejercicios resueltos)
│   ├── week_1/         (7 notebooks ML clásico)
│   ├── week_2/         (6 notebooks Deep Learning)
│   ├── Proyecto_final_SAFEGUARD/  (16 scripts Python + 7 docs .md)
│   └── README.md
│
├── CLAUDE.md
└── README.md
```

---

## Tecnologías utilizadas

| Categoría | Herramientas |
|-----------|-------------|
| Lenguajes | Python 3.x |
| Data manipulation | NumPy, Pandas |
| Visualización | Matplotlib, Seaborn |
| ML clásico | scikit-learn, imbalanced-learn |
| Deep Learning | TensorFlow / Keras |
| Visión computacional | MediaPipe BlazePose, OpenCV |
| NLP | Word2Vec, GloVe, FastText |
| Dimensionalidad | UMAP, t-SNE |
| Big Data | PySpark, Databricks |
