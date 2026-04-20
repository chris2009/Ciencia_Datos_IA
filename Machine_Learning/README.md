# Machine Learning

Colección de notebooks de Machine Learning organizados por sesión, cubriendo modelos supervisados, no supervisados y reducción de dimensionalidad.
Todos los algoritmos se implementan **desde cero** con NumPy y luego se contrastan con scikit-learn.

Curso: **Machine Learning** — Maestría en Ciencia de Datos, UTEC.

---

## Notebooks

| # | Archivo | Algoritmo(s) | Tipo | Dataset |
|---|---------|-------------|------|---------|
| 1 | [Session2.0_Gradient Descent.ipynb](Session2.0_Gradient%20Descent.ipynb) | Gradient Descent | Optimización | Sintético |
| 2 | [Session2.0_OLS.ipynb](Session2.0_OLS.ipynb) | OLS (Regresión Lineal) | Supervisado - Regresión | Sintético |
| 3 | [Session2.1_Linear model.ipynb](Session2.1_Linear%20model.ipynb) | OLS, Ridge, Lasso, GD, Polynomial | Supervisado - Regresión | BandGap (energía eV) |
| 4 | [Session3.0_LDA_Logistic.ipynb](Session3.0_LDA_Logistic.ipynb) | LDA, Regresión Logística (Newton/SGD) | Supervisado - Clasificación | Sintético |
| 5 | [Session3.1_SVM.ipynb](Session3.1_SVM.ipynb) | SVM lineal, Kernel SVM, SVR | Supervisado - Clasificación/Regresión | Sintético / Breast Cancer |
| 6 | [Session3.2_Random Forest.ipynb](Session3.2_Random%20Forest.ipynb) | Random Forest | Supervisado - Clasificación | Sintético / Breast Cancer |
| 7 | [Session4.0_kMeans.ipynb](Session4.0_kMeans.ipynb) | K-Means, K-Means++, MiniBatch K-Means | No supervisado - Clustering | Sintético |
| 8 | [Session4.0_kmedoids.ipynb](Session4.0_kmedoids.ipynb) | PAM, CLARA | No supervisado - Clustering | Sintético / Wine |
| 9 | [Session4.0_Mean Shift.ipynb](Session4.0_Mean%20Shift.ipynb) | KDE, Mean Shift | No supervisado - Clustering | Sintético |
| 10 | [Session4.0_EM.ipynb](Session4.0_EM.ipynb) | GMM + EM, AIC/BIC | No supervisado - Clustering | Wine |
| 11 | [Session4.1_Agglomerative.ipynb](Session4.1_Agglomerative.ipynb) | Agglomerative Clustering (4 linkages) | No supervisado - Clustering | Sintético |
| 12 | [PCA.ipynb](PCA.ipynb) | PCA | Reducción de dimensionalidad | Sintético / Iris / Wine |
| 13 | [Dimensional reduction.ipynb](Dimensional%20reduction.ipynb) | PCA, t-SNE, UMAP | Reducción de dimensionalidad | Swiss Roll / Digits / Fashion MNIST |
| 14 | [Unbalanced.ipynb](Unbalanced.ipynb) | Undersampling, Tomek Links, Oversampling | Manejo de desbalance | Sintético / Breast Cancer |

---

## Patrón común de todos los notebooks

```
TEORÍA + MATEMÁTICA
    │
    ▼
IMPLEMENTACIÓN FROM SCRATCH (NumPy puro)
    ├─ Funciones auxiliares reutilizables
    ├─ Visualización del proceso interno
    └─ Métricas de evaluación
    │
    ▼
VALIDACIÓN CON SCIKIT-LEARN
    └─ Comparación numérica de resultados
    │
    ▼
APLICACIÓN A DATASET REAL
    └─ Conclusiones e interpretación
```

---

## Detalle por notebook

---

### 1. `Session2.0_Gradient Descent.ipynb` — Optimización

**Concepto central:** El gradiente descendente como algoritmo general de optimización numérica.

**Implementaciones desde cero:**
- `gd(f, grad, x0, alpha, n_steps)` — GD genérico para cualquier función diferenciable
- `gd_1d(f, grad, x0, alpha, n_steps)` — versión 1D con trayectoria completa

**Experimentos:**
| Experimento | Qué demuestra |
|-------------|---------------|
| Función cuadrática 2D (elipse) | Trayectoria de descenso en espacio de parámetros vs curva de costo |
| Efecto del learning rate α | α pequeño = lento, α óptimo ≈ 1/L = rápido, α > 2/L = diverge |
| Función convexa vs no convexa | Garantía de convergencia al mínimo global solo en convexas |
| GD vs OLS en regresión lineal | Ambos convergen a los mismos pesos; OLS es solución exacta |

**Fórmula implementada:**
```
w_{t+1} = w_t - α · ∇f(w_t)
```

---

### 2. `Session2.0_OLS.ipynb` — Regresión Lineal (OLS)

**Concepto central:** Estimador de Mínimos Cuadrados Ordinarios y sus propiedades geométricas/estadísticas.

**Implementaciones desde cero:**
- `ols_fit(X, y)` — ajuste por pseudoinversa: `ŵ = (XᵀX)⁻¹Xᵀy`
- `regression_metrics(y, ŷ, residual, X_design)` — RSS, MSE, RMSE, MAE, R²
- `summarize_orthogonality(X_design, residual)` — verifica que `Xᵀe ≈ 0`
- `condition_number_sym_psd(A)` — número de condición de XᵀX

**Análisis profundos:**
| Experimento | Hallazgo |
|-------------|----------|
| Varianza del estimador vs n | Con más datos, los coeficientes se estabilizan alrededor de los valores reales |
| Colinealidad (`ε`-perturbación) | Cuando x₁ ≈ x₂, el número de condición de XᵀX explota → coeficientes inestables |
| Mala especificación | Ajustar una línea a datos cuadráticos → sesgo sistemático (underfitting) |
| Heterocedasticidad | Los residuos crecen con x → OLS es ineficiente (varianza no constante) |

---

### 3. `Session2.1_Linear model.ipynb` — Modelos Lineales sobre BandGap

**Dataset real:** BandGap energético de materiales sólidos (`Eg(G0W0; eV)`) — 10 folds de train/val + test.

**Pipeline de preprocesamiento:**
```
Mediana (imputación NaN) → Estandarización (μ, σ) → One-Hot (tipo cristalino) → Bias column
```

**Algoritmos implementados desde cero:**

| Modelo | Método | Fórmula / Detalle |
|--------|--------|-------------------|
| OLS | Pseudoinversa | `ŵ = X⁺y` |
| Ridge | Solución cerrada | `ŵ = (XᵀX + λI)⁻¹Xᵀy` — no regulariza bias |
| Gradient Descent | Batch GD | `w -= (2/n) · Xᵀ(Xw - y)` |
| Lasso | Coordinate Descent | Soft-threshold: `S(a,λ) = sign(a)·max(0, |a|-λ)` |
| Polynomial | Expansión de monomios | Combina features numéricos hasta grado d |

**Evaluación con 10-fold cross-validation:**
- Grid search sobre λ ∈ {1e-4, …, 100} para Ridge y Lasso
- Grid search sobre (grado ∈ {1,2,3,4}, λ) para Polynomial+Ridge
- Métricas: RMSE, MAE, R²
- El mejor modelo se reentrena con train+val completo y se evalúa en test

**Observación:** Usa `google.colab.drive` — requiere adaptación para entorno local.

---

### 4. `Session3.0_LDA_Logistic.ipynb` — LDA y Regresión Logística

**Dataset:** Sintético 2-clases, misma covarianza (supuesto LDA).

#### LDA desde cero — `LDAFromScratch`
```python
# Parámetros estimados:
π_k  = N_k / N                        # priors
μ_k  = media de X para clase k
Σ    = covarianza compartida (pooled)  # (N-K) denominador
# Función discriminante:
δ_k(x) = xᵀΣ⁻¹μ_k - ½ μ_kᵀΣ⁻¹μ_k + log(π_k)
```

#### Regresión Logística desde cero — 2 optimizadores

| Método | Clase | Detalle |
|--------|-------|---------|
| Newton-Raphson (IRLS) | `LogisticNewtonIRLS` | Hessiana + L2 sin penalizar intercept, convergencia en ~10-30 iter |
| SGD con mini-batches | `LogisticSGD` | `lr=0.2`, `epochs=200`, `batch_size=64` |

- **Sigmoid numericamente estable:** usa `np.where` para evitar overflow en valores negativos
- Comparación de curvas NLL (Newton vs SGD), fronteras de decisión, accuracy y log-loss contra sklearn + LDA

---

### 5. `Session3.1_SVM.ipynb` — Support Vector Machines

**Secciones:**

#### SVM Lineal
- Efecto del parámetro C: bajo (margen amplio, más errores) → alto (margen estrecho, overfitting)
- `make_pipeline(StandardScaler(), SVC(kernel="linear", C=C))`

#### Kernel SVM
- Comparación: `linear`, `rbf` con distintos C y γ
- El kernel RBF con γ alto → overfitting (frontera muy compleja)

#### Hinge Loss + SGD desde cero
```python
hinge_loss(y, f) = max(0, 1 - y·f)
# Actualización SGD:
if y*f < 1:  w -= lr*(−y·x + 2*lam*w)
else:        w -= lr*(2*lam*w)        # solo regularización
```

#### SVR (Regresión)
- ε-insensitive loss: ignora errores menores a ε
- Aplicado a señal senoidal con ruido

#### Práctica: Breast Cancer (569 muestras, 30 features)
- `train_test_split` estratificado 75/25
- Comparación de kernels y configuraciones

---

### 6. `Session3.2_Random Forest.ipynb` — Random Forest

**Parámetros del modelo:**
```python
RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",    # sqrt(p) features por árbol
    min_samples_leaf=2,
    bootstrap=True,
    oob_score=True,         # OOB error sin cross-val
    n_jobs=-1
)
```

**Experimentos:**
| Experimento | Resultado |
|-------------|-----------|
| Accuracy vs nº de árboles (1→300) | Mejora rápida hasta ~30 árboles, luego se estabiliza |
| Dataset sintético 2D | Frontera de decisión no lineal visualizada |
| Breast Cancer (569 muestras) | Alta accuracy; top features por importancia de Gini |

**Feature importance:** Barplot de las 12 variables más importantes en Breast Cancer, ordenadas por impureza de Gini media.

---

### 7. `Session4.0_kMeans.ipynb` — K-Means

**Implementaciones desde cero:**

| Clase/Función | Detalle |
|---------------|---------|
| `init_random_centers` | Centros iniciales aleatorios sin reemplazo |
| `init_kmeans_plus_plus` | Distancias al cuadrado como probabilidades → menos mínimos locales |
| `kmeans_from_scratch` | E-step (asignación), M-step (actualización), convergencia por `‖Δcentros‖ < tol` |
| `minibatch_kmeans_from_scratch` | Actualiza centros con mini-batches usando learning rate decreciente |

**Comparaciones:**
- Random init vs K-Means++ → K-Means++ converge más rápido y con menor inercia
- From scratch vs sklearn `KMeans` + `MiniBatchKMeans` → resultados idénticos
- **Limitación demostrada:** K-Means falla en `make_moons` (clusters no convexos)

---

### 8. `Session4.0_kmedoids.ipynb` — K-Medoids

**Diferencia clave con K-Means:** Los centros son puntos reales del dataset (medoids), no medias — más robusto ante outliers y compatible con métricas no euclidianas.

**Algoritmos implementados:**

#### PAM (Partitioning Around Medoids)
```
BUILD: greedy — elige k medoids que minimizan la suma total de distancias
SWAP: itera intercambiando cada medoid con cada no-medoid → acepta si reduce el costo total
```

#### CLARA (Clustering Large Applications)
```
Repite num_samples veces:
  1. Muestrea sample_size puntos del dataset
  2. Aplica PAM sobre la muestra
  3. Asigna todos los puntos a los medoids encontrados
  4. Conserva la configuración con menor costo global
```

**Ventaja de CLARA:** O(sample_size² · k) por iteración vs O(n² · k) de PAM → escala mejor.

**Observación de código:** La última celda aplica PAM con `k=178` sobre el dataset Wine (178 muestras) — cada punto sería su propio medoid, lo cual es un error — debería ser `k=3` para las 3 variedades de vino.

---

### 9. `Session4.0_Mean Shift.ipynb` — KDE y Mean Shift

**KDE implementado desde cero:**
```python
# 1D:  density(x) = (1/nh) Σ K((x - xᵢ)/h)
# 2D:  density(x) = (1/nh²) Σ K((x - xᵢ)/h)
# Kernel Gaussiano: K(u) = exp(-½‖u‖²) / (2π)
```

**Efecto del bandwidth h en KDE:**
- h pequeño → sobreajuste (muchos picos)
- h óptimo → captura la estructura real
- h grande → suavizado excesivo (un solo pico)

**Mean Shift desde cero:**
```
Cada punto x se desplaza iterativamente hacia la media ponderada de sus vecinos:
  x_{t+1} = Σ K((x-xᵢ)/h)·xᵢ / Σ K((x-xᵢ)/h)
Converge al modo local de la densidad.
Los puntos que convergen al mismo modo → mismo cluster.
```

- `cluster_modes`: agrupa puntos convergidos con distancia < `merge_tol`
- **Ventaja:** No requiere especificar k; descubre el número de clusters automáticamente
- Validado contra `sklearn.cluster.MeanShift`

---

### 10. `Session4.0_EM.ipynb` — Modelos de Mezcla Gaussiana (GMM)

**Dataset:** Wine (178 muestras, 13 features, 3 variedades) — estandarizado con `StandardScaler`.

**GMM con covarianza diagonal — `gmm_em_diag` desde cero:**

#### Inicialización: K-Means++ (`init_means_kpp`)
```python
# Asegura centros iniciales bien separados
```

#### Algoritmo EM:
```
E-step: r_{ik} = π_k · N(xᵢ | μ_k, diag(σ_k²)) / Σ_j π_j · N(xᵢ | μ_j, diag(σ_j²))
M-step: π_k = N_k/n,  μ_k = Σ r_{ik}·xᵢ / N_k,  σ_k² = Σ r_{ik}·(xᵢ-μ_k)² / N_k
```
- **Truco numérico:** `logsumexp` para evitar underflow en probabilidades pequeñas
- Converge cuando `|ΔlogLik| < tol`

#### Selección de k con AIC/BIC:
```python
AIC = 2p - 2·logLik
BIC = p·log(n) - 2·logLik
# p = parámetros = (k-1) + 2·k·d  (diag)
```
- Barrido de k ∈ {1…8}: BIC penaliza más la complejidad que AIC
- Visualización en PCA 2D de los clusters asignados

**sklearn GaussianMixture:** comparación `diag` vs `full` covariance con densidad 2D.

---

### 11. `Session4.1_Agglomerative.ipynb` — Clustering Jerárquico

**Dataset:** Sintético 36 puntos, 3 clusters (pequeño para seguir los pasos manualmente).

**Algoritmo desde cero — `agglomerative_clustering`:**
```
Inicializar: cada punto = un cluster
Repetir hasta un solo cluster:
  Encontrar el par (i, j) con menor distancia entre clusters
  Fusionar → registrar en history
```

**4 criterios de enlace implementados:**

| Linkage | Distancia entre clusters A y B | Efecto |
|---------|-------------------------------|--------|
| `single` | `min_{a∈A, b∈B} d(a,b)` | Tiende a "chaining" (cadenas largas) |
| `complete` | `max_{a∈A, b∈B} d(a,b)` | Clusters más compactos y esféricos |
| `average` | `mean_{a∈A, b∈B} d(a,b)` | Balance entre single y complete |
| `ward` | Minimiza el incremento de SSE total | Clusters más homogéneos en varianza |

**Extras:**
- Visualización paso a paso de fusiones
- Generación de la **matriz de enlaces (linkage matrix)** desde cero compatible con `scipy.dendrogram`
- Dendrogramas para los 4 linkages
- Comparación from scratch vs `sklearn.AgglomerativeClustering`

---

### 12. `PCA.ipynb` — Análisis de Componentes Principales

**Implementación desde cero — `PCAscratch(X, n_components)`:**
```python
1. μ = mean(X, axis=0)
2. X_c = X - μ
3. C = cov(X_c)                    # matriz de covarianza (d×d)
4. λ, V = eigh(C)                  # eigendescomposición
5. Ordenar λ descendente → tomar primeros n_components eigenvectores
6. Z = X_c @ V[:, :k]              # proyección
7. X_rec = Z @ V[:, :k].T + μ     # reconstrucción
```

**Experimentos:**
| Experimento | Hallazgo |
|-------------|----------|
| Datos 2D correlacionados | PC1 apunta en la dirección de máxima varianza |
| Proyección a 1D y reconstrucción | MSE mide la información perdida |
| **Efecto de la escala** | Sin estandarizar: PC1 dominada por variable de escala 100×; con estandarizar: componentes equilibradas |
| Scree plot | Acumulado de varianza explicada para elegir k |
| Iris + Wine con sklearn | PCA reduce a 2D, separación visual de clases |

---

### 13. `Dimensional reduction.ipynb` — PCA vs t-SNE vs UMAP

**Objetivo:** Comparar tres métodos de reducción no lineal en datasets con estructura compleja.

#### Swiss Roll (3D → 2D)
| Método | Resultado |
|--------|-----------|
| PCA | No desdobla el rollo (proyección lineal) |
| t-SNE | Desdobla parcialmente la estructura |
| UMAP | Preserva mejor la topología 1D del rollo |

#### Digits (64D → 2D) y Fashion MNIST (784D → 2D)

**Hiperparámetros explorados:**

| Método | Parámetros estudiados | Efecto |
|--------|-----------------------|--------|
| t-SNE | `perplexity` ∈ {5, 30, 80} | Bajo = clusters atomizados; alto = estructura global difusa |
| UMAP | `n_neighbors` ∈ {5,15,50}, `min_dist` ∈ {0.01,0.1,0.5} | `n_neighbors` controla escala local/global; `min_dist` compacidad |

**Conclusión visual:** UMAP produce la separación más clara de dígitos con configuración `n_neighbors=15, min_dist=0.1`.

**Observación:** Usa `!pip install umap-learn` — requiere instalación previa en entorno local.

---

### 14. `Unbalanced.ipynb` — Manejo de Clases Desbalanceadas

**Dataset sintético:** 90% clase 0 / 10% clase 1 — caso extremo de desbalance.

**Técnicas implementadas desde cero y con imblearn:**

#### Undersampling
| Técnica | Mecanismo | Desde cero | imblearn |
|---------|-----------|------------|---------|
| Random Undersampling | Elimina muestras de la clase mayoritaria aleatoriamente | ✅ | `RandomUnderSampler` |
| Tomek Links | Elimina solo los pares frontera (mayoritario-minoritario más cercanos entre sí) | ✅ | `TomekLinks` |

**Implementación de Tomek Links desde cero:**
```python
# Par (i,j) es Tomek Link si:
# 1. y[i] ≠ y[j]
# 2. nearest_neighbor(i) == j  AND  nearest_neighbor(j) == i
# Se eliminan los puntos de la clase mayoritaria en esos pares
```

#### Oversampling
| Técnica | Mecanismo | Desde cero | imblearn |
|---------|-----------|------------|---------|
| Random Oversampling | Duplica muestras minoritarias con reemplazo | ✅ | `RandomOverSampler` |

**Tabla comparativa de métricas (Logistic Regression como clasificador base):**

| Técnica | Precision (minoritaria) | Recall (minoritaria) | F1 |
|---------|------------------------|---------------------|-----|
| Baseline (sin balanceo) | Alta | Muy baja | Bajo |
| Random Undersampling | Media | Alta | Medio |
| Tomek Links | Media-Alta | Media | Medio |
| Random Oversampling | Media | Alta | Medio-Alto |

**Aplicación real:** Breast Cancer con clase artificial minoritaria (25 muestras clase 0 vs ~265 clase 1) — mismo pipeline con `StandardScaler` + cada técnica de balanceo.

---

## Análisis de calidad del código

### Fortalezas generales ✅
- **Patrón consistente** en todos: from scratch → sklearn → dataset real
- `random_state` / `random_seed` en todos los experimentos → reproducibilidad garantizada
- Funciones auxiliares bien abstraídas y reutilizadas (`ols_fit`, `eval_model`, `plot_clusters`, etc.)
- `logsumexp` en EM para estabilidad numérica — práctica de producción
- `np.where` en sigmoid para evitar overflow — correcto
- `np.linalg.pinv` en OLS y Ridge en lugar de invertir directamente → más estable
- Uso de `make_pipeline` con `StandardScaler` en SVM → sin data leakage

### Observaciones por notebook

| Notebook | Observación |
|----------|-------------|
| `Session2.1_Linear model` | Usa `google.colab.drive` — rutas hardcodeadas a Colab |
| `Session4.0_kmedoids` | Última celda aplica PAM con `k=178` sobre Wine (178 samples) — debería ser `k=3` |
| `Session4.0_kMeans` | `make_moons` demuestra correctamente la limitación de K-Means, pero no propone solución (Spectral Clustering / DBSCAN) |
| `Dimensional reduction` | Usa `!pip install umap-learn` y `tensorflow` para Fashion MNIST — dependencias pesadas |
| `Unbalanced` | No implementa SMOTE (oversampling sintético) — la técnica más utilizada en producción |

---

## Datasets utilizados

| Dataset | Fuente | Tamaño | Notebooks |
|---------|--------|--------|-----------|
| Sintético (make_blobs, make_moons, etc.) | sklearn | Variable | Todos |
| BandGap (Eg G0W0 eV) | CSV local (10 folds) | ~N×4 | Lab2.1 |
| Breast Cancer | sklearn | 569×30 | SVM, RF, Unbalanced |
| Wine | sklearn | 178×13 | EM, k-medoids |
| Iris | sklearn | 150×4 | PCA |
| Swiss Roll | sklearn | 2000×3 | Dim. Reduction |
| Digits | sklearn | 1797×64 | Dim. Reduction, PCA |
| Fashion MNIST | TensorFlow/Keras | 70000×784 | Dim. Reduction |

---

## Stack tecnológico

| Herramienta | Uso |
|-------------|-----|
| NumPy | Todos los algoritmos from scratch |
| Matplotlib | Visualizaciones (trayectorias, clusters, dendrogramas, fronteras) |
| scikit-learn | Datasets, métricas, validación de implementaciones |
| SciPy | `scipy.cluster.hierarchy.dendrogram`, `chi2`, `boxcox` |
| imbalanced-learn | `RandomUnderSampler`, `TomekLinks`, `RandomOverSampler` |
| umap-learn | UMAP dimensionality reduction |
| TensorFlow/Keras | Fashion MNIST dataset |
