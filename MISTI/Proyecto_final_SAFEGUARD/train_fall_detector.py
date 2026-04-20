"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🛡️  SAFEGUARD VISION AI - ENTRENAMIENTO DE MODELO                         ║
║                                                                              ║
║   Entrena un clasificador Random Forest para detectar caídas                 ║
║   usando keypoints extraídos con BlazePose                                   ║
║                                                                              ║
║   Autor: Christian Cajusol                                                   ║
║   Proyecto: MIT Global Teaching Labs                                         ║
║   Fecha: Enero 2026                                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

DESCRIPCIÓN:
============
Este script toma el CSV generado por BlazePose (keypoints_dataset.csv) y entrena
un modelo de Machine Learning para clasificar entre:
    - Clase 0 (ADL): Actividades normales (caminar, sentarse, pararse)
    - Clase 1 (Fall): Caídas

ENTRADA:
========
    - keypoints_dataset.csv: CSV con 132 features de keypoints + etiquetas

SALIDA:
=======
    - modelo_caidas.pkl: Modelo Random Forest entrenado
    - scaler_caidas.pkl: Normalizador de features
    - training_report.json: Métricas y configuración del entrenamiento
    - confusion_matrix.png: Visualización de la matriz de confusión
    - feature_importance.png: Importancia de cada keypoint

USO:
====
    python train_fall_detector.py

REQUISITOS:
===========
    pip install pandas scikit-learn matplotlib seaborn joblib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import learning_curve
import joblib
import warnings
warnings.filterwarnings('ignore')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURACIÓN                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Ruta al CSV de keypoints (generado por BlazePose)
CSV_PATH = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\safeguard_keypoints_lite\keypoints_LITE.csv"

# Carpeta donde guardar el modelo entrenado
OUTPUT_FOLDER = r"D:\APRENDIZAJE\MAESTRIA\CICLO_III\MISTI\Project_local\train_model"

# Configuración del entrenamiento
CONFIG = {
    "test_size": 0.2,           # 20% para test, 80% para entrenamiento
    "random_state": 42,         # Semilla para reproducibilidad
    "cross_validation_folds": 5, # Número de folds para validación cruzada
    "optimize_hyperparameters": True  # Si hacer búsqueda de hiperparámetros
}

# Hiperparámetros para búsqueda (si optimize_hyperparameters = True)
PARAM_GRID = {
    'n_estimators': [100, 200, 300],      # Número de árboles
    'max_depth': [10, 20, 30, None],       # Profundidad máxima
    'min_samples_split': [2, 5, 10],       # Mínimo de muestras para dividir
    'min_samples_leaf': [1, 2, 4]          # Mínimo de muestras en hoja
}

# Hiperparámetros por defecto (si optimize_hyperparameters = False)
DEFAULT_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1  # Usar todos los cores
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         FUNCIONES PRINCIPALES                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def load_and_prepare_data(csv_path):
    """
    Carga el CSV de keypoints y prepara los datos para entrenamiento.
    
    Args:
        csv_path: Ruta al archivo CSV
        
    Returns:
        X: Features (keypoints)
        y: Labels (0=ADL, 1=Fall)
        feature_names: Nombres de las columnas de features
        df: DataFrame original
    """
    print("\n📂 Cargando datos...")
    
    # Cargar CSV
    df = pd.read_csv(csv_path)
    print(f"   ✅ Cargadas {len(df):,} muestras")
    
    # Identificar columnas de features (todos los keypoints)
    metadata_columns = ['filename', 'folder', 'dataset', 'label', 'label_name']
    feature_columns = [col for col in df.columns if col not in metadata_columns]
    
    print(f"   📊 Features: {len(feature_columns)} columnas")
    print(f"   🏷️  Distribución de clases:")
    print(f"      - ADL (0):  {len(df[df['label']==0]):,} muestras")
    print(f"      - Fall (1): {len(df[df['label']==1]):,} muestras")
    
    # Separar features y labels
    X = df[feature_columns].values
    y = df['label'].values
    
    # Verificar valores nulos o infinitos
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("   ⚠️  Detectados valores nulos/infinitos, limpiando...")
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return X, y, feature_columns, df


def create_additional_features(X, feature_names):
    """
    Crea features adicionales derivadas de los keypoints.
    Estas features ayudan al modelo a detectar patrones de caídas.
    
    Features creadas:
    - Ángulo del torso (relación nariz-cadera)
    - Altura del cuerpo (dispersión vertical)
    - Ancho del cuerpo (dispersión horizontal)
    - Centro de masa aproximado
    - Simetría del cuerpo
    
    Args:
        X: Array de features originales
        feature_names: Nombres de las columnas
        
    Returns:
        X_enhanced: Features originales + nuevas features
        new_feature_names: Lista actualizada de nombres
    """
    print("\n🔧 Creando features adicionales...")
    
    # Convertir a DataFrame para facilitar acceso
    df_features = pd.DataFrame(X, columns=feature_names)
    
    new_features = []
    new_names = []
    
    # 1. Ángulo del torso (diferencia Y entre nariz y caderas)
    #    Negativo = cabeza arriba (normal), Positivo = cabeza abajo (caída)
    torso_angle = df_features['nose_y'] - (df_features['left_hip_y'] + df_features['right_hip_y']) / 2
    new_features.append(torso_angle.values)
    new_names.append('torso_angle')
    
    # 2. Altura del cuerpo (diferencia entre punto más alto y más bajo)
    #    Valor alto = persona de pie, Valor bajo = persona acostada
    y_columns = [col for col in feature_names if col.endswith('_y')]
    y_values = df_features[y_columns]
    body_height = y_values.max(axis=1) - y_values.min(axis=1)
    new_features.append(body_height.values)
    new_names.append('body_height')
    
    # 3. Ancho del cuerpo (dispersión horizontal)
    #    Valor alto = cuerpo extendido (posible caída)
    x_columns = [col for col in feature_names if col.endswith('_x')]
    x_values = df_features[x_columns]
    body_width = x_values.max(axis=1) - x_values.min(axis=1)
    new_features.append(body_width.values)
    new_names.append('body_width')
    
    # 4. Ratio altura/ancho
    #    Alto = persona vertical, Bajo = persona horizontal
    aspect_ratio = body_height / (body_width + 0.001)  # +0.001 para evitar división por cero
    new_features.append(aspect_ratio.values)
    new_names.append('aspect_ratio')
    
    # 5. Centro de masa Y (promedio de puntos clave del torso)
    center_y = (df_features['nose_y'] + df_features['left_hip_y'] + 
                df_features['right_hip_y'] + df_features['left_shoulder_y'] + 
                df_features['right_shoulder_y']) / 5
    new_features.append(center_y.values)
    new_names.append('center_mass_y')
    
    # 6. Distancia hombros (simetría superior)
    shoulder_dist = np.sqrt((df_features['left_shoulder_x'] - df_features['right_shoulder_x'])**2 +
                           (df_features['left_shoulder_y'] - df_features['right_shoulder_y'])**2)
    new_features.append(shoulder_dist.values)
    new_names.append('shoulder_distance')
    
    # 7. Distancia caderas (simetría inferior)
    hip_dist = np.sqrt((df_features['left_hip_x'] - df_features['right_hip_x'])**2 +
                       (df_features['left_hip_y'] - df_features['right_hip_y'])**2)
    new_features.append(hip_dist.values)
    new_names.append('hip_distance')
    
    # 8. Ángulo de piernas (diferencia Y entre caderas y tobillos)
    leg_angle = ((df_features['left_hip_y'] + df_features['right_hip_y']) / 2 - 
                 (df_features['left_ankle_y'] + df_features['right_ankle_y']) / 2)
    new_features.append(leg_angle.values)
    new_names.append('leg_angle')
    
    # Combinar features originales con nuevas
    new_features_array = np.column_stack(new_features)
    X_enhanced = np.hstack([X, new_features_array])
    
    enhanced_feature_names = list(feature_names) + new_names
    
    print(f"   ✅ Creadas {len(new_names)} features adicionales")
    print(f"   📊 Total features: {len(enhanced_feature_names)}")
    
    return X_enhanced, enhanced_feature_names


def train_model(X_train, y_train, optimize=True):
    """
    Entrena el modelo Random Forest.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Labels de entrenamiento
        optimize: Si True, busca mejores hiperparámetros
        
    Returns:
        model: Modelo entrenado
        best_params: Mejores parámetros encontrados
    """
    print("\n🧠 Entrenando modelo Random Forest...")
    
    if optimize:
        print("   🔍 Buscando mejores hiperparámetros (esto puede tomar unos minutos)...")
        
        # Búsqueda de hiperparámetros con validación cruzada
        rf = RandomForestClassifier(random_state=CONFIG['random_state'], n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=PARAM_GRID,
            cv=CONFIG['cross_validation_folds'],
            scoring='f1',  # Optimizar para F1-score
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        model = grid_search.best_estimator_
        
        print(f"\n   ✅ Mejores parámetros encontrados:")
        for param, value in best_params.items():
            print(f"      - {param}: {value}")
    else:
        print("   📋 Usando parámetros por defecto...")
        
        model = RandomForestClassifier(**DEFAULT_PARAMS)
        model.fit(X_train, y_train)
        best_params = DEFAULT_PARAMS
    
    return model, best_params


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, output_folder):
    """
    Evalúa el modelo y genera visualizaciones corregidas.
    Incluye:
    - Matriz de confusión (Fall como clase positiva)
    - Curva ROC
    - Curva de aprendizaje (Train vs Validation)
    - Distribución de probabilidades
    """
    print("\n📊 Evaluando modelo...")

    # =============================
    # PREDICCIONES
    # =============================
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # =============================
    # MÉTRICAS
    # =============================
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_prob_test)

    cv_scores = cross_val_score(
        model,
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        cv=CONFIG['cross_validation_folds'],
        scoring='f1'
    )

    metrics = {
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "test_f1_score": round(test_f1, 4),
        "test_auc_roc": round(test_auc, 4),
        "cv_f1_mean": round(cv_scores.mean(), 4),
        "cv_f1_std": round(cv_scores.std(), 4)
    }

    # =============================
    # FIGURA GENERAL
    # =============================
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        '🛡️ SafeGuard Vision AI - Evaluación del Modelo\nRandom Forest + BlazePose',
        fontsize=14,
        fontweight='bold'
    )

    # =============================
    # 1. MATRIZ DE CONFUSIÓN (CORREGIDA)
    # =============================
    ax1 = fig.add_subplot(2, 2, 1)

    cm = confusion_matrix(y_test, y_pred_test, labels=[1, 0])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Fall', 'ADL'],
        yticklabels=['Fall', 'ADL'],
        annot_kws={'size': 16},
        ax=ax1
    )

    TP, FN, FP, TN = cm.ravel()

    ax1.set_xlabel('Predicción', fontweight='bold')
    ax1.set_ylabel('Real', fontweight='bold')
    ax1.set_title('Matriz de Confusión (Fall = clase positiva)', fontweight='bold')
    ax1.text(
        0.5, -0.18,
        f'TP={TP} | FN={FN} | FP={FP} | TN={TN}',
        transform=ax1.transAxes,
        ha='center',
        fontsize=10
    )

    # =============================
    # 2. CURVA ROC
    # =============================
    ax2 = fig.add_subplot(2, 2, 2)

    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    ax2.plot(fpr, tpr, linewidth=2, label=f'AUC = {test_auc:.3f}')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax2.fill_between(fpr, tpr, alpha=0.3)

    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')
    ax2.set_title('Curva ROC', fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    # =============================
    # 3. CURVA DE APRENDIZAJE (NUEVA)
    # =============================
    ax3 = fig.add_subplot(2, 2, 3)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
        cv=CONFIG['cross_validation_folds'],
        scoring='f1',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    ax3.plot(train_sizes, train_mean, 'o-', label='Train F1')
    ax3.plot(train_sizes, val_mean, 'o-', label='Validation F1')

    ax3.set_xlabel('Número de muestras', fontweight='bold')
    ax3.set_ylabel('F1-score', fontweight='bold')
    ax3.set_title('Curva de Aprendizaje', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # =============================
    # 4. DISTRIBUCIÓN DE PROBABILIDADES
    # =============================
    ax4 = fig.add_subplot(2, 2, 4)

    ax4.hist(y_prob_test[y_test == 0], bins=30, alpha=0.7, label='ADL')
    ax4.hist(y_prob_test[y_test == 1], bins=30, alpha=0.7, label='Fall')
    ax4.axvline(x=0.5, linestyle='--', color='black', label='Threshold = 0.5')

    ax4.set_xlabel('Probabilidad de caída', fontweight='bold')
    ax4.set_ylabel('Frecuencia', fontweight='bold')
    ax4.set_title('Distribución de Probabilidades', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # =============================
    # GUARDAR FIGURA
    # =============================
    plot_path = os.path.join(output_folder, "model_evaluation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n   💾 Gráfico guardado: {plot_path}")

    return metrics


def save_model(model, scaler, feature_names, metrics, best_params, output_folder):
    """
    Guarda el modelo entrenado y archivos asociados.
    
    Args:
        model: Modelo entrenado
        scaler: Normalizador de features
        feature_names: Nombres de features
        metrics: Métricas de evaluación
        best_params: Parámetros del modelo
        output_folder: Carpeta de salida
    """
    print("\n💾 Guardando modelo y archivos...")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Guardar modelo
    model_path = os.path.join(output_folder, "modelo_caidas.pkl")
    joblib.dump(model, model_path)
    print(f"   ✅ Modelo: {model_path}")
    
    # 2. Guardar scaler
    scaler_path = os.path.join(output_folder, "scaler_caidas.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler: {scaler_path}")
    
    # 3. Guardar nombres de features
    features_path = os.path.join(output_folder, "feature_names.json")
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"   ✅ Features: {features_path}")
    
    # 4. Guardar reporte completo
    report = {
        "project": "SafeGuard Vision AI",
        "author": "Christian Cajusol - MIT Global Teaching Labs",
        "model_type": "RandomForestClassifier",
        "training_date": datetime.now().isoformat(),
        "parameters": best_params,
        "metrics": metrics,
        "feature_count": len(feature_names),
        "classes": {
            "0": "ADL (Actividad Normal)",
            "1": "Fall (Caída)"
        },
        "files_generated": [
            "modelo_caidas.pkl",
            "scaler_caidas.pkl",
            "feature_names.json",
            "model_evaluation.png"
        ],
        "usage_example": """
# Cargar modelo
import joblib
model = joblib.load('modelo_caidas.pkl')
scaler = joblib.load('scaler_caidas.pkl')

# Predecir
X_new = scaler.transform(keypoints)
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)[:, 1]
"""
    }
    
    report_path = os.path.join(output_folder, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Reporte: {report_path}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """Función principal de entrenamiento."""
    
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🛡️  SAFEGUARD VISION AI - ENTRENAMIENTO DE MODELO ".center(70) + "║")
    print("║" + " Random Forest para Detección de Caídas ".center(70) + "║")
    print("║" + " MIT Global Teaching Labs ".center(70) + "║")
    print("╚" + "═"*70 + "╝")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 1: Cargar datos
    # ═══════════════════════════════════════════════════════════════════════════
    
    if not os.path.exists(CSV_PATH):
        print(f"\n❌ ERROR: No se encuentra el archivo CSV:")
        print(f"   {CSV_PATH}")
        print("\n   Primero debes ejecutar el script de extracción de keypoints.")
        return
    
    X, y, feature_names, df = load_and_prepare_data(CSV_PATH)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 2: Crear features adicionales
    # ═══════════════════════════════════════════════════════════════════════════
    
    X, feature_names = create_additional_features(X, feature_names)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 3: Dividir datos
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n📊 Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y  # Mantener proporción de clases
    )
    print(f"   Train: {len(X_train):,} muestras")
    print(f"   Test:  {len(X_test):,} muestras")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 4: Normalizar features
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n⚖️  Normalizando features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   ✅ Features normalizadas (StandardScaler)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 5: Entrenar modelo
    # ═══════════════════════════════════════════════════════════════════════════
    
    model, best_params = train_model(
        X_train_scaled, 
        y_train, 
        optimize=CONFIG['optimize_hyperparameters']
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 6: Evaluar modelo
    # ═══════════════════════════════════════════════════════════════════════════
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    metrics = evaluate_model(
        model, 
        X_train_scaled, 
        X_test_scaled, 
        y_train, 
        y_test,
        feature_names,
        OUTPUT_FOLDER
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PASO 7: Guardar modelo
    # ═══════════════════════════════════════════════════════════════════════════
    
    save_model(model, scaler, feature_names, metrics, best_params, OUTPUT_FOLDER)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n")
    print("╔" + "═"*70 + "╗")
    print("║" + " 🎉 ENTRENAMIENTO COMPLETADO ".center(70) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📊 Accuracy:  {metrics['test_accuracy']:.2%}".ljust(71) + "║")
    print(f"║  🎯 Recall:    {metrics['test_recall']:.2%}  (detección de caídas)".ljust(71) + "║")
    print(f"║  ⚡ F1-Score:  {metrics['test_f1_score']:.2%}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print(f"║  📂 Modelo guardado en: {OUTPUT_FOLDER}".ljust(71) + "║")
    print("╠" + "═"*70 + "╣")
    print("║  📁 Archivos generados:".ljust(71) + "║")
    print("║      • modelo_caidas.pkl    (modelo entrenado)".ljust(71) + "║")
    print("║      • scaler_caidas.pkl    (normalizador)".ljust(71) + "║")
    print("║      • feature_names.json   (nombres de features)".ljust(71) + "║")
    print("║      • training_report.json (métricas y config)".ljust(71) + "║")
    print("║      • model_evaluation.png (visualizaciones)".ljust(71) + "║")
    print("╚" + "═"*70 + "╝")
    
    print("\n🚀 SIGUIENTE PASO: Ejecutar demo_video_safeguard.py")
    print()


if __name__ == "__main__":
    main()
