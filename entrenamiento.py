import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Cargar datos con manejo de errores
try:
    df = pd.read_csv("hand_gestures.csv")
except FileNotFoundError:
    print("Error: Archivo 'hand_gestures.csv' no encontrado.")
    exit()

# Separar características y etiquetas
if "label" not in df.columns:
    print("Error: La columna 'label' no está en el dataset.")
    exit()

X = df.drop(columns=["label"])
y = df["label"]

# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Optimización de hiperparámetros usando GridSearchCV
param_grid = {"n_neighbors": range(3, 20, 2)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_knn = grid_search.best_estimator_
print(f"Mejor número de vecinos: {grid_search.best_params_['n_neighbors']}")

# Evaluación del modelo
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Guardar modelo y normalizador
joblib.dump(best_knn, "modelo_knn.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Modelo guardado como 'modelo_knn.pkl' y escalador como 'scaler.pkl'")
