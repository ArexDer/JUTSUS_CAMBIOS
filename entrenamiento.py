import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib  # Importar joblib para guardar el modelo

# Cargar datos
df = pd.read_csv("hand_gestures.csv")
X = df.drop(columns=["label"])
y = df["label"]

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluar el modelo
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Guardar el modelo entrenado en un archivo .pkl
joblib.dump(knn, "modelo_knn.pkl")
print("Modelo guardado como 'modelo_knn.pkl'")
