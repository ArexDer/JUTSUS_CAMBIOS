import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import random
from collections import deque

# Cargar el modelo y el escalador
try:
    knn = joblib.load("modelo_knn.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: No se encontraron los archivos del modelo ('modelo_knn.pkl' o 'scaler.pkl').")
    exit()

# Definir los jutsus y sus secuencias
jutsus = {
    "Katon: Gōkakyū no Jutsu": ["snake", "ram", "monkey", "boar", "boar", "horse", "tiger"],
    "Kage Bunshin no Jutsu": ["ram", "snake", "tiger"]
}

# Seleccionar un jutsu aleatorio
jutsu_actual, secuencia_correcta = random.choice(list(jutsus.items()))

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(0)

# Hacer que la ventana sea ajustable
cv2.namedWindow("Detección de Jutsus", cv2.WINDOW_NORMAL)

# Lista para almacenar las posturas detectadas
posturas_detectadas = []
ultimas_detecciones = deque(maxlen=5)  # Almacena las últimas 5 detecciones para evitar repeticiones

ultimo_tiempo = time.time()
delay_postura = 1.5  # Tiempo en segundos antes de aceptar una nueva postura

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Obtener dimensiones de la ventana
    height, width, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer landmarks y normalizarlos
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)
            landmarks = scaler.transform(landmarks)  # Normalización

            # Realizar predicción
            prediction = knn.predict(landmarks)[0]

            tiempo_actual = time.time()

            # Verificar si la postura es la siguiente correcta
            if len(posturas_detectadas) < len(secuencia_correcta) and prediction == secuencia_correcta[len(posturas_detectadas)]:
                if prediction not in ultimas_detecciones and (tiempo_actual - ultimo_tiempo) > delay_postura:
                    posturas_detectadas.append(prediction)
                    ultimas_detecciones.append(prediction)
                    ultimo_tiempo = tiempo_actual

            cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Dibujar secuencia de jutsu en la pantalla
    rect_width = 80
    rect_height = 60
    spacing = 10
    total_width = len(secuencia_correcta) * (rect_width + spacing) - spacing
    start_x = (width - total_width) // 2
    start_y = height - 100

    for i, postura in enumerate(secuencia_correcta):
        color = (255, 255, 255) if i >= len(posturas_detectadas) else (0, 255, 0)
        x1, y1 = start_x + i * (rect_width + spacing), start_y
        x2, y2 = x1 + rect_width, y1 + rect_height
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        text_size = cv2.getTextSize(postura, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (rect_width - text_size[0]) // 2
        text_y = y1 + (rect_height + text_size[1]) // 2
        cv2.putText(frame, postura, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Mensaje de finalización
    if posturas_detectadas == secuencia_correcta:
        cv2.putText(frame, f"COMPLETASTE EL {jutsu_actual}!", (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

    # Mostrar el jutsu seleccionado
    cv2.putText(frame, jutsu_actual, (width // 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Detección de Jutsus", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
