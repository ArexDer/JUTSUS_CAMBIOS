import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import random
from collections import deque
import os

# Cargar el modelo y el escalador
try:
    knn = joblib.load("modelo_knn.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: No se encontraron los archivos del modelo ('modelo_knn.pkl' o 'scaler.pkl').")
    exit()

# Definir los jutsus y sus secuencias
jutsus = {
    "Katon: Gokakyu no Jutsu": ["snake", "ram", "monkey", "boar", "rat", "horse", "tiger"],
    "Kage Bunshin no Jutsu": ["ram", "snake", "tiger"]
}

# Seleccionar un jutsu aleatorio
jutsu_actual, secuencia_correcta = random.choice(list(jutsus.items()))

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Cargar las imágenes de las posturas
imagenes_posturas = {}
carpeta_imagenes = "imagenes_posturas"  # Carpeta donde se encuentran las imágenes
for postura in ["snake", "ram", "monkey", "boar", "horse", "tiger" ,"dog", "rat"]:
    imagen_path = os.path.join(carpeta_imagenes, f"{postura}.png")
    imagenes_posturas[postura] = cv2.imread(imagen_path)

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

            # Dibujar la imagen de la postura en lugar del texto
            if prediction in imagenes_posturas:
                imagen_postura = imagenes_posturas[prediction]
                frame[50:50+imagen_postura.shape[0], 50:50+imagen_postura.shape[1]] = imagen_postura

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

        if postura in imagenes_posturas:
            imagen_postura = imagenes_posturas[postura]
            imagen_postura = cv2.resize(imagen_postura, (rect_width, rect_height))
            frame[y1:y2, x1:x2] = imagen_postura

    # Mostrar el jutsu seleccionado en la parte derecha de la ventana
    text_size = cv2.getTextSize(jutsu_actual, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = width - text_size[0] - 10  # 10 píxeles de margen desde el borde derecho
    cv2.putText(frame, jutsu_actual, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Mostrar las posturas detectadas en orden
    for i, postura in enumerate(posturas_detectadas):
        if postura in imagenes_posturas:
            imagen_postura = imagenes_posturas[postura]
            imagen_postura = cv2.resize(imagen_postura, (rect_width, rect_height))
            y1 = 100 + i * (rect_height + spacing)
            y2 = y1 + rect_height
            if y2 <= frame.shape[0]:  # Asegúrate de que la imagen encaje en el marco
                frame[y1:y2, text_x:text_x + rect_width] = imagen_postura

    # Mensaje de finalización
    if posturas_detectadas == secuencia_correcta:
        completion_text = f"COMPLETASTE EL {jutsu_actual}!"
        # Calcular el tamaño de la fuente en función del ancho de la ventana
        font_scale = width / 1000
        text_size = cv2.getTextSize(completion_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(frame, completion_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Detección de Jutsus", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()