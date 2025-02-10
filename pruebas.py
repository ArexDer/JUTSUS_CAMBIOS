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
import pygame  # Libreria para reproducir sonido

# Musica
pygame.mixer.init()

WIDTH = 1024
HEIGHT = 768

MIRROR_MODE = True

# Boton
detection_started = False  
exit_program = False       

current_music = None

# Definir dimensiones y posicion del boton
button_width = 100
button_height = 30
button_x = WIDTH // 2 - button_width // 2
button_y = 50

# Funcion callback para detectar clicks del mouse en la ventana
def on_mouse(event, x, y, flags, param):
    global detection_started, exit_program
    if event == cv2.EVENT_LBUTTONDOWN:
        # Verificar si el click se realizo dentro del boton
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            if not detection_started:
                detection_started = True
            else:
                exit_program = True

# Cargar el modelo y el escalador
try:
    knn = joblib.load("modelo_knn.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: No se encontraron los archivos del modelo ('modelo_knn.pkl' o 'scaler.pkl').")
    exit()

# Definir los jutsus y sus secuencias
jutsus = {
    "Katon: Gran Bola de Fuego": ["snake", "ram", "monkey", "boar", "rat", "tiger"],
    "Clon de Sombra": ["ram", "snake", "tiger"],
    "Raiton: Chidori": ["ox", "hare", "monkey"],
    "Suiton: Jutsu de Misil Dragon de Agua": ["dragon", "hare", "ox", "horse", "monkey", "snake"],
    "Doton: Muralla de Tierra": ["tiger", "hare", "boar", "dog"],
    "Katon: Jutsu de Cenizas Ardientes": ["snake", "ram", "bird", "tiger"],
    "Fuuton: Gran Rafaga de Viento": ["ram", "ox", "dog", "snake"],
    "Suiton: Jutsu de la Gran Cascada": ["bird", "boar", "dog", "monkey", "dragon"],
    "Doton: Decapitacion Subterranea": ["ox", "tiger", "snake"],
    "Katon: Jutsu de Fenix de Fuego": ["tiger", "boar", "dog", "bird"],
    "Raiton: Lanza Relampago": ["ox", "snake", "dragon", "ram"],
    "Fuuton: Jutsu de la Bala de Aire": ["monkey", "hare", "ox", "ram"],
    "Invocacion: Edo Tensei": ["tiger", "snake", "dog", "dragon"]
}

# Seleccionar un jutsu aleatorio
jutsu_actual, secuencia_correcta = random.choice(list(jutsus.items()))

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Cargar las imagenes de las posturas
imagenes_posturas = {}
carpeta_imagenes = "imagenes_posturas"  # Carpeta donde se encuentran las imagenes
for postura in ["snake", "ram", "monkey", "boar", "horse", "tiger", "dog", "rat", "ox", "hare", "bird", "dragon"]:
    imagen_path = os.path.join(carpeta_imagenes, f"{postura}.png")
    imagenes_posturas[postura] = cv2.imread(imagen_path)

# Captura de video
cap = cv2.VideoCapture(0)
# Configurar ancho y alto de la captura de video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Hacer que la ventana sea ajustable
cv2.namedWindow("Jutsus detector", cv2.WINDOW_NORMAL)
# Registrar la funcion callback del mouse
cv2.setMouseCallback("Jutsus detector", on_mouse)

# Lista para almacenar las posturas detectadas
posturas_detectadas = []
ultimas_detecciones = deque(maxlen=5)  # Almacena las ultimas 5 detecciones para evitar repeticiones

ultimo_tiempo = time.time()
delay_postura = 1.5  # Tiempo en segundos antes de aceptar una nueva postura

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar modo espejo si esta activado
    if MIRROR_MODE:
        frame = cv2.flip(frame, 1)

    # Obtener dimensiones reales del frame
    height, width, _ = frame.shape

    # Control de musica segun el estado de la deteccion
    if not detection_started:
        if current_music != "intro":
            pygame.mixer.music.stop()
            pygame.mixer.music.load(os.path.join("sounds", "hebi_theme.mp3"))
            pygame.mixer.music.play(-1)  # Reproducir en loop
            current_music = "intro"
    else:
        if current_music != "tema":
            pygame.mixer.music.stop()
            pygame.mixer.music.load(os.path.join("sounds", "anger_theme.mp3"))
            pygame.mixer.music.play(-1)
            current_music = "tema"

    # Convertir la imagen a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Solo ejecutar la deteccion de manos si se ha iniciado
    if detection_started:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer landmarks y normalizarlos
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)
                landmarks = scaler.transform(landmarks)  # Normalizacion

                # Realizar prediccion
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
        rect_width = 80      # Ancho del rectangulo
        rect_height = 60     # Alto del rectangulo
        spacing = 10         # Espacio entre rectangulos
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

        # Mostrar el jutsu seleccionado en la parte derecha de la pantalla
        text_size = cv2.getTextSize(jutsu_actual, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = width - text_size[0] + 200
        cv2.putText(frame, jutsu_actual, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        # Mostrar las posturas detectadas en orden
        for i, postura in enumerate(posturas_detectadas):
            if postura in imagenes_posturas:
                imagen_postura = imagenes_posturas[postura]
                imagen_postura = cv2.resize(imagen_postura, (rect_width, rect_height))
                y1 = 100 + i * (rect_height + spacing)
                y2 = y1 + rect_height
                if y2 <= height:
                    frame[y1:y2, text_x:text_x + rect_width] = imagen_postura

        # Mensaje de finalizacion si se completa la secuencia
        if posturas_detectadas == secuencia_correcta:
            completion_text = f"COMPLETASTE EL {jutsu_actual}!"
            font_scale_text = width / 1000
            text_size_comp = cv2.getTextSize(completion_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, 3)[0]
            text_x_comp = (width - text_size_comp[0]) // 2
            text_y_comp = (height + text_size_comp[1]) // 2
            cv2.putText(frame, completion_text, (text_x_comp, text_y_comp), cv2.FONT_HERSHEY_SIMPLEX, font_scale_text, (0, 255, 0), 3, cv2.LINE_AA)

    # Dibujar el boton (si detection_started es False, el boton dira "INICIAR", de lo contrario "TERMINAR")
    button_text = "TERMINAR" if detection_started else "INICIAR"
    cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), (50, 50, 50), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_btn = 1
    thickness = 2
    text_size_btn, _ = cv2.getTextSize(button_text, font, font_scale_btn, thickness)
    text_x_btn = button_x + (button_width - text_size_btn[0]) // 2
    text_y_btn = button_y + (button_height + text_size_btn[1]) // 2
    cv2.putText(frame, button_text, (text_x_btn, text_y_btn), font, font_scale_btn, (255, 255, 255), thickness, cv2.LINE_AA)

    # Si se presiono el boton "TERMINAR", salir del ciclo
    if exit_program:
        break

    cv2.imshow("Jutsus detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
pygame.quit()
