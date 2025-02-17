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
import pygame
from ffpyplayer.player import MediaPlayer


# -----------------------------
# CONFIGURACION GLOBAL
# -----------------------------
WIDTH = 1440
HEIGHT = 900
MIRROR_MODE = True

# Configuracion del boton
BUTTON_CONFIG = {
    'width': 100,
    'height': 30,
    'x': WIDTH // 2 - 100 // 2,
    'y': 50
}

# Variables para las dimensiones de los rectangulos de la secuencia de jutsus
RECT_WIDTH = 160      # ancho del rectangulo
RECT_HEIGHT = 120     # alto del rectangulo
RECT_SPACING = 10     # espacio entre rectangulos
JUTSU_SEQUENCE_BOTTOM_MARGIN = 150  # margen inferior aumentado para que el rectangulo no se salga

# Variables de estado global
detection_started = False
exit_program = False
current_music = None
cinematic_played = False  # Variable global para controlar si el video ya se reprodujo

# Rutas de archivos de musica
MUSIC_PATHS = {
    'intro': os.path.join("sounds", "hebi_theme.mp3"),
    'tema': os.path.join("sounds", "anger_theme.mp3")
}

# Tiempo de retardo entre posturas (en segundos)
DELAY_POSTURA = 1.5

# Diccionario de jutsus y sus secuencias
JUTSUS = {
    "Katon: Gran Bola de Fuego": ["snake", "ram", "monkey", "boar", "rat", "tiger"],
    "Clon de Sombra": ["ram", "snake", "tiger"],
    "Raiton: Chidori": ["hare", "monkey"],
    "Suiton: Jutsu de Misil Dragon de Agua": ["dragon", "hare", "horse", "monkey", "snake"],
    "Doton: Muralla de Tierra": ["tiger", "hare", "boar", "dog"],
    "Katon: Jutsu de Cenizas Ardientes": ["snake", "ram", "bird", "tiger"],
    "Fuuton: Gran Rafaga de Viento": ["ram", "dog", "snake"],
    "Suiton: Jutsu de la Gran Cascada": ["bird", "boar", "dog", "monkey", "dragon"],
    "Doton: Decapitacion Subterranea": ["tiger", "snake"],
    "Katon: Jutsu de Fenix de Fuego": ["tiger", "boar", "dog", "bird"],
    "Raiton: Lanza Relampago": ["snake", "dragon", "ram"],
    "Fuuton: Jutsu de la Bala de Aire": ["monkey", "hare", "ram"],
    "Invocacion: Edo Tensei": ["tiger", "snake", "dog", "dragon"]
}

CINEMATIC_PATHS = {
    "Katon: Gran Bola de Fuego": os.path.join("cinematics", "bolaFuego1.mp4"),
    "Clon de Sombra": os.path.join("cinematics", "clonSombra.mp4"),
    "Raiton: Chidori": os.path.join("cinematics", "chidori1.mp4"),
    "Suiton: Jutsu de Misil Dragon de Agua": os.path.join("cinematics", "aguaDragon1.mp4"),
    "Katon: Jutsu de Cenizas Ardientes": os.path.join("cinematics", "cenizaArdientes.mp4"),
    "Fuuton: Gran Rafaga de Viento": os.path.join("cinematics", "rafagaViento.mp4"),
    "Suiton: Jutsu de la Gran Cascada": os.path.join("cinematics", "granCascada.mp4"),
    "Doton: Decapitacion Subterranea": os.path.join("cinematics", "doton.mp4"),
    "Katon: Jutsu de Fenix de Fuego": os.path.join("cinematics", "fenix1.mp4"),
    "Raiton: Lanza Relampago": os.path.join("cinematics", "raiton.mp4"),
    "Fuuton: Jutsu de la Bala de Aire": os.path.join("cinematics", "balaAire.mp4"),
    "Invocacion: Edo Tensei": os.path.join("cinematics", "edoTensei.mp4"),
    # Agrega aquí las rutas de los videos para cada jutsu
}

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------

# Función para reproducir un video
def play_cinematic(video_path, frame):
    global current_music  # Acceder a la variable global de la música

    # Pausar la música de fondo
    pygame.mixer.music.pause()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    # Obtener las dimensiones del frame principal
    frame_height, frame_width, _ = frame.shape

    # Iniciar el reproductor de audio en un hilo separado
    player = MediaPlayer(video_path)
    
    while cap.isOpened():
        ret, cinematic_frame = cap.read()
        if not ret:
            break

        # Redimensionar el video al tamaño del frame principal
        cinematic_frame = cv2.resize(cinematic_frame, (frame_width, frame_height))

        # Mostrar el video en la misma ventana
        cv2.imshow("Jutsus detector", cinematic_frame)

        # Sincronizar el audio con el video
        audio_frame, val = player.get_frame()
        if val == 'eof':
            break

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    player = None

    # Reanudar la música de fondo
    pygame.mixer.music.unpause()
    cv2.destroyWindow("Jutsus detector")

# Funcion callback para detectar clicks del mouse en la ventana
def on_mouse(event, x, y, flags, param):
    global detection_started, exit_program
    if event == cv2.EVENT_LBUTTONDOWN:
        # Verificar si el click se realizo dentro del boton
        if (BUTTON_CONFIG['x'] <= x <= BUTTON_CONFIG['x'] + BUTTON_CONFIG['width'] and
            BUTTON_CONFIG['y'] <= y <= BUTTON_CONFIG['y'] + BUTTON_CONFIG['height']):
            if not detection_started:
                detection_started = True
            else:
                detection_started = False
                #exit_program = True

# Cargar el modelo y el escalador
def load_model():
    try:
        knn = joblib.load("modelo_knn.pkl")
        scaler = joblib.load("scaler.pkl")
        return knn, scaler
    except FileNotFoundError:
        print("Error: No se encontraron los archivos del modelo (modelo_knn.pkl o scaler.pkl).")
        exit()

# Seleccionar un jutsu aleatorio
def choose_jutsu(jutsus):
    return random.choice(list(jutsus.items()))

# Inicializar mediapipe Hands
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    return mp_hands, hands, mp_draw

# Cargar las imagenes de las posturas desde la carpeta
def load_images_posturas(carpeta="imagenes_posturas"):
    posturas = ["snake", "ram", "monkey", "boar", "horse", "tiger", "dog", "rat", "hare", "bird", "dragon"]
    imagenes = {}
    for postura in posturas:
        imagen_path = os.path.join(carpeta, f"{postura}.png")
        imagenes[postura] = cv2.imread(imagen_path)
    return imagenes

# Actualizar la reproduccion de musica segun el estado de la deteccion
def update_music(detection_started, current_music):
    if not detection_started:
        if current_music != "intro":
            pygame.mixer.music.stop()
            pygame.mixer.music.load(MUSIC_PATHS['intro'])
            pygame.mixer.music.play(-1)
            return "intro"
    else:
        if current_music != "tema":
            pygame.mixer.music.stop()
            pygame.mixer.music.load(MUSIC_PATHS['tema'])
            pygame.mixer.music.play(-1)
            return "tema"
    return current_music

# Dibujar el boton en la ventana
def draw_button(frame, detection_started):
    button_text = "Intentar Otra vez" if detection_started else "INICIAR"
    x = BUTTON_CONFIG['x']
    y = BUTTON_CONFIG['y']
    width_b = BUTTON_CONFIG['width']
    height_b = BUTTON_CONFIG['height']
    cv2.rectangle(frame, (x, y), (x + width_b, y + height_b), (50, 50, 50), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size, _ = cv2.getTextSize(button_text, font, font_scale, thickness)
    text_x = x + (width_b - text_size[0]) // 2
    text_y = y + (height_b + text_size[1]) // 2
    cv2.putText(frame, button_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# Dibujar la secuencia de jutsu en la pantalla
def draw_jutsu_sequence(frame, secuencia, posturas_detectadas, imagenes_posturas):
    height, width, _ = frame.shape
    total_width = len(secuencia) * (RECT_WIDTH + RECT_SPACING) - RECT_SPACING
    start_x = (width - total_width) // 2
    start_y = height - JUTSU_SEQUENCE_BOTTOM_MARGIN

    for i, postura in enumerate(secuencia):
        color = (255, 255, 255) if i >= len(posturas_detectadas) else (0, 255, 0)
        x1 = start_x + i * (RECT_WIDTH + RECT_SPACING)
        y1 = start_y
        x2 = x1 + RECT_WIDTH
        y2 = y1 + RECT_HEIGHT
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

        if postura in imagenes_posturas and imagenes_posturas[postura] is not None:
            # Redimensionar la imagen al tamano definido globalmente
            img_postura = cv2.resize(imagenes_posturas[postura], (RECT_WIDTH, RECT_HEIGHT))
            frame[y1:y2, x1:x2] = img_postura

def draw_detected_posturas(frame, posturas_detectadas, imagenes_posturas, x_origin):
    # x_origin es el valor que se obtuvo de draw_jutsu_text
    height, width = frame.shape[:2]
    # Ajustar x_origin si la region sale de la imagen
    if x_origin + RECT_WIDTH > width:
        x_origin = width - RECT_WIDTH
    for i, postura in enumerate(posturas_detectadas):
        if postura in imagenes_posturas and imagenes_posturas[postura] is not None:
            img_postura = cv2.resize(imagenes_posturas[postura], (RECT_WIDTH, RECT_HEIGHT))
            y1 = 100 + i * (RECT_HEIGHT + RECT_SPACING)
            y2 = y1 + RECT_HEIGHT
            if y2 <= height:
                frame[y1:y2, x_origin:x_origin+RECT_WIDTH] = img_postura



# Dibujar el nombre del jutsu en la pantalla
def draw_jutsu_text(frame, jutsu_actual):
    height, width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(jutsu_actual, font, 1, 2)
    text_x = width - text_size[0] + 200
    cv2.putText(frame, jutsu_actual, (text_x, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    return text_x

# Dibujar el mensaje de finalizacion si se completa la secuencia
def draw_completion_message(frame, jutsu_actual):
    global cinematic_played  # Acceder a la variable global
    height, width, _ = frame.shape
    completion_text = f"COMPLETASTE EL {jutsu_actual}!"
    font_scale = width / 1000
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(completion_text, font, font_scale, 3)
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, completion_text, (text_x, text_y), font, font_scale, (0, 255, 0), 3, cv2.LINE_AA)
    # Reproducir la cinemática correspondiente al jutsu completado
    if jutsu_actual in CINEMATIC_PATHS and not cinematic_played:
        play_cinematic(CINEMATIC_PATHS[jutsu_actual], frame)
        cinematic_played = True
        

# Procesar la deteccion de manos y actualizar las posturas detectadas
def process_hand_detection(frame, hands, scaler, knn, imagenes_posturas, secuencia_correcta,
                           posturas_detectadas, ultimas_detecciones, ultimo_tiempo, delay_postura,
                           mp_draw, mp_hands):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extraer landmarks y normalizarlos
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)
            landmarks = scaler.transform(landmarks)
            # Realizar prediccion
            prediction = knn.predict(landmarks)[0]
            tiempo_actual = time.time()
            # Verificar si la postura es la siguiente correcta
            if (len(posturas_detectadas) < len(secuencia_correcta) and
                prediction == secuencia_correcta[len(posturas_detectadas)]):
                if prediction not in ultimas_detecciones and (tiempo_actual - ultimo_tiempo) > delay_postura:
                    posturas_detectadas.append(prediction)
                    ultimas_detecciones.append(prediction)
                    ultimo_tiempo = tiempo_actual

            # Dibujar la imagen de la postura en lugar del texto
            if prediction in imagenes_posturas and imagenes_posturas[prediction] is not None:
                img_postura = imagenes_posturas[prediction]
                frame[50:50+img_postura.shape[0], 50:50+img_postura.shape[1]] = img_postura

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return ultimo_tiempo


# -----------------------------
# FUNCION PRINCIPAL
# -----------------------------
def main():
    global detection_started, exit_program, current_music

    # Inicializar pygame mixer
    pygame.mixer.init()

    # Cargar el modelo y el escalador
    knn, scaler = load_model()

    # Seleccionar un jutsu aleatorio
    jutsu_actual, secuencia_correcta = choose_jutsu(JUTSUS)
    cinematic_played = False  # Reiniciar la variable para permitir la reproducción del video

    # Inicializar mediapipe
    mp_hands, hands, mp_draw = init_mediapipe()

    # Cargar las imagenes de las posturas
    imagenes_posturas = load_images_posturas()

    # Configurar captura de video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    cv2.namedWindow("Jutsus detector", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Jutsus detector", on_mouse)

    posturas_detectadas = []
    ultimas_detecciones = deque(maxlen=5)
    ultimo_tiempo = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Actualizar musica segun el estado de la deteccion
        current_music = update_music(detection_started, current_music)

        # Procesar deteccion de manos solo si se inicio la deteccion
        if detection_started:
            ultimo_tiempo = process_hand_detection(frame, hands, scaler, knn, imagenes_posturas,
                                                     secuencia_correcta, posturas_detectadas,
                                                     ultimas_detecciones, ultimo_tiempo, DELAY_POSTURA,
                                                     mp_draw, mp_hands)

            # Dibujar la secuencia de jutsu
            draw_jutsu_sequence(frame, secuencia_correcta, posturas_detectadas, imagenes_posturas)

            # Mostrar el jutsu seleccionado en la parte derecha
            text_x = draw_jutsu_text(frame, jutsu_actual)

            # Mostrar las posturas detectadas en orden
            draw_detected_posturas(frame, posturas_detectadas, imagenes_posturas, text_x)

            # Mostrar mensaje de finalizacion si se completa la secuencia
            if posturas_detectadas == secuencia_correcta:
                draw_completion_message(frame, jutsu_actual)

        # Dibujar el boton en pantalla
        draw_button(frame, detection_started)

        if exit_program:
            break

        cv2.imshow("Jutsus detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    pygame.quit()

if __name__ == "__main__":
    main()
