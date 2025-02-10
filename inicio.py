import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import subprocess
import os
import wave
import json
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer

def iniciar_pruebas():
    ventana.destroy()  # Cerrar la ventana principal
    # Usa una ruta absoluta para ejecutar pruebas.py
    script_path = os.path.join(os.path.dirname(__file__), "pruebas.py")
    try:
        result = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = result.communicate()
        print(stdout)
        print(stderr)
    except Exception as e:
        print(f"Error al ejecutar pruebas.py: {e}")

def deteccion_voz():
    ventana.destroy()  # Cerrar la ventana principal

    # Cargar el modelo de Vosk con una ruta absoluta
    model_path = "d:/ANDDY/UCE/9noSemestre/MineriaDeDatos/Proyecto_Final/JUTSUS_CAMBIOS/vosk-model-small-es-0.42"
    model = Model(model_path)

    recognizer = KaldiRecognizer(model, 16000)

    # Listar dispositivos de audio disponibles
    print(sd.query_devices())

    # Especificar el dispositivo de entrada (ajusta el índice según el dispositivo que desees usar)
    device_index = 1  # Ajusta este índice según el dispositivo que desees usar
    device_info = sd.query_devices(device_index, kind='input')
    print(f"Usando dispositivo de entrada: {device_info['name']}")

    def callback(indata, frames, time, status):
        if status:
            print(status)
        # Convertir los datos de audio a bytes antes de pasarlos a AcceptWaveform
        audio_data = np.frombuffer(indata, dtype=np.int16).tobytes()
        if recognizer.AcceptWaveform(audio_data):
            result = recognizer.Result()
            result_dict = json.loads(result)
            texto = result_dict.get("text", "")
            print(f"Has dicho: {texto}")
            if "iniciar" in texto.lower():
                iniciar_pruebas()

    print("Di 'iniciar' para comenzar...")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback, device=device_index):
        sd.sleep(1000000)

def on_enter(e):
    e.widget['background'] = 'black'

def on_leave(e):
    e.widget['background'] = 'red'

def on_enter2(e):
    e.widget['background'] = 'lightblue'

def on_leave2(e):
    e.widget['background'] = 'blue'

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Interfaz de Inicio")

# Configurar la ventana para que ocupe una gran parte de la pantalla
ancho_pantalla = ventana.winfo_screenwidth()
alto_pantalla = ventana.winfo_screenheight()
ancho_ventana = int(ancho_pantalla * 0.8)  # 80% del ancho de la pantalla
alto_ventana = int(alto_pantalla * 0.8)    # 80% del alto de la pantalla
x_ventana = int((ancho_pantalla - ancho_ventana) / 2)
y_ventana = int((alto_pantalla - alto_ventana) / 2)

ventana.geometry(f"{ancho_ventana}x{alto_ventana}+{x_ventana}+{y_ventana}")

# Crear un canvas para la imagen de fondo
canvas = tk.Canvas(ventana, width=ancho_ventana, height=alto_ventana)
canvas.pack(fill="both", expand=True)

# Cargar y redimensionar la imagen de fondo
imagen = Image.open("narutofondo.png")
imagen = imagen.resize((ancho_ventana, alto_ventana), Image.LANCZOS)
imagen_fondo = ImageTk.PhotoImage(imagen)
canvas.create_image(0, 0, image=imagen_fondo, anchor="nw")

# Crear los botones con colores sobre el canvas
boton1 = tk.Button(ventana, text="Iniciar Pruebas", command=iniciar_pruebas, bg="red", fg="white")
boton1_window = canvas.create_window(ancho_ventana//2, alto_ventana//2 - 20, anchor="center", window=boton1)
boton1.bind("<Enter>", on_enter)
boton1.bind("<Leave>", on_leave)

boton2 = tk.Button(ventana, text="Detección de Voz", command=deteccion_voz, bg="blue", fg="white")
boton2_window = canvas.create_window(ancho_ventana//2, alto_ventana//2 + 20, anchor="center", window=boton2)
boton2.bind("<Enter>", on_enter2)
boton2.bind("<Leave>", on_leave2)

# Iniciar el bucle principal de la interfaz
ventana.mainloop()