import cv2
import mediapipe as mp
import os
import pandas as pd

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Directorio donde están las imágenes organizadas en carpetas por postura
DATASET_PATH = "dataset"
data = []

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                    landmarks.append(landmark.z)
                
                data.append([label] + landmarks)

# Guardar los datos en un CSV
columns = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
df = pd.DataFrame(data, columns=columns)
df.to_csv("hand_gestures.csv", index=False)
print("Datos guardados en hand_gestures.csv")
 