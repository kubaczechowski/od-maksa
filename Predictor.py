import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Ścieżka do wytrenowanego modelu Keras
MODEL_PATH = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\trained_model_30.h5'

# Ścieżka do zdjęcia
IMAGE_PATH = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\Data\\dziad2.jpg'

# Ładowanie wytrenowanego modelu
model = load_model(MODEL_PATH)

# Definicja słownika etykiet
gender_dict = {0: 'Male', 1: 'Female'}

# Ładowanie zdjęcia za pomocą OpenCV
image = cv2.imread(IMAGE_PATH)

# Konwersja do skali szarości (potrzebne do detekcji twarzy)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Ładowanie predefiniowanego klasyfikatora do detekcji twarzy z OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detekcja twarzy
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Sprawdzenie, czy twarze zostały wykryte
if len(faces) == 0:
    print("Nie wykryto twarzy na obrazie.")
else:
    for (x, y, w, h) in faces:
        # Wycinanie obszaru twarzy
        face = gray_image[y:y+h, x:x+w]

        # Przeskalowanie obrazu twarzy do rozmiaru akceptowanego przez model (np. 128x128)
        face_resized = cv2.resize(face, (128, 128))

        # Normalizacja wartości pikseli
        face_normalized = face_resized / 255.0

        # Konwersja do tablicy odpowiedniego formatu (1, 128, 128, 1) - dodajemy wymiar dla batcha
        face_array = img_to_array(face_normalized)
        face_array = np.expand_dims(face_array, axis=0)

        # Predykcja przy użyciu modelu
        predictions = model.predict(face_array)

        # Przewidywana płeć
        predicted_gender = gender_dict[round(predictions[0][0][0])]

        # Przewidywany wiek
        predicted_age = round(predictions[1][0][0])

        # Wyświetlanie wyników
        print(f"Przewidywana płeć: {predicted_gender}")
        print(f"Przewidywany wiek: {predicted_age}")

        # Rysowanie wykrytej twarzy na oryginalnym obrazie
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f'{predicted_gender}, Age: {predicted_age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Wyświetlenie obrazu z oznaczoną twarzą
plt.axis('off')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
