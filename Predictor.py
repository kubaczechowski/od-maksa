import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

MODEL_PATH = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\trained_model_30.h5'
IMAGE_PATH = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\Data\\camera.jpg'

model = load_model(MODEL_PATH)
gender_dict = {0: 'Male', 1: 'Female'}

image = cv2.imread(IMAGE_PATH)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Lista wartości offsetu
offsets = [-10, 0, 10]

if len(faces) == 0:
    print("Nie wykryto twarzy na obrazie.")
else:
    for (x, y, w, h) in faces:
        age_predictions = []

        for offset in offsets:
            # Zwiększenie lub zmniejszenie obszaru wycinka twarzy za pomocą offsetu
            x_offset = max(x - offset, 0)
            y_offset = max(y - offset, 0)
            w_offset = min(x + w + offset, gray_image.shape[1]) - x_offset
            h_offset = min(y + h + offset, gray_image.shape[0]) - y_offset

            # Wycinanie większego/mniejszego obszaru twarzy
            face = gray_image[y_offset:y_offset + h_offset, x_offset:x_offset + w_offset]

            # Przeskalowanie obrazu twarzy do rozmiaru akceptowanego przez model (np. 128x128)
            face_resized = cv2.resize(face, (128, 128))
            face_normalized = face_resized / 255.0
            face_array = img_to_array(face_normalized)
            face_array = np.expand_dims(face_array, axis=0)

            # Predykcja przy użyciu modelu
            predictions = model.predict(face_array)

            # Przewidywana płeć (bierzemy pierwszy offset, ponieważ płeć się nie zmienia)
            if offset == 10:
                predicted_gender = gender_dict[round(predictions[0][0][0])]

            # Zbieranie przewidywanych wartości wieku
            predicted_age = predictions[1][0][0]
            age_predictions.append(predicted_age)

        # Obliczanie średniej wieku
        average_age = round(np.mean(age_predictions))

        # Wyświetlanie wyników
        print(f"Przewidywana płeć: {predicted_gender}")
        print(f"Średni przewidywany wiek: {average_age}")

        # Rysowanie wykrytej twarzy na oryginalnym obrazie
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, f'{predicted_gender}, Age: {average_age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Wyświetlenie obrazu z oznaczoną twarzą
plt.axis('off')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
