import cv2
import numpy as np

# Ścieżki do plików modeli
age_proto = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\Data2\\age_deploy.prototxt'
age_model = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\Data2\\age_net.caffemodel'
gender_proto = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\Data2\\gender_deploy.prototxt'
gender_model = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\Data2\\gender_net.caffemodel'

# Ładowanie modeli
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Definicje etykiet dla płci i grup wiekowych
gender_list = ['Male', 'Female']
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Ładowanie obrazu
image_path = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\Data\\ksiazulo.jpg'
image = cv2.imread(image_path)

# Detekcja twarzy za pomocą kaskady Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Przetwarzanie wykrytych twarzy
for (x, y, w, h) in faces:
    # Wycinanie obszaru twarzy
    face = image[y:y+h, x:x+w]

    # Przekształcanie twarzy na format dla sieci neuronowej
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)

    # Predykcja płci
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]

    # Predykcja wieku
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    # Wyświetlanie wyników
    label = f"{gender}, Age: {age}"
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# Wyświetlanie obrazu
cv2.imshow("Wyniki", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
