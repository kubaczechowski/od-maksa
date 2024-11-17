import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Load the trained model
MODEL_PATH = 'E:\\PJATK\\rok_4\\SUM\\Age_predictor\\trained_model_30.h5'
model = load_model(MODEL_PATH)

# Gender dictionary
gender_dict = {0: 'Male', 1: 'Female'}

# Function to process image and predict age and gender
def process_image_with_model(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    offsets = [-10, 0, 10, 20]
    result = ""

    if len(faces) == 0:
        result = "No face detected in the image."
    else:
        for (x, y, w, h) in faces:
            age_predictions = []
            gender_predictions = []

            for offset in offsets:
                x_offset = max(x - offset, 0)
                y_offset = max(y - offset, 0)
                w_offset = min(x + w + offset, gray_image.shape[1]) - x_offset
                h_offset = min(y + h + offset, gray_image.shape[0]) - y_offset

                face = gray_image[y_offset:y_offset + h_offset, x_offset:x_offset + w_offset]
                face_resized = cv2.resize(face, (128, 128))
                face_normalized = face_resized / 255.0
                face_array = img_to_array(face_normalized)
                face_array = np.expand_dims(face_array, axis=0)

                predictions = model.predict(face_array)

                # Zbieranie przewidywań płci dla każdego offsetu
                predicted_gender_index = round(predictions[0][0][0])
                gender_predictions.append(predicted_gender_index)

                # Zbieranie przewidywań wieku
                predicted_age = predictions[1][0][0]
                age_predictions.append(predicted_age)

            # Wybieranie najczęściej występującej płci
            most_common_gender_index = Counter(gender_predictions).most_common(1)[0][0]
            predicted_gender = gender_dict[most_common_gender_index]

            # Obliczanie średniej wieku
            average_age = round(np.mean(age_predictions))
            result = f"Predicted Gender: {predicted_gender}, Average Age: {average_age}"

    return result


# def process_image_with_model(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     offsets = [-10, 0, 10, 20, 30]
#     result = ""
#
#     if len(faces) == 0:
#         result = "No face detected in the image."
#     else:
#         for (x, y, w, h) in faces:
#             age_predictions = []
#
#             for offset in offsets:
#                 x_offset = max(x - offset, 0)
#                 y_offset = max(y - offset, 0)
#                 w_offset = min(x + w + offset, gray_image.shape[1]) - x_offset
#                 h_offset = min(y + h + offset, gray_image.shape[0]) - y_offset
#
#                 face = gray_image[y_offset:y_offset + h_offset, x_offset:x_offset + w_offset]
#                 face_resized = cv2.resize(face, (128, 128))
#                 face_normalized = face_resized / 255.0
#                 face_array = img_to_array(face_normalized)
#                 face_array = np.expand_dims(face_array, axis=0)
#
#                 predictions = model.predict(face_array)
#                 if offset == -10:
#                     predicted_gender = gender_dict[round(predictions[0][0][0])]
#
#                 predicted_age = predictions[1][0][0]
#                 age_predictions.append(predicted_age)
#
#             average_age = round(np.mean(age_predictions))
#             result = f"Predicted Gender: {predicted_gender}, Average Age: {average_age}"
#
#     return result

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "landing"

# Function to set the current page
def set_page(page):
    st.session_state.page = page

# Landing Page
if st.session_state.page == "landing":
    st.title("🌟 SUML SZABROWNICY!")
    st.write("This is a coffee shop camera app...")

    # "Get Started" button
    st.button("Get Started 🚀", on_click=set_page, args=["dashboard"])

# Dashboard Page
elif st.session_state.page == "dashboard":
    st.markdown("""<style>.centered-title { text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 20px; }
                   .stTabs [role="tablist"] { display: flex; justify-content: space-between; text-align: center; }
                   .stTabs [role="tab"] { flex: 1; text-align: center; }</style>""", unsafe_allow_html=True)

    st.markdown('<div class="centered-title">🏠 Welcome to the Coffee Shop!</div>', unsafe_allow_html=True)

    # Create tabs for Camera and Upload Photo
    tab1, tab2 = st.tabs(["Camera", "Upload Photo"])

    # Tab 1 - Camera
    with tab1:
        camera_input = st.camera_input("Take a photo")

        if camera_input:
            image = Image.open(camera_input)
            st.image(image, caption="Captured Image", use_container_width=True)

            # Process the image with the age and gender prediction model
            image = np.array(image)
            result = process_image_with_model(image)
            st.write(result)

    # Tab 2 - Upload Photo
    with tab2:
        st.subheader("Upload a Photo")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

            # Process the image with the age and gender prediction model
            image = np.array(img)
            result = process_image_with_model(image)
            st.write(result)
