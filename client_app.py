import json
from collections import Counter
from datetime import date, datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from keras.losses import MeanAbsoluteError
from keras.models import load_model
from keras.utils import img_to_array

MODEL_PATH = '..//trained_model_30.h5'
model = load_model(MODEL_PATH, custom_objects={"mae": MeanAbsoluteError})

gender_dict = {0: 'Male', 1: 'Female'}


def get_coffee_recommendations(age, gender, season, time_of_day, coffee_data):
    age = int(age)
    if age <= 21:
        age_group = "17-21"
    elif 22 <= age <= 25:
        age_group = "22-25"
    elif 26 <= age <= 30:
        age_group = "26-30"
    else:
        age_group = "31-100"

    recommendations = coffee_data[age_group][gender][season][time_of_day]
    print(recommendations)
    return recommendations


def load_coffee_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            coffee_data = json.load(file)
        return coffee_data
    except Exception:
        return {}


def get_season(current_date=None):
    if current_date is None:
        current_date = date.today()

    year = current_date.year

    spring_start = date(year, 3, 20)
    summer_start = date(year, 6, 21)
    autumn_start = date(year, 9, 23)
    winter_start = date(year, 12, 21)

    if spring_start <= current_date < summer_start:
        return "spring"
    elif summer_start <= current_date < autumn_start:
        return "summer"
    elif autumn_start <= current_date < winter_start:
        return "autumn"
    else:
        return "winter"


def get_time_of_day(current_time=None):
    if current_time is None:
        current_time = datetime.now().time()

    if 6 <= current_time.hour < 12:
        return "morning"
    elif 12 <= current_time.hour < 18:
        return "afternoon"
    elif 18 <= current_time.hour < 22:
        return "evening"
    else:
        return "night"


print(get_season())
print(get_time_of_day())


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

                predicted_gender_index = round(predictions[0][0][0])
                gender_predictions.append(predicted_gender_index)

                predicted_age = predictions[1][0][0]
                age_predictions.append(predicted_age)

            most_common_gender_index = Counter(gender_predictions).most_common(1)[0][0]
            predicted_gender = gender_dict[most_common_gender_index]

            average_age = round(np.mean(age_predictions))
            result = f"Predicted Gender: {predicted_gender}, Average Age: {average_age} \n"
            season = get_season()
            time_of_day = get_time_of_day()
            coffee = load_coffee_data('..//Data//coffeeV4.json')

            predicted_gender = predicted_gender.lower()

            recommendations = get_coffee_recommendations(average_age, predicted_gender, season=season,
                                                         time_of_day=time_of_day, coffee_data=coffee)
            print(type(recommendations))
            if 'classic' in recommendations:
                result += "Classic Recommendations:\n"
                for coffee in recommendations['classic']:
                    result += f"- {coffee['name']} (Base: {coffee['base']}, Additions: {', '.join(coffee['additions'])})\n"
                result += "\n"

            if 'seasonal' in recommendations:
                result += "Seasonal Recommendations:\n"
                for coffee in recommendations['seasonal']:
                    result += f"- {coffee['name']} (Base: {coffee['base']}, Additions: {', '.join(coffee['additions'])})\n"
                result += "\n"

    return result


if "page" not in st.session_state:
    st.session_state.page = "landing"


def set_page(page):
    st.session_state.page = page


if st.session_state.page == "landing":
    st.title("🌟 SUML SZABROWNICY!")
    st.write("This is a coffee shop camera app...")

    st.button("Get Started 🚀", on_click=set_page, args=["dashboard"])

elif st.session_state.page == "dashboard":
    st.markdown("""<style>.centered-title { text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 20px; }
                   .stTabs [role="tablist"] { display: flex; justify-content: space-between; text-align: center; }
                   .stTabs [role="tab"] { flex: 1; text-align: center; }</style>""", unsafe_allow_html=True)

    st.markdown('<div class="centered-title">🏠 Welcome to the Coffee Shop!</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Camera", "Upload Photo"])

    with tab1:
        camera_input = st.camera_input("Take a photo")

        if camera_input:
            image = Image.open(camera_input)
            st.image(image, caption="Captured Image", use_container_width=True)

            image = np.array(image)
            result = process_image_with_model(image)
            st.write(result)

    with tab2:
        st.subheader("Upload a Photo")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)

            image = np.array(img)
            result = process_image_with_model(image)
            st.write(result)
