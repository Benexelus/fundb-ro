import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os
import json

# ------------------------
# Einstellungen
# ------------------------

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"
DB_FILE = "database.json"

np.set_printoptions(suppress=True)

# Ordner erstellen falls nicht vorhanden
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Modell laden
# ------------------------

@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_my_model()
class_names = open(LABELS_PATH, "r").readlines()

# ------------------------
# Mini-Datenbank laden
# ------------------------

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

with open(DB_FILE, "r") as f:
    database = json.load(f)

# ------------------------
# Vorhersage Funktion
# ------------------------

def predict_image(image):

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    
    class_name = class_names[index].strip()
    confidence_score = float(prediction[0][index])

    return class_name, confidence_score

# ------------------------
# UI
# ------------------------

st.title("🧥 Fundbüro KI")
st.write("Lade ein Bild hoch und die KI erkennt: Mütze, Hoodie, Hose oder Schuhe.")

uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🔍 Klassifizieren"):
        class_name, confidence = predict_image(image)

        st.success(f"Erkannt: **{class_name}**")
        st.write(f"Confidence: {round(confidence * 100, 2)} %")

        # Bild speichern
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        image.save(file_path)

        # In Datenbank speichern
        database.append({
            "filename": uploaded_file.name,
            "category": class_name,
            "confidence": confidence
        })

        with open(DB_FILE, "w") as f:
            json.dump(database, f)

# ------------------------
# Suchfunktion
# ------------------------

st.markdown("---")
st.header("🔎 Bereits hochgeladene Bilder durchsuchen")

categories = ["Alle", "Mütze", "Hoodie", "Hose", "Schuhe"]
selected_category = st.selectbox("Kategorie auswählen", categories)

for item in database:
    if selected_category == "Alle" or item["category"] == selected_category:
        image_path = os.path.join(UPLOAD_FOLDER, item["filename"])
        if os.path.exists(image_path):
            st.image(image_path, caption=f"{item['category']} ({round(item['confidence']*100,2)}%)", width=200)
