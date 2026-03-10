import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os
import json
import uuid
import hashlib

from supabase import create_client

# ------------------------
# Einstellungen
# ------------------------

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"
DB_FILE = "database.json"

np.set_printoptions(suppress=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Supabase Verbindung
# ------------------------

SUPABASE_URL = "https://gbbwzeuhtjxxjiyzkpig.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdiYnd6ZXVodGp4eGppeXprcGlnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0NTU5ODYsImV4cCI6MjA4ODAzMTk4Nn0.IuaU1dd1_Xu7ZTd5l2FEdUSBigOWoLOky7h4HhAA_JE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------
# Kategorien Mapping
# ------------------------

category_map = {
    "0 mütze": "Mütze",
    "1 hose": "Hose",
    "2 hoodie": "Hoodie",
    "3 schuh": "Schuhe"
}

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
# Datenbank laden
# ------------------------

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

with open(DB_FILE, "r") as f:
    database = json.load(f)

# ------------------------
# Bild Hash Funktion
# ------------------------

def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

# ------------------------
# KI Vorhersage
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

    raw_class = class_names[index].strip()

    class_name = category_map.get(raw_class, raw_class)

    confidence = float(prediction[0][index])

    return class_name, confidence

# ------------------------
# Navigation
# ------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Seite auswählen",
    ["Bild hochladen", "Bild suchen", "Galerie"]
)

st.title("🧥 Fundbüro KI")

# ------------------------
# BILD HOCHLADEN
# ------------------------

if page == "Bild hochladen":

    st.header("📤 Bild hochladen")

    uploaded_file = st.file_uploader(
        "Bild auswählen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width=300)

        if st.button("Klassifizieren"):

            image_bytes = uploaded_file.getvalue()

            image_hash = get_image_hash(image_bytes)

            # ------------------------
            # DUPLIKATE PRÜFEN
            # ------------------------

            for item in database:
                if item["hash"] == image_hash:
                    st.error("❌ Dieses Bild wurde bereits hochgeladen.")
                    st.stop()

            # ------------------------
            # KI Analyse
            # ------------------------

            class_name, confidence = predict_image(image)

            st.success(f"Erkannt: {class_name}")
            st.write(f"Confidence: {round(confidence * 100, 2)} %")

            # ------------------------
            # Datei speichern
            # ------------------------

            unique_name = f"{uuid.uuid4()}.png"

            file_path = os.path.join(UPLOAD_FOLDER, unique_name)

            image.save(file_path)

            # ------------------------
            # Lokale DB
            # ------------------------

            entry = {
                "filename": unique_name,
                "category": class_name,
                "confidence": confidence,
                "hash": image_hash
            }

            database.append(entry)

            with open(DB_FILE, "w") as f:
                json.dump(database, f)

            # ------------------------
            # Supabase Upload
            # ------------------------

            category_folder = class_name.lower()

            storage_path = f"{category_folder}/{unique_name}"

            supabase.storage.from_("bilder").upload(
                storage_path,
                image_bytes,
                {"content-type": "image/png"}
            )

            supabase.table("items").insert({
                "filename": unique_name,
                "path": storage_path,
                "category": class_name,
                "confidence": confidence
            }).execute()

            st.success("Bild erfolgreich gespeichert ✅")

# ------------------------
# BILD SUCHEN
# ------------------------

elif page == "Bild suchen":

    st.header("🔎 Bilder durchsuchen")

    categories = ["Alle", "Mütze", "Hose", "Hoodie", "Schuhe"]

    selected = st.selectbox("Kategorie auswählen", categories)

    for item in database:

        if selected == "Alle" or item["category"] == selected:

            image_path = os.path.join(UPLOAD_FOLDER, item["filename"])

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    caption=f"{item['category']} ({round(item['confidence']*100,2)}%)",
                    width=200
                )

# ------------------------
# GALERIE
# ------------------------

elif page == "Galerie":

    st.header("🖼️ Galerie")

    response = supabase.table("items").select("*").execute()

    if response.data:

        cols = st.columns(3)

        i = 0

        for item in response.data:

            public_url = supabase.storage.from_("bilder").get_public_url(item["path"])

            with cols[i % 3]:

                st.image(
                    public_url,
                    caption=item["category"],
                    use_column_width=True
                )

            i += 1
