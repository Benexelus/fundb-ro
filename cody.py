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

MIN_CONFIDENCE = 0.85

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
    "3 schuh": "Schuhe",
    "mütze": "Mütze",
    "hose": "Hose",
    "hoodie": "Hoodie",
    "schuhe": "Schuhe"
}

# ------------------------
# Modell laden
# ------------------------

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

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
# Bild Hash
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

    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    data[0] = normalized

    prediction = model.predict(data)

    index = np.argmax(prediction)

    raw_class = class_names[index].strip().lower()

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
            # DUPLIKATE CHECK
            # ------------------------

            for item in database:

                if item.get("hash") == image_hash:

                    st.error("❌ Dieses Bild wurde bereits hochgeladen.")

                    st.stop()

            # ------------------------
            # KI Analyse
            # ------------------------

            class_name, confidence = predict_image(image)

            st.write(f"Erkannt: {class_name}")
            st.write(f"Confidence: {round(confidence*100,2)} %")

            # ------------------------
            # CONFIDENCE CHECK
            # ------------------------

            if confidence < MIN_CONFIDENCE:

                st.error("❌ KI ist sich nicht sicher genug (<85%).")

                st.stop()

            st.success("✅ Bild wird gespeichert")

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

            try:

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

            except:

                st.warning("Supabase Upload fehlgeschlagen")

            st.success("Bild gespeichert ✅")

# ------------------------
# BILD SUCHEN
# ------------------------

elif page == "Bild suchen":

    st.header("🔎 Bilder durchsuchen")

    categories = ["Alle", "Mütze", "Hose", "Hoodie", "Schuhe"]

    selected = st.selectbox("Kategorie", categories)

    # ------------------------
    # LOKALE BILDER
    # ------------------------

    for item in database:

        category = category_map.get(item["category"].lower(), item["category"])

        if selected == "Alle" or category == selected:

            path = os.path.join(UPLOAD_FOLDER, item["filename"])

            if os.path.exists(path):

                st.image(
                    path,
                    caption=f"{category} ({round(item['confidence']*100,2)}%)",
                    width=200
                )

    # ------------------------
    # SUPABASE BILDER
    # ------------------------

    try:

        response = supabase.table("items").select("*").execute()

        if response.data:

            for item in response.data:

                category = category_map.get(item["category"].lower(), item["category"])

                if selected == "Alle" or category == selected:

                    public_url = supabase.storage.from_("bilder").get_public_url(item["path"])

                    st.image(
                        public_url,
                        caption=f"{category} ({round(item['confidence']*100,2)}%)",
                        width=200
                    )

    except:

        st.warning("Supabase Bilder konnten nicht geladen werden")

# ------------------------
# GALERIE
# ------------------------

elif page == "Galerie":

    st.header("🖼️ Galerie")

    cols = st.columns(3)

    i = 0

    # LOKALE BILDER

    for item in database:

        path = os.path.join(UPLOAD_FOLDER, item["filename"])

        if os.path.exists(path):

            with cols[i % 3]:

                st.image(
                    path,
                    caption=item["category"],
                    use_column_width=True
                )

            i += 1

    # SUPABASE BILDER

    try:

        response = supabase.table("items").select("*").execute()

        if response.data:

            for item in response.data:

                public_url = supabase.storage.from_("bilder").get_public_url(item["path"])

                with cols[i % 3]:

                    st.image(
                        public_url,
                        caption=item["category"],
                        use_column_width=True
                    )

                i += 1

    except:

        st.warning("Supabase Galerie konnte nicht geladen werden")
