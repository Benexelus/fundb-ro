import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os
import json
import uuid
import hashlib
from datetime import datetime

from supabase import create_client

# ------------------------
# Einstellungen
# ------------------------

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"
DB_FILE = "database.json"

MIN_CONFIDENCE = 0.85
CHEATCODE = "Google1#23#"

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
def load_model_ai():
    return load_model(MODEL_PATH, compile=False)

model = load_model_ai()
class_names = open(LABELS_PATH).readlines()

# ------------------------
# Lokale Datenbank laden
# ------------------------

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

with open(DB_FILE, "r") as f:
    database = json.load(f)

# ------------------------
# Bild Hash Funktion
# ------------------------

def get_hash(data):
    return hashlib.md5(data).hexdigest()

# ------------------------
# KI Vorhersage
# ------------------------

def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized
    prediction = model.predict(data)
    index = np.argmax(prediction)
    raw = class_names[index].strip().lower()
    class_name = category_map.get(raw, raw)
    confidence = float(prediction[0][index])
    return class_name, confidence

# ------------------------
# Navigation
# ------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Seite auswählen",
    ["Bild hochladen", "Bild suchen", "Galerie", "⚙️ Einstellungen"]
)

st.title("🧥 Fundbüro KI")

# ------------------------
# Bild hochladen
# ------------------------

if page == "Bild hochladen":
    uploaded = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, width=300)

        # Zusätzliche Metadaten abfragen
        finder_name = st.text_input("Dein Name")
        location = st.text_input("Ort / Stadt")
        fundort = st.text_input("Fundort")
        found_at = st.date_input("Datum des Fundes", value=datetime.today())

        if st.button("Klassifizieren und speichern"):
            image_bytes = uploaded.getvalue()
            image_hash = get_hash(image_bytes)

            # ------------------------
            # Duplikat prüfen
            # ------------------------
            duplicate = False
            for item in database:
                if item.get("hash") == image_hash:
                    duplicate = True
                    break
            try:
                response = supabase.table("items").select("filename").eq("hash", image_hash).execute()
                if response.data:
                    duplicate = True
            except:
                pass

            if duplicate:
                st.error("❌ Dieses Bild wurde bereits hochgeladen.")
                st.stop()

            # ------------------------
            # KI Vorhersage
            # ------------------------
            class_name, confidence = predict_image(image)
            st.write("Erkannt:", class_name)
            st.write("Confidence:", round(confidence*100,2), "%")

            if confidence < MIN_CONFIDENCE:
                st.error("❌ KI ist sich nicht sicher genug (<85%).")
                st.stop()

            st.success("✅ Bild wird gespeichert")
            unique_name = str(uuid.uuid4()) + ".png"
            local_path = os.path.join(UPLOAD_FOLDER, unique_name)
            image.save(local_path)

            # ------------------------
            # Lokale Datenbank speichern
            # ------------------------
            entry = {
                "filename": unique_name,
                "category": class_name,
                "confidence": confidence,
                "hash": image_hash,
                "finder_name": finder_name,
                "location": location,
                "fundort": fundort,
                "found_at": str(found_at)
            }
            database.append(entry)
            with open(DB_FILE, "w") as f:
                json.dump(database, f)

            # ------------------------
            # Supabase speichern
            # ------------------------
            storage_path = f"{class_name.lower()}/{unique_name}"
            try:
                supabase.storage.from_("bilder").upload(storage_path, image_bytes, {"content-type":"image/png"})
                supabase.table("items").insert({
                    "filename": unique_name,
                    "path": storage_path,
                    "category": class_name,
                    "confidence": confidence,
                    "hash": image_hash,
                    "finder_name": finder_name,
                    "location": location,
                    "fundort": fundort,
                    "found_at": str(found_at)
                }).execute()
            except:
                st.warning("Supabase Upload fehlgeschlagen")
            st.success("Bild erfolgreich gespeichert ✅")

# ------------------------
# Suche & Galerie bleiben unverändert (können Metadaten anzeigen)
# ------------------------
