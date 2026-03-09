import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os
import json

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> NEU: Supabase Import
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from supabase import create_client
import uuid

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
# >>> NEU: Supabase Verbindung
# ------------------------
SUPABASE_URL = "https://gbbwzeuhtjxxjiyzkpig.supabase.co"

SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdiYnd6ZXVodGp4eGppeXprcGlnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI0NTU5ODYsImV4cCI6MjA4ODAzMTk4Nn0.IuaU1dd1_Xu7ZTd5l2FEdUSBigOWoLOky7h4HhAA_JE"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
# Mini-Datenbank laden (LOKAL bleibt bestehen)
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

        # ------------------------
        # LOKAL speichern (ALT)
        # ------------------------

        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        image.save(file_path)

        database.append({
            "filename": uploaded_file.name,
            "category": class_name,
            "confidence": confidence
        })

        with open(DB_FILE, "w") as f:
            json.dump(database, f)

        # ------------------------
        # >>> NEU: Zusätzlich in Supabase speichern
        # ------------------------

        unique_name = f"{uuid.uuid4()}.png"
        image_bytes = uploaded_file.getvalue()

        # >>> NEU: Kategorie Ordner
        category_folder = class_name.lower().replace("ü", "ue")

        # >>> NEU: Speicherpfad
        storage_path = f"{category_folder}/{unique_name}"

        # >>> NEU: Upload zu Supabase Storage
        supabase.storage.from_("bilder").upload(
            path=storage_path,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )

        # >>> NEU: Metadaten speichern
        supabase.table("items").insert({
            "filename": unique_name,
            "path": storage_path,
            "category": class_name,
            "confidence": confidence
        }).execute()

        st.success("Zusätzlich in Supabase gespeichert ✅")

# ------------------------
# Suchfunktion
# ------------------------

st.markdown("---")
st.header("🔎 Bereits hochgeladene Bilder durchsuchen")

categories = ["Alle", "Mütze", "Hoodie", "Hose", "Schuhe"]
selected_category = st.selectbox("Kategorie auswählen", categories)

# ------------------------
# LOKALE Bilder anzeigen (ALT)
# ------------------------

for item in database:
    if selected_category == "Alle" or item["category"] == selected_category:
        image_path = os.path.join(UPLOAD_FOLDER, item["filename"])
        if os.path.exists(image_path):
            st.image(
                image_path,
                caption=f"(Lokal) {item['category']} ({round(item['confidence']*100,2)}%)",
                width=200
            )

# ------------------------
# >>> NEU: Supabase Bilder anzeigen
# ------------------------

if selected_category == "Alle":
    response = supabase.table("items").select("*").execute()
else:
    response = supabase.table("items")\
        .select("*")\
        .eq("category", selected_category)\
        .execute()

if response.data:
    for item in response.data:
        public_url = supabase.storage.from_("bilder").get_public_url(item["path"])
        st.image(
            public_url,
            caption=f"(Cloud) {item['category']} ({round(item['confidence']*100,2)}%)",
            width=200
        )

# ------------------------
# NEU: Funktion zum Speichern hochgeladener Bilder
# ------------------------

def save_uploaded_image(uploaded_file):

    if uploaded_file is None:
        return None

    filename = uploaded_file.name
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if os.path.exists(file_path):
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(file_path):
            filename = f"{name}_{counter}{ext}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            counter += 1

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return filename
