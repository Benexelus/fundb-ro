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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
np.set_printoptions(suppress=True)

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
class_names = [line.strip() for line in open(LABELS_PATH)]

# ------------------------
# Lokale Datenbank laden
# ------------------------
if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

with open(DB_FILE, "r") as f:
    database = json.load(f)

# ------------------------
# Hash Funktion
# ------------------------
def get_hash(data):
    return hashlib.md5(data).hexdigest()

# ------------------------
# KI Vorhersage
# ------------------------
def predict_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    arr = np.asarray(image)
    norm = (arr.astype(np.float32)/127.5) - 1
    data = np.ndarray((1,224,224,3), dtype=np.float32)
    data[0] = norm
    pred = model.predict(data)
    index = np.argmax(pred)
    raw_class = class_names[index].lower()
    class_name = category_map.get(raw_class, raw_class)
    confidence = float(pred[0][index])
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
# 1️⃣ Bild hochladen
# ------------------------
if page=="Bild hochladen":
    uploaded = st.file_uploader("Bild hochladen", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, width=300)

        finder_name = st.text_input("Dein Name")
        location = st.text_input("Ort / Stadt")
        fundort = st.text_input("Fundort")
        found_at = st.date_input("Datum des Fundes", datetime.today())

        if st.button("Klassifizieren & Speichern"):
            image_bytes = uploaded.getvalue()
            img_hash = get_hash(image_bytes)

            # Duplikat prüfen
            duplicate = any(item.get("hash")==img_hash for item in database)
            try:
                resp = supabase.table("items").select("filename").eq("hash", img_hash).execute()
                if resp.data: duplicate=True
            except: pass

            if duplicate:
                st.error("❌ Dieses Bild wurde bereits hochgeladen.")
                st.stop()

            # KI Vorhersage
            class_name, confidence = predict_image(image)
            st.write("Erkannt:", class_name)
            st.write("Confidence:", round(confidence*100,2), "%")

            if confidence < MIN_CONFIDENCE:
                st.error("❌ KI ist sich nicht sicher genug (<85%).")
                st.stop()

            st.success("✅ Bild wird gespeichert")
            unique_name = f"{uuid.uuid4()}.png"
            local_path = os.path.join(UPLOAD_FOLDER, unique_name)
            image.save(local_path)

            # Lokale DB speichern
            entry = {
                "filename": unique_name,
                "category": class_name,
                "confidence": confidence,
                "hash": img_hash,
                "finder_name": finder_name,
                "location": location,
                "fundort": fundort,
                "found_at": str(found_at)
            }
            database.append(entry)
            with open(DB_FILE,"w") as f:
                json.dump(database,f)

            # Supabase speichern
            storage_path = f"{class_name.lower()}/{unique_name}"
            try:
                supabase.storage.from_("bilder").upload(storage_path, image_bytes, {"content-type":"image/png"})
                supabase.table("items").insert({
                    "filename": unique_name,
                    "path": storage_path,
                    "category": class_name,
                    "confidence": confidence,
                    "hash": img_hash,
                    "finder_name": finder_name,
                    "location": location,
                    "fundort": fundort,
                    "found_at": str(found_at)
                }).execute()
            except:
                st.warning("Supabase Upload fehlgeschlagen")
            st.success("✅ Bild erfolgreich gespeichert")

# ------------------------
# 2️⃣ Bild suchen
# ------------------------
elif page=="Bild suchen":
    categories = ["Alle","Mütze","Hose","Hoodie","Schuhe"]
    selected = st.selectbox("Kategorie", categories)

    # Lokale Bilder
    for item in database:
        if selected=="Alle" or item["category"]==selected:
            path = os.path.join(UPLOAD_FOLDER,item["filename"])
            if os.path.exists(path):
                st.image(path, caption=f"{item['category']} ({round(item['confidence']*100,2)}%)", width=200)

    # Supabase Bilder
    try:
        resp = supabase.table("items").select("*").execute()
        if resp.data:
            for item in resp.data:
                if selected=="Alle" or item["category"]==selected:
                    url = supabase.storage.from_("bilder").get_public_url(item["path"])
                    st.image(url, caption=f"{item['category']} ({round(item['confidence']*100,2)}%)", width=200)
    except:
        st.warning("Supabase Bilder konnten nicht geladen werden")

# ------------------------
# 3️⃣ Galerie
# ------------------------
elif page=="Galerie":
    st.header("🖼️ Galerie")
    gallery_cat = st.selectbox("Kategorie wählen", ["Alle","Mütze","Hose","Hoodie","Schuhe"])
    cols = st.columns(3)
    i=0

    for item in database:
        if gallery_cat=="Alle" or item["category"]==gallery_cat:
            path = os.path.join(UPLOAD_FOLDER,item["filename"])
            if os.path.exists(path):
                with cols[i%3]:
                    st.image(path, caption=item["category"], use_column_width=True)
                i+=1

    try:
        resp = supabase.table("items").select("*").execute()
        if resp.data:
            for item in resp.data:
                if gallery_cat=="Alle" or item["category"]==gallery_cat:
                    url = supabase.storage.from_("bilder").get_public_url(item["path"])
                    with cols[i%3]:
                        st.image(url, caption=item["category"], use_column_width=True)
                    i+=1
    except:
        st.warning("Supabase Galerie konnte nicht geladen werden")

# ------------------------
# 4️⃣ Adminbereich
# ------------------------
elif page=="⚙️ Einstellungen":
    code = st.text_input("Cheatcode", type="password")
    if code==CHEATCODE:
        st.success("Adminmodus aktiviert")
        for item in database[:]:
            col1,col2 = st.columns([3,1])
            with col1:
                path = os.path.join(UPLOAD_FOLDER,item["filename"])
                if os.path.exists(path):
                    st.image(path, width=200)
            with col2:
                if st.button("🗑 Löschen", key=item["filename"]):
                    # 1️⃣ Lokal
                    try: os.remove(path)
                    except: pass
                    # 2️⃣ Supabase Storage
                    try: supabase.storage.from_("bilder").remove([item["path"]])
                    except: pass
                    # 3️⃣ Supabase Table
                    try: supabase.table("items").delete().eq("filename",item["filename"]).execute()
                    except: pass
                    # 4️⃣ Lokale DB
                    database.remove(item)
                    with open(DB_FILE,"w") as f: json.dump(database,f)
                    st.success(f"{item['filename']} gelöscht ✅")
                    st.experimental_rerun()
    else:
        st.info("Adminbereich gesperrt")
