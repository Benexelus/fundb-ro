import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os
import json
import uuid
import hashlib
from datetime import datetime
from io import BytesIO
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
SUPABASE_KEY = "YOUR_SUPABASE_KEY"  # bitte ersetzen
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
# Hilfsfunktionen
# ------------------------
def get_hash(data_bytes):
    return hashlib.md5(data_bytes).hexdigest()

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

        finder_name = st.text_input("Dein Name / Initialen")
        location = st.text_input("Ort / Stadt")
        fundort = st.text_input("Fundort")
        found_at = st.date_input("Datum des Fundes", datetime.today())

        if st.button("Klassifizieren & Speichern"):
            image_bytes = uploaded.getbuffer()
            img_hash = get_hash(image_bytes)

            # Duplikat prüfen
            duplicate_local = any(item.get("hash")==img_hash for item in database)
            duplicate_cloud = False
            try:
                resp = supabase.table("items").select("filename").eq("hash", img_hash).execute()
                if resp.data: duplicate_cloud = True
            except: pass
            if duplicate_local or duplicate_cloud:
                st.error("❌ Dieses Bild wurde bereits hochgeladen.")
                st.stop()

            # KI Vorhersage
            class_name, confidence = predict_image(image)
            st.write("Erkannt:", class_name)
            st.write("Confidence:", round(confidence*100,2), "%")
            if confidence < MIN_CONFIDENCE:
                st.error("❌ KI ist sich nicht sicher genug (<85%).")
                st.stop()

            # Speicherung
            unique_name = f"{uuid.uuid4()}.png"
            local_path = os.path.join(UPLOAD_FOLDER, unique_name)
            image.save(local_path)

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
            with open(DB_FILE,"w") as f: json.dump(database,f)

            # Supabase Upload
            storage_path = f"{class_name}/{unique_name}"
            try:
                supabase.storage.from_("bilder").upload(storage_path, BytesIO(image_bytes), {"content-type":"image/png"})
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
                st.success("✅ Bild erfolgreich hochgeladen")
            except Exception as e:
                st.error(f"Supabase Upload fehlgeschlagen: {e}")

# ------------------------
# 2️⃣ Bild suchen
# ------------------------
elif page=="Bild suchen":
    categories = ["Alle","Mütze","Hose","Hoodie","Schuhe"]
    selected = st.selectbox("Kategorie", categories)
    st.subheader("Lokale Bilder")
    for idx,item in enumerate(database):
        if selected=="Alle" or item["category"]==selected:
            path = os.path.join(UPLOAD_FOLDER,item["filename"])
            if os.path.exists(path):
                if st.button(f"Bild_{idx}"):
                    st.image(path, width=300)
                    st.write(f"**Finder:** {item['finder_name']}  |  **Ort:** {item['location']}  |  **Fundort:** {item['fundort']}  |  **Datum:** {item['found_at']}")

    st.subheader("Cloud Bilder")
    try:
        resp = supabase.table("items").select("*").execute()
        if resp.data:
            for idx,item in enumerate(resp.data):
                if selected=="Alle" or item["category"]==selected:
                    url = supabase.storage.from_("bilder").get_public_url(item["path"])
                    if st.button(f"CloudBild_{idx}"):
                        st.image(url, width=300)
                        st.write(f"**Finder:** {item['finder_name']}  |  **Ort:** {item['location']}  |  **Fundort:** {item['fundort']}  |  **Datum:** {item['found_at']}")
    except:
        st.warning("Cloud Bilder konnten nicht geladen werden")

# ------------------------
# 3️⃣ Galerie
# ------------------------
elif page=="Galerie":
    st.header("🖼️ Galerie")
    gallery_cat = st.selectbox("Kategorie wählen", ["Alle","Mütze","Hose","Hoodie","Schuhe"])
    cols = st.columns(3)
    i=0
    for idx,item in enumerate(database):
        if gallery_cat=="Alle" or item["category"]==gallery_cat:
            path = os.path.join(UPLOAD_FOLDER,item["filename"])
            if os.path.exists(path):
                with cols[i%3]:
                    if st.button(f"Local_{idx}"):
                        st.image(path, width=300)
                        st.write(f"**Finder:** {item['finder_name']}  |  **Ort:** {item['location']}  |  **Fundort:** {item['fundort']}  |  **Datum:** {item['found_at']}")
                    st.image(path, caption=item["category"], use_column_width=True)
                i+=1
    try:
        resp = supabase.table("items").select("*").execute()
        if resp.data:
            for idx,item in enumerate(resp.data):
                if gallery_cat=="Alle" or item["category"]==gallery_cat:
                    url = supabase.storage.from_("bilder").get_public_url(item["path"])
                    with cols[i%3]:
                        if st.button(f"Cloud_{idx}"):
                            st.image(url, width=300)
                            st.write(f"**Finder:** {item['finder_name']}  |  **Ort:** {item['location']}  |  **Fundort:** {item['fundort']}  |  **Datum:** {item['found_at']}")
                        st.image(url, caption=item["category"], use_column_width=True)
                    i+=1
    except:
        st.warning("Cloud Galerie konnte nicht geladen werden")

# ------------------------
# 4️⃣ Adminbereich
# ------------------------
elif page=="⚙️ Einstellungen":
    code = st.text_input("Cheatcode", type="password")
    if code==CHEATCODE:
        st.success("Adminmodus aktiviert")
        for idx,item in enumerate(database[:]):
            st.write(f"**{item['category']}** - {item['filename']}")
            st.image(os.path.join(UPLOAD_FOLDER,item["filename"]), width=200)
            if st.button(f"🗑 Löschen_{idx}"):
                try: os.remove(os.path.join(UPLOAD_FOLDER,item["filename"]))
                except: pass
                try: supabase.storage.from_("bilder").remove([item["path"]])
                except: pass
                try: supabase.table("items").delete().eq("filename",item["filename"]).execute()
                except: pass
                database.remove(item)
                with open(DB_FILE,"w") as f: json.dump(database,f)
                st.success(f"{item['filename']} gelöscht ✅")
                st.experimental_rerun()
    else:
        st.info("Adminbereich gesperrt")
