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
CHEATCODE = "Google1#23#"

np.set_printoptions(suppress=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------
# Supabase Verbindung
# ------------------------

SUPABASE_URL = "https://gbbwzeuhtjxxjiyzkpig.supabase.co"
SUPABASE_KEY = "YOUR_SUPABASE_KEY"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------
# Kategorien
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
# Lokale Datenbank
# ------------------------

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

with open(DB_FILE) as f:
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

    size = (224,224)

    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    normalized = (image_array.astype(np.float32)/127.5)-1

    data = np.ndarray((1,224,224,3), dtype=np.float32)
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
    "Seite",
    ["Bild hochladen","Bild suchen","Galerie","⚙️ Einstellungen"]
)

st.title("🧥 Fundbüro KI")

# ------------------------
# Upload
# ------------------------

if page == "Bild hochladen":

    uploaded = st.file_uploader("Bild hochladen",type=["jpg","jpeg","png"])

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(image,width=300)

        if st.button("Klassifizieren"):

            image_bytes = uploaded.getvalue()

            image_hash = get_hash(image_bytes)

            # DUPLIKATE LOKAL
            for item in database:
                if item.get("hash")==image_hash:
                    st.error("❌ Bild existiert bereits.")
                    st.stop()

            # DUPLIKATE SUPABASE
            try:
                response = supabase.table("items").select("filename").eq("hash",image_hash).execute()
                if response.data:
                    st.error("❌ Bild existiert bereits in Supabase.")
                    st.stop()
            except:
                pass

            class_name, confidence = predict_image(image)

            st.write("Erkannt:",class_name)
            st.write("Confidence:",round(confidence*100,2),"%")

            if confidence < MIN_CONFIDENCE:

                st.error("❌ KI ist nicht sicher genug (<85%).")

                st.stop()

            unique = str(uuid.uuid4())+".png"

            path = os.path.join(UPLOAD_FOLDER,unique)

            image.save(path)

            entry = {
                "filename":unique,
                "category":class_name,
                "confidence":confidence,
                "hash":image_hash
            }

            database.append(entry)

            with open(DB_FILE,"w") as f:
                json.dump(database,f)

            # SUPABASE

            storage_path = f"{class_name.lower()}/{unique}"

            try:

                supabase.storage.from_("bilder").upload(
                    storage_path,
                    image_bytes,
                    {"content-type":"image/png"}
                )

                supabase.table("items").insert({
                    "filename":unique,
                    "path":storage_path,
                    "category":class_name,
                    "confidence":confidence,
                    "hash":image_hash
                }).execute()

            except:
                st.warning("Supabase Upload fehlgeschlagen")

            st.success("✅ Bild gespeichert")

# ------------------------
# Suche
# ------------------------

elif page == "Bild suchen":

    categories=["Alle","Mütze","Hose","Hoodie","Schuhe"]

    selected=st.selectbox("Kategorie",categories)

    for item in database:

        cat=category_map.get(item["category"].lower(),item["category"])

        if selected=="Alle" or cat==selected:

            path=os.path.join(UPLOAD_FOLDER,item["filename"])

            if os.path.exists(path):

                st.image(path,caption=cat,width=200)

# ------------------------
# Galerie
# ------------------------

elif page == "Galerie":

    cols=st.columns(3)

    i=0

    for item in database:

        path=os.path.join(UPLOAD_FOLDER,item["filename"])

        if os.path.exists(path):

            with cols[i%3]:

                st.image(path,caption=item["category"],use_column_width=True)

            i+=1

# ------------------------
# Einstellungen (Admin)
# ------------------------

elif page == "⚙️ Einstellungen":

    code=st.text_input("Cheatcode eingeben",type="password")

    if code==CHEATCODE:

        st.success("Adminmodus aktiviert")

        for item in database:

            col1,col2=st.columns([3,1])

            with col1:

                path=os.path.join(UPLOAD_FOLDER,item["filename"])

                if os.path.exists(path):

                    st.image(path,width=200)

            with col2:

                if st.button("🗑 Löschen",key=item["filename"]):

                    try:
                        os.remove(path)
                    except:
                        pass

                    try:
                        supabase.storage.from_("bilder").remove([item["path"]])
                        supabase.table("items").delete().eq("filename",item["filename"]).execute()
                    except:
                        pass

                    database.remove(item)

                    with open(DB_FILE,"w") as f:
                        json.dump(database,f)

                    st.rerun()

    else:

        st.info("Adminbereich gesperrt")
