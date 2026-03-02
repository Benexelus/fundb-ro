import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from supabase import create_client
import uuid

# ------------------------
# Einstellungen
# ------------------------

MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

np.set_printoptions(suppress=True)

# ------------------------
# Supabase Verbindung
# ------------------------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

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

        # Eindeutiger Dateiname
        file_name = f"{uuid.uuid4()}.png"

        # Bild als Bytes
        image_bytes = uploaded_file.getvalue()

        # 1️⃣ Upload in Supabase Storage
        supabase.storage.from_("uploads").upload(
            file_name,
            image_bytes,
            {"content-type": "image/png"}
        )

        # 2️⃣ Metadaten speichern
        supabase.table("items").insert({
            "filename": file_name,
            "category": class_name,
            "confidence": confidence
        }).execute()

        st.success("Bild erfolgreich in Supabase gespeichert ✅")

# ------------------------
# Suchfunktion
# ------------------------

st.markdown("---")
st.header("🔎 Bereits hochgeladene Bilder durchsuchen")

categories = ["Alle", "Mütze", "Hoodie", "Hose", "Schuhe"]
selected_category = st.selectbox("Kategorie auswählen", categories)

if selected_category == "Alle":
    response = supabase.table("items").select("*").execute()
else:
    response = supabase.table("items")\
        .select("*")\
        .eq("category", selected_category)\
        .execute()

items = response.data

for item in items:
    image_url = supabase.storage.from_("uploads").get_public_url(item["filename"])
    
    st.image(
        image_url,
        caption=f"{item['category']} ({round(item['confidence']*100,2)}%)",
        width=200
    )
