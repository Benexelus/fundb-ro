import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Titel der App
st.title("🔍 Fundbüro-Bilderkennung")
st.write("Lade ein Bild hoch, um verlorene Gegenstände zu identifizieren.")

# Modell und Labels laden (mit Caching für Performance)
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_Model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

# Bild klassifizieren
def classify_image(img, model, labels):
    # Bild vorverarbeiten
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_img, axis=0)
    
    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence = float(prediction[0][index])
    return class_name, confidence

# Hauptlogik
model, labels = load_model_and_labels()
uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
        
        # Klassifizierung
        class_name, confidence = classify_image(image, model, labels)
        st.success(f"**Erkannt:** {class_name.split(' ')[1]} (Genauigkeit: {confidence:.1%})")
        st.progress(confidence)
        
    except Exception as e:
        st.error(f"Fehler: {e}")
