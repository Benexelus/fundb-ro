import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Streamlit-UI
st.title("🔍 Fundbüro-Klassifizierung")
st.write("Lade ein Bild hoch, um es als Mütze, Hoodie, Hose oder Schuhe zu identifizieren.")

# Modell laden (Cache für Performance)
@st.cache_resource
def load_model_and_labels():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = f.readlines()
    return model, labels

# Bildklassifizierung
def classify_image(img):
    model, labels = load_model_and_labels()
    
    # Bild vorverarbeiten
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_img = (img_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_img, axis=0)
    
    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index].strip()
    confidence = float(prediction[0][index])
    
    return class_name, confidence

# Bild-Upload
uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", width=300)
    
    # Klassifizierung und Ergebnis anzeigen
    class_name, confidence = classify_image(image)
    st.success(f"**Ergebnis:** {class_name.split(' ')[1]} (Genauigkeit: {confidence:.1%})")
    st.progress(confidence)
