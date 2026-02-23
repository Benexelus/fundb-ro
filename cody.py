import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Konfiguration
np.set_printoptions(suppress=True)
st.set_page_config(page_title="Fundbüro Bilderkennung", layout="wide")

@st.cache_resource
def load_keras_model():
    return load_model("keras_Model.h5", compile=False)

@st.cache_resource
def load_class_names():
    with open("labels.txt", "r") as f:
        return f.readlines()

# Funktion zur Bildklassifizierung
def classify_image(image):
    size = (224, 224)
    processed_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(processed_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    model = load_keras_model()
    class_names = load_class_names()
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# UI
st.title("📦 Fundbüro Bilderkennung")
st.markdown("""
Lade ein Bild hoch, und das Modell sagt dir, ob es eine **Mütze**, **Hoodie**, **Hose** oder **Schuhe** sind!
""")

# Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bildverarbeitung
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Klassifizierung
    try:
        class_name, confidence_score = classify_image(image)
        with col2:
            st.subheader("Ergebnis")
            st.metric(label="Erkannte Kategorie", value=class_name.split(" ")[1])
            st.progress(float(confidence_score))
            st.write(f"Genauigkeit: {confidence_score:.2%}")
    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {str(e)}")
else:
    st.write("Noch keine Bilder hochgeladen.")
