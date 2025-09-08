import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_multiclass_cnn.h5")
    return model

model = load_model()

# Class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# App layout
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI scan image and the model will classify the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    st.subheader("🔎 Prediction:")
    st.write(f"**{class_names[predicted_class]}** ({confidence:.2f}%)")

    # Show probabilities for all classes
    st.subheader("Class probabilities:")
    st.bar_chart(dict(zip(class_names, prediction[0])))
