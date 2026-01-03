
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Monkeypox Detector", layout="centered")

# Title
st.title("🩺  Monkeypox Detector")
st.markdown("Upload a skin image to check for Monkeypox")

# Load the model (update path as needed)
@st.cache_resource
def load_model():
    model_path = "model/mpox.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()
    
# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# File uploader
uploaded_file = st.file_uploader("Upload skin image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")
    
    # Predict button
    if st.button("🔍 Classify Image", type="primary"):
        with st.spinner("Analyzing image..."):

            # Generate image Array
            img_array = preprocess_image(img)
            # Make prediction
            prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Display result
        st.markdown("---")
        
        if prediction > 0.5:
            st.error(f"⚠️ **Monkeypox Detected**")
            st.progress(float(prediction), text=f"Confidence: {prediction*100:.1f}%")
        else:
            st.success(f"✅ **Healthy Skin Or Not Monkeypox**")
            healthy_prob = 1 - prediction
            st.progress(float(healthy_prob), text=f"Confidence: {healthy_prob*100:.1f}%")

# Add some instructions in the sidebar
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload a skin image (JPG/PNG)
    2. Click the **Classify Image** button
    3. View the prediction results
    
    **Supported Conditions:**
    - Healthy skin or not Monkeypox
    - Monkeypox
    """)
    
    st.header("⚠️ Disclaimer")
    st.caption("This tool is for educational purposes only. Always consult a healthcare professional for medical diagnosis.")            