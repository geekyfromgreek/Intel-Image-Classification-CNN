import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import os

try:
    from tensorflow.keras.models import load_model
except ModuleNotFoundError:
    from keras.models import load_model

# page config
st.set_page_config(page_title="Intel Image Classification", layout="centered")

# minimal styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        font-size: 14px;
        color: #666;
        margin-bottom: 30px;
    }
    .prediction-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-top: 15px;
    }
    .pred-label {
        font-size: 22px;
        font-weight: 600;
        color: #333;
    }
    .pred-confidence {
        font-size: 18px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Intel Image Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a natural scene image to classify it</div>', unsafe_allow_html=True)

CLASS_NAMES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
IMG_SIZE = (150, 150)

# load model
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "intel_cnn_model.keras")
    model = load_model(model_path)
    return model

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # predict
    model = get_model()
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # show prediction
    st.markdown(f"""
    <div class="prediction-card">
        <div class="pred-label">Predicted Class: {predicted_class}</div>
        <div class="pred-confidence">Confidence: {confidence:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # probability table
    st.subheader("Class Probabilities")
    prob_data = {
        "Class": CLASS_NAMES,
        "Confidence (%)": [round(p * 100, 2) for p in predictions[0]]
    }
    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values(by="Confidence (%)", ascending=False).reset_index(drop=True)
    st.table(prob_df)
