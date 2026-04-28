import os
import json
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Tomato Leaf Disease AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SETTINGS
# ============================================================
MODEL_PATH = "model.keras"
MODEL_URL = "https://drive.google.com/uc?id=18QFKZwrzdHLfIYEtfo-0yLg77KcO-ge5"
RECOMMENDATION_PATH = "recommendations.json"
IMAGE_SIZE = 128

CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

CONFIDENCE_THRESHOLD = 65.0
GREEN_RATIO_THRESHOLD = 0.08

# ============================================================
# CSS DESIGN
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 45%, #fff7ed 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #064e3b 0%, #166534 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        padding: 32px;
        border-radius: 28px;
        background: linear-gradient(135deg, #14532d 0%, #16a34a 55%, #84cc16 100%);
        color: white;
        box-shadow: 0px 18px 45px rgba(22, 101, 52, 0.25);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 44px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.95;
        margin-bottom: 0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0px 14px 35px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(10px);
    }

    .result-success {
        padding: 22px;
        border-radius: 22px;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #86efac;
        box-shadow: 0px 10px 25px rgba(22, 163, 74, 0.14);
    }

    .result-danger {
        padding: 22px;
        border-radius: 22px;
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 1px solid #fda4af;
        box-shadow: 0px 10px 25px rgba(225, 29, 72, 0.12);
    }

    .result-warning {
        padding: 22px;
        border-radius: 22px;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
        box-shadow: 0px 10px 25px rgba(245, 158, 11, 0.12);
    }

    .big-pred {
        font-size: 30px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 6px;
    }

    .confidence {
        font-size: 48px;
        font-weight: 900;
        color: #166534;
        margin-bottom: 4px;
    }

    .muted {
        color: #64748b;
        font-size: 15px;
    }

    .small-badge {
        display: inline-block;
        padding: 7px 13px;
        border-radius: 999px;
        background: #ecfdf5;
        color: #166534;
        font-weight: 700;
        border: 1px solid #bbf7d0;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .footer-note {
        padding: 16px;
        border-radius: 18px;
        background: #f8fafc;
        color: #475569;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD MODEL + DATA
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model from Google Drive... Please wait."):
            output = gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        if output is None or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model download failed. Please check Google Drive sharing: Anyone with the link → Viewer."
            )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_recommendations():
    if not Path(RECOMMENDATION_PATH).exists():
        raise FileNotFoundError("recommendations.json is missing from GitHub repository.")

    with open(RECOMMENDATION_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    recommendations = load_recommendations()
except Exception as error:
    st.error("App failed to start.")
    st.write("Please check these three things:")
    st.write("1. Google Drive model link is public: Anyone with link → Viewer")
    st.write("2. recommendations.json is uploaded to GitHub")
    st.write("3. requirements.txt contains gdown")
    st.exception(error)
    st.stop()

# ============================================================
# FUNCTIONS
# ============================================================
def get_green_ratio(image: Image.Image) -> float:
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 45)
    return float(np.mean(green_mask))


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_disease(image: Image.Image):
    image_array = preprocess_image(image)
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_key = CLASS_NAMES[predicted_index]
    predicted_confidence = float(probabilities[predicted_index] * 100)

    top_indices = probabilities.argsort()[-3:][::-1]
    top_3 = []
    for idx in top_indices:
        key = CLASS_NAMES[int(idx)]
        top_3.append({
            "Disease": recommendations[key]["display_name"],
            "Confidence": float(probabilities[int(idx)] * 100)
        })

    all_rows = []
    for i, key in enumerate(CLASS_NAMES):
        all_rows.append({
            "Disease": recommendations[key]["display_name"],
            "Confidence (%)": round(float(probabilities[i] * 100), 2)
        })

    prediction_df = pd.DataFrame(all_rows).sort_values("Confidence (%)", ascending=False)

    return predicted_key, predicted_confidence, top_3, prediction_df


def show_recommendation(disease_key: str):
    info = recommendations[disease_key]

    st.markdown("### 🌿 Disease Details")
    st.write(f"**Cause:** {info['cause']}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔍 Common Symptoms")
        for symptom in info["symptoms"]:
            st.write(f"✅ {symptom}")

    with col_b:
        st.markdown("#### 🛡 Recommended Actions")
        for action in info["recommendations"]:
            st.write(f"🌱 {action}")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🍅 Tomato AI")
    st.write("Deep learning dashboard for tomato leaf disease classification.")
    st.divider()
    st.markdown("### Model Info")
    st.write("Image size: **128 × 128**")
    st.write("Classes: **10 tomato categories**")
    st.write("Confidence threshold: **65%**")
    st.divider()
    st.markdown("### Important")
    st.write(
        "For perfect non-tomato detection, train one extra class called "
        "**Not_Tomato_Leaf**."
    )

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🍅 Tomato Leaf Disease Detection</h1>
        <p>Upload a leaf image or capture using camera. The app predicts disease, shows confidence %, and gives practical recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="small-badge">📸 Camera Input</span>
    <span class="small-badge">🖼 Image Upload</span>
    <span class="small-badge">📊 Confidence Score</span>
    <span class="small-badge">🌱 Treatment Tips</span>
    """,
    unsafe_allow_html=True
)

st.write("")

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

input_method = st.radio(
    "Choose how you want to check the leaf:",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

uploaded_image = None

if input_method == "Upload Image":
    uploaded_image = st.file_uploader(
        "Upload tomato leaf image",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_image = st.camera_input("Take a clear photo of the tomato leaf")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# OUTPUT SECTION
# ============================================================
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    st.write("")
    col1, col2 = st.columns([1, 1.25], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption="Selected Leaf Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        green_ratio = get_green_ratio(image)
        predicted_key, confidence, top_3, prediction_df = predict_disease(image)
        predicted_name = recommendations[predicted_key]["display_name"]

        if green_ratio < GREEN_RATIO_THRESHOLD:
            st.markdown('<div class="result-warning">', unsafe_allow_html=True)
            st.markdown('<div class="big-pred">⚠️ This does not look like a clear tomato leaf image.</div>', unsafe_allow_html=True)
            st.write(
                "Please upload a clear tomato leaf image. The image may be unclear, background-heavy, "
                "too dark, or not a leaf."
            )
            st.markdown(f"**Model's highest guess:** {predicted_name} ({confidence:.2f}%)")
            st.markdown("</div>", unsafe_allow_html=True)

        elif confidence < CONFIDENCE_THRESHOLD:
            st.markdown('<div class="result-warning">', unsafe_allow_html=True)
            st.markdown('<div class="big-pred">⚠️ Prediction is uncertain</div>', unsafe_allow_html=True)
            st.markdown(f"**Highest model prediction:** {predicted_name}")
            st.markdown(f'<div class="confidence">{confidence:.2f}%</div>', unsafe_allow_html=True)
            st.write(
                "The image may not be very clear, or the disease pattern is not strongly recognized by the model."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            if predicted_key == "Tomato_healthy":
                st.markdown('<div class="result-success">', unsafe_allow_html=True)
                st.markdown('<div class="big-pred">✅ Leaf Status: Healthy</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-danger">', unsafe_allow_html=True)
                st.markdown('<div class="big-pred">🚨 Disease Detected</div>', unsafe_allow_html=True)

            st.markdown(f"### Predicted Disease: **{predicted_name}**")
            st.markdown(f'<div class="confidence">{confidence:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="muted">Confidence for the predicted disease</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Top 3 Predictions")
        for item in top_3:
            st.write(f"**{item['Disease']} — {item['Confidence']:.2f}%**")
            st.progress(item["Confidence"] / 100)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["🌱 Recommendation", "📈 All Confidence Scores", "ℹ️ Important Note"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if green_ratio >= GREEN_RATIO_THRESHOLD:
            show_recommendation(predicted_key)
        else:
            st.warning("Recommendation is shown only after a clear tomato leaf image is detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.dataframe(prediction_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="footer-note">', unsafe_allow_html=True)
        st.write(
            "This app is for academic/project use. Your current model was trained only on tomato disease classes. "
            "So the app uses confidence score and a green-leaf check to warn when the image may not be a tomato leaf. "
            "For the best result, retrain the model with an additional class named Not_Tomato_Leaf."
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image or use the camera to start disease detection.")