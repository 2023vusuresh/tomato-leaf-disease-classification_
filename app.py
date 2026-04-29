import os
import json
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError, ImageStat

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
MODEL_URL = "https://drive.google.com/file/d/1yFu154R2rA_fXcIrGE7eIC74FyfatT0o/view?usp=drive_link"
RECOMMENDATION_PATH = "recommendations.json"

IMAGE_SIZE = 128
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

# Strict rules to reduce wrong prediction on non-tomato/unclear images
CONFIDENCE_THRESHOLD = 70.0
GREEN_RATIO_THRESHOLD = 0.06
MIN_BRIGHTNESS = 35
MAX_BRIGHTNESS = 235
MIN_SHARPNESS = 8

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

# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #dcfce7 0%, #ffffff 38%, #fff7ed 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #052e16 0%, #166534 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        padding: 34px;
        border-radius: 30px;
        background: linear-gradient(135deg, #064e3b 0%, #16a34a 55%, #84cc16 100%);
        color: white;
        box-shadow: 0px 20px 48px rgba(22, 101, 52, 0.28);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: 44px;
        font-weight: 900;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.96;
        margin-bottom: 0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(226, 232, 240, 0.95);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0px 14px 36px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(10px);
    }

    .result-success {
        padding: 24px;
        border-radius: 24px;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 1px solid #86efac;
        box-shadow: 0px 10px 28px rgba(22, 163, 74, 0.14);
    }

    .result-danger {
        padding: 24px;
        border-radius: 24px;
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 1px solid #fda4af;
        box-shadow: 0px 10px 28px rgba(225, 29, 72, 0.12);
    }

    .result-warning {
        padding: 24px;
        border-radius: 24px;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
        box-shadow: 0px 10px 28px rgba(245, 158, 11, 0.12);
    }

    .big-pred {
        font-size: 29px;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .confidence {
        font-size: 50px;
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
        padding: 8px 14px;
        border-radius: 999px;
        background: #ecfdf5;
        color: #166534;
        font-weight: 800;
        border: 1px solid #bbf7d0;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .footer-note {
        padding: 18px;
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
# LOAD MODEL + RECOMMENDATIONS
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model from Google Drive... Please wait."):
            output = gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

        if output is None or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model download failed. Make sure Google Drive sharing is set to Anyone with link → Viewer."
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
    st.write("Please check:")
    st.write("1. Google Drive model link is public: Anyone with link → Viewer")
    st.write("2. recommendations.json is uploaded to GitHub")
    st.write("3. requirements.txt contains gdown")
    st.exception(error)
    st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def safe_open_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image, None
    except UnidentifiedImageError:
        return None, "Invalid image file. Please upload JPG, JPEG, PNG, WEBP, or BMP."
    except Exception as error:
        return None, f"Image could not be processed: {error}"


def image_brightness(image: Image.Image) -> float:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    return float(stat.mean[0])


def image_sharpness(image: Image.Image) -> float:
    """
    Lightweight sharpness check using grayscale gradient.
    Higher value = sharper image.
    """
    gray = np.array(image.convert("L").resize((224, 224))).astype(np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))


def green_leaf_ratio(image: Image.Image) -> float:
    """
    Practical green-region check.
    This does not perfectly detect tomato leaves, but helps reject obvious non-leaf/unclear images.
    """
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 40)
    return float(np.mean(green_mask))


def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype(np.float32)

    # IMPORTANT:
    # Your notebook model already had Rescaling(1./255) inside the model.
    # So we do NOT divide by 255 here. If your saved model does not include Rescaling,
    # change the next line to: arr = arr / 255.0
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


def quality_decision(image: Image.Image, confidence: float):
    brightness = image_brightness(image)
    sharpness = image_sharpness(image)
    green_ratio = green_leaf_ratio(image)

    issues = []

    if brightness < MIN_BRIGHTNESS:
        issues.append("image is too dark")
    elif brightness > MAX_BRIGHTNESS:
        issues.append("image is too bright")

    if green_ratio < GREEN_RATIO_THRESHOLD:
        issues.append("image does not look like a clear green leaf")

    if confidence < CONFIDENCE_THRESHOLD:
        issues.append("model confidence is low")

    # Do not reject a highly confident tomato prediction just because the image is slightly blurry.
    # Blur is shown as a quality warning, but it will not block prediction when confidence is high.
    blur_warning = sharpness < MIN_SHARPNESS

    is_valid = len(issues) == 0

    metrics = {
        "Brightness": round(brightness, 2),
        "Sharpness": round(sharpness, 2),
        "Green leaf ratio": round(green_ratio * 100, 2),
        "Model confidence": round(confidence, 2),
        "Blur warning": "Yes" if blur_warning else "No"
    }

    return is_valid, issues, metrics


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🍅 Tomato AI Pro")
    st.write("Upload or capture a tomato leaf image for disease detection.")
    st.divider()
    st.markdown("### Features")
    st.write("✅ Upload image")
    st.write("✅ Camera input")
    st.write("✅ Image quality check")
    st.write("✅ Non-tomato/unclear image rejection")
    st.write("✅ Confidence percentage")
    st.write("✅ Disease recommendations")
    st.divider()
    st.markdown("### Supported Images")
    st.write("JPG, JPEG, PNG, WEBP, BMP")

# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🍅 Tomato Leaf Disease Detection Pro</h1>
        <p>AI-powered disease classification with image upload, camera capture, confidence score, quality checks, and practical recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="small-badge">📸 Camera</span>
    <span class="small-badge">🖼 Upload</span>
    <span class="small-badge">🧪 Quality Check</span>
    <span class="small-badge">📊 Confidence</span>
    <span class="small-badge">🌱 Recommendations</span>
    """,
    unsafe_allow_html=True
)

st.write("")

# ============================================================
# INPUT SECTION
# ============================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

input_method = st.radio(
    "Choose input method:",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

uploaded_image = None

if input_method == "Upload Image":
    uploaded_image = st.file_uploader(
        "Upload a clear tomato leaf image",
        type=SUPPORTED_TYPES,
        help="Supported formats: JPG, JPEG, PNG, WEBP, BMP"
    )
else:
    uploaded_image = st.camera_input("Take a clear photo of one tomato leaf")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# OUTPUT SECTION
# ============================================================
if uploaded_image is not None:
    image, image_error = safe_open_image(uploaded_image)

    if image_error:
        st.error(image_error)
        st.stop()

    predicted_key, confidence, top_3, prediction_df = predict_disease(image)
    predicted_name = recommendations[predicted_key]["display_name"]

    is_valid_leaf, issues, quality_metrics = quality_decision(image, confidence)

    st.write("")
    col1, col2 = st.columns([1, 1.25], gap="large")

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption="Selected Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if not is_valid_leaf:
            st.markdown('<div class="result-warning">', unsafe_allow_html=True)
            st.markdown('<div class="big-pred">❌ This does not appear to be a clear tomato leaf image.</div>', unsafe_allow_html=True)
            st.write("Please upload a close, clear, well-lit tomato leaf image with minimum background.")
            st.write("Reason:")
            for issue in issues:
                st.write(f"• {issue}")

            st.markdown(f"**Model's highest guess:** {predicted_name} ({confidence:.2f}%)")
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
            st.markdown('<div class="muted">Confidence for this predicted disease</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Top 3 Predictions")
        for item in top_3:
            st.write(f"**{item['Disease']} — {item['Confidence']:.2f}%**")
            st.progress(item["Confidence"] / 100)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    tab1, tab2, tab3 = st.tabs(["🌱 Recommendation", "📈 All Confidence Scores", "🧪 Image Quality"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if is_valid_leaf:
            show_recommendation(predicted_key)
        else:
            st.warning("Recommendation is shown only when a clear tomato leaf image is detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.dataframe(prediction_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        metric_df = pd.DataFrame(
            [{"Metric": key, "Value": value} for key, value in quality_metrics.items()]
        )
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.info(
            "For best results, upload one tomato leaf at a time, keep the leaf centered, avoid shadows, "
            "and use a clear image with minimum background."
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Upload an image or use the camera to start disease detection.")
