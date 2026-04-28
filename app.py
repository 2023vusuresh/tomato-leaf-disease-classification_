import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json
from pathlib import Path

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Tomato Leaf Disease Detection Dashboard",
    page_icon="🍅",
    layout="wide"
)

# ============================================================
# MODEL SETTINGS
# ============================================================
MODEL_PATH = "tomato_disease_final_model.keras"
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

CONFIDENCE_THRESHOLD = 65.0      # below this, result is treated as uncertain / possibly not tomato leaf
GREEN_RATIO_THRESHOLD = 0.08     # basic image check: leaf image should have some green pixels

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #0f172a;
}
.subtitle {
    color: #475569;
    font-size: 1.05rem;
}
.result-card {
    background: #ffffff;
    padding: 1.3rem;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
}
.good-card {
    background: #ecfdf5;
    padding: 1.1rem;
    border-radius: 16px;
    border: 1px solid #bbf7d0;
}
.bad-card {
    background: #fff1f2;
    padding: 1.1rem;
    border-radius: 16px;
    border: 1px solid #fecdd3;
}
.warn-card {
    background: #fffbeb;
    padding: 1.1rem;
    border-radius: 16px;
    border: 1px solid #fde68a;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL + RECOMMENDATIONS
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_recommendations():
    with open(RECOMMENDATION_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

model = load_model()
recommendations = load_recommendations()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def green_leaf_check(image: Image.Image) -> float:
    """
    Basic green-pixel ratio check.
    This is only a practical filter, not a perfect tomato-leaf detector.
    """
    img = image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 45)
    green_ratio = float(np.mean(green_mask))
    return green_ratio

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(image: Image.Image):
    processed = preprocess_image(image)
    probs = model.predict(processed, verbose=0)[0]

    top_index = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_index]
    top_confidence = float(probs[top_index] * 100)

    top_3_indices = probs.argsort()[-3:][::-1]
    top_3 = []
    for idx in top_3_indices:
        disease_key = CLASS_NAMES[int(idx)]
        disease_name = recommendations[disease_key]["display_name"]
        confidence = float(probs[int(idx)] * 100)
        top_3.append((disease_name, confidence))

    all_predictions = []
    for i, key in enumerate(CLASS_NAMES):
        all_predictions.append({
            "Disease": recommendations[key]["display_name"],
            "Confidence (%)": round(float(probs[i] * 100), 2)
        })

    return top_class, top_confidence, top_3, pd.DataFrame(all_predictions).sort_values(
        by="Confidence (%)", ascending=False
    )

def show_recommendations(disease_key):
    info = recommendations[disease_key]

    st.subheader("Disease Information")
    st.write(f"**Cause:** {info['cause']}")

    st.write("**Common Symptoms:**")
    for item in info["symptoms"]:
        st.write(f"- {item}")

    st.write("**Recommended Actions:**")
    for item in info["recommendations"]:
        st.write(f"- {item}")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🍅 Dashboard")
    st.write("Upload or capture a tomato leaf image to classify disease.")
    st.divider()
    st.write("**Model input size:** 128 × 128")
    st.write("**Classes:** 10 tomato leaf categories")
    st.write("**Uncertain rule:** Confidence below 65% is treated carefully.")
    st.warning(
        "Best improvement: train one extra class named 'Not Tomato Leaf' using non-tomato images."
    )

# ============================================================
# MAIN UI
# ============================================================
st.markdown("<div class='main-title'>Tomato Leaf Disease Detection Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Upload a leaf image or capture using camera. The app shows predicted disease, confidence %, and recommendations.</div>",
    unsafe_allow_html=True
)

st.divider()

input_method = st.radio(
    "Choose input method",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

image_file = None

if input_method == "Upload Image":
    image_file = st.file_uploader(
        "Upload tomato leaf image",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("Take a photo of the tomato leaf")

if image_file is not None:
    image = Image.open(image_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Selected Leaf Image", use_container_width=True)

    with col2:
        green_ratio = green_leaf_check(image)
        predicted_key, predicted_confidence, top_3, prediction_table = predict(image)
        predicted_name = recommendations[predicted_key]["display_name"]

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        # Not tomato / uncertain warning logic
        if green_ratio < GREEN_RATIO_THRESHOLD:
            st.markdown("<div class='warn-card'>", unsafe_allow_html=True)
            st.error("This does not look like a tomato leaf image.")
            st.write(
                "Please upload a clear tomato leaf photo. The image may be a non-leaf object, background-heavy image, or unclear photo."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        elif predicted_confidence < CONFIDENCE_THRESHOLD:
            st.markdown("<div class='warn-card'>", unsafe_allow_html=True)
            st.warning("Prediction is uncertain.")
            st.write(
                "The image may not be a clear tomato leaf, or the disease pattern may not be strongly recognized by the model."
            )
            st.write(f"Highest model confidence: **{predicted_confidence:.2f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            if predicted_key == "Tomato_healthy":
                st.markdown("<div class='good-card'>", unsafe_allow_html=True)
                st.success("Leaf Status: Healthy")
            else:
                st.markdown("<div class='bad-card'>", unsafe_allow_html=True)
                st.error("Disease Detected")

            st.markdown(f"### Predicted Disease: **{predicted_name}**")
            st.metric(
                label=f"Confidence for {predicted_name}",
                value=f"{predicted_confidence:.2f}%"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Top 3 Disease Probabilities")
        for disease, conf in top_3:
            st.write(f"**{disease}: {conf:.2f}%**")
            st.progress(conf / 100)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Recommendation", "All Class Confidence", "Important Note"])

    with tab1:
        if green_ratio >= GREEN_RATIO_THRESHOLD:
            show_recommendations(predicted_key)
        else:
            st.warning("Recommendation is shown only after a valid tomato leaf image is detected.")

    with tab2:
        st.dataframe(prediction_table, use_container_width=True, hide_index=True)

    with tab3:
        st.info(
            "This model was trained on tomato leaf disease classes. It can classify tomato leaf diseases, "
            "but it is not a perfect detector of whether an image is tomato or not. For highest accuracy, "
            "add a separate training class called 'Not Tomato Leaf' with images of other leaves, hands, soil, "
            "plain background, and unrelated objects."
        )

else:
    st.info("Upload an image or use the camera to start disease detection.")