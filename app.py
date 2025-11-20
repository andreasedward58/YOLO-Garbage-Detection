import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Trash Detection", layout="wide")

st.title("♻️ Trash Detection using YOLO")

# =========================================
# LOAD MODEL (Hanya sekali)
# =========================================
@st.cache_resource
def load_model():
    model_path = "weights/best.pt"  # pastikan file ada di folder weights/
    return YOLO(model_path)

model = load_model()


# =========================================
# UPLOAD IMAGE
# =========================================
uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True)

    # Predict
    results = model.predict(img)

    # YOLO already returns result image as numpy array with bounding boxes
    result_img = results[0].plot()

    with col2:
        st.subheader("Prediction Result")
        st.image(result_img, use_column_width=True)

    # =========================================
    # SHOW LABEL RESULTS
    # =========================================
    st.subheader("Detection Details")

    names = model.names
    boxes = results[0].boxes

    if len(boxes) == 0:
        st.info("No objects detected.")
    else:
        for box in boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = names[cls_id]

            st.write(f"**{label}** — Confidence: `{score:.2f}`")
