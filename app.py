import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Trash Detection with YOLOv8")

model = YOLO("best.torchscript")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    img_np = np.array(img)

    results = model(img_np)

    result_img = results[0].plot()

    st.image(result_img, channels="BGR")
